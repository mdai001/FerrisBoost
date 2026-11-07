//! Prediction-only CUDA model and bounded reusable staging.
//!
//! The immutable tree ensemble is flattened and uploaded once when a Python
//! model first uses GPU prediction. Small staging buffers can be reused, but
//! retention is capped independently of the largest prediction request.

use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::{Arc, Mutex};

use crate::inference::CompactInferenceModel;
use crate::tree::Model;

use super::{device_state, DeviceState};

const PREDICT_THREADS: u32 = 256;
const MAX_STAGING_BYTES: usize = 512 * 1024 * 1024;
const MAX_CACHED_STAGING_BYTES: usize = 64 * 1024 * 1024;
const DEVICE_HEADROOM_BYTES: usize = 256 * 1024 * 1024;

struct PredictionStaging {
    input: CudaSlice<f32>,
    output: CudaSlice<f32>,
    batch_rows: usize,
}

// Pageable host copies are not guaranteed to complete before an Async API
// returns. On an early error, wait before the local output Vec is dropped.
// Successful calls already synchronize and disarm this guard, so they incur
// no additional wait. The original error is preserved if cleanup also fails.
struct PredictionHostCopyGuard<'a> {
    stream: &'a CudaStream,
    completed: bool,
}

impl Drop for PredictionHostCopyGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
    }
}

impl PredictionStaging {
    fn bytes(&self) -> usize {
        self.input.num_bytes() + self.output.num_bytes()
    }
}

/// Compact prediction-only SoA. Training-only gain/Hessian metadata is not
/// uploaded. Child indices are absolute in the flattened node arrays.
pub struct GpuInferenceModel {
    state: Arc<DeviceState>,
    stream: Arc<CudaStream>,
    predict_fn: CudaFunction,
    tree_offsets: CudaSlice<u32>,
    features: CudaSlice<u32>,
    split_conditions: CudaSlice<f32>,
    left_children: CudaSlice<u32>,
    right_children: CudaSlice<u32>,
    flags: CudaSlice<u8>,
    leaf_values: CudaSlice<f32>,
    input_features: Vec<usize>,
    n_features: usize,
    n_trees: usize,
    base_margin: f32,
    // A cached model owns one stream. Serialize calls on that stream while
    // still allowing different PyModel/device caches to run independently.
    call_lock: Mutex<Option<PredictionStaging>>,
}

impl GpuInferenceModel {
    pub fn new(model: &Model, device_ordinal: usize) -> Result<Self> {
        let compact = CompactInferenceModel::new(model)?;
        Self::from_compact(&compact, device_ordinal)
    }

    pub(crate) fn from_compact(
        compact: &CompactInferenceModel,
        device_ordinal: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            !compact.has_categorical_splits(),
            "GPU prediction does not support categorical splits; score such models on the CPU"
        );
        let model = compact.model();
        let input_features = compact.canonical_features().to_vec();
        let state = device_state(device_ordinal)?;
        let stream = state
            .ctx
            .new_stream()
            .context("创建 prediction CUDA stream")?;
        let predict_fn = state
            .module
            .load_function("predict_margin_soa")
            .context("PTX 中找不到 predict_margin_soa")?;

        let mut tree_offsets = Vec::with_capacity(model.trees.len() + 1);
        let mut features = Vec::new();
        let mut split_conditions = Vec::new();
        let mut left_children = Vec::new();
        let mut right_children = Vec::new();
        let mut flags = Vec::new();
        let mut leaf_values = Vec::new();

        for tree in &model.trees {
            let base = u32::try_from(features.len()).context("GPU prediction 节点数超过 u32")?;
            tree_offsets.push(base);
            for node in &tree.nodes {
                if !node.is_leaf {
                    if node.feat as usize >= model.n_features {
                        bail!("GPU prediction 模型含越界 feature {}", node.feat);
                    }
                    if node.left < 0
                        || node.right < 0
                        || node.left as usize >= tree.nodes.len()
                        || node.right as usize >= tree.nodes.len()
                    {
                        bail!("GPU prediction 模型含越界 child index");
                    }
                }
                features.push(node.feat);
                split_conditions.push(node.split_cond);
                left_children.push(if node.is_leaf {
                    0
                } else {
                    base.checked_add(node.left as u32)
                        .context("GPU prediction child index 溢出")?
                });
                right_children.push(if node.is_leaf {
                    0
                } else {
                    base.checked_add(node.right as u32)
                        .context("GPU prediction child index 溢出")?
                });
                flags.push(u8::from(node.is_leaf) | (u8::from(node.default_left) << 1));
                leaf_values.push(node.leaf_value);
            }
        }
        tree_offsets.push(u32::try_from(features.len()).context("GPU prediction 节点数超过 u32")?);

        // CUDA allocations cannot be zero-sized. Empty ensembles still use
        // zero tree ranges, so one unreachable sentinel node is sufficient.
        if features.is_empty() {
            features.push(0);
            split_conditions.push(0.0);
            left_children.push(0);
            right_children.push(0);
            flags.push(1);
            leaf_values.push(0.0);
        }

        let tree_offsets = stream
            .clone_htod(&tree_offsets)
            .context("prediction tree offsets H2D")?;
        let features = stream
            .clone_htod(&features)
            .context("prediction feature ids H2D")?;
        let split_conditions = stream
            .clone_htod(&split_conditions)
            .context("prediction thresholds H2D")?;
        let left_children = stream
            .clone_htod(&left_children)
            .context("prediction left children H2D")?;
        let right_children = stream
            .clone_htod(&right_children)
            .context("prediction right children H2D")?;
        let flags = stream.clone_htod(&flags).context("prediction flags H2D")?;
        let leaf_values = stream
            .clone_htod(&leaf_values)
            .context("prediction leaf values H2D")?;
        stream
            .synchronize()
            .context("等待 prediction model upload")?;

        Ok(Self {
            state,
            stream,
            predict_fn,
            tree_offsets,
            features,
            split_conditions,
            left_children,
            right_children,
            flags,
            input_features,
            leaf_values,
            n_features: model.n_features,
            n_trees: model.trees.len(),
            base_margin: crate::train::base_margin(model.objective, model.base_score),
            call_lock: Mutex::new(None),
        })
    }

    pub fn input_features(&self) -> &[usize] {
        &self.input_features
    }

    pub fn predict_margins(
        &self,
        row_major: &[f32],
        n_rows: usize,
        n_trees: usize,
    ) -> Result<Vec<f32>> {
        let expected = n_rows
            .checked_mul(self.n_features)
            .context("GPU prediction input size 溢出")?;
        if row_major.len() != expected {
            bail!(
                "GPU prediction 输入有 {} 个值,预期 {n_rows} × {} = {expected}",
                row_major.len(),
                self.n_features
            );
        }
        if n_rows == 0 {
            return Ok(Vec::new());
        }
        let mut guard = self
            .call_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU prediction cache poisoned"))?;

        let bytes_per_row = self
            .n_features
            .checked_add(1)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("GPU prediction row size 溢出")?;
        let (free, _) = self
            .state
            .ctx
            .mem_get_info()
            .context("读取 prediction 可用显存")?;
        let usable = free
            .saturating_add(guard.as_ref().map_or(0, PredictionStaging::bytes))
            .saturating_sub(DEVICE_HEADROOM_BYTES)
            .min(MAX_STAGING_BYTES);
        let batch_rows = (usable / bytes_per_row).max(1).min(n_rows);
        // Keep ownership local until the call completes successfully. Errors
        // cannot leave partially submitted staging cached for the next call.
        let cached = guard.take();
        let reusable = cached
            .as_ref()
            .is_some_and(|staging| staging.batch_rows >= batch_rows && staging.bytes() <= usable);
        let mut staging = if reusable {
            cached.unwrap()
        } else {
            if cached.is_some() {
                drop(cached);
                // Cudarc frees on the stream. Finish the release before a
                // replacement allocation, including under memory pressure.
                self.stream
                    .synchronize()
                    .context("释放 prediction staging")?;
            }
            self.allocate_staging(batch_rows)?
        };
        let result = self.predict_with_staging(row_major, n_rows, n_trees, &mut staging);
        if result.is_ok() && staging.bytes() <= MAX_CACHED_STAGING_BYTES {
            *guard = Some(staging);
        } else {
            drop(staging);
            // Preserve the original error if cleanup also fails. Successful
            // large calls retain no staging allocation after returning.
            if let Err(error) = self.stream.synchronize() {
                if result.is_ok() {
                    return Err(error).context("释放 prediction staging");
                }
            }
        }
        result
    }

    fn allocate_staging(&self, batch_rows: usize) -> Result<PredictionStaging> {
        let input_values = batch_rows
            .checked_mul(self.n_features)
            .context("GPU prediction staging size 溢出")?;
        input_values
            .checked_add(batch_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("GPU prediction staging bytes 溢出")?;
        // SAFETY: every active input value is uploaded before the kernel
        // reads it; predict_margin_soa writes every active output row (also
        // for zero trees) before D2H. Unused capacity is never read/copied.
        // CompactInferenceModel retains one feature even for leaf-only models.
        let input = unsafe { self.stream.alloc::<f32>(input_values) }
            .context("分配 GPU prediction input staging")?;
        let output = unsafe { self.stream.alloc::<f32>(batch_rows) }
            .context("分配 GPU prediction output staging")?;
        Ok(PredictionStaging {
            input,
            output,
            batch_rows,
        })
    }

    fn predict_with_staging(
        &self,
        row_major: &[f32],
        n_rows: usize,
        n_trees: usize,
        staging: &mut PredictionStaging,
    ) -> Result<Vec<f32>> {
        let batch_rows = staging.batch_rows.min(n_rows);
        let mut output = vec![0.0f32; n_rows];
        // Declared after output: reverse drop order keeps host storage alive
        // until pending copies have been drained on every early-return path.
        let mut host_copy_guard = PredictionHostCopyGuard {
            stream: &self.stream,
            completed: false,
        };
        let n_features =
            u32::try_from(self.n_features).context("GPU prediction feature 数超过 u32")?;
        let n_trees =
            u32::try_from(n_trees.min(self.n_trees)).context("GPU prediction tree 数超过 u32")?;

        for first_row in (0..n_rows).step_by(batch_rows) {
            let rows = (n_rows - first_row).min(batch_rows);
            let values = rows * self.n_features;
            let first_value = first_row * self.n_features;
            {
                let mut input_view = staging.input.slice_mut(..values);
                self.stream
                    .memcpy_htod(
                        &row_major[first_value..first_value + values],
                        &mut input_view,
                    )
                    .context("prediction input H2D")?;
            }
            let input_view = staging.input.slice(..values);
            let mut output_view = staging.output.slice_mut(..rows);
            let rows_u32 = u32::try_from(rows).context("GPU prediction batch 行数超过 u32")?;
            let config = LaunchConfig {
                grid_dim: (rows_u32.div_ceil(PREDICT_THREADS), 1, 1),
                block_dim: (PREDICT_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream.launch_builder(&self.predict_fn);
            launch
                .arg(&input_view)
                .arg(&rows_u32)
                .arg(&n_features)
                .arg(&self.tree_offsets)
                .arg(&n_trees)
                .arg(&self.features)
                .arg(&self.split_conditions)
                .arg(&self.left_children)
                .arg(&self.right_children)
                .arg(&self.flags)
                .arg(&self.leaf_values)
                .arg(&self.base_margin)
                .arg(&mut output_view);
            unsafe { launch.launch(config) }.context("launch predict_margin_soa")?;
            self.stream
                .memcpy_dtoh(&output_view, &mut output[first_row..first_row + rows])
                .context("prediction output D2H")?;
        }
        self.stream
            .synchronize()
            .context("等待 GPU prediction 完成")?;
        host_copy_guard.completed = true;
        Ok(output)
    }
}

#[cfg(test)]
mod cleanup_tests {
    use super::*;

    #[test]
    fn host_copy_guard_drains_queued_work_before_early_return() {
        let state = device_state(0).unwrap();
        let stream = state.ctx.new_stream().unwrap();
        let completion = state.ctx.new_event(None).unwrap();
        // Pinned storage ensures this regression covers genuinely asynchronous
        // host copies, independently of pageable staging behavior on a driver.
        let mut output = unsafe { state.ctx.alloc_pinned::<f32>(4096) }.unwrap();
        let input = vec![7.25f32; 4096];
        let device = stream.clone_htod(&input).unwrap();
        let result = (|| -> Result<()> {
            let _guard = PredictionHostCopyGuard {
                stream: &stream,
                completed: false,
            };
            stream.memcpy_dtoh(&device, &mut output)?;
            completion.record(&stream)?;
            bail!("injected error after queued host copy")
        })();
        assert_eq!(
            result.unwrap_err().to_string(),
            "injected error after queued host copy"
        );
        assert!(
            completion.is_complete(),
            "early return must drain the stream"
        );
        assert!(output
            .as_slice()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 7.25f32.to_bits()));
    }
}
