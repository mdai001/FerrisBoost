//! SHAP contributions, SHAP interaction values and leaf indices on the device.
//!
//! The kernels (`shx_*` in `histogram.cu`) run the same quadrature TreeSHAP as
//! `crate::explain` and replay its float32 operation order, so every output is
//! bit-identical to the host path; see the kernel comment for how. This module
//! flattens the ensemble once (all trees, with the per-node metadata the walk
//! needs), keeps it on the device for the model's lifetime and streams rows
//! through bounded batches: device memory per call is capped (input rows,
//! output, and a per-thread traversal scratch sized from the deepest tree of
//! the requested prefix) whatever the number of rows.

use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, LaunchArgs, LaunchConfig, PushKernelArg,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{device_state, DeviceState};
use crate::explain::{Explainer, Kind};
use crate::tree::Model;

/// Threads per block of the per-row kernels (must match nothing on the device:
/// the kernels index by `blockIdx * blockDim`).
const THREADS: u32 = 128;
/// Words of one traversal frame (`SHX_FRAME_WORDS` in `histogram.cu`).
const FRAME_WORDS: usize = 22;
/// Upper bound on the device memory one call's batch may use.
const MAX_BATCH_BYTES: usize = 512 * 1024 * 1024;
/// Free device memory always left to the driver and other users.
const DEVICE_HEADROOM_BYTES: usize = 256 * 1024 * 1024;
/// CUDA's limit on `gridDim.y` (the tree axis of the tree-parallel launch).
const MAX_GRID_Y: usize = 65_535;
/// Below `CONTRIBS_TREE_PARALLEL_PER_SM` (row, output) pairs per SM the
/// per-row exact-contribution kernel leaves the device mostly idle and its
/// latency is one thread's walk over every tree, so the tree-parallel pair is
/// used instead. Measured crossover: about 64 pairs per SM
/// (200 depth-6 trees, binary and 5-class); above it the per-tree scratch
/// traffic makes the tree-parallel pair up to ~1.9x slower.
const CONTRIBS_TREE_PARALLEL_PER_SM: usize = 64;
/// The same switch for interactions. The tree-parallel log variant measured
/// faster at every tested size (up to 20,000 pairs on 30 SMs: 0.05x to
/// 0.89x of the per-row kernel); larger batches keep the per-row kernel.
const INTERACTIONS_TREE_PARALLEL_PER_SM: usize = 512;
/// Per-row scratch cap of the tree-parallel launch, so a very large ensemble
/// keeps the per-row kernel (whose scratch does not grow with the tree count).
const TREE_PARALLEL_MAX_ROW_SCRATCH: usize = 16 * 1024 * 1024;

/// Early-return guard: pageable host copies are not guaranteed to be finished
/// when an async copy call returns, so drain the stream before the host
/// buffer can be dropped. Successful calls synchronize and disarm it.
struct HostCopyGuard<'a> {
    stream: &'a CudaStream,
    done: bool,
}

impl Drop for HostCopyGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.stream.synchronize();
        }
    }
}

/// An ensemble flattened for the explanation kernels.
pub struct GpuExplainModel {
    state: Arc<DeviceState>,
    stream: Arc<CudaStream>,
    contribs_fn: CudaFunction,
    contribs_per_tree_fn: CudaFunction,
    contribs_reduce_fn: CudaFunction,
    saabas_fn: CudaFunction,
    interactions_fn: CudaFunction,
    interactions_log_fn: CudaFunction,
    interactions_apply_fn: CudaFunction,
    symmetrize_fn: CudaFunction,
    diagonal_fn: CudaFunction,
    leaf_fn: CudaFunction,
    tree_node_begin: CudaSlice<u32>,
    node_feature: CudaSlice<u32>,
    node_split: CudaSlice<f32>,
    node_left: CudaSlice<u32>,
    node_right: CudaSlice<u32>,
    node_flags: CudaSlice<u8>,
    node_leaf: CudaSlice<f32>,
    node_child_weight: CudaSlice<f32>,
    node_prev_same: CudaSlice<u32>,
    node_slot: CudaSlice<u32>,
    node_cat_begin: CudaSlice<u32>,
    node_cat_len: CudaSlice<u32>,
    cat_words: CudaSlice<u32>,
    tree_uniq_begin: CudaSlice<u32>,
    uniq_feats: CudaSlice<u32>,
    node_mean: CudaSlice<f32>,
    rule: CudaSlice<f32>,
    /// Per tree: frames its walk needs (deepest node level + 1).
    tree_levels: Vec<u32>,
    /// Per tree: number of distinct split features.
    tree_slots: Vec<u32>,
    /// Tree-parallel interactions (`shx_interactions_log`): per-tree log
    /// offsets (host copy for sizing, device copy for the kernels) and the
    /// per-slot log entries.
    ix_tree_begin_host: Vec<u32>,
    ix_tree_begin: CudaSlice<u32>,
    ix_slot_ptr: CudaSlice<u32>,
    ix_entry_index: CudaSlice<u32>,
    ix_entry_col: CudaSlice<u32>,
    n_features: usize,
    n_trees: usize,
    k: usize,
    /// Streaming multiprocessors of the device (tree-parallel switch).
    sm_count: usize,
    // One stream per cached model: calls are serialized on it.
    call_lock: Mutex<()>,
}

/// The per-node tables, built on the host.
#[derive(Default)]
struct Flat {
    tree_node_begin: Vec<u32>,
    feature: Vec<u32>,
    split: Vec<f32>,
    left: Vec<u32>,
    right: Vec<u32>,
    flags: Vec<u8>,
    leaf: Vec<f32>,
    child_weight: Vec<f32>,
    prev_same: Vec<u32>,
    slot: Vec<u32>,
    cat_begin: Vec<u32>,
    cat_len: Vec<u32>,
    cat_words: Vec<u32>,
    uniq_begin: Vec<u32>,
    uniq_feats: Vec<u32>,
    mean: Vec<f32>,
    tree_levels: Vec<u32>,
    tree_slots: Vec<u32>,
    ix_tree_begin: Vec<u32>,
    ix_slot_ptr: Vec<u32>,
    ix_entry_index: Vec<u32>,
    ix_entry_col: Vec<u32>,
}

/// The cells `shx_run_tree<1>` adds to, in order, for the subtree under `nd`
/// (the tree structure alone fixes them; see `shx_interactions_log`). Edge
/// `nd -> child` is unwound after the child's subtree: first the diagonal of
/// the parent's feature, then one partner per newest occurrence of every
/// other feature on the path (newest to oldest), skipping the parent's own
/// newest occurrence. `efeats[j - 1]` is the split feature above level `j`.
fn interaction_writes(
    nodes: &[crate::tree::Node],
    nd: usize,
    efeats: &mut Vec<u32>,
    out: &mut Vec<(u32, u32)>,
) {
    let node = &nodes[nd];
    if node.is_leaf {
        return;
    }
    let feat = node.feat;
    for child in [node.left as usize, node.right as usize] {
        efeats.push(feat);
        interaction_writes(nodes, child, efeats, out);
        out.push((feat, feat));
        let d = efeats.len();
        let mut skipped_current = false;
        for j in (1..=d).rev() {
            let split = efeats[j - 1];
            if efeats[j..d].contains(&split) {
                continue;
            }
            if !skipped_current && split == feat {
                skipped_current = true;
                continue;
            }
            out.push((feat, split));
        }
        efeats.pop();
    }
}

fn to_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).with_context(|| format!("GPU explanation {what} exceeds u32"))
}

fn flatten(model: &Model) -> Result<Flat> {
    let mut f = Flat::default();
    for (ti, tree) in model.trees.iter().enumerate() {
        let n = tree.nodes.len();
        let base = f.feature.len();
        f.tree_node_begin.push(to_u32(base, "node count")?);
        f.uniq_begin
            .push(to_u32(f.uniq_feats.len(), "feature slots")?);
        for node in &tree.nodes {
            if !node.is_leaf {
                ensure!(
                    (node.feat as usize) < model.n_features,
                    "tree {ti} splits on out-of-range feature {}",
                    node.feat
                );
                ensure!(
                    node.left >= 0
                        && node.right >= 0
                        && (node.left as usize) < n
                        && (node.right as usize) < n,
                    "tree {ti} has an out-of-range child index"
                );
            }
            f.feature.push(if node.is_leaf { 0 } else { node.feat });
            f.split.push(node.split_cond);
            let child = |c: i32| -> Result<u32> {
                if node.is_leaf {
                    Ok(0)
                } else {
                    to_u32(base + c as usize, "child index")
                }
            };
            f.left.push(child(node.left)?);
            f.right.push(child(node.right)?);
            let cats = node.cats.as_deref();
            f.flags.push(
                u8::from(node.is_leaf)
                    | (u8::from(node.default_left) << 1)
                    | (u8::from(cats.is_some()) << 2),
            );
            f.leaf.push(node.leaf_value);
            f.child_weight.push(1.0);
            f.prev_same.push(0);
            f.slot.push(0);
            match cats {
                Some(words) => {
                    f.cat_begin
                        .push(to_u32(f.cat_words.len(), "category words")?);
                    f.cat_len.push(to_u32(words.len(), "category words")?);
                    f.cat_words.extend_from_slice(words);
                }
                None => {
                    f.cat_begin.push(0);
                    f.cat_len.push(0);
                }
            }
        }
        f.ix_tree_begin
            .push(to_u32(f.ix_entry_index.len(), "interaction log")?);
        let mut means = vec![0.0f32; n];
        let (mut max_level, mut slots) = (0u32, 0u32);
        if n > 0 {
            crate::explain::fill_node_means(&tree.nodes, 0, &mut means);
            // Preorder walk: the path features of a node's ancestors stay in
            // `path_feat[..level]` while its subtree is visited.
            let mut stack = vec![(0usize, 0usize)];
            let mut path_feat: Vec<u32> = Vec::new();
            let mut slot_of: HashMap<u32, u32> = HashMap::new();
            let mut visited = 0usize;
            while let Some((i, level)) = stack.pop() {
                visited += 1;
                ensure!(visited <= n, "tree {ti} is not a tree (node visited twice)");
                max_level = max_level.max(to_u32(level, "tree depth")?);
                let node = &tree.nodes[i];
                if node.is_leaf {
                    continue;
                }
                let g = base + i;
                f.prev_same[g] = (0..level)
                    .rev()
                    .find(|&la| path_feat[la] == node.feat)
                    .map_or(0, |la| (level - la) as u32);
                if path_feat.len() <= level {
                    path_feat.resize(level + 1, 0);
                }
                path_feat[level] = node.feat;
                let next_slot = slot_of.len() as u32;
                let s = *slot_of.entry(node.feat).or_insert_with(|| {
                    f.uniq_feats.push(node.feat);
                    next_slot
                });
                f.slot[g] = s;
                for child in [node.left as usize, node.right as usize] {
                    f.child_weight[base + child] =
                        crate::explain::branch_weight(tree.nodes[child].sum_hess, node.sum_hess);
                    stack.push((child, level + 1));
                }
            }
            slots = slot_of.len() as u32;
            // Interaction log entries of this tree, grouped by matrix row
            // (= feature slot), log order inside a slot.
            let mut writes = Vec::new();
            interaction_writes(&tree.nodes, 0, &mut Vec::new(), &mut writes);
            let mut per_slot: Vec<Vec<(u32, u32)>> = vec![Vec::new(); slot_of.len()];
            for (j, &(row, col)) in writes.iter().enumerate() {
                per_slot[slot_of[&row] as usize].push((to_u32(j, "interaction log")?, col));
            }
            // Every log value is exactly one entry, so the entry count before
            // this tree (pushed to `ix_tree_begin` above) is also its log offset.
            for entries in per_slot {
                f.ix_slot_ptr
                    .push(to_u32(f.ix_entry_index.len(), "interaction entries")?);
                for (j, col) in entries {
                    f.ix_entry_index.push(j);
                    f.ix_entry_col.push(col);
                }
            }
        }
        f.mean.extend_from_slice(&means);
        f.tree_levels.push(max_level + 1);
        f.tree_slots.push(slots);
    }
    f.tree_node_begin
        .push(to_u32(f.feature.len(), "node count")?);
    f.uniq_begin
        .push(to_u32(f.uniq_feats.len(), "feature slots")?);
    f.ix_tree_begin
        .push(to_u32(f.ix_entry_index.len(), "interaction log")?);
    f.ix_slot_ptr
        .push(to_u32(f.ix_entry_index.len(), "interaction entries")?);
    // CUDA allocations cannot be empty: keep one inert entry per table.
    if f.ix_entry_index.is_empty() {
        f.ix_entry_index.push(0);
        f.ix_entry_col.push(0);
    }
    if f.feature.is_empty() {
        f.feature.push(0);
        f.split.push(0.0);
        f.left.push(0);
        f.right.push(0);
        f.flags.push(1);
        f.leaf.push(0.0);
        f.child_weight.push(1.0);
        f.prev_same.push(0);
        f.slot.push(0);
        f.cat_begin.push(0);
        f.cat_len.push(0);
        f.mean.push(0.0);
    }
    if f.cat_words.is_empty() {
        f.cat_words.push(0);
    }
    if f.uniq_feats.is_empty() {
        f.uniq_feats.push(0);
    }
    Ok(f)
}

/// Batch row count for `per_row_bytes` of device memory per row.
fn batch_rows(usable: usize, per_row_bytes: usize, n_rows: usize) -> Result<usize> {
    ensure!(
        per_row_bytes <= usable,
        "GPU explanation needs {per_row_bytes} bytes of device memory per row but only {usable} \
         are available; use the CPU (use_gpu=False) for this model and output type"
    );
    Ok((usable / per_row_bytes).clamp(1, n_rows.max(1)))
}

impl GpuExplainModel {
    pub fn new(model: &Model, device_ordinal: usize) -> Result<Self> {
        let flat = flatten(model)?;
        let state = device_state(device_ordinal)?;
        let stream = state
            .ctx
            .new_stream()
            .context("create explanation CUDA stream")?;
        let load = |name: &str| -> Result<CudaFunction> {
            state
                .module
                .load_function(name)
                .with_context(|| format!("{name} not found in PTX"))
        };
        let up = |what: &str| format!("explanation {what} H2D");
        Ok(Self {
            contribs_fn: load("shx_contribs")?,
            contribs_per_tree_fn: load("shx_contribs_per_tree")?,
            contribs_reduce_fn: load("shx_contribs_reduce")?,
            saabas_fn: load("shx_saabas")?,
            interactions_fn: load("shx_interactions")?,
            interactions_log_fn: load("shx_interactions_log")?,
            interactions_apply_fn: load("shx_interactions_apply")?,
            symmetrize_fn: load("shx_interactions_symmetrize")?,
            diagonal_fn: load("shx_interactions_diagonal")?,
            leaf_fn: load("shx_leaf")?,
            tree_node_begin: stream
                .clone_htod(&flat.tree_node_begin)
                .with_context(|| up("tree offsets"))?,
            node_feature: stream
                .clone_htod(&flat.feature)
                .with_context(|| up("features"))?,
            node_split: stream
                .clone_htod(&flat.split)
                .with_context(|| up("thresholds"))?,
            node_left: stream
                .clone_htod(&flat.left)
                .with_context(|| up("left children"))?,
            node_right: stream
                .clone_htod(&flat.right)
                .with_context(|| up("right children"))?,
            node_flags: stream
                .clone_htod(&flat.flags)
                .with_context(|| up("flags"))?,
            node_leaf: stream
                .clone_htod(&flat.leaf)
                .with_context(|| up("leaf values"))?,
            node_child_weight: stream
                .clone_htod(&flat.child_weight)
                .with_context(|| up("branch weights"))?,
            node_prev_same: stream
                .clone_htod(&flat.prev_same)
                .with_context(|| up("path offsets"))?,
            node_slot: stream
                .clone_htod(&flat.slot)
                .with_context(|| up("feature slots"))?,
            node_cat_begin: stream
                .clone_htod(&flat.cat_begin)
                .with_context(|| up("category offsets"))?,
            node_cat_len: stream
                .clone_htod(&flat.cat_len)
                .with_context(|| up("category lengths"))?,
            cat_words: stream
                .clone_htod(&flat.cat_words)
                .with_context(|| up("category sets"))?,
            tree_uniq_begin: stream
                .clone_htod(&flat.uniq_begin)
                .with_context(|| up("tree feature offsets"))?,
            uniq_feats: stream
                .clone_htod(&flat.uniq_feats)
                .with_context(|| up("tree features"))?,
            node_mean: stream
                .clone_htod(&flat.mean)
                .with_context(|| up("node means"))?,
            rule: stream
                .clone_htod(&crate::explain::rule_values())
                .with_context(|| up("quadrature rule"))?,
            ix_tree_begin: stream
                .clone_htod(&flat.ix_tree_begin)
                .with_context(|| up("interaction log offsets"))?,
            ix_slot_ptr: stream
                .clone_htod(&flat.ix_slot_ptr)
                .with_context(|| up("interaction slot offsets"))?,
            ix_entry_index: stream
                .clone_htod(&flat.ix_entry_index)
                .with_context(|| up("interaction log entries"))?,
            ix_entry_col: stream
                .clone_htod(&flat.ix_entry_col)
                .with_context(|| up("interaction log columns"))?,
            ix_tree_begin_host: flat.ix_tree_begin,
            sm_count: usize::try_from(
                state
                    .ctx
                    .attribute(
                        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    )
                    .context("query multiprocessor count")?,
            )
            .unwrap_or(1)
            .max(1),
            tree_levels: flat.tree_levels,
            tree_slots: flat.tree_slots,
            n_features: model.n_features,
            n_trees: model.trees.len(),
            k: model.num_outputs(),
            call_lock: Mutex::new(()),
            state,
            stream,
        })
    }
}

impl GpuExplainModel {
    /// Push the 16 model tables, in the order of `SHX_MODEL_PARAMS`.
    fn push_model<'a>(&'a self, l: &mut LaunchArgs<'a>) {
        l.arg(&self.tree_node_begin)
            .arg(&self.node_feature)
            .arg(&self.node_split)
            .arg(&self.node_left)
            .arg(&self.node_right)
            .arg(&self.node_flags)
            .arg(&self.node_leaf)
            .arg(&self.node_child_weight)
            .arg(&self.node_prev_same)
            .arg(&self.node_slot)
            .arg(&self.node_cat_begin)
            .arg(&self.node_cat_len)
            .arg(&self.cat_words)
            .arg(&self.tree_uniq_begin)
            .arg(&self.uniq_feats)
            .arg(&self.node_mean);
    }

    /// Explain `n_rows` row-major rows of the model's features on the device.
    /// The result has the layout (and, bit for bit, the values) of
    /// `Explainer::compute_with_margin` on the host. `margin` is the optional
    /// per-row `base_margin` (row-major `n_rows x groups`).
    pub fn compute(
        &self,
        ex: &Explainer,
        kind: Kind,
        rows: &[f32],
        n_rows: usize,
        margin: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let (nf, k) = (self.n_features, self.k);
        let n_trees = ex.n_trees();
        ensure!(
            n_trees <= self.n_trees && ex.groups() == k,
            "explainer does not match the uploaded model"
        );
        ensure!(
            rows.len() == n_rows.checked_mul(nf).context("input size overflow")?,
            "expected {n_rows} rows x {nf} features = {} values, got {}",
            n_rows * nf,
            rows.len()
        );
        if let Some(m) = margin {
            ensure!(
                m.len() == n_rows.checked_mul(k).context("margin size overflow")?,
                "base_margin has {} values, expected {n_rows} rows x {k} outputs",
                m.len()
            );
        }
        ensure!(
            k <= 65_535,
            "GPU explanation supports at most 65535 outputs"
        );
        let out_width = ex.row_width(kind)?;
        let mut out = vec![
            0.0f32;
            n_rows
                .checked_mul(out_width)
                .context("output size overflow")?
        ];
        if out.is_empty() {
            return Ok(out);
        }
        let nc = nf + 1;
        let levels = self.tree_levels[..n_trees]
            .iter()
            .copied()
            .max()
            .unwrap_or(1) as usize;
        let slots = self.tree_slots[..n_trees]
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let scratch_words = match kind {
            Kind::Contribs { approx: false } => levels * FRAME_WORDS + slots,
            Kind::Contribs { approx: true } => slots,
            Kind::Interactions => levels * FRAME_WORDS,
            Kind::Leaf => 0,
        };
        // Exact contributions of a small batch run one thread per (row, tree)
        // instead of per (row, output) (see `shx_contribs_per_tree`); the
        // scratch then holds one frame stack and slot set per tree.
        let tree_parallel = matches!(kind, Kind::Contribs { approx: false })
            && n_trees > 0
            && n_trees <= MAX_GRID_Y
            && n_rows.saturating_mul(k) < self.sm_count * CONTRIBS_TREE_PARALLEL_PER_SM
            && n_trees
                .saturating_mul(scratch_words)
                .saturating_mul(std::mem::size_of::<f32>())
                <= TREE_PARALLEL_MAX_ROW_SCRATCH;
        // Interactions likewise run one thread per (row, tree) for a small
        // batch, recording values into a per-row log (`shx_interactions_log`).
        let log_words = self.ix_tree_begin_host[n_trees] as usize;
        let interactions_parallel = matches!(kind, Kind::Interactions)
            && n_trees > 0
            && n_trees <= MAX_GRID_Y
            && n_rows.saturating_mul(k) < self.sm_count * INTERACTIONS_TREE_PARALLEL_PER_SM
            && n_trees
                .saturating_mul(scratch_words)
                .saturating_add(log_words)
                .saturating_mul(std::mem::size_of::<f32>())
                <= TREE_PARALLEL_MAX_ROW_SCRATCH;
        let scratch_units = if tree_parallel || interactions_parallel {
            n_trees
        } else {
            k
        };
        let row_log_words = if interactions_parallel { log_words } else { 0 };
        let f32_size = std::mem::size_of::<f32>();
        let per_row_bytes = nf
            .checked_add(if margin.is_some() { k } else { 0 })
            .and_then(|v| v.checked_add(out_width))
            .and_then(|v| v.checked_add(scratch_units.checked_mul(scratch_words)?))
            .and_then(|v| v.checked_add(row_log_words))
            .and_then(|v| v.checked_mul(f32_size))
            .context("GPU explanation row size overflow")?;

        let _serial = self
            .call_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU explanation cache poisoned"))?;
        let (free, _) = self
            .state
            .ctx
            .mem_get_info()
            .context("query free device memory for explanation")?;
        let usable = free
            .saturating_sub(DEVICE_HEADROOM_BYTES)
            .min(MAX_BATCH_BYTES);
        let batch = batch_rows(usable, per_row_bytes, n_rows)?;
        let stream = &self.stream;

        // SAFETY: every region a kernel reads is written first: input and
        // margin are uploaded, the output is zeroed per batch, and the scratch
        // frames / slots are initialised by the kernels before they are read.
        let mut input =
            unsafe { stream.alloc::<f32>(batch * nf) }.context("allocate GPU explanation input")?;
        let mut margin_buf = match margin {
            Some(_) => Some(
                unsafe { stream.alloc::<f32>(batch * k) }
                    .context("allocate GPU explanation margin")?,
            ),
            None => None,
        };
        let mut out_buf = unsafe { stream.alloc::<f32>(batch * out_width) }
            .context("allocate GPU explanation output")?;
        let mut scratch =
            unsafe { stream.alloc::<f32>((batch * scratch_units * scratch_words).max(1)) }
                .context("allocate GPU explanation scratch")?;
        let mut log_buf = unsafe { stream.alloc::<f32>((batch * row_log_words).max(1)) }
            .context("allocate GPU interaction log")?;
        let group_root_mean = stream
            .clone_htod(ex.group_root_mean())
            .context("explanation group means H2D")?;
        let group_bias = stream
            .clone_htod(ex.group_bias())
            .context("explanation group bias H2D")?;

        let mut guard = HostCopyGuard {
            stream,
            done: false,
        };
        let null_margin = 0u64;
        let (n_features, n_trees_u32, k_u32, levels_u32, nc_u32) = (
            to_u32(nf, "feature count")?,
            to_u32(n_trees, "tree count")?,
            to_u32(k, "output count")?,
            to_u32(levels, "tree depth")?,
            to_u32(nc, "feature count")?,
        );
        for first in (0..n_rows).step_by(batch) {
            let r = (n_rows - first).min(batch);
            let r_u32 = to_u32(r, "batch rows")?;
            {
                let mut view = input.slice_mut(..r * nf);
                stream
                    .memcpy_htod(&rows[first * nf..(first + r) * nf], &mut view)
                    .context("explanation input H2D")?;
            }
            if let (Some(m), Some(buf)) = (margin, margin_buf.as_mut()) {
                let mut view = buf.slice_mut(..r * k);
                stream
                    .memcpy_htod(&m[first * k..(first + r) * k], &mut view)
                    .context("explanation margin H2D")?;
            }
            let margin_view = margin_buf.as_ref().map(|b| b.slice(..r * k));
            let input_view = input.slice(..r * nf);
            let mut out_view = out_buf.slice_mut(..r * out_width);
            stream
                .memset_zeros(&mut out_view)
                .context("explanation output memset")?;
            let config = LaunchConfig {
                grid_dim: (r_u32.div_ceil(THREADS), k_u32, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            match kind {
                Kind::Contribs { approx: false } if tree_parallel => {
                    let per_tree = LaunchConfig {
                        grid_dim: (r_u32.div_ceil(THREADS), n_trees_u32, 1),
                        ..config
                    };
                    let mut l = stream.launch_builder(&self.contribs_per_tree_fn);
                    self.push_model(&mut l);
                    l.arg(&self.rule)
                        .arg(&input_view)
                        .arg(&r_u32)
                        .arg(&n_features)
                        .arg(&n_trees_u32)
                        .arg(&levels_u32)
                        .arg(&mut scratch);
                    unsafe { l.launch(per_tree) }.context("launch shx_contribs_per_tree")?;
                    let mut l = stream.launch_builder(&self.contribs_reduce_fn);
                    self.push_model(&mut l);
                    l.arg(&r_u32)
                        .arg(&n_features)
                        .arg(&n_trees_u32)
                        .arg(&k_u32)
                        .arg(&levels_u32)
                        .arg(&group_root_mean)
                        .arg(&group_bias);
                    match &margin_view {
                        Some(v) => l.arg(v),
                        None => l.arg(&null_margin),
                    };
                    l.arg(&scratch).arg(&mut out_view);
                    unsafe { l.launch(config) }.context("launch shx_contribs_reduce")?;
                }
                Kind::Contribs { approx: false } => {
                    let mut l = stream.launch_builder(&self.contribs_fn);
                    self.push_model(&mut l);
                    l.arg(&self.rule)
                        .arg(&input_view)
                        .arg(&r_u32)
                        .arg(&n_features)
                        .arg(&n_trees_u32)
                        .arg(&k_u32)
                        .arg(&levels_u32)
                        .arg(&group_root_mean)
                        .arg(&group_bias);
                    match &margin_view {
                        Some(v) => l.arg(v),
                        None => l.arg(&null_margin),
                    };
                    l.arg(&mut scratch).arg(&mut out_view);
                    unsafe { l.launch(config) }.context("launch shx_contribs")?;
                }
                Kind::Contribs { approx: true } => {
                    let mut l = stream.launch_builder(&self.saabas_fn);
                    self.push_model(&mut l);
                    l.arg(&input_view)
                        .arg(&r_u32)
                        .arg(&n_features)
                        .arg(&n_trees_u32)
                        .arg(&k_u32)
                        .arg(&group_bias);
                    match &margin_view {
                        Some(v) => l.arg(v),
                        None => l.arg(&null_margin),
                    };
                    l.arg(&mut scratch).arg(&mut out_view);
                    unsafe { l.launch(config) }.context("launch shx_saabas")?;
                }
                Kind::Interactions => {
                    if interactions_parallel {
                        let per_tree = LaunchConfig {
                            grid_dim: (r_u32.div_ceil(THREADS), n_trees_u32, 1),
                            ..config
                        };
                        let mut l = stream.launch_builder(&self.interactions_log_fn);
                        self.push_model(&mut l);
                        l.arg(&self.rule)
                            .arg(&input_view)
                            .arg(&r_u32)
                            .arg(&n_features)
                            .arg(&n_trees_u32)
                            .arg(&self.ix_tree_begin)
                            .arg(&mut scratch)
                            .arg(&mut log_buf);
                        unsafe { l.launch(per_tree) }.context("launch shx_interactions_log")?;
                        let cells =
                            u32::try_from(r * nc).context("GPU interaction batch too large")?;
                        let apply = LaunchConfig {
                            grid_dim: (cells.div_ceil(256), k_u32, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut l = stream.launch_builder(&self.interactions_apply_fn);
                        self.push_model(&mut l);
                        l.arg(&r_u32)
                            .arg(&n_features)
                            .arg(&n_trees_u32)
                            .arg(&k_u32)
                            .arg(&self.ix_tree_begin)
                            .arg(&self.ix_slot_ptr)
                            .arg(&self.ix_entry_index)
                            .arg(&self.ix_entry_col)
                            .arg(&log_buf)
                            .arg(&mut out_view);
                        unsafe { l.launch(apply) }.context("launch shx_interactions_apply")?;
                    } else {
                        let mut l = stream.launch_builder(&self.interactions_fn);
                        self.push_model(&mut l);
                        l.arg(&self.rule)
                            .arg(&input_view)
                            .arg(&r_u32)
                            .arg(&n_features)
                            .arg(&n_trees_u32)
                            .arg(&k_u32)
                            .arg(&mut scratch)
                            .arg(&mut out_view);
                        unsafe { l.launch(config) }.context("launch shx_interactions")?;
                    }
                    let groups_total = (r * k) as u64;
                    let pair_threads =
                        u32::try_from(r * k * nc).context("GPU interaction batch too large")?;
                    let post = LaunchConfig {
                        grid_dim: (pair_threads.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut l = stream.launch_builder(&self.symmetrize_fn);
                    l.arg(&mut out_view).arg(&groups_total).arg(&nc_u32);
                    unsafe { l.launch(post) }.context("launch shx_interactions_symmetrize")?;
                    let mut l = stream.launch_builder(&self.diagonal_fn);
                    l.arg(&mut out_view)
                        .arg(&groups_total)
                        .arg(&nc_u32)
                        .arg(&k_u32)
                        .arg(&group_root_mean)
                        .arg(&group_bias);
                    match &margin_view {
                        Some(v) => l.arg(v),
                        None => l.arg(&null_margin),
                    };
                    unsafe { l.launch(post) }.context("launch shx_interactions_diagonal")?;
                }
                Kind::Leaf => {
                    let leaf_config = LaunchConfig {
                        grid_dim: (r_u32.div_ceil(THREADS), 1, 1),
                        ..config
                    };
                    let mut l = stream.launch_builder(&self.leaf_fn);
                    self.push_model(&mut l);
                    l.arg(&input_view)
                        .arg(&r_u32)
                        .arg(&n_features)
                        .arg(&n_trees_u32)
                        .arg(&mut out_view);
                    unsafe { l.launch(leaf_config) }.context("launch shx_leaf")?;
                }
            }
            stream
                .memcpy_dtoh(
                    &out_view,
                    &mut out[first * out_width..(first + r) * out_width],
                )
                .context("explanation output D2H")?;
            // Per-batch sync: the host slice must be complete before the next
            // batch reuses the staging buffers.
            stream
                .synchronize()
                .context("wait for GPU explanation to finish")?;
        }
        guard.done = true;
        drop(guard);
        // Release the staging before returning (stream-ordered frees).
        drop((input, margin_buf, out_buf, scratch, log_buf));
        stream
            .synchronize()
            .context("release explanation staging")?;
        Ok(out)
    }
}
