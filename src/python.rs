//! Python 绑定。
//!
//! **这一层不做任何数值处理。** 参数解析、dtype 归一化、输入类型分发
//! 都在 `python/ferrisboost/__init__.py` 里;这里只负责把已经规整好的
//! 数据交给训练循环。预测是例外:它是公开的 PyO3 边界，必须正确解释
//! NumPy 的 stride，不能把 F-order/切片视为 C row-major 内存。

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(not(feature = "cuda"))]
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::callback::{Callback, EarlyStopping, RecordHistory, RoundMetrics, VerboseEval};
use crate::columns::BinningStrategy;
use crate::comm::Local;
use crate::source::parquet_source::{ParquetBlockSource, QuantizedCacheOptions};
use crate::source::{csv_source, parquet_source, DenseSource};
use crate::train::{train_weighted, BlockSource, EvalSet, TrainConfig};

mod explain;
mod margin;

fn on_off(flag: bool) -> &'static str {
    if flag {
        "on"
    } else {
        "off"
    }
}
use crate::tree::Model;
use crate::types::{Objective, TrainParams};

/// anyhow 的错误转 Python 异常。
///
/// 尽量给具体的异常类型:文件找不到就是 FileNotFoundError,参数不对
/// (带 `crate::error::InputError` 标记的校验错误)就是 ValueError,
/// 其余(I/O、CUDA、内部错误)才是 RuntimeError。全都甩成 RuntimeError 的话,调用方没法按类型处理,
/// 而且 traceback 里只剩一句没头没尾的话。
fn to_py_err(e: anyhow::Error) -> PyErr {
    let msg = format!("{e:#}");
    if let Some(io) = e.downcast_ref::<std::io::Error>() {
        if io.kind() == std::io::ErrorKind::NotFound {
            return pyo3::exceptions::PyFileNotFoundError::new_err(msg);
        }
    }
    // Rejected input is tagged where it is validated (`crate::error`); the two
    // message checks cover file-open and missing-column errors that predate it.
    if crate::error::is_input_error(&e)
        || msg.contains("cannot open ")
        || msg.contains("has no column")
    {
        return PyValueError::new_err(msg);
    }
    PyRuntimeError::new_err(msg)
}

/// panic 不能穿过 FFI 边界(那是 UB),兜住转成异常。
fn guard<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "Rust panic without a message".to_string());
            Err(PyRuntimeError::new_err(format!(
                "ferrisboost internal error: {msg}"
            )))
        }
    }
}

/// 用户传进来的 Python 回调。
///
/// 训练主体是在 `allow_threads` 里跑的(GIL 已经放掉),所以进 Python
/// 之前必须重新拿 GIL。这也是最容易死锁的地方:回调里**不要**再去拿
/// 其它 Rust 侧的锁。
struct PyCallback {
    func: Py<PyAny>,
    /// 所有用户回调共用:第一个抛出的异常存在这里,训练返回后原样抛出。
    ///
    /// ⚠️ 不能靠 `PyErr::restore` 把异常留在解释器里:后面的回调再进
    /// Python 会把它冲掉,训练于是静默返回一个被截断的模型。
    error: Arc<Mutex<Option<PyErr>>>,
}

impl Callback for PyCallback {
    fn after_iteration(&mut self, m: &RoundMetrics) -> bool {
        if self.error.lock().expect("callback error slot").is_some() {
            return false; // 已经有回调抛了异常:后面的不再调用
        }
        Python::attach(|py| {
            let metrics = PyDict::new(py);
            for (eval_name, metric, value) in &m.entries {
                let per_eval = match metrics.get_item(eval_name) {
                    Ok(Some(d)) => d
                        .cast_into::<PyDict>()
                        .expect("checked to be a PyDict above"),
                    _ => {
                        let d = PyDict::new(py);
                        metrics.set_item(eval_name, &d).ok();
                        d
                    }
                };
                per_eval.set_item(metric, *value).ok();
            }
            match self.func.call1(py, (m.round, metrics)) {
                // 只有显式返回 False 才停;返回 None 当继续
                Ok(ret) => !matches!(ret.extract::<bool>(py), Ok(false)),
                Err(e) => {
                    *self.error.lock().expect("callback error slot") = Some(e);
                    false // 回调抛异常就停下来,异常在训练返回后抛出去
                }
            }
        })
    }
}

/// 一次分块决策的**请求值**和**生效值**。
///
/// 把两者分开是为了别让「用户要什么」和「实际用了什么」混成一个数:
/// benchmark 和日志要能同时看到 requested / effective / 块数 / 线程数,
/// 否则自动选择一旦选错,现场根本看不出来是谁定的。
#[derive(Clone, Copy, Debug)]
struct BlockingPlan {
    #[allow(dead_code)]
    // Retain requested/resolved diagnostic metadata alongside execution geometry.
    requested: Option<usize>,
    effective: usize,
    #[allow(dead_code)] // Retained plan metadata; not every path consumes it yet.
    n_blocks: usize,
    #[allow(dead_code)] // Retained plan metadata; not every path consumes it yet.
    nthread: usize,
    /// 显存规划器解出来的 stream 数 / 常驻块数。
    ///
    /// ⚠️ **规划器解出来的几何必须原样到达运行时。** 三者(块宽、stream 数、
    /// 常驻块数)是在**同一个内存模型**里一起解的;运行时若按自己的默认
    /// 再挑一个,预算和实际分配就对不上 —— 规划器为了装下而选了 1 条流,
    /// 运行时却按路径又开了 2 条,就是一次 OOM。
    /// `None` = 规划器尚未参与(仅 CPU 或 deferred source 的 pre-open 几何)。
    hist_streams: Option<usize>,
    resident_blocks: Option<usize>,
    /// 规划器实际用的预算(字节),= `min(用户给的, 空闲显存)`。
    /// 记下来是为了让运行时回显说出**生效值**,而不是原始请求值。
    effective_gpu_budget: Option<usize>,
}

impl BlockingPlan {
    /// 已经定好的宽度(用户显式给的,或规划器解出来的)。
    fn explicit(effective: usize, n_features: usize, nthread: usize) -> Self {
        Self {
            requested: Some(effective),
            effective,
            n_blocks: n_features.max(1).div_ceil(effective.max(1)),
            nthread,
            hist_streams: None,
            resident_blocks: None,
            effective_gpu_budget: None,
        }
    }

    /// 显存规划器解出来的完整几何:块宽 + stream 数 + 常驻块数。
    fn from_plan(
        p: &crate::gpu_mem_plan::GpuStreamingPlan,
        n_features: usize,
        nthread: usize,
    ) -> Self {
        let effective = (p.resolved_cols_per_block as usize).max(1);
        Self {
            requested: Some(effective),
            effective,
            n_blocks: n_features.max(1).div_ceil(effective),
            nthread,
            hist_streams: Some(p.resolved_hist_streams as usize),
            resident_blocks: Some(p.resident_blocks as usize),
            effective_gpu_budget: Some(p.free_vram as usize),
        }
    }

    /// CPU blocking: `requested = None` 才跑启发式;显式给值就原样用。
    fn resolve(
        requested: Option<usize>,
        n_features: usize,
        nthread: usize,
        max_bin: usize,
    ) -> Self {
        let effective = match requested {
            Some(c) if c > 0 => c,
            _ => {
                crate::columns::auto_cols_per_block(n_features, effective_nthread(nthread), max_bin)
            }
        };
        Self {
            requested,
            effective,
            n_blocks: n_features.max(1).div_ceil(effective.max(1)),
            nthread,
            hist_streams: None,
            resident_blocks: None,
            effective_gpu_budget: None,
        }
    }
}

/// CPU 走原来的缓存友好启发式;GPU 走显存预算 + 块数拐点的规划器。
/// GPU 上显式 `cols_per_block` 只约束物理几何;stream / residency / budget
/// 仍由同一个显存规划器完成,不能因为手动定宽就退回 runtime 默认值。
fn resolve_blocking(
    cfg: &Parsed,
    n_features: usize,
    // ⚠️ **真实行数**,不是轮数。显存模型的每一项都按行数走(gpair、
    // row index、prediction、列块 buffer),传错这个数解出来的块宽在
    // 真实数据上根本放不下。`None` = 拿不到(例如 CSV 不读完不知道行数)。
    n_rows: Option<u64>,
) -> PyResult<BlockingPlan> {
    match cfg.device {
        #[cfg(feature = "cuda")]
        Some(ordinal) => {
            use crate::gpu_mem_plan::resolve_gpu_streaming_plan;
            // ⚠️ **必须查用户真正要用的那张卡。** 这里以前固定
            // `CudaContext::new(0)`,而 buffer 实际分配在 `train_gpu(ordinal, ..)`
            // 指定的卡上 —— `device="cuda:1"` 时预算读的是 GPU 0 的空闲显存,
            // 分配却发生在 GPU 1。两张卡型号不同、或占用不同时,轻则规划过小
            // (白白 streaming),重则在 GPU 1 上 **OOM**。
            // 规划和分配必须看同一张卡。
            let ctx = crate::backend::cuda::cuda_context(ordinal).map_err(|e| {
                PyValueError::new_err(format!("cannot open CUDA device {ordinal}: {e}"))
            })?;
            let (free, total) = ctx.mem_get_info().map_err(|e| {
                PyValueError::new_err(format!("cannot read device memory info: {e}"))
            })?;
            // ⚠️ 列块 buffer 是 streaming 预算里最大的一项,而 `per_col_bytes`
            // **整个乘以 stream 数** —— 所以块宽和 stream 数必须**一起解**。
            // 分开解会得到一个装不下的组合:规划器按单流选了宽块,运行时
            // 却按路径开了双流,然后 CUDA OOM(而不是明确的规划失败)。
            // ⚠️ **拿不到真实行数就明确报错,不要拿别的数顶上。**
            // 这里曾经传的是 `n_rounds`(注释还写着"下面用真实行数覆盖",
            // 而下面根本没有覆盖),于是模型把上千万行当成几十行算,
            // 解出来的块宽在真实数据上放不下。
            // 「规划失败必须报错,不能悄悄回退」——顶一个假行数比报错更糟。
            let rows = n_rows.ok_or_else(|| {
                PyValueError::new_err(
                    "GPU planning needs the row count, but this input source cannot provide it before open; construct the source with an explicit cols_per_block, and planning finishes with the real aggregate row count after open",
                )
            })?;
            let p = resolve_gpu_streaming_plan(
                cfg.requested_cols_per_block,
                cfg.params.hist_streams,
                cfg.params.resident_blocks,
                n_features as u64,
                cfg.params
                    .gpu_memory_budget
                    .map_or(free as u64, |b| (b as u64).min(free as u64)),
                total as u64,
                |streams| gpu_plan_model(cfg, rows, streams, None),
            )
            .map_err(to_py_err)?;
            if !cfg.quiet {
                eprintln!("{}", p.report());
            }
            Ok(BlockingPlan::from_plan(&p, n_features, cfg.params.nthread))
        }
        #[cfg(not(feature = "cuda"))]
        Some(_) => Err(PyValueError::new_err(
            "this build has no CUDA support: rebuild with --features cuda, or use device=\"cpu\"",
        )),
        None => Ok(BlockingPlan::resolve(
            cfg.requested_cols_per_block,
            n_features,
            cfg.params.nthread,
            cfg.params.max_bin as usize,
        )),
    }
}

/// The streaming memory model the GPU planner prices one candidate stream
/// count with. Pre-open (`shape = None`) every column is priced at the scalar
/// `max_bin` and one byte per cell; after open, `Some((shape, cols))` prices the
/// real blocks (16-bit cells, actual per-feature bins) at the fixed width.
#[cfg(feature = "cuda")]
fn gpu_plan_model(
    cfg: &Parsed,
    rows: u64,
    streams: u64,
    shape: Option<(&crate::gpu_mem_plan::GpuBlockShape, u64)>,
) -> crate::gpu_mem_plan::StreamingMemModel {
    use crate::gpu_mem_plan::StreamingMemModel;
    let fast = matches!(cfg.params.gpu_math, crate::types::GpuMath::Fast);
    let max_bin = cfg.params.max_bin as u64;
    let cap = cfg.params.hist_nodes_per_batch.unwrap_or(16) as u64;
    let device_quantize = cfg.params.device_quantize.unwrap_or(true);
    let model = match shape {
        None => StreamingMemModel::new(
            rows,
            max_bin,
            streams,
            cap,
            device_quantize,
            fast,
            true,
            crate::gpu_mem_plan::RM_STAGE_BYTES,
        ),
        Some((shape, cols)) => StreamingMemModel::new_for_shape(
            rows,
            shape,
            cols,
            streams,
            cap,
            device_quantize,
            fast,
            crate::gpu_mem_plan::RM_STAGE_BYTES,
        ),
    };
    // fast 模式下梯度权重常驻显存(4 B/行);exact 在 host 上乘,不占显存。
    // 梯度权重 = 样本权重,或 scale_pos_weight ≠ 1 时的隐式权重 ——
    // 和训练时的分配用同一个判断。
    let model = if fast
        && crate::train::has_gradient_weights(
            cfg.objective,
            cfg.weighted,
            cfg.params.scale_pos_weight,
        ) {
        model.with_fast_weights(rows)
    } else {
        model
    };
    // fast 多分类:K 份常驻 prediction + softmax 统计量。
    // exact 的多分类在 host 算梯度,device 上不多占。
    let model = if fast && cfg.objective.is_multi_quantile() {
        // 多个 quantile_alpha:K 份 prediction,没有 softmax 统计量。
        model.with_fast_multi_quantile(rows, cfg.objective.num_outputs() as u64)
    } else if fast {
        model.with_fast_multiclass(rows, cfg.objective.num_outputs() as u64)
    } else {
        model
    };
    // fast ranking: the device LambdaMART buffers. The group count is
    // known up front for arrays; file inputs plan with the row count
    // (an upper bound) and allocate for the real groups.
    let model = match (fast, cfg.objective.rank()) {
        (true, Some((kind, params))) => model.with_fast_rank(
            rows,
            &crate::gpu_mem_plan::FastRankShape {
                n_groups: cfg
                    .group_ptr
                    .as_ref()
                    .map_or(rows, |g| (g.len() - 1) as u64),
                map: kind == crate::rank::RankKind::Map,
                mean: params.pair_method == crate::rank::PairMethod::Mean,
                // the largest group is not known for file inputs: the
                // row count is an upper bound
                n_positions: params.device_bias_positions(
                    cfg.group_ptr
                        .as_deref()
                        .map_or(rows as usize, crate::rank::max_group_size),
                ),
            },
        ),
        _ => model,
    };
    if !cfg.params.gpu_split_cache
        || matches!(
            std::env::var("FB_HIST_BUCKET_PROFILE").as_deref(),
            Ok("1") | Ok("true")
        )
    {
        return model;
    }
    let node_masks = cfg.params.colsample_bylevel < 1.0 || cfg.params.colsample_bynode < 1.0;
    match shape {
        // Categorical set outputs are reserved unconditionally:
        // Arrow dictionary columns make a file input categorical
        // without a declaration (tiny: ~0.5 KB / column / slot).
        None => {
            let model = model
                .with_resident_split_cache(max_bin, cap)
                .with_categorical_split_cache(cap);
            if node_masks {
                model.with_node_feature_masks(cap)
            } else {
                model
            }
        }
        Some((shape, cols)) => {
            model.with_resident_split_cache_for_shape(shape, cols, cap, true, node_masks)
        }
    }
}

/// The GPU geometry the user asked for, before the planner overwrote
/// `TrainParams` with its resolved values.
#[derive(Clone, Copy)]
struct UserGpuGeometry {
    hist_streams: Option<usize>,
    resident_blocks: Option<usize>,
    gpu_memory_budget: Option<usize>,
}

fn user_gpu_geometry(cfg: &Parsed) -> UserGpuGeometry {
    UserGpuGeometry {
        hist_streams: cfg.params.hist_streams,
        resident_blocks: cfg.params.resident_blocks,
        gpu_memory_budget: cfg.params.gpu_memory_budget,
    }
}

/// Re-plan the GPU memory once the source is open and its real blocks are
/// known, when the pre-open pricing could not know them: any 16-bit block
/// (features with more than 255 bins or 254 categories) or `max_bin > 255`
/// (pre-open pricing charges every column `max_bin` bins). The block width is
/// kept (the source is already built with it); streams and residency are
/// solved again against the real block count and slot bytes, from the user's
/// original requests. 8-bit data with `max_bin <= 255` keeps its pre-open plan.
#[allow(unused_variables)]
fn refine_gpu_plan_for_source(
    cfg: &Parsed,
    user: UserGpuGeometry,
    n_features: usize,
    source: &dyn crate::train::BlockSource,
    params: &mut crate::types::TrainParams,
) -> PyResult<()> {
    #[cfg(feature = "cuda")]
    if let Some(ordinal) = cfg.device {
        let shape = crate::train::gpu_block_shape(source).map_err(to_py_err)?;
        if !shape.has_wide() && cfg.params.max_bin <= 255 {
            return Ok(());
        }
        let ctx = crate::backend::cuda::cuda_context(ordinal).map_err(|e| {
            PyValueError::new_err(format!("cannot open CUDA device {ordinal}: {e}"))
        })?;
        let (free, total) = ctx
            .mem_get_info()
            .map_err(|e| PyValueError::new_err(format!("cannot read device memory info: {e}")))?;
        let budget = user
            .gpu_memory_budget
            .map_or(free as u64, |b| (b as u64).min(free as u64));
        let rows = source.n_rows() as u64;
        let cols = params.cols_per_block.max(1);
        let model = |streams| gpu_plan_model(cfg, rows, streams, Some((&shape, cols as u64)));
        let plan = |streams: Option<usize>| {
            crate::gpu_mem_plan::resolve_gpu_streaming_plan_for_blocks(
                cols,
                streams,
                user.resident_blocks,
                n_features as u64,
                source.n_blocks() as u64,
                budget,
                total as u64,
                model,
            )
            .map_err(to_py_err)
        };
        // Streaming-only floor of a plan: its block width and streams, nothing
        // resident. The second stream is dropped (when the user did not fix
        // the count) before the configuration is declared too large.
        let floor = |p: &crate::gpu_mem_plan::GpuStreamingPlan| {
            model(p.resolved_hist_streams).peak_for_layout(cols as u64, p.resolved_hist_streams, 0)
        };
        let mut p = plan(user.hist_streams)?;
        if user.hist_streams.is_none()
            && p.resolved_hist_streams > 1
            && floor(&p) > budget.saturating_sub(p.reserve)
        {
            p = plan(Some(1))?;
        }
        let floor = floor(&p);
        if floor > budget.saturating_sub(p.reserve) {
            return Err(PyValueError::new_err(format!(
                "the streaming configuration of these column blocks does not fit in GPU memory: \
                 needs about {:.1} MB, budget is only {:.1} MB (free {:.1} MB − reserve {:.1} MB); \
                 lower cols_per_block or max_bin, or raise gpu_memory_budget",
                floor as f64 / 1048576.0,
                budget.saturating_sub(p.reserve) as f64 / 1048576.0,
                budget as f64 / 1048576.0,
                p.reserve as f64 / 1048576.0,
            )));
        }
        if !cfg.quiet {
            eprintln!(
                "{}\n               priced from the opened blocks: {} blocks, widest {} B/row, \
                 largest histogram {} bins",
                p.report(),
                source.n_blocks(),
                shape.max_block_row_bytes,
                shape.max_hist_bins
            );
        }
        params.hist_streams = Some(p.resolved_hist_streams as usize);
        params.resident_blocks = Some(p.resident_blocks as usize);
        params.gpu_memory_budget = Some(p.free_vram as usize);
    }
    Ok(())
}

/// 把规划器解出来的几何写回训练参数。
///
/// ⚠️ **只在规划器真的参与时才覆盖。** CPU 或 deferred source 的
/// pre-open 几何里这些值是 `None`;deferred plan 在 open 后必须再调用一次。
fn apply_resolved_geometry(cfg: &mut Parsed, plan: &BlockingPlan) {
    if let Some(n) = plan.hist_streams {
        cfg.params.hist_streams = Some(n);
    }
    if let Some(n) = plan.resident_blocks {
        cfg.params.resident_blocks = Some(n);
    }
    // 规划之后这个字段保存**生效预算**,回显读的就是它。
    if let Some(b) = plan.effective_gpu_budget {
        cfg.params.gpu_memory_budget = Some(b);
    }
}

fn process_read_chars() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/io").ok()?;
    text.lines().find_map(|line| {
        line.strip_prefix("rchar:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

/// Resolve file-ingest concurrency once from stable, observable inputs.
/// Decoder and column-transform stages divide this resolved total internally.
/// This scales with the machine and replaces the old machine-specific
/// four-worker cap. Explicit values bypass the auto CPU policy, while work and
/// transient-memory bounds still apply.
fn resolved_ingest_workers(
    requested: usize,
    fmt: &str,
    paths: &[std::path::PathBuf],
    n_features: usize,
    cols_per_block: usize,
) -> usize {
    let feature_blocks = n_features.max(1).div_ceil(cols_per_block.max(1));
    let work_units = if fmt == "parquet" {
        paths
            .iter()
            .map(|path| crate::source::parquet_source::peek_n_row_groups(path).unwrap_or(1))
            .sum::<usize>()
            .max(feature_blocks)
    } else if paths.len() == 1 {
        let bytes = std::fs::metadata(&paths[0])
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        bytes
            .max(1)
            .div_ceil(crate::source::csv_source::CSV_CHUNK_BYTES)
            .max(feature_blocks)
    } else {
        paths.len().max(feature_blocks)
    };
    let bytes_per_worker = if fmt == "parquet" {
        // One decoded Arrow batch, conversion scratch, and the maximum bounded
        // dense-projection Parquet read window. Sparse readers do not allocate it.
        n_features
            .saturating_mul(8192)
            .saturating_mul(8)
            .saturating_add(16 * 1024 * 1024)
            .saturating_add(if n_features >= 256 {
                crate::source::parquet_source::PARQUET_READ_WINDOW_BYTES
            } else {
                0
            })
    } else {
        // Bounded record chunk, parsed Arrow arrays, and channel slack.
        crate::source::csv_source::CSV_CHUNK_BYTES
            .saturating_mul(8)
            .saturating_add(16 * 1024 * 1024)
    };
    crate::threading::bounded_io_workers(requested, work_units, bytes_per_worker)
}

/// Low-frequency ingest telemetry. It samples process I/O outside reader and
/// quantization hot paths, so enabling normal logs adds no per-batch locking.
struct IngestReporter {
    quiet: bool,
    started: std::time::Instant,
    read0: Option<u64>,
    files: usize,
    workers: usize,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl IngestReporter {
    fn start(
        quiet: bool,
        fmt: &'static str,
        paths: &[std::path::PathBuf],
        requested: usize,
        workers: usize,
        rows_total: Option<u64>,
    ) -> Self {
        use std::sync::atomic::Ordering;
        let started = std::time::Instant::now();
        let read0 = process_read_chars();
        let files = paths.len();
        let bytes_total = paths
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok().map(|m| m.len()))
            .sum::<u64>();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        if quiet {
            return Self {
                quiet,
                started,
                read0,
                files,
                workers,
                stop,
                handle: None,
            };
        }
        let requested = if requested == 0 {
            "auto".to_string()
        } else {
            requested.to_string()
        };
        match rows_total {
            Some(rows) => eprintln!(
                "INGEST start input={fmt} files={files} bytes_total={bytes_total} rows_total={rows} ingest_threads={requested} workers={workers}"
            ),
            None => eprintln!(
                "INGEST start input={fmt} files={files} bytes_total={bytes_total} rows_total=unknown ingest_threads={requested} workers={workers}"
            ),
        }
        let thread_stop = std::sync::Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                std::thread::park_timeout(std::time::Duration::from_secs(3));
                if thread_stop.load(Ordering::Relaxed) {
                    break;
                }
                let elapsed = started.elapsed().as_secs_f64();
                let bytes = process_read_chars()
                    .zip(read0)
                    .map(|(now, before)| now.saturating_sub(before));
                match bytes {
                    Some(bytes) => eprintln!(
                        "INGEST progress files={files} bytes_read={bytes} workers={workers} throughput={:.1}MiB/s elapsed={elapsed:.1}s",
                        bytes as f64 / (1024.0 * 1024.0) / elapsed.max(1e-9)
                    ),
                    None => eprintln!(
                        "INGEST progress files={files} workers={workers} elapsed={elapsed:.1}s"
                    ),
                }
            }
        });
        Self {
            quiet,
            started,
            read0,
            files,
            workers,
            stop,
            handle: Some(handle),
        }
    }

    fn complete(mut self, rows: usize) {
        self.stop_thread();
        if !self.quiet {
            let elapsed = self.started.elapsed().as_secs_f64();
            let bytes = process_read_chars()
                .zip(self.read0)
                .map(|(now, before)| now.saturating_sub(before));
            match bytes {
                Some(bytes) => eprintln!(
                    "INGEST complete files={} rows={} bytes_read={} workers={} throughput={:.1}MiB/s elapsed={elapsed:.3}s",
                    self.files,
                    rows,
                    bytes,
                    self.workers,
                    bytes as f64 / (1024.0 * 1024.0) / elapsed.max(1e-9)
                ),
                None => eprintln!(
                    "INGEST complete files={} rows={} workers={} elapsed={elapsed:.3}s",
                    self.files, rows, self.workers
                ),
            }
        }
    }

    fn stop_thread(&mut self) {
        use std::sync::atomic::Ordering;
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

impl Drop for IngestReporter {
    fn drop(&mut self) {
        self.stop_thread();
    }
}

const MODEL_IO_CHUNK: usize = 1024 * 1024;

fn resolved_model_io_workers(total: usize, requested: usize) -> usize {
    crate::threading::bounded_io_workers(
        requested,
        total.max(1).div_ceil(MODEL_IO_CHUNK),
        MODEL_IO_CHUNK,
    )
}

struct ModelIoReporter {
    quiet: bool,
    kind: &'static str,
    total: usize,
    workers: usize,
    started: std::time::Instant,
    processed: std::sync::Arc<std::sync::atomic::AtomicU64>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ModelIoReporter {
    fn start(
        quiet: bool,
        kind: &'static str,
        path: &str,
        total: usize,
        requested: usize,
        workers: usize,
    ) -> Self {
        use std::sync::atomic::Ordering;
        let started = std::time::Instant::now();
        let processed = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        if quiet {
            return Self {
                quiet,
                kind,
                total,
                workers,
                started,
                processed,
                stop,
                handle: None,
            };
        }
        let requested = if requested == 0 {
            "auto".to_string()
        } else {
            requested.to_string()
        };
        eprintln!(
            "MODEL_{kind} start path={path} bytes_total={total} model_io_threads={requested} workers={workers}"
        );
        let thread_processed = std::sync::Arc::clone(&processed);
        let thread_stop = std::sync::Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                std::thread::park_timeout(std::time::Duration::from_secs(3));
                if thread_stop.load(Ordering::Relaxed) {
                    break;
                }
                let bytes = thread_processed.load(Ordering::Relaxed);
                eprintln!(
                    "MODEL_{kind} progress bytes={bytes}/{total} workers={workers} elapsed={:.1}s",
                    started.elapsed().as_secs_f64()
                );
            }
        });
        Self {
            quiet,
            kind,
            total,
            workers,
            started,
            processed,
            stop,
            handle: Some(handle),
        }
    }

    fn counter(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        std::sync::Arc::clone(&self.processed)
    }

    fn complete(mut self) {
        self.stop_thread();
        if !self.quiet {
            eprintln!(
                "MODEL_{} complete bytes={}/{} workers={} elapsed={:.3}s",
                self.kind,
                self.processed.load(std::sync::atomic::Ordering::Relaxed),
                self.total,
                self.workers,
                self.started.elapsed().as_secs_f64()
            );
        }
    }

    fn stop_thread(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

impl Drop for ModelIoReporter {
    fn drop(&mut self) {
        self.stop_thread();
    }
}

#[cfg(unix)]
fn write_all_at(file: &std::fs::File, mut bytes: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !bytes.is_empty() {
        let n = file.write_at(bytes, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "write_at returned 0",
            ));
        }
        bytes = &bytes[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &std::fs::File, mut bytes: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !bytes.is_empty() {
        let n = file.seek_write(bytes, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "seek_write returned 0",
            ));
        }
        bytes = &bytes[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(
    file: &std::fs::File,
    mut bytes: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !bytes.is_empty() {
        let n = file.read_at(bytes, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "read_at reached EOF",
            ));
        }
        let (_, rest) = bytes.split_at_mut(n);
        bytes = rest;
        offset += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(
    file: &std::fs::File,
    mut bytes: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !bytes.is_empty() {
        let n = file.seek_read(bytes, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "seek_read reached EOF",
            ));
        }
        let (_, rest) = bytes.split_at_mut(n);
        bytes = rest;
        offset += n as u64;
    }
    Ok(())
}

#[cfg(any(unix, windows))]
fn parallel_model_write(
    path: &str,
    bytes: &[u8],
    requested: usize,
    quiet: bool,
) -> anyhow::Result<()> {
    use std::sync::atomic::Ordering;
    let workers = resolved_model_io_workers(bytes.len(), requested);
    let reporter = ModelIoReporter::start(quiet, "SAVE", path, bytes.len(), requested, workers);
    crate::model_io::atomic_write(std::path::Path::new(path), |file| {
        file.set_len(bytes.len() as u64)?;
        let progress = reporter.counter();
        let chunk_size = bytes.len().max(1).div_ceil(workers);
        std::thread::scope(|scope| -> anyhow::Result<()> {
            let mut handles = Vec::new();
            for (index, chunk) in bytes.chunks(chunk_size).enumerate() {
                let output = file.try_clone()?;
                let progress = std::sync::Arc::clone(&progress);
                handles.push(scope.spawn(move || -> std::io::Result<()> {
                    let base = index * chunk_size;
                    for (part_index, part) in chunk.chunks(MODEL_IO_CHUNK).enumerate() {
                        write_all_at(&output, part, (base + part_index * MODEL_IO_CHUNK) as u64)?;
                        progress.fetch_add(part.len() as u64, Ordering::Relaxed);
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("model save worker panicked"))??;
            }
            Ok(())
        })?;
        Ok(())
    })?;
    reporter.complete();
    Ok(())
}

#[cfg(any(unix, windows))]
fn parallel_model_read(path: &str, requested: usize, quiet: bool) -> anyhow::Result<Vec<u8>> {
    use std::sync::atomic::Ordering;
    let file = std::fs::File::open(path)?;
    let total = usize::try_from(file.metadata()?.len())
        .map_err(|_| anyhow::anyhow!("model file is too large for this process"))?;
    let workers = resolved_model_io_workers(total, requested);
    let reporter = ModelIoReporter::start(quiet, "LOAD", path, total, requested, workers);
    let progress = reporter.counter();
    let mut bytes = vec![0u8; total];
    let chunk_size = total.max(1).div_ceil(workers);
    std::thread::scope(|scope| -> anyhow::Result<()> {
        let mut handles = Vec::new();
        for (index, chunk) in bytes.chunks_mut(chunk_size).enumerate() {
            let input = file.try_clone()?;
            let progress = std::sync::Arc::clone(&progress);
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                let base = index * chunk_size;
                for (part_index, part) in chunk.chunks_mut(MODEL_IO_CHUNK).enumerate() {
                    read_exact_at(&input, part, (base + part_index * MODEL_IO_CHUNK) as u64)?;
                    progress.fetch_add(part.len() as u64, Ordering::Relaxed);
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("model load worker panicked"))??;
        }
        Ok(())
    })?;
    reporter.complete();
    Ok(bytes)
}

// Detached model I/O carries native errors only. Python exceptions are built
// after reattachment, preserving the existing exception classes/messages.
enum ModelIoError {
    Runtime(String),
    Value(String),
    FileNotFound(String),
}

impl From<ModelIoError> for PyErr {
    fn from(error: ModelIoError) -> Self {
        match error {
            ModelIoError::Runtime(message) => PyRuntimeError::new_err(message),
            ModelIoError::Value(message) => PyValueError::new_err(message),
            ModelIoError::FileNotFound(message) => {
                pyo3::exceptions::PyFileNotFoundError::new_err(message)
            }
        }
    }
}

/// 启发式必须按**真正会起来的线程数**算,所以直接复用建线程池那一套
/// 解析(`nthread=0` → 物理核数),而不是自己再实现一遍 ——
/// 两处如果各算各的,块宽就会按一个和实际线程数不同的值来切。
use crate::threading::effective_threads as effective_nthread;

/// Auto prediction stays serial below this conservative work estimate. The
/// estimate uses the maximum traversed path in each selected tree; the exact
/// crossover is benchmarked independently.
const AUTO_PREDICT_MIN_WORK: usize = 256 * 1024;

fn prediction_work_per_row(model: &Model, n_trees: usize) -> usize {
    fn tree_path_upper_bound(tree: &crate::tree::Tree) -> usize {
        if tree.nodes.is_empty() {
            return 0;
        }
        let mut longest = 1usize;
        let mut stack = vec![(0usize, 1usize)];
        while let Some((node_id, depth)) = stack.pop() {
            longest = longest.max(depth);
            let node = &tree.nodes[node_id];
            if !node.is_leaf {
                stack.push((node.left as usize, depth + 1));
                stack.push((node.right as usize, depth + 1));
            }
        }
        longest
    }

    model.trees[..n_trees.min(model.trees.len())]
        .iter()
        .map(tree_path_upper_bound)
        .sum::<usize>()
        .max(1)
}

fn resolved_predict_workers(
    requested: usize,
    worker_cap: usize,
    n_rows: usize,
    work_per_row: usize,
) -> usize {
    let resolved = effective_nthread(requested)
        .min(worker_cap.max(1))
        .min(n_rows.max(1));
    if requested == 0 && n_rows.saturating_mul(work_per_row) < AUTO_PREDICT_MIN_WORK {
        1
    } else {
        resolved.max(1)
    }
}

/// Score independent row indices into deterministic, disjoint output chunks.
/// Every row still traverses trees serially and in model order, so scheduling
/// cannot change floating-point accumulation order.
fn score_rows<F>(
    n_rows: usize,
    workers: usize,
    pool: &mut Option<std::sync::Arc<rayon::ThreadPool>>,
    score_row: F,
) -> anyhow::Result<Vec<f32>>
where
    F: Fn(usize) -> f32 + Sync,
{
    let mut out = vec![0.0f32; n_rows];
    if n_rows == 0 {
        return Ok(out);
    }
    if workers <= 1 {
        for (row, value) in out.iter_mut().enumerate() {
            *value = score_row(row);
        }
        return Ok(out);
    }
    if pool.is_none() {
        *pool = Some(crate::threading::prediction_pool(workers)?);
    }
    let chunks = workers.saturating_mul(4).max(1);
    let rows_per_chunk = n_rows.div_ceil(chunks).max(1);
    pool.as_ref()
        .expect("prediction pool just initialized")
        .install(|| {
            out.par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk, values)| {
                    let first = chunk * rows_per_chunk;
                    for (offset, value) in values.iter_mut().enumerate() {
                        *value = score_row(first + offset);
                    }
                });
        });
    Ok(out)
}

/// Rows per parallel task of the blocked CPU scorers, or `None` to score on
/// the calling thread. Tasks cover whole `CPU_PREDICTION_BLOCK_ROWS` row
/// blocks (a smaller task loses the per-block tree reuse) and at least
/// `AUTO_PREDICT_MIN_WORK` traversal steps each, the same crossover the
/// automatic worker count uses. A small batch therefore occupies fewer
/// workers than requested, or runs serially (waking pool threads costs more
/// than scoring it); the pool keeps the requested size. Every row is still
/// scored on its own, in tree order, so the split cannot change a result.
fn blocked_rows_per_chunk(n_rows: usize, workers: usize, work_per_row: usize) -> Option<usize> {
    const BLOCK: usize = crate::inference::CPU_PREDICTION_BLOCK_ROWS;
    let blocks = n_rows.div_ceil(BLOCK);
    let by_work = n_rows.saturating_mul(work_per_row) / AUTO_PREDICT_MIN_WORK;
    let chunks = workers.saturating_mul(4).min(blocks).min(by_work);
    if workers <= 1 || chunks <= 1 {
        return None;
    }
    Some(n_rows.div_ceil(chunks).div_ceil(BLOCK) * BLOCK)
}

/// Cache-friendly fast path for C-contiguous NumPy input. Parallelism still
/// owns disjoint deterministic row ranges; inside each range the CPU inference
/// model blocks rows while preserving every row's original tree-add order. An
/// optional per-row starting margin (`base_margin`, one value per row) replaces the
/// model intercept.
#[allow(clippy::too_many_arguments)]
fn score_contiguous_rows_from(
    model: &crate::inference::CpuInferenceModel,
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    n_trees: usize,
    workers: usize,
    pool: &mut Option<std::sync::Arc<rayon::ThreadPool>>,
    init: Option<&[f32]>,
) -> anyhow::Result<Vec<f32>> {
    let mut out = vec![0.0f32; n_rows];
    if n_rows == 0 {
        return Ok(out);
    }
    let Some(rows_per_chunk) = blocked_rows_per_chunk(n_rows, workers, model.work_per_row(n_trees))
    else {
        model.predict_margins_from(data, n_features, n_trees, init, &mut out)?;
        return Ok(out);
    };
    if pool.is_none() {
        *pool = Some(crate::threading::prediction_pool(workers)?);
    }
    pool.as_ref()
        .expect("prediction pool just initialized")
        .install(|| {
            out.par_chunks_mut(rows_per_chunk)
                .enumerate()
                .try_for_each(|(chunk, values)| {
                    let first = chunk * rows_per_chunk;
                    let value_first = first * n_features;
                    let value_last = value_first + values.len() * n_features;
                    model.predict_margins_from(
                        &data[value_first..value_last],
                        n_features,
                        n_trees,
                        init.map(|m| &m[first..first + values.len()]),
                        values,
                    )
                })
        })?;
    Ok(out)
}

/// Reject the reserved value f32::MAX in prediction input, as training does
/// (every array prediction path, CPU and GPU, checks here).
fn reject_reserved_max(x: &PyReadonlyArray2<'_, f32>) -> PyResult<()> {
    let has_max = if x.is_c_contiguous() {
        x.as_slice()?.contains(&f32::MAX)
    } else {
        x.as_array().iter().any(|&v| v == f32::MAX)
    };
    if has_max {
        let view = x.as_array();
        let (r, c) = view
            .indexed_iter()
            .find(|(_, &v)| v == f32::MAX)
            .map(|(ix, _)| ix)
            .expect("found above");
        return Err(PyValueError::new_err(format!(
            "prediction data at row {r}, column {c} is f32::MAX: this value is reserved for the missing-value split boundary and is rejected in training and prediction; rescale this feature first"
        )));
    }
    Ok(())
}

/// Multi-output counterpart of `score_contiguous_rows_from`: row-major
/// `rows x K` margins, disjoint deterministic row ranges per worker.
fn score_contiguous_multi_from(
    model: &crate::inference::CpuInferenceModel,
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    n_trees: usize,
    workers: usize,
    init: Option<&[f32]>,
) -> anyhow::Result<Vec<f32>> {
    let k = model.n_outputs();
    let mut out = vec![0.0f32; n_rows * k];
    if n_rows == 0 {
        return Ok(out);
    }
    let Some(rows_per_chunk) = blocked_rows_per_chunk(n_rows, workers, model.work_per_row(n_trees))
    else {
        model.predict_multi_margins_from(data, n_features, n_trees, init, &mut out)?;
        return Ok(out);
    };
    let pool = crate::threading::prediction_pool(workers)?;
    pool.install(|| {
        out.par_chunks_mut(rows_per_chunk * k)
            .enumerate()
            .try_for_each(|(chunk, values)| {
                let first = chunk * rows_per_chunk;
                let rows = values.len() / k;
                model.predict_multi_margins_from(
                    &data[first * n_features..(first + rows) * n_features],
                    n_features,
                    n_trees,
                    init.map(|m| &m[first * k..(first + rows) * k]),
                    values,
                )
            })
    })?;
    Ok(out)
}

fn predict_value(model: &Model, row: &[f32], n_trees: usize, output_margin: bool) -> f32 {
    let margin = model.predict_margin_upto(row, n_trees);
    if output_margin {
        margin
    } else {
        model.objective.transform(margin)
    }
}

const PREDICTION_PARQUET_ROW_GROUP_ROWS: usize = 1024 * 1024;
const PREDICTION_PARQUET_CHANNEL_BATCHES: usize = 2;

/// One batch of output columns (one `Vec` per Parquet column, equal lengths).
enum PredictionParquetMessage {
    Batch(Vec<Vec<f32>>),
    Finish,
}

struct IncompletePredictionOutput {
    path: std::path::PathBuf,
    complete: bool,
}

impl Drop for IncompletePredictionOutput {
    fn drop(&mut self) {
        if !self.complete {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Ordered Parquet writer thread. Incoming batches (whose sizes follow the ingest
/// batches) are regrouped into exact row groups of `PREDICTION_PARQUET_ROW_GROUP_ROWS`
/// rows before they reach the Arrow writer, so the file is byte-identical whatever the
/// ingest batch sizes or worker counts were, and identical to writing the whole
/// output in one call. Memory is bounded by one row group per column.
fn prediction_parquet_writer(
    file: std::fs::File,
    output_path: std::path::PathBuf,
    column_names: Vec<String>,
    receiver: std::sync::mpsc::Receiver<PredictionParquetMessage>,
) -> anyhow::Result<usize> {
    use arrow::array::{ArrayRef, Float32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    let mut cleanup = IncompletePredictionOutput {
        path: output_path,
        complete: false,
    };
    let n_columns = column_names.len();
    let schema = Arc::new(Schema::new(
        column_names
            .into_iter()
            .map(|name| Field::new(name, DataType::Float32, false))
            .collect::<Vec<_>>(),
    ));
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_size(PREDICTION_PARQUET_ROW_GROUP_ROWS)
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties))
        .context("failed to create the prediction Parquet writer")?;
    let mut rows_written = 0usize;
    let mut pending: Vec<Vec<f32>> = vec![Vec::new(); n_columns];
    let write_group =
        |writer: &mut ArrowWriter<std::fs::File>, columns: Vec<Vec<f32>>| -> anyhow::Result<()> {
            let arrays: Vec<ArrayRef> = columns
                .into_iter()
                .map(|values| Arc::new(Float32Array::from(values)) as ArrayRef)
                .collect();
            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
                .context("failed to create a prediction Arrow batch")?;
            writer
                .write(&batch)
                .context("failed to write a prediction Parquet batch")
        };
    loop {
        match receiver.recv() {
            Ok(PredictionParquetMessage::Batch(columns)) => {
                anyhow::ensure!(
                    columns.len() == n_columns,
                    "prediction batch has {} columns, schema has {n_columns}",
                    columns.len()
                );
                let rows = columns.first().map_or(0, Vec::len);
                anyhow::ensure!(
                    columns.iter().all(|c| c.len() == rows),
                    "prediction batch columns have different lengths"
                );
                rows_written = rows_written
                    .checked_add(rows)
                    .context("prediction output row count overflows")?;
                if pending[0].is_empty() && rows == PREDICTION_PARQUET_ROW_GROUP_ROWS {
                    write_group(&mut writer, columns)?;
                    continue;
                }
                for (buffer, column) in pending.iter_mut().zip(columns) {
                    buffer.extend_from_slice(&column);
                }
                while pending[0].len() >= PREDICTION_PARQUET_ROW_GROUP_ROWS {
                    let group: Vec<Vec<f32>> = pending
                        .iter_mut()
                        .map(|buffer| {
                            let rest = buffer.split_off(PREDICTION_PARQUET_ROW_GROUP_ROWS);
                            std::mem::replace(buffer, rest)
                        })
                        .collect();
                    write_group(&mut writer, group)?;
                }
            }
            Ok(PredictionParquetMessage::Finish) => break,
            Err(_) => anyhow::bail!("prediction Parquet pipeline was cancelled before completion"),
        }
    }
    if !pending[0].is_empty() {
        write_group(&mut writer, std::mem::take(&mut pending))?;
    }
    writer
        .finish()
        .context("failed to finish the prediction Parquet footer")?;
    writer
        .inner()
        .sync_all()
        .context("failed to sync the prediction Parquet file")?;
    cleanup.complete = true;
    Ok(rows_written)
}

#[allow(clippy::too_many_arguments)]
fn predict_files_to_parquet(
    model: &PyModel,
    py: Python<'_>,
    path_specs: Vec<String>,
    output_path: &str,
    prediction_column: &str,
    output_margin: bool,
    n_trees: Option<usize>,
    header: bool,
    predict_threads: usize,
    missing: f32,
    init: Option<&[f32]>,
) -> PyResult<usize> {
    if prediction_column.trim().is_empty() {
        return Err(PyValueError::new_err(
            "prediction_column cannot be empty or whitespace only",
        ));
    }
    let output_path = std::path::PathBuf::from(output_path);
    let valid_extension = output_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("parquet") || extension.eq_ignore_ascii_case("pq")
        });
    if !valid_extension {
        return Err(PyValueError::new_err(
            "prediction output path must use the .parquet or .pq extension",
        ));
    }

    let resolved = crate::source::input::resolve(&path_specs)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    model.check_file_schema(&resolved, header)?;

    let names = model.inner.feature_names.clone();
    let canonical_n_features = model.inner.n_features;
    let upto = n_trees.unwrap_or(model.inner.trees.len());
    // Multi-output models (multiclass, several quantiles): the file is read once, every
    // batch is scored for all outputs and written before the next one is read.
    let k_out = model.inner.num_outputs();
    let multi: Option<Arc<crate::inference::CompactInferenceModel>> = if k_out > 1 {
        Some(model.inference_model().map_err(to_py_err)?)
    } else {
        None
    };
    let single = if multi.is_none() {
        Some(model.inference_model().map_err(to_py_err)?)
    } else {
        None
    };
    let projected_features: Vec<usize> = match (&single, &multi) {
        (Some(inference), _) => inference.canonical_features().to_vec(),
        (None, Some(inference)) => inference.canonical_features().to_vec(),
        (None, None) => unreachable!("either a single or a multi-output predictor"),
    };
    let n_features = projected_features.len();
    // `multi:softmax` writes one class-index column; every other multi-output result
    // (probabilities, margins, quantiles) writes `{column}_{k}`.
    let label_column = !output_margin
        && matches!(
            model.inner.objective,
            crate::types::Objective::Softmax { .. }
        );
    let column_names: Vec<String> = if multi.is_none() || label_column {
        vec![prediction_column.to_string()]
    } else {
        (0..k_out)
            .map(|c| format!("{prediction_column}_{c}"))
            .collect()
    };
    let rows_total = if resolved.format == crate::source::input::InputFormat::Parquet {
        resolved
            .paths
            .iter()
            .try_fold(0u64, |total, path| {
                crate::source::parquet_source::peek_n_rows(path).map(|rows| total + rows)
            })
            .ok()
    } else {
        None
    };
    let initially_resolved_ingest = resolved_ingest_workers(
        model.ingest_threads,
        resolved.format.label(),
        &resolved.paths,
        n_features,
        n_features.max(1),
    );
    let work_per_row = match (&single, &multi) {
        (Some(inference), _) => prediction_work_per_row(inference.model(), upto),
        (None, Some(_)) => prediction_work_per_row(&model.inner, upto),
        (None, None) => unreachable!("either a single or a multi-output predictor"),
    };
    let (ingest_workers, predict_worker_cap, oversubscribed) = prediction_output_pipeline_workers(
        model.ingest_threads,
        predict_threads,
        initially_resolved_ingest,
    );
    if oversubscribed && !model.quiet {
        eprintln!(
            "PREDICT_OUTPUT warning=explicit_cpu_oversubscription ingest_workers={} \
             predict_workers={} writer_workers=1 available={}",
            ingest_workers,
            predict_worker_cap,
            effective_nthread(0),
        );
    }

    let file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(pyo3::exceptions::PyFileExistsError::new_err(format!(
                "prediction output already exists, not overwriting: {}",
                output_path.display()
            )))
        }
        Err(error) => {
            return Err(PyValueError::new_err(format!(
                "cannot create prediction output {}: {error}",
                output_path.display()
            )))
        }
    };

    let (sender, receiver) = std::sync::mpsc::sync_channel(PREDICTION_PARQUET_CHANNEL_BATCHES);
    let writer_path = output_path.clone();
    let writer_columns = column_names.clone();
    let writer_handle = match std::thread::Builder::new()
        .name("fb-predict-parquet".to_string())
        .spawn(move || prediction_parquet_writer(file, writer_path, writer_columns, receiver))
    {
        Ok(handle) => handle,
        Err(error) => {
            let _ = std::fs::remove_file(&output_path);
            return Err(PyRuntimeError::new_err(format!(
                "cannot start the prediction Parquet writer: {error}"
            )));
        }
    };

    if !model.quiet {
        eprintln!(
            "PREDICT_OUTPUT start format=parquet output={} column={} outputs={} \
             ingest_workers={} predict_workers={} writer_workers=1 queue_batches={}",
            output_path.display(),
            prediction_column,
            column_names.len(),
            ingest_workers,
            predict_worker_cap,
            PREDICTION_PARQUET_CHANNEL_BATCHES,
        );
    }
    let started = std::time::Instant::now();
    let reporter = IngestReporter::start(
        model.quiet,
        resolved.format.label(),
        &resolved.paths,
        model.ingest_threads,
        ingest_workers,
        rows_total,
    );
    let mut predict_pool = None;
    let mut row_offset = 0usize;
    let input_result = py.detach(|| {
        let predictor = match &single {
            Some(inference) => Some(inference.cpu_predictor()?),
            None => None,
        };
        crate::source::predict_input::for_each_projected_contiguous_batch(
            &resolved,
            &names,
            canonical_n_features,
            &projected_features,
            header,
            ingest_workers,
            missing,
            |data, n_rows| {
                let workers = resolved_predict_workers(
                    predict_threads,
                    predict_worker_cap,
                    n_rows,
                    work_per_row,
                );
                // This batch's slice of the per-row start (row-major `n_rows x k_out`;
                // one value per row for single-output models).
                let batch_init = match init {
                    Some(m) => {
                        let (lo, hi) = (row_offset * k_out, (row_offset + n_rows) * k_out);
                        anyhow::ensure!(hi <= m.len(), "base_margin has fewer rows than the data");
                        Some(&m[lo..hi])
                    }
                    None => None,
                };
                row_offset += n_rows;
                let columns: Vec<Vec<f32>> = match (&single, &multi, &predictor) {
                    (Some(inference), _, Some(predictor)) => {
                        let mut batch = score_contiguous_rows_from(
                            predictor,
                            data,
                            n_rows,
                            n_features,
                            upto,
                            workers,
                            &mut predict_pool,
                            batch_init,
                        )?;
                        transform_prediction_margins(inference.model(), &mut batch, output_margin);
                        vec![batch]
                    }
                    (_, Some(inference), _) => {
                        let mut margins = score_multi_output_batch(
                            model, inference, upto, data, n_rows, false, workers, batch_init,
                        )?;
                        if output_margin {
                            deinterleave_columns(&margins, k_out)
                        } else if let Some(labels) =
                            model.inner.objective.transform_rows(&mut margins, k_out)
                        {
                            vec![labels]
                        } else {
                            deinterleave_columns(&margins, k_out)
                        }
                    }
                    _ => unreachable!("either a single or a multi-output predictor"),
                };
                sender
                    .send(PredictionParquetMessage::Batch(columns))
                    .map_err(|_| {
                        anyhow::anyhow!("prediction Parquet writer stopped before batch commit")
                    })
            },
        )
    });

    if let Err(input_error) = input_result {
        drop(sender);
        let writer_result = writer_handle.join();
        if input_error
            .to_string()
            .contains("prediction Parquet writer stopped before batch commit")
        {
            match writer_result {
                Ok(Err(writer_error)) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "prediction Parquet write failed: {writer_error:#}"
                    )))
                }
                Err(_) => return Err(PyRuntimeError::new_err("prediction Parquet writer panic")),
                Ok(Ok(_)) => {}
            }
        }
        return Err(PyValueError::new_err(format!("{input_error:#}")));
    }
    let rows_read = input_result.expect("checked above");
    if let Some(m) = init {
        if m.len() != rows_read * k_out {
            drop(sender);
            let _ = writer_handle.join();
            let _ = std::fs::remove_file(&output_path);
            return Err(PyValueError::new_err(format!(
                "base_margin has {} values, the files have {rows_read} rows x {k_out} outputs",
                m.len()
            )));
        }
    }
    if sender.send(PredictionParquetMessage::Finish).is_err() {
        drop(sender);
        return match writer_handle.join() {
            Ok(Err(writer_error)) => Err(PyRuntimeError::new_err(format!(
                "prediction Parquet write failed: {writer_error:#}"
            ))),
            Err(_) => Err(PyRuntimeError::new_err("prediction Parquet writer panic")),
            Ok(Ok(_)) => Err(PyRuntimeError::new_err(
                "prediction Parquet writer stopped before the completion signal",
            )),
        };
    }
    drop(sender);
    let rows_written = match writer_handle.join() {
        Ok(Ok(rows)) => rows,
        Ok(Err(error)) => {
            return Err(PyRuntimeError::new_err(format!(
                "prediction Parquet write failed: {error:#}"
            )))
        }
        Err(_) => return Err(PyRuntimeError::new_err("prediction Parquet writer panic")),
    };
    if rows_read != rows_written {
        let _ = std::fs::remove_file(&output_path);
        return Err(PyRuntimeError::new_err(format!(
            "prediction output row count mismatch: read {rows_read}, wrote {rows_written}"
        )));
    }

    reporter.complete(rows_read);
    if !model.quiet {
        let bytes = std::fs::metadata(&output_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        eprintln!(
            "PREDICT_OUTPUT complete rows={} bytes={} seconds={:.3} output={}",
            rows_written,
            bytes,
            started.elapsed().as_secs_f64(),
            output_path.display(),
        );
    }
    Ok(rows_written)
}

/// Row-major `n × k` values → `k` columns of `n` values.
fn deinterleave_columns(values: &[f32], k: usize) -> Vec<Vec<f32>> {
    let n = values.len() / k;
    (0..k)
        .map(|c| (0..n).map(|i| values[i * k + c]).collect())
        .collect()
}

/// One batch of a multi-output model (multiclass classes, or one quantile per
/// alpha), row-major `n_rows x K`. `data` holds the columns of `inference`'s
/// compact feature list, which covers every output; all outputs are scored in
/// one pass (one upload on the device), bit-identical to scoring each class
/// view on its own.
#[allow(clippy::too_many_arguments)]
fn score_multi_output_batch(
    owner: &PyModel,
    inference: &crate::inference::CompactInferenceModel,
    upto: usize,
    data: &[f32],
    n_rows: usize,
    gpu_batch: bool,
    workers: usize,
    // Per-row start margins of this batch, row-major `n_rows x K` (replaces the intercepts).
    init: Option<&[f32]>,
) -> anyhow::Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    if gpu_batch {
        return owner
            .gpu_predictor(0)?
            .predict_multi_margins_from(data, n_rows, upto, init);
    }
    let _ = (gpu_batch, owner);
    score_contiguous_multi_from(
        &*inference.cpu_predictor()?,
        data,
        n_rows,
        inference.n_features(),
        upto,
        workers,
        init,
    )
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn project_canonical_rows(
    row_major: &[f32],
    n_rows: usize,
    canonical_width: usize,
    projected_features: &[usize],
) -> anyhow::Result<Vec<f32>> {
    let expected = n_rows
        .checked_mul(canonical_width)
        .context("prediction canonical input size overflows")?;
    if row_major.len() != expected {
        anyhow::bail!(
            "prediction canonical input has {} values, expected {expected}",
            row_major.len()
        );
    }
    let output_len = n_rows
        .checked_mul(projected_features.len())
        .context("prediction compact input size overflows")?;
    let mut projected = Vec::with_capacity(output_len);
    for row in row_major.chunks_exact(canonical_width) {
        for &feature in projected_features {
            projected.push(row[feature]);
        }
    }
    Ok(projected)
}

/// File ingest and prediction are independently configurable, but auto values
/// share one process-visible CPU budget. Explicit values are never rewritten.
fn prediction_pipeline_workers(
    ingest_requested: usize,
    predict_requested: usize,
    resolved_ingest: usize,
) -> (usize, usize, bool) {
    prediction_pipeline_workers_with_reserve(
        ingest_requested,
        predict_requested,
        resolved_ingest,
        0,
    )
}

/// A single Parquet file has one ordered writer, but it runs concurrently with
/// ingest and scoring. Auto mode reserves one process-visible CPU for that
/// writer, then divides the rest between ingest and prediction.
fn prediction_output_pipeline_workers(
    ingest_requested: usize,
    predict_requested: usize,
    resolved_ingest: usize,
) -> (usize, usize, bool) {
    prediction_pipeline_workers_with_reserve(
        ingest_requested,
        predict_requested,
        resolved_ingest,
        1,
    )
}

fn prediction_pipeline_workers_with_reserve(
    ingest_requested: usize,
    predict_requested: usize,
    resolved_ingest: usize,
    reserved_workers: usize,
) -> (usize, usize, bool) {
    let total_budget = effective_nthread(0).max(1);
    let budget = total_budget.saturating_sub(reserved_workers).max(1);
    let (ingest, predict) = match (ingest_requested, predict_requested) {
        (0, 0) if budget > 1 => {
            let ingest = resolved_ingest.min((budget / 2).max(1));
            (ingest, budget.saturating_sub(ingest).max(1))
        }
        (0, 0) => (1, 1),
        (0, explicit_predict) => {
            let predict = effective_nthread(explicit_predict);
            (
                resolved_ingest.min(budget.saturating_sub(predict).max(1)),
                predict,
            )
        }
        (_, 0) => (
            resolved_ingest,
            budget.saturating_sub(resolved_ingest).max(1),
        ),
        (_, explicit_predict) => (resolved_ingest, effective_nthread(explicit_predict)),
    };
    let explicit_oversubscription = if reserved_workers == 0 {
        ingest_requested > 0 && predict_requested > 0 && ingest + predict > total_budget
    } else {
        (ingest_requested > 0 || predict_requested > 0)
            && ingest
                .saturating_add(predict)
                .saturating_add(reserved_workers)
                > total_budget
    };
    (ingest.max(1), predict.max(1), explicit_oversubscription)
}

const AUTO_GPU_PREDICT_MIN_WORK: usize = 8 * 1024 * 1024;

fn ensure_gpu_prediction_available(use_gpu: bool) -> PyResult<()> {
    #[cfg(not(feature = "cuda"))]
    if use_gpu {
        return Err(PyNotImplementedError::new_err(
            "this FerrisBoost build has no CUDA prediction support; use use_gpu=False or install the CUDA wheel",
        ));
    }
    let _ = use_gpu;
    Ok(())
}

fn gpu_prediction_selected(
    use_gpu: bool,
    n_rows: usize,
    n_features: usize,
    work_per_row: usize,
) -> bool {
    // Full-row H2D competes with traversal work that touches only split
    // features. Wide/shallow inputs can therefore favor CPU even at a large
    // row count (confirmed by Epsilon 20k x 2000); require at least one unit
    // of traversal work per transferred feature in addition to total work.
    use_gpu
        && work_per_row >= n_features
        && n_rows.saturating_mul(work_per_row) >= AUTO_GPU_PREDICT_MIN_WORK
}

fn transform_prediction_margins(model: &Model, values: &mut [f32], output_margin: bool) {
    if !output_margin && !model.objective.transform_is_identity() {
        for value in values {
            *value = model.objective.transform(*value);
        }
    }
}

#[pyclass(name = "Model", module = "ferrisboost._core")]
pub struct PyModel {
    inner: Model,
    history: Vec<RoundMetrics>,
    quiet: bool,
    model_io_threads: usize,
    ingest_threads: usize,
    cpu_predictor: Mutex<Option<Arc<crate::inference::CpuInferenceModel>>>,
    inference_model: Mutex<Option<Arc<crate::inference::CompactInferenceModel>>>,
    #[cfg(feature = "cuda")]
    gpu_predictors:
        Mutex<std::collections::HashMap<usize, Arc<crate::backend::cuda::GpuInferenceModel>>>,
    /// Device SHAP / leaf-index kernels' flattened ensemble, per device.
    #[cfg(feature = "cuda")]
    gpu_explainers:
        Mutex<std::collections::HashMap<usize, Arc<crate::backend::cuda::GpuExplainModel>>>,
}

/// `iteration_range` 的执行方式:`begin == 0` 是树的前缀(不拷贝),否则是
/// 只含这些轮的模型切片(同样的 base_score / objective)。
enum IterationPlan {
    Prefix(usize),
    Slice(PyModel),
}

impl PyModel {
    /// XGBoost `iteration_range=(begin, end)` 的语义:半开区间 `[begin, end)` 的
    /// **轮次**(多分类一轮 `num_class` 棵树);`end == 0` 表示到最后一轮;
    /// 越界或 `begin > end` 报错;`begin == end` 是空区间,只剩 base_score。
    /// 和 `n_trees`(树的前缀,不是轮次)互斥。
    fn plan_iteration_range(
        &self,
        (begin, end): (usize, usize),
        n_trees: Option<usize>,
    ) -> PyResult<IterationPlan> {
        if n_trees.is_some() {
            return Err(PyValueError::new_err(
                "iteration_range and n_trees cannot both be given: the former counts rounds, the latter counts trees",
            ));
        }
        let k = self.inner.num_outputs();
        let rounds = self.inner.trees.len() / k;
        let end = if end == 0 { rounds } else { end };
        if end > rounds {
            return Err(PyValueError::new_err(format!(
                "iteration_range end = {end} exceeds the model's {rounds} rounds"
            )));
        }
        if begin > end {
            return Err(PyValueError::new_err(format!(
                "iteration_range begin = {begin} is greater than end = {end}"
            )));
        }
        if begin == 0 {
            return Ok(IterationPlan::Prefix(end * k));
        }
        Ok(IterationPlan::Slice(PyModel::plain(
            self.inner.slice_rounds(begin, end),
            self.quiet,
            self.model_io_threads,
            self.ingest_threads,
        )))
    }

    /// A plain wrapper without training history (`iteration_range` slices).
    fn plain(inner: Model, quiet: bool, model_io_threads: usize, ingest_threads: usize) -> Self {
        PyModel {
            inner,
            history: Vec::new(),
            quiet,
            model_io_threads,
            ingest_threads,
            cpu_predictor: Mutex::new(None),
            inference_model: Mutex::new(None),
            #[cfg(feature = "cuda")]
            gpu_predictors: Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "cuda")]
            gpu_explainers: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Multi-output (multiclass, multi-quantile) margins, row-major `n x K`.
    /// Arrays score every output in one pass (`multi_output_array_margins`);
    /// file input goes through `predict_files_from`, which reads each file once
    /// and scores every batch for all outputs (`score_multi_output_batch`).
    #[allow(clippy::too_many_arguments)]
    fn multiclass_margins<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        use_gpu: bool,
        missing: f32,
        init: Option<&[f32]>,
    ) -> PyResult<(usize, Vec<f32>)> {
        let k = self.inner.num_outputs();
        if let Some(specs) = path_specs_of(data)? {
            let arr = self.predict_files_from(
                py,
                specs,
                true,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                missing,
                init,
            )?;
            let arr = arr
                .cast::<numpy::PyArray2<f32>>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let margins = arr.readonly().as_slice()?.to_vec();
            return Ok((margins.len() / k, margins));
        }
        let upto = n_trees.unwrap_or(self.inner.trees.len());
        let x = array_input(py, data)?;
        self.multi_output_array_margins(py, x, upto, predict_threads, use_gpu, init)
    }

    /// Row-major `rows x K` margins of a multi-output model from one array,
    /// validated like the single-output array path. Every output is scored in
    /// one pass over the rows (one upload on the device), bit-identical to
    /// scoring each class view on its own: see
    /// `CpuInferenceModel::predict_multi_margins_from` and
    /// `GpuInferenceModel::predict_multi_margins_from`.
    fn multi_output_array_margins<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        upto: usize,
        predict_threads: usize,
        use_gpu: bool,
        init: Option<&[f32]>,
    ) -> PyResult<(usize, Vec<f32>)> {
        let k = self.inner.num_outputs();
        let shape = x.shape();
        let (n_rows, n_features) = (shape[0], shape[1]);
        if n_features != self.inner.n_features {
            return Err(PyValueError::new_err(format!(
                "model was trained on {} features, got {n_features} columns",
                self.inner.n_features
            )));
        }
        if let Some(m) = init {
            if m.len() != n_rows * k {
                return Err(PyValueError::new_err(format!(
                    "base_margin has {} values, the data has {n_rows} rows x {k} outputs",
                    m.len()
                )));
            }
        }
        reject_reserved_max(&x)?;
        let work_per_row = prediction_work_per_row(&self.inner, upto);
        if use_gpu {
            let inference = self.inference_model().map_err(to_py_err)?;
            if gpu_prediction_selected(use_gpu, n_rows, inference.n_features(), work_per_row) {
                #[cfg(feature = "cuda")]
                {
                    let predictor = self.gpu_predictor(0).map_err(to_py_err)?;
                    let margins = if x.is_c_contiguous() && inference.is_identity(n_features) {
                        let data = x.as_slice()?;
                        py.detach(|| predictor.predict_multi_margins_from(data, n_rows, upto, init))
                    } else {
                        let view = x.as_array();
                        py.detach(|| {
                            let mut projected = Vec::with_capacity(n_rows * inference.n_features());
                            for row in view.rows() {
                                for &feature in inference.canonical_features() {
                                    projected.push(row[feature]);
                                }
                            }
                            predictor.predict_multi_margins_from(&projected, n_rows, upto, init)
                        })
                    }
                    .map_err(to_py_err)?;
                    return Ok((n_rows, margins));
                }
            }
        }
        let gathered: Vec<f32>;
        let data: &[f32] = if x.is_c_contiguous() {
            x.as_slice()?
        } else {
            gathered = x.as_array().iter().copied().collect();
            &gathered
        };
        let workers = resolved_predict_workers(
            predict_threads,
            effective_nthread(predict_threads),
            n_rows,
            work_per_row,
        );
        let predictor = self.cpu_predictor().map_err(to_py_err)?;
        let margins = py
            .detach(|| {
                score_contiguous_multi_from(
                    &predictor, data, n_rows, n_features, upto, workers, init,
                )
            })
            .map_err(to_py_err)?;
        Ok((n_rows, margins))
    }

    /// 多分类预测的最终形状:`output_margin` / softprob → `(n, K)`,
    /// softmax → `(n,)` 类别下标(和 XGBoost `Booster.predict` 一致)。
    #[allow(clippy::too_many_arguments)]
    fn predict_multiclass<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        use_gpu: bool,
        missing: f32,
        init: Option<&[f32]>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // 文件输入:单遍读、按 batch 给每类打分(见 predict_files)。
        if let Some(specs) = path_specs_of(data)? {
            return self.predict_files_from(
                py,
                specs,
                output_margin,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                missing,
                init,
            );
        }
        let (n, mut margins) = self.multiclass_margins(
            py,
            data,
            n_trees,
            header,
            predict_threads,
            use_gpu,
            missing,
            init,
        )?;
        let k = self.inner.num_outputs();
        if !output_margin {
            // softprob → 概率,softmax → 类别下标(此时返回一维),
            // 多分位数 → 每行升序(防交叉)。
            if let Some(labels) = self.inner.objective.transform_rows(&mut margins, k) {
                return Ok(PyArray1::from_vec(py, labels).into_any());
            }
        }
        let arr = PyArray1::from_vec(py, margins)
            .reshape([n, k])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_any())
    }

    /// `predict_array` with an optional per-row starting margin (`base_margin`,
    /// one value per row, single-output models only) in place of the intercept.
    /// The device predictor takes the same per-row start.
    #[allow(clippy::too_many_arguments)]
    fn predict_array_from<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        output_margin: bool,
        n_trees: Option<usize>,
        predict_threads: usize,
        use_gpu: bool,
        init: Option<&[f32]>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        guard(|| {
            ensure_gpu_prediction_available(use_gpu)?;
            if self.inner.num_outputs() > 1 {
                return Err(PyValueError::new_err(
                    "multiclass predictions are two-dimensional; use Model.predict()",
                ));
            }
            let shape = x.shape();
            let (n_rows, n_features) = (shape[0], shape[1]);
            if n_features != self.inner.n_features {
                return Err(PyValueError::new_err(format!(
                    "model was trained on {} features, got {n_features} columns",
                    self.inner.n_features
                )));
            }
            if let Some(m) = init {
                if m.len() != n_rows {
                    return Err(PyValueError::new_err(format!(
                        "base_margin has {} values, the data has {n_rows} rows",
                        m.len()
                    )));
                }
            }
            // 和训练一样拒绝保留值 f32::MAX(CPU 和 GPU 预测都走这里)。
            reject_reserved_max(&x)?;
            let upto = n_trees.unwrap_or(self.inner.trees.len());
            let work_per_row = prediction_work_per_row(&self.inner, upto);
            let inference = if use_gpu {
                Some(self.inference_model().map_err(to_py_err)?)
            } else {
                None
            };
            let transfer_width = inference
                .as_ref()
                .map_or(n_features, |model| model.n_features());
            if gpu_prediction_selected(use_gpu, n_rows, transfer_width, work_per_row) {
                #[cfg(feature = "cuda")]
                {
                    let inference = inference.expect("CUDA selection built compact model");
                    let predictor = self.gpu_predictor(0).map_err(to_py_err)?;
                    let mut out = if x.is_c_contiguous() {
                        let data = x.as_slice()?;
                        if inference.is_identity(n_features) {
                            py.detach(|| predictor.predict_margins_from(data, n_rows, upto, init))
                        } else {
                            py.detach(|| {
                                let projected = project_canonical_rows(
                                    data,
                                    n_rows,
                                    n_features,
                                    inference.canonical_features(),
                                )?;
                                predictor.predict_margins_from(&projected, n_rows, upto, init)
                            })
                        }
                    } else {
                        let view = x.as_array();
                        py.detach(|| {
                            let mut projected = Vec::with_capacity(n_rows * inference.n_features());
                            for row in 0..n_rows {
                                for &feature in inference.canonical_features() {
                                    projected.push(view[(row, feature)]);
                                }
                            }
                            predictor.predict_margins_from(&projected, n_rows, upto, init)
                        })
                    }
                    .map_err(to_py_err)?;
                    transform_prediction_margins(&self.inner, &mut out, output_margin);
                    return Ok(PyArray1::from_vec(py, out));
                }
            }
            let worker_cap = effective_nthread(predict_threads);
            let workers =
                resolved_predict_workers(predict_threads, worker_cap, n_rows, work_per_row);
            let mut pool = None;

            // `as_slice` alone is insufficient: NumPy can return a linear slice
            // for an F-contiguous array too, whose physical order is column-major.
            // Only C-contiguous input may use the zero-copy row-major fast path.
            // Other layouts are enumerated logically, so storage order is never
            // interpreted as row-major.
            // A per-row start always goes through the blocked contiguous scorer;
            // a non-contiguous array is gathered once (O(rows x features)).
            let gathered: Vec<f32>;
            let contiguous: Option<&[f32]> = if x.is_c_contiguous() {
                Some(x.as_slice()?)
            } else if init.is_some() {
                gathered = x.as_array().iter().copied().collect();
                Some(&gathered)
            } else {
                None
            };
            let out = if let Some(data) = contiguous {
                let predictor = self.cpu_predictor().map_err(to_py_err)?;
                py.detach(|| {
                    let mut out = score_contiguous_rows_from(
                        &predictor, data, n_rows, n_features, upto, workers, &mut pool, init,
                    )?;
                    transform_prediction_margins(&self.inner, &mut out, output_margin);
                    Ok(out)
                })
            } else {
                let view = x.as_array();
                py.detach(|| {
                    score_rows(n_rows, workers, &mut pool, |row| {
                        // Materialize only this logical row, never the whole matrix.
                        let values = view.row(row).iter().copied().collect::<Vec<_>>();
                        predict_value(&self.inner, &values, upto, output_margin)
                    })
                })
            }
            .map_err(to_py_err)?;
            Ok(PyArray1::from_vec(py, out))
        })
    }

    /// `predict_files` with an optional per-row starting margin (`base_margin`,
    /// row-major `rows x num_outputs` over all inputs in file order) replacing
    /// the intercept (the device predictor takes the same per-row start).
    #[allow(clippy::too_many_arguments)]
    fn predict_files_from<'py>(
        &self,
        py: Python<'py>,
        path_specs: Vec<String>,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        use_gpu: bool,
        missing: f32,
        init: Option<&[f32]>,
    ) -> PyResult<Bound<'py, PyAny>> {
        ensure_gpu_prediction_available(use_gpu)?;
        let resolved = crate::source::input::resolve(&path_specs)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        self.check_file_schema(&resolved, header)?;

        let names = self.inner.feature_names.clone();
        let canonical_n_features = self.inner.n_features;
        let upto = n_trees.unwrap_or(self.inner.trees.len());
        let k_out = self.inner.num_outputs();
        // 多分类:文件只读一遍。每个 batch 只读各类别视图用到的特征的**并集**,
        // 然后每个类别视图在内存里 gather 自己(压缩过)的特征子集、用自己的
        // 单输出预测器打分,按行交错后做 softmax / argmax。
        let multiclass: Option<Arc<crate::inference::CompactInferenceModel>> = if k_out > 1 {
            Some(self.inference_model().map_err(to_py_err)?)
        } else {
            None
        };
        let single = if k_out == 1 {
            Some(self.inference_model().map_err(to_py_err)?)
        } else {
            None
        };
        let projected_features: Vec<usize> = match (&single, &multiclass) {
            (Some(inference), _) => inference.canonical_features().to_vec(),
            (None, Some(inference)) => inference.canonical_features().to_vec(),
            (None, None) => unreachable!("either a single or a multi-output predictor"),
        };
        let n_features = projected_features.len();
        let mut out: Vec<f32> = Vec::new();
        let rows_total = if resolved.format == crate::source::input::InputFormat::Parquet {
            resolved
                .paths
                .iter()
                .try_fold(0u64, |total, path| {
                    crate::source::parquet_source::peek_n_rows(path).map(|rows| total + rows)
                })
                .ok()
        } else {
            None
        };
        let initially_resolved_ingest = resolved_ingest_workers(
            self.ingest_threads,
            resolved.format.label(),
            &resolved.paths,
            n_features,
            n_features.max(1),
        );
        let work_per_row = match (&single, &multiclass) {
            (Some(inference), _) => prediction_work_per_row(inference.model(), upto),
            (None, Some(_)) => prediction_work_per_row(&self.inner, upto),
            (None, None) => unreachable!(),
        };
        let known_gpu_selection = rows_total
            .and_then(|rows| usize::try_from(rows).ok())
            .map(|rows| gpu_prediction_selected(use_gpu, rows, n_features, work_per_row));
        let (ingest_workers, predict_worker_cap, oversubscribed) =
            if known_gpu_selection == Some(true) {
                (initially_resolved_ingest, 1, false)
            } else {
                prediction_pipeline_workers(
                    self.ingest_threads,
                    predict_threads,
                    initially_resolved_ingest,
                )
            };
        if oversubscribed && !self.quiet {
            eprintln!(
                "PREDICT warning=explicit_cpu_oversubscription ingest_workers={} \
                 predict_workers={} available={}",
                ingest_workers,
                predict_worker_cap,
                effective_nthread(0),
            );
        }
        let reporter = IngestReporter::start(
            self.quiet,
            resolved.format.label(),
            &resolved.paths,
            self.ingest_threads,
            ingest_workers,
            rows_total,
        );
        let mut predict_pool = None;
        let mut row_offset = 0usize;
        let rows = py
            .detach(|| {
                #[cfg(feature = "cuda")]
                let mut gpu_predictor = None;
                let mut cpu_predictor = None;
                crate::source::predict_input::for_each_projected_contiguous_batch(
                    &resolved,
                    &names,
                    canonical_n_features,
                    &projected_features,
                    header,
                    ingest_workers,
                    missing,
                    |data, n_rows| {
                        // This batch's slice of the per-row start (row-major
                        // `n_rows x k_out`); batches arrive in file/row order.
                        let batch_init = match init {
                            Some(m) => {
                                let (lo, hi) = (row_offset * k_out, (row_offset + n_rows) * k_out);
                                anyhow::ensure!(
                                    hi <= m.len(),
                                    "base_margin has fewer rows than the data"
                                );
                                Some(&m[lo..hi])
                            }
                            None => None,
                        };
                        row_offset += n_rows;
                        let gpu_batch = known_gpu_selection.unwrap_or_else(|| {
                            gpu_prediction_selected(use_gpu, n_rows, n_features, work_per_row)
                        });
                        if let Some(inference) = &multiclass {
                            let workers = resolved_predict_workers(
                                predict_threads,
                                predict_worker_cap,
                                n_rows,
                                work_per_row,
                            );
                            let mut margins = score_multi_output_batch(
                                self, inference, upto, data, n_rows, gpu_batch, workers, batch_init,
                            )?;
                            if output_margin {
                                out.extend(margins);
                            } else if let Some(labels) =
                                self.inner.objective.transform_rows(&mut margins, k_out)
                            {
                                out.extend(labels);
                            } else {
                                out.extend(margins);
                            }
                            return Ok(());
                        }
                        let inference = single.as_ref().expect("single-output inference");
                        let inner = inference.model();
                        #[cfg(feature = "cuda")]
                        if gpu_batch {
                            let predictor = match &gpu_predictor {
                                Some(predictor) => Arc::clone(predictor),
                                None => {
                                    let predictor = self.gpu_predictor(0)?;
                                    gpu_predictor = Some(Arc::clone(&predictor));
                                    predictor
                                }
                            };
                            let mut batch =
                                predictor.predict_margins_from(data, n_rows, upto, batch_init)?;
                            transform_prediction_margins(inner, &mut batch, output_margin);
                            out.extend(batch);
                            return Ok(());
                        }
                        let _ = gpu_batch;
                        let workers = resolved_predict_workers(
                            predict_threads,
                            predict_worker_cap,
                            n_rows,
                            work_per_row,
                        );
                        let predictor = match &cpu_predictor {
                            Some(predictor) => Arc::clone(predictor),
                            None => {
                                let predictor = inference.cpu_predictor()?;
                                cpu_predictor = Some(Arc::clone(&predictor));
                                predictor
                            }
                        };
                        let mut batch = score_contiguous_rows_from(
                            &predictor,
                            data,
                            n_rows,
                            n_features,
                            upto,
                            workers,
                            &mut predict_pool,
                            batch_init,
                        )?;
                        transform_prediction_margins(inner, &mut batch, output_margin);
                        // Batch completion order is irrelevant: the reader yields
                        // batches in canonical file/row order and this callback
                        // commits each complete batch synchronously.
                        out.extend(batch);
                        Ok(())
                    },
                )
            })
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        reporter.complete(rows);
        if let Some(m) = init {
            if m.len() != rows * k_out {
                return Err(PyValueError::new_err(format!(
                    "base_margin has {} values, the files have {rows} rows x {k_out} outputs",
                    m.len()
                )));
            }
        }
        let two_d = k_out > 1
            && (output_margin
                || !matches!(
                    self.inner.objective,
                    crate::types::Objective::Softmax { .. }
                ));
        if two_d {
            let n = out.len() / k_out;
            return PyArray1::from_vec(py, out)
                .reshape([n, k_out])
                .map(Bound::into_any)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()));
        }
        Ok(PyArray1::from_vec(py, out).into_any())
    }

    fn cpu_predictor(&self) -> anyhow::Result<Arc<crate::inference::CpuInferenceModel>> {
        let mut cache = self
            .cpu_predictor
            .lock()
            .map_err(|_| anyhow::anyhow!("CPU prediction model cache poisoned"))?;
        if let Some(model) = cache.as_ref() {
            return Ok(Arc::clone(model));
        }
        let model = Arc::new(crate::inference::CpuInferenceModel::new(&self.inner)?);
        *cache = Some(Arc::clone(&model));
        Ok(model)
    }

    fn inference_model(&self) -> anyhow::Result<Arc<crate::inference::CompactInferenceModel>> {
        let mut cache = self
            .inference_model
            .lock()
            .map_err(|_| anyhow::anyhow!("compact inference model cache poisoned"))?;
        if let Some(model) = cache.as_ref() {
            return Ok(Arc::clone(model));
        }
        let model = Arc::new(crate::inference::CompactInferenceModel::new(&self.inner)?);
        *cache = Some(Arc::clone(&model));
        Ok(model)
    }

    /// The flattened ensemble the device explanation kernels read, built once
    /// per device on first use.
    #[cfg(feature = "cuda")]
    fn gpu_explainer(
        &self,
        device_ordinal: usize,
    ) -> anyhow::Result<Arc<crate::backend::cuda::GpuExplainModel>> {
        let mut cache = self
            .gpu_explainers
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU explanation model cache poisoned"))?;
        if let Some(model) = cache.get(&device_ordinal) {
            return Ok(Arc::clone(model));
        }
        let model = Arc::new(crate::backend::cuda::GpuExplainModel::new(
            &self.inner,
            device_ordinal,
        )?);
        cache.insert(device_ordinal, Arc::clone(&model));
        Ok(model)
    }

    #[cfg(feature = "cuda")]
    fn gpu_predictor(
        &self,
        device_ordinal: usize,
    ) -> anyhow::Result<Arc<crate::backend::cuda::GpuInferenceModel>> {
        let mut cache = self
            .gpu_predictors
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU prediction model cache poisoned"))?;
        if let Some(predictor) = cache.get(&device_ordinal) {
            return Ok(Arc::clone(predictor));
        }
        let inference = self.inference_model()?;
        let predictor = Arc::new(crate::backend::cuda::GpuInferenceModel::from_compact(
            &inference,
            device_ordinal,
        )?);
        cache.insert(device_ordinal, Arc::clone(&predictor));
        Ok(predictor)
    }
}

#[pymethods]
impl PyModel {
    /// 预测。`output_margin=True` 给链接函数之前的原始分数。
    ///
    /// 和 XGBoost 的 `Booster.predict()` 一样,**默认用全部树**,
    /// 即使早停记了 best_iteration。要截断传 `n_trees`。
    #[pyo3(signature = (data, output_margin=false, n_trees=None, header=true, *,
                        predict_threads=0, use_gpu=false, iteration_range=None,
                        pred_contribs=false, approx_contribs=false, pred_interactions=false,
                        pred_leaf=false, missing=f64::NAN, base_margin=None,
                        strict_shape=false))]
    /// 预测。`data` 可以是:
    ///
    /// * 文件路径 / 路径列表 / 目录 / glob —— 走 Rust 的文件路径,
    ///   按模型 schema 绑列(**产品的主路径**);
    /// * 二维 f32 numpy 数组 —— 仍然支持,小数据方便。
    ///
    /// 两条都返回一维 `numpy.ndarray`。
    #[allow(clippy::too_many_arguments)] // Public Python keyword API must remain unchanged.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        use_gpu: bool,
        iteration_range: Option<(usize, usize)>,
        pred_contribs: bool,
        approx_contribs: bool,
        pred_interactions: bool,
        pred_leaf: bool,
        missing: f64,
        base_margin: Option<&Bound<'py, PyAny>>,
        strict_shape: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // `strict_shape` (XGBoost semantics) only reshapes the finished output,
        // so it is applied to the ordinary result of the same call.
        if strict_shape {
            let out = self.predict(
                py,
                data,
                output_margin,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                iteration_range,
                pred_contribs,
                approx_contribs,
                pred_interactions,
                pred_leaf,
                missing,
                base_margin,
                false,
            )?;
            let request = explain::ExplainRequest {
                contribs: pred_contribs,
                approx: approx_contribs,
                interactions: pred_interactions,
                leaf: pred_leaf,
            };
            return self.strict_shaped(out, request);
        }
        // `missing` (XGBoost semantics): feature values equal to it score as
        // NaN. Array-likes are normalized (and the sentinel mapped) here, once,
        // so every branch below sees a plain f32 matrix; file inputs carry the
        // sentinel into the reader.
        let missing = crate::source::validate_missing(missing)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        // A pandas DataFrame / pyarrow Table is bound to the model's feature
        // names (when it has them) before anything else, like evals in training.
        let data = &bind_named_table(py, data, &self.inner.feature_names)?;
        let data = &normalize_array_input(py, data, missing)?;
        // iteration_range 先于一切:begin == 0 折成 n_trees 前缀,否则换成只含
        // 这些轮的模型切片再走同一条路径(SHAP / leaf 也只看被选中的树)。
        let mut n_trees = n_trees;
        if let Some(range) = iteration_range {
            match self.plan_iteration_range(range, n_trees)? {
                IterationPlan::Prefix(n) => n_trees = Some(n),
                IterationPlan::Slice(sliced) => {
                    return sliced.predict(
                        py,
                        data,
                        output_margin,
                        None,
                        header,
                        predict_threads,
                        use_gpu,
                        None,
                        pred_contribs,
                        approx_contribs,
                        pred_interactions,
                        pred_leaf,
                        f64::from(missing),
                        base_margin,
                        false,
                    );
                }
            }
        }
        // `base_margin` (XGBoost semantics): a per-row starting margin that
        // replaces the model's intercept. Resolved once into row-major
        // `rows x num_outputs` values; every path below consumes the same slice.
        let margin_values: Option<Vec<f32>> = match base_margin {
            Some(bm) if !bm.is_none() => Some(self.resolve_predict_margin(py, data, bm, header)?),
            _ => None,
        };
        let init = margin_values.as_deref();
        // SHAP contributions / interactions / leaf indices: on the CPU, or on
        // the device with `use_gpu` (same values bit for bit). Single code path
        // for all objectives and inputs.
        let explain = explain::ExplainRequest {
            contribs: pred_contribs,
            approx: approx_contribs,
            interactions: pred_interactions,
            leaf: pred_leaf,
        };
        if explain.any() {
            return self.predict_explain(
                py,
                data,
                explain,
                output_margin,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                missing,
                init,
            );
        }
        ensure_gpu_prediction_available(use_gpu)?;
        if self.inner.num_outputs() > 1 {
            return self.predict_multiclass(
                py,
                data,
                output_margin,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                missing,
                init,
            );
        }
        // 先认路径:字符串 / PathLike / 它们的序列。认不出来才当数组。
        if let Some(specs) = path_specs_of(data)? {
            return self.predict_files_from(
                py,
                specs,
                output_margin,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                missing,
                init,
            );
        }
        let x: PyReadonlyArray2<'py, f32> = array_input(py, data)?;
        self.predict_array_from(
            py,
            x,
            output_margin,
            n_trees,
            predict_threads,
            use_gpu,
            init,
        )
        .map(Bound::into_any)
    }

    #[pyo3(signature = (x, output_margin=false, n_trees=None, *,
                        predict_threads=0, use_gpu=false))]
    fn predict_array<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        output_margin: bool,
        n_trees: Option<usize>,
        predict_threads: usize,
        use_gpu: bool,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.predict_array_from(
            py,
            x,
            output_margin,
            n_trees,
            predict_threads,
            use_gpu,
            None,
        )
    }

    /// 存成 XGBoost 能加载的 JSON。
    /// 文件预测。`path_specs` 和训练接受的写法完全一样(单文件、列表、
    /// 目录、glob),复用同一套发现 / 排序 / 格式判定 —— **不另起一套路径解析**。
    ///
    /// 输出恒为 1 维 `numpy.ndarray`,长度等于所有输入文件的总行数,
    /// 顺序就是文件顺序(字典序)加文件内行序。
    ///
    /// ⚠️ **不需要标签列。** 标签是训练期元信息;文件里带着也不影响,
    /// 因为选列是按模型的特征名单来的,不是"除某列外全要"。
    #[pyo3(signature = (path_specs, output_margin=false, n_trees=None, header=true, *,
                        predict_threads=0, use_gpu=false, missing=f32::NAN))]
    #[allow(clippy::too_many_arguments)] // Keep the installed public keyword API stable.
    fn predict_files<'py>(
        &self,
        py: Python<'py>,
        path_specs: Vec<String>,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        use_gpu: bool,
        missing: f32,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.predict_files_from(
            py,
            path_specs,
            output_margin,
            n_trees,
            header,
            predict_threads,
            use_gpu,
            missing,
            None,
        )
    }

    /// Stream file prediction directly to one Parquet file. CPU scoring uses
    /// deterministic row-parallel workers; one ordered writer thread overlaps
    /// Parquet encoding/output with the next ingest/score batch.
    #[pyo3(signature = (data, output_path, output_margin=false, n_trees=None, header=true, *,
                        predict_threads=0, prediction_column="prediction",
                        iteration_range=None, missing=f64::NAN, base_margin=None))]
    #[allow(clippy::too_many_arguments)] // Public output/lifecycle options are independently specified.
    fn predict_to_parquet(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        output_path: &str,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        prediction_column: &str,
        iteration_range: Option<(usize, usize)>,
        missing: f64,
        base_margin: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        let missing = crate::source::validate_missing(missing)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut n_trees = n_trees;
        if let Some(range) = iteration_range {
            match self.plan_iteration_range(range, n_trees)? {
                IterationPlan::Prefix(n) => n_trees = Some(n),
                IterationPlan::Slice(sliced) => {
                    return sliced.predict_to_parquet(
                        py,
                        data,
                        output_path,
                        output_margin,
                        None,
                        header,
                        predict_threads,
                        prediction_column,
                        None,
                        f64::from(missing),
                        base_margin,
                    );
                }
            }
        }
        guard(|| {
            let path_specs = path_specs_of(data)?.ok_or_else(|| {
                PyValueError::new_err(
                    "predict_to_parquet only accepts a file path, a list of paths, a directory or a glob",
                )
            })?;
            // A per-row starting margin is read from file column(s) next to the features.
            let margin_values: Option<Vec<f32>> = match base_margin {
                Some(bm) if !bm.is_none() => {
                    Some(self.resolve_predict_margin(py, data, bm, header)?)
                }
                _ => None,
            };
            let init = margin_values.as_deref();
            predict_files_to_parquet(
                self,
                py,
                path_specs,
                output_path,
                prediction_column,
                output_margin,
                n_trees,
                header,
                predict_threads,
                missing,
                init,
            )
        })
    }

    #[pyo3(signature = (path, quiet=None, model_io_threads=None))]
    fn save_model(
        &self,
        py: Python<'_>,
        path: &str,
        quiet: Option<bool>,
        model_io_threads: Option<usize>,
    ) -> PyResult<()> {
        let quiet = quiet.unwrap_or(self.quiet);
        let requested = model_io_threads.unwrap_or(self.model_io_threads);
        let inner = &self.inner;
        guard(|| {
            py.detach(|| -> Result<(), ModelIoError> {
                let json = inner
                    .to_xgboost_json_with_threads(requested)
                    .map_err(|e| ModelIoError::Runtime(format!("export failed: {e}")))?;
                parallel_model_write(path, json.as_bytes(), requested, quiet)
                    .map_err(|e| ModelIoError::Value(format!("cannot write {path}: {e}")))
            })
            .map_err(PyErr::from)
        })
    }

    /// 读一个 XGBoost 格式的模型(它自己训的也行)。
    #[staticmethod]
    #[pyo3(signature = (path, quiet=false, model_io_threads=0, ingest_threads=0))]
    fn load_model(
        py: Python<'_>,
        path: &str,
        quiet: bool,
        model_io_threads: usize,
        ingest_threads: usize,
    ) -> PyResult<Self> {
        guard(|| {
            let inner = py
                .detach(|| -> Result<Model, ModelIoError> {
                    let raw = parallel_model_read(path, model_io_threads, quiet).map_err(|e| {
                        if e.downcast_ref::<std::io::Error>()
                            .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
                        {
                            ModelIoError::FileNotFound(format!("{path} does not exist"))
                        } else {
                            ModelIoError::Value(format!("cannot read {path}: {e}"))
                        }
                    })?;
                    let raw = String::from_utf8(raw).map_err(|e| {
                        ModelIoError::Value(format!("{path} is not UTF-8 JSON: {e}"))
                    })?;
                    Model::from_xgboost_json_with_threads(&raw, model_io_threads)
                        .map_err(|e| ModelIoError::Value(format!("failed to parse {path}: {e}")))
                })
                .map_err(PyErr::from)?;
            Ok(Self {
                inner,
                history: Vec::new(),
                quiet,
                model_io_threads,
                ingest_threads,
                cpu_predictor: Mutex::new(None),
                #[cfg(feature = "cuda")]
                gpu_predictors: Mutex::new(std::collections::HashMap::new()),
                #[cfg(feature = "cuda")]
                gpu_explainers: Mutex::new(std::collections::HashMap::new()),
                inference_model: Mutex::new(None),
            })
        })
    }

    #[getter]
    fn best_iteration(&self) -> Option<usize> {
        self.inner.best_iteration
    }

    #[getter]
    fn best_score(&self) -> Option<f32> {
        self.inner.best_score
    }

    #[getter]
    fn num_trees(&self) -> usize {
        self.inner.trees.len()
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features
    }

    /// The model's feature names (empty for a positional model: numpy input).
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names.clone()
    }

    /// XGBoost objective 名字(如 `binary:logistic`)。
    #[getter]
    fn objective(&self) -> &'static str {
        self.inner.objective.name()
    }

    /// `reg:quantileerror` 的 `quantile_alpha` 列表;其它模型是 `None`。
    #[getter]
    fn quantile_alpha(&self) -> Option<Vec<f32>> {
        self.inner
            .objective
            .quantile_alphas()
            .map(|a| a.as_slice().to_vec())
    }

    /// 多分类的类别数;其它模型是 1。
    #[getter]
    fn num_class(&self) -> usize {
        self.inner.num_outputs()
    }

    /// Feature importance with XGBoost's `Booster.get_score` semantics:
    /// `importance_type` is "weight", "gain", "cover", "total_gain" or
    /// "total_cover". Features never used in a split are omitted; names are
    /// the model's feature names, or `f0`, `f1`, ... when it has none.
    #[pyo3(signature = (importance_type="weight"))]
    fn get_score<'py>(
        &self,
        py: Python<'py>,
        importance_type: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.feature_score_dict(py, importance_type)
    }

    /// evals_result:{eval 名: {metric 名: [每轮的值]}}
    fn evals_result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for m in &self.history {
            for (eval_name, metric, value) in &m.entries {
                let per_eval = match out.get_item(eval_name)? {
                    Some(d) => d.cast_into::<PyDict>()?,
                    None => {
                        let d = PyDict::new(py);
                        out.set_item(eval_name, &d)?;
                        d
                    }
                };
                let series = match per_eval.get_item(metric)? {
                    Some(l) => l.cast_into::<PyList>()?,
                    None => {
                        let l = PyList::empty(py);
                        per_eval.set_item(metric, &l)?;
                        l
                    }
                };
                series.append(*value)?;
            }
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "<ferrisboost.Model {} trees, {} features{}>",
            self.inner.trees.len(),
            self.inner.n_features,
            match self.inner.best_iteration {
                Some(b) => format!(", best_iteration={b}"),
                None => String::new(),
            }
        )
    }
}

/// 规整好的参数(Python 侧已经查过别名和拼写)。
struct Parsed {
    params: TrainParams,
    /// 用户**显式**传进来的 `cols_per_block`;`None` 表示交给启发式。
    /// 和 `params.cols_per_block`(生效值)分开存,这样日志能同时报
    /// requested / effective,自动选择选错时看得出来是谁定的。
    requested_cols_per_block: Option<usize>,
    /// `Some(ordinal)` = 在 GPU 上训练。`None` = CPU。
    ///
    /// ⚠️ **Python 只说"用哪块卡",不构造 GPU source、也不选物理块宽** ——
    /// 那些由 Rust 侧的显存规划器决定,见 `gpu_mem_plan`。
    device: Option<usize>,
    objective: Objective,
    base_score: Option<f32>,
    cache_budget_bytes: Option<usize>,
    cache_path: Option<String>,
    /// File ingest concurrency, independent of training `nthread`.
    ingest_threads: usize,
    model_io_threads: usize,
    quiet: bool,
    /// XGBoost `eval_metric`,已按 objective 解析。空 = objective 默认指标。
    eval_metrics: Vec<String>,
    /// 训练集带样本权重。由入口在规划**之前**设上:fast 模式的显存计划要算它。
    weighted: bool,
    /// XGBoost `monotone_constraints`,还没按 schema 解析。见 `resolve_monotone`。
    monotone: Option<MonotoneSpec>,
    /// Declared categorical features (`feature_types`), one flag per feature in
    /// input order; empty = none declared. Arrow dictionary columns in file
    /// inputs are categorical even when this is empty.
    categorical: Vec<bool>,
    /// XGBoost `missing`: feature values equal to it are treated as NaN in the
    /// training data and every validation set. NaN = no sentinel.
    missing: f32,
    /// Query-group row offsets of the training set (ranking). Set by the entry
    /// point once the groups are known, before `train_config`.
    group_ptr: Option<Vec<u32>>,
}

/// Python 层归一后的单调约束:按位置,或按列名(要等拿到列名才能解析)。
enum MonotoneSpec {
    Positional(Vec<i8>),
    Named(Vec<(String, i8)>),
}

impl Parsed {
    /// 按 schema 解析单调约束,写进 `params.monotone_constraints`。
    ///
    /// `feature_names` 为空 = 位置模式(numpy、无表头 CSV),此时按列名给的
    /// 约束没法解析,直接报错而不是悄悄忽略。
    fn resolve_monotone(&mut self, n_features: usize, feature_names: &[String]) -> PyResult<()> {
        let resolved = match &self.monotone {
            None => return Ok(()),
            Some(MonotoneSpec::Positional(v)) => {
                if v.len() > n_features {
                    return Err(PyValueError::new_err(format!(
                        "monotone_constraints has {} entries but there are only {n_features} features",
                        v.len()
                    )));
                }
                v.clone()
            }
            Some(MonotoneSpec::Named(pairs)) => {
                if feature_names.is_empty() {
                    return Err(PyValueError::new_err(
                        "monotone_constraints given by name need column names: this input is positional \
                         (numpy or headerless CSV); use a positional tuple instead",
                    ));
                }
                let mut out = vec![0i8; feature_names.len()];
                for (name, dir) in pairs {
                    let at = feature_names.iter().position(|n| n == name).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "column {name:?} in monotone_constraints is not a feature column; feature columns: {feature_names:?}"
                        ))
                    })?;
                    out[at] = *dir;
                }
                out
            }
        };
        self.params.monotone_constraints = resolved;
        Ok(())
    }
}

impl Parsed {
    /// 组装 TrainConfig,同时返回 early stopping 的方向(看最后一个 metric)。
    fn train_config(&self) -> PyResult<(TrainConfig, bool)> {
        let mut cfg = TrainConfig {
            base_score: self.base_score,
            quiet: self.quiet,
            group_ptr: self.group_ptr.clone(),
            ..TrainConfig::new(self.params.clone(), self.objective)
        };
        if !self.eval_metrics.is_empty() {
            cfg.eval_metrics = self
                .eval_metrics
                .iter()
                .map(|name| crate::metrics::for_objective(name, self.objective))
                .collect::<Result<_, _>>()
                .map_err(PyValueError::new_err)?;
        }
        let lower_is_better = cfg.eval_metrics.last().is_none_or(|m| m.lower_is_better());
        Ok((cfg, lower_is_better))
    }
}

/// The XGBoost `lambdarank_*` / `ndcg_exp_gain` parameters of a ranking objective.
fn parse_rank_params(d: &Bound<'_, PyDict>) -> PyResult<crate::rank::RankParams> {
    use crate::rank::{PairMethod, RankParams};
    fn get<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        d.get_item(key)?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "parameter {key} is missing (the Python layer should supply a default)"
            ))
        })
    }
    let get = |key: &str| get(d, key);
    let pair_method: String = get("lambdarank_pair_method")?.extract()?;
    let pair_method = match pair_method.as_str() {
        "topk" => PairMethod::TopK,
        "mean" => PairMethod::Mean,
        other => {
            return Err(PyValueError::new_err(format!(
                "lambdarank_pair_method must be \"topk\" or \"mean\", got {other:?}"
            )))
        }
    };
    let num_pair: Option<u32> = get("lambdarank_num_pair_per_sample")?.extract()?;
    Ok(RankParams {
        pair_method,
        num_pair_per_sample: num_pair,
        normalization: get("lambdarank_normalization")?.extract()?,
        score_normalization: get("lambdarank_score_normalization")?.extract()?,
        ndcg_exp_gain: get("ndcg_exp_gain")?.extract()?,
        unbiased: get("lambdarank_unbiased")?.extract()?,
        bias_norm: get("lambdarank_bias_norm")?.extract()?,
    })
}

/// Group offsets and per-group weights of a file dataset from its query-id column
/// and (optional) per-row weight column.
fn file_groups(
    ids: &[u64],
    row_weights: Option<&[f32]>,
    weight_column: Option<&str>,
) -> anyhow::Result<(Vec<u32>, Option<Vec<f32>>)> {
    let ptr = crate::rank::group_ptr_from_qid(ids)?;
    let weights = match row_weights {
        None => None,
        Some(w) => {
            anyhow::ensure!(
                w.len() == ids.len(),
                "weight column has {} rows, the query-id column {}",
                w.len(),
                ids.len()
            );
            Some(crate::rank::group_weights_from_rows(
                w,
                &ptr,
                weight_column.unwrap_or("weight"),
            )?)
        }
    };
    Ok((ptr, weights))
}

fn parse_params(d: &Bound<'_, PyDict>) -> PyResult<Parsed> {
    // 写成宏而不是泛型函数:pyo3 0.29 的 FromPyObject 带两个生命周期,
    // 泛型 helper 要把它们全串起来,不如就地展开清楚。
    macro_rules! get {
        ($key:expr) => {
            d.get_item($key)?
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "parameter {} is missing (the Python layer should supply a default)",
                        $key
                    ))
                })?
                .extract()?
        };
    }
    let objective: String = get!("objective");
    let requested_device: Option<String> = get!("device");
    let requested_hist_cap: Option<usize> = get!("hist_nodes_per_batch");
    let hist_cap = if requested_device.as_deref().is_some_and(|d| d != "cpu") {
        let cap = match requested_hist_cap {
            Some(cap) => cap,
            None => match std::env::var("FB_HIST_NODES_PER_BATCH") {
                Ok(v) => v.trim().parse::<usize>().map_err(|_| {
                    PyValueError::new_err("FB_HIST_NODES_PER_BATCH must be an integer")
                })?,
                Err(_) => 16,
            },
        };
        if !(1..=32).contains(&cap) {
            return Err(PyValueError::new_err(
                "hist_nodes_per_batch must be in 1..=32",
            ));
        }
        Some(cap)
    } else {
        requested_hist_cap
    };
    // 同一个 max_delta_step 同时喂给树和 count:poisson 的 Hessian 稳定项。
    // 没给时 XGBoost 的 learner 对 count:poisson 把它设成 0.7(两边都是),
    // 其它 objective 树上是 0。
    let max_delta_step: Option<f32> = get!("max_delta_step");
    // Python 层已经把标量 / 序列统一成 `list[float]`(或 None)。
    let quantile_alpha: Option<Vec<f32>> = get!("quantile_alpha");
    if objective == "reg:quantileerror" && quantile_alpha.is_none() {
        return Err(PyValueError::new_err(
            "reg:quantileerror requires quantile_alpha (one or more ascending quantiles in [0, 1], \
             e.g. 0.9 or [0.1, 0.5, 0.9]); XGBoost has no default",
        ));
    }
    let objective = Objective::from_name(&objective)
        .filter(|_| objective != "reg:linear")
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "objective {objective:?} is not supported; supported: {}",
                Objective::supported_names()
            ))
        })?
        .with_params(
            max_delta_step,
            get!("tweedie_variance_power"),
            get!("huber_slope"),
            {
                let k: Option<u32> = get!("num_class");
                k.unwrap_or(0)
            },
        )
        .with_quantile_alphas(quantile_alpha.as_deref().unwrap_or(&[0.5]))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let objective = match objective {
        Objective::Rank { kind, .. } => Objective::Rank {
            kind,
            params: parse_rank_params(d)?,
        },
        other => other,
    };
    objective
        .validate_params()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut parsed = Parsed {
        params: TrainParams {
            n_rounds: get!("num_boost_round"),
            max_depth: get!("max_depth"),
            max_bin: get!("max_bin"),
            learning_rate: get!("eta"),
            lambda: get!("lambda"),
            gamma: get!("gamma"),
            min_child_weight: get!("min_child_weight"),
            alpha: get!("alpha"),
            max_delta_step: max_delta_step.unwrap_or(objective.default_tree_max_delta_step()),
            scale_pos_weight: get!("scale_pos_weight"),
            subsample: get!("subsample"),
            colsample_bytree: get!("colsample_bytree"),
            colsample_bylevel: get!("colsample_bylevel"),
            colsample_bynode: get!("colsample_bynode"),
            // 占位:真正的值要等拿到 schema 才能定,见 BlockingPlan。
            cols_per_block: 0,
            nthread: get!("nthread"),
            // 和 cols_per_block 一样保留 Option 语义:Python 层给 None 就是
            // 「让 planner 按模式选」,给 1/2 就完全按用户的来。
            hist_streams: get!("hist_streams"),
            resident_blocks: get!("resident_blocks"),
            hist_nodes_per_batch: hist_cap,
            // Resolve once: planner and runtime must see the same default and
            // explicit override. An unset variable enables the bounded cache;
            // a present non-true value (including empty) disables it.
            gpu_split_cache: std::env::var("FB_GPU_SPLIT_CACHE")
                .map_or(crate::types::DEFAULT_GPU_SPLIT_CACHE, |v| {
                    matches!(v.trim(), "1" | "true")
                }),
            gpu_memory_budget: get!("gpu_memory_budget"),
            gpu_math: {
                let raw: Option<String> = get!("gpu_math");
                match raw {
                    None => crate::types::GpuMath::default(),
                    Some(v) => v.parse().map_err(pyo3::exceptions::PyValueError::new_err)?,
                }
            },
            device_quantize: get!("device_quantize"),
            seed: get!("seed"),
            // 占位:按名字给的约束要等拿到 schema 才能解析,见 `MonotoneSpec`。
            monotone_constraints: Vec::new(),
            max_cat_to_onehot: get!("max_cat_to_onehot"),
            max_cat_threshold: get!("max_cat_threshold"),
        },
        requested_cols_per_block: get!("cols_per_block"),
        device: {
            // `device="cuda"` / `"cuda:0"` / `"gpu"` → GPU;不给或 "cpu" → CPU。
            let raw = requested_device;
            match raw.as_deref() {
                None | Some("cpu") => None,
                Some(d) if d == "cuda" || d == "gpu" => Some(0),
                Some(d) if d.starts_with("cuda:") => {
                    Some(d[5..].parse::<usize>().map_err(|_| {
                        PyValueError::new_err(format!("cannot parse device: {d:?}"))
                    })?)
                }
                Some(other) => {
                    return Err(PyValueError::new_err(format!(
                        "device must be \"cpu\" / \"cuda\" / \"cuda:N\", got {other:?}"
                    )))
                }
            }
        },
        objective,
        base_score: get!("base_score"),
        cache_budget_bytes: get!("cache_budget_bytes"),
        cache_path: get!("cache_path"),
        ingest_threads: get!("ingest_threads"),
        model_io_threads: get!("model_io_threads"),
        quiet: get!("_quiet"),
        eval_metrics: {
            let raw: Option<Vec<String>> = get!("eval_metric");
            let names = raw.unwrap_or_default();
            // 提前校验名字和 objective 组合:错了在读数据之前就报。
            for name in &names {
                crate::metrics::for_objective(name, objective).map_err(PyValueError::new_err)?;
            }
            names
        },
        weighted: false,
        group_ptr: None,
        missing: {
            let raw: f64 = get!("missing");
            crate::source::validate_missing(raw)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        },
        categorical: {
            let flags: Option<Vec<bool>> = get!("_feature_types");
            flags.unwrap_or_default()
        },
        monotone: {
            match d.get_item("monotone_constraints")? {
                None => None,
                Some(v) if v.is_none() => None,
                Some(v) => {
                    if let Ok(dict) = v.cast::<PyDict>() {
                        let mut pairs = Vec::with_capacity(dict.len());
                        for (k, val) in dict.iter() {
                            pairs.push((
                                k.extract::<String>()?,
                                val.extract::<i64>()?.signum() as i8,
                            ));
                        }
                        Some(MonotoneSpec::Named(pairs))
                    } else {
                        let values: Vec<i64> = v.extract()?;
                        Some(MonotoneSpec::Positional(
                            values.into_iter().map(|x| x.signum() as i8).collect(),
                        ))
                    }
                }
            }
        },
    };
    Ok(parsed)
}

/// 组装 callbacks:早停和 verbose 在 Rust 侧,用户的函数包一层。
struct Callbacks {
    list: Vec<Box<dyn Callback>>,
    history: Arc<Mutex<Vec<RoundMetrics>>>,
    error: Arc<Mutex<Option<PyErr>>>,
}

/// 训练返回后调用:用户回调抛过异常就把它抛出去。
fn raise_callback_error(slot: &Mutex<Option<PyErr>>) -> PyResult<()> {
    match slot.lock().expect("callback error slot").take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn build_callbacks(
    lower_is_better: bool,
    early_stopping_rounds: Option<usize>,
    verbose_eval: Option<usize>,
    user: Option<&Bound<'_, PyList>>,
) -> PyResult<Callbacks> {
    let error = Arc::new(Mutex::new(None));
    let recorder = RecordHistory::new();
    let handle = recorder.handle();
    let mut cbs: Vec<Box<dyn Callback>> = Vec::new();
    if let Some(rounds) = early_stopping_rounds {
        // 方向跟被盯的那个 metric 走(最后一个 metric;auc 越大越好)。
        cbs.push(Box::new(EarlyStopping::new(rounds, lower_is_better)));
    }
    if let Some(period) = verbose_eval {
        if period > 0 {
            cbs.push(Box::new(VerboseEval { period }));
        }
    }
    if let Some(list) = user {
        for f in list.iter() {
            cbs.push(Box::new(PyCallback {
                func: f.unbind(),
                error: Arc::clone(&error),
            }));
        }
    }
    cbs.push(Box::new(recorder));
    Ok(Callbacks {
        list: cbs,
        history: handle,
        error,
    })
}

/// 收尾:把规范特征名钉进模型。
///
/// ⚠️ **只有知道 schema 的那一层能填这个。** 训练层看到的是量化后的块,
/// 没有列名概念。空名单 = 位置模式(numpy 入口、无表头 CSV),
/// 非空 = 命名模式,文件预测据此按列名重排。
/// 认出「路径规格」:字符串 / `os.PathLike` / 它们的序列。
///
/// 不是路径就返回 `None`,让调用方去当数组解 —— numpy 数组也是序列,
/// 所以这里必须先排除掉它,不能只看"是不是可迭代"。
fn path_specs_of(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<String>>> {
    // numpy 数组优先排除:它可迭代,但显然不是路径。
    if obj.extract::<PyReadonlyArray2<f32>>().is_ok() {
        return Ok(None);
    }
    // pandas DataFrames iterate over their column names, and PyArrow tables are
    // tables: neither is a path spec.
    if obj.hasattr("columns")? || obj.hasattr("column_names")? {
        return Ok(None);
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Some(vec![s]));
    }
    if let Ok(list) = obj.extract::<Vec<String>>() {
        if !list.is_empty() {
            return Ok(Some(list));
        }
    }
    // A sequence of pathlib.Path objects does not extract as Vec<String>.
    // Resolve __fspath__ item-by-item, while leaving arbitrary iterables (most
    // importantly NumPy arrays) to the array branch below.
    if let Ok(iter) = obj.try_iter() {
        let mut paths = Vec::new();
        let mut all_paths = true;
        for item in iter {
            let item = item?;
            if let Ok(s) = item.extract::<String>() {
                paths.push(s);
            } else if let Ok(fs) = item.call_method0("__fspath__") {
                if let Ok(s) = fs.extract::<String>() {
                    paths.push(s);
                } else {
                    all_paths = false;
                    break;
                }
            } else {
                all_paths = false;
                break;
            }
        }
        if all_paths && !paths.is_empty() {
            return Ok(Some(paths));
        }
    }
    // os.PathLike:走 __fspath__。
    if let Ok(fs) = obj.call_method0("__fspath__") {
        if let Ok(s) = fs.extract::<String>() {
            return Ok(Some(vec![s]));
        }
    }
    Ok(None)
}

fn finish(
    model: Model,
    handle: Arc<Mutex<Vec<RoundMetrics>>>,
    feature_names: Vec<String>,
    quiet: bool,
    model_io_threads: usize,
    ingest_threads: usize,
) -> PyModel {
    let history = handle.lock().map(|h| h.clone()).unwrap_or_default();
    let mut model = model;
    // 名单和模式**一起设**,它们不可能各自为政:有名字就是命名模式,
    // 没名字就是位置模式。分两处赋值早晚会漂移成"有名字但标着位置"。
    model.schema_mode = Some(if feature_names.is_empty() {
        crate::tree::SchemaMode::Positional
    } else {
        crate::tree::SchemaMode::Named
    });
    model.feature_names = feature_names;
    PyModel {
        inner: model,
        history,
        quiet,
        model_io_threads,
        ingest_threads,
        cpu_predictor: Mutex::new(None),
        inference_model: Mutex::new(None),
        #[cfg(feature = "cuda")]
        gpu_predictors: Mutex::new(std::collections::HashMap::new()),
        #[cfg(feature = "cuda")]
        gpu_explainers: Mutex::new(std::collections::HashMap::new()),
    }
}

/// A 2-D f32 matrix from `predict` input: a numpy array as is, a pandas
/// DataFrame / PyArrow table through the Python layer (categorical columns
/// become their category codes).
fn array_input<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f32>> {
    match data.extract() {
        Ok(x) => Ok(x),
        Err(_) => {
            let converted = py
                .import("ferrisboost")?
                .getattr("_as_predict_matrix")?
                .call1((data,))?;
            converted.extract().map_err(|_| {
                PyValueError::new_err(
                    "predict only accepts file paths (single / list / directory / glob), a 2-D f32 numpy array, \
                     a pandas DataFrame or a pyarrow Table",
                )
            })
        }
    }
}

/// Convert a pandas DataFrame / pyarrow Table to an f32 matrix whose columns
/// are selected and ordered by the model's feature names. Other inputs (paths,
/// numpy arrays), and any input for a model without names, pass through.
fn bind_named_table<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    names: &[String],
) -> PyResult<Bound<'py, PyAny>> {
    if names.is_empty()
        || path_specs_of(data)?.is_some()
        || data.extract::<PyReadonlyArray2<f32>>().is_ok()
    {
        return Ok(data.clone());
    }
    let fb = py.import("ferrisboost")?;
    let is_table = fb.getattr("_is_named_table")?.call1((data,))?.is_truthy()?;
    if !is_table {
        return Ok(data.clone());
    }
    fb.getattr("_as_predict_matrix")?
        .call1((data, names.to_vec()))
}

/// Apply the XGBoost `missing` sentinel to array-like prediction input.
///
/// Path inputs, and any input when `missing` is NaN, are returned untouched
/// (file readers apply the sentinel while decoding). Otherwise the input is
/// normalized to an f32 matrix; if some value equals the sentinel a corrected
/// copy with those values set to NaN is returned, never mutating the caller's
/// array.
fn normalize_array_input<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    missing: f32,
) -> PyResult<Bound<'py, PyAny>> {
    if missing.is_nan() || path_specs_of(data)?.is_some() {
        return Ok(data.clone());
    }
    let x = array_input(py, data)?;
    let view = x.as_array();
    if !view.iter().any(|&v| v == missing) {
        return Ok(x.as_any().clone());
    }
    let mapped = view.mapv(|v| if v == missing { f32::NAN } else { v });
    Ok(numpy::PyArray2::from_owned_array(py, mapped).into_any())
}

/// 特征里的 ±inf 直接拒绝(XGBoost 同样拒绝);缺失值要用 NaN。
fn reject_inf(data: &[f32], n_features: usize, what: &str) -> PyResult<()> {
    match data.iter().position(|v| v.is_infinite() || *v == f32::MAX) {
        None => Ok(()),
        Some(i) => Err(PyValueError::new_err(format!(
            "{what} at row {}, column {} is {}: features cannot be ±inf or the reserved f32::MAX (use NaN for missing values; rescale extreme values)",
            i / n_features.max(1),
            i % n_features.max(1),
            data[i]
        ))),
    }
}

/// Training categorical columns must hold integer codes `0..=65533` (NaN =
/// missing). Report the first offender by feature index before binning.
fn check_categorical_codes(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    flags: &[bool],
) -> PyResult<()> {
    if flags.is_empty() {
        return Ok(());
    }
    if flags.len() != n_features {
        return Err(PyValueError::new_err(format!(
            "feature_types has {} entries, the data has {n_features} features",
            flags.len()
        )));
    }
    for (f, _) in flags.iter().enumerate().filter(|(_, &c)| c) {
        for r in 0..n_rows {
            let v = data[r * n_features + f];
            if !v.is_nan() && crate::categorical::invalid_code(v) {
                return Err(PyValueError::new_err(format!(
                    "{} (row {r})",
                    crate::categorical::invalid_code_message(&format!("#{f}"), v)
                )));
            }
        }
    }
    Ok(())
}

/// numpy 入口。`x` 是行主序的 f32 矩阵(Python 层负责转 dtype)。
#[pyfunction]
#[pyo3(signature = (x, y, params, evals=None, early_stopping_rounds=None,
                    verbose_eval=None, callbacks=None, weight=None, feature_names=None,
                    qid=None, group=None, base_margin=None, eval_base_margins=None,
                    prior_model=None))]
#[allow(clippy::too_many_arguments)]
fn train_dense(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, f32>,
    params: &Bound<'_, PyDict>,
    evals: Option<&Bound<'_, PyList>>,
    early_stopping_rounds: Option<usize>,
    verbose_eval: Option<usize>,
    callbacks: Option<&Bound<'_, PyList>>,
    weight: Option<PyReadonlyArray1<'_, f32>>,
    feature_names: Option<Vec<String>>,
    qid: Option<PyReadonlyArray1<'_, u64>>,
    group: Option<PyReadonlyArray1<'_, u64>>,
    // Per-row starting margin `(rows, num_outputs)` (for continuation: the
    // prior model's margin on this data); one entry per eval set in
    // `eval_base_margins`, `None` for a set that starts from the intercept.
    base_margin: Option<PyReadonlyArray2<'_, f32>>,
    eval_base_margins: Option<&Bound<'_, PyList>>,
    // Training continuation: trees, intercept and metadata carried over.
    prior_model: Option<PyRef<'_, PyModel>>,
) -> PyResult<PyModel> {
    let overall_started = std::time::Instant::now();
    let mut cfg = parse_params(params)?;
    let weights: Option<Vec<f32>> = weight
        .map(|w| w.as_slice().map(<[f32]>::to_vec))
        .transpose()?;
    let ranking = cfg.objective.is_ranking();
    cfg.weighted = weights.is_some() && !ranking;
    let shape = x.shape();
    let (n_rows, n_features) = (shape[0], shape[1]);
    let labels = y.as_slice()?.to_vec();
    if labels.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "got {} labels, the feature matrix has {n_rows} rows",
            labels.len()
        )));
    }
    // Query groups (ranking): flat row offsets next to the labels.
    let group_ptr = crate::rank::group_ptr_from_inputs(
        qid.as_ref().map(|q| q.as_slice()).transpose()?,
        group.as_ref().map(|g| g.as_slice()).transpose()?,
        n_rows,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    match (ranking, group_ptr.is_some()) {
        (true, false) => {
            return Err(PyValueError::new_err(format!(
                "{} needs query groups: pass qid= or group=",
                cfg.objective.name()
            )))
        }
        // Other objectives accept query groups too (XGBoost does): they do not change
        // training, but `auc` / `aucpr` on data with groups are scored per group.
        _ => {}
    }
    cfg.group_ptr = group_ptr.clone();

    // 量化在 GIL 里做:它要读 numpy 的内存。之后训练不再碰 Python 对象,
    // 就可以放掉 GIL 了。
    // 先定 blocking,再建 source。策略只在这里决定一次,后面的训练只消费
    // 已经定好的分块。
    // ⚠️ **GPU 和 CPU 用不同的块宽解析器。** GPU streaming 的启发式是拿
    // GPU 墙钟测出来的(块数拐点),套到 CPU 上没有证据支持;反过来也一样。
    // 这里是 Python 侧唯一一次决定物理块宽的地方。
    let plan = resolve_blocking(&cfg, n_features, Some(n_rows as u64))?;
    cfg.params.cols_per_block = plan.effective;
    let user_gpu = user_gpu_geometry(&cfg);
    // 规划器解出来的几何是权威的(见 `BlockingPlan`)。
    apply_resolved_geometry(&mut cfg, &plan);
    if let Some(w) = &weights {
        // Ranking weights are one per query group.
        let expected = match (&group_ptr, ranking) {
            (Some(g), true) => g.len() - 1,
            _ => n_rows,
        };
        if w.len() != expected {
            return Err(PyValueError::new_err(if ranking {
                format!(
                    "ranking weights are per query group: weight has {} entries for {expected} groups",
                    w.len()
                )
            } else {
                format!(
                    "weight has {} entries, the feature matrix has {n_rows} rows",
                    w.len()
                )
            }));
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(PyValueError::new_err(
                "sample weights must be finite and non-negative",
            ));
        }
    }
    // The `missing` sentinel becomes NaN before any check or binning (borrowed
    // when no value matches; otherwise one corrected copy, never the caller's
    // array). Categorical codes are subject to it too, as in XGBoost.
    let x_values = crate::source::missing_as_nan(x.as_slice()?, cfg.missing);
    reject_inf(&x_values, n_features, "training data")?;
    check_categorical_codes(&x_values, n_rows, n_features, &cfg.categorical)?;
    // 分箱边界按样本权重加权(XGBoost hist 同样如此);全 1 / None 逐位相同。
    // Ranking: the group weight is repeated on the rows of its group.
    let unrolled_weights: Option<Vec<f32>> = match (&weights, &group_ptr, ranking) {
        (Some(w), Some(g), true) => Some(crate::rank::unroll_group_weights(w, g)),
        _ => None,
    };
    let sketch_weights: Option<&[f32]> = unrolled_weights.as_deref().or(weights.as_deref());
    let train_src = DenseSource::new_weighted_with_nthread(
        &x_values,
        n_rows,
        n_features,
        if cfg.categorical.iter().any(|&c| c) {
            BinningStrategy::SketchTyped {
                max_bins: cfg.params.max_bin as usize,
                categorical: &cfg.categorical,
            }
        } else {
            BinningStrategy::Sketch {
                max_bins: cfg.params.max_bin as usize,
            }
        },
        plan.effective,
        cfg.params.nthread,
        sketch_weights,
    );
    {
        let mut params = cfg.params.clone();
        refine_gpu_plan_for_source(&cfg, user_gpu, n_features, &train_src, &mut params)?;
        cfg.params = params;
    }

    // 验证集必须复用训练集的 cuts,否则同一个值会落进不同的 bin
    #[allow(clippy::type_complexity)]
    let mut eval_data: Vec<(
        String,
        DenseSource,
        Vec<f32>,
        Option<Vec<f32>>,
        Option<Vec<u32>>,
    )> = Vec::new();
    if let Some(list) = evals {
        for item in list.iter() {
            // Python 层统一成 (data, label, weight-or-None, name, qid-or-None, group-or-None)。
            #[allow(clippy::type_complexity)]
            let (ex, ey, ew, name, eq, eg): (
                PyReadonlyArray2<f32>,
                PyReadonlyArray1<f32>,
                Option<PyReadonlyArray1<f32>>,
                String,
                Option<PyReadonlyArray1<u64>>,
                Option<PyReadonlyArray1<u64>>,
            ) = item.extract()?;
            let eshape = ex.shape();
            if eshape[1] != n_features {
                return Err(PyValueError::new_err(format!(
                    "validation set {name} has {} columns, the training set has {n_features}",
                    eshape[1]
                )));
            }
            let ex_values = crate::source::missing_as_nan(ex.as_slice()?, cfg.missing);
            reject_inf(&ex_values, n_features, &format!("validation set {name}"))?;
            let src = DenseSource::new(
                &ex_values,
                eshape[0],
                n_features,
                BinningStrategy::Provided(train_src.cuts()),
                cfg.params.cols_per_block,
            );
            let ew = ew.map(|w| w.as_slice().map(<[f32]>::to_vec)).transpose()?;
            let egp = crate::rank::group_ptr_from_inputs(
                eq.as_ref().map(|q| q.as_slice()).transpose()?,
                eg.as_ref().map(|g| g.as_slice()).transpose()?,
                eshape[0],
            )
            .map_err(|e| PyValueError::new_err(format!("validation set {name}: {e}")))?;
            eval_data.push((name, src, ey.as_slice()?.to_vec(), ew, egp));
        }
    }

    let device = cfg.device;
    // numpy 入口没有列名:只接受按位置的约束(Python 层已把 pyarrow 列名解析掉)。
    cfg.resolve_monotone(n_features, &[])?;
    let (mut train_cfg, lower_is_better) = cfg.train_config()?;
    let k_out = train_cfg.objective.num_outputs();
    if let Some(bm) = &base_margin {
        train_cfg.base_margin = Some(margin::class_major_margin(
            bm,
            n_rows,
            k_out,
            "base_margin",
        )?);
    }
    if let Some(list) = eval_base_margins {
        for (i, item) in list.iter().enumerate() {
            let rows = eval_data.get(i).map_or(0, |e| e.1.n_rows());
            train_cfg.eval_base_margins.push(if item.is_none() {
                None
            } else {
                let m: PyReadonlyArray2<f32> = item.extract()?;
                Some(margin::class_major_margin(
                    &m,
                    rows,
                    k_out,
                    &format!("eval set {:?} base_margin", eval_data[i].0),
                )?)
            });
        }
    }
    if let Some(prior) = &prior_model {
        train_cfg.prior_model = Some(prior.inner.clone());
    }
    let Callbacks {
        list: mut cbs,
        history: handle,
        error: cb_error,
    } = build_callbacks(
        lower_is_better,
        early_stopping_rounds,
        verbose_eval,
        callbacks,
    )?;
    if !cfg.quiet && device.is_none() {
        eprintln!(
            "CPU_THREADS requested={} resolved={}",
            train_cfg.params.nthread,
            effective_nthread(train_cfg.params.nthread)
        );
    }
    if !cfg.quiet {
        eprintln!(
            "SETUP input=dense rows={n_rows} features={n_features} weights={} seconds={:.3}",
            on_off(weights.is_some()),
            overall_started.elapsed().as_secs_f64()
        );
    }

    let evals_ref: Vec<EvalSet> = eval_data
        .iter()
        .map(|(name, src, labels, w, gp)| EvalSet {
            name: name.clone(),
            source: src,
            labels,
            weights: w.as_deref(),
            group_ptr: gp.as_deref(),
        })
        .collect();

    // 训练期间放掉 GIL:不然调用方的其它线程全被卡住。
    // 用户回调进 Python 时会自己重新拿(见 PyCallback)。
    let train_started = std::time::Instant::now();
    let model = py
        .detach(|| match device {
            #[cfg(feature = "cuda")]
            Some(ord) => crate::train::train_gpu_weighted(
                ord,
                &train_src,
                &labels,
                weights.as_deref(),
                &evals_ref,
                &train_cfg,
                &mut cbs,
                &Local,
            ),
            #[cfg(not(feature = "cuda"))]
            Some(_) => Err(anyhow::anyhow!("this build has no CUDA support")),
            None => train_weighted(
                &train_src,
                &labels,
                weights.as_deref(),
                &evals_ref,
                &train_cfg,
                &mut cbs,
                &Local,
            ),
        })
        .map_err(to_py_err)?;
    if !cfg.quiet {
        eprintln!(
            "TRAIN backend={} trees={} seconds={:.3}",
            device.map_or_else(|| "cpu".to_string(), |d| format!("cuda:{d}")),
            model.trees.len(),
            train_started.elapsed().as_secs_f64()
        );
        eprintln!(
            "END_TO_END seconds={:.3}",
            overall_started.elapsed().as_secs_f64()
        );
    }

    // 回调里抛的异常在这儿冒出来
    raise_callback_error(&cb_error)?;
    // numpy 入口没有列名 —— 位置模式;pandas / pyarrow 入口带列名 —— 命名模式
    // (get_score / feature_importances_ 用真名,按文件 / DataFrame 预测按列名绑定)。
    let feature_names = match feature_names {
        None => Vec::new(),
        Some(names) => {
            if names.len() != n_features {
                return Err(PyValueError::new_err(format!(
                    "feature_names has {} entries, the data has {n_features} columns",
                    names.len()
                )));
            }
            let mut seen = std::collections::HashSet::new();
            if let Some(dup) = names.iter().find(|n| !seen.insert(n.as_str())) {
                return Err(PyValueError::new_err(format!(
                    "duplicate feature column name: {dup:?}"
                )));
            }
            names
        }
    };
    let feature_names = margin::continued_feature_names(prior_model.as_deref(), feature_names)?;
    Ok(finish(
        model,
        handle,
        feature_names,
        cfg.quiet,
        cfg.model_io_threads,
        cfg.ingest_threads,
    ))
}

/// 文件入口。**格式由 Rust 从解析出的文件判定**,不再由 Python 按后缀猜。
#[pyfunction]
#[pyo3(signature = (path_specs, label, params, header=true, evals=None, early_stopping_rounds=None,
                    verbose_eval=None, callbacks=None, weight=None, qid=None, base_margin=None,
                    start_margin=None, eval_start_margins=None, prior_model=None))]
#[allow(clippy::too_many_arguments)]
fn train_file(
    py: Python<'_>,
    // ⚠️ **Python 只递路径规格,发现和校验在 Rust。** 可以是单个文件、
    // 目录、glob,或者它们的列表 —— 展开、去重、排序、格式一致性全部
    // 由 `source::input::resolve` 负责,Python 不做文件系统判断。
    path_specs: Vec<String>,
    label: &Bound<'_, PyAny>,
    params: &Bound<'_, PyDict>,
    header: bool,
    evals: Option<&Bound<'_, PyList>>,
    early_stopping_rounds: Option<usize>,
    verbose_eval: Option<usize>,
    callbacks: Option<&Bound<'_, PyList>>,
    weight: Option<&Bound<'_, PyAny>>,
    qid: Option<&Bound<'_, PyAny>>,
    // `base_margin`: file column name(s) holding the per-row starting margin
    // (one per output), excluded from the features like the label / weight.
    // `start_margin` / `eval_start_margins`: explicit `(rows, outputs)` arrays
    // that take precedence (continuation passes the prior model's margins).
    base_margin: Option<&Bound<'_, PyAny>>,
    start_margin: Option<&Bound<'_, PyAny>>,
    eval_start_margins: Option<&Bound<'_, PyList>>,
    prior_model: Option<PyRef<'_, PyModel>>,
) -> PyResult<PyModel> {
    let overall_started = std::time::Instant::now();
    let mut cfg = parse_params(params)?;
    let ranking = cfg.objective.is_ranking();
    cfg.weighted = weight.is_some() && !ranking;
    // 先把输入规格解成确定顺序的文件列表。格式也由它判定 ——
    // Python 那边不再靠后缀猜。
    // ⚠️ 输入规格的问题是**用户输入错误**,不是运行时故障 —— 映射成
    // `ValueError`,而不是笼统的 `RuntimeError`。找不到文件、glob 空、
    // 混格式都属于这一类,调用方应该能用 `except ValueError` 一把接住。
    let resolved = crate::source::input::resolve(&path_specs)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let paths = resolved.paths.clone();
    let fmt = resolved.format.label();
    let path_disp = paths[0].display().to_string();
    let path = path_disp.as_str();
    // CSV schema inference samples text rows, so keep the result used for
    // blocking and pass it into open instead of rescanning the first file.
    let csv_preflight = match fmt {
        "csv" => Some(csv_source::infer_schema(&paths[0], header).map_err(|e| {
            PyValueError::new_err(format!("failed to read the schema of {path}: {e}"))
        })?),
        _ => None,
    };
    let label_owned = match (fmt, header) {
        ("parquet", false) => {
            return Err(PyValueError::new_err("header=False only applies to CSV"));
        }
        (_, true) => label.extract::<String>().map_err(|_| {
            PyValueError::new_err("with header=True, label must be a column name string")
        })?,
        ("csv", false) => {
            let index = label.extract::<isize>().map_err(|_| {
                PyValueError::new_err("with header=False, label must be a zero-based column index")
            })?;
            if index < 0 {
                return Err(PyValueError::new_err(format!(
                    "label column index cannot be negative, got {index}"
                )));
            }
            csv_source::positional_label_name_from_schema(
                csv_preflight.as_deref().expect("CSV preflight exists"),
                index as usize,
            )
            .map_err(|e| PyValueError::new_err(format!("{e}")))?
        }
        _ => unreachable!(),
    };
    let label = label_owned.as_str();
    // 权重列和标签列同样的写法:有表头给列名,无表头 CSV 给列下标。
    let weight_owned: Option<String> = match weight {
        None => None,
        Some(w) if header => Some(w.extract::<String>().map_err(|_| {
            PyValueError::new_err("with header=True, weight must be a column name string")
        })?),
        Some(w) => {
            let index = w.extract::<isize>().map_err(|_| {
                PyValueError::new_err("with header=False, weight must be a zero-based column index")
            })?;
            if index < 0 {
                return Err(PyValueError::new_err(format!(
                    "weight column index cannot be negative, got {index}"
                )));
            }
            Some(
                csv_source::positional_label_name_from_schema(
                    csv_preflight.as_deref().expect("CSV preflight exists"),
                    index as usize,
                )
                .map_err(|e| PyValueError::new_err(format!("{e}")))?,
            )
        }
    };
    // The query-id column: same convention as label / weight (a name with a
    // header, a column index for a headerless CSV).
    let qid_owned: Option<String> = match qid {
        None => None,
        Some(q) if header => Some(q.extract::<String>().map_err(|_| {
            PyValueError::new_err("with header=True, qid must be a column name string")
        })?),
        Some(q) => {
            let index = q.extract::<isize>().map_err(|_| {
                PyValueError::new_err("with header=False, qid must be a zero-based column index")
            })?;
            if index < 0 {
                return Err(PyValueError::new_err(format!(
                    "qid column index cannot be negative, got {index}"
                )));
            }
            Some(
                csv_source::positional_label_name_from_schema(
                    csv_preflight.as_deref().expect("CSV preflight exists"),
                    index as usize,
                )
                .map_err(|e| PyValueError::new_err(format!("{e}")))?,
            )
        }
    };
    match (ranking, qid_owned.is_some()) {
        (true, false) => {
            return Err(PyValueError::new_err(format!(
                "{} needs query groups: pass qid=<query-id column>",
                cfg.objective.name()
            )))
        }
        (false, true) => {
            return Err(PyValueError::new_err(format!(
                "qid is only used by the rank:* objectives, not {}",
                cfg.objective.name()
            )))
        }
        _ => {}
    }
    if qid_owned.is_some() && qid_owned.as_deref() == Some(label) {
        return Err(PyValueError::new_err(
            "qid column must differ from the label column",
        ));
    }
    if qid_owned.is_some() && qid_owned == weight_owned {
        return Err(PyValueError::new_err(
            "qid column must differ from the weight column",
        ));
    }
    let k_out = cfg.objective.num_outputs();
    let margin_names: Vec<String> = match base_margin {
        Some(v) if !v.is_none() => {
            margin::file_margin_names(v, k_out, header, csv_preflight.as_deref(), label)?
        }
        _ => Vec::new(),
    };
    if let (Some(w), true) = (weight_owned.as_deref(), !margin_names.is_empty()) {
        if margin_names.iter().any(|n| n == w) {
            return Err(PyValueError::new_err(format!(
                "the base_margin column and the weight column cannot be the same ({w:?})"
            )));
        }
    }
    // A continued model with feature names indexes its features by those names:
    // bind the training (and eval) files' columns by name, in the model's order
    // (file column order is free; a missing or surplus column is an error).
    let prior_names: Vec<String> = match (&prior_model, header) {
        (Some(prior), true) => prior.inner.feature_names.clone(),
        _ => Vec::new(),
    };
    let mut targets = crate::source::Targets::new(label, weight_owned.as_deref())
        .map_err(|e| PyValueError::new_err(format!("{e}")))?
        .with_qid(qid_owned.as_deref())
        .with_base_margin(&margin_names);
    if !prior_names.is_empty() {
        targets = targets.with_features(&prior_names);
    }
    // **peek schema → 定 blocking → 才真正 open**。Parquet / CSV 在 open
    // 之前不知道特征数,而自动选块宽需要它;单独 peek 一次(只读 footer /
    // 只推断 schema,不解压数据)就能把策略决定留在这一层,
    // 不必把 `nthread` 传进底层的 `partition_features`。
    let n_features = match fmt {
        "parquet" => parquet_source::peek_n_features(path, targets),
        "csv" => csv_source::n_features_from_schema(
            csv_preflight.as_deref().expect("CSV preflight exists"),
            targets,
        ),
        other => Err(anyhow::anyhow!("unknown format {other:?}")),
    }
    .map_err(|e| PyValueError::new_err(format!("failed to read the schema of {path}: {e}")))?;
    // Declared feature types are positional over the resolved features (label
    // and weight excluded), like XGBoost's `feature_types`.
    if !cfg.categorical.is_empty() && cfg.categorical.len() != n_features {
        return Err(PyValueError::new_err(format!(
            "feature_types has {} entries, the file has {n_features} feature columns (excluding label/weight columns)",
            cfg.categorical.len()
        )));
    }
    let categorical_flags = cfg.categorical.clone();
    let targets = targets
        .with_categorical(&categorical_flags)
        .with_missing(cfg.missing);
    // 行数同样要在 open 之前拿到 —— 显存模型的每一项都按它走。
    // parquet 从 footer 读,不解压任何数据;CSV 不扫全文件就是不知道。
    // ⚠️ **规划器要看到整个数据集的行数,不是第一个文件的。** 只报第一个
    // 文件会让显存模型按 1/N 的规模规划,解出来的块宽在真实数据上放不下。
    let n_rows_peek: Option<u64> = match fmt {
        "parquet" => {
            let mut total = 0u64;
            for p in &paths {
                total += parquet_source::peek_n_rows(p).map_err(|e| {
                    PyValueError::new_err(format!(
                        "failed to read the row count of {}: {e}",
                        p.display()
                    ))
                })?;
            }
            Some(total)
        }
        _ => None,
    };
    // ⚠️ **路径入口是大数据 GPU 训练的首选路径**:Rust 读数据、量化/缓存、
    // 定块宽、建 source,Python 只给路径和参数。
    // 走 numpy 那条路等于先在 Python 里把整份数据物化一遍,
    // streaming / 部分列读 / colsample 的 I/O 削减 / 显存规划全都用不上。
    // ⚠️ 任意拿不到 pre-open row count 的 GPU source 都走 deferred planning。
    // 显式宽度**只用于构造 source**,不是完整 GPU plan；open 后必须用聚合的
    // 真实行数补齐 streams / residency / effective budget。这个 seam 按能力
    // (`n_rows_peek`)分派,不能按 CSV/Parquet 格式名特判,否则下一个 streaming
    // 或远端 source 会重现同一个 bug。
    let (plan, deferred_gpu_plan) = match (n_rows_peek, cfg.device) {
        #[cfg(feature = "cuda")]
        (None, Some(_)) => {
            let c = cfg.requested_cols_per_block.filter(|c| *c > 0).ok_or_else(|| {
                PyValueError::new_err(
                    "GPU input needs an explicit cols_per_block when the row count is unavailable before open; memory, stream and residency planning finish with the real row count after open",
                )
            })?;
            (
                BlockingPlan::explicit(c, n_features, cfg.params.nthread),
                true,
            )
        }
        _ => (resolve_blocking(&cfg, n_features, n_rows_peek)?, false),
    };
    cfg.params.cols_per_block = plan.effective;
    let user_gpu = user_gpu_geometry(&cfg);
    // 规划器解出来的几何是权威的(见 `BlockingPlan`)。
    apply_resolved_geometry(&mut cfg, &plan);
    let cols_per_block = plan.effective;
    let ingest_threads =
        resolved_ingest_workers(cfg.ingest_threads, fmt, &paths, n_features, cols_per_block);
    let ingest_reporter = IngestReporter::start(
        cfg.quiet,
        fmt,
        &paths,
        cfg.ingest_threads,
        ingest_threads,
        n_rows_peek,
    );
    // 一个入口同时吃单文件和多文件:调用方给的是**已经解析好的路径列表**,
    // 单文件只是 `len() == 1` 的退化情况,不需要第二条代码路径。
    let max_bin = cfg.params.max_bin;
    let ingest_quiet = cfg.quiet;
    let open_paths = |ps: &[std::path::PathBuf],
                      targets: crate::source::Targets<'_>,
                      cuts: Option<crate::columns::BinCuts>,
                      first_csv_schema: Option<arrow::datatypes::SchemaRef>|
     -> anyhow::Result<crate::source::parquet_source::Dataset> {
        match fmt {
            "parquet" => parquet_source::open_many_observed(
                ps,
                targets,
                cuts,
                max_bin,
                cols_per_block,
                ingest_threads,
                |phase, readers, transforms| {
                    if !ingest_quiet {
                        eprintln!("INGEST_PLAN phase={phase} reader_slots={readers} transform_threads={transforms} inline_reader={} total_budget={ingest_threads}", readers == 1);
                    }
                },
            ),
            "csv" => match first_csv_schema {
                Some(schema) => csv_source::open_many_with_header_and_nthread_prepared(
                    ps,
                    targets,
                    cuts,
                    max_bin,
                    cols_per_block,
                    header,
                    ingest_threads,
                    schema,
                ),
                None => csv_source::open_many_with_header_and_nthread(
                    ps,
                    targets,
                    cuts,
                    max_bin,
                    cols_per_block,
                    header,
                    ingest_threads,
                ),
            },
            other => Err(anyhow::anyhow!("unknown format {other:?}")),
        }
    };
    // 验证集同样接受多文件规格。
    // ⚠️ 验证集**按列名**绑定到训练集的特征(无表头 CSV 只能按位置):
    // 按位置读的话,列顺序不同的验证文件会被静默错配,早停跟着选错轮数。
    let open_spec = |spec: &str,
                     train_features: &[String],
                     cuts: Option<crate::columns::BinCuts>|
     -> anyhow::Result<crate::source::parquet_source::Dataset> {
        let r = crate::source::input::resolve(&[spec.to_string()])?;
        anyhow::ensure!(
            r.format == resolved.format,
            "validation set format ({}) differs from the training set ({})",
            r.format.label(),
            resolved.format.label()
        );
        let eval_targets = if header || fmt == "parquet" {
            targets.with_features(train_features)
        } else {
            targets
        };
        open_paths(&r.paths, eval_targets, cuts, None)
    };

    let device = cfg.device;

    // Parquet 走**按需加载**:列块用时才读,读完就还回去。
    // CSV 做不到 —— 文本格式没法只取某几列,只能全量量化后驻留。
    // ⚠️ **单文件 parquet 保持原来的按需加载 / 量化缓存路径**(streaming、
    // 部分列读、显存规划都挂在它上面);多文件落到下面的 Dataset 路径。
    // `ParquetBlockSource` 从 path 到 cache manifest 都是单文件形状,
    // 为多文件重构它不属于 v0.0.1 的范围。
    // ⚠️ 代价要说清楚:多文件走的是量化后常驻内存的 `Dataset` 路径,
    // 和单文件的按需加载不是同一条内存曲线。
    if fmt == "parquet" && paths.len() == 1 {
        let names = parquet_source::feature_names_of(&paths[0], targets).map_err(to_py_err)?;
        cfg.resolve_monotone(n_features, &names)?;
        margin::continued_feature_names(prior_model.as_deref(), names.clone())?;
        let (mut train_cfg, lower_is_better) = cfg.train_config()?;
        train_cfg.prior_model = prior_model.as_ref().map(|p| p.inner.clone());
        let Callbacks {
            list: mut cbs,
            history: handle,
            error: cb_error,
        } = build_callbacks(
            lower_is_better,
            early_stopping_rounds,
            verbose_eval,
            callbacks,
        )?;
        if !cfg.quiet && device.is_none() {
            eprintln!(
                "CPU_THREADS requested={} resolved={}",
                train_cfg.params.nthread,
                effective_nthread(train_cfg.params.nthread)
            );
        }
        let cache_options = QuantizedCacheOptions {
            budget_bytes: cfg.cache_budget_bytes,
            path: cfg.cache_path.as_ref().map(std::path::PathBuf::from),
        };
        let (src, labels) = ParquetBlockSource::open_with_cache_observed(
            path,
            targets,
            cfg.params.max_bin,
            cfg.params.cols_per_block,
            cache_options,
            ingest_threads,
            |phase, readers, transforms| {
                if !ingest_quiet {
                    eprintln!("INGEST_PLAN phase={phase} reader_slots={readers} transform_threads={transforms} inline_reader={} total_budget={ingest_threads}", readers == 1);
                }
            },
        )
        .map_err(to_py_err)?;
        refine_gpu_plan_for_source(&cfg, user_gpu, n_features, &src, &mut train_cfg.params)?;
        // 权重是单独一列,不进量化缓存;每次 open 都从源文件读。
        let read_weights = |p: &str| -> PyResult<Option<Vec<f32>>> {
            let Some(w) = targets.weight else {
                return Ok(None);
            };
            let values = parquet_source::read_f32_column(p, w).map_err(to_py_err)?;
            crate::source::check_weight_column(&values, w)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            Ok(Some(values))
        };
        let weights = read_weights(path)?;
        // Ranking: the query-id column gives the groups; the weight column must hold
        // one weight per group (repeated on its rows).
        let read_qid = |p: &str| -> PyResult<Option<Vec<u64>>> {
            let Some(q) = targets.qid else {
                return Ok(None);
            };
            parquet_source::read_u64_columns(&[std::path::PathBuf::from(p)], q)
                .map(Some)
                .map_err(to_py_err)
        };
        let group_inputs = |ids: Option<Vec<u64>>,
                            w: Option<Vec<f32>>|
         -> PyResult<(Option<Vec<f32>>, Option<Vec<u32>>)> {
            match ids {
                None => Ok((w, None)),
                Some(ids) => {
                    let (gp, gw) =
                        file_groups(&ids, w.as_deref(), targets.weight).map_err(to_py_err)?;
                    Ok((gw, Some(gp)))
                }
            }
        };
        let (weights, train_group_ptr) = group_inputs(read_qid(path)?, weights)?;
        let mut train_cfg = train_cfg;
        train_cfg.group_ptr = train_group_ptr;
        ingest_reporter.complete(src.n_rows());
        let mut eval_data = Vec::new();
        let mut eval_paths: Vec<String> = Vec::new();
        if let Some(list) = evals {
            for item in list.iter() {
                let (p, name): (String, String) = item.extract()?;
                eval_paths.push(p.clone());
                // 验证集继承**预算**但绝不继承 **cache_path**:它的量化
                // 结果和训练集不同,共用一个目录会互相覆盖(而且下次复用
                // 时校验会直接报错)。给它自己的路径要用户显式指定。
                let d = ParquetBlockSource::with_cuts_and_cache_and_nthread(
                    &p,
                    targets.with_features(&names),
                    src.cuts().clone(),
                    cfg.params.cols_per_block,
                    QuantizedCacheOptions {
                        budget_bytes: cfg.cache_budget_bytes,
                        path: None,
                    },
                    ingest_threads,
                )
                .map_err(to_py_err)?;
                let ew = read_weights(&p)?;
                let (ew, egp) = group_inputs(read_qid(&p)?, ew)?;
                eval_data.push((name, d, ew, egp));
            }
        }
        // Starting margins (see `margin::dataset_margin`): train, then each eval.
        train_cfg.base_margin = margin::dataset_margin(
            start_margin,
            &resolved,
            &margin_names,
            header,
            src.n_rows(),
            k_out,
            "training set",
        )?;
        for (i, (name, (_, l), _, _)) in eval_data.iter().enumerate() {
            let eval_resolved = crate::source::input::resolve(std::slice::from_ref(&eval_paths[i]))
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            train_cfg.eval_base_margins.push(margin::dataset_margin(
                margin::list_item(eval_start_margins, i)?.as_ref(),
                &eval_resolved,
                &margin_names,
                header,
                l.len(),
                k_out,
                &format!("eval set {name:?}"),
            )?);
        }
        let evals_ref: Vec<EvalSet> = eval_data
            .iter()
            .map(|(name, (s, l), w, gp)| EvalSet {
                name: name.clone(),
                source: s,
                labels: l,
                weights: w.as_deref(),
                group_ptr: gp.as_deref(),
            })
            .collect();
        if !cfg.quiet {
            eprintln!(
                "SETUP input=parquet files=1 rows={} features={n_features} weights={} seconds={:.3}",
                src.n_rows(),
                on_off(weights.is_some()),
                overall_started.elapsed().as_secs_f64()
            );
        }
        let train_started = std::time::Instant::now();
        let model = py
            .detach(|| match device {
                #[cfg(feature = "cuda")]
                Some(ord) => crate::train::train_gpu_weighted(
                    ord,
                    &src,
                    &labels,
                    weights.as_deref(),
                    &evals_ref,
                    &train_cfg,
                    &mut cbs,
                    &Local,
                ),
                #[cfg(not(feature = "cuda"))]
                Some(_) => Err(anyhow::anyhow!("this build has no CUDA support")),
                None => train_weighted(
                    &src,
                    &labels,
                    weights.as_deref(),
                    &evals_ref,
                    &train_cfg,
                    &mut cbs,
                    &Local,
                ),
            })
            .map_err(to_py_err)?;
        if !cfg.quiet {
            eprintln!(
                "TRAIN backend={} trees={} seconds={:.3}",
                device.map_or_else(|| "cpu".to_string(), |d| format!("cuda:{d}")),
                model.trees.len(),
                train_started.elapsed().as_secs_f64()
            );
            eprintln!(
                "END_TO_END seconds={:.3}",
                overall_started.elapsed().as_secs_f64()
            );
        }
        raise_callback_error(&cb_error)?;
        // 命名模式:列名取自 parquet schema,标签列已排除(分支开头已读出)。
        let names = margin::continued_feature_names(prior_model.as_deref(), names)?;
        return Ok(finish(
            model,
            handle,
            names,
            cfg.quiet,
            cfg.model_io_threads,
            cfg.ingest_threads,
        ));
    }

    let ds = open_paths(&paths, targets, None, csv_preflight.clone()).map_err(to_py_err)?;
    ingest_reporter.complete(ds.source.n_rows());
    // A deferred source only learns its aggregate row count during open.
    // Complete the GPU plan now, preserving the requested block width while
    // resolving memory budget, stream count and residency from the real shape.
    #[cfg(feature = "cuda")]
    if deferred_gpu_plan {
        let completed = resolve_blocking(&cfg, n_features, Some(ds.source.n_rows() as u64))?;
        if completed.effective != cols_per_block {
            return Err(PyRuntimeError::new_err(format!(
                "deferred GPU planner changed source geometry from {cols_per_block} to {}; explicit cols_per_block must be preserved",
                completed.effective
            )));
        }
        cfg.params.cols_per_block = completed.effective;
        apply_resolved_geometry(&mut cfg, &completed);
    }
    {
        let mut params = cfg.params.clone();
        refine_gpu_plan_for_source(&cfg, user_gpu, n_features, &ds.source, &mut params)?;
        cfg.params = params;
    }
    cfg.resolve_monotone(n_features, &ds.feature_names)?;
    margin::continued_feature_names(prior_model.as_deref(), ds.feature_names.clone())?;
    let (mut train_cfg, lower_is_better) = cfg.train_config()?;
    train_cfg.prior_model = prior_model.as_ref().map(|p| p.inner.clone());
    // Ranking: query ids come from the qid column of every file, in dataset order.
    let read_qid_paths = |ps: &[std::path::PathBuf]| -> PyResult<Option<Vec<u64>>> {
        let Some(q) = targets.qid else {
            return Ok(None);
        };
        match fmt {
            "parquet" => parquet_source::read_u64_columns(ps, q),
            _ => csv_source::read_u64_column(ps, header, q),
        }
        .map(Some)
        .map_err(to_py_err)
    };
    let group_inputs = |ids: Option<Vec<u64>>,
                        w: Option<&[f32]>|
     -> PyResult<(Option<Vec<f32>>, Option<Vec<u32>>)> {
        match ids {
            None => Ok((w.map(<[f32]>::to_vec), None)),
            Some(ids) => {
                let (gp, gw) = file_groups(&ids, w, targets.weight).map_err(to_py_err)?;
                Ok((gw, Some(gp)))
            }
        }
    };
    let (ds_weights, train_group_ptr) =
        group_inputs(read_qid_paths(&paths)?, ds.weights.as_deref())?;
    train_cfg.group_ptr = train_group_ptr;
    let Callbacks {
        list: mut cbs,
        history: handle,
        error: cb_error,
    } = build_callbacks(
        lower_is_better,
        early_stopping_rounds,
        verbose_eval,
        callbacks,
    )?;
    if !cfg.quiet && device.is_none() {
        eprintln!(
            "CPU_THREADS requested={} resolved={}",
            train_cfg.params.nthread,
            effective_nthread(train_cfg.params.nthread)
        );
    }
    let mut eval_data = Vec::new();
    let mut eval_paths: Vec<String> = Vec::new();
    if let Some(list) = evals {
        for item in list.iter() {
            let (p, name): (String, String) = item.extract()?;
            let d = open_spec(&p, &ds.feature_names, Some(ds.source.cuts().clone()))
                .map_err(to_py_err)?;
            eval_paths.push(p.clone());
            let eval_paths = crate::source::input::resolve(&[p.clone()])
                .map_err(|e| PyValueError::new_err(format!("{e}")))?
                .paths;
            let (ew, egp) = group_inputs(read_qid_paths(&eval_paths)?, d.weights.as_deref())?;
            eval_data.push((name, d, ew, egp));
        }
    }
    train_cfg.base_margin = margin::dataset_margin(
        start_margin,
        &resolved,
        &margin_names,
        header,
        ds.source.n_rows(),
        k_out,
        "training set",
    )?;
    for (i, (name, d, _, _)) in eval_data.iter().enumerate() {
        let eval_resolved = crate::source::input::resolve(std::slice::from_ref(&eval_paths[i]))
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        train_cfg.eval_base_margins.push(margin::dataset_margin(
            margin::list_item(eval_start_margins, i)?.as_ref(),
            &eval_resolved,
            &margin_names,
            header,
            d.labels.len(),
            k_out,
            &format!("eval set {name:?}"),
        )?);
    }
    let evals_ref: Vec<EvalSet> = eval_data
        .iter()
        .map(|(name, d, w, gp)| EvalSet {
            name: name.clone(),
            source: &d.source,
            labels: &d.labels,
            weights: w.as_deref(),
            group_ptr: gp.as_deref(),
        })
        .collect();

    if !cfg.quiet {
        eprintln!(
            "SETUP input={fmt} files={} rows={} features={n_features} weights={} seconds={:.3}",
            paths.len(),
            ds.source.n_rows(),
            on_off(ds.weights.is_some()),
            overall_started.elapsed().as_secs_f64()
        );
    }
    let train_started = std::time::Instant::now();

    let model = py
        .detach(|| match device {
            #[cfg(feature = "cuda")]
            Some(ord) => crate::train::train_gpu_weighted(
                ord,
                &ds.source,
                &ds.labels,
                ds_weights.as_deref(),
                &evals_ref,
                &train_cfg,
                &mut cbs,
                &Local,
            ),
            #[cfg(not(feature = "cuda"))]
            Some(_) => Err(anyhow::anyhow!("this build has no CUDA support")),
            None => train_weighted(
                &ds.source,
                &ds.labels,
                ds_weights.as_deref(),
                &evals_ref,
                &train_cfg,
                &mut cbs,
                &Local,
            ),
        })
        .map_err(to_py_err)?;
    if !cfg.quiet {
        eprintln!(
            "TRAIN backend={} trees={} seconds={:.3}",
            device.map_or_else(|| "cpu".to_string(), |d| format!("cuda:{d}")),
            model.trees.len(),
            train_started.elapsed().as_secs_f64()
        );
        eprintln!(
            "END_TO_END seconds={:.3}",
            overall_started.elapsed().as_secs_f64()
        );
    }
    raise_callback_error(&cb_error)?;
    let names = margin::continued_feature_names(prior_model.as_deref(), ds.feature_names.clone())?;
    Ok(finish(
        model,
        handle,
        names,
        cfg.quiet,
        cfg.model_io_threads,
        cfg.ingest_threads,
    ))
}

/// A ranking metric (`ndcg`, `ndcg@k`, `ndcg@k-`, `map`, `map@k`, `pre@k`, `auc`, `aucpr`) of scores
/// against labels, with XGBoost's semantics. `qid` gives the query groups and
/// `weight` is one value per group. Used by `FerrisBoostRanker.score`.
#[pyfunction]
#[pyo3(signature = (metric, pred, label, qid, weight=None, ndcg_exp_gain=true))]
fn rank_score(
    metric: &str,
    pred: PyReadonlyArray1<'_, f32>,
    label: PyReadonlyArray1<'_, f32>,
    qid: PyReadonlyArray1<'_, u64>,
    weight: Option<PyReadonlyArray1<'_, f32>>,
    ndcg_exp_gain: bool,
) -> PyResult<f64> {
    let value_err = |e: anyhow::Error| PyValueError::new_err(e.to_string());
    let (pred, label) = (pred.as_slice()?, label.as_slice()?);
    if pred.len() != label.len() {
        return Err(PyValueError::new_err(format!(
            "pred has {} entries, label {}",
            pred.len(),
            label.len()
        )));
    }
    let group_ptr = crate::rank::group_ptr_from_qid(qid.as_slice()?).map_err(value_err)?;
    crate::rank::validate_group_ptr(&group_ptr, pred.len(), "qid").map_err(value_err)?;
    let weight = weight
        .map(|w| w.as_slice().map(<[f32]>::to_vec))
        .transpose()?;
    if let Some(w) = &weight {
        if w.len() != group_ptr.len() - 1 {
            return Err(PyValueError::new_err(format!(
                "ranking weights are per query group: {} weights for {} groups",
                w.len(),
                group_ptr.len() - 1
            )));
        }
    }
    let auc_kind = match metric {
        "auc" => Some(crate::rank::GroupAucKind::Roc),
        "aucpr" => Some(crate::rank::GroupAucKind::Pr),
        _ => None,
    };
    if let Some(kind) = auc_kind {
        use crate::metrics::Metric;
        let m = crate::rank::GroupAuc::new(kind, false);
        m.check_labels(label, "labels").map_err(value_err)?;
        return Ok(m.score(pred, label, weight.as_deref(), &group_ptr));
    }
    let m = crate::rank::RankMetric::parse(metric, ndcg_exp_gain).map_err(value_err)?;
    m.validate_labels(label, "labels").map_err(value_err)?;
    Ok(m.score(pred, label, weight.as_deref(), Some(&group_ptr)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(rank_score, m)?)?;
    m.add_function(wrap_pyfunction!(train_dense, m)?)?;
    m.add_function(wrap_pyfunction!(train_file, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
