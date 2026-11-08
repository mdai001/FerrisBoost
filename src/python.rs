//! Python 绑定。
//!
//! **这一层不做任何数值处理。** 参数解析、dtype 归一化、输入类型分发
//! 都在 `python/ferrisboost/__init__.py` 里;这里只负责把已经规整好的
//! 数据交给训练循环。预测是例外:它是公开的 PyO3 边界，必须正确解释
//! NumPy 的 stride，不能把 F-order/切片视为 C row-major 内存。

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray,
};
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
/// 就是 ValueError。全都甩成 RuntimeError 的话,调用方没法按类型处理,
/// 而且 traceback 里只剩一句没头没尾的话。
fn to_py_err(e: anyhow::Error) -> PyErr {
    let msg = format!("{e:#}");
    if let Some(io) = e.downcast_ref::<std::io::Error>() {
        if io.kind() == std::io::ErrorKind::NotFound {
            return pyo3::exceptions::PyFileNotFoundError::new_err(msg);
        }
    }
    if msg.contains("打不开") || msg.contains("没有列") {
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
                .unwrap_or_else(|| "Rust 侧 panic,没有消息".to_string());
            Err(PyRuntimeError::new_err(format!(
                "ferrisboost 内部错误:{msg}"
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
                    Ok(Some(d)) => d.cast_into::<PyDict>().expect("上一步就是 PyDict"),
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
            use crate::gpu_mem_plan::{resolve_gpu_streaming_plan, StreamingMemModel};
            // ⚠️ **必须查用户真正要用的那张卡。** 这里以前固定
            // `CudaContext::new(0)`,而 buffer 实际分配在 `train_gpu(ordinal, ..)`
            // 指定的卡上 —— `device="cuda:1"` 时预算读的是 GPU 0 的空闲显存,
            // 分配却发生在 GPU 1。两张卡型号不同、或占用不同时,轻则规划过小
            // (白白 streaming),重则在 GPU 1 上 **OOM**。
            // 规划和分配必须看同一张卡。
            let ctx = crate::backend::cuda::cuda_context(ordinal)
                .map_err(|e| PyValueError::new_err(format!("打不开 CUDA device {ordinal}:{e}")))?;
            let (free, total) = ctx
                .mem_get_info()
                .map_err(|e| PyValueError::new_err(format!("读不到显存信息:{e}")))?;
            // ⚠️ 列块 buffer 是 streaming 预算里最大的一项,而 `per_col_bytes`
            // **整个乘以 stream 数** —— 所以块宽和 stream 数必须**一起解**。
            // 分开解会得到一个装不下的组合:规划器按单流选了宽块,运行时
            // 却按路径开了双流,然后 CUDA OOM(而不是明确的规划失败)。
            let fast = matches!(cfg.params.gpu_math, crate::types::GpuMath::Fast);
            let max_bin = cfg.params.max_bin as u64;
            // ⚠️ **拿不到真实行数就明确报错,不要拿别的数顶上。**
            // 这里曾经传的是 `n_rounds`(注释还写着"下面用真实行数覆盖",
            // 而下面根本没有覆盖),于是模型把上千万行当成几十行算,
            // 解出来的块宽在真实数据上放不下。
            // 「规划失败必须报错,不能悄悄回退」——顶一个假行数比报错更糟。
            let rows = n_rows.ok_or_else(|| {
                PyValueError::new_err(
                    "GPU 规划需要数据行数,但当前输入源在 open 前无法提供；请先用显式 cols_per_block 构造 source，再用 open 后的真实聚合行数完成规划",
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
                |streams| {
                    let model = StreamingMemModel::new(
                        rows,
                        max_bin,
                        streams,
                        cfg.params.hist_nodes_per_batch.unwrap_or(16) as u64,
                        cfg.params.device_quantize.unwrap_or(true),
                        fast,
                        true,
                        crate::gpu_mem_plan::RM_STAGE_BYTES,
                    );
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
                    let model = if fast {
                        model.with_fast_multiclass(rows, cfg.objective.num_outputs() as u64)
                    } else {
                        model
                    };
                    if cfg.params.gpu_split_cache
                        && !matches!(
                            std::env::var("FB_HIST_BUCKET_PROFILE").as_deref(),
                            Ok("1") | Ok("true")
                        )
                    {
                        model.with_resident_split_cache(
                            max_bin,
                            cfg.params.hist_nodes_per_batch.unwrap_or(16) as u64,
                        )
                    } else {
                        model
                    }
                },
            )
            .map_err(to_py_err)?;
            if !cfg.quiet {
                eprintln!("{}", p.report());
            }
            Ok(BlockingPlan::from_plan(&p, n_features, cfg.params.nthread))
        }
        #[cfg(not(feature = "cuda"))]
        Some(_) => Err(PyValueError::new_err(
            "这个构建没有 CUDA 支持:请用 --features cuda 重新构建,或改用 device=\"cpu\"",
        )),
        None => Ok(BlockingPlan::resolve(
            cfg.requested_cols_per_block,
            n_features,
            cfg.params.nthread,
            cfg.params.max_bin as usize,
        )),
    }
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
    pool: &mut Option<rayon::ThreadPool>,
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
        *pool = Some(crate::threading::build_pool(workers)?);
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

/// Cache-friendly fast path for C-contiguous NumPy input. Parallelism still
/// owns disjoint deterministic row ranges; inside each range the CPU inference
/// model blocks rows while preserving every row's original tree-add order.
fn score_contiguous_rows(
    model: &crate::inference::CpuInferenceModel,
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    n_trees: usize,
    workers: usize,
    pool: &mut Option<rayon::ThreadPool>,
) -> anyhow::Result<Vec<f32>> {
    let mut out = vec![0.0f32; n_rows];
    if n_rows == 0 {
        return Ok(out);
    }
    if workers <= 1 {
        model.predict_margins(data, n_features, n_trees, &mut out)?;
        return Ok(out);
    }
    if pool.is_none() {
        *pool = Some(crate::threading::build_pool(workers)?);
    }
    let chunks = workers.saturating_mul(4).max(1);
    let rows_per_chunk = n_rows.div_ceil(chunks).max(1);
    pool.as_ref()
        .expect("prediction pool just initialized")
        .install(|| {
            out.par_chunks_mut(rows_per_chunk)
                .enumerate()
                .try_for_each(|(chunk, values)| {
                    let first = chunk * rows_per_chunk;
                    let value_first = first * n_features;
                    let value_last = value_first + values.len() * n_features;
                    model.predict_margins(
                        &data[value_first..value_last],
                        n_features,
                        n_trees,
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

enum PredictionParquetMessage {
    Batch(Vec<f32>),
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

fn prediction_parquet_writer(
    file: std::fs::File,
    output_path: std::path::PathBuf,
    prediction_column: String,
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
    let schema = Arc::new(Schema::new(vec![Field::new(
        prediction_column,
        DataType::Float32,
        false,
    )]));
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_size(PREDICTION_PARQUET_ROW_GROUP_ROWS)
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties))
        .context("创建 prediction Parquet writer 失败")?;
    let mut rows_written = 0usize;
    loop {
        match receiver.recv() {
            Ok(PredictionParquetMessage::Batch(values)) => {
                rows_written = rows_written
                    .checked_add(values.len())
                    .context("prediction 输出行数溢出")?;
                let values: ArrayRef = Arc::new(Float32Array::from(values));
                let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values])
                    .context("创建 prediction Arrow batch 失败")?;
                writer
                    .write(&batch)
                    .context("写 prediction Parquet batch 失败")?;
            }
            Ok(PredictionParquetMessage::Finish) => break,
            Err(_) => anyhow::bail!("prediction Parquet pipeline 在完成前被取消"),
        }
    }
    writer
        .finish()
        .context("完成 prediction Parquet footer 失败")?;
    writer
        .inner()
        .sync_all()
        .context("同步 prediction Parquet 文件失败")?;
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
) -> PyResult<usize> {
    use crate::tree::SchemaMode;

    if prediction_column.trim().is_empty() {
        return Err(PyValueError::new_err(
            "prediction_column 不能为空或只包含空白",
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
            "prediction 输出路径必须使用 .parquet 或 .pq 扩展名",
        ));
    }

    let resolved = crate::source::input::resolve(&path_specs)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let mode = model.inner.schema_mode.ok_or_else(|| {
        PyValueError::new_err(
            "这个模型是在记录 schema 之前保存的,无法安全地按文件预测:\
             空的特征名单既可能是位置模式,也可能是命名模式没存名字,\
             二者无法区分。请用当前版本重新训练,或改用 numpy 数组预测。",
        )
    })?;
    if mode == SchemaMode::Named && model.inner.feature_names.is_empty() {
        return Err(PyValueError::new_err("模型标为命名模式但没有特征名单"));
    }
    if resolved.format == crate::source::input::InputFormat::Parquet && !header {
        return Err(PyValueError::new_err("header=False 只适用于 CSV"));
    }
    if resolved.format == crate::source::input::InputFormat::Csv {
        match (mode, header) {
            (SchemaMode::Named, false) => {
                return Err(PyValueError::new_err(
                    "命名模式模型不能用 header=False 的位置 CSV 预测",
                ))
            }
            (SchemaMode::Positional, true) => {
                return Err(PyValueError::new_err(
                    "位置模式模型预测 CSV 时必须显式传 header=False",
                ))
            }
            _ => {}
        }
    }

    let names = model.inner.feature_names.clone();
    let canonical_n_features = model.inner.n_features;
    let inference = model.inference_model().map_err(to_py_err)?;
    let projected_features = inference.canonical_features().to_vec();
    let n_features = inference.n_features();
    let upto = n_trees.unwrap_or(model.inner.trees.len());
    let inner = inference.model();
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
    let work_per_row = prediction_work_per_row(inner, upto);
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
                "prediction 输出已存在,不会覆盖: {}",
                output_path.display()
            )))
        }
        Err(error) => {
            return Err(PyValueError::new_err(format!(
                "无法创建 prediction 输出 {}: {error}",
                output_path.display()
            )))
        }
    };

    let (sender, receiver) = std::sync::mpsc::sync_channel(PREDICTION_PARQUET_CHANNEL_BATCHES);
    let writer_path = output_path.clone();
    let writer_column = prediction_column.to_string();
    let writer_handle = match std::thread::Builder::new()
        .name("fb-predict-parquet".to_string())
        .spawn(move || prediction_parquet_writer(file, writer_path, writer_column, receiver))
    {
        Ok(handle) => handle,
        Err(error) => {
            let _ = std::fs::remove_file(&output_path);
            return Err(PyRuntimeError::new_err(format!(
                "无法启动 prediction Parquet writer: {error}"
            )));
        }
    };

    if !model.quiet {
        eprintln!(
            "PREDICT_OUTPUT start format=parquet output={} column={} \
             ingest_workers={} predict_workers={} writer_workers=1 queue_batches={}",
            output_path.display(),
            prediction_column,
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
    let input_result = py.detach(|| {
        let predictor = inference.cpu_predictor()?;
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
                let mut batch = score_contiguous_rows(
                    &predictor,
                    data,
                    n_rows,
                    n_features,
                    upto,
                    workers,
                    &mut predict_pool,
                )?;
                transform_prediction_margins(inner, &mut batch, output_margin);
                sender
                    .send(PredictionParquetMessage::Batch(batch))
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
                        "prediction Parquet 写出失败: {writer_error:#}"
                    )))
                }
                Err(_) => return Err(PyRuntimeError::new_err("prediction Parquet writer panic")),
                Ok(Ok(_)) => {}
            }
        }
        return Err(PyValueError::new_err(format!("{input_error:#}")));
    }
    let rows_read = input_result.expect("checked above");
    if sender.send(PredictionParquetMessage::Finish).is_err() {
        drop(sender);
        return match writer_handle.join() {
            Ok(Err(writer_error)) => Err(PyRuntimeError::new_err(format!(
                "prediction Parquet 写出失败: {writer_error:#}"
            ))),
            Err(_) => Err(PyRuntimeError::new_err("prediction Parquet writer panic")),
            Ok(Ok(_)) => Err(PyRuntimeError::new_err(
                "prediction Parquet writer 在完成信号前停止",
            )),
        };
    }
    drop(sender);
    let rows_written = match writer_handle.join() {
        Ok(Ok(rows)) => rows,
        Ok(Err(error)) => {
            return Err(PyRuntimeError::new_err(format!(
                "prediction Parquet 写出失败: {error:#}"
            )))
        }
        Err(_) => return Err(PyRuntimeError::new_err("prediction Parquet writer panic")),
    };
    if rows_read != rows_written {
        let _ = std::fs::remove_file(&output_path);
        return Err(PyRuntimeError::new_err(format!(
            "prediction 输出行数不一致:读取 {rows_read},写入 {rows_written}"
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

fn project_canonical_rows(
    row_major: &[f32],
    n_rows: usize,
    canonical_width: usize,
    projected_features: &[usize],
) -> anyhow::Result<Vec<f32>> {
    let expected = n_rows
        .checked_mul(canonical_width)
        .context("prediction canonical input size 溢出")?;
    if row_major.len() != expected {
        anyhow::bail!(
            "prediction canonical input 有 {} 个值,预期 {expected}",
            row_major.len()
        );
    }
    let output_len = n_rows
        .checked_mul(projected_features.len())
        .context("prediction compact input size 溢出")?;
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
            "这个 FerrisBoost build 没有 CUDA prediction 支持;请使用 use_gpu=False 或安装 CUDA wheel。",
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

/// 多分类文件预测:每类的单输出视图和它的压缩推理模型。
type MulticlassPredictors = (
    Arc<Vec<PyModel>>,
    Vec<Arc<crate::inference::CompactInferenceModel>>,
);

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
    /// 多分类:每类一个单输出视图(`Model::class_view`),带各自的预测缓存。
    class_views: Mutex<Option<Arc<Vec<PyModel>>>>,
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
                "iteration_range 和 n_trees 不能同时给:前者按轮次,后者按树的个数",
            ));
        }
        let k = self.inner.num_outputs();
        let rounds = self.inner.trees.len() / k;
        let end = if end == 0 { rounds } else { end };
        if end > rounds {
            return Err(PyValueError::new_err(format!(
                "iteration_range 的 end = {end} 超出了模型的 {rounds} 轮"
            )));
        }
        if begin > end {
            return Err(PyValueError::new_err(format!(
                "iteration_range 的 begin = {begin} 大于 end = {end}"
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

    /// 不带训练历史的普通包装(多分类的类视图用)。
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
            class_views: Mutex::new(None),
        }
    }

    fn class_views(&self) -> anyhow::Result<Arc<Vec<PyModel>>> {
        let mut cache = self
            .class_views
            .lock()
            .map_err(|_| anyhow::anyhow!("class view cache poisoned"))?;
        if let Some(v) = cache.as_ref() {
            return Ok(Arc::clone(v));
        }
        let views = Arc::new(
            (0..self.inner.num_outputs())
                .map(|c| {
                    PyModel::plain(
                        self.inner.class_view(c),
                        self.quiet,
                        self.model_io_threads,
                        self.ingest_threads,
                    )
                })
                .collect::<Vec<_>>(),
        );
        *cache = Some(Arc::clone(&views));
        Ok(views)
    }

    /// 多分类预测:每类视图各跑一遍单输出路径(CPU / GPU / 文件都复用),
    /// 再按行组合。返回按行排的 `n × K` margin。
    ///
    /// ⚠️ 文件输入会被读 K 遍(每类一遍);0.0.6 再做单遍多输出。
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
    ) -> PyResult<(usize, Vec<f32>)> {
        let k = self.inner.num_outputs();
        if let Some(specs) = path_specs_of(data)? {
            let arr = self.predict_files(
                py,
                specs,
                true,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                missing,
            )?;
            let arr = arr
                .cast::<numpy::PyArray2<f32>>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let margins = arr.readonly().as_slice()?.to_vec();
            return Ok((margins.len() / k, margins));
        }
        let views = self.class_views().map_err(to_py_err)?;
        let upto = n_trees.unwrap_or(self.inner.trees.len());
        let mut cols: Vec<Vec<f32>> = Vec::with_capacity(k);
        for (c, view) in views.iter().enumerate() {
            let nt = self.inner.class_tree_count(upto, c);
            let out = view.predict(
                py,
                data,
                true,
                Some(nt),
                header,
                predict_threads,
                use_gpu,
                None,
                false,
                false,
                false,
                false,
                // Array input: the sentinel was already mapped to NaN by `predict`.
                f64::NAN,
            )?;
            let arr = out
                .cast::<PyArray1<f32>>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            cols.push(arr.readonly().as_slice()?.to_vec());
        }
        let n = cols[0].len();
        let mut margins = vec![0.0f32; n * k];
        for (c, col) in cols.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                margins[i * k + c] = v;
            }
        }
        Ok((n, margins))
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
    ) -> PyResult<Bound<'py, PyAny>> {
        // 文件输入:单遍读、按 batch 给每类打分(见 predict_files)。
        if let Some(specs) = path_specs_of(data)? {
            return self.predict_files(
                py,
                specs,
                output_margin,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                missing,
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
        )?;
        let k = self.inner.num_outputs();
        if !output_margin {
            if let crate::types::Objective::Softmax { .. } = self.inner.objective {
                let labels: Vec<f32> = margins
                    .chunks_exact(k)
                    .map(|row| crate::objective::argmax(row) as f32)
                    .collect();
                return Ok(PyArray1::from_vec(py, labels).into_any());
            }
            margins
                .par_chunks_exact_mut(k)
                .for_each(crate::objective::softmax_in_place);
        }
        let arr = PyArray1::from_vec(py, margins)
            .reshape([n, k])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_any())
    }

    /// 多分类写 Parquet:`softprob` / margin 写 K 列 `{column}_{k}`,
    /// `softmax` 写一列类别下标。
    ///
    /// ⚠️ 仍先在内存里算完再写(不是单输出那条有界流式管线);输入只读一遍。
    /// 流式的多输出写出是后续工作(memo 0.0.7 carry-over 8)。
    #[allow(clippy::too_many_arguments)]
    fn multiclass_to_parquet(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        output_path: &str,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        prediction_column: &str,
        missing: f32,
    ) -> PyResult<usize> {
        use arrow::array::{ArrayRef, Float32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::basic::Compression;
        use parquet::file::properties::WriterProperties;

        // 和单输出路径同样的输出规则:列名非空、扩展名、不覆盖已有文件、
        // 失败时删掉半成品。全部在预测之前检查,不白算一遍。
        if prediction_column.trim().is_empty() {
            return Err(PyValueError::new_err(
                "prediction_column 不能为空或只包含空白",
            ));
        }
        let out_path = std::path::PathBuf::from(output_path);
        let valid_extension = out_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("parquet") || e.eq_ignore_ascii_case("pq"));
        if !valid_extension {
            return Err(PyValueError::new_err(
                "prediction 输出路径必须使用 .parquet 或 .pq 扩展名",
            ));
        }
        // 先拒绝已存在的目标(不白算一遍),但**不在这里创建文件**:
        // 输出若落在输入目录 / glob 里,先建出来的空文件会被当成输入读进去。
        // 输入全部读完、预测算完之后才 create_new(仍然原子地拒绝覆盖)。
        if out_path.exists() {
            return Err(pyo3::exceptions::PyFileExistsError::new_err(format!(
                "prediction 输出已存在,不会覆盖: {}",
                out_path.display()
            )));
        }
        let (n, mut margins) = self.multiclass_margins(
            py,
            data,
            n_trees,
            header,
            predict_threads,
            false,
            missing,
        )?;
        let file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&out_path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(pyo3::exceptions::PyFileExistsError::new_err(format!(
                    "prediction 输出已存在,不会覆盖: {}",
                    out_path.display()
                )))
            }
            Err(error) => {
                return Err(PyValueError::new_err(format!(
                    "无法创建 prediction 输出 {}: {error}",
                    out_path.display()
                )))
            }
        };
        let mut cleanup = IncompletePredictionOutput {
            path: out_path,
            complete: false,
        };

        let k = self.inner.num_outputs();
        let softmax_labels = !output_margin
            && matches!(
                self.inner.objective,
                crate::types::Objective::Softmax { .. }
            );
        let columns: Vec<(String, Vec<f32>)> = if softmax_labels {
            vec![(
                prediction_column.to_string(),
                margins
                    .chunks_exact(k)
                    .map(|r| crate::objective::argmax(r) as f32)
                    .collect(),
            )]
        } else {
            if !output_margin {
                margins
                    .par_chunks_exact_mut(k)
                    .for_each(crate::objective::softmax_in_place);
            }
            (0..k)
                .map(|c| {
                    (
                        format!("{prediction_column}_{c}"),
                        (0..n).map(|i| margins[i * k + c]).collect(),
                    )
                })
                .collect()
        };
        let write = || -> anyhow::Result<usize> {
            let schema = Arc::new(Schema::new(
                columns
                    .iter()
                    .map(|(name, _)| Field::new(name, DataType::Float32, false))
                    .collect::<Vec<_>>(),
            ));
            let arrays: Vec<ArrayRef> = columns
                .iter()
                .map(|(_, v)| Arc::new(Float32Array::from(v.clone())) as ArrayRef)
                .collect();
            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
                .context("创建 prediction Arrow batch 失败")?;
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .set_max_row_group_size(PREDICTION_PARQUET_ROW_GROUP_ROWS)
                .build();
            let mut writer = ArrowWriter::try_new(file, schema, Some(props))
                .context("创建 prediction Parquet writer 失败")?;
            writer.write(&batch).context("写 prediction Parquet 失败")?;
            writer.finish().context("完成 prediction Parquet 失败")?;
            writer
                .inner()
                .sync_all()
                .context("同步 prediction Parquet 文件失败")?;
            Ok(n)
        };
        let n = py.detach(write).map_err(to_py_err)?;
        cleanup.complete = true;
        Ok(n)
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
                        pred_leaf=false, missing=f64::NAN))]
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
    ) -> PyResult<Bound<'py, PyAny>> {
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
                    );
                }
            }
        }
        // SHAP contributions / interactions / leaf indices (CPU only; `use_gpu`
        // does not apply). Single code path for all objectives and inputs.
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
                missing,
            );
        }
        ensure_gpu_prediction_available(use_gpu)?;
        // The device predictor has no categorical decision: such models score on the CPU.
        let use_gpu = use_gpu && !self.inner.has_categorical_splits();
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
            );
        }
        // 先认路径:字符串 / PathLike / 它们的序列。认不出来才当数组。
        if let Some(specs) = path_specs_of(data)? {
            return self.predict_files(
                py,
                specs,
                output_margin,
                n_trees,
                header,
                predict_threads,
                use_gpu,
                missing,
            );
        }
        let x: PyReadonlyArray2<'py, f32> = array_input(py, data)?;
        self.predict_array(py, x, output_margin, n_trees, predict_threads, use_gpu)
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
        guard(|| {
            ensure_gpu_prediction_available(use_gpu)?;
            // The device predictor has no categorical decision: such models score on the CPU.
            let use_gpu = use_gpu && !self.inner.has_categorical_splits();
            if self.inner.num_outputs() > 1 {
                return Err(PyValueError::new_err(
                    "多分类模型的预测是二维的,请用 Model.predict()",
                ));
            }
            let shape = x.shape();
            let (n_rows, n_features) = (shape[0], shape[1]);
            if n_features != self.inner.n_features {
                return Err(PyValueError::new_err(format!(
                    "模型是按 {} 个特征训的,传进来的是 {n_features} 列",
                    self.inner.n_features
                )));
            }
            // 和训练一样拒绝保留值 f32::MAX(CPU 和 GPU 预测都走这里)。
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
                    "预测数据第 {r} 行第 {c} 列是 f32::MAX:这个值保留给缺失值分裂边界,训练和预测都拒绝它;请先缩放这个特征"
                )));
            }
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
                            py.detach(|| predictor.predict_margins(data, n_rows, upto))
                        } else {
                            py.detach(|| {
                                let projected = project_canonical_rows(
                                    data,
                                    n_rows,
                                    n_features,
                                    inference.canonical_features(),
                                )?;
                                predictor.predict_margins(&projected, n_rows, upto)
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
                            predictor.predict_margins(&projected, n_rows, upto)
                        })
                    }
                    .map_err(to_py_err)?;
                    transform_prediction_margins(&self.inner, &mut out, output_margin);
                    return Ok(out.to_pyarray(py));
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
            let out = if x.is_c_contiguous() {
                let data = x.as_slice()?;
                let predictor = self.cpu_predictor().map_err(to_py_err)?;
                py.detach(|| {
                    let mut out = score_contiguous_rows(
                        &predictor, data, n_rows, n_features, upto, workers, &mut pool,
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
            Ok(out.to_pyarray(py))
        })
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
        use crate::tree::SchemaMode;
        ensure_gpu_prediction_available(use_gpu)?;
        // The device predictor has no categorical decision: such models score on the CPU.
        let use_gpu = use_gpu && !self.inner.has_categorical_splits();
        let resolved = crate::source::input::resolve(&path_specs)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        // ⚠️ 老模型没记过绑定模式,而空名单既可能是"真·位置模式",也可能是
        // "命名模式但当时没存名字" —— 从 JSON 里分不出来。**猜错会静默算错**,
        // 所以这里直接拒,并说清楚怎么办。
        let mode = self.inner.schema_mode.ok_or_else(|| {
            PyValueError::new_err(
                "这个模型是在记录 schema 之前保存的,无法安全地按文件预测:\
                 空的特征名单既可能是位置模式,也可能是命名模式没存名字,\
                 二者无法区分。请用当前版本重新训练,或改用 numpy 数组预测。",
            )
        })?;
        if mode == SchemaMode::Named && self.inner.feature_names.is_empty() {
            return Err(PyValueError::new_err("模型标为命名模式但没有特征名单"));
        }
        if resolved.format == crate::source::input::InputFormat::Parquet && !header {
            return Err(PyValueError::new_err("header=False 只适用于 CSV"));
        }
        if resolved.format == crate::source::input::InputFormat::Csv {
            match (mode, header) {
                (SchemaMode::Named, false) => {
                    return Err(PyValueError::new_err(
                        "命名模式模型不能用 header=False 的位置 CSV 预测",
                    ))
                }
                (SchemaMode::Positional, true) => {
                    return Err(PyValueError::new_err(
                        "位置模式模型预测 CSV 时必须显式传 header=False",
                    ))
                }
                _ => {}
            }
        }

        let names = self.inner.feature_names.clone();
        let canonical_n_features = self.inner.n_features;
        let upto = n_trees.unwrap_or(self.inner.trees.len());
        let k_out = self.inner.num_outputs();
        // 多分类:文件只读一遍。每个 batch 按**完整**规范特征投影一次,
        // 然后每个类别视图在内存里 gather 自己(压缩过)的特征子集、用自己的
        // 单输出预测器打分,按行交错后做 softmax / argmax。
        let multiclass: Option<MulticlassPredictors> = if k_out > 1 {
            let views = self.class_views().map_err(to_py_err)?;
            let infs = views
                .iter()
                .map(|v| v.inference_model())
                .collect::<anyhow::Result<Vec<_>>>()
                .map_err(to_py_err)?;
            Some((views, infs))
        } else {
            None
        };
        let single = if k_out == 1 {
            Some(self.inference_model().map_err(to_py_err)?)
        } else {
            None
        };
        let projected_features: Vec<usize> = match &single {
            Some(inference) => inference.canonical_features().to_vec(),
            None => (0..canonical_n_features).collect(),
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
            (None, Some((views, _))) => views
                .iter()
                .enumerate()
                .map(|(c, v)| {
                    prediction_work_per_row(&v.inner, self.inner.class_tree_count(upto, c))
                })
                .sum(),
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
                        let gpu_batch = known_gpu_selection.unwrap_or_else(|| {
                            gpu_prediction_selected(use_gpu, n_rows, n_features, work_per_row)
                        });
                        if let Some((views, infs)) = &multiclass {
                            let workers = resolved_predict_workers(
                                predict_threads,
                                predict_worker_cap,
                                n_rows,
                                work_per_row,
                            );
                            let mut margins = vec![0.0f32; n_rows * k_out];
                            for (c, (view, vinf)) in views.iter().zip(infs).enumerate() {
                                let nt = self.inner.class_tree_count(upto, c);
                                let gathered;
                                let vdata: &[f32] = if vinf.is_identity(canonical_n_features) {
                                    data
                                } else {
                                    gathered = project_canonical_rows(
                                        data,
                                        n_rows,
                                        canonical_n_features,
                                        vinf.canonical_features(),
                                    )?;
                                    &gathered
                                };
                                #[cfg(feature = "cuda")]
                                let col = if gpu_batch {
                                    view.gpu_predictor(0)?.predict_margins(vdata, n_rows, nt)?
                                } else {
                                    score_contiguous_rows(
                                        &*vinf.cpu_predictor()?,
                                        vdata,
                                        n_rows,
                                        vinf.n_features(),
                                        nt,
                                        workers,
                                        &mut predict_pool,
                                    )?
                                };
                                #[cfg(not(feature = "cuda"))]
                                let col = {
                                    let _ = (gpu_batch, view);
                                    score_contiguous_rows(
                                        &*vinf.cpu_predictor()?,
                                        vdata,
                                        n_rows,
                                        vinf.n_features(),
                                        nt,
                                        workers,
                                        &mut predict_pool,
                                    )?
                                };
                                for (i, v) in col.into_iter().enumerate() {
                                    margins[i * k_out + c] = v;
                                }
                            }
                            if output_margin {
                                out.extend(margins);
                            } else if let crate::types::Objective::Softmax { .. } =
                                self.inner.objective
                            {
                                out.extend(
                                    margins
                                        .chunks_exact(k_out)
                                        .map(|r| crate::objective::argmax(r) as f32),
                                );
                            } else {
                                margins
                                    .par_chunks_exact_mut(k_out)
                                    .for_each(crate::objective::softmax_in_place);
                                out.extend(margins);
                            }
                            return Ok(());
                        }
                        let inference = single.as_ref().expect("单输出 inference");
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
                            let mut batch = predictor.predict_margins(data, n_rows, upto)?;
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
                        let mut batch = score_contiguous_rows(
                            &predictor,
                            data,
                            n_rows,
                            n_features,
                            upto,
                            workers,
                            &mut predict_pool,
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

    /// Stream file prediction directly to one Parquet file. CPU scoring uses
    /// deterministic row-parallel workers; one ordered writer thread overlaps
    /// Parquet encoding/output with the next ingest/score batch.
    #[pyo3(signature = (data, output_path, output_margin=false, n_trees=None, header=true, *,
                        predict_threads=0, prediction_column="prediction",
                        iteration_range=None, missing=f64::NAN))]
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
                    );
                }
            }
        }
        guard(|| {
            let path_specs = path_specs_of(data)?.ok_or_else(|| {
                PyValueError::new_err(
                    "predict_to_parquet 只接受文件路径、路径列表、目录或 glob 输入",
                )
            })?;
            if self.inner.num_outputs() > 1 {
                return self.multiclass_to_parquet(
                    py,
                    data,
                    output_path,
                    output_margin,
                    n_trees,
                    header,
                    predict_threads,
                    prediction_column,
                    missing,
                );
            }
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
                    .map_err(|e| ModelIoError::Runtime(format!("导出失败:{e}")))?;
                parallel_model_write(path, json.as_bytes(), requested, quiet)
                    .map_err(|e| ModelIoError::Value(format!("写不了 {path}:{e}")))
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
                            ModelIoError::FileNotFound(format!("{path} 不存在"))
                        } else {
                            ModelIoError::Value(format!("读不了 {path}:{e}"))
                        }
                    })?;
                    let raw = String::from_utf8(raw)
                        .map_err(|e| ModelIoError::Value(format!("{path} 不是 UTF-8 JSON:{e}")))?;
                    Model::from_xgboost_json_with_threads(&raw, model_io_threads)
                        .map_err(|e| ModelIoError::Value(format!("{path} 解析失败:{e}")))
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
                class_views: Mutex::new(None),
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

    /// XGBoost objective 名字(如 `binary:logistic`)。
    #[getter]
    fn objective(&self) -> &'static str {
        self.inner.objective.name()
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
            "<ferrisboost.Model {} 棵树,{} 个特征{}>",
            self.inner.trees.len(),
            self.inner.n_features,
            match self.inner.best_iteration {
                Some(b) => format!(",best_iteration={b}"),
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
                        "monotone_constraints 有 {} 个,但只有 {n_features} 个特征",
                        v.len()
                    )));
                }
                v.clone()
            }
            Some(MonotoneSpec::Named(pairs)) => {
                if feature_names.is_empty() {
                    return Err(PyValueError::new_err(
                        "按列名给的 monotone_constraints 需要列名:这个输入是位置模式\
                         (numpy 或无表头 CSV),请改用按位置的元组",
                    ));
                }
                let mut out = vec![0i8; feature_names.len()];
                for (name, dir) in pairs {
                    let at = feature_names.iter().position(|n| n == name).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "monotone_constraints 里的列 {name:?} 不是特征列;特征列有 {feature_names:?}"
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

fn parse_params(d: &Bound<'_, PyDict>) -> PyResult<Parsed> {
    // 写成宏而不是泛型函数:pyo3 0.29 的 FromPyObject 带两个生命周期,
    // 泛型 helper 要把它们全串起来,不如就地展开清楚。
    macro_rules! get {
        ($key:expr) => {
            d.get_item($key)?
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "参数 {} 没给(Python 层应该补上默认值)",
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
    let quantile_alpha: Option<f32> = get!("quantile_alpha");
    if objective == "reg:quantileerror" && quantile_alpha.is_none() {
        return Err(PyValueError::new_err(
            "reg:quantileerror 需要 quantile_alpha(单个 [0, 1] 内的分位数,例如 0.9);\
             XGBoost 没有默认值",
        ));
    }
    let objective = Objective::from_name(&objective)
        .filter(|_| objective != "reg:linear")
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "objective {objective:?} 不支持;支持的有 {}",
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
        .with_quantile_alpha(quantile_alpha.unwrap_or(0.5));
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
                Some(d) if d.starts_with("cuda:") => Some(
                    d[5..]
                        .parse::<usize>()
                        .map_err(|_| PyValueError::new_err(format!("device 解析不了:{d:?}")))?,
                ),
                Some(other) => {
                    return Err(PyValueError::new_err(format!(
                        "device 只支持 \"cpu\" / \"cuda\" / \"cuda:N\",收到 {other:?}"
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
    // The device split kernels only know ordered bins, so categorical training
    // keeps the exact host split search and allocates no resident split cache.
    // Plan memory with the same decision the runtime makes.
    if parsed.categorical.iter().any(|&c| c) {
        parsed.params.gpu_split_cache = false;
    }
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
        class_views: Mutex::new(None),
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
                    "predict 只接受文件路径(单个 / 列表 / 目录 / glob)、二维 f32 numpy 数组、\
                     pandas DataFrame 或 pyarrow Table",
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
    fb.getattr("_as_predict_matrix")?.call1((data, names.to_vec()))
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
            "{what}第 {} 行第 {} 列是 {}:特征不能是 ±inf 或保留的 f32::MAX(缺失值请用 NaN;极端值请先缩放)",
            i / n_features.max(1),
            i % n_features.max(1),
            data[i]
        ))),
    }
}

/// Training categorical columns must hold integer codes `0..=253` (NaN =
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
            "feature_types 有 {} 项,数据有 {n_features} 个特征",
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
                    verbose_eval=None, callbacks=None, weight=None, feature_names=None))]
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
) -> PyResult<PyModel> {
    let overall_started = std::time::Instant::now();
    let mut cfg = parse_params(params)?;
    let weights: Option<Vec<f32>> = weight
        .map(|w| w.as_slice().map(<[f32]>::to_vec))
        .transpose()?;
    cfg.weighted = weights.is_some();
    let shape = x.shape();
    let (n_rows, n_features) = (shape[0], shape[1]);
    let labels = y.as_slice()?.to_vec();
    if labels.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "标签有 {} 个,特征矩阵有 {n_rows} 行",
            labels.len()
        )));
    }

    // 量化在 GIL 里做:它要读 numpy 的内存。之后训练不再碰 Python 对象,
    // 就可以放掉 GIL 了。
    // 先定 blocking,再建 source。策略只在这里决定一次,后面的训练只消费
    // 已经定好的分块。
    // ⚠️ **GPU 和 CPU 用不同的块宽解析器。** GPU streaming 的启发式是拿
    // GPU 墙钟测出来的(块数拐点),套到 CPU 上没有证据支持;反过来也一样。
    // 这里是 Python 侧唯一一次决定物理块宽的地方。
    let plan = resolve_blocking(&cfg, n_features, Some(n_rows as u64))?;
    cfg.params.cols_per_block = plan.effective;
    // 规划器解出来的几何是权威的(见 `BlockingPlan`)。
    apply_resolved_geometry(&mut cfg, &plan);
    if let Some(w) = &weights {
        if w.len() != n_rows {
            return Err(PyValueError::new_err(format!(
                "weight 有 {} 个,特征矩阵有 {n_rows} 行",
                w.len()
            )));
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(PyValueError::new_err("样本权重必须是有限的非负数"));
        }
    }
    // The `missing` sentinel becomes NaN before any check or binning (borrowed
    // when no value matches; otherwise one corrected copy, never the caller's
    // array). Categorical codes are subject to it too, as in XGBoost.
    let x_values = crate::source::missing_as_nan(x.as_slice()?, cfg.missing);
    reject_inf(&x_values, n_features, "训练集")?;
    check_categorical_codes(&x_values, n_rows, n_features, &cfg.categorical)?;
    // 分箱边界按样本权重加权(XGBoost hist 同样如此);全 1 / None 逐位相同。
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
        weights.as_deref(),
    );

    // 验证集必须复用训练集的 cuts,否则同一个值会落进不同的 bin
    #[allow(clippy::type_complexity)]
    let mut eval_data: Vec<(String, DenseSource, Vec<f32>, Option<Vec<f32>>)> = Vec::new();
    if let Some(list) = evals {
        for item in list.iter() {
            // Python 层统一成 (data, label, weight-or-None, name)。
            #[allow(clippy::type_complexity)]
            let (ex, ey, ew, name): (
                PyReadonlyArray2<f32>,
                PyReadonlyArray1<f32>,
                Option<PyReadonlyArray1<f32>>,
                String,
            ) = item.extract()?;
            let eshape = ex.shape();
            if eshape[1] != n_features {
                return Err(PyValueError::new_err(format!(
                    "验证集 {name} 有 {} 列,训练集是 {n_features} 列",
                    eshape[1]
                )));
            }
            let ex_values = crate::source::missing_as_nan(ex.as_slice()?, cfg.missing);
            reject_inf(&ex_values, n_features, &format!("验证集 {name} "))?;
            let src = DenseSource::new(
                &ex_values,
                eshape[0],
                n_features,
                BinningStrategy::Provided(train_src.cuts()),
                cfg.params.cols_per_block,
            );
            let ew = ew.map(|w| w.as_slice().map(<[f32]>::to_vec)).transpose()?;
            eval_data.push((name, src, ey.as_slice()?.to_vec(), ew));
        }
    }

    let device = cfg.device;
    // numpy 入口没有列名:只接受按位置的约束(Python 层已把 pyarrow 列名解析掉)。
    cfg.resolve_monotone(n_features, &[])?;
    let (train_cfg, lower_is_better) = cfg.train_config()?;
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
        .map(|(name, src, labels, w)| EvalSet {
            name: name.clone(),
            source: src,
            labels,
            weights: w.as_deref(),
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
            Some(_) => Err(anyhow::anyhow!("这个构建没有 CUDA 支持")),
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
                    "feature_names 有 {} 个,数据有 {n_features} 列",
                    names.len()
                )));
            }
            let mut seen = std::collections::HashSet::new();
            if let Some(dup) = names.iter().find(|n| !seen.insert(n.as_str())) {
                return Err(PyValueError::new_err(format!("特征列名重复:{dup:?}")));
            }
            names
        }
    };
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
                    verbose_eval=None, callbacks=None, weight=None))]
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
) -> PyResult<PyModel> {
    let overall_started = std::time::Instant::now();
    let mut cfg = parse_params(params)?;
    cfg.weighted = weight.is_some();
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
        "csv" => Some(
            csv_source::infer_schema(&paths[0], header)
                .map_err(|e| PyValueError::new_err(format!("读 {path} 的 schema 失败:{e}")))?,
        ),
        _ => None,
    };
    let label_owned = match (fmt, header) {
        ("parquet", false) => {
            return Err(PyValueError::new_err("header=False 只适用于 CSV"));
        }
        (_, true) => label
            .extract::<String>()
            .map_err(|_| PyValueError::new_err("header=True 时 label 必须是列名字符串"))?,
        ("csv", false) => {
            let index = label.extract::<isize>().map_err(|_| {
                PyValueError::new_err("header=False 时 label 必须是从 0 开始的列下标")
            })?;
            if index < 0 {
                return Err(PyValueError::new_err(format!(
                    "label 列下标不能是负数,给的是 {index}"
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
        Some(w) if header => Some(
            w.extract::<String>()
                .map_err(|_| PyValueError::new_err("header=True 时 weight 必须是列名字符串"))?,
        ),
        Some(w) => {
            let index = w.extract::<isize>().map_err(|_| {
                PyValueError::new_err("header=False 时 weight 必须是从 0 开始的列下标")
            })?;
            if index < 0 {
                return Err(PyValueError::new_err(format!(
                    "weight 列下标不能是负数,给的是 {index}"
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
    let targets = crate::source::Targets::new(label, weight_owned.as_deref())
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
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
        other => Err(anyhow::anyhow!("不认识的格式 {other:?}")),
    }
    .map_err(|e| PyValueError::new_err(format!("读 {path} 的 schema 失败:{e}")))?;
    // Declared feature types are positional over the resolved features (label
    // and weight excluded), like XGBoost's `feature_types`.
    if !cfg.categorical.is_empty() && cfg.categorical.len() != n_features {
        return Err(PyValueError::new_err(format!(
            "feature_types 有 {} 项,文件有 {n_features} 个特征列(不含标签/权重列)",
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
                    PyValueError::new_err(format!("读 {} 的行数失败:{e}", p.display()))
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
                    "GPU 输入在 open 前拿不到行数时需要显式 cols_per_block；open 后会用真实行数完成显存、stream 和 residency 规划",
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
            other => Err(anyhow::anyhow!("不认识的格式 {other:?}")),
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
            "验证集格式({})和训练集({})不一致",
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
        let (train_cfg, lower_is_better) = cfg.train_config()?;
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
        ingest_reporter.complete(src.n_rows());
        let mut eval_data = Vec::new();
        if let Some(list) = evals {
            for item in list.iter() {
                let (p, name): (String, String) = item.extract()?;
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
                eval_data.push((name, d, ew));
            }
        }
        let evals_ref: Vec<EvalSet> = eval_data
            .iter()
            .map(|(name, (s, l), w)| EvalSet {
                name: name.clone(),
                source: s,
                labels: l,
                weights: w.as_deref(),
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
                Some(_) => Err(anyhow::anyhow!("这个构建没有 CUDA 支持")),
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
    cfg.resolve_monotone(n_features, &ds.feature_names)?;
    let (train_cfg, lower_is_better) = cfg.train_config()?;
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
    if let Some(list) = evals {
        for item in list.iter() {
            let (p, name): (String, String) = item.extract()?;
            let d = open_spec(&p, &ds.feature_names, Some(ds.source.cuts().clone()))
                .map_err(to_py_err)?;
            eval_data.push((name, d));
        }
    }
    let evals_ref: Vec<EvalSet> = eval_data
        .iter()
        .map(|(name, d)| EvalSet {
            name: name.clone(),
            source: &d.source,
            labels: &d.labels,
            weights: d.weights.as_deref(),
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
                ds.weights.as_deref(),
                &evals_ref,
                &train_cfg,
                &mut cbs,
                &Local,
            ),
            #[cfg(not(feature = "cuda"))]
            Some(_) => Err(anyhow::anyhow!("这个构建没有 CUDA 支持")),
            None => train_weighted(
                &ds.source,
                &ds.labels,
                ds.weights.as_deref(),
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
    Ok(finish(
        model,
        handle,
        ds.feature_names.clone(),
        cfg.quiet,
        cfg.model_io_threads,
        cfg.ingest_threads,
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(train_dense, m)?)?;
    m.add_function(wrap_pyfunction!(train_file, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
