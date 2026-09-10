//! Python 绑定。
//!
//! **这一层不做任何数值处理。** 参数解析、dtype 归一化、输入类型分发
//! 都在 `python/ferrisboost/__init__.py` 里;这里只负责把已经规整好的
//! 数据交给训练循环。预测是例外:它是公开的 PyO3 边界，必须正确解释
//! NumPy 的 stride，不能把 F-order/切片视为 C row-major 内存。

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
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
use crate::train::{train as rust_train, BlockSource, EvalSet, TrainConfig};
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
}

impl Callback for PyCallback {
    fn after_iteration(&mut self, m: &RoundMetrics) -> bool {
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
                    e.restore(py);
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
    requested: Option<usize>,
    effective: usize,
    n_blocks: usize,
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
            let ctx = cudarc::driver::CudaContext::new(ordinal)
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
                    StreamingMemModel::new(
                        rows,
                        max_bin,
                        streams,
                        16,
                        true,
                        fast,
                        true,
                        32 * 1024 * 1024,
                    )
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
        // One decoded Arrow batch plus conversion/quantization scratch.
        n_features
            .saturating_mul(8192)
            .saturating_mul(8)
            .saturating_add(16 * 1024 * 1024)
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
    let file = std::fs::File::create(path)?;
    file.set_len(bytes.len() as u64)?;
    let reporter = ModelIoReporter::start(quiet, "SAVE", path, bytes.len(), requested, workers);
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
    file.sync_all()?;
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
        match model.objective {
            Objective::SquaredError => margin,
            Objective::Logistic => 1.0 / (1.0 + (-margin).exp()),
        }
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
        crate::source::predict_input::for_each_projected_batch(
            &resolved,
            &names,
            canonical_n_features,
            &projected_features,
            header,
            ingest_workers,
            |rows| {
                let workers = resolved_predict_workers(
                    predict_threads,
                    predict_worker_cap,
                    rows.len(),
                    work_per_row,
                );
                let batch = score_rows(rows.len(), workers, &mut predict_pool, |row| {
                    predict_value(inner, &rows[row], upto, output_margin)
                })?;
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
    if !output_margin && model.objective == Objective::Logistic {
        for value in values {
            *value = 1.0 / (1.0 + (-*value).exp());
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
}

impl PyModel {
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
                        predict_threads=0, use_gpu=false))]
    /// 预测。`data` 可以是:
    ///
    /// * 文件路径 / 路径列表 / 目录 / glob —— 走 Rust 的文件路径,
    ///   按模型 schema 绑列(**产品的主路径**);
    /// * 二维 f32 numpy 数组 —— 仍然支持,小数据方便。
    ///
    /// 两条都返回一维 `numpy.ndarray`。
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        use_gpu: bool,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        ensure_gpu_prediction_available(use_gpu)?;
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
            );
        }
        let x: PyReadonlyArray2<'py, f32> = data.extract().map_err(|_| {
            PyValueError::new_err(
                "predict 只接受文件路径(单个 / 列表 / 目录 / glob)或二维 f32 numpy 数组",
            )
        })?;
        self.predict_array(py, x, output_margin, n_trees, predict_threads, use_gpu)
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
            let shape = x.shape();
            let (n_rows, n_features) = (shape[0], shape[1]);
            if n_features != self.inner.n_features {
                return Err(PyValueError::new_err(format!(
                    "模型是按 {} 个特征训的,传进来的是 {n_features} 列",
                    self.inner.n_features
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
                        predict_threads=0, use_gpu=false))]
    fn predict_files<'py>(
        &self,
        py: Python<'py>,
        path_specs: Vec<String>,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
        use_gpu: bool,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        use crate::tree::SchemaMode;
        ensure_gpu_prediction_available(use_gpu)?;
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
        let inference = self.inference_model().map_err(to_py_err)?;
        let projected_features = inference.canonical_features().to_vec();
        let n_features = inference.n_features();
        let upto = n_trees.unwrap_or(self.inner.trees.len());
        let inner = inference.model();
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
        let work_per_row = prediction_work_per_row(inner, upto);
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
                crate::source::predict_input::for_each_projected_batch(
                    &resolved,
                    &names,
                    canonical_n_features,
                    &projected_features,
                    header,
                    ingest_workers,
                    |rows| {
                        let gpu_batch = known_gpu_selection.unwrap_or_else(|| {
                            gpu_prediction_selected(use_gpu, rows.len(), n_features, work_per_row)
                        });
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
                            let mut data = Vec::with_capacity(rows.len() * n_features);
                            for row in rows {
                                data.extend_from_slice(row);
                            }
                            let mut batch = predictor.predict_margins(&data, rows.len(), upto)?;
                            transform_prediction_margins(inner, &mut batch, output_margin);
                            out.extend(batch);
                            return Ok(());
                        }
                        let _ = gpu_batch;
                        let workers = resolved_predict_workers(
                            predict_threads,
                            predict_worker_cap,
                            rows.len(),
                            work_per_row,
                        );
                        let batch = score_rows(rows.len(), workers, &mut predict_pool, |row| {
                            predict_value(inner, &rows[row], upto, output_margin)
                        })?;
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
        Ok(PyArray1::from_vec(py, out))
    }

    /// Stream file prediction directly to one Parquet file. CPU scoring uses
    /// deterministic row-parallel workers; one ordered writer thread overlaps
    /// Parquet encoding/output with the next ingest/score batch.
    #[pyo3(signature = (data, output_path, output_margin=false, n_trees=None, header=true, *,
                        predict_threads=0, prediction_column="prediction"))]
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
    ) -> PyResult<usize> {
        guard(|| {
            let path_specs = path_specs_of(data)?.ok_or_else(|| {
                PyValueError::new_err(
                    "predict_to_parquet 只接受文件路径、路径列表、目录或 glob 输入",
                )
            })?;
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
            )
        })
    }

    #[pyo3(signature = (path, quiet=None, model_io_threads=None))]
    fn save_model(
        &self,
        path: &str,
        quiet: Option<bool>,
        model_io_threads: Option<usize>,
    ) -> PyResult<()> {
        let quiet = quiet.unwrap_or(self.quiet);
        let requested = model_io_threads.unwrap_or(self.model_io_threads);
        guard(|| {
            let json = self
                .inner
                .to_xgboost_json_with_threads(requested)
                .map_err(|e| PyRuntimeError::new_err(format!("导出失败:{e}")))?;
            parallel_model_write(path, json.as_bytes(), requested, quiet)
                .map_err(|e| PyValueError::new_err(format!("写不了 {path}:{e}")))
        })
    }

    /// 读一个 XGBoost 格式的模型(它自己训的也行)。
    #[staticmethod]
    #[pyo3(signature = (path, quiet=false, model_io_threads=0, ingest_threads=0))]
    fn load_model(
        path: &str,
        quiet: bool,
        model_io_threads: usize,
        ingest_threads: usize,
    ) -> PyResult<Self> {
        guard(|| {
            let raw = parallel_model_read(path, model_io_threads, quiet).map_err(|e| {
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
                {
                    pyo3::exceptions::PyFileNotFoundError::new_err(format!("{path} 不存在"))
                } else {
                    PyValueError::new_err(format!("读不了 {path}:{e}"))
                }
            })?;
            let raw = String::from_utf8(raw)
                .map_err(|e| PyValueError::new_err(format!("{path} 不是 UTF-8 JSON:{e}")))?;
            let inner = Model::from_xgboost_json_with_threads(&raw, model_io_threads)
                .map_err(|e| PyValueError::new_err(format!("{path} 解析失败:{e}")))?;
            Ok(Self {
                inner,
                history: Vec::new(),
                quiet,
                model_io_threads,
                ingest_threads,
                cpu_predictor: Mutex::new(None),
                #[cfg(feature = "cuda")]
                gpu_predictors: Mutex::new(std::collections::HashMap::new()),
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
    base_score: f32,
    cache_budget_bytes: Option<usize>,
    cache_path: Option<String>,
    /// File ingest concurrency, independent of training `nthread`.
    ingest_threads: usize,
    model_io_threads: usize,
    quiet: bool,
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
    let objective = match objective.as_str() {
        "binary:logistic" => Objective::Logistic,
        "reg:squarederror" => Objective::SquaredError,
        other => {
            return Err(PyValueError::new_err(format!(
                "objective {other:?} 不支持,现在只有 binary:logistic 和 reg:squarederror"
            )))
        }
    };
    Ok(Parsed {
        params: TrainParams {
            n_rounds: get!("num_boost_round"),
            max_depth: get!("max_depth"),
            max_bin: get!("max_bin"),
            learning_rate: get!("eta"),
            lambda: get!("lambda"),
            gamma: get!("gamma"),
            min_child_weight: get!("min_child_weight"),
            subsample: get!("subsample"),
            colsample_bytree: get!("colsample_bytree"),
            // 占位:真正的值要等拿到 schema 才能定,见 BlockingPlan。
            cols_per_block: 0,
            nthread: get!("nthread"),
            // 和 cols_per_block 一样保留 Option 语义:Python 层给 None 就是
            // 「让 planner 按模式选」,给 1/2 就完全按用户的来。
            hist_streams: get!("hist_streams"),
            resident_blocks: get!("resident_blocks"),
            hist_nodes_per_batch: get!("hist_nodes_per_batch"),
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
        },
        requested_cols_per_block: get!("cols_per_block"),
        device: {
            // `device="cuda"` / `"cuda:0"` / `"gpu"` → GPU;不给或 "cpu" → CPU。
            let raw: Option<String> = get!("device");
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
    })
}

/// 组装 callbacks:早停和 verbose 在 Rust 侧,用户的函数包一层。
fn build_callbacks(
    early_stopping_rounds: Option<usize>,
    verbose_eval: Option<usize>,
    user: Option<&Bound<'_, PyList>>,
) -> PyResult<(Vec<Box<dyn Callback>>, Arc<Mutex<Vec<RoundMetrics>>>)> {
    let recorder = RecordHistory::new();
    let handle = recorder.handle();
    let mut cbs: Vec<Box<dyn Callback>> = Vec::new();
    if let Some(rounds) = early_stopping_rounds {
        // 目前两个 metric 都是越小越好
        cbs.push(Box::new(EarlyStopping::new(rounds, true)));
    }
    if let Some(period) = verbose_eval {
        if period > 0 {
            cbs.push(Box::new(VerboseEval { period }));
        }
    }
    if let Some(list) = user {
        for f in list.iter() {
            cbs.push(Box::new(PyCallback { func: f.unbind() }));
        }
    }
    cbs.push(Box::new(recorder));
    Ok((cbs, handle))
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
    }
}

/// numpy 入口。`x` 是行主序的 f32 矩阵(Python 层负责转 dtype)。
#[pyfunction]
#[pyo3(signature = (x, y, params, evals=None, early_stopping_rounds=None,
                    verbose_eval=None, callbacks=None))]
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
) -> PyResult<PyModel> {
    let overall_started = std::time::Instant::now();
    let mut cfg = parse_params(params)?;
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
    let train_src = DenseSource::new_with_nthread(
        x.as_slice()?,
        n_rows,
        n_features,
        BinningStrategy::Sketch {
            max_bins: cfg.params.max_bin as usize,
        },
        plan.effective,
        cfg.params.nthread,
    );

    // 验证集必须复用训练集的 cuts,否则同一个值会落进不同的 bin
    let mut eval_data: Vec<(String, DenseSource, Vec<f32>)> = Vec::new();
    if let Some(list) = evals {
        for item in list.iter() {
            let (ex, ey, name): (PyReadonlyArray2<f32>, PyReadonlyArray1<f32>, String) =
                item.extract()?;
            let eshape = ex.shape();
            if eshape[1] != n_features {
                return Err(PyValueError::new_err(format!(
                    "验证集 {name} 有 {} 列,训练集是 {n_features} 列",
                    eshape[1]
                )));
            }
            let src = DenseSource::new(
                ex.as_slice()?,
                eshape[0],
                n_features,
                BinningStrategy::Provided(train_src.cuts()),
                cfg.params.cols_per_block,
            );
            eval_data.push((name, src, ey.as_slice()?.to_vec()));
        }
    }

    let (mut cbs, handle) = build_callbacks(early_stopping_rounds, verbose_eval, callbacks)?;
    // `cfg.params` 下一行就被移进 TrainConfig,所以先把 device 取出来。
    let device = cfg.device;
    let train_cfg = TrainConfig {
        base_score: cfg.base_score,
        quiet: cfg.quiet,
        ..TrainConfig::new(cfg.params, cfg.objective)
    };
    if !cfg.quiet && device.is_none() {
        eprintln!(
            "CPU_THREADS requested={} resolved={}",
            train_cfg.params.nthread,
            effective_nthread(train_cfg.params.nthread)
        );
    }
    if !cfg.quiet {
        eprintln!(
            "SETUP input=dense rows={n_rows} features={n_features} seconds={:.3}",
            overall_started.elapsed().as_secs_f64()
        );
    }

    let evals_ref: Vec<EvalSet> = eval_data
        .iter()
        .map(|(name, src, labels)| EvalSet {
            name: name.clone(),
            source: src,
            labels,
        })
        .collect();

    // 训练期间放掉 GIL:不然调用方的其它线程全被卡住。
    // 用户回调进 Python 时会自己重新拿(见 PyCallback)。
    let train_started = std::time::Instant::now();
    let model = py
        .detach(|| match device {
            #[cfg(feature = "cuda")]
            Some(ord) => crate::train::train_gpu(
                ord, &train_src, &labels, &evals_ref, &train_cfg, &mut cbs, &Local,
            ),
            #[cfg(not(feature = "cuda"))]
            Some(_) => Err(anyhow::anyhow!("这个构建没有 CUDA 支持")),
            None => rust_train(
                &train_src, &labels, &evals_ref, &train_cfg, &mut cbs, &Local,
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
    if PyErr::occurred(py) {
        return Err(PyErr::fetch(py));
    }
    // numpy 入口没有列名 —— 位置模式。
    Ok(finish(
        model,
        handle,
        Vec::new(),
        cfg.quiet,
        cfg.model_io_threads,
        cfg.ingest_threads,
    ))
}

/// 文件入口。**格式由 Rust 从解析出的文件判定**,不再由 Python 按后缀猜。
#[pyfunction]
#[pyo3(signature = (path_specs, label, params, header=true, evals=None, early_stopping_rounds=None,
                    verbose_eval=None, callbacks=None))]
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
) -> PyResult<PyModel> {
    let overall_started = std::time::Instant::now();
    let mut cfg = parse_params(params)?;
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
    // **peek schema → 定 blocking → 才真正 open**。Parquet / CSV 在 open
    // 之前不知道特征数,而自动选块宽需要它;单独 peek 一次(只读 footer /
    // 只推断 schema,不解压数据)就能把策略决定留在这一层,
    // 不必把 `nthread` 传进底层的 `partition_features`。
    let n_features = match fmt {
        "parquet" => parquet_source::peek_n_features(path, label),
        "csv" => csv_source::n_features_from_schema(
            csv_preflight.as_deref().expect("CSV preflight exists"),
            label,
        ),
        other => Err(anyhow::anyhow!("不认识的格式 {other:?}")),
    }
    .map_err(|e| PyValueError::new_err(format!("读 {path} 的 schema 失败:{e}")))?;
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
    let open_paths = |ps: &[std::path::PathBuf],
                      cuts: Option<crate::columns::BinCuts>,
                      first_csv_schema: Option<arrow::datatypes::SchemaRef>|
     -> anyhow::Result<crate::source::parquet_source::Dataset> {
        match fmt {
            "parquet" => parquet_source::open_many_with_nthread(
                ps,
                label,
                cuts,
                max_bin,
                cols_per_block,
                ingest_threads,
            ),
            "csv" => match first_csv_schema {
                Some(schema) => csv_source::open_many_with_header_and_nthread_prepared(
                    ps,
                    label,
                    cuts,
                    max_bin,
                    cols_per_block,
                    header,
                    ingest_threads,
                    schema,
                ),
                None => csv_source::open_many_with_header_and_nthread(
                    ps,
                    label,
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
    let open_spec = |spec: &str,
                     cuts: Option<crate::columns::BinCuts>|
     -> anyhow::Result<crate::source::parquet_source::Dataset> {
        let r = crate::source::input::resolve(&[spec.to_string()])?;
        anyhow::ensure!(
            r.format == resolved.format,
            "验证集格式({})和训练集({})不一致",
            r.format.label(),
            resolved.format.label()
        );
        open_paths(&r.paths, cuts, None)
    };

    let (mut cbs, handle) = build_callbacks(early_stopping_rounds, verbose_eval, callbacks)?;
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
        let train_cfg = TrainConfig {
            base_score: cfg.base_score,
            quiet: cfg.quiet,
            ..TrainConfig::new(cfg.params.clone(), cfg.objective)
        };
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
        let (src, labels) = ParquetBlockSource::open_with_cache_and_nthread(
            path,
            label,
            cfg.params.max_bin,
            cfg.params.cols_per_block,
            cache_options,
            ingest_threads,
        )
        .map_err(to_py_err)?;
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
                    label,
                    src.cuts().clone(),
                    cfg.params.cols_per_block,
                    QuantizedCacheOptions {
                        budget_bytes: cfg.cache_budget_bytes,
                        path: None,
                    },
                    ingest_threads,
                )
                .map_err(to_py_err)?;
                eval_data.push((name, d));
            }
        }
        let evals_ref: Vec<EvalSet> = eval_data
            .iter()
            .map(|(name, (s, l))| EvalSet {
                name: name.clone(),
                source: s,
                labels: l,
            })
            .collect();
        if !cfg.quiet {
            eprintln!(
                "SETUP input=parquet files=1 rows={} features={n_features} seconds={:.3}",
                src.n_rows(),
                overall_started.elapsed().as_secs_f64()
            );
        }
        let train_started = std::time::Instant::now();
        let model = py
            .detach(|| match device {
                #[cfg(feature = "cuda")]
                Some(ord) => crate::train::train_gpu(
                    ord, &src, &labels, &evals_ref, &train_cfg, &mut cbs, &Local,
                ),
                #[cfg(not(feature = "cuda"))]
                Some(_) => Err(anyhow::anyhow!("这个构建没有 CUDA 支持")),
                None => rust_train(&src, &labels, &evals_ref, &train_cfg, &mut cbs, &Local),
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
        if PyErr::occurred(py) {
            return Err(PyErr::fetch(py));
        }
        // 命名模式:列名取自 parquet schema,标签列已排除。
        let names = parquet_source::feature_names_of(&paths[0], label).map_err(to_py_err)?;
        return Ok(finish(
            model,
            handle,
            names,
            cfg.quiet,
            cfg.model_io_threads,
            cfg.ingest_threads,
        ));
    }

    let ds = open_paths(&paths, None, csv_preflight.clone()).map_err(to_py_err)?;
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
    let train_cfg = TrainConfig {
        base_score: cfg.base_score,
        quiet: cfg.quiet,
        ..TrainConfig::new(cfg.params.clone(), cfg.objective)
    };
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
            let d = open_spec(&p, Some(ds.source.cuts().clone())).map_err(to_py_err)?;
            eval_data.push((name, d));
        }
    }
    let evals_ref: Vec<EvalSet> = eval_data
        .iter()
        .map(|(name, d)| EvalSet {
            name: name.clone(),
            source: &d.source,
            labels: &d.labels,
        })
        .collect();

    if !cfg.quiet {
        eprintln!(
            "SETUP input={fmt} files={} rows={} features={n_features} seconds={:.3}",
            paths.len(),
            ds.source.n_rows(),
            overall_started.elapsed().as_secs_f64()
        );
    }
    let train_started = std::time::Instant::now();

    let model = py
        .detach(|| match device {
            #[cfg(feature = "cuda")]
            Some(ord) => crate::train::train_gpu(
                ord, &ds.source, &ds.labels, &evals_ref, &train_cfg, &mut cbs, &Local,
            ),
            #[cfg(not(feature = "cuda"))]
            Some(_) => Err(anyhow::anyhow!("这个构建没有 CUDA 支持")),
            None => rust_train(
                &ds.source, &ds.labels, &evals_ref, &train_cfg, &mut cbs, &Local,
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
    if PyErr::occurred(py) {
        return Err(PyErr::fetch(py));
    }
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
