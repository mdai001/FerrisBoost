//! 数据源 —— 这个项目最实际的卖点所在。
//!
//! 目标:S3 上的 Parquet 直接进训练,不落地、不转 libsvm、
//! 内存不翻倍。现在的常见流程是 S3 -> pandas -> numpy -> DMatrix,
//! 中间内存翻两三倍,几百 GB 的数据光这步就装不下。

pub mod arrow_source;
pub mod csv_source;
pub mod dense;
pub mod input;
pub mod openprof;
pub(crate) mod parallel_reader;
pub mod parquet_source;
pub mod predict_input;

pub use dense::DenseSource;

/// Public file-ingest concurrency default: zero means system-adaptive auto.
/// It remains independent of training `nthread`.
pub const DEFAULT_INGEST_THREADS: usize = 0;

/// Divide one resolved concurrency budget between an upstream decoder and a
/// downstream column transform. Their worker pools run concurrently.
pub(crate) fn pipeline_threads(total: usize) -> (usize, usize) {
    let total = if total == 0 {
        crate::threading::effective_threads(0)
    } else {
        total
    };
    match total.max(1) {
        1 => (1, 1),
        n => (n.div_ceil(2), n / 2),
    }
}

// 设计要点:
//
// 1. Arrow RecordBatch 是唯一输入接口。这样 Parquet、S3、DuckDB、
//    Polars、任何数据库的 ADBC 驱动全部自动支持,零拷贝。
//
// 2. Parquet 本来就是列存,按列取一组 column chunk 正是它的最优
//    访问模式 —— S3 range read 直接拿需要的字节。行分批要读全部列
//    再丢弃,IO 放大好几倍。列分块在这里不只是「能」,是「更快」。
//
// 3. Parquet footer 里有每个 column chunk 的 min/max 和 null count。
//    分箱的第一遍扫描可以用这些统计做初始估计,省掉一整轮 IO。
//    这个优化在行分块下拿不到。
