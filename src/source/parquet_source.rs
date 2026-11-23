//! 从 Parquet 读数据。
//!
//! Parquet 本来就是列存,按列取一组 column chunk 正是它的最优访问
//! 模式;行分批要读全部列再丢弃,IO 放大好几倍。列分块在这里不只是
//! 「能」,是「更快」。
//!
//! 这一版是**本地文件 + 全量量化驻留**:读进来量化成 u8 的 bin
//! (比原始 f32 小 4 倍)全驻内存。按需从对象存储流式加载列块
//! (S3 range read、footer 统计跳过第一遍扫描)还没做 —— 那需要
//! async + object_store,是下一步。

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReader,
    ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::ProjectionMask;
use parquet::file::reader::ChunkReader;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bins::{BinCode, BinData, BinSlice, BinWidth};
use crate::columns::{quantize_value, BinCuts, ColumnBlock};
use crate::source::arrow_source::ArrowSource;
use crate::source::Targets;
use crate::train::{BlockSource, BlockSourceExt};
use crate::types::FeatId;

pub(crate) const PARQUET_READ_WINDOW_BYTES: usize = crate::source::parquet_io::READ_WINDOW_BYTES;

/// 一份带标签的数据集:特征列进 `source`,标签列单独拎出来。
/// Parquet 和 CSV 共用这个形状。
///
/// 标签不进特征矩阵 —— 拿标签当特征训练,树会直接照着它分裂,
/// 训练集指标完美、线上全错。这里用列名显式排除掉。
pub struct Dataset {
    pub source: ArrowSource,
    pub labels: Vec<f32>,
    /// 样本权重列(给了 `weight` 才有),和 `labels` 同行序。
    pub weights: Option<Vec<f32>>,
    /// 特征列名,顺序和 `feat_id` 一致。导出模型时能填进 feature_names。
    pub feature_names: Vec<String>,
}

impl Dataset {
    /// 组装并做最后的校验。各个格式的读取器都走这条,免得校验各写一遍。
    pub(crate) fn assemble(
        source: ArrowSource,
        labels: Vec<f32>,
        feature_names: Vec<String>,
        label_name: &str,
        weights: Option<(Vec<f32>, &str)>,
    ) -> anyhow::Result<Self> {
        use crate::train::BlockSource as _;
        crate::ensure_input!(
            labels.len() == source.n_rows(),
            "labels have {} rows, features have {} rows",
            labels.len(),
            source.n_rows()
        );
        crate::ensure_input!(
            !labels.iter().any(|v| v.is_nan()),
            "label column {label_name:?} has missing values; handle missing labels upstream, \
             since filling them with 0 would train on 0 as a real label"
        );
        let weights = match weights {
            None => None,
            Some((w, name)) => {
                crate::ensure_input!(
                    w.len() == source.n_rows(),
                    "weights have {} rows, features have {} rows",
                    w.len(),
                    source.n_rows()
                );
                crate::source::check_weight_column(&w, name)?;
                Some(w)
            }
        };
        Ok(Self {
            source,
            labels,
            weights,
            feature_names,
        })
    }
}

fn batch_reader(
    path: &Path,
    columns: &[usize],
    batch_size: usize,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    let file =
        File::open(path).map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.display()))?;
    let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())?;
    batch_reader_with_metadata(file, metadata, columns, batch_size)
}

fn batch_reader_with_metadata(
    file: File,
    metadata: ArrowReaderMetadata,
    columns: &[usize],
    batch_size: usize,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    if coalesced_projection(&metadata, columns) {
        let input = crate::source::parquet_io::CoalescedFile::new(file)?;
        coalesced_row_group_reader(input, metadata, columns, batch_size)
    } else {
        let reader = build_projected_reader(file, metadata, columns, batch_size, None)?;
        Ok(Box::new(reader.map(|r| r.map_err(anyhow::Error::from))))
    }
}

/// An Arrow batch spanning small row groups reads column A from both groups,
/// then column B from both groups. That makes a single lookahead window bounce
/// between distant file regions for every column. Scope decoding to one group
/// at a time, retaining the same shared window and strictly increasing row order.
/// Sketch insertion is per value, not per batch; quantization uses row offsets.
fn coalesced_row_group_reader(
    input: crate::source::parquet_io::CoalescedFile,
    metadata: ArrowReaderMetadata,
    columns: &[usize],
    batch_size: usize,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    let n_groups = metadata.metadata().num_row_groups();
    anyhow::ensure!(n_groups > 0, "Parquet file has no row groups");
    let columns = columns.to_vec();
    let mut current =
        bounded_coalesced_group(input.clone(), metadata.clone(), &columns, batch_size, 0)?;
    let mut next_group = 1;
    let mut failed = false;
    Ok(Box::new(std::iter::from_fn(move || loop {
        if failed {
            return None;
        }
        if let Some(batch) = current.next() {
            failed = batch.is_err();
            return Some(batch);
        }
        if next_group == n_groups {
            return None;
        }
        match bounded_coalesced_group(
            input.clone(),
            metadata.clone(),
            &columns,
            batch_size,
            next_group,
        ) {
            Ok(reader) => {
                current = reader;
                next_group += 1;
            }
            Err(error) => {
                failed = true;
                return Some(Err(error));
            }
        }
    })))
}

fn coalesced_projection(metadata: &ArrowReaderMetadata, columns: &[usize]) -> bool {
    let fields = metadata.schema().fields().len();
    if metadata.metadata().num_row_groups() == 0
        || columns.len() < 256
        || columns.len().saturating_add(1) < fields
    {
        return false;
    }
    // Nested roots do not map one-to-one to physical chunks. Keep them, sparse
    // projections, and long multi-batch groups on the original File reader.
    let schema = metadata.metadata().file_metadata().schema_descr();
    if schema.num_columns() != fields
        || schema
            .root_schema()
            .get_fields()
            .iter()
            .any(|field| !field.is_primitive())
    {
        return false;
    }
    let mut selected = vec![false; fields];
    for &column in columns {
        let Some(slot) = selected.get_mut(column) else {
            return false;
        };
        *slot = true;
    }
    let selected_count = selected.iter().filter(|&&yes| yes).count();
    if selected_count < 256 || selected_count.saturating_add(1) < fields {
        return false;
    }
    for group in metadata.metadata().row_groups() {
        if group.num_rows() < 0 || group.num_rows() as u64 > (2 * BATCH_SIZE) as u64 {
            return false;
        }
        let mut all = 0u128;
        let mut projected = 0u128;
        for (i, column) in group.columns().iter().enumerate() {
            let bytes = column.compressed_size();
            if bytes < 0 {
                return false;
            }
            all += bytes as u128;
            if selected[i] {
                projected += bytes as u128;
            }
        }
        if all == 0 || projected * 100 < all * 95 {
            return false;
        }
    }
    true
}

/// 投影读取器:列**按 `columns` 给的顺序**返回。
///
/// ⚠️ Parquet 投影只决定读哪些列,返回顺序永远是文件里的物理顺序。
/// 下标不是升序时(按列名绑定的验证集、列序不同的文件)必须重排回来,
/// 否则每个特征会静默读到别的列。升序时 `order` 是 `None`,零开销。
pub(crate) struct ProjectedReader {
    inner: ParquetRecordBatchReader,
    order: Option<Vec<usize>>,
}

impl Iterator for ProjectedReader {
    type Item = Result<RecordBatch, arrow::error::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        Some(match &self.order {
            None => batch,
            Some(order) => batch.map(|b| crate::source::input::reorder(&b, order)),
        })
    }
}

fn build_projected_reader<T: ChunkReader + 'static>(
    input: T,
    metadata: ArrowReaderMetadata,
    columns: &[usize],
    batch_size: usize,
    row_group: Option<usize>,
) -> anyhow::Result<ProjectedReader> {
    let (mask_cols, order) = crate::source::input::projection_and_order(columns);
    let identity = order.iter().enumerate().all(|(i, &o)| i == o) && order.len() == mask_cols.len();
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(input, metadata);
    let mask = ProjectionMask::roots(builder.parquet_schema(), mask_cols);
    let mut builder = builder.with_projection(mask).with_batch_size(batch_size);
    if let Some(row_group) = row_group {
        builder = builder.with_row_groups(vec![row_group]);
    }
    Ok(ProjectedReader {
        inner: builder.build()?,
        order: (!identity).then_some(order),
    })
}

fn parallel_row_groups(
    path: PathBuf,
    columns: Vec<usize>,
    batch_size: usize,
    nthread: usize,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    if nthread <= 1 {
        return batch_reader(&path, &columns, batch_size);
    }
    let file = File::open(&path)?;
    let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())?;
    let n_groups = metadata.metadata().num_row_groups();
    if n_groups <= 1 {
        return batch_reader_with_metadata(file, metadata, &columns, batch_size);
    }
    // Share the parsed footer and projection decision across row groups.
    // Each group still opens its own file handle for an independent cursor.
    let coalesced = coalesced_projection(&metadata, &columns);
    Ok(Box::new(
        crate::source::parallel_reader::ordered_parallel_batches(
            n_groups,
            nthread,
            move |row_group| {
                let file = File::open(&path)
                    .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.display()))?;
                if coalesced {
                    let end = row_group_data_end(metadata.metadata().row_group(row_group));
                    return bounded_coalesced_group(
                        crate::source::parquet_io::CoalescedFile::for_row_group(file, end)?,
                        metadata.clone(),
                        &columns,
                        batch_size,
                        row_group,
                    );
                }
                let reader = build_projected_reader(
                    file,
                    metadata.clone(),
                    &columns,
                    batch_size,
                    Some(row_group),
                )?;
                Ok(Box::new(reader.map(|r| r.map_err(anyhow::Error::from))))
            },
        ),
    ))
}

// Bound decoder state independently of the full projection width. The existing
// coalesced eligibility check limits each row group to two standard batches.
const COALESCED_DECODE_COLUMNS: usize = 256;

fn coalesced_fixed_width_projection(metadata: &ArrowReaderMetadata, columns: &[usize]) -> bool {
    use arrow::datatypes::DataType;
    columns.iter().all(|&column| {
        metadata.schema().fields().get(column).is_some_and(|field| {
            matches!(
                field.data_type(),
                DataType::Boolean
                    | DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
                    | DataType::Float16
                    | DataType::Float32
                    | DataType::Float64
                    | DataType::Timestamp(_, _)
                    | DataType::Date32
                    | DataType::Date64
                    | DataType::Time32(_)
                    | DataType::Time64(_)
                    | DataType::Duration(_)
                    | DataType::Interval(_)
                    | DataType::Decimal128(_, _)
                    | DataType::Decimal256(_, _)
            )
        })
    })
}

fn bounded_coalesced_group(
    input: crate::source::parquet_io::CoalescedFile,
    metadata: ArrowReaderMetadata,
    columns: &[usize],
    batch_size: usize,
    row_group: usize,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    let rows = metadata.metadata().row_group(row_group).num_rows();
    if columns.len() <= COALESCED_DECODE_COLUMNS
        || !(0..=(2 * BATCH_SIZE) as i64).contains(&rows)
        || !coalesced_fixed_width_projection(&metadata, columns)
    {
        let reader = build_projected_reader(input, metadata, columns, batch_size, Some(row_group))?;
        return Ok(Box::new(
            reader.map(|batch| batch.map_err(anyhow::Error::from)),
        ));
    }
    let columns = columns.to_vec();
    // Defer page reads until iteration, matching the ordinary reader. Errors are
    // yielded once; a partially assembled row group is never exposed.
    Ok(Box::new(
        std::iter::once_with(move || {
            decode_coalesced_group(input, metadata, &columns, batch_size, row_group)
        })
        .flat_map(|result| match result {
            Ok(batches) => batches.into_iter().map(Ok).collect::<Vec<_>>(),
            Err(error) => vec![Err(error)],
        }),
    ))
}

/// Arrow keeps dictionaries and page decoders for every projected column alive
/// until its reader is dropped. Decode bounded column sets and drop each reader
/// before opening the next set. Reassemble the original row batches by sharing
/// their arrays, without copying values or changing row/feature order. This only
/// materializes one short row group of fixed-width values. Variable-width and
/// dictionary arrays retain the ordinary streaming path rather than staging
/// potentially large payloads across multiple batches.
fn decode_coalesced_group(
    input: crate::source::parquet_io::CoalescedFile,
    metadata: ArrowReaderMetadata,
    columns: &[usize],
    batch_size: usize,
    row_group: usize,
) -> anyhow::Result<Vec<RecordBatch>> {
    let (physical, order) = crate::source::input::projection_and_order(columns);
    let mut arrays: Vec<Vec<arrow::array::ArrayRef>> = Vec::new();
    let mut rows = Vec::new();
    let mut fields: Vec<arrow::datatypes::FieldRef> = Vec::new();
    for (part, chunk) in physical.chunks(COALESCED_DECODE_COLUMNS).enumerate() {
        let reader = build_projected_reader(
            input.clone(),
            metadata.clone(),
            chunk,
            batch_size,
            Some(row_group),
        )?;
        let mut count = 0;
        for batch in reader {
            let batch = batch?;
            if part == 0 {
                rows.push(batch.num_rows());
                arrays.push(Vec::with_capacity(physical.len()));
            }
            anyhow::ensure!(
                rows.get(count) == Some(&batch.num_rows()),
                "Parquet column decoder row batches differ"
            );
            if count == 0 {
                fields.extend(batch.schema().fields().iter().cloned());
            }
            arrays[count].extend(batch.columns().iter().cloned());
            count += 1;
        }
        anyhow::ensure!(
            count == rows.len(),
            "Parquet column decoder batch count differs"
        );
    }
    anyhow::ensure!(
        rows.iter().sum::<usize>() == metadata.metadata().row_group(row_group).num_rows() as usize,
        "Parquet column decoder row count differs from row-group metadata"
    );
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new_with_metadata(
        fields,
        metadata.schema().metadata().clone(),
    ));
    let identity = order.iter().enumerate().all(|(i, &o)| i == o) && order.len() == physical.len();
    arrays
        .into_iter()
        .map(|arrays| {
            let batch = RecordBatch::try_new(schema.clone(), arrays)?;
            Ok(if identity {
                batch
            } else {
                crate::source::input::reorder(&batch, &order)
            })
        })
        .collect()
}

fn row_group_data_end(group: &parquet::file::metadata::RowGroupMetaData) -> Option<u64> {
    group.columns().iter().try_fold(0, |end, column| {
        let offset = column
            .dictionary_page_offset()
            .unwrap_or_else(|| column.data_page_offset());
        let offset = u64::try_from(offset).ok()?;
        let len = u64::try_from(column.compressed_size()).ok()?;
        Some(end.max(offset.checked_add(len)?))
    })
}

/// 列名 -> 下标。
fn column_index(path: &Path, name: &str) -> anyhow::Result<usize> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    schema_column_index(builder.schema(), name)
}

fn schema_column_index(schema: &arrow::datatypes::Schema, name: &str) -> anyhow::Result<usize> {
    schema
        .fields()
        .iter()
        .position(|f| f.name() == name)
        .ok_or_else(|| {
            let have: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
            crate::input_err!("file has no column {name:?}; available columns: {have:?}")
        })
}

/// 只读 footer 的 schema,数一下特征列有多少个 —— 不解压任何数据。
///
/// 给「先 peek schema、再决定 blocking、最后才真正 open」这条线用:
/// `cols_per_block` 的自动选择需要 `n_features`,而 Parquet 在 open 之前
/// 是不知道的。把这一步单独拿出来,就不必把 `nthread` 一路传进
/// `partition_features`(那会让底层分块知道训练的线程策略)。
pub fn peek_n_features<'a>(
    path: impl AsRef<Path>,
    label: impl Into<Targets<'a>>,
) -> anyhow::Result<usize> {
    let (idx, _) = feature_columns(path.as_ref(), label)?;
    Ok(idx.len())
}

/// 文件里**全部**列名(含标签列),顺序即物理顺序。预测绑定要用它。
pub fn all_column_names(path: impl AsRef<Path>) -> anyhow::Result<Vec<String>> {
    let file = File::open(path.as_ref())
        .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.as_ref().display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect())
}

/// 按给定投影读原始 batch(不量化)。预测路径用。
pub fn raw_batches(
    path: impl AsRef<Path>,
    projection: &[usize],
) -> anyhow::Result<impl Iterator<Item = anyhow::Result<RecordBatch>> + 'static> {
    batch_reader(path.as_ref(), projection, BATCH_SIZE)
}

pub(crate) fn raw_batches_with_nthread(
    path: impl AsRef<Path>,
    projection: &[usize],
    nthread: usize,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    parallel_row_groups(
        path.as_ref().to_path_buf(),
        projection.to_vec(),
        BATCH_SIZE,
        nthread,
    )
}

/// 特征列名(已排除标签列),顺序就是训练时的规范顺序。
///
/// 模型要靠它做命名模式的预测绑定,所以这里必须和训练用的顺序完全一致 ——
/// 两边各推一遍就会有漂移的风险,故只此一处。
pub fn feature_names_of<'a>(
    path: impl AsRef<Path>,
    label: impl Into<Targets<'a>>,
) -> anyhow::Result<Vec<String>> {
    let (_, names) = feature_columns(path.as_ref(), label)?;
    Ok(names)
}

/// 只读 footer 拿行数,**不解压任何数据**。
///
/// 显存规划器的每一项都按行数走(gpair、row index、prediction、列块 buffer),
/// 所以在 open 之前就必须知道真实行数。以前这里传的是 `n_rounds`,
/// 模型于是把 10M 行当成 100 行来算 —— 解出来的块宽在真实数据上放不下。
pub fn peek_n_rows(path: impl AsRef<Path>) -> anyhow::Result<u64> {
    let file = File::open(path.as_ref())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    parquet_row_count(builder.metadata())
}

fn parquet_row_count(metadata: &parquet::file::metadata::ParquetMetaData) -> anyhow::Result<u64> {
    let n = metadata.file_metadata().num_rows();
    anyhow::ensure!(n >= 0, "negative row count in the parquet footer: {n}");
    Ok(n as u64)
}

pub(crate) fn peek_n_row_groups(path: impl AsRef<Path>) -> anyhow::Result<usize> {
    let file = File::open(path.as_ref())?;
    Ok(ParquetRecordBatchReaderBuilder::try_new(file)?
        .metadata()
        .num_row_groups())
}

fn row_group_work(path: &Path) -> anyhow::Result<(usize, usize)> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let metadata = builder.metadata();
    let max_rows = metadata
        .row_groups()
        .iter()
        .map(|group| group.num_rows().max(0) as usize)
        .max()
        .unwrap_or(0);
    Ok((metadata.num_row_groups(), max_rows.div_ceil(BATCH_SIZE)))
}

fn feature_columns<'a>(
    path: &Path,
    targets: impl Into<Targets<'a>>,
) -> anyhow::Result<(Vec<usize>, Vec<String>)> {
    let targets = targets.into();
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    features_in_schema(builder.schema(), targets)
}

fn features_in_schema(
    schema: &arrow::datatypes::Schema,
    targets: Targets<'_>,
) -> anyhow::Result<(Vec<usize>, Vec<String>)> {
    if let Some(w) = targets.weight {
        // 权重列找不到要报错,不能静默当成「没给权重」。
        schema_column_index(schema, w)?;
    }
    if let Some(q) = targets.qid {
        schema_column_index(schema, q)?;
    }
    let columns: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    targets.select_features(&columns)
}

/// The final categorical flags of the feature columns `feat_cols`, resolved
/// exactly as a fresh cut build does: an explicit `declared` list decides;
/// without one, dictionary-typed columns are categorical.
fn resolve_categorical_flags(
    path: &Path,
    feat_cols: &[usize],
    declared: Option<&[bool]>,
) -> anyhow::Result<Vec<bool>> {
    let declared = declared.unwrap_or(&[]);
    crate::ensure_input!(
        declared.is_empty() || declared.len() == feat_cols.len(),
        "feature_types has {} entries but the data has {} feature columns",
        declared.len(),
        feat_cols.len()
    );
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    Ok(feat_cols
        .iter()
        .enumerate()
        .map(|(f, &col)| match declared.get(f) {
            Some(&d) => d,
            None => matches!(
                schema.field(col).data_type(),
                arrow::datatypes::DataType::Dictionary(_, _)
            ),
        })
        .collect())
}

fn named_column_batches(
    path: &Path,
    name: &str,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    let file = File::open(path)?;
    let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())?;
    let column = schema_column_index(metadata.schema(), name)?;
    batch_reader_with_metadata(file, metadata, &[column], BATCH_SIZE)
}

/// 读一整列 f32(单文件 parquet 的权重列用)。不量化、不留其它列。
#[cfg_attr(not(feature = "python"), allow(dead_code))] // Called from the Python ingest path.
pub(crate) fn read_f32_column(path: impl AsRef<Path>, name: &str) -> anyhow::Result<Vec<f32>> {
    let mut out = Vec::new();
    for batch in named_column_batches(path.as_ref(), name)? {
        crate::source::arrow_source::for_each_column_f32(&batch?, 0, |v| out.push(v))?;
    }
    Ok(out)
}

/// Read a whole query-id column from every file, in the given order (the order the
/// dataset concatenates the files in). Integer columns of any width, or floating
/// point columns holding integral values.
pub(crate) fn read_u64_columns(paths: &[PathBuf], name: &str) -> anyhow::Result<Vec<u64>> {
    let mut out = Vec::new();
    for path in paths {
        for batch in named_column_batches(path, name)? {
            out.extend(crate::source::arrow_source::column_as_u64(
                &batch?, 0, name,
            )?);
        }
    }
    Ok(out)
}

/// 默认的读取批大小。够摊薄每批的固定开销,又不至于一批就把内存吃满。
const BATCH_SIZE: usize = 8192;

/// 打开一个 Parquet 文件,自己建分箱边界。
pub fn open(
    path: impl AsRef<Path>,
    label: &str,
    max_bins: u32,
    cols_per_block: usize,
) -> anyhow::Result<Dataset> {
    open_inner(path.as_ref(), label, None, max_bins, cols_per_block)
}

/// 只跑第一遍:扫一遍数据建分箱边界,不量化、不留数据。
///
/// 除了内存实测要用,它本身也有意义:多个分片要共用一套 cuts 时,
/// 先在一个分片(或全量)上把边界定下来,再分发给各方 —— 各自建
/// 边界会让同一个值落进不同的 bin。
pub fn cuts_only(path: impl AsRef<Path>, label: &str, max_bins: u32) -> anyhow::Result<BinCuts> {
    cuts_only_with_nthread(path, label, max_bins, crate::source::DEFAULT_INGEST_THREADS)
}

/// `cuts_only`,但同一 batch 内按特征并行。`nthread=0` 使用 Rayon 默认值。
pub fn cuts_only_with_nthread(
    path: impl AsRef<Path>,
    label: &str,
    max_bins: u32,
    nthread: usize,
) -> anyhow::Result<BinCuts> {
    cuts_only_observed(path, label, max_bins, nthread, &mut |_, _, _| {})
}

fn cuts_only_observed<'a>(
    path: impl AsRef<Path>,
    label: impl Into<Targets<'a>>,
    max_bins: u32,
    nthread: usize,
    observe: &mut impl FnMut(&'static str, usize, usize),
) -> anyhow::Result<BinCuts> {
    let path = path.as_ref().to_path_buf();
    let targets = label.into();
    let (feat_cols, _) = feature_columns(&path, targets)?;
    let (tasks, batches_per_task) = row_group_work(&path)?;
    let (reader_threads, transform_threads) = crate::source::parquet_pipeline_threads(
        nthread,
        tasks,
        batches_per_task,
        feat_cols.len(),
        feat_cols.len(),
    );
    observe("cuts", reader_threads, transform_threads);
    let categorical = targets.categorical.unwrap_or(&[]);
    let Some(weight) = targets.weight else {
        return crate::source::arrow_source::build_cuts_and_count_typed(
            parallel_row_groups(path, feat_cols, BATCH_SIZE, reader_threads)?,
            max_bins as usize,
            transform_threads,
            false,
            categorical,
            targets.missing,
            targets.categories,
        )
        .map(|(cuts, _)| cuts);
    };
    // 加权分箱:权重列跟在特征后面一起读。Parquet 投影按文件物理顺序
    // 返回,所以要投影后重排,保证权重是最后一列。
    let mut cols = feat_cols;
    cols.push(column_index(&path, weight)?);
    let (mask, order) = crate::source::input::projection_and_order(&cols);
    let batches = parallel_row_groups(path, mask, BATCH_SIZE, reader_threads)?
        .map(move |b| b.map(|b| crate::source::input::reorder(&b, &order)));
    crate::source::arrow_source::build_cuts_and_count_typed(
        batches,
        max_bins as usize,
        transform_threads,
        true,
        categorical,
        targets.missing,
        targets.categories,
    )
    .map(|(cuts, _)| cuts)
}

/// 打开一个 Parquet 文件,**复用给定的分箱边界**。
///
/// 验证集走这条。自己重新建草图会让同一个特征值在训练/验证里落进
/// 不同的 bin,模型直接失效 —— 跑起来不报错,只是指标莫名其妙地差。
pub fn open_with_cuts(
    path: impl AsRef<Path>,
    label: &str,
    cuts: BinCuts,
    cols_per_block: usize,
) -> anyhow::Result<Dataset> {
    open_inner(path.as_ref(), label, Some(cuts), 0, cols_per_block)
}

fn open_inner(
    path: &Path,
    label: &str,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
) -> anyhow::Result<Dataset> {
    open_many(&[path.to_path_buf()], label, cuts, max_bins, cols_per_block)
}

/// 多文件版本。单文件是它的退化情况,两条路走同一份代码。
///
/// **按列名绑定,不按物理顺序。** 各文件的列顺序可以不同,这里为每个文件
/// 单独解出「规范特征顺序 → 该文件列下标」的投影,所以读出来的 batch
/// 一定是同一个逻辑顺序。规范顺序取**第一个文件**的顺序(文件本身已按
/// 路径字典序排好,所以这个"第一个"是确定的)。
///
/// ⚠️ **不把多文件拼成一个大数组。** batch reader 按文件顺序惰性串起来,
/// 内存里同时只有一个 batch —— 否则"支持多文件"就变成了"先把数据全读进来",
/// 那正好违反这个项目的产品边界。
pub fn open_many(
    paths: &[PathBuf],
    label: &str,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
) -> anyhow::Result<Dataset> {
    open_many_with_nthread(
        paths,
        label,
        cuts,
        max_bins,
        cols_per_block,
        crate::source::DEFAULT_INGEST_THREADS,
    )
}

pub fn open_many_with_nthread(
    paths: &[PathBuf],
    label: &str,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
    nthread: usize,
) -> anyhow::Result<Dataset> {
    open_many_observed(
        paths,
        label,
        cuts,
        max_bins,
        cols_per_block,
        nthread,
        |_, _, _| {},
    )
}

pub(crate) fn open_many_observed<'a>(
    paths: &[PathBuf],
    targets: impl Into<Targets<'a>>,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
    nthread: usize,
    mut observe: impl FnMut(&'static str, usize, usize),
) -> anyhow::Result<Dataset> {
    let targets = targets.into();
    let metadata_profile = crate::source::openprof::start("multi metadata");
    let ManyFileLayout {
        features: feat_proj,
        feature_names,
        labels: label_proj,
        weights: weight_proj,
        max_batches: batches_per_task,
        ..
    } = inspect_many_files(paths, targets)?;
    let reader_tasks = paths.len();
    // A single-file dataset reader remains inline; the single-file training
    // route uses ParquetBlockSource and row-group-parallel cut construction.
    let cuts_plan = crate::source::parquet_pipeline_threads(
        nthread,
        reader_tasks,
        batches_per_task,
        feature_names.len(),
        feature_names.len(),
    );
    let bins_plan = crate::source::parquet_pipeline_threads(
        nthread,
        reader_tasks,
        batches_per_task,
        feature_names.len().div_ceil(cols_per_block.max(1)),
        feature_names.len(),
    );
    drop(metadata_profile);
    if cuts.is_none() {
        observe("cuts", cuts_plan.0, cuts_plan.1);
    }
    observe("bins", bins_plan.0, bins_plan.1);
    let owned = paths.to_vec();
    let feats = |quantize: bool| {
        let reader_threads = if quantize { bins_plan.0 } else { cuts_plan.0 };
        Ok(dataset_reader(
            owned.clone(),
            feat_proj.clone(),
            BATCH_SIZE,
            reader_threads,
        ))
    };
    // 加权分箱:先单独扫一遍「特征 + 权重列」建 cuts,再走给定 cuts 的量化。
    // 验证集(`cuts` 已给)沿用训练集的 cuts,不重建。
    let cuts = match (cuts, targets.weight) {
        (None, Some(_)) => {
            let proj: Vec<Vec<usize>> = feat_proj
                .iter()
                .zip(&weight_proj)
                .map(|(f, &w)| {
                    let mut cols = f.clone();
                    cols.push(w);
                    cols
                })
                .collect();
            let (c, _) = crate::source::arrow_source::build_cuts_and_count_typed(
                dataset_reader(owned.clone(), proj, BATCH_SIZE, cuts_plan.0),
                max_bins as usize,
                cuts_plan.1,
                true,
                targets.categorical.unwrap_or(&[]),
                targets.missing,
                targets.categories,
            )?;
            Some(c)
        }
        (c, _) => c,
    };
    let source = match cuts {
        Some(c) => ArrowSource::with_cuts_and_nthread_missing(
            || feats(true),
            c,
            cols_per_block,
            bins_plan.1,
            targets.missing,
        )?,
        None => ArrowSource::new_typed_with_phase_threads(
            feats,
            max_bins,
            cols_per_block,
            cuts_plan.1,
            bins_plan.1,
            targets.categorical.unwrap_or(&[]),
            targets.missing,
            targets.categories,
        )?,
    };

    let _target_profile = crate::source::openprof::start("multi labels and weights");
    let (labels, weights) = read_many_targets(paths, &label_proj, &weight_proj, nthread)?;
    Dataset::assemble(
        source,
        labels,
        feature_names,
        targets.label,
        weights.zip(targets.weight),
    )
}

fn dataset_reader(
    paths: Vec<PathBuf>,
    projections: Vec<Vec<usize>>,
    batch_size: usize,
    nthread: usize,
) -> crate::source::parallel_reader::BatchIter {
    if paths.len() <= 1 || nthread <= 1 {
        return Box::new(chained_reader(paths, projections, batch_size));
    }
    let paths = std::sync::Arc::new(paths);
    let projections = std::sync::Arc::new(projections);
    let n_files = paths.len();
    Box::new(crate::source::parallel_reader::ordered_parallel_batches(
        n_files,
        nthread,
        move |index| {
            let path = paths[index].clone();
            let cols = projections[index].clone();
            let (mask, order) = crate::source::input::projection_and_order(&cols);
            let iter = batch_reader(&path, &mask, batch_size)?;
            Ok(Box::new(iter.map(move |batch| {
                batch.map(|batch| crate::source::input::reorder(&batch, &order))
            })))
        },
    ))
}

/// Native lazy multi-file source with optional in-memory budget or persistent
/// cache. The existing eager `open_many` route remains unchanged.
pub(crate) fn open_many_with_cache_observed<'a>(
    paths: &[PathBuf],
    targets: impl Into<Targets<'a>>,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
    options: QuantizedCacheOptions,
    nthread: usize,
    mut observe: impl FnMut(&'static str, usize, usize),
) -> anyhow::Result<(ParquetBlockSource, Vec<f32>, Option<Vec<f32>>, Vec<String>)> {
    let targets = targets.into();
    let ManyFileLayout {
        features: file_cols,
        feature_names,
        labels: label_cols,
        weights: weight_cols,
        n_rows,
        max_batches,
    } = inspect_many_files(paths, targets)?;
    let cuts_plan = crate::source::parquet_pipeline_threads(
        nthread,
        paths.len(),
        max_batches,
        feature_names.len(),
        feature_names.len(),
    );
    let bins_plan = crate::source::parquet_pipeline_threads(
        nthread,
        paths.len(),
        max_batches,
        feature_names.len().div_ceil(cols_per_block.max(1)),
        feature_names.len(),
    );
    if cuts.is_none() {
        observe("cuts", cuts_plan.0, cuts_plan.1);
    }
    observe("bins", bins_plan.0, bins_plan.1);

    if let Some(cache_path) = options.path.as_ref() {
        if cache_path.join("manifest.json").exists() {
            let (source, labels, weights) = ParquetBlockSource::reuse_many_cache(
                paths,
                &file_cols,
                &label_cols,
                &weight_cols,
                targets,
                max_bins,
                cols_per_block,
                cache_path,
                cuts.as_ref(),
            )?;
            return Ok((source, labels, weights, feature_names));
        }
        crate::ensure_input!(
            !cache_path.exists() || std::fs::read_dir(cache_path)?.next().is_none(),
            "cache path {} exists and is not a valid ferrisboost cache",
            cache_path.display()
        );
    }

    let initial = if options.path.is_some() {
        file_identities(paths)?
    } else {
        Vec::new()
    };
    let cuts = match cuts {
        Some(cuts) => cuts,
        None if targets.weight.is_some() => {
            let projections: Vec<Vec<usize>> = file_cols
                .iter()
                .zip(&weight_cols)
                .map(|(f, &w)| {
                    let mut p = f.clone();
                    p.push(w);
                    p
                })
                .collect();
            crate::source::arrow_source::build_cuts_and_count_typed(
                dataset_reader(paths.to_vec(), projections, BATCH_SIZE, cuts_plan.0),
                max_bins as usize,
                cuts_plan.1,
                true,
                targets.categorical.unwrap_or(&[]),
                targets.missing,
                targets.categories,
            )?
            .0
        }
        None => {
            crate::source::arrow_source::build_cuts_and_count_typed(
                dataset_reader(paths.to_vec(), file_cols.clone(), BATCH_SIZE, cuts_plan.0),
                max_bins as usize,
                cuts_plan.1,
                false,
                targets.categorical.unwrap_or(&[]),
                targets.missing,
                targets.categories,
            )?
            .0
        }
    };
    let (labels, weights) = read_many_labels_weights(paths, &label_cols, &weight_cols, targets)?;
    crate::ensure_input!(
        cuts.n_feats() == feature_names.len(),
        "cuts and Parquet features have different widths"
    );
    crate::ensure_input!(
        labels.len() == n_rows && weights.as_ref().is_none_or(|w| w.len() == n_rows),
        "Parquet label or weight row count differs from the source"
    );
    cuts.check_widths()?;
    let groups = crate::columns::partition_features_by_width(&cuts.widths(), cols_per_block);
    let cache_bytes = quantized_bytes(&cuts, n_rows)?;
    let mut source = ParquetBlockSource {
        path: paths[0].clone(),
        source_paths: Some(paths.to_vec()),
        cuts,
        groups,
        file_col: file_cols[0].clone(),
        multi_file_cols: Some(file_cols),
        n_rows,
        n_features: feature_names.len(),
        cache: QuantizedCache::Memory(Vec::new()),
        cache_bytes,
        cache_reads: AtomicUsize::new(0),
        source_bytes: std::sync::atomic::AtomicU64::new(0),
        parquet_loads: AtomicUsize::new(0),
        missing: targets.missing,
    };
    crate::source::openprof::shape(source.n_rows, source.n_features, source.groups.len());
    source.build_cache(
        targets,
        max_bins,
        cols_per_block,
        options,
        nthread,
        &initial,
    )?;
    Ok((source, labels, weights, feature_names))
}

struct ManyFileLayout {
    features: Vec<Vec<usize>>,
    feature_names: Vec<String>,
    labels: Vec<usize>,
    weights: Vec<usize>,
    n_rows: usize,
    max_batches: usize,
}

fn inspect_many_files(paths: &[PathBuf], targets: Targets<'_>) -> anyhow::Result<ManyFileLayout> {
    anyhow::ensure!(!paths.is_empty(), "no input files");
    let mut layout = ManyFileLayout {
        features: Vec::with_capacity(paths.len()),
        feature_names: Vec::new(),
        labels: Vec::with_capacity(paths.len()),
        weights: Vec::new(),
        n_rows: 0,
        max_batches: 0,
    };
    for path in paths {
        let metadata = ArrowReaderMetadata::load(&File::open(path)?, ArrowReaderOptions::new())?;
        let schema = metadata.schema();
        let (columns, names) = features_in_schema(schema, targets)?;
        if layout.features.is_empty() {
            layout.features.push(columns);
            layout.feature_names = names;
        } else {
            crate::ensure_input!(
                names.len() == layout.feature_names.len(),
                "schemas differ across files: {} has {} feature columns, {} has {}",
                paths[0].display(),
                layout.feature_names.len(),
                path.display(),
                names.len()
            );
            let by_name: std::collections::HashMap<_, _> = names.iter().zip(columns).collect();
            let mapped = layout
                .feature_names
                .iter()
                .map(|name| {
                    by_name.get(name).copied().ok_or_else(|| {
                        crate::input_err!(
                            "schemas differ across files: {} has no feature column {name:?}",
                            path.display()
                        )
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            layout.features.push(mapped);
        }
        layout
            .labels
            .push(schema_column_index(schema, targets.label)?);
        if let Some(weight) = targets.weight {
            layout.weights.push(schema_column_index(schema, weight)?);
        }
        let rows = usize::try_from(parquet_row_count(metadata.metadata())?)
            .map_err(|_| anyhow::anyhow!("Parquet rows exceed usize"))?;
        layout.n_rows = layout
            .n_rows
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("row count overflows usize"))?;
        layout.max_batches = layout.max_batches.max(rows.div_ceil(BATCH_SIZE));
    }
    Ok(layout)
}

fn read_many_targets(
    paths: &[PathBuf],
    label_cols: &[usize],
    weight_cols: &[usize],
    nthread: usize,
) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
    let projections = label_cols
        .iter()
        .enumerate()
        .map(|(i, &label)| match weight_cols.get(i) {
            Some(&weight) => vec![label, weight],
            None => vec![label],
        })
        .collect();
    let mut labels = Vec::new();
    let mut weights = (!weight_cols.is_empty()).then(Vec::new);
    for batch in dataset_reader(paths.to_vec(), projections, BATCH_SIZE, nthread) {
        let batch = batch?;
        crate::source::arrow_source::for_each_column_f32(&batch, 0, |v| labels.push(v))?;
        if let Some(weights) = &mut weights {
            crate::source::arrow_source::for_each_column_f32(&batch, 1, |v| weights.push(v))?;
        }
    }
    Ok((labels, weights))
}

fn read_many_labels_weights(
    paths: &[PathBuf],
    label_cols: &[usize],
    weight_cols: &[usize],
    targets: Targets<'_>,
) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
    let (labels, weights) = read_many_targets(paths, label_cols, weight_cols, 1)?;
    crate::ensure_input!(
        !labels.iter().any(|v| v.is_nan()),
        "label column {:?} has missing values",
        targets.label
    );
    if let Some((weights, name)) = weights.as_ref().zip(targets.weight) {
        crate::source::check_weight_column(weights, name)?;
    }
    Ok((labels, weights))
}

/// 按文件顺序惰性串起来的 batch 流。打开失败作为一个 `Err` 元素传下去,
/// 这样调用方在正常的迭代里就能看见,不需要一条单独的错误通道。
fn chained_reader(
    paths: Vec<PathBuf>,
    projections: Vec<Vec<usize>>,
    batch_size: usize,
) -> impl Iterator<Item = anyhow::Result<RecordBatch>> {
    paths
        .into_iter()
        .zip(projections)
        .flat_map(move |(p, cols)| {
            // ⚠️ **投影只选列,不排序** —— Arrow 按文件物理顺序返回。
            // 所以按名字绑定必须投影后再重排,否则列序不同的文件会
            // **静默**把每个特征读成别人的值。预测路径上已经栽过一次。
            let (mask, order) = crate::source::input::projection_and_order(&cols);
            match batch_reader(&p, &mask, batch_size) {
                Ok(it) => {
                    Box::new(it.map(move |b| b.map(|b| crate::source::input::reorder(&b, &order))))
                        as Box<dyn Iterator<Item = anyhow::Result<RecordBatch>>>
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        })
}

/// 量化缓存的放置策略。
#[derive(Clone, Debug, Default)]
pub struct QuantizedCacheOptions {
    /// 内存缓存上限。`None` = `/proc/meminfo` 中可用内存的 50%;`Some(0)`
    /// 强制落盘。
    pub budget_bytes: Option<usize>,
    /// 持久缓存目录。给了就落盘并允许跨 run 复用;目录已存在但元数据或
    /// cuts 不一致时直接报错,绝不静默使用旧 bin。
    pub path: Option<PathBuf>,
}

#[derive(Serialize, Deserialize)]
struct CacheManifest {
    version: u32,
    source_path: String,
    source_len: u64,
    source_mtime_ns: u64,
    /// Ordered identities for native multi-file caches. Empty for legacy and
    /// single-file caches, preserving their on-disk representation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    source_files: Vec<SourceFileIdentity>,
    label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    label_upper_bound: Option<String>,
    /// 被排除出特征的权重列。旧缓存没有这个字段 = 没有权重列。
    /// 必须进身份校验:换一个权重列,特征集合就变了,而特征**数**可能不变。
    #[serde(default)]
    weight: Option<String>,
    /// Excluded `base_margin` columns (same identity argument as `weight`).
    #[serde(default)]
    base_margin: Vec<String>,
    /// Excluded query-id column (ranking); absent in older caches = none.
    #[serde(default)]
    qid: Option<String>,
    max_bins: u32,
    cols_per_block: usize,
    n_rows: usize,
    n_features: usize,
    cuts_values_bits: Vec<u32>,
    cuts_offsets: Vec<u32>,
    cuts_hash: String,
    /// Categorical flag per feature; absent in caches written before
    /// categorical support (= all numeric).
    #[serde(default)]
    categorical: Vec<bool>,
    /// Bits of the `missing` sentinel the bins were built with (canonical NaN =
    /// none). Older caches predate it and were always built without one.
    #[serde(default = "no_missing_bits")]
    missing_bits: u32,
    groups: Vec<Vec<FeatId>>,
    /// BLAKE3 of each `block-{i}.bin`, in `groups` order.
    block_hashes: Vec<String>,
    /// Bytes per bin of each block (version 3 only; version 2 = all 1).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    block_widths: Vec<u8>,
    /// Category names of named categorical features (XGBoost `cats` layout);
    /// absent when the data has none, so other caches keep their bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    categories: Option<crate::category_names::XgbCats>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SourceFileIdentity {
    path: String,
    len: u64,
    mtime_ns: u64,
}

/// Cache format of all-8-bit caches.
const CACHE_VERSION: u32 = 2;
/// Cache format with at least one 16-bit block (`block_widths` present).
const CACHE_VERSION_WIDE: u32 = 3;

/// Bytes of the whole quantized dataset at each feature's storage width.
fn quantized_bytes(cuts: &BinCuts, n_rows: usize) -> anyhow::Result<usize> {
    (0..cuts.n_feats() as FeatId)
        .try_fold(0usize, |acc, f| {
            n_rows
                .checked_mul(cuts.bin_width(f).bytes())
                .and_then(|b| acc.checked_add(b))
        })
        .ok_or_else(|| anyhow::anyhow!("quantized cache size overflows"))
}

fn no_missing_bits() -> u32 {
    f32::NAN.to_bits()
}

/// Bit pattern used to compare `missing` sentinels: every NaN is "none".
fn missing_bits(missing: f32) -> u32 {
    if missing.is_nan() {
        no_missing_bits()
    } else {
        missing.to_bits()
    }
}

enum QuantizedCache {
    Memory(Vec<ColumnBlock>),
    Disk {
        dir: PathBuf,
        delete_on_drop: bool,
        block_hashes: Vec<String>,
        /// Hash each block only on its first actual read. This catches a
        /// same-length corrupt payload without rereading a 12 GB cache at open.
        verified: Vec<AtomicBool>,
    },
}

impl Drop for QuantizedCache {
    fn drop(&mut self) {
        if let Self::Disk {
            dir,
            delete_on_drop: true,
            ..
        } = self
        {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

struct CacheWriteLock(PathBuf);

impl CacheWriteLock {
    fn acquire(cache_path: &Path) -> anyhow::Result<Self> {
        let mut name = cache_path.as_os_str().to_os_string();
        name.push(".lock");
        let lock = PathBuf::from(name);
        std::fs::create_dir(&lock).map_err(|e| {
            anyhow::anyhow!(
                "cannot acquire cache writer lock {} (another writer may be active): {e}",
                lock.display()
            )
        })?;
        Ok(Self(lock))
    }
}

impl Drop for CacheWriteLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}

struct CacheStage(Option<PathBuf>);

impl Drop for CacheStage {
    fn drop(&mut self) {
        if let Some(dir) = &self.0 {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// Parquet 数据源。原始列只在建 cuts 和量化缓存时读取;训练阶段按需读取
/// 的是紧凑的 u8 列块,不再回源解压和重复量化。
pub struct ParquetBlockSource {
    path: PathBuf,
    /// `None` for legacy single-file sources; otherwise ordered concatenation.
    source_paths: Option<Vec<PathBuf>>,
    cuts: BinCuts,
    /// 每个块持有哪些**全局特征 id**
    groups: Vec<Vec<FeatId>>,
    /// 全局特征 id → 文件里的列下标(标签列已经排除,所以两者不等)
    file_col: Vec<usize>,
    /// Canonical feature projection for each source file, in row order.
    multi_file_cols: Option<Vec<Vec<usize>>>,
    n_rows: usize,
    n_features: usize,
    cache: QuantizedCache,
    cache_bytes: usize,
    cache_reads: AtomicUsize,
    /// 真实从 cache 读入的量化字节数。和 H2D 字节分开统计。
    source_bytes: std::sync::atomic::AtomicU64,
    parquet_loads: AtomicUsize,
    /// XGBoost `missing` sentinel (NaN = none), applied when quantizing features.
    missing: f32,
}

impl ParquetBlockSource {
    fn quantize_block(&self, i: usize) -> anyhow::Result<ColumnBlock> {
        self.parquet_loads.fetch_add(1, Ordering::Relaxed);
        let feat_ids = self
            .groups
            .get(i)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "column block {i} does not exist ({} blocks)",
                    self.groups.len()
                )
            })?
            .clone();
        // 只投影这一组列,别的 column chunk 根本不读
        let cols: Vec<usize> = feat_ids
            .iter()
            .map(|&f| self.file_col[f as usize])
            .collect();

        let len = feat_ids.len() * self.n_rows;
        let data = match self.cuts.block_width(&feat_ids) {
            BinWidth::U8 => {
                let mut data = vec![0u8; len];
                if let Some(file_cols) = &self.multi_file_cols {
                    self.read_many_into(file_cols, &feat_ids, self.n_rows, &mut data)?;
                } else {
                    self.read_into(&cols, &feat_ids, self.n_rows, &mut data)?;
                }
                BinData::U8(data)
            }
            BinWidth::U16 => {
                let mut data = vec![0u16; len];
                if let Some(file_cols) = &self.multi_file_cols {
                    self.read_many_into(file_cols, &feat_ids, self.n_rows, &mut data)?;
                } else {
                    self.read_into(&cols, &feat_ids, self.n_rows, &mut data)?;
                }
                BinData::U16(data)
            }
        };
        Ok(ColumnBlock {
            feat_ids,
            data,
            n_rows: self.n_rows,
        })
    }

    /// 兼容旧的基准字段:现在它表示缓存读取次数,不是 Parquet 解压次数。
    pub fn load_count(&self) -> usize {
        self.cache_read_count()
    }

    pub fn cache_read_count(&self) -> usize {
        self.cache_reads.load(Ordering::Relaxed)
    }

    /// 真实读入的量化字节数(见 `BlockSource::source_bytes_read`)。
    pub fn source_bytes(&self) -> u64 {
        self.source_bytes.load(Ordering::Relaxed)
    }

    pub fn parquet_load_count(&self) -> usize {
        self.parquet_loads.load(Ordering::Relaxed)
    }

    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    pub fn cache_kind(&self) -> &'static str {
        match self.cache {
            QuantizedCache::Memory(_) => "memory",
            QuantizedCache::Disk { .. } => "disk",
        }
    }

    /// 扫一遍建分箱边界,然后**只记住怎么去取数据**,不留数据本身。
    pub fn open(
        path: impl AsRef<Path>,
        label: &str,
        max_bins: u32,
        cols_per_block: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        Self::open_with_cache(
            path,
            label,
            max_bins,
            cols_per_block,
            QuantizedCacheOptions::default(),
        )
    }

    /// `open`,但 cuts sketch 和量化都服从 `nthread`。
    pub fn open_with_nthread(
        path: impl AsRef<Path>,
        label: &str,
        max_bins: u32,
        cols_per_block: usize,
        nthread: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        Self::open_with_cache_and_nthread(
            path,
            label,
            max_bins,
            cols_per_block,
            QuantizedCacheOptions::default(),
            nthread,
        )
    }

    pub fn open_with_cache(
        path: impl AsRef<Path>,
        label: &str,
        max_bins: u32,
        cols_per_block: usize,
        options: QuantizedCacheOptions,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        Self::open_with_cache_and_nthread(
            path,
            label,
            max_bins,
            cols_per_block,
            options,
            crate::source::DEFAULT_INGEST_THREADS,
        )
    }

    /// `open_with_cache`,但 cuts sketch 和量化都服从 `nthread`。
    /// `nthread=0` 使用 Rayon 默认值。缓存复用时没有 cuts scan 或量化,
    /// 因此不会创建线程池。
    pub fn open_with_cache_and_nthread(
        path: impl AsRef<Path>,
        label: &str,
        max_bins: u32,
        cols_per_block: usize,
        options: QuantizedCacheOptions,
        nthread: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        Self::open_with_cache_observed(
            path,
            label,
            max_bins,
            cols_per_block,
            options,
            nthread,
            |_, _, _| {},
        )
    }

    pub(crate) fn open_with_cache_observed<'a>(
        path: impl AsRef<Path>,
        targets: impl Into<Targets<'a>>,
        max_bins: u32,
        cols_per_block: usize,
        options: QuantizedCacheOptions,
        nthread: usize,
        mut observe: impl FnMut(&'static str, usize, usize),
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        let path = path.as_ref().to_path_buf();
        let targets = targets.into();
        let label = targets.label;
        let (feat_cols, label_col) = {
            let _p = crate::source::openprof::start("metadata");
            let (feat_cols, _) = feature_columns(&path, targets)?;
            (feat_cols, column_index(&path, label)?)
        };
        if let Some(cache_path) = options.path.as_ref() {
            if cache_path.join("manifest.json").exists() {
                return Self::reuse_cache(
                    path,
                    feat_cols,
                    label_col,
                    targets,
                    max_bins,
                    cols_per_block,
                    cache_path,
                );
            }
            crate::ensure_input!(
                !cache_path.exists() || std::fs::read_dir(cache_path)?.next().is_none(),
                "cache path {} exists and is not a valid ferrisboost cache",
                cache_path.display()
            );
        }
        let cuts = {
            let _p = crate::source::openprof::start("cuts");
            cuts_only_observed(&path, targets, max_bins, nthread, &mut observe)?
        };
        Self::with_cuts_inner(
            path,
            feat_cols,
            label_col,
            cuts,
            max_bins,
            cols_per_block,
            targets,
            options,
            nthread,
        )
    }

    /// 复用给定的分箱边界(验证集走这条)。
    pub fn with_cuts(
        path: impl AsRef<Path>,
        label: &str,
        cuts: BinCuts,
        cols_per_block: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        Self::with_cuts_and_cache(
            path,
            label,
            cuts,
            cols_per_block,
            QuantizedCacheOptions::default(),
        )
    }

    /// `with_cuts`,但量化服从 `nthread`。
    pub fn with_cuts_and_nthread(
        path: impl AsRef<Path>,
        label: &str,
        cuts: BinCuts,
        cols_per_block: usize,
        nthread: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        Self::with_cuts_and_cache_and_nthread(
            path,
            label,
            cuts,
            cols_per_block,
            QuantizedCacheOptions::default(),
            nthread,
        )
    }

    pub fn with_cuts_and_cache(
        path: impl AsRef<Path>,
        label: &str,
        cuts: BinCuts,
        cols_per_block: usize,
        options: QuantizedCacheOptions,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        Self::with_cuts_and_cache_and_nthread(
            path,
            label,
            cuts,
            cols_per_block,
            options,
            crate::source::DEFAULT_INGEST_THREADS,
        )
    }

    pub fn with_cuts_and_cache_and_nthread<'a>(
        path: impl AsRef<Path>,
        targets: impl Into<Targets<'a>>,
        cuts: BinCuts,
        cols_per_block: usize,
        options: QuantizedCacheOptions,
        nthread: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        let path = path.as_ref().to_path_buf();
        let targets = targets.into();
        let (feat_cols, _) = feature_columns(&path, targets)?;
        let label_col = column_index(&path, targets.label)?;
        if let Some(cache_path) = options.path.as_ref() {
            if cache_path.join("manifest.json").exists() {
                let (src, labels) = Self::reuse_cache(
                    path,
                    feat_cols,
                    label_col,
                    targets,
                    0,
                    cols_per_block,
                    cache_path,
                )?;
                // 逐值比,不只比哈希:manifest 里存着完整的 cuts,
                // 直接比是免费的,而且比任何哈希都强。哈希只用来发现
                // manifest 自身被改坏。
                crate::ensure_input!(
                    cuts.categories().map(|t| &**t) == src.cuts().categories().map(|t| &**t),
                    "the given category names differ from the cached ones; refusing to use \
                     quantized data coded with other categories"
                );
                crate::ensure_input!(
                    cuts.values == src.cuts().values && cuts.offsets == src.cuts().offsets,
                    "the given cuts differ from the cached ones; refusing to use stale quantized data; \
                     a validation set must use the training cuts but must **not share cache_path** with the training set: \
                     their quantized data differ and would overwrite each other"
                );
                return Ok((src, labels));
            }
        }
        Self::with_cuts_inner(
            path,
            feat_cols,
            label_col,
            cuts,
            0,
            cols_per_block,
            targets,
            options,
            nthread,
        )
    }

    // Keep this private constructor seam explicit: collapsing its inputs into a
    // config object would only add ownership/plumbing churn for a lint threshold.
    #[allow(clippy::too_many_arguments)]
    fn with_cuts_inner(
        path: PathBuf,
        feat_cols: Vec<usize>,
        label_col: usize,
        cuts: BinCuts,
        max_bins: u32,
        cols_per_block: usize,
        targets: Targets<'_>,
        options: QuantizedCacheOptions,
        nthread: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        let label = targets.label;
        let n_features = feat_cols.len();
        crate::ensure_input!(
            cuts.n_feats() == n_features,
            "cuts were built for {} features, the file has {n_features}",
            cuts.n_feats()
        );

        // 标签得留在内存里:训练每轮都要用,而且它只有一列。
        let mut labels = Vec::new();
        let mut n_rows = 0usize;
        {
            let _p = crate::source::openprof::start("label");
            for batch in batch_reader(&path, &[label_col], BATCH_SIZE)? {
                let batch = batch?;
                n_rows += batch.num_rows();
                labels.extend(crate::source::arrow_source::column_as_f32(&batch, 0)?);
            }
        }
        crate::ensure_input!(
            !labels.iter().any(|v| v.is_nan()),
            "label column {label:?} has missing values"
        );

        cuts.check_widths()?;
        let groups = crate::columns::partition_features_by_width(&cuts.widths(), cols_per_block);
        let cache_bytes = quantized_bytes(&cuts, n_rows)?;
        let mut source = Self {
            path,
            source_paths: None,
            cuts,
            groups,
            file_col: feat_cols,
            multi_file_cols: None,
            n_rows,
            n_features,
            cache: QuantizedCache::Memory(Vec::new()),
            cache_bytes,
            cache_reads: AtomicUsize::new(0),
            source_bytes: std::sync::atomic::AtomicU64::new(0),
            parquet_loads: AtomicUsize::new(0),
            missing: targets.missing,
        };
        crate::source::openprof::shape(source.n_rows, source.n_features, source.groups.len());
        source.build_cache(targets, max_bins, cols_per_block, options, nthread, &[])?;
        Ok((source, labels))
    }

    fn build_cache(
        &mut self,
        targets: Targets<'_>,
        max_bins: u32,
        cols_per_block: usize,
        options: QuantizedCacheOptions,
        nthread: usize,
        initial: &[SourceFileIdentity],
    ) -> anyhow::Result<()> {
        let budget = options.budget_bytes.unwrap_or_else(default_cache_budget);
        if options.path.is_none() && self.cache_bytes <= budget {
            let pool = crate::threading::build_pool(nthread).map_err(|e| {
                anyhow::anyhow!("failed to create the Parquet quantize thread pool: {e}")
            })?;
            let blocks = {
                let _p = crate::source::openprof::start("quantize");
                pool.install(|| {
                    (0..self.groups.len())
                        .into_par_iter()
                        .map(|i| self.quantize_block(i))
                        .collect::<anyhow::Result<Vec<_>>>()
                })?
            };
            let _p = crate::source::openprof::start("cache commit");
            self.cache = QuantizedCache::Memory(blocks);
            return Ok(());
        }

        let (dir, write_dir, _lock) = match options.path {
            Some(path) => {
                let parent = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."));
                std::fs::create_dir_all(parent)?;
                let lock = CacheWriteLock::acquire(&path)?;
                crate::ensure_input!(
                    !path.exists() || std::fs::read_dir(&path)?.next().is_none(),
                    "cache path {} already exists and is not empty",
                    path.display()
                );
                let stage = parent.join(format!(
                    ".ferrisboost-cache-{}-{}-staging",
                    std::process::id(),
                    TEMP_CACHE_ID.fetch_add(1, Ordering::Relaxed)
                ));
                (path, stage, Some(lock))
            }
            None => {
                let path = std::env::temp_dir().join(format!(
                    "ferrisboost-cache-{}-{}",
                    std::process::id(),
                    TEMP_CACHE_ID.fetch_add(1, Ordering::Relaxed)
                ));
                (path.clone(), path, None)
            }
        };
        std::fs::create_dir(&write_dir)?;
        let mut staging = CacheStage(Some(write_dir.clone()));
        let write_block = |i| -> anyhow::Result<String> {
            let block = self.quantize_block(i)?;
            let bytes = block.data.as_le_bytes();
            File::create(write_dir.join(format!("block-{i}.bin")))?.write_all(bytes)?;
            Ok(block_hash(bytes))
        };
        let block_hashes = {
            let _p = crate::source::openprof::start("quantize+write");
            if self.source_paths.is_some() {
                // Keep only one multi-file block in memory while writing.
                (0..self.groups.len())
                    .map(write_block)
                    .collect::<anyhow::Result<Vec<_>>>()?
            } else {
                let pool = crate::threading::build_pool(nthread).map_err(|e| {
                    anyhow::anyhow!("failed to create the Parquet quantize thread pool: {e}")
                })?;
                pool.install(|| {
                    (0..self.groups.len())
                        .into_par_iter()
                        .map(write_block)
                        .collect::<anyhow::Result<Vec<_>>>()
                })?
            }
        };
        let _p = crate::source::openprof::start("cache commit");
        let manifest = self.manifest(targets, max_bins, cols_per_block, block_hashes.clone())?;
        if !initial.is_empty() {
            crate::ensure_input!(
                manifest.source_files == initial,
                "the Parquet files changed while building the quantized cache"
            );
        }
        File::create(write_dir.join("manifest.json"))?
            .write_all(&serde_json::to_vec(&manifest)?)?;
        let delete_on_drop = write_dir == dir;
        if !delete_on_drop {
            if dir.exists() {
                std::fs::remove_dir(&dir)?;
            }
            std::fs::rename(&write_dir, &dir)?;
        }
        staging.0 = None;
        self.cache = QuantizedCache::Disk {
            dir,
            delete_on_drop,
            verified: block_hashes
                .iter()
                .map(|_| AtomicBool::new(false))
                .collect(),
            block_hashes,
        };
        Ok(())
    }

    fn manifest(
        &self,
        targets: Targets<'_>,
        max_bins: u32,
        cols_per_block: usize,
        block_hashes: Vec<String>,
    ) -> anyhow::Result<CacheManifest> {
        let (source_path, source_len, source_mtime_ns) = source_identity(&self.path)?;
        let source_files = self
            .source_paths
            .as_deref()
            .map(file_identities)
            .transpose()?
            .unwrap_or_default();
        let widths: Vec<u8> = self
            .groups
            .iter()
            .map(|g| self.cuts.block_width(g).bytes() as u8)
            .collect();
        // Version 2 stays the format of all-8-bit caches (byte-identical to
        // 0.0.7, so existing caches remain valid); a cache with any 16-bit
        // block is version 3 and records the width of every block.
        let wide = widths.iter().any(|&w| w != 1);
        Ok(CacheManifest {
            version: if wide {
                CACHE_VERSION_WIDE
            } else {
                CACHE_VERSION
            },
            source_path,
            source_len,
            source_mtime_ns,
            source_files,
            label: targets.label.to_string(),
            label_upper_bound: targets.label_upper_bound.map(str::to_string),
            weight: targets.weight.map(str::to_string),
            base_margin: targets
                .base_margin
                .map(<[String]>::to_vec)
                .unwrap_or_default(),
            qid: targets.qid.map(str::to_string),
            max_bins,
            cols_per_block,
            n_rows: self.n_rows,
            n_features: self.n_features,
            cuts_values_bits: self.cuts.values.iter().map(|v| v.to_bits()).collect(),
            cuts_offsets: self.cuts.offsets.clone(),
            cuts_hash: cuts_hash(&self.cuts),
            categorical: self.cuts.categorical_flags().to_vec(),
            missing_bits: missing_bits(self.missing),
            groups: self.groups.clone(),
            block_hashes,
            block_widths: if wide { widths } else { Vec::new() },
            categories: self.cuts.categories().map(|t| t.to_xgb()),
        })
    }

    fn reuse_many_cache(
        paths: &[PathBuf],
        file_cols: &[Vec<usize>],
        label_cols: &[usize],
        weight_cols: &[usize],
        targets: Targets<'_>,
        max_bins: u32,
        cols_per_block: usize,
        cache_path: &Path,
        provided_cuts: Option<&BinCuts>,
    ) -> anyhow::Result<(Self, Vec<f32>, Option<Vec<f32>>)> {
        let manifest: CacheManifest =
            serde_json::from_slice(&std::fs::read(cache_path.join("manifest.json"))?)?;
        crate::ensure_input!(
            manifest.version == CACHE_VERSION || manifest.version == CACHE_VERSION_WIDE,
            "unsupported quantized cache version {}",
            manifest.version
        );
        let identities = paths
            .iter()
            .map(|p| {
                let (path, len, mtime_ns) = source_identity(p)?;
                Ok(SourceFileIdentity {
                    path,
                    len,
                    mtime_ns,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        crate::ensure_input!(
            manifest.source_files == identities,
            "the cache's ordered Parquet files have changed or are not the current files"
        );
        crate::ensure_input!(
            manifest.label == targets.label,
            "cache label column differs"
        );
        crate::ensure_input!(
            manifest.label_upper_bound.as_deref() == targets.label_upper_bound,
            "cache upper-bound label column differs"
        );
        crate::ensure_input!(
            manifest.weight.as_deref() == targets.weight,
            "cache weight column differs"
        );
        crate::ensure_input!(
            manifest.base_margin.as_slice() == targets.base_margin.unwrap_or(&[]),
            "cache base_margin column differs"
        );
        crate::ensure_input!(
            manifest.qid.as_deref() == targets.qid,
            "cache query-id column differs"
        );
        crate::ensure_input!(
            manifest.cols_per_block == cols_per_block,
            "cache cols_per_block differs (cached {}, this run {})",
            manifest.cols_per_block,
            cols_per_block
        );
        if max_bins != 0 {
            crate::ensure_input!(manifest.max_bins == max_bins, "cache max_bin differs");
        }
        crate::ensure_input!(
            manifest.n_features == file_cols[0].len(),
            "cache feature count differs"
        );
        crate::ensure_input!(
            manifest.missing_bits == missing_bits(targets.missing),
            "cache was built with a different missing value"
        );
        let resolved = resolve_categorical_flags(&paths[0], &file_cols[0], targets.categorical)?;
        crate::ensure_input!(
            (0..manifest.n_features)
                .all(|f| resolved[f] == manifest.categorical.get(f).copied().unwrap_or(false)),
            "cached feature types differ from the resolved feature types"
        );
        let cuts = BinCuts::new(
            manifest
                .cuts_values_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect(),
            manifest.cuts_offsets.clone(),
        )
        .with_categorical(&manifest.categorical)
        .map_err(|e| anyhow::anyhow!("cached categorical feature metadata is corrupt: {e}"))?;
        let categories = match &manifest.categories {
            None => None,
            Some(c) => crate::category_names::CategoryTable::from_xgb(c, manifest.n_features)
                .and_then(|t| {
                    if let Some(t) = &t {
                        t.check_against(manifest.n_features, &manifest.categorical)?;
                    }
                    Ok(t)
                })
                .map_err(|e| anyhow::anyhow!("cached category names are corrupt: {e}"))?,
        };
        if let Some(seed) = targets.categories {
            for f in 0..manifest.n_features {
                if let Some(c) = seed.get(f) {
                    crate::ensure_input!(
                        categories.as_ref().and_then(|t| t.get(f)) == Some(c),
                        "cached category names differ from the existing model at feature {f}"
                    );
                }
            }
        }
        let cuts = cuts.with_categories(categories.map(std::sync::Arc::new));
        anyhow::ensure!(
            cuts_hash(&cuts) == manifest.cuts_hash,
            "cache cuts hash differs"
        );
        if let Some(given) = provided_cuts {
            crate::ensure_input!(
                given.categories().map(|t| &**t) == cuts.categories().map(|t| &**t),
                "the given category names differ from the cached ones"
            );
            crate::ensure_input!(
                given.values == cuts.values && given.offsets == cuts.offsets,
                "the given cuts differ from the cached ones; refusing to use stale quantized data"
            );
        }
        cuts.check_widths()?;
        let groups = crate::columns::partition_features_by_width(&cuts.widths(), cols_per_block);
        anyhow::ensure!(
            manifest.groups == groups,
            "cache column block map is corrupt"
        );
        anyhow::ensure!(
            manifest.block_hashes.len() == groups.len(),
            "cache block checksum count differs"
        );
        let widths: Vec<BinWidth> = groups.iter().map(|g| cuts.block_width(g)).collect();
        let recorded: Vec<BinWidth> = if manifest.version == CACHE_VERSION {
            vec![BinWidth::U8; groups.len()]
        } else {
            manifest
                .block_widths
                .iter()
                .map(|&w| {
                    BinWidth::from_bytes(w).ok_or_else(|| {
                        anyhow::anyhow!("unsupported cache block width of {w} bytes")
                    })
                })
                .collect::<anyhow::Result<_>>()?
        };
        anyhow::ensure!(
            recorded == widths,
            "cache block bin widths do not match the cuts"
        );
        let (labels, weights) = read_many_labels_weights(paths, label_cols, weight_cols, targets)?;
        let n_rows = labels.len();
        anyhow::ensure!(
            weights.as_ref().map_or(true, |w| w.len() == n_rows),
            "cache weight row count differs"
        );
        anyhow::ensure!(n_rows == manifest.n_rows, "cache row count differs");
        for (i, group) in groups.iter().enumerate() {
            let expected = group
                .len()
                .saturating_mul(n_rows)
                .saturating_mul(widths[i].bytes()) as u64;
            anyhow::ensure!(
                std::fs::metadata(cache_path.join(format!("block-{i}.bin")))?.len() == expected,
                "cache block {i} length differs"
            );
        }
        let first_identity = source_identity(&paths[0])?;
        crate::ensure_input!(
            (
                manifest.source_path.as_str(),
                manifest.source_len,
                manifest.source_mtime_ns
            ) == (
                first_identity.0.as_str(),
                first_identity.1,
                first_identity.2
            ),
            "cache first-file identity differs"
        );
        let cache_bytes = quantized_bytes(&cuts, n_rows)?;
        Ok((
            Self {
                path: paths[0].clone(),
                source_paths: Some(paths.to_vec()),
                cuts,
                groups,
                file_col: file_cols[0].clone(),
                multi_file_cols: Some(file_cols.to_vec()),
                n_rows,
                n_features: file_cols[0].len(),
                cache: QuantizedCache::Disk {
                    dir: cache_path.to_path_buf(),
                    delete_on_drop: false,
                    verified: manifest
                        .block_hashes
                        .iter()
                        .map(|_| AtomicBool::new(false))
                        .collect(),
                    block_hashes: manifest.block_hashes,
                },
                cache_bytes,
                cache_reads: AtomicUsize::new(0),
                source_bytes: std::sync::atomic::AtomicU64::new(0),
                parquet_loads: AtomicUsize::new(0),
                missing: targets.missing,
            },
            labels,
            weights,
        ))
    }

    fn reuse_cache(
        path: PathBuf,
        feat_cols: Vec<usize>,
        label_col: usize,
        targets: Targets<'_>,
        max_bins: u32,
        cols_per_block: usize,
        cache_path: &Path,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        let label = targets.label;
        let raw = std::fs::read(cache_path.join("manifest.json"))?;
        let manifest: CacheManifest = serde_json::from_slice(&raw)?;
        crate::ensure_input!(
            manifest.version == CACHE_VERSION || manifest.version == CACHE_VERSION_WIDE,
            "unsupported quantized cache version {} (payload checksums start at version 2; rebuild the old cache)",
            manifest.version
        );
        crate::ensure_input!(
            manifest.source_files.is_empty(),
            "the cache was built for multiple Parquet files"
        );
        let identity = source_identity(&path)?;
        crate::ensure_input!(
            (
                manifest.source_path.as_str(),
                manifest.source_len,
                manifest.source_mtime_ns
            ) == (identity.0.as_str(), identity.1, identity.2),
            "the cache's Parquet file has changed or is not the current file"
        );
        crate::ensure_input!(manifest.label == label, "cache label column differs");
        crate::ensure_input!(
            manifest.label_upper_bound.as_deref() == targets.label_upper_bound,
            "cache upper-bound label column differs"
        );
        crate::ensure_input!(
            manifest.weight.as_deref() == targets.weight,
            "cache weight column differs (cached {:?}, current {:?})",
            manifest.weight,
            targets.weight
        );
        crate::ensure_input!(
            manifest.base_margin.as_slice() == targets.base_margin.unwrap_or(&[]),
            "cache base_margin column differs (cached {:?}, current {:?})",
            manifest.base_margin,
            targets.base_margin
        );
        crate::ensure_input!(
            manifest.qid.as_deref() == targets.qid,
            "cache query-id column differs (cache {:?}, this run {:?})",
            manifest.qid,
            targets.qid
        );
        crate::ensure_input!(
            manifest.cols_per_block == cols_per_block,
            "cache cols_per_block differs (cached {}, this run {}); pass the cached value or rebuild the cache",
            manifest.cols_per_block,
            cols_per_block
        );
        if max_bins != 0 {
            crate::ensure_input!(manifest.max_bins == max_bins, "cache max_bin differs");
        }
        crate::ensure_input!(
            manifest.n_features == feat_cols.len(),
            "cache feature count differs"
        );
        anyhow::ensure!(
            manifest.block_hashes.len() == manifest.groups.len(),
            "cache block checksum count differs"
        );
        anyhow::ensure!(
            manifest.cuts_offsets.len() == manifest.n_features + 1
                && manifest.cuts_offsets.first() == Some(&0)
                && manifest.cuts_offsets.last().copied()
                    == Some(manifest.cuts_values_bits.len() as u32)
                && manifest.cuts_offsets.windows(2).all(|w| w[0] <= w[1]),
            "cache cuts offsets are corrupt"
        );
        // The sentinel decides which values are missing, so cached bins are only
        // valid for the same `missing` (a stale cache would silently bin
        // sentinel values as ordinary numbers or vice versa).
        crate::ensure_input!(
            manifest.missing_bits == missing_bits(targets.missing),
            "cache was built with a different missing value (cached {}, current {}); use another cache_path or rebuild the cache",
            describe_missing(manifest.missing_bits),
            describe_missing(missing_bits(targets.missing))
        );
        // The cached binning is only valid for the same final feature types.
        // Resolve them exactly as a fresh build does (declared "c", or a
        // dictionary-typed column when nothing is declared), then compare:
        // omitting `feature_types` on a cache built with categorical features
        // must not silently keep training them as categorical, and the same
        // declaration must always reuse the cache it built.
        let resolved = resolve_categorical_flags(&path, &feat_cols, targets.categorical)?;
        let cached = |f: usize| manifest.categorical.get(f).copied().unwrap_or(false);
        crate::ensure_input!(
            (0..manifest.n_features).all(|f| resolved[f] == cached(f)),
            "cached feature types (categorical) differ from the resolved feature types \
             (feature_types or the file's dictionary columns changed); use another cache_path or rebuild the cache"
        );
        let cuts = BinCuts::new(
            manifest
                .cuts_values_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect(),
            manifest.cuts_offsets.clone(),
        )
        .with_categorical(&manifest.categorical)
        .map_err(|e| anyhow::anyhow!("cached categorical feature metadata is corrupt: {e}"))?;
        let cached_categories = match &manifest.categories {
            None => None,
            Some(c) => crate::category_names::CategoryTable::from_xgb(c, manifest.n_features)
                .and_then(|t| {
                    if let Some(t) = &t {
                        t.check_against(manifest.n_features, &manifest.categorical)?;
                    }
                    Ok(t)
                })
                .map_err(|e| anyhow::anyhow!("cached category names are corrupt: {e}"))?,
        };
        // A continued model codes named columns with its own lists: the cache
        // must have been built with exactly those (never re-code silently).
        if let Some(seed) = targets.categories {
            for f in 0..manifest.n_features {
                if let Some(c) = seed.get(f) {
                    crate::ensure_input!(
                        cached_categories.as_ref().and_then(|t| t.get(f)) == Some(c),
                        "the cached category names of feature {f} differ from the existing \
                         model's; use another cache_path or rebuild the cache"
                    );
                }
            }
        }
        let cuts = cuts.with_categories(cached_categories.map(std::sync::Arc::new));
        anyhow::ensure!(
            cuts_hash(&cuts) == manifest.cuts_hash,
            "cache cuts hash differs"
        );
        cuts.check_widths()?;
        let expected_groups =
            crate::columns::partition_features_by_width(&cuts.widths(), cols_per_block);
        anyhow::ensure!(
            manifest.groups == expected_groups,
            "cache column block map is corrupt: groups do not cover all features by cols_per_block"
        );
        // Block widths are derived from the cuts; the manifest must agree
        // (version 2 = every block 8-bit, version 3 = recorded per block).
        let expected_widths: Vec<BinWidth> = manifest
            .groups
            .iter()
            .map(|g| cuts.block_width(g))
            .collect();
        let recorded_widths: Vec<BinWidth> = if manifest.version == CACHE_VERSION {
            vec![BinWidth::U8; manifest.groups.len()]
        } else {
            manifest
                .block_widths
                .iter()
                .map(|&b| {
                    BinWidth::from_bytes(b).ok_or_else(|| {
                        anyhow::anyhow!("unsupported cache block width of {b} bytes")
                    })
                })
                .collect::<anyhow::Result<_>>()?
        };
        anyhow::ensure!(
            recorded_widths == expected_widths,
            "cache block bin widths do not match the cuts (manifest version {})",
            manifest.version
        );

        let mut labels = Vec::new();
        for batch in batch_reader(&path, &[label_col], BATCH_SIZE)? {
            labels.extend(crate::source::arrow_source::column_as_f32(&batch?, 0)?);
        }
        anyhow::ensure!(labels.len() == manifest.n_rows, "cache row count differs");
        for (i, group) in manifest.groups.iter().enumerate() {
            let expected = group
                .len()
                .saturating_mul(manifest.n_rows)
                .saturating_mul(expected_widths[i].bytes()) as u64;
            let actual = std::fs::metadata(cache_path.join(format!("block-{i}.bin")))?.len();
            anyhow::ensure!(actual == expected, "cache block {i} length differs");
        }
        let cache_bytes = quantized_bytes(&cuts, manifest.n_rows)?;
        Ok((
            Self {
                path,
                source_paths: None,
                cuts,
                groups: manifest.groups,
                file_col: feat_cols,
                multi_file_cols: None,
                n_rows: manifest.n_rows,
                n_features: manifest.n_features,
                cache: QuantizedCache::Disk {
                    dir: cache_path.to_path_buf(),
                    delete_on_drop: false,
                    verified: manifest
                        .block_hashes
                        .iter()
                        .map(|_| AtomicBool::new(false))
                        .collect(),
                    block_hashes: manifest.block_hashes,
                },
                cache_bytes,
                cache_reads: AtomicUsize::new(0),
                source_bytes: std::sync::atomic::AtomicU64::new(0),
                parquet_loads: AtomicUsize::new(0),
                missing: targets.missing,
            },
            labels,
        ))
    }
}

static TEMP_CACHE_ID: AtomicUsize = AtomicUsize::new(0);

fn describe_missing(bits: u32) -> String {
    let value = f32::from_bits(bits);
    if value.is_nan() {
        "NaN".to_string()
    } else {
        value.to_string()
    }
}

fn default_cache_budget() -> usize {
    let available_kib = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("MemAvailable:")?
                    .split_whitespace()
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
        });
    available_kib
        .and_then(|v| v.checked_mul(1024))
        .map(|v| v / 2)
        .unwrap_or(4 * 1024 * 1024 * 1024usize)
}

fn file_identities(paths: &[PathBuf]) -> anyhow::Result<Vec<SourceFileIdentity>> {
    paths
        .iter()
        .map(|p| {
            let (path, len, mtime_ns) = source_identity(p)?;
            Ok(SourceFileIdentity {
                path,
                len,
                mtime_ns,
            })
        })
        .collect()
}

fn source_identity(path: &Path) -> anyhow::Result<(String, u64, u64)> {
    let canonical = std::fs::canonicalize(path)?;
    let meta = std::fs::metadata(&canonical)?;
    let mtime = meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    Ok((canonical.to_string_lossy().into_owned(), meta.len(), mtime))
}

fn cuts_hash(cuts: &BinCuts) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for value in &cuts.values {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
        }
    }
    for offset in &cuts.offsets {
        for byte in offset.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
        }
    }
    // Mixed in only when present, so numeric caches keep their old hash.
    for (f, &categorical) in cuts.categorical_flags().iter().enumerate() {
        if categorical {
            for byte in (f as u32).to_le_bytes() {
                hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
            }
        }
    }
    // Likewise category names: only caches of named categories mix them in.
    if let Some(t) = cuts.categories() {
        hash = t.hash_into(hash);
    }
    format!("{hash:016x}")
}

/// Columns `cols` (block-local, in order) of a column-major block.
fn gather_columns<B: BinCode>(data: &[B], cols: &[usize], n_rows: usize) -> Vec<B> {
    let mut out = Vec::with_capacity(cols.len() * n_rows);
    for &l in cols {
        out.extend_from_slice(&data[l * n_rows..(l + 1) * n_rows]);
    }
    out
}

fn block_hash(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

impl BlockSource for ParquetBlockSource {
    /// 只读块里选中的那几列。
    ///
    /// **spill cache 的块文件就是 `ColumnBlock.data` 原样落盘,而它是列主序**
    /// —— 所以局部特征 `l` 的那一列正好是文件里
    /// `[l * n_rows, (l+1) * n_rows)` 这段**连续**字节,可以直接 seek 过去读。
    /// 这是 `colsample` 能在 source 这一层真省字节的原因,不只是少传。
    ///
    /// ⚠️ **checksum 的取舍**:`block_hashes` 是整块 payload 的 BLAKE3,
    /// 部分读取无法验证它。所以这里的规则是:**没验过的块先整块读一次并
    /// 验证**(付一次全量代价,换掉「静默跳过校验」),之后同一个块才允许
    /// 部分读取。缓存损坏仍然会在第一次读到时被抓住。
    fn with_block_cols(
        &self,
        i: usize,
        cols: &[usize],
        f: &mut dyn FnMut(&ColumnBlock),
    ) -> anyhow::Result<()> {
        let feat_ids = self
            .groups
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("column block {i} does not exist"))?;
        // 全选(或空)就走原路,别为一个退化情况多写一条路径。
        if cols.is_empty() || cols.len() == feat_ids.len() {
            return self.with_block(i, f);
        }
        if matches!(self.cache, QuantizedCache::Memory(_)) {
            // A resident block can be borrowed directly.
            return self.with_block(i, f);
        }
        let data = self.read_disk_columns(i, cols)?;
        f(&ColumnBlock {
            feat_ids: cols.iter().map(|&l| feat_ids[l]).collect(),
            data,
            n_rows: self.n_rows,
        });
        Ok(())
    }

    /// 只有落盘(spill)的 cache 能真的少读:块文件是列主序,
    /// 一列就是一段连续字节。内存 cache 没有"读取"这件事。
    fn supports_column_selection(&self) -> bool {
        matches!(self.cache, QuantizedCache::Disk { .. })
    }

    fn source_bytes_read(&self) -> Option<u64> {
        Some(self.source_bytes.load(Ordering::Relaxed))
    }

    fn source_block_reads(&self) -> Option<u64> {
        Some(self.cache_reads.load(Ordering::Relaxed) as u64)
    }

    fn with_block(&self, i: usize, f: &mut dyn FnMut(&ColumnBlock)) -> anyhow::Result<()> {
        self.cache_reads.fetch_add(1, Ordering::Relaxed);
        match &self.cache {
            QuantizedCache::Memory(blocks) => {
                let block = blocks
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("column block {i} does not exist"))?;
                f(block);
            }
            QuantizedCache::Disk {
                dir,
                block_hashes,
                verified,
                ..
            } => {
                let feat_ids = self
                    .groups
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("column block {i} does not exist"))?
                    .clone();
                let width = self.cuts.block_width(&feat_ids);
                let mut file = File::open(dir.join(format!("block-{i}.bin")))?;
                anyhow::ensure!(
                    file.metadata()?.len() == (feat_ids.len() * self.n_rows * width.bytes()) as u64,
                    "cache block {i} length differs"
                );
                // Read straight into the block's typed cells (no byte-to-cell copy).
                let mut cells = BinData::missing(width, feat_ids.len() * self.n_rows);
                file.read_exact(cells.as_le_bytes_mut())?;
                let data = cells.as_le_bytes();
                // 整块读也如实计数 —— 只有把「读了多少」和「传了多少」
                // 分开记,才看得出省的到底是哪一段。
                self.source_bytes
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                let is_verified = verified
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("cache block {i} has no checksum state"))?;
                if !is_verified.load(Ordering::Acquire) {
                    let expected = block_hashes
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("cache block {i} has no checksum"))?;
                    let actual = block_hash(&data);
                    anyhow::ensure!(
                        &actual == expected,
                        "cache block {i} payload checksum mismatch: expected {expected}, got {actual}"
                    );
                    is_verified.store(true, Ordering::Release);
                }
                f(&ColumnBlock {
                    feat_ids,
                    data: cells,
                    n_rows: self.n_rows,
                });
            }
        }
        Ok(())
    }

    fn n_blocks(&self) -> usize {
        self.groups.len()
    }
    fn n_rows(&self) -> usize {
        self.n_rows
    }
    fn n_features(&self) -> usize {
        self.n_features
    }
    fn cuts(&self) -> &BinCuts {
        &self.cuts
    }

    /// 纯元数据,不碰文件。默认实现会加载整块来读这个,那对按需加载
    /// 是灾难 —— 建树前问一遍每块有哪些列,就等于把数据集读一遍。
    fn block_features(&self, i: usize) -> anyhow::Result<Vec<FeatId>> {
        self.groups.get(i).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "column block {i} does not exist ({} blocks)",
                self.groups.len()
            )
        })
    }

    fn feature_column(&self, feat: FeatId) -> anyhow::Result<BinData> {
        let (block_id, local) = self
            .groups
            .iter()
            .enumerate()
            .find_map(|(b, group)| group.iter().position(|&f| f == feat).map(|l| (b, l)))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "feature {feat} is out of range ({} columns)",
                    self.n_features
                )
            })?;
        // ⚠️ **选择性特征 I/O 必须同时覆盖直方图物化和 split-feature 取列。**
        // 这里以前走 `with_block` —— 为了一列把整块读回来,而 partition 每层
        // 对每个 split feature 都调一次。上游 colsample 省下的字节会在这条
        // 旁路上被重新读回去:实测 spill 上 source bytes 的稳态比例因此被钉在
        // 约 47%,而不是继续向采样比例收敛。
        //
        // spill cache 是列主序 raw payload,单列本来就是一段连续字节,
        // 所以直接复用 `with_block_cols` 的部分读路径(连同它的 checksum
        // 首触规则和字节计数)。不支持部分读的 source 会走默认整块回退,
        // 行为和以前一致。
        if self.supports_column_selection() && self.groups[block_id].len() > 1 {
            return self.read_disk_columns(block_id, &[local]);
        }
        self.with_block_ret(block_id, |block| block.column(local).to_owned())
    }

    /// 内存 cache 直接借出列,零拷贝;落盘 cache 返回 `None`,调用方走
    /// `feature_column` 的部分读(那条路本来就要读进一份新内存)。
    fn has_resident_columns(&self) -> bool {
        matches!(self.cache, QuantizedCache::Memory(_))
    }

    fn resident_feature_column(&self, feat: FeatId) -> Option<BinSlice<'_>> {
        let QuantizedCache::Memory(blocks) = &self.cache else {
            return None;
        };
        let (block_id, local) = self
            .groups
            .iter()
            .enumerate()
            .find_map(|(b, group)| group.iter().position(|&x| x == feat).map(|l| (b, l)))?;
        // 和 `with_block` 一样记一次缓存读取,遥测口径不因借用而变。
        self.cache_reads.fetch_add(1, Ordering::Relaxed);
        Some(blocks.get(block_id)?.column(local))
    }
}

impl ParquetBlockSource {
    fn read_disk_columns(&self, i: usize, cols: &[usize]) -> anyhow::Result<BinData> {
        use std::io::{Seek, SeekFrom};

        let QuantizedCache::Disk { dir, verified, .. } = &self.cache else {
            unreachable!("disk column reads require a spill cache");
        };
        let feat_ids = &self.groups[i];
        for &local in cols {
            anyhow::ensure!(
                local < feat_ids.len(),
                "column block {i} has no local feature {local}"
            );
        }
        let checked = verified
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("cache block {i} has no checksum state"))?
            .load(Ordering::Acquire);
        if !checked {
            // Verify the full payload on first touch and gather directly
            // from that buffer while with_block still owns it.
            return self.with_block_ret(i, |block| match &block.data {
                BinData::U8(data) => BinData::U8(gather_columns(data, cols, self.n_rows)),
                BinData::U16(data) => BinData::U16(gather_columns(data, cols, self.n_rows)),
            });
        }

        let width = self.cuts.block_width(feat_ids);
        let col_bytes = self.n_rows * width.bytes();
        let mut file = File::open(dir.join(format!("block-{i}.bin")))?;
        let mut data = BinData::missing(width, cols.len() * self.n_rows);
        for (slot, &local) in cols.iter().enumerate() {
            file.seek(SeekFrom::Start((local * col_bytes) as u64))?;
            file.read_exact(&mut data.as_le_bytes_mut()[slot * col_bytes..(slot + 1) * col_bytes])?;
        }
        self.cache_reads.fetch_add(1, Ordering::Relaxed);
        self.source_bytes
            .fetch_add(data.byte_len() as u64, Ordering::Relaxed);
        Ok(data)
    }

    /// 读若干列并量化,列主序写进 `out`。
    ///
    /// `block()` 和 `feature_column()` **共用这一份量化代码**。分成两份
    /// 写的话,两条路的 bin 迟早会飘 —— 而那会悄悄破坏位精确基线:
    /// 建直方图用的是块里的 bin,行重分区用的是单列的 bin,两者对不上
    /// 就等于训练和分区看到了不同的数据。
    fn read_into<B: BinCode>(
        &self,
        file_cols: &[usize],
        feat_ids: &[FeatId],
        n_rows: usize,
        out: &mut [B],
    ) -> anyhow::Result<()> {
        debug_assert_eq!(out.len(), feat_ids.len() * n_rows);
        let mut row_base = 0usize;
        let mut recoder =
            crate::source::cat_recode::Recoder::fixed(self.cuts.categories().map(|t| &**t));
        for batch in batch_reader(&self.path, file_cols, BATCH_SIZE)? {
            let batch = recoder.recode(batch?, |c| feat_ids.get(c).map(|&f| f as usize))?;
            let n = batch.num_rows();
            anyhow::ensure!(
                n <= n_rows - row_base,
                "Parquet has more than {n_rows} expected rows"
            );
            for (local, &f) in feat_ids.iter().enumerate() {
                let dst = &mut out[local * n_rows + row_base..][..n];
                let mut row = 0usize;
                crate::source::arrow_source::for_each_feature_f32(
                    &batch,
                    local,
                    self.missing,
                    |v| {
                        // null 和 NaN 都是缺失
                        dst[row] = quantize_value::<B>(&self.cuts, f, v);
                        row += 1;
                    },
                )?;
                debug_assert_eq!(row, n);
            }
            row_base += n;
        }
        anyhow::ensure!(
            row_base == n_rows,
            "read {row_base} rows, metadata says {n_rows}"
        );
        Ok(())
    }

    fn read_many_into<B: BinCode>(
        &self,
        file_cols: &[Vec<usize>],
        feat_ids: &[FeatId],
        n_rows: usize,
        out: &mut [B],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            file_cols.len() == self.source_paths.as_ref().map_or(0, Vec::len),
            "multi-file projection count differs from source file count"
        );
        let projections: Vec<Vec<usize>> = file_cols
            .iter()
            .map(|cols| feat_ids.iter().map(|&f| cols[f as usize]).collect())
            .collect();
        let paths = self
            .source_paths
            .as_ref()
            .expect("multi-file source paths")
            .clone();
        let mut row_base = 0usize;
        let mut recoder =
            crate::source::cat_recode::Recoder::fixed(self.cuts.categories().map(|t| &**t));
        for batch in dataset_reader(paths, projections, BATCH_SIZE, 1) {
            let batch = recoder.recode(batch?, |c| feat_ids.get(c).map(|&f| f as usize))?;
            let n = batch.num_rows();
            anyhow::ensure!(
                n <= n_rows - row_base,
                "Parquet has more than {n_rows} expected rows"
            );
            for (local, &f) in feat_ids.iter().enumerate() {
                let dst = &mut out[local * n_rows + row_base..][..n];
                let mut row = 0usize;
                crate::source::arrow_source::for_each_feature_f32(
                    &batch,
                    local,
                    self.missing,
                    |v| {
                        dst[row] = quantize_value::<B>(&self.cuts, f, v);
                        row += 1;
                    },
                )?;
                debug_assert_eq!(row, n);
            }
            row_base += n;
        }
        anyhow::ensure!(
            row_base == n_rows,
            "read {row_base} rows, metadata says {n_rows}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::DenseSource;
    use arrow::array::{
        Array, DictionaryArray, Float32Array, Float64Array, Int64Array, Int8Array, StringArray,
    };
    use arrow::datatypes::{DataType, Field, Int8Type, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    /// 每个测试写自己的文件:cargo 默认并行跑测试,共用一个路径会
    /// 互相删掉对方的文件 —— 单跑全过、一起跑就挂,最难查的那种。
    fn write_fixture(name: &str) -> PathBuf {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f0", DataType::Float32, false),
            Field::new("target", DataType::Float32, false),
            Field::new("f1", DataType::Float64, true),
            Field::new("f2", DataType::Int64, false),
        ]));
        let dir = std::env::temp_dir().join(format!("fb_parquet_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.parquet"));
        let file = File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        // 分两批写,顺带压一下 batch 边界那条路
        for (a, y, b, c) in [
            (
                vec![1.0f32, 2.0, 3.0],
                vec![0.0f32, 1.0, 0.0],
                vec![Some(10.0f64), None, Some(30.0)],
                vec![7i64, 8, 9],
            ),
            (
                vec![4.0, 5.0, 6.0],
                vec![1.0, 1.0, 0.0],
                vec![Some(40.0), Some(50.0), None],
                vec![1, 2, 3],
            ),
        ] {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Float32Array::from(a)),
                    Arc::new(Float32Array::from(y)),
                    Arc::new(Float64Array::from(b)),
                    Arc::new(Int64Array::from(c)),
                ],
            )
            .unwrap();
            w.write(&batch).unwrap();
        }
        w.close().unwrap();
        path
    }

    /// Parquet 进来的 bin 必须和同样数据走 DenseSource 的**逐位相同**。
    /// 这一条成立,阶段 1a 的位精确基线就自动覆盖了 Parquet 这条路。
    #[test]
    fn parquet_matches_dense_bin_for_bin() {
        let path = write_fixture("match_dense");
        let ds = open(&path, "target", 16, 2).unwrap();

        assert_eq!(ds.source.n_rows(), 6);
        assert_eq!(
            ds.source.n_features(),
            3,
            "the label column must not count as a feature"
        );
        assert_eq!(ds.feature_names, vec!["f0", "f1", "f2"]);
        assert_eq!(ds.labels, vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);

        let n = f32::NAN;
        let row_major: Vec<f32> = vec![
            1.0, 10.0, 7.0, //
            2.0, n, 8.0, //
            3.0, 30.0, 9.0, //
            4.0, 40.0, 1.0, //
            5.0, 50.0, 2.0, //
            6.0, n, 3.0,
        ];
        let dense = DenseSource::from_row_major(&row_major, 6, 3, 16, 2);
        for row in 0..6 {
            for f in 0..3u32 {
                assert_eq!(
                    ds.source.bin_at(row, f),
                    dense.bin_at(row, f),
                    "row {row} feature {f}"
                );
            }
        }
        std::fs::remove_file(&path).ok();
    }

    /// 端到端:从 Parquet 训出来的模型,预测值和从稠密矩阵训出来的
    /// **逐位相同**。
    ///
    /// 换输入层的全部风险就在这一条上。它成立,阶段 1a 那套位精确基线
    /// (拿 DenseSource 建的)就自动覆盖了 Parquet 这条路,训练那一半
    /// 不用重验。
    #[test]
    fn training_from_parquet_matches_training_from_dense() {
        use crate::comm::Local;
        use crate::train::{train, TrainConfig};
        use crate::types::{Objective, TrainParams};

        let path = write_fixture("train_e2e");
        let ds = open(&path, "target", 16, 2).unwrap();

        let n = f32::NAN;
        let row_major: Vec<f32> = vec![
            1.0, 10.0, 7.0, 2.0, n, 8.0, 3.0, 30.0, 9.0, 4.0, 40.0, 1.0, 5.0, 50.0, 2.0, 6.0, n,
            3.0,
        ];
        let dense = DenseSource::from_row_major(&row_major, 6, 3, 16, 2);

        let p = TrainParams {
            n_rounds: 5,
            max_depth: 3,
            max_bin: 16,
            min_child_weight: 0.0,
            cols_per_block: 2,
            nthread: 1,
            ..Default::default()
        };
        let cfg = TrainConfig::new(p, Objective::Logistic);

        let from_parquet = train(&ds.source, &ds.labels, &[], &cfg, &mut [], &Local).unwrap();
        let from_dense = train(&dense, &ds.labels, &[], &cfg, &mut [], &Local).unwrap();

        assert_eq!(from_parquet.trees.len(), from_dense.trees.len());
        for row in 0..6 {
            let x = &row_major[row * 3..(row + 1) * 3];
            assert_eq!(
                from_parquet.predict_margin(x).to_bits(),
                from_dense.predict_margin(x).to_bits(),
                "row {row}: Parquet-trained and Dense-trained results differ"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    /// 按需加载的块,内容必须和全量加载的**逐位相同**。
    /// 生命周期变了,数据没变。
    #[test]
    fn lazy_blocks_match_eager_blocks() {
        let path = write_fixture("lazy_eager");
        let eager = open(&path, "target", 16, 2).unwrap();
        let (lazy, labels) = ParquetBlockSource::open(&path, "target", 16, 2).unwrap();

        assert_eq!(lazy.n_rows(), eager.source.n_rows());
        assert_eq!(lazy.n_features(), eager.source.n_features());
        assert_eq!(lazy.n_blocks(), eager.source.n_blocks());
        assert_eq!(labels, eager.labels);
        for f in 0..lazy.n_features() as u32 {
            assert_eq!(lazy.cuts().cuts_for(f), eager.source.cuts().cuts_for(f));
        }
        for b in 0..lazy.n_blocks() {
            lazy.with_block(b, &mut |a| {
                eager
                    .source
                    .with_block(b, &mut |e| {
                        assert_eq!(a.feat_ids, e.feat_ids, "block {b} feature list");
                        assert_eq!(a.n_rows, e.n_rows);
                        assert_eq!(a.data, e.data, "block {b} bin data");
                    })
                    .unwrap();
            })
            .unwrap();
        }
    }

    /// 单列读取和整块读取必须给出**同一份 bin**。
    ///
    /// 这两条路分别喂给"建直方图"和"行重分区",一旦飘了就等于训练和
    /// 分区看到了不同的数据 —— 树还是长得出来,只是行走错边,而且
    /// 位精确基线在小数据上未必抓得到。
    /// 值域要铺开:负数、重复值、极值、null、正好落在切分点上的值。
    /// 漂移常常只在某些值域上出现(边界比较、NaN 处理、饱和),
    /// 拿一个 6 行的小样本测等于没测。
    fn write_wide_range(name: &str, n_rows: usize, n_cols: usize) -> PathBuf {
        let mut fields: Vec<Field> = (0..n_cols)
            .map(|i| Field::new(format!("f{i}"), DataType::Float64, true))
            .collect();
        fields.push(Field::new("target", DataType::Float32, false));
        let schema = Arc::new(Schema::new(fields));

        let dir = std::env::temp_dir().join(format!("fb_parquet_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.parquet"));
        let file = File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema.clone(), None).unwrap();

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u32
        };
        // 分两批写,顺带压 batch 边界
        for half in 0..2 {
            let mut cols: Vec<std::sync::Arc<dyn arrow::array::Array>> = Vec::new();
            for _ in 0..n_cols {
                let vals: Vec<Option<f64>> = (0..n_rows / 2)
                    .map(|r| {
                        let k = next() % 13;
                        match k {
                            0 => None, // null
                            1 => Some(0.0),
                            2 => Some(-0.0),
                            3 => Some(f64::MAX / 1e300),
                            4 => Some(-1e30),
                            5 => Some((r as f64) * -1.5), // 负数
                            6 => Some(7.0),               // 重复值
                            _ => Some(((next() % 1000) as f64 - 500.0) / 7.0),
                        }
                    })
                    .collect();
                cols.push(Arc::new(Float64Array::from(vals)));
            }
            let y: Vec<f32> = (0..n_rows / 2).map(|r| ((r + half) % 2) as f32).collect();
            cols.push(Arc::new(Float32Array::from(y)));
            w.write(&RecordBatch::try_new(schema.clone(), cols).unwrap())
                .unwrap();
        }
        w.close().unwrap();
        path
    }

    fn rewrite_with_row_group_size(source: &Path, name: &str, max_rows: usize) -> PathBuf {
        let metadata =
            ArrowReaderMetadata::load(&File::open(source).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        let schema = metadata.schema().clone();
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(source).unwrap())
            .unwrap()
            .with_batch_size(BATCH_SIZE)
            .build()
            .unwrap();
        let dir = std::env::temp_dir().join(format!("fb_parquet_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.parquet"));
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(max_rows))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(properties)).unwrap();
        for batch in reader {
            writer.write(&batch.unwrap()).unwrap();
        }
        writer.close().unwrap();
        path
    }

    fn batch_rows_as_bits(batches: &[RecordBatch]) -> Vec<Vec<Option<u64>>> {
        let mut columns: Vec<Vec<Option<u64>>> = (0..256).map(|_| Vec::new()).collect();
        for batch in batches {
            for (column, output) in columns.iter_mut().enumerate() {
                let values = batch
                    .column(column)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                output.extend(
                    (0..values.len())
                        .map(|row| (!values.is_null(row)).then(|| values.value(row).to_bits())),
                );
            }
        }
        columns
    }

    fn assert_cuts_bitwise_equal(left: &BinCuts, right: &BinCuts, context: &str) {
        assert_eq!(left.offsets, right.offsets, "{context}: cut offsets");
        assert_eq!(
            left.values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{context}: cut value bits"
        );
    }

    fn write_wide_f32_range(
        name: &str,
        n_rows: usize,
        n_cols: usize,
        dictionary_column: Option<usize>,
    ) -> PathBuf {
        // Keep the bounded wide fixture fixed-width, while allowing one
        // dictionary column for the ordinary-reader fallback test.
        let mut fields: Vec<_> = (0..n_cols)
            .map(|i| {
                let data_type = if Some(i) == dictionary_column {
                    DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8))
                } else {
                    DataType::Float32
                };
                Field::new(format!("f{i}"), data_type, true)
            })
            .collect();
        fields.push(Field::new("target", DataType::Float32, false));
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<std::sync::Arc<dyn arrow::array::Array>> = (0..n_cols)
            .map(|column| {
                if Some(column) == dictionary_column {
                    let keys = (0..n_rows)
                        .map(|row| (row % 11 != 0).then_some((row % 3) as i8))
                        .collect::<Vec<_>>();
                    return Arc::new(
                        DictionaryArray::<Int8Type>::try_new(
                            Int8Array::from(keys),
                            Arc::new(StringArray::from(vec!["red", "green", "blue"])),
                        )
                        .unwrap(),
                    ) as std::sync::Arc<dyn arrow::array::Array>;
                }
                let values = (0..n_rows)
                    .map(|row| match (row * 31 + column * 17) % 19 {
                        0 => None,
                        1 => Some(-0.0),
                        2 => Some(0.0),
                        value => Some((value as f32 - 9.0) / 4.0),
                    })
                    .collect::<Vec<_>>();
                Arc::new(Float32Array::from(values)) as std::sync::Arc<dyn arrow::array::Array>
            })
            .chain(std::iter::once(Arc::new(Float32Array::from(
                (0..n_rows).map(|row| (row % 2) as f32).collect::<Vec<_>>(),
            ))
                as std::sync::Arc<dyn arrow::array::Array>))
            .collect();
        let dir = std::env::temp_dir().join(format!("fb_parquet_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.parquet"));
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema.clone(), None).unwrap();
        writer
            .write(&RecordBatch::try_new(schema, arrays).unwrap())
            .unwrap();
        writer.close().unwrap();
        path
    }

    fn assert_f32_batches_bitwise_equal(actual: &[RecordBatch], expected: &[RecordBatch]) {
        assert_eq!(actual.len(), expected.len(), "batch count");
        for (batch_index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.schema().as_ref(),
                expected.schema().as_ref(),
                "batch {batch_index} schema"
            );
            assert_eq!(
                actual.num_rows(),
                expected.num_rows(),
                "batch {batch_index} row count"
            );
            assert_eq!(actual.num_columns(), expected.num_columns());
            for column in 0..actual.num_columns() {
                if let (Some(actual_values), Some(expected_values)) = (
                    actual
                        .column(column)
                        .as_any()
                        .downcast_ref::<Float32Array>(),
                    expected
                        .column(column)
                        .as_any()
                        .downcast_ref::<Float32Array>(),
                ) {
                    assert_eq!(actual_values.len(), expected_values.len());
                    for row in 0..actual_values.len() {
                        let actual_bits = (!actual_values.is_null(row))
                            .then(|| actual_values.value(row).to_bits());
                        let expected_bits = (!expected_values.is_null(row))
                            .then(|| expected_values.value(row).to_bits());
                        assert_eq!(
                            actual_bits, expected_bits,
                            "batch {batch_index}, column {column}, row {row}"
                        );
                    }
                } else {
                    let actual_dictionary = actual
                        .column(column)
                        .as_any()
                        .downcast_ref::<DictionaryArray<Int8Type>>()
                        .unwrap();
                    let expected_dictionary = expected
                        .column(column)
                        .as_any()
                        .downcast_ref::<DictionaryArray<Int8Type>>()
                        .unwrap();
                    assert_eq!(actual_dictionary.keys(), expected_dictionary.keys());
                    let actual_values = actual_dictionary
                        .values()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap();
                    let expected_values = expected_dictionary
                        .values()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap();
                    assert_eq!(actual_values, expected_values);
                }
            }
        }
    }

    fn assert_quantized_sources_bitwise_equal(
        left: &ParquetBlockSource,
        right: &ParquetBlockSource,
        context: &str,
    ) {
        assert_eq!(left.n_rows(), right.n_rows(), "{context}: row count");
        assert_eq!(
            left.n_features(),
            right.n_features(),
            "{context}: feature count"
        );
        assert_eq!(left.n_blocks(), right.n_blocks(), "{context}: block count");
        assert_cuts_bitwise_equal(left.cuts(), right.cuts(), context);
        for block in 0..left.n_blocks() {
            left.with_block(block, &mut |left_block| {
                right
                    .with_block(block, &mut |right_block| {
                        assert_eq!(left_block.feat_ids, right_block.feat_ids);
                        assert_eq!(left_block.n_rows, right_block.n_rows);
                        assert_eq!(
                            left_block.data, right_block.data,
                            "{context}: block {block}"
                        );
                    })
                    .unwrap();
            })
            .unwrap();
        }
    }

    #[test]
    fn wide_coalesced_groups_match_ordinary_f32_batches_in_requested_order() {
        // Widths 257 and 513 force 256+1 and 256+256+1 decoder chunks.
        // The latter also spans several short row groups and a short final group.
        for (n_cols, row_group_size, batch_size) in [(257usize, None, 13usize), (513, Some(16), 7)]
        {
            let initial =
                write_wide_f32_range(&format!("coalesced_f32_{n_cols}_source"), 50, n_cols, None);
            let path = if let Some(row_group_size) = row_group_size {
                let path = rewrite_with_row_group_size(
                    &initial,
                    &format!("coalesced_f32_{n_cols}_groups"),
                    row_group_size,
                );
                std::fs::remove_file(initial).unwrap();
                path
            } else {
                initial
            };
            let metadata =
                ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                    .unwrap();
            let row_groups = metadata.metadata().row_groups();
            assert_eq!(
                row_groups
                    .iter()
                    .map(|group| group.num_rows())
                    .collect::<Vec<_>>(),
                if n_cols == 257 {
                    vec![50]
                } else {
                    vec![16, 16, 16, 2]
                }
            );

            let features: Vec<_> = (0..n_cols).collect();
            let mut reversed = features.clone();
            reversed.reverse();
            let rotated = (37..n_cols).chain(0..37).collect::<Vec<_>>();
            for columns in [features, reversed, rotated] {
                assert!(coalesced_projection(&metadata, &columns));
                assert!(coalesced_fixed_width_projection(&metadata, &columns));

                // The expected arrays come from independent ordinary File
                // readers, each explicitly scoped to one row group.
                let mut expected = Vec::new();
                let mut expected_group_rows = Vec::new();
                for row_group in 0..row_groups.len() {
                    let ordinary = build_projected_reader(
                        File::open(&path).unwrap(),
                        metadata.clone(),
                        &columns,
                        batch_size,
                        Some(row_group),
                    )
                    .unwrap();
                    let batches = ordinary.map(Result::unwrap).collect::<Vec<_>>();
                    expected_group_rows.push(
                        batches
                            .iter()
                            .map(RecordBatch::num_rows)
                            .collect::<Vec<_>>(),
                    );
                    expected.extend(batches);
                }

                let input = crate::source::parquet_io::CoalescedFile::test_with_capacity(
                    File::open(&path).unwrap(),
                    4096,
                )
                .unwrap();
                let probe = input.clone();
                let mut reader =
                    coalesced_row_group_reader(input, metadata.clone(), &columns, batch_size)
                        .unwrap();
                assert_eq!(probe.test_backing_bytes(), 0, "construction must be lazy");
                let actual = reader.by_ref().map(Result::unwrap).collect::<Vec<_>>();
                drop(reader);
                let backing_bytes = probe.test_backing_bytes();
                drop(probe);
                assert!(backing_bytes > 0, "iteration should read the file");
                assert_f32_batches_bitwise_equal(&actual, &expected);

                let ordinary_group_rows = row_groups
                    .iter()
                    .map(|group| {
                        let rows = group.num_rows() as usize;
                        (0..rows)
                            .step_by(batch_size)
                            .map(|start| (rows - start).min(batch_size))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(ordinary_group_rows, expected_group_rows);
                let actual_batch_rows =
                    actual.iter().map(RecordBatch::num_rows).collect::<Vec<_>>();
                assert_eq!(
                    actual_batch_rows,
                    expected_group_rows
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    actual_batch_rows.len(),
                    expected_group_rows
                        .iter()
                        .map(|rows| rows.len())
                        .sum::<usize>()
                );
                assert!(
                    actual_batch_rows.iter().any(|&rows| rows < batch_size),
                    "fixture must include a partial batch tail"
                );
            }
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn wide_coalesced_group_is_lazy_and_truncation_yields_one_error_without_batches() {
        let path = write_wide_f32_range("coalesced_f32_truncated", 50, 257, None);
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        let group = metadata.metadata().row_group(0);
        // Preserve the first 256-column decode, then truncate the one-column
        // tail chunk so assembly fails after earlier chunks have succeeded.
        let failing_chunk_page = group.columns()[256]
            .dictionary_page_offset()
            .unwrap_or_else(|| group.columns()[256].data_page_offset());
        let input = crate::source::parquet_io::CoalescedFile::test_with_capacity(
            File::open(&path).unwrap(),
            4096,
        )
        .unwrap();
        let probe = input.clone();
        let columns: Vec<_> = (0..257).collect();
        let mut reader = bounded_coalesced_group(input, metadata, &columns, 13, 0).unwrap();
        assert_eq!(probe.test_backing_bytes(), 0, "construction must be lazy");

        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(failing_chunk_page as u64)
            .unwrap();
        let first = reader.next().expect("truncated group must yield an error");
        assert!(first.is_err(), "no partially decoded group may be yielded");
        assert!(reader.next().is_none(), "decode error must be yielded once");
        assert!(
            reader.next().is_none(),
            "reader must stay fused after its error"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn wide_dictionary_projection_uses_ordinary_reader_batches() {
        let path = write_wide_f32_range("coalesced_dictionary_fallback", 50, 257, Some(256));
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        let columns = (0..257).rev().collect::<Vec<_>>();
        assert!(coalesced_projection(&metadata, &columns));
        assert!(
            !coalesced_fixed_width_projection(&metadata, &columns),
            "dictionary/Utf8 columns must use the original reader"
        );

        let expected = build_projected_reader(
            File::open(&path).unwrap(),
            metadata.clone(),
            &columns,
            11,
            None,
        )
        .unwrap()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
        let actual = batch_reader_with_metadata(File::open(&path).unwrap(), metadata, &columns, 11)
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert!(actual.len() > 1, "fixture must exercise multiple batches");
        assert!(actual.last().unwrap().num_rows() < 11, "partial tail batch");
        assert_f32_batches_bitwise_equal(&actual, &expected);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn wide_coalesced_routing_is_bitwise_invariant_across_thread_counts() {
        let initial = write_wide_f32_range("coalesced_f32_threads_source", 50, 513, None);
        let path = rewrite_with_row_group_size(&initial, "coalesced_f32_threads_groups", 16);
        std::fs::remove_file(initial).unwrap();
        let mut one_cut_readers = None;
        let (one, labels_one) = ParquetBlockSource::open_with_cache_observed(
            &path,
            "target",
            255,
            300,
            QuantizedCacheOptions::default(),
            1,
            |phase, readers, _| {
                if phase == "cuts" {
                    one_cut_readers = Some(readers);
                }
            },
        )
        .unwrap();
        assert_eq!(one_cut_readers, Some(1));

        let mut parallel_cut_readers = None;
        let (parallel, labels_parallel) = ParquetBlockSource::open_with_cache_observed(
            &path,
            "target",
            255,
            300,
            QuantizedCacheOptions::default(),
            4,
            |phase, readers, _| {
                if phase == "cuts" {
                    parallel_cut_readers = Some(readers);
                }
            },
        )
        .unwrap();
        assert_eq!(parallel_cut_readers, Some(2));
        assert_eq!(labels_one, labels_parallel, "nthread=4");
        assert_quantized_sources_bitwise_equal(&one, &parallel, "nthread 1 vs 4");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn coalesced_small_row_groups_bound_reads_and_preserve_source_values() {
        let initial = write_wide_range("coalesced_groups_source", 64, 256);
        let path = rewrite_with_row_group_size(&initial, "coalesced_groups_16", 16);
        std::fs::remove_file(initial).ok();

        let file = File::open(&path).unwrap();
        let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).unwrap();
        assert_eq!(metadata.metadata().num_row_groups(), 4);
        assert!(metadata
            .metadata()
            .row_groups()
            .iter()
            .all(|group| group.num_rows() == 16));
        let features: Vec<_> = (0..256).collect();
        assert!(coalesced_projection(&metadata, &features));

        // Reproduce the old multi-group access through one shared bounded
        // cache, then compare it with the new one-row-group-at-a-time reader.
        let baseline_input = crate::source::parquet_io::CoalescedFile::test_with_capacity(
            File::open(&path).unwrap(),
            4096,
        )
        .unwrap();
        let baseline_probe = baseline_input.clone();
        let baseline =
            build_projected_reader(baseline_input, metadata.clone(), &features, 32, None).unwrap();
        let baseline_batches: Vec<_> = baseline.map(Result::unwrap).collect();

        let scoped_input = crate::source::parquet_io::CoalescedFile::test_with_capacity(
            File::open(&path).unwrap(),
            4096,
        )
        .unwrap();
        let scoped_probe = scoped_input.clone();
        let scoped =
            coalesced_row_group_reader(scoped_input, metadata.clone(), &features, 32).unwrap();
        let scoped_batches: Vec<_> = scoped.map(Result::unwrap).collect();

        let ordinary = build_projected_reader(
            File::open(&path).unwrap(),
            metadata.clone(),
            &features,
            32,
            None,
        )
        .unwrap();
        let ordinary_batches: Vec<_> = ordinary.map(Result::unwrap).collect();
        assert_eq!(
            baseline_batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![32, 32]
        );
        assert_eq!(
            scoped_batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![16, 16, 16, 16]
        );
        assert_eq!(
            ordinary_batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![32, 32]
        );
        let expected_rows = batch_rows_as_bits(&ordinary_batches);
        assert_eq!(batch_rows_as_bits(&baseline_batches), expected_rows);
        assert_eq!(batch_rows_as_bits(&scoped_batches), expected_rows);

        let file_bytes = std::fs::metadata(&path).unwrap().len();
        let baseline_reads = baseline_probe.test_backing_bytes();
        let scoped_reads = scoped_probe.test_backing_bytes();
        assert!(
            scoped_reads <= file_bytes + 2 * 4096,
            "scoped reader read {scoped_reads} backing bytes for {file_bytes} file bytes"
        );
        assert!(
            baseline_reads > scoped_reads * 4,
            "unscoped reads {baseline_reads} bytes, scoped reads {scoped_reads} bytes"
        );

        // A batch smaller than one row group must also preserve row order
        // across the scoped-reader handoff and within each row group.
        let baseline_small_input = crate::source::parquet_io::CoalescedFile::test_with_capacity(
            File::open(&path).unwrap(),
            4096,
        )
        .unwrap();
        let baseline_small =
            build_projected_reader(baseline_small_input, metadata.clone(), &features, 8, None)
                .unwrap();
        let baseline_small_batches: Vec<_> = baseline_small.map(Result::unwrap).collect();
        let scoped_small_input = crate::source::parquet_io::CoalescedFile::test_with_capacity(
            File::open(&path).unwrap(),
            4096,
        )
        .unwrap();
        let scoped_small =
            coalesced_row_group_reader(scoped_small_input, metadata.clone(), &features, 8).unwrap();
        let scoped_small_batches: Vec<_> = scoped_small.map(Result::unwrap).collect();
        let ordinary_small = build_projected_reader(
            File::open(&path).unwrap(),
            metadata.clone(),
            &features,
            8,
            None,
        )
        .unwrap();
        let ordinary_small_batches: Vec<_> = ordinary_small.map(Result::unwrap).collect();
        assert_eq!(
            baseline_small_batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![8; 8]
        );
        assert_eq!(
            scoped_small_batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![8; 8]
        );
        assert_eq!(
            ordinary_small_batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![8; 8]
        );
        assert_eq!(
            batch_rows_as_bits(&baseline_small_batches),
            batch_rows_as_bits(&ordinary_small_batches)
        );
        assert_eq!(
            batch_rows_as_bits(&scoped_small_batches),
            batch_rows_as_bits(&ordinary_small_batches)
        );

        // The public path and row-group parallel path must produce the same
        // max-bin sketch as the ordinary File reader at either thread count.
        let plain_cuts = crate::source::arrow_source::build_cuts_with_nthread(
            Box::new(
                build_projected_reader(
                    File::open(&path).unwrap(),
                    metadata.clone(),
                    &features,
                    BATCH_SIZE,
                    None,
                )
                .unwrap()
                .map(|batch| batch.map_err(anyhow::Error::from)),
            ),
            255,
            1,
        )
        .unwrap();
        for nthread in [1, 2] {
            let routed_cuts = crate::source::arrow_source::build_cuts_with_nthread(
                parallel_row_groups(path.clone(), features.clone(), BATCH_SIZE, nthread).unwrap(),
                255,
                nthread,
            )
            .unwrap();
            assert_cuts_bitwise_equal(&plain_cuts, &routed_cuts, &format!("nthread={nthread}"));
        }

        let (one, labels_one) =
            ParquetBlockSource::open_with_nthread(&path, "target", 255, 256, 1).unwrap();
        let (two, labels_two) =
            ParquetBlockSource::open_with_nthread(&path, "target", 255, 256, 2).unwrap();
        assert_eq!(labels_one, labels_two);
        assert_cuts_bitwise_equal(one.cuts(), two.cuts(), "quantization nthread 1 vs 2");
        assert_cuts_bitwise_equal(&plain_cuts, one.cuts(), "plain vs quantized cuts");
        for block in 0..one.n_blocks() {
            one.with_block(block, &mut |left| {
                two.with_block(block, &mut |right| {
                    assert_eq!(left.feat_ids, right.feat_ids);
                    assert_eq!(left.n_rows, right.n_rows);
                    assert_eq!(left.data, right.data, "quantized block {block}");
                })
                .unwrap();
            })
            .unwrap();
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn coalesced_reader_preserves_a_short_last_row_group() {
        let initial = write_wide_range("coalesced_short_source", 50, 256);
        let path = rewrite_with_row_group_size(&initial, "coalesced_short_16", 16);
        std::fs::remove_file(initial).ok();
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        assert_eq!(
            metadata
                .metadata()
                .row_groups()
                .iter()
                .map(|group| group.num_rows())
                .collect::<Vec<_>>(),
            vec![16, 16, 16, 2]
        );
        let features: Vec<_> = (0..256).collect();
        let scoped = coalesced_row_group_reader(
            crate::source::parquet_io::CoalescedFile::test_with_capacity(
                File::open(&path).unwrap(),
                4096,
            )
            .unwrap(),
            metadata.clone(),
            &features,
            32,
        )
        .unwrap();
        let scoped_batches: Vec<_> = scoped.map(Result::unwrap).collect();
        let ordinary =
            build_projected_reader(File::open(&path).unwrap(), metadata, &features, 32, None)
                .unwrap();
        let ordinary_batches: Vec<_> = ordinary.map(Result::unwrap).collect();
        assert_eq!(
            scoped_batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![16, 16, 16, 2]
        );
        assert_eq!(
            batch_rows_as_bits(&scoped_batches),
            batch_rows_as_bits(&ordinary_batches)
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn empty_wide_parquet_uses_the_ordinary_reader_path() {
        let path = write_wide_range("coalesced_empty", 0, 256);
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        let features: Vec<_> = (0..256).collect();
        assert!(!coalesced_projection(&metadata, &features));
        let batches: Vec<_> = batch_reader(&path, &features, BATCH_SIZE)
            .unwrap()
            .collect::<anyhow::Result<Vec<_>>>()
            .unwrap();
        assert!(batches.is_empty());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn coalesced_row_group_reader_stops_after_first_decode_error() {
        let initial = write_wide_range("coalesced_error_source", 64, 256);
        let path = rewrite_with_row_group_size(&initial, "coalesced_error_16", 16);
        std::fs::remove_file(initial).ok();
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        let group_two_first_data_page =
            metadata.metadata().row_groups()[2].columns()[0].data_page_offset();
        let input = crate::source::parquet_io::CoalescedFile::test_with_capacity(
            File::open(&path).unwrap(),
            4096,
        )
        .unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(group_two_first_data_page as u64)
            .unwrap();

        let features: Vec<_> = (0..256).collect();
        let mut reader = coalesced_row_group_reader(input, metadata, &features, 32).unwrap();
        let mut batch_count = 0;
        let mut error_count = 0;
        for result in reader.by_ref() {
            match result {
                Ok(_) => batch_count += 1,
                Err(_) => error_count += 1,
            }
        }
        assert!(
            batch_count > 0,
            "earlier row groups should decode before truncation"
        );
        assert_eq!(
            error_count, 1,
            "the first decode error must terminate the stream"
        );
        assert!(
            reader.next().is_none(),
            "reader must stay fused after its error"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn coalesced_projection_routes_only_dense_small_groups() {
        let path = write_wide_range("coalesced_routing", 64, 256);
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        let features: Vec<_> = (0..256).collect();
        assert!(coalesced_projection(&metadata, &features));
        assert!(!coalesced_projection(&metadata, &features[..255]));
        assert!(!coalesced_projection(&metadata, &[256]));
        let mut duplicates = features.clone();
        duplicates[255] = 0;
        assert!(!coalesced_projection(&metadata, &duplicates));
        let mut invalid = features.clone();
        invalid[255] = 257;
        assert!(!coalesced_projection(&metadata, &invalid));
        // Compare actual decoded arrays with the ordinary File reader, not
        // merely two thread counts that could both use the same cached path.
        let ordinary = build_projected_reader(
            File::open(&path).unwrap(),
            metadata.clone(),
            &features,
            17,
            None,
        )
        .unwrap();
        let cached = batch_reader(&path, &features, 17).unwrap();
        let a: Vec<_> = ordinary.map(Result::unwrap).collect();
        let b: Vec<_> = cached.map(Result::unwrap).collect();
        assert_eq!(a, b);
        std::fs::remove_file(path).unwrap();

        let path = write_wide_range("coalesced_sparse", 64, 512);
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        assert!(!coalesced_projection(&metadata, &features));
        std::fs::remove_file(path).unwrap();

        let path = write_wide_range("coalesced_long_group", 2 * BATCH_SIZE + 2, 256);
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        assert!(!coalesced_projection(&metadata, &features));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn coalesced_projection_rejects_skewed_omitted_bytes() {
        use arrow::array::{ArrayRef, BinaryArray};
        let mut fields: Vec<_> = (0..256)
            .map(|i| Field::new(format!("f{i}"), DataType::Float32, false))
            .collect();
        fields.push(Field::new("target", DataType::Binary, false));
        let schema = Arc::new(Schema::new(fields));
        let mut arrays: Vec<ArrayRef> = (0..256)
            .map(|_| Arc::new(Float32Array::from(vec![0.0; 64])) as ArrayRef)
            .collect();
        let label: Vec<Vec<u8>> = (0..64)
            .map(|row| (0..4096).map(|i| ((i * 17 + row) % 251) as u8).collect())
            .collect();
        arrays.push(Arc::new(BinaryArray::from_iter_values(
            label.iter().map(Vec::as_slice),
        )));
        let path =
            std::env::temp_dir().join(format!("fb-coalesced-skew-{}.parquet", std::process::id()));
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema.clone(), None).unwrap();
        writer
            .write(&RecordBatch::try_new(schema, arrays).unwrap())
            .unwrap();
        writer.close().unwrap();
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        assert!(!coalesced_projection(
            &metadata,
            &(0..256).collect::<Vec<_>>()
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn coalesced_projection_rejects_one_leaf_nested_root() {
        use arrow::array::{ArrayRef, StructArray};
        let nested_field = Arc::new(Field::new("child", DataType::Float32, false));
        let mut fields: Vec<_> = (0..256)
            .map(|i| Field::new(format!("f{i}"), DataType::Float32, false))
            .collect();
        fields.push(Field::new(
            "target",
            DataType::Struct(vec![nested_field.clone()].into()),
            false,
        ));
        let schema = Arc::new(Schema::new(fields));
        let mut arrays: Vec<ArrayRef> = (0..256)
            .map(|_| Arc::new(Float32Array::from(vec![0.0; 64])) as ArrayRef)
            .collect();
        arrays.push(Arc::new(StructArray::from(vec![(
            nested_field,
            Arc::new(Float32Array::from(vec![0.0; 64])) as ArrayRef,
        )])));
        let path = std::env::temp_dir().join(format!(
            "fb-coalesced-nested-{}.parquet",
            std::process::id()
        ));
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema.clone(), None).unwrap();
        writer
            .write(&RecordBatch::try_new(schema, arrays).unwrap())
            .unwrap();
        writer.close().unwrap();
        let metadata =
            ArrowReaderMetadata::load(&File::open(&path).unwrap(), ArrowReaderOptions::new())
                .unwrap();
        assert_eq!(
            metadata
                .metadata()
                .file_metadata()
                .schema_descr()
                .num_columns(),
            metadata.schema().fields().len()
        );
        assert!(!coalesced_projection(
            &metadata,
            &(0..256).collect::<Vec<_>>()
        ));
        std::fs::remove_file(path).unwrap();
    }

    /// 单列读取和整块读取必须给出**逐字节相同**的 bin。
    ///
    /// 共用一个 helper 只保证"现在"一致,不保证将来不分叉 —— 比如有人
    /// 给块级路径加个批量量化的快速路径,单列路径没跟上。那时两条路
    /// 各自看着都对,但不再一致,而位精确基线未必抓得到(它只在特定
    /// 数据上跑,漂移可能只在某些值域出现)。这条测试盯的就是这个契约。
    #[test]
    fn single_column_read_matches_the_block_byte_for_byte() {
        for (rows, cols, cpb) in [(6usize, 3usize, 2usize), (400, 9, 4), (400, 9, 1)] {
            let path = write_wide_range(&format!("col_vs_block_{rows}_{cols}_{cpb}"), rows, cols);
            let (src, _) = ParquetBlockSource::open(&path, "target", 32, cpb).unwrap();
            let mut checked = 0;
            for b in 0..src.n_blocks() {
                src.with_block(b, &mut |block| {
                    for (local, &f) in block.feat_ids.iter().enumerate() {
                        assert_eq!(
                            src.feature_column(f).unwrap().as_slice(),
                            block.column(local),
                            "{rows} rows {cols} cols cpb={cpb}, feature {f}: column read and block read differ"
                        );
                        // 内存 cache 借出的列同样必须逐字节一致(落盘 cache 返回 None)。
                        if let Some(borrowed) = src.resident_feature_column(f) {
                            assert_eq!(borrowed, block.column(local), "feature {f}: borrowed column differs");
                        }
                        checked += 1;
                    }
                })
                .unwrap();
            }
            assert_eq!(checked, cols, "every column must be compared");
            std::fs::remove_file(&path).ok();
        }
    }

    /// cuts、量化缓存和最终模型都不能随 cuts sketch 的线程数变化。
    #[test]
    fn parquet_cut_threads_are_byte_identical_end_to_end() {
        use crate::comm::Local;
        use crate::train::{train, TrainConfig};
        use crate::types::{Objective, TrainParams};

        let path = write_wide_range("cut_threads", 1024, 9);
        let (one, labels_one) =
            ParquetBlockSource::open_with_nthread(&path, "target", 32, 4, 1).unwrap();
        let (eight, labels_eight) =
            ParquetBlockSource::open_with_nthread(&path, "target", 32, 4, 8).unwrap();
        let (rayon_default, labels_default) =
            ParquetBlockSource::open_with_nthread(&path, "target", 32, 4, 0).unwrap();

        assert_eq!(labels_one, labels_eight);
        assert_eq!(labels_one, labels_default);
        assert_eq!(one.cuts().offsets, eight.cuts().offsets);
        assert_eq!(
            one.cuts()
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            eight
                .cuts()
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(one.cuts().offsets, rayon_default.cuts().offsets);
        assert_eq!(
            one.cuts()
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            rayon_default
                .cuts()
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        for (threads, src) in [(1, &one), (8, &eight), (0, &rayon_default)] {
            assert_eq!(
                src.parquet_load_count(),
                src.n_blocks(),
                "nthread={threads}: each block must be quantized exactly once"
            );
        }
        for b in 0..one.n_blocks() {
            one.with_block(b, &mut |a| {
                eight
                    .with_block(b, &mut |e| assert_eq!(a.data, e.data, "block {b}: 1 vs 8"))
                    .unwrap();
                rayon_default
                    .with_block(b, &mut |d| assert_eq!(a.data, d.data, "block {b}: 1 vs 0"))
                    .unwrap();
            })
            .unwrap();
        }

        let params = TrainParams {
            n_rounds: 3,
            max_depth: 3,
            max_bin: 32,
            min_child_weight: 0.0,
            cols_per_block: 4,
            nthread: 4,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, Objective::Logistic);
        let model_one = train(&one, &labels_one, &[], &cfg, &mut [], &Local).unwrap();
        let model_eight = train(&eight, &labels_eight, &[], &cfg, &mut [], &Local).unwrap();
        assert_eq!(
            model_one.to_xgboost_json().unwrap(),
            model_eight.to_xgboost_json().unwrap()
        );
        std::fs::remove_file(path).ok();
    }

    /// 两次调用是两次缓存读,但 Parquet 解压和量化只发生一次。
    #[test]
    fn repeated_reads_hit_cache_without_requantizing() {
        let path = write_fixture("memory_cache");
        let (src, _) = ParquetBlockSource::open(&path, "target", 16, 2).unwrap();
        assert_eq!(src.cache_kind(), "memory");
        assert_eq!(
            src.parquet_load_count(),
            src.n_blocks(),
            "each block is quantized only once"
        );
        assert_eq!(src.cache_read_count(), 0);

        let mut seen = Vec::new();
        src.with_block(0, &mut |b| seen.push(b.data.clone()))
            .unwrap();
        assert_eq!(src.cache_read_count(), 1);

        src.with_block(0, &mut |b| seen.push(b.data.clone()))
            .unwrap();
        assert_eq!(src.cache_read_count(), 2);
        assert_eq!(
            src.parquet_load_count(),
            src.n_blocks(),
            "repeated access must not reload from source"
        );
        assert_eq!(seen[0], seen[1], "two reads must return the same content");

        // 训练里每层每块一次:两层两块 = 4 次
        for _ in 0..2 {
            for b in 0..src.n_blocks() {
                src.with_block(b, &mut |_| {}).unwrap();
            }
        }
        assert_eq!(src.cache_read_count(), 2 + 2 * src.n_blocks());
        assert_eq!(src.parquet_load_count(), src.n_blocks());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn spill_cache_matches_memory_cache() {
        let path = write_fixture("spill_cache");
        let (memory, _) = ParquetBlockSource::open(&path, "target", 16, 2).unwrap();
        let (disk, _) = ParquetBlockSource::open_with_cache(
            &path,
            "target",
            16,
            2,
            QuantizedCacheOptions {
                budget_bytes: Some(0),
                path: None,
            },
        )
        .unwrap();
        assert_eq!(disk.cache_kind(), "disk");
        assert_eq!(disk.parquet_load_count(), disk.n_blocks());
        let first_block = memory.block_features(0).unwrap();
        let selected = first_block[1];
        let expected = memory.feature_column(selected).unwrap();
        assert_eq!(disk.feature_column(selected).unwrap(), expected);
        let full_bytes = (expected.byte_len() * first_block.len()) as u64;
        assert_eq!(disk.source_bytes_read(), Some(full_bytes));
        disk.with_block_cols(0, &[1], &mut |block| {
            assert_eq!(block.feat_ids, [selected]);
            assert_eq!(block.data, expected);
        })
        .unwrap();
        assert_eq!(
            disk.source_bytes_read(),
            Some(full_bytes + expected.byte_len() as u64)
        );
        for b in 0..memory.n_blocks() {
            memory
                .with_block(b, &mut |a| {
                    disk.with_block(b, &mut |d| assert_eq!(a.data, d.data))
                        .unwrap();
                })
                .unwrap();
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn persistent_cache_reuses_bins_and_rejects_different_cuts() {
        let path = write_fixture("persistent_cache");
        let cache = path.with_extension("fbcache");
        let options = || QuantizedCacheOptions {
            budget_bytes: Some(0),
            path: Some(cache.clone()),
        };
        let (first, labels_a) =
            ParquetBlockSource::open_with_cache(&path, "target", 16, 2, options()).unwrap();
        assert_eq!(first.parquet_load_count(), first.n_blocks());
        let (second, labels_b) =
            ParquetBlockSource::open_with_cache(&path, "target", 16, 2, options()).unwrap();
        assert_eq!(
            second.parquet_load_count(),
            0,
            "the second open must not re-quantize"
        );
        assert_eq!(labels_a, labels_b);
        for b in 0..first.n_blocks() {
            first
                .with_block(b, &mut |a| {
                    second
                        .with_block(b, &mut |d| assert_eq!(a.data, d.data))
                        .unwrap();
                })
                .unwrap();
        }

        let mut different = second.cuts().clone();
        different.values[0] = f32::from_bits(different.values[0].to_bits() ^ 1);
        let err = ParquetBlockSource::with_cuts_and_cache(&path, "target", different, 2, options())
            .err()
            .expect("different cuts must be rejected");
        // 现在是**逐值**比较,不再只比哈希,所以报错措辞也换了
        assert!(
            err.to_string()
                .contains("refusing to use stale quantized data"),
            "{err:#}"
        );
        std::fs::remove_dir_all(cache).ok();
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn persistent_cache_is_bound_to_the_missing_sentinel() {
        let path = write_fixture("cache_missing");
        let cache = path.with_extension("fbcache_missing");
        let options = || QuantizedCacheOptions {
            budget_bytes: Some(0),
            path: Some(cache.clone()),
        };
        let with = |missing: f32| Targets::new("target", None).unwrap().with_missing(missing);
        let open = |missing: f32| {
            ParquetBlockSource::open_with_cache_observed(
                &path,
                with(missing),
                16,
                2,
                options(),
                0,
                |_, _, _| {},
            )
        };
        let (first, _) = open(-999.0).unwrap();
        assert_eq!(first.parquet_load_count(), first.n_blocks());
        let (second, _) = open(-999.0).unwrap();
        assert_eq!(
            second.parquet_load_count(),
            0,
            "same sentinel reuses the cache"
        );
        for other in [0.0, f32::NAN] {
            let Err(err) = open(other) else {
                panic!("a cache built for a different sentinel must be rejected");
            };
            assert!(err.to_string().contains("missing"), "{err:#}");
        }
        std::fs::remove_dir_all(cache).ok();
        std::fs::remove_file(path).ok();
    }

    /// 写缓存写到一半(有 block 文件、没有 manifest)时,下次打开必须
    /// **明确报错**,不能当成有效缓存用,也不能静默重建覆盖。
    ///
    /// manifest 是最后一步 rename 上去的,所以「有 manifest」等价于
    /// 「写完了」。半成品目录只能靠"非空但没 manifest"识别。
    #[test]
    fn half_written_cache_is_rejected_not_silently_reused() {
        let path = write_fixture("half_cache");
        let dir = std::env::temp_dir().join(format!("fb_half_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // 只有块文件,没有 manifest —— 就是被中断的样子
        std::fs::write(dir.join("block-0.bin"), b"garbage").unwrap();

        let opts = QuantizedCacheOptions {
            budget_bytes: None,
            path: Some(dir.clone()),
        };
        let Err(e) = ParquetBlockSource::open_with_cache(&path, "target", 16, 2, opts) else {
            panic!("a partial cache directory must be an error");
        };
        assert!(
            e.to_string().contains("is not a valid ferrisboost cache"),
            "error should say the cache directory is the problem, got: {e}"
        );
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn same_length_cache_corruption_is_rejected_on_first_read() {
        let path = write_fixture("checksum_cache");
        let dir = std::env::temp_dir().join(format!("fb_checksum_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let opts = || QuantizedCacheOptions {
            budget_bytes: Some(0),
            path: Some(dir.clone()),
        };

        let (source, _) =
            ParquetBlockSource::open_with_cache(&path, "target", 16, 2, opts()).unwrap();
        drop(source);

        let block_path = dir.join("block-0.bin");
        let mut bytes = std::fs::read(&block_path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 1;
        std::fs::write(&block_path, &bytes).unwrap();

        let (reused, _) =
            ParquetBlockSource::open_with_cache(&path, "target", 16, 2, opts()).unwrap();
        // The corruption is in the second column. Selecting only the first
        // must still verify the entire block and suppress the callback.
        let err = reused
            .with_block_cols(0, &[0], &mut |_| panic!("invalid block was exposed"))
            .expect_err("same-length payload corruption must be caught by the checksum");
        assert!(
            err.to_string().contains("payload checksum mismatch"),
            "{err:#}"
        );

        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_file(path).ok();
    }

    /// 验证集**不能**和训练集共用 cache_path:两者量化结果不同。
    /// 共用时给的 cuts 和缓存里的对不上,必须报错而不是拿旧 bin 训练。
    #[test]
    fn sharing_cache_path_between_different_cuts_is_rejected() {
        let path = write_fixture("shared_cache");
        let dir = std::env::temp_dir().join(format!("fb_shared_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let opts = |d: &std::path::Path| QuantizedCacheOptions {
            budget_bytes: None,
            path: Some(d.to_path_buf()),
        };
        let (train, _) =
            ParquetBlockSource::open_with_cache(&path, "target", 16, 2, opts(&dir)).unwrap();

        // 同一个目录、但换一套 cuts(把第一个切分点挪一下)
        let mut other = train.cuts().clone();
        other.values[0] += 1.0;
        let Err(e) = ParquetBlockSource::with_cuts_and_cache(&path, "target", other, 2, opts(&dir))
        else {
            panic!("sharing cache_path with different cuts must fail");
        };
        assert!(
            e.to_string()
                .contains("refusing to use stale quantized data"),
            "{e}"
        );

        // 同一套 cuts 则可以复用,而且不再回源量化
        let (again, _) = ParquetBlockSource::with_cuts_and_cache(
            &path,
            "target",
            train.cuts().clone(),
            2,
            opts(&dir),
        )
        .unwrap();
        assert_eq!(
            again.parquet_load_count(),
            0,
            "reuse must not quantize again"
        );

        drop(train);
        drop(again);
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&path).ok();
    }

    /// 端到端:按需加载训出来的模型和全量加载的**逐位相同**。
    #[test]
    fn training_on_lazy_source_matches_eager() {
        use crate::comm::Local;
        use crate::train::{train, TrainConfig};
        use crate::types::{Objective, TrainParams};

        let path = write_fixture("lazy_train");
        let eager = open(&path, "target", 16, 2).unwrap();
        let (lazy, labels) = ParquetBlockSource::open(&path, "target", 16, 2).unwrap();

        let p = TrainParams {
            n_rounds: 5,
            max_depth: 3,
            max_bin: 16,
            min_child_weight: 0.0,
            cols_per_block: 2,
            nthread: 1,
            ..Default::default()
        };
        let cfg = TrainConfig::new(p, Objective::Logistic);
        let a = train(&lazy, &labels, &[], &cfg, &mut [], &Local).unwrap();
        let b = train(&eager.source, &eager.labels, &[], &cfg, &mut [], &Local).unwrap();

        let n = f32::NAN;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 10.0, 7.0],
            vec![2.0, n, 8.0],
            vec![3.0, 30.0, 9.0],
            vec![4.0, 40.0, 1.0],
            vec![5.0, 50.0, 2.0],
            vec![6.0, n, 3.0],
        ];
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(
                a.predict_margin(r).to_bits(),
                b.predict_margin(r).to_bits(),
                "row {i}: on-demand and full loading trained differently"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_label_column_says_what_is_available() {
        let path = write_fixture("missing_label");
        let Err(e) = open(&path, "nope", 16, 2) else {
            panic!("should fail")
        };
        let msg = e.to_string();
        assert!(
            msg.contains("nope") && msg.contains("f0"),
            "error should list the available columns: {msg}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn valid_set_reuses_training_cuts() {
        let path = write_fixture("valid_cuts");
        let train = open(&path, "target", 16, 2).unwrap();
        let cuts = train.source.cuts().clone();
        let valid = open_with_cuts(&path, "target", cuts, 2).unwrap();
        for f in 0..3u32 {
            assert_eq!(
                valid.source.cuts().cuts_for(f),
                train.source.cuts().cuts_for(f)
            );
        }
        for row in 0..6 {
            for f in 0..3u32 {
                assert_eq!(valid.source.bin_at(row, f), train.source.bin_at(row, f));
            }
        }
        std::fs::remove_file(&path).ok();
    }

    /// Two row groups per file, every row group with its own string dictionary
    /// order: one category list (first appearance in rows), the same bins from the lazy
    /// single-file source, the multi-file dataset and a validation file.
    #[test]
    fn string_dictionaries_get_one_category_list() {
        use crate::category_names::CategoryValues;
        use arrow::array::{DictionaryArray, Int32Array, StringArray};
        use arrow::datatypes::Int32Type;

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "c",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new("target", DataType::Float32, false),
        ]));
        let dir = std::env::temp_dir().join(format!("fb_parquet_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, groups: &[(&[&str], &[Option<i32>])]| -> PathBuf {
            let path = dir.join(format!("{name}.parquet"));
            let mut w =
                ArrowWriter::try_new(File::create(&path).unwrap(), schema.clone(), None).unwrap();
            for (dict, keys) in groups {
                let c = DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from(keys.to_vec()),
                    Arc::new(StringArray::from(dict.to_vec())),
                )
                .unwrap();
                let y = Float32Array::from((0..keys.len()).map(|i| i as f32).collect::<Vec<_>>());
                let b =
                    RecordBatch::try_new(schema.clone(), vec![Arc::new(c), Arc::new(y)]).unwrap();
                w.write(&b).unwrap();
                w.flush().unwrap(); // one row group per batch
            }
            w.close().unwrap();
            path
        };
        // rows: b a b ∅ | c a | (file 2) a d c
        let a = write(
            "dict_a",
            &[
                (&["b", "a"], &[Some(0), Some(1), Some(0), None]),
                (&["a", "c"], &[Some(1), Some(0)]),
            ],
        );
        let b = write(
            "dict_b",
            &[(&["d", "c", "a"], &[Some(2), Some(0), Some(1)])],
        );

        let (lazy, _) = ParquetBlockSource::open(&a, "target", 16, 1).unwrap();
        let t = lazy.cuts().categories().expect("named categories").clone();
        assert_eq!(
            t.get(0).unwrap().values(),
            &CategoryValues::Str(vec!["b".into(), "a".into(), "c".into()])
        );
        let column = lazy.feature_column(0).unwrap();
        let missing = lazy.cuts().bin_width(0).missing();
        let bins: Vec<u32> = (0..6).map(|r| column.as_slice().get(r)).collect();
        assert_eq!(bins, vec![0, 1, 0, missing, 2, 1]);

        // A validation file is coded with the training list: "d" is unseen.
        let Err(e) = ParquetBlockSource::with_cuts(&b, "target", lazy.cuts().clone(), 1) else {
            panic!("an unseen category must be rejected")
        };
        assert!(e.to_string().contains("\"d\""), "{e}");

        let many = open_many(&[a.clone(), b.clone()], "target", None, 16, 1).unwrap();
        let t2 = many.source.cuts().categories().unwrap();
        assert_eq!(
            t2.get(0).unwrap().values(),
            &CategoryValues::Str(vec!["b".into(), "a".into(), "c".into(), "d".into()])
        );
        let got: Vec<u32> = (6..9).map(|r| many.source.bin_at(r, 0)).collect();
        assert_eq!(got, vec![1, 3, 2]); // rows a, d, c
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
    }
}
