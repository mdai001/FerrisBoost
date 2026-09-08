//! 从 CSV 读数据。
//!
//! 和 Parquet 走同一条路(两遍扫描 → `ArrowSource`),只是多了一步
//! **schema 推断**:CSV 没有类型信息,得先采样一遍猜出每列是什么。
//!
//! CSV 是最差的训练数据格式 —— 没有类型、没有统计信息、解析要过文本、
//! 每次都得从头扫。之所以还是支持,是因为它是用户手边最常有的东西,
//! 拦着不让用只会逼他们先转一道。真跑大数据请用 Parquet。

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};

use arrow::csv::reader::Format;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::columns::BinCuts;
use crate::source::arrow_source::ArrowSource;
use crate::source::parquet_source::Dataset;

/// 推断 schema 时最多看多少行。
///
/// 太少会把"前 1000 行恰好都是整数"的列判成 Int64,后面出现小数就变
/// null —— 那是静默丢数据。给大一点,这一遍只读不解析,不贵。
const INFER_ROWS: usize = 10_000;

const BATCH_SIZE: usize = 8192;

/// Open a streaming reader. Schema inference and data parsing each open their
/// own reader, so gzip never needs to be made seekable by inflating the entire
/// file into a `Vec`.
fn readable(path: &Path) -> anyhow::Result<Box<dyn Read + Send>> {
    let file = File::open(path).map_err(|e| anyhow::anyhow!("打不开 {}:{e}", path.display()))?;
    if crate::source::input::is_gzipped(path) {
        Ok(Box::new(flate2::read::GzDecoder::new(BufReader::new(file))))
    } else {
        Ok(Box::new(file))
    }
}

fn infer(path: &Path, has_header: bool) -> anyhow::Result<SchemaRef> {
    let mut file = readable(path)?;
    let format = Format::default().with_header(has_header);
    let (schema, _) = format
        .infer_schema(&mut file, Some(INFER_ROWS))
        .map_err(|e| anyhow::anyhow!("{} 的 schema 推断失败:{e}", path.display()))?;
    Ok(Arc::new(schema))
}

/// 只推断 schema,数一下特征列有多少个 —— 不做量化、不建 cuts。
///
/// 和 `parquet_source::peek_n_features` 同一个用途:让 blocking 的策略
/// 决定留在 wiring 边界上,而不是渗进底层分块。
pub fn peek_n_features(
    path: impl AsRef<Path>,
    label: &str,
    has_header: bool,
) -> anyhow::Result<usize> {
    let schema = infer(path.as_ref(), has_header)?;
    columns(&schema, label).map(|(features, _, _)| features.len())
}

/// Resolve a positional label against the inferred CSV schema.  Arrow assigns
/// stable synthetic field names to headerless columns; the name is only an
/// internal handle used by the existing projection machinery.
pub fn positional_label_name(path: impl AsRef<Path>, label_index: usize) -> anyhow::Result<String> {
    let schema = infer(path.as_ref(), false)?;
    let n_columns = schema.fields().len();
    anyhow::ensure!(
        label_index < n_columns,
        "label 列下标 {label_index} 越界:无表头 CSV 只有 {n_columns} 列"
    );
    anyhow::ensure!(n_columns > 1, "无表头 CSV 除了标签列之外没有特征列");
    Ok(schema.field(label_index).name().clone())
}

/// 把推断出来的列类型收紧成训练能用的形状。
///
/// 推断出 Utf8 的列直接报错,而不是留到量化时才炸 —— 那时候的错误
/// 信息里只有列号,这里还能说出列名。
fn check_numeric(schema: &Schema, skip: &str) -> anyhow::Result<()> {
    for f in schema.fields() {
        if f.name() == skip {
            continue;
        }
        match f.data_type() {
            DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64 => {}
            other => anyhow::bail!(
                "列 {:?} 推断成了 {other:?},只支持数值列。\
                 类别特征要先在上游编码成数值;如果它其实是数字,\
                 检查一下是不是有非法字符或者千分位逗号",
                f.name()
            ),
        }
    }
    Ok(())
}

fn batch_reader(
    path: &Path,
    schema: SchemaRef,
    has_header: bool,
    projection: Vec<usize>,
) -> anyhow::Result<impl Iterator<Item = anyhow::Result<RecordBatch>>> {
    let file = readable(path)?;
    let reader = ReaderBuilder::new(schema)
        .with_header(has_header)
        .with_batch_size(BATCH_SIZE)
        .with_projection(projection)
        .build(BufReader::new(file))?;
    Ok(reader.map(|r| r.map_err(anyhow::Error::from)))
}

fn columns(schema: &Schema, label: &str) -> anyhow::Result<(Vec<usize>, usize, Vec<String>)> {
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let label_col = names
        .iter()
        .position(|n| *n == label)
        .ok_or_else(|| anyhow::anyhow!("文件里没有列 {label:?},只有 {names:?}"))?;
    let feat_cols: Vec<usize> = (0..names.len()).filter(|i| *i != label_col).collect();
    anyhow::ensure!(
        !feat_cols.is_empty(),
        "除了标签列 {label:?} 之外没有别的列了"
    );
    let feat_names = feat_cols.iter().map(|&i| names[i].to_string()).collect();
    Ok((feat_cols, label_col, feat_names))
}

/// 打开一个 CSV 文件,自己建分箱边界。第一行必须是表头(要靠列名找标签)。
pub fn open(
    path: impl AsRef<Path>,
    label: &str,
    max_bins: u32,
    cols_per_block: usize,
) -> anyhow::Result<Dataset> {
    open_inner(path.as_ref(), label, None, max_bins, cols_per_block)
}

/// 打开一个 CSV 文件,**复用给定的分箱边界**。验证集走这条。
pub fn open_with_cuts(
    path: impl AsRef<Path>,
    label: &str,
    cuts: BinCuts,
    cols_per_block: usize,
) -> anyhow::Result<Dataset> {
    open_inner(path.as_ref(), label, Some(cuts), 0, cols_per_block)
}

/// 多文件 CSV。单文件是它的退化情况。
///
/// **按列名绑定**:各文件的列顺序可以不同,这里为每个文件单独解出投影,
/// 所以读出来的 batch 一定是同一个逻辑顺序。规范顺序取第一个文件的顺序
/// (文件列表已按路径字典序排好,所以"第一个"是确定的)。
///
/// ⚠️ 每个文件**各自**推断 schema,然后要求特征列名集合一致 ——
/// 拿第一个文件的 schema 去读其它文件,列错位了也不会报错,只会静默
/// 训出一个错模型。
pub fn open_many(
    paths: &[PathBuf],
    label: &str,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
) -> anyhow::Result<Dataset> {
    open_many_with_header(paths, label, cuts, max_bins, cols_per_block, true)
}

/// CSV dataset with an explicit header contract.  `has_header=false` is the
/// positional product mode: schemas must agree by column count/order and the
/// returned model metadata intentionally contains no feature names.
pub fn open_many_with_header(
    paths: &[PathBuf],
    label: &str,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
    has_header: bool,
) -> anyhow::Result<Dataset> {
    open_many_with_header_and_nthread(
        paths,
        label,
        cuts,
        max_bins,
        cols_per_block,
        has_header,
        crate::source::DEFAULT_INGEST_THREADS,
    )
}

pub fn open_many_with_header_and_nthread(
    paths: &[PathBuf],
    label: &str,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
    has_header: bool,
    nthread: usize,
) -> anyhow::Result<Dataset> {
    anyhow::ensure!(!paths.is_empty(), "没有输入文件");
    let schema_pool = crate::threading::build_pool(nthread)
        .map_err(|e| anyhow::anyhow!("CSV schema ingest 线程池创建失败:{e}"))?;
    let schemas = schema_pool.install(|| {
        paths
            .par_iter()
            .map(|path| infer(path, has_header))
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    let mut feat_proj = Vec::with_capacity(paths.len());
    let mut label_proj = Vec::with_capacity(paths.len());
    let mut feature_names: Option<Vec<String>> = None;

    for (p, schema) in paths.iter().zip(&schemas) {
        check_numeric(&schema, label)?;
        let (cols, label_col, names) = columns(&schema, label)?;
        match &feature_names {
            None => {
                feature_names = Some(names);
                feat_proj.push(cols);
            }
            Some(canon) if has_header => {
                anyhow::ensure!(
                    names.len() == canon.len(),
                    "多文件 schema 不一致:{} 有 {} 个特征列,{} 有 {} 个",
                    paths[0].display(),
                    canon.len(),
                    p.display(),
                    names.len()
                );
                let mut mapped = Vec::with_capacity(canon.len());
                for want in canon {
                    let at = names.iter().position(|n| n == want).ok_or_else(|| {
                        anyhow::anyhow!("多文件 schema 不一致:{} 里没有列 {want:?}", p.display())
                    })?;
                    mapped.push(cols[at]);
                }
                feat_proj.push(mapped);
            }
            Some(canon) => {
                anyhow::ensure!(
                    names.len() == canon.len(),
                    "多文件位置 schema 不一致:{} 有 {} 个特征列,{} 有 {} 个",
                    paths[0].display(),
                    canon.len(),
                    p.display(),
                    names.len()
                );
                feat_proj.push(cols);
            }
        }
        label_proj.push(label_col);
    }
    let feature_names = feature_names.expect("非空");

    let (reader_threads, transform_threads) = crate::source::pipeline_threads(nthread);
    let owned = paths.to_vec();
    let (sc, fp) = (schemas.clone(), feat_proj.clone());
    let feats = || {
        Ok(dataset_reader(
            owned.clone(),
            sc.clone(),
            fp.clone(),
            has_header,
            reader_threads,
        ))
    };
    let source = match cuts {
        Some(c) => ArrowSource::with_cuts_and_nthread(feats, c, cols_per_block, transform_threads)?,
        None => ArrowSource::new_with_nthread(feats, max_bins, cols_per_block, transform_threads)?,
    };

    let mut labels = Vec::new();
    let lp: Vec<Vec<usize>> = label_proj.iter().map(|c| vec![*c]).collect();
    for batch in dataset_reader(paths.to_vec(), schemas, lp, has_header, nthread) {
        let batch = batch?;
        labels.extend(crate::source::arrow_source::column_as_f32(&batch, 0)?);
    }
    Dataset::assemble(
        source,
        labels,
        if has_header {
            feature_names
        } else {
            Vec::new()
        },
        label,
    )
}

fn dataset_reader(
    paths: Vec<PathBuf>,
    schemas: Vec<SchemaRef>,
    projections: Vec<Vec<usize>>,
    has_header: bool,
    nthread: usize,
) -> crate::source::parallel_reader::BatchIter {
    if nthread <= 1 {
        return Box::new(chained_csv(paths, schemas, projections, has_header));
    }
    if paths.len() == 1 {
        let (mask, order) = crate::source::input::projection_and_order(&projections[0]);
        return parallel_csv_chunks(
            paths[0].clone(),
            schemas[0].clone(),
            has_header,
            mask,
            order,
            nthread,
        );
    }
    let paths = Arc::new(paths);
    let schemas = Arc::new(schemas);
    let projections = Arc::new(projections);
    let n_files = paths.len();
    Box::new(crate::source::parallel_reader::ordered_parallel_batches(
        n_files,
        nthread,
        move |index| {
            let path = paths[index].clone();
            let schema = schemas[index].clone();
            let (mask, order) = crate::source::input::projection_and_order(&projections[index]);
            let iter = batch_reader(&path, schema, has_header, mask)?;
            Ok(Box::new(iter.map(move |batch| {
                batch.map(|batch| crate::source::input::reorder(&batch, &order))
            })))
        },
    ))
}

pub(crate) const CSV_CHUNK_BYTES: usize = 2 * 1024 * 1024;

struct CsvChunkJob {
    bytes: Vec<u8>,
    rows: usize,
    result: mpsc::SyncSender<anyhow::Result<Vec<RecordBatch>>>,
}

enum CsvChunkMessage {
    Chunk(mpsc::Receiver<anyhow::Result<Vec<RecordBatch>>>),
    Error(anyhow::Error),
    Done,
}

struct ParallelCsvChunks {
    messages: mpsc::Receiver<CsvChunkMessage>,
    ready: VecDeque<RecordBatch>,
}

impl Iterator for ParallelCsvChunks {
    type Item = anyhow::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(batch) = self.ready.pop_front() {
                return Some(Ok(batch));
            }
            match self.messages.recv() {
                Ok(CsvChunkMessage::Chunk(result)) => match result.recv() {
                    Ok(Ok(batches)) => self.ready.extend(batches),
                    Ok(Err(error)) => return Some(Err(error)),
                    Err(_) => return Some(Err(anyhow::anyhow!("CSV parse worker stopped"))),
                },
                Ok(CsvChunkMessage::Error(error)) => return Some(Err(error)),
                Ok(CsvChunkMessage::Done) | Err(_) => return None,
            }
        }
    }
}

/// Split only at complete CSV record boundaries. Quoted newlines remain inside
/// one record and doubled quotes do not toggle quote state.
fn read_csv_record(reader: &mut dyn BufRead, record: &mut Vec<u8>) -> anyhow::Result<usize> {
    record.clear();
    let mut in_quotes = false;
    loop {
        let before = record.len();
        let read = reader.read_until(b'\n', record)?;
        if read == 0 {
            return Ok(record.len());
        }
        let mut i = before;
        while i < record.len() {
            if record[i] == b'"' {
                if in_quotes && i + 1 < record.len() && record[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_quotes = !in_quotes;
            }
            i += 1;
        }
        if !in_quotes {
            return Ok(record.len());
        }
    }
}

fn parse_csv_chunk(
    bytes: Vec<u8>,
    rows: usize,
    schema: SchemaRef,
    projection: Vec<usize>,
    order: Vec<usize>,
) -> anyhow::Result<Vec<RecordBatch>> {
    let reader = ReaderBuilder::new(schema)
        .with_header(false)
        // Avoid Arrow reserving a full 8192-row × wide-schema batch for a
        // chunk that may contain only a few dozen very wide records.
        .with_batch_size(rows.clamp(1, BATCH_SIZE))
        .with_projection(projection)
        .build(Cursor::new(bytes))?;
    reader
        .map(|batch| {
            batch
                .map(|batch| crate::source::input::reorder(&batch, &order))
                .map_err(anyhow::Error::from)
        })
        .collect()
}

/// One sequential reader/decompressor feeds complete-record chunks to bounded
/// parse workers. Gzip itself remains serial, while decompression overlaps
/// parsing; plain CSV additionally gets parallel record parsing.
fn parallel_csv_chunks(
    path: PathBuf,
    schema: SchemaRef,
    has_header: bool,
    projection: Vec<usize>,
    order: Vec<usize>,
    nthread: usize,
) -> crate::source::parallel_reader::BatchIter {
    let parser_threads = nthread.saturating_sub(1).max(1);
    let (job_tx, job_rx) = mpsc::sync_channel::<CsvChunkJob>(parser_threads);
    let job_rx = Arc::new(Mutex::new(job_rx));
    for _ in 0..parser_threads {
        let jobs = Arc::clone(&job_rx);
        let schema = schema.clone();
        let projection = projection.clone();
        let order = order.clone();
        std::thread::spawn(move || loop {
            let job = match jobs.lock() {
                Ok(rx) => rx.recv(),
                Err(_) => return,
            };
            let Ok(job) = job else { return };
            let parsed = parse_csv_chunk(
                job.bytes,
                job.rows,
                schema.clone(),
                projection.clone(),
                order.clone(),
            );
            if job.result.send(parsed).is_err() {
                return;
            }
        });
    }

    let (message_tx, message_rx) = mpsc::sync_channel(parser_threads);
    std::thread::spawn(move || {
        let produce = || -> anyhow::Result<()> {
            let input = readable(&path)?;
            let mut reader = BufReader::new(input);
            let mut record = Vec::new();
            if has_header && read_csv_record(&mut reader, &mut record)? == 0 {
                anyhow::bail!("{} 没有 CSV 表头", path.display());
            }
            loop {
                let mut bytes = Vec::with_capacity(CSV_CHUNK_BYTES);
                let mut rows = 0usize;
                while bytes.len() < CSV_CHUNK_BYTES {
                    if read_csv_record(&mut reader, &mut record)? == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&record);
                    rows += 1;
                }
                if bytes.is_empty() {
                    break;
                }
                let (result_tx, result_rx) = mpsc::sync_channel(1);
                message_tx
                    .send(CsvChunkMessage::Chunk(result_rx))
                    .map_err(|_| anyhow::anyhow!("CSV batch consumer stopped"))?;
                job_tx
                    .send(CsvChunkJob {
                        bytes,
                        rows,
                        result: result_tx,
                    })
                    .map_err(|_| anyhow::anyhow!("CSV parse workers stopped"))?;
            }
            Ok(())
        };
        if let Err(error) = produce() {
            let _ = message_tx.send(CsvChunkMessage::Error(error));
        }
        let _ = message_tx.send(CsvChunkMessage::Done);
    });
    Box::new(ParallelCsvChunks {
        messages: message_rx,
        ready: VecDeque::new(),
    })
}

/// 按文件顺序惰性串起来的 batch 流。打开失败作为一个 `Err` 元素传下去。
fn chained_csv(
    paths: Vec<PathBuf>,
    schemas: Vec<SchemaRef>,
    projections: Vec<Vec<usize>>,
    has_header: bool,
) -> impl Iterator<Item = anyhow::Result<RecordBatch>> {
    paths
        .into_iter()
        .zip(schemas)
        .zip(projections)
        .flat_map(move |((p, sc), cols)| {
            // 同 parquet:投影之后必须按规范顺序重排。
            let (mask, order) = crate::source::input::projection_and_order(&cols);
            match batch_reader(&p, sc, has_header, mask) {
                Ok(it) => {
                    Box::new(it.map(move |b| b.map(|b| crate::source::input::reorder(&b, &order))))
                        as Box<dyn Iterator<Item = anyhow::Result<RecordBatch>>>
                }
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        })
}

/// schema + 全部列名(含标签列)。预测绑定要用。
pub fn schema_and_names(
    path: impl AsRef<Path>,
    has_header: bool,
) -> anyhow::Result<(SchemaRef, Vec<String>)> {
    let schema = infer(path.as_ref(), has_header)?;
    let names = schema.fields().iter().map(|f| f.name().clone()).collect();
    Ok((schema, names))
}

/// 按给定投影读原始 batch(不量化)。预测路径用。
pub fn raw_batches(
    path: impl AsRef<Path>,
    schema: SchemaRef,
    projection: &[usize],
    has_header: bool,
) -> anyhow::Result<impl Iterator<Item = anyhow::Result<RecordBatch>> + 'static> {
    batch_reader(path.as_ref(), schema, has_header, projection.to_vec())
}

pub(crate) fn raw_batches_with_nthread(
    path: impl AsRef<Path>,
    schema: SchemaRef,
    projection: &[usize],
    has_header: bool,
    nthread: usize,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    if nthread <= 1 {
        return Ok(Box::new(batch_reader(
            path.as_ref(),
            schema,
            has_header,
            projection.to_vec(),
        )?));
    }
    Ok(parallel_csv_chunks(
        path.as_ref().to_path_buf(),
        schema,
        has_header,
        projection.to_vec(),
        (0..projection.len()).collect(),
        nthread,
    ))
}

fn open_inner(
    path: &Path,
    label: &str,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
) -> anyhow::Result<Dataset> {
    open_inner_with_header(path, label, cuts, max_bins, cols_per_block, true)
}

fn open_inner_with_header(
    path: &Path,
    label: &str,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
    has_header: bool,
) -> anyhow::Result<Dataset> {
    open_many_with_header_and_nthread(
        &[path.to_path_buf()],
        label,
        cuts,
        max_bins,
        cols_per_block,
        has_header,
        crate::source::DEFAULT_INGEST_THREADS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::DenseSource;
    use crate::train::BlockSource as _;

    fn write(name: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fb_csv_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.csv"));
        std::fs::write(&path, body).unwrap();
        path
    }

    /// 空字段是缺失,不是 0。
    const BODY: &str = "f0,target,f1,f2\n\
                        1.0,0,10.0,7\n\
                        2.0,1,,8\n\
                        3.0,0,30.0,9\n\
                        4.0,1,40.0,1\n\
                        5.0,1,50.0,2\n\
                        6.0,0,,3\n";

    #[test]
    fn csv_matches_dense_bin_for_bin() {
        let path = write("match_dense", BODY);
        let ds = open(&path, "target", 16, 2).unwrap();

        assert_eq!(ds.source.n_rows(), 6);
        assert_eq!(ds.source.n_features(), 3, "标签列不算特征");
        assert_eq!(ds.feature_names, vec!["f0", "f1", "f2"]);
        assert_eq!(ds.labels, vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);

        let n = f32::NAN;
        let row_major: Vec<f32> = vec![
            1.0, 10.0, 7.0, 2.0, n, 8.0, 3.0, 30.0, 9.0, 4.0, 40.0, 1.0, 5.0, 50.0, 2.0, 6.0, n,
            3.0,
        ];
        let dense = DenseSource::from_row_major(&row_major, 6, 3, 16, 2);
        for row in 0..6 {
            for f in 0..3u32 {
                assert_eq!(
                    ds.source.bin_at(row, f),
                    dense.bin_at(row, f),
                    "行 {row} 特征 {f}"
                );
            }
        }
        std::fs::remove_file(&path).ok();
    }

    /// 空字段 → null → 缺失哨兵。填成 0 的话它会变成一个真实取值,
    /// 而 0 在这一列里本来就有含义,错得很隐蔽。
    #[test]
    fn empty_fields_are_missing_not_zero() {
        let path = write("empty_missing", BODY);
        let ds = open(&path, "target", 16, 3).unwrap();
        assert_eq!(ds.source.bin_at(1, 1), crate::types::MISSING_BIN);
        assert_eq!(ds.source.bin_at(5, 1), crate::types::MISSING_BIN);
        for row in [0usize, 2, 3, 4] {
            assert_ne!(ds.source.bin_at(row, 1), crate::types::MISSING_BIN);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn text_columns_are_rejected_by_name() {
        let path = write(
            "text_col",
            "f0,target,city\n1.0,0,beijing\n2.0,1,shanghai\n",
        );
        let Err(e) = open(&path, "target", 16, 2) else {
            panic!("字符串列应该被拒绝")
        };
        let msg = e.to_string();
        assert!(msg.contains("city"), "报错要点名是哪一列:{msg}");
        assert!(msg.contains("只支持数值列"), "{msg}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_label_column_lists_what_is_there() {
        let path = write("no_label", BODY);
        let Err(e) = open(&path, "nope", 16, 2) else {
            panic!("应该报错")
        };
        assert!(e.to_string().contains("f0"), "{e}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn valid_set_reuses_training_cuts() {
        let path = write("valid_cuts", BODY);
        let train = open(&path, "target", 16, 2).unwrap();
        let valid = open_with_cuts(&path, "target", train.source.cuts().clone(), 2).unwrap();
        for f in 0..3u32 {
            assert_eq!(
                valid.source.cuts().cuts_for(f),
                train.source.cuts().cuts_for(f)
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn headless_csv_uses_a_positional_label() {
        let path = write(
            "headless",
            "1.0,10.0,0,7\n2.0,20.0,1,8\n3.0,30.0,0,9\n4.0,40.0,1,1\n",
        );
        let label = positional_label_name(&path, 2).unwrap();
        let ds =
            open_many_with_header(std::slice::from_ref(&path), &label, None, 16, 2, false).unwrap();
        assert_eq!(ds.source.n_rows(), 4, "第一行不能被当成表头吃掉");
        assert_eq!(ds.source.n_features(), 3);
        assert_eq!(ds.labels, vec![0.0, 1.0, 0.0, 1.0]);
        assert!(ds.feature_names.is_empty(), "无表头模型必须是位置模式");
        assert!(positional_label_name(&path, 4).is_err());
        std::fs::remove_file(&path).ok();
    }
}
