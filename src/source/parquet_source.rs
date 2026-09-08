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
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::columns::{BinCuts, ColumnBlock};
use crate::source::arrow_source::ArrowSource;
use crate::train::BlockSource;
use crate::types::{Bin, FeatId, MISSING_BIN};

/// 一份带标签的数据集:特征列进 `source`,标签列单独拎出来。
/// Parquet 和 CSV 共用这个形状。
///
/// 标签不进特征矩阵 —— 拿标签当特征训练,树会直接照着它分裂,
/// 训练集指标完美、线上全错。这里用列名显式排除掉。
pub struct Dataset {
    pub source: ArrowSource,
    pub labels: Vec<f32>,
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
    ) -> anyhow::Result<Self> {
        use crate::train::BlockSource as _;
        anyhow::ensure!(
            labels.len() == source.n_rows(),
            "标签有 {} 行,特征有 {} 行",
            labels.len(),
            source.n_rows()
        );
        anyhow::ensure!(
            !labels.iter().any(|v| v.is_nan()),
            "标签列 {label_name:?} 里有缺失值。缺失的标签得在上游处理掉 —— \
             填 0 会被当成一个真实标签学进去"
        );
        Ok(Self {
            source,
            labels,
            feature_names,
        })
    }
}

fn batch_reader(
    path: &Path,
    columns: &[usize],
    batch_size: usize,
) -> anyhow::Result<impl Iterator<Item = anyhow::Result<RecordBatch>>> {
    let file = File::open(path).map_err(|e| anyhow::anyhow!("打不开 {}:{e}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), columns.to_vec());
    let reader = builder
        .with_projection(mask)
        .with_batch_size(batch_size)
        .build()?;
    Ok(reader.map(|r| r.map_err(anyhow::Error::from)))
}

fn batch_reader_row_group(
    path: &Path,
    columns: &[usize],
    batch_size: usize,
    row_group: usize,
) -> anyhow::Result<impl Iterator<Item = anyhow::Result<RecordBatch>>> {
    let file = File::open(path).map_err(|e| anyhow::anyhow!("打不开 {}:{e}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), columns.to_vec());
    let reader = builder
        .with_projection(mask)
        .with_row_groups(vec![row_group])
        .with_batch_size(batch_size)
        .build()?;
    Ok(reader.map(|r| r.map_err(anyhow::Error::from)))
}

fn parallel_row_groups(
    path: PathBuf,
    columns: Vec<usize>,
    batch_size: usize,
    nthread: usize,
) -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
    if nthread <= 1 {
        return Ok(Box::new(batch_reader(&path, &columns, batch_size)?));
    }
    let n_groups = {
        let file = File::open(&path)?;
        ParquetRecordBatchReaderBuilder::try_new(file)?
            .metadata()
            .num_row_groups()
    };
    if n_groups <= 1 {
        return Ok(Box::new(batch_reader(&path, &columns, batch_size)?));
    }
    let path = std::sync::Arc::new(path);
    let columns = std::sync::Arc::new(columns);
    Ok(Box::new(
        crate::source::parallel_reader::ordered_parallel_batches(
            n_groups,
            nthread,
            move |row_group| {
                Ok(Box::new(batch_reader_row_group(
                    &path, &columns, batch_size, row_group,
                )?))
            },
        ),
    ))
}

/// 列名 -> 下标。
fn column_index(path: &Path, name: &str) -> anyhow::Result<usize> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    builder
        .schema()
        .fields()
        .iter()
        .position(|f| f.name() == name)
        .ok_or_else(|| {
            let have: Vec<&str> = builder
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect();
            anyhow::anyhow!("文件里没有列 {name:?},只有 {have:?}")
        })
}

/// 只读 footer 的 schema,数一下特征列有多少个 —— 不解压任何数据。
///
/// 给「先 peek schema、再决定 blocking、最后才真正 open」这条线用:
/// `cols_per_block` 的自动选择需要 `n_features`,而 Parquet 在 open 之前
/// 是不知道的。把这一步单独拿出来,就不必把 `nthread` 一路传进
/// `partition_features`(那会让底层分块知道训练的线程策略)。
pub fn peek_n_features(path: impl AsRef<Path>, label: &str) -> anyhow::Result<usize> {
    let (idx, _) = feature_columns(path.as_ref(), label)?;
    Ok(idx.len())
}

/// 文件里**全部**列名(含标签列),顺序即物理顺序。预测绑定要用它。
pub fn all_column_names(path: impl AsRef<Path>) -> anyhow::Result<Vec<String>> {
    let file = File::open(path.as_ref())
        .map_err(|e| anyhow::anyhow!("打不开 {}:{e}", path.as_ref().display()))?;
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
pub fn feature_names_of(path: impl AsRef<Path>, label: &str) -> anyhow::Result<Vec<String>> {
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
    let n = builder.metadata().file_metadata().num_rows();
    anyhow::ensure!(n >= 0, "parquet footer 里的行数是负数:{n}");
    Ok(n as u64)
}

pub(crate) fn peek_n_row_groups(path: impl AsRef<Path>) -> anyhow::Result<usize> {
    let file = File::open(path.as_ref())?;
    Ok(ParquetRecordBatchReaderBuilder::try_new(file)?
        .metadata()
        .num_row_groups())
}

fn feature_columns(path: &Path, label: &str) -> anyhow::Result<(Vec<usize>, Vec<String>)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let mut idx = Vec::new();
    let mut names = Vec::new();
    for (i, f) in schema.fields().iter().enumerate() {
        if f.name() != label {
            idx.push(i);
            names.push(f.name().clone());
        }
    }
    anyhow::ensure!(!idx.is_empty(), "除了标签列 {label:?} 之外没有别的列了");
    Ok((idx, names))
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
    let path = path.as_ref().to_path_buf();
    let (feat_cols, _) = feature_columns(&path, label)?;
    let (reader_threads, transform_threads) = crate::source::pipeline_threads(nthread);
    crate::source::arrow_source::build_cuts_with_nthread(
        parallel_row_groups(path, feat_cols, BATCH_SIZE, reader_threads)?,
        max_bins as usize,
        transform_threads,
    )
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
    anyhow::ensure!(!paths.is_empty(), "没有输入文件");
    let (first_cols, feature_names) = feature_columns(&paths[0], label)?;
    let label_col_first = column_index(&paths[0], label)?;

    // 每个文件各自解投影:列名相同、物理顺序不同也能一起训。
    let mut feat_proj: Vec<Vec<usize>> = vec![first_cols];
    let mut label_proj: Vec<usize> = vec![label_col_first];
    for p in &paths[1..] {
        let (cols, names) = feature_columns(p, label)?;
        anyhow::ensure!(
            names.len() == feature_names.len(),
            "多文件 schema 不一致:{} 有 {} 个特征列,{} 有 {} 个",
            paths[0].display(),
            feature_names.len(),
            p.display(),
            names.len()
        );
        // 按**名字**取该文件里的下标,排成规范顺序。
        let mut mapped = Vec::with_capacity(feature_names.len());
        for want in &feature_names {
            let at = names.iter().position(|n| n == want).ok_or_else(|| {
                anyhow::anyhow!("多文件 schema 不一致:{} 里没有特征列 {want:?}", p.display())
            })?;
            mapped.push(cols[at]);
        }
        feat_proj.push(mapped);
        label_proj.push(column_index(p, label)?);
    }

    let (reader_threads, transform_threads) = crate::source::pipeline_threads(nthread);
    let owned = paths.to_vec();
    let feats = || {
        Ok(dataset_reader(
            owned.clone(),
            feat_proj.clone(),
            BATCH_SIZE,
            reader_threads,
        ))
    };
    let source = match cuts {
        Some(c) => ArrowSource::with_cuts_and_nthread(feats, c, cols_per_block, transform_threads)?,
        None => ArrowSource::new_with_nthread(feats, max_bins, cols_per_block, transform_threads)?,
    };

    let mut labels = Vec::new();
    for batch in dataset_reader(
        paths.to_vec(),
        label_proj.iter().map(|c| vec![*c]).collect(),
        BATCH_SIZE,
        nthread,
    ) {
        let batch = batch?;
        labels.extend(crate::source::arrow_source::column_as_f32(&batch, 0)?);
    }
    Dataset::assemble(source, labels, feature_names, label)
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
    label: String,
    max_bins: u32,
    cols_per_block: usize,
    n_rows: usize,
    n_features: usize,
    cuts_values_bits: Vec<u32>,
    cuts_offsets: Vec<u32>,
    cuts_hash: String,
    groups: Vec<Vec<FeatId>>,
    /// BLAKE3 of each `block-{i}.bin`, in `groups` order.
    block_hashes: Vec<String>,
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

/// Parquet 数据源。原始列只在建 cuts 和量化缓存时读取;训练阶段按需读取
/// 的是紧凑的 u8 列块,不再回源解压和重复量化。
pub struct ParquetBlockSource {
    path: PathBuf,
    cuts: BinCuts,
    /// 每个块持有哪些**全局特征 id**
    groups: Vec<Vec<FeatId>>,
    /// 全局特征 id → 文件里的列下标(标签列已经排除,所以两者不等)
    file_col: Vec<usize>,
    n_rows: usize,
    n_features: usize,
    cache: QuantizedCache,
    cache_bytes: usize,
    cache_reads: AtomicUsize,
    /// 真实从 cache 读入的量化字节数。和 H2D 字节分开统计。
    source_bytes: std::sync::atomic::AtomicU64,
    parquet_loads: AtomicUsize,
}

impl ParquetBlockSource {
    fn quantize_block(&self, i: usize) -> anyhow::Result<ColumnBlock> {
        self.parquet_loads.fetch_add(1, Ordering::Relaxed);
        let feat_ids = self
            .groups
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在,一共 {} 个", self.groups.len()))?
            .clone();
        // 只投影这一组列,别的 column chunk 根本不读
        let cols: Vec<usize> = feat_ids
            .iter()
            .map(|&f| self.file_col[f as usize])
            .collect();

        let mut data = vec![0u8; feat_ids.len() * self.n_rows];
        self.read_into(&cols, &feat_ids, self.n_rows, &mut data)?;
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
        let path = path.as_ref().to_path_buf();
        let (feat_cols, label_col) = {
            let _p = crate::source::openprof::start("metadata");
            let (feat_cols, _) = feature_columns(&path, label)?;
            (feat_cols, column_index(&path, label)?)
        };
        if let Some(cache_path) = options.path.as_ref() {
            if cache_path.join("manifest.json").exists() {
                return Self::reuse_cache(
                    path,
                    feat_cols,
                    label_col,
                    label,
                    max_bins,
                    cols_per_block,
                    cache_path,
                );
            }
            anyhow::ensure!(
                !cache_path.exists() || std::fs::read_dir(cache_path)?.next().is_none(),
                "缓存路径 {} 已存在且不是有效的 ferrisboost 缓存",
                cache_path.display()
            );
        }
        let cuts = {
            let _p = crate::source::openprof::start("cuts");
            cuts_only_with_nthread(&path, label, max_bins, nthread)?
        };
        Self::with_cuts_inner(
            path,
            feat_cols,
            label_col,
            cuts,
            max_bins,
            cols_per_block,
            label,
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

    pub fn with_cuts_and_cache_and_nthread(
        path: impl AsRef<Path>,
        label: &str,
        cuts: BinCuts,
        cols_per_block: usize,
        options: QuantizedCacheOptions,
        nthread: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        let path = path.as_ref().to_path_buf();
        let (feat_cols, _) = feature_columns(&path, label)?;
        let label_col = column_index(&path, label)?;
        if let Some(cache_path) = options.path.as_ref() {
            if cache_path.join("manifest.json").exists() {
                let (src, labels) = Self::reuse_cache(
                    path,
                    feat_cols,
                    label_col,
                    label,
                    0,
                    cols_per_block,
                    cache_path,
                )?;
                // 逐值比,不只比哈希:manifest 里存着完整的 cuts,
                // 直接比是免费的,而且比任何哈希都强。哈希只用来发现
                // manifest 自身被改坏。
                anyhow::ensure!(
                    cuts.values == src.cuts().values && cuts.offsets == src.cuts().offsets,
                    "给定的 cuts 和缓存里的不一致:拒绝使用旧量化数据。\
                     验证集要用训练集的 cuts,但**不能和训练集共用 cache_path** —— \
                     它们的量化结果不同,共用会互相覆盖"
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
            label,
            options,
            nthread,
        )
    }

    fn with_cuts_inner(
        path: PathBuf,
        feat_cols: Vec<usize>,
        label_col: usize,
        cuts: BinCuts,
        max_bins: u32,
        cols_per_block: usize,
        label: &str,
        options: QuantizedCacheOptions,
        nthread: usize,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        let n_features = feat_cols.len();
        anyhow::ensure!(
            cuts.n_feats() == n_features,
            "cuts 是按 {} 个特征建的,文件里有 {n_features} 个",
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
        anyhow::ensure!(
            !labels.iter().any(|v| v.is_nan()),
            "标签列 {label:?} 里有缺失值"
        );

        let groups = crate::columns::partition_features(n_features, cols_per_block);
        let cache_bytes = n_rows
            .checked_mul(n_features)
            .ok_or_else(|| anyhow::anyhow!("量化缓存大小溢出"))?;
        let mut source = Self {
            path,
            cuts,
            groups,
            file_col: feat_cols,
            n_rows,
            n_features,
            cache: QuantizedCache::Memory(Vec::new()),
            cache_bytes,
            cache_reads: AtomicUsize::new(0),
            source_bytes: std::sync::atomic::AtomicU64::new(0),
            parquet_loads: AtomicUsize::new(0),
        };
        crate::source::openprof::shape(source.n_rows, source.n_features, source.groups.len());
        source.build_cache(label, max_bins, cols_per_block, options, nthread)?;
        Ok((source, labels))
    }

    fn build_cache(
        &mut self,
        label: &str,
        max_bins: u32,
        cols_per_block: usize,
        options: QuantizedCacheOptions,
        nthread: usize,
    ) -> anyhow::Result<()> {
        let budget = options.budget_bytes.unwrap_or_else(default_cache_budget);
        let pool = crate::threading::build_pool(nthread)
            .map_err(|e| anyhow::anyhow!("Parquet quantize 线程池创建失败:{e}"))?;
        if options.path.is_none() && self.cache_bytes <= budget {
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

        let (dir, delete_on_drop) = match options.path {
            Some(path) => (path, false),
            None => (
                std::env::temp_dir().join(format!(
                    "ferrisboost-cache-{}-{}",
                    std::process::id(),
                    TEMP_CACHE_ID.fetch_add(1, Ordering::Relaxed)
                )),
                true,
            ),
        };
        std::fs::create_dir_all(&dir)?;
        let block_hashes = {
            // 每个 worker 只持有自己正在量化的块,写完立即释放。量化和写盘
            // 在 worker 内相邻发生,两者会重叠,所以 profile 明确报合并墙钟,
            // 不把重叠的 worker 累计时间冒充阶段 wall。
            let _p = crate::source::openprof::start("quantize+write");
            pool.install(|| {
                (0..self.groups.len())
                    .into_par_iter()
                    .map(|i| -> anyhow::Result<String> {
                        let block = self.quantize_block(i)?;
                        let hash = block_hash(&block.data);
                        let tmp = dir.join(format!("block-{i}.bin.tmp"));
                        let final_path = dir.join(format!("block-{i}.bin"));
                        File::create(&tmp)?.write_all(&block.data)?;
                        std::fs::rename(tmp, final_path)?;
                        Ok(hash)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })?
        };
        let _p = crate::source::openprof::start("cache commit");
        let manifest = self.manifest(label, max_bins, cols_per_block, block_hashes.clone())?;
        let tmp = dir.join("manifest.json.tmp");
        File::create(&tmp)?.write_all(&serde_json::to_vec(&manifest)?)?;
        std::fs::rename(tmp, dir.join("manifest.json"))?;
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
        label: &str,
        max_bins: u32,
        cols_per_block: usize,
        block_hashes: Vec<String>,
    ) -> anyhow::Result<CacheManifest> {
        let (source_path, source_len, source_mtime_ns) = source_identity(&self.path)?;
        Ok(CacheManifest {
            version: 2,
            source_path,
            source_len,
            source_mtime_ns,
            label: label.to_string(),
            max_bins,
            cols_per_block,
            n_rows: self.n_rows,
            n_features: self.n_features,
            cuts_values_bits: self.cuts.values.iter().map(|v| v.to_bits()).collect(),
            cuts_offsets: self.cuts.offsets.clone(),
            cuts_hash: cuts_hash(&self.cuts),
            groups: self.groups.clone(),
            block_hashes,
        })
    }

    fn reuse_cache(
        path: PathBuf,
        feat_cols: Vec<usize>,
        label_col: usize,
        label: &str,
        max_bins: u32,
        cols_per_block: usize,
        cache_path: &Path,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        let raw = std::fs::read(cache_path.join("manifest.json"))?;
        let manifest: CacheManifest = serde_json::from_slice(&raw)?;
        anyhow::ensure!(
            manifest.version == 2,
            "不支持的量化缓存版本 {}（payload checksum 从版本 2 开始；请重建旧缓存）",
            manifest.version
        );
        let identity = source_identity(&path)?;
        anyhow::ensure!(
            (
                manifest.source_path.as_str(),
                manifest.source_len,
                manifest.source_mtime_ns
            ) == (identity.0.as_str(), identity.1, identity.2),
            "缓存对应的 Parquet 文件已经变化或不是当前文件"
        );
        anyhow::ensure!(manifest.label == label, "缓存标签列不一致");
        anyhow::ensure!(
            manifest.cols_per_block == cols_per_block,
            "缓存列块宽度不一致"
        );
        if max_bins != 0 {
            anyhow::ensure!(manifest.max_bins == max_bins, "缓存 max_bin 不一致");
        }
        anyhow::ensure!(manifest.n_features == feat_cols.len(), "缓存特征数不一致");
        let expected_groups =
            crate::columns::partition_features(manifest.n_features, cols_per_block);
        anyhow::ensure!(
            manifest.groups == expected_groups,
            "缓存列块映射损坏:groups 没有按 cols_per_block 覆盖全部特征"
        );
        anyhow::ensure!(
            manifest.block_hashes.len() == manifest.groups.len(),
            "缓存 block checksum 数量不一致"
        );
        anyhow::ensure!(
            manifest.cuts_offsets.len() == manifest.n_features + 1
                && manifest.cuts_offsets.first() == Some(&0)
                && manifest.cuts_offsets.last().copied()
                    == Some(manifest.cuts_values_bits.len() as u32)
                && manifest.cuts_offsets.windows(2).all(|w| w[0] <= w[1]),
            "缓存 cuts offsets 损坏"
        );
        let cuts = BinCuts::new(
            manifest
                .cuts_values_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect(),
            manifest.cuts_offsets.clone(),
        );
        anyhow::ensure!(
            cuts_hash(&cuts) == manifest.cuts_hash,
            "缓存 cuts 哈希不一致"
        );

        let mut labels = Vec::new();
        for batch in batch_reader(&path, &[label_col], BATCH_SIZE)? {
            labels.extend(crate::source::arrow_source::column_as_f32(&batch?, 0)?);
        }
        anyhow::ensure!(labels.len() == manifest.n_rows, "缓存行数不一致");
        for (i, group) in manifest.groups.iter().enumerate() {
            let expected = group
                .len()
                .checked_mul(manifest.n_rows)
                .unwrap_or(usize::MAX) as u64;
            let actual = std::fs::metadata(cache_path.join(format!("block-{i}.bin")))?.len();
            anyhow::ensure!(actual == expected, "缓存块 {i} 长度不一致");
        }
        Ok((
            Self {
                path,
                cuts,
                groups: manifest.groups,
                file_col: feat_cols,
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
                cache_bytes: manifest.n_rows * manifest.n_features,
                cache_reads: AtomicUsize::new(0),
                source_bytes: std::sync::atomic::AtomicU64::new(0),
                parquet_loads: AtomicUsize::new(0),
            },
            labels,
        ))
    }
}

static TEMP_CACHE_ID: AtomicUsize = AtomicUsize::new(0);

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
    format!("{hash:016x}")
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
            .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在"))?;
        // 全选(或空)就走原路,别为一个退化情况多写一条路径。
        if cols.is_empty() || cols.len() == feat_ids.len() {
            return self.with_block(i, f);
        }
        match &self.cache {
            QuantizedCache::Disk { dir, verified, .. } => {
                let already_verified = verified
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("缓存块 {i} 缺少 checksum 状态"))?
                    .load(Ordering::Acquire);
                if !already_verified {
                    // 第一次碰这个块:整块读 + 校验(付一次全量代价,
                    // 换掉"静默跳过校验")。
                    //
                    // ⚠️ 但**交回去的仍然只能是调用方要的那几列** ——
                    // 调用方已经按压实算好了 offsets,交回整块会直接对不上。
                    // 早一版就是在这里 `return self.with_block(i, f)`,
                    // 被一致性断言抓了个正着。
                    let mut whole = None;
                    self.with_block(i, &mut |b| {
                        whole = Some((b.data.clone(), b.feat_ids.clone()));
                    })?;
                    let (whole_data, whole_feats) = whole.expect("with_block 必须恰好调用一次 f");
                    let mut data = vec![0u8; cols.len() * self.n_rows];
                    for (slot, &l) in cols.iter().enumerate() {
                        anyhow::ensure!(l < whole_feats.len(), "列块 {i} 没有局部特征 {l}");
                        data[slot * self.n_rows..(slot + 1) * self.n_rows]
                            .copy_from_slice(&whole_data[l * self.n_rows..(l + 1) * self.n_rows]);
                    }
                    let block = ColumnBlock {
                        feat_ids: cols.iter().map(|&l| whole_feats[l]).collect(),
                        data,
                        n_rows: self.n_rows,
                    };
                    f(&block);
                    return Ok(());
                }
                use std::io::{Read, Seek, SeekFrom};
                let mut file = File::open(dir.join(format!("block-{i}.bin")))?;
                let mut data = vec![0u8; cols.len() * self.n_rows];
                for (slot, &l) in cols.iter().enumerate() {
                    anyhow::ensure!(l < feat_ids.len(), "列块 {i} 没有局部特征 {l}");
                    file.seek(SeekFrom::Start((l * self.n_rows) as u64))?;
                    file.read_exact(&mut data[slot * self.n_rows..(slot + 1) * self.n_rows])?;
                }
                self.cache_reads.fetch_add(1, Ordering::Relaxed);
                self.source_bytes
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                let block = ColumnBlock {
                    feat_ids: cols.iter().map(|&l| feat_ids[l]).collect(),
                    data,
                    n_rows: self.n_rows,
                };
                f(&block);
                Ok(())
            }
            // 内存 cache 本来就没有「读取」这件事,整块借出去是零拷贝,
            // 再挑列反而多一次复制。
            QuantizedCache::Memory(_) => self.with_block(i, f),
        }
    }

    /// 只有落盘(spill)的 cache 能真的少读:块文件是列主序,
    /// 一列就是一段连续字节。内存 cache 没有"读取"这件事。
    fn supports_column_selection(&self) -> bool {
        matches!(self.cache, QuantizedCache::Disk { .. })
    }

    fn source_bytes_read(&self) -> Option<u64> {
        Some(self.source_bytes.load(Ordering::Relaxed))
    }

    fn with_block(&self, i: usize, f: &mut dyn FnMut(&ColumnBlock)) -> anyhow::Result<()> {
        self.cache_reads.fetch_add(1, Ordering::Relaxed);
        match &self.cache {
            QuantizedCache::Memory(blocks) => {
                let block = blocks
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在"))?;
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
                    .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在"))?
                    .clone();
                let mut data = Vec::new();
                File::open(dir.join(format!("block-{i}.bin")))?.read_to_end(&mut data)?;
                anyhow::ensure!(
                    data.len() == feat_ids.len() * self.n_rows,
                    "缓存块 {i} 长度不一致"
                );
                // 整块读也如实计数 —— 只有把「读了多少」和「传了多少」
                // 分开记,才看得出省的到底是哪一段。
                self.source_bytes
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                let is_verified = verified
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("缓存块 {i} 缺少 checksum 状态"))?;
                if !is_verified.load(Ordering::Acquire) {
                    let expected = block_hashes
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("缓存块 {i} 缺少 checksum"))?;
                    let actual = block_hash(&data);
                    anyhow::ensure!(
                        &actual == expected,
                        "缓存块 {i} payload checksum 不一致:期望 {expected},实际 {actual}"
                    );
                    is_verified.store(true, Ordering::Release);
                }
                f(&ColumnBlock {
                    feat_ids,
                    data,
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
        self.groups
            .get(i)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在,一共 {} 个", self.groups.len()))
    }

    fn feature_column(&self, feat: FeatId) -> anyhow::Result<Vec<Bin>> {
        let (block_id, local) = self
            .groups
            .iter()
            .enumerate()
            .find_map(|(b, group)| group.iter().position(|&f| f == feat).map(|l| (b, l)))
            .ok_or_else(|| anyhow::anyhow!("特征 {feat} 越界,共 {} 列", self.n_features))?;
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
        let mut result = Vec::new();
        self.with_block_cols(block_id, &[local], &mut |block| {
            // 部分读时块里只有这一列;整块回退时仍要按 local 取。
            let col = if block.n_feats() == 1 { 0 } else { local };
            result.extend_from_slice(block.column(col));
        })?;
        Ok(result)
    }
}

impl ParquetBlockSource {
    /// 读若干列并量化,列主序写进 `out`。
    ///
    /// `block()` 和 `feature_column()` **共用这一份量化代码**。分成两份
    /// 写的话,两条路的 bin 迟早会飘 —— 而那会悄悄破坏位精确基线:
    /// 建直方图用的是块里的 bin,行重分区用的是单列的 bin,两者对不上
    /// 就等于训练和分区看到了不同的数据。
    fn read_into(
        &self,
        file_cols: &[usize],
        feat_ids: &[FeatId],
        n_rows: usize,
        out: &mut [Bin],
    ) -> anyhow::Result<()> {
        debug_assert_eq!(out.len(), feat_ids.len() * n_rows);
        let mut row_base = 0usize;
        for batch in batch_reader(&self.path, file_cols, BATCH_SIZE)? {
            let batch = batch?;
            let n = batch.num_rows();
            for (local, &f) in feat_ids.iter().enumerate() {
                let values = crate::source::arrow_source::column_as_f32(&batch, local)?;
                let dst = &mut out[local * n_rows + row_base..][..n];
                for (slot, v) in dst.iter_mut().zip(values) {
                    // null 和 NaN 都是缺失
                    *slot = if v.is_nan() {
                        MISSING_BIN
                    } else {
                        self.cuts.find_bin(f, v)
                    };
                }
            }
            row_base += n;
        }
        anyhow::ensure!(row_base == n_rows, "读到 {row_base} 行,元数据说 {n_rows}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::DenseSource;
    use crate::train::BlockSource as _;
    use arrow::array::{Float32Array, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
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
        assert_eq!(ds.source.n_features(), 3, "标签列不能算进特征");
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
                    "行 {row} 特征 {f}"
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
                "第 {row} 行:Parquet 训的和 Dense 训的对不上"
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
                        assert_eq!(a.feat_ids, e.feat_ids, "块 {b} 的特征列表");
                        assert_eq!(a.n_rows, e.n_rows);
                        assert_eq!(a.data, e.data, "块 {b} 的 bin 数据");
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
            for c in 0..n_cols {
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
                            src.feature_column(f).unwrap(),
                            block.column(local),
                            "{rows} 行 {cols} 列 cpb={cpb},特征 {f}:单列读和整块读不一致"
                        );
                        checked += 1;
                    }
                })
                .unwrap();
            }
            assert_eq!(checked, cols, "每一列都要比到");
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
                "nthread={threads}:每个块必须且只能量化一次"
            );
        }
        for b in 0..one.n_blocks() {
            one.with_block(b, &mut |a| {
                eight
                    .with_block(b, &mut |e| assert_eq!(a.data, e.data, "块 {b}:1 vs 8"))
                    .unwrap();
                rayon_default
                    .with_block(b, &mut |d| assert_eq!(a.data, d.data, "块 {b}:1 vs 0"))
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
        assert_eq!(src.parquet_load_count(), src.n_blocks(), "每块只量化一次");
        assert_eq!(src.cache_read_count(), 0);

        let mut seen = Vec::new();
        src.with_block(0, &mut |b| seen.push(b.data.clone()))
            .unwrap();
        assert_eq!(src.cache_read_count(), 1);

        src.with_block(0, &mut |b| seen.push(b.data.clone()))
            .unwrap();
        assert_eq!(src.cache_read_count(), 2);
        assert_eq!(src.parquet_load_count(), src.n_blocks(), "重复访问不能回源");
        assert_eq!(seen[0], seen[1], "两次读出来的内容当然要一样");

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
        assert_eq!(second.parquet_load_count(), 0, "第二次不能重新量化");
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
            .expect("不同 cuts 必须拒绝");
        // 现在是**逐值**比较,不再只比哈希,所以报错措辞也换了
        assert!(err.to_string().contains("拒绝使用旧量化数据"), "{err:#}");
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
            panic!("半成品缓存目录必须报错");
        };
        assert!(
            e.to_string().contains("不是有效的 ferrisboost 缓存"),
            "报错要说清楚是缓存目录的问题,实际是:{e}"
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
        let err = reused
            .with_block(0, &mut |_| {})
            .expect_err("同长度 payload 损坏必须被 checksum 抓住");
        assert!(
            err.to_string().contains("payload checksum 不一致"),
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
            panic!("cuts 不一致却共用 cache_path,必须报错");
        };
        assert!(e.to_string().contains("拒绝使用旧量化数据"), "{e}");

        // 同一套 cuts 则可以复用,而且不再回源量化
        let (again, _) = ParquetBlockSource::with_cuts_and_cache(
            &path,
            "target",
            train.cuts().clone(),
            2,
            opts(&dir),
        )
        .unwrap();
        assert_eq!(again.parquet_load_count(), 0, "复用时不该再量化");

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
                "第 {i} 行:按需加载和全量加载训出来的不一样"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_label_column_says_what_is_available() {
        let path = write_fixture("missing_label");
        let Err(e) = open(&path, "nope", 16, 2) else {
            panic!("应该报错")
        };
        let msg = e.to_string();
        assert!(
            msg.contains("nope") && msg.contains("f0"),
            "报错要列出有哪些列:{msg}"
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
}
