//! 从 Arrow RecordBatch 构建列块。
//!
//! Arrow 是**唯一**的输入接口:Parquet、S3、DuckDB、Polars、任何数据库
//! 的 ADBC 驱动都能吐 RecordBatch,接住这一层就等于全都支持了。
//!
//! 两遍扫描:第一遍建草图定分箱边界,第二遍按边界量化。两遍都按
//! batch 流式处理,不需要把整个数据集拉进内存 —— 这正是「内存需求从
//! O(数据集) 降到 O(列块)」那条主张在输入侧的落点。

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::columns::{partition_features, BinCuts, ColumnBlock};
use crate::sketch::SketchSet;
use crate::types::{Bin, FeatId, MAX_BIN_LIMIT, MISSING_BIN};

/// A query-id column as u64. Integer columns of any width, and floating point
/// columns holding integral values; nulls and negative values are errors.
pub(crate) fn column_as_u64(
    batch: &RecordBatch,
    col: usize,
    name: &str,
) -> anyhow::Result<Vec<u64>> {
    use arrow::array::{cast::AsArray, ArrayRef};
    use arrow::compute::cast;
    use arrow::datatypes::{Float64Type, UInt64Type};

    let array: &ArrayRef = batch.column(col);
    crate::ensure_input!(
        array.null_count() == 0,
        "query-id column {name:?} has missing values"
    );
    let ty = array.data_type();
    if matches!(ty, DataType::Float32 | DataType::Float64) {
        let as_f64 = cast(array, &DataType::Float64)?;
        let values = as_f64.as_primitive::<Float64Type>();
        let mut out = Vec::with_capacity(values.len());
        for &v in values.values().iter() {
            crate::ensure_input!(
                v.is_finite() && v >= 0.0 && v.fract() == 0.0,
                "query-id column {name:?} must hold non-negative integers, found {v}"
            );
            out.push(v as u64);
        }
        return Ok(out);
    }
    crate::ensure_input!(
        ty.is_integer(),
        "query-id column {name:?} must be an integer column, found {ty:?}"
    );
    let as_u64 = cast(array, &DataType::UInt64)?;
    crate::ensure_input!(
        as_u64.null_count() == 0,
        "query-id column {name:?} must hold non-negative integers"
    );
    Ok(as_u64.as_primitive::<UInt64Type>().values().to_vec())
}

/// 把一列取成 f32,null 变 NaN。
///
/// 只认数值列。**null 和 NaN 走同一条路**(都变成 MISSING_BIN),
/// 因为 XGBoost 的语义里两者都是"缺失",分裂时由 default direction
/// 决定去向 —— 不是填 0,更不是丢掉整行。
pub(crate) fn column_as_f32(batch: &RecordBatch, col: usize) -> anyhow::Result<Vec<f32>> {
    let mut output = Vec::with_capacity(batch.num_rows());
    for_each_column_f32(batch, col, |value| output.push(value))?;
    Ok(output)
}

/// 预测输入用:±inf 原样保留(只决定走哪个分支,和 XGBoost 预测一致)。
/// 训练入口用 `column_as_f32`,它拒绝 ±inf。
pub(crate) fn column_as_f32_allow_inf(batch: &RecordBatch, col: usize) -> anyhow::Result<Vec<f32>> {
    let mut output = Vec::with_capacity(batch.num_rows());
    visit_column_f32::<false, false>(batch, col, f32::NAN, |value| output.push(value))?;
    Ok(output)
}

/// Visit one numeric Arrow column as f32 without allocating an intermediate
/// buffer. Values, null handling, and casts are identical to `column_as_f32`.
///
/// 训练入口:±inf(含 f64 换成 f32 时溢出的值)报错,XGBoost 同样拒绝。
pub(crate) fn for_each_column_f32(
    batch: &RecordBatch,
    col: usize,
    visit: impl FnMut(f32),
) -> anyhow::Result<()> {
    visit_column_f32::<true, false>(batch, col, f32::NAN, visit)
}

/// Features reserve f32::MAX for the finite missing/nonmissing split sentinel.
/// Labels and weights use the ordinary numeric visitor and are not restricted.
///
/// `missing` is the XGBoost `missing` sentinel: a value equal to it (after the
/// cast to f32) is visited as NaN, exactly like a null. NaN means no sentinel.
/// This is the single decode point for training features, so the sketch and
/// the quantizer both see sentinel values as missing and never as a bin.
pub(crate) fn for_each_feature_f32(
    batch: &RecordBatch,
    col: usize,
    missing: f32,
    visit: impl FnMut(f32),
) -> anyhow::Result<()> {
    visit_column_f32::<true, true>(batch, col, missing, visit)
}

pub(crate) fn feature_as_f32(
    batch: &RecordBatch,
    col: usize,
    missing: f32,
) -> anyhow::Result<Vec<f32>> {
    let mut output = Vec::with_capacity(batch.num_rows());
    for_each_feature_f32(batch, col, missing, |value| output.push(value))?;
    Ok(output)
}

fn visit_column_f32<const REJECT_INF: bool, const REJECT_MAX: bool>(
    batch: &RecordBatch,
    col: usize,
    missing: f32,
    mut visit: impl FnMut(f32),
) -> anyhow::Result<()> {
    // A dictionary column carries category codes: visit its decoded values.
    // Only numeric dictionaries are stable codes; string dictionaries have a
    // per-batch ordering that cannot be turned into consistent codes here.
    let decoded;
    let array = match batch.column(col).data_type() {
        DataType::Dictionary(_, value) => {
            if !(value.is_numeric()) {
                let name = batch.schema().field(col).name().clone();
                crate::bail_input!(
                    "column {name:?} is a dictionary of {value:?} values: categorical \
                     file columns must hold integer category codes (the codes of a string \
                     dictionary differ between row groups). Encode the categories as integers \
                     upstream, or train from a pandas / PyArrow in-memory table"
                );
            }
            decoded = arrow::compute::cast(batch.column(col), &DataType::Float64)?;
            &decoded
        }
        _ => batch.column(col),
    };
    let n = array.len();

    macro_rules! visit_values {
        ($ty:ty) => {{
            let values = array
                .as_any()
                .downcast_ref::<$ty>()
                .expect("类型已经 match 过了");
            let mut any_inf = false;
            let mut any_max = false;
            for i in 0..n {
                let v = if values.is_null(i) {
                    f32::NAN
                } else {
                    values.value(i) as f32
                };
                any_inf |= v.is_infinite();
                if REJECT_MAX { any_max |= v == f32::MAX; }
                // `missing` is finite or NaN (validated upstream), so it never
                // equals an inf / f32::MAX that the checks above reject.
                visit(if v == missing { f32::NAN } else { v });
            }
            if REJECT_MAX && any_max {
                // 报列名:`col` 是这一批里的列号,文件只读了部分列时它不是文件里的列号。
                let name = batch.schema().field(col).name().clone();
                crate::bail_input!("feature column {name:?} contains f32::MAX, reserved for missing-value split boundaries; rescale this feature");
            }
            if REJECT_INF && any_inf {
                let i = (0..n)
                    .find(|&i| !values.is_null(i) && (values.value(i) as f32).is_infinite())
                    .expect("any_inf");
                crate::bail_input!(
                    "第 {col} 列第 {i} 行是 {}(换成 f32 是 {}):不支持 ±inf \
                     或超出 f32 范围的值(XGBoost 同样拒绝);缺失值请用 null / NaN",
                    values.value(i),
                    values.value(i) as f32
                );
            }
        }};
    }

    match array.data_type() {
        DataType::Float32 => visit_values!(Float32Array),
        DataType::Float64 => visit_values!(Float64Array),
        DataType::Int32 => visit_values!(Int32Array),
        DataType::Int64 => visit_values!(Int64Array),
        other => crate::bail_input!(
            "第 {col} 列是 {other:?},只支持数值列。\\
             类别特征要先在上游编码成整数类别码(0..=253),再用 feature_types=\"c\" 声明"
        ),
    }
    Ok(())
}

/// 第一遍:扫描建草图,得到分箱边界。
///
/// 每个 batch 喂进同一组草图。草图本身是流式的(见 sketch.rs 的层级
/// 归并),所以这一遍的内存是 O(特征数 × 草图大小),和行数无关。
pub fn build_cuts(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    max_bins: usize,
) -> anyhow::Result<BinCuts> {
    build_cuts_with_nthread(batches, max_bins, crate::source::DEFAULT_INGEST_THREADS)
}

/// `build_cuts`,但同一 batch 内按特征并行。`nthread=0` 交给 Rayon
/// 使用可用线程数。
///
/// batch 仍然严格顺序消费,每个特征也始终由同一个草图按 batch 顺序
/// 接收数据。这样不需要合并草图,线程数不会改变 cuts。
pub fn build_cuts_with_nthread(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    max_bins: usize,
    nthread: usize,
) -> anyhow::Result<BinCuts> {
    build_cuts_and_count_with_nthread(batches, max_bins, nthread).map(|(cuts, _)| cuts)
}

pub(crate) fn build_cuts_and_count_with_nthread(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    max_bins: usize,
    nthread: usize,
) -> anyhow::Result<(BinCuts, usize)> {
    build_cuts_and_count_weighted(batches, max_bins, nthread, false)
}

/// 加权分箱(XGBoost hist 的 sketch 同样按样本权重加权)。
///
/// `weighted = true` 时每个 batch 的**最后一列**是样本权重,不参与分箱;
/// 其余列是特征。权重必须有限且非负,0 权重的行不影响 cuts。
/// 全 1 权重和不加权的 cuts 逐位相同(不加权时本来就按权重 1 插入)。
pub(crate) fn build_cuts_and_count_weighted(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    max_bins: usize,
    nthread: usize,
    weighted: bool,
) -> anyhow::Result<(BinCuts, usize)> {
    build_cuts_and_count_typed(batches, max_bins, nthread, weighted, &[], f32::NAN)
}

/// `build_cuts_and_count_weighted` with categorical features.
///
/// `categorical[f]` marks feature `f` (in batch column order, weights
/// excluded) as categorical; an empty slice means "none declared", and then
/// dictionary-typed columns are detected as categorical. A non-empty list is
/// authoritative (a dictionary column declared "q" is numeric). A categorical
/// feature's cuts are one bin per integer code `0..=max_code`.
///
/// Feature values equal to `missing` are skipped by the sketch like NaN (NaN =
/// no sentinel), so a sentinel never becomes a bin boundary or a category.
pub(crate) fn build_cuts_and_count_typed(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    max_bins: usize,
    nthread: usize,
    weighted: bool,
    categorical: &[bool],
    missing: f32,
) -> anyhow::Result<(BinCuts, usize)> {
    let scan_profile = crate::source::openprof::start("cuts scan");
    assert!(
        max_bins as u32 <= MAX_BIN_LIMIT,
        "max_bin 上限是 {MAX_BIN_LIMIT}(255 留给缺失哨兵),给的是 {max_bins}"
    );

    // 整次扫描只建一个 pool。每 batch 重建一次在线程数较多、batch 很小
    // 时会让调度开销盖过 sketch 本身。
    let pool = crate::threading::build_pool(nthread)
        .map_err(|e| anyhow::anyhow!("Parquet cuts 线程池创建失败:{e}"))?;
    let profile_columns = pool.current_num_threads() == 1;
    let mut sketches: Option<SketchSet> = None;
    let mut feature_names: Vec<String> = Vec::new();
    let mut n_rows = 0usize;
    // 手动迭代:`next()` 里才是 Parquet 解压,要和 sketch 分开计时。
    let mut batches = batches;
    loop {
        let next = {
            let _p = crate::source::openprof::sub("read+decode");
            batches.next()
        };
        let Some(batch) = next else { break };
        let batch = batch?;
        let row_offset = n_rows;
        n_rows = n_rows
            .checked_add(batch.num_rows())
            .ok_or_else(|| anyhow::anyhow!("输入行数溢出 usize"))?;
        let n_feats = batch.num_columns() - usize::from(weighted);
        let weights: Option<Vec<f32>> = if weighted {
            let mut w = Vec::with_capacity(batch.num_rows());
            for_each_column_f32(&batch, n_feats, |v| w.push(v))?;
            if let Some(i) = w.iter().position(|v| !v.is_finite() || *v < 0.0) {
                crate::bail_input!(
                    "样本权重第 {} 行是 {};权重必须是有限的非负数",
                    row_offset + i,
                    w[i]
                );
            }
            Some(w)
        } else {
            None
        };
        let push_column = |col: usize, sketch: &mut crate::sketch::FeatureSketch| match &weights {
            None => for_each_feature_f32(&batch, col, missing, |value| sketch.push(value, 1.0)),
            Some(w) => {
                let mut row = 0usize;
                for_each_feature_f32(&batch, col, missing, |value| {
                    sketch.push(value, w[row]);
                    row += 1;
                })
            }
        };
        if sketches.is_none() {
            crate::ensure_input!(
                categorical.is_empty() || categorical.len() == n_feats,
                "feature_types has {} entries but the data has {n_feats} feature columns",
                categorical.len()
            );
            let schema = batch.schema();
            feature_names = schema
                .fields()
                .iter()
                .take(n_feats)
                .map(|f| f.name().clone())
                .collect();
            // An explicit feature_types list decides; otherwise dictionary-typed
            // columns are detected as categorical (same rule as in-memory input).
            let flags: Vec<bool> = (0..n_feats)
                .map(|f| match categorical.get(f) {
                    Some(&declared) => declared,
                    None => matches!(schema.field(f).data_type(), DataType::Dictionary(_, _)),
                })
                .collect();
            sketches = Some(SketchSet::with_categorical(n_feats, max_bins, &flags));
        }
        let sk = sketches.as_mut().expect("created above");
        anyhow::ensure!(
            n_feats == sk.n_feats(),
            "batch 之间的列数不一致:先前 {},这个 batch {n_feats}",
            sk.n_feats()
        );
        if profile_columns {
            // Single-thread profiling keeps one wall bucket around conversion
            // and sketch insertion. Direct visitation intentionally removes the
            // materialized f32 buffer that used to separate those operations.
            for (col, sketch) in sk.per_feat.iter_mut().enumerate() {
                let _p = crate::source::openprof::sub("convert+sketch");
                push_column(col, sketch)?;
            }
        } else {
            // 这里的计时是整个 batch 的 elapsed wall,不是各 worker 用时之和。
            let _p = crate::source::openprof::sub("parallel batch");
            pool.install(|| {
                sk.per_feat.par_iter_mut().enumerate().try_for_each(
                    |(col, sketch)| -> anyhow::Result<()> { push_column(col, sketch) },
                )
            })?;
        }
    }
    drop(scan_profile);
    let _finalize_profile = crate::source::openprof::start("cuts finalize");
    let cuts = sketches
        .ok_or_else(|| crate::input_err!("一个 batch 都没有,建不了分箱边界(数据没有行)"))?
        .finalize_checked(&feature_names)?;
    Ok((cuts, n_rows))
}

/// 第二遍:按 cuts 量化,产出列块。
///
/// 列主序,`data[local_feat * n_rows + row]`,和 DenseSource 出来的
/// 布局完全一样 —— 训练侧看不出数据是从哪来的。
pub fn quantize(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    cuts: &BinCuts,
    cols_per_block: usize,
) -> anyhow::Result<(Vec<ColumnBlock>, usize)> {
    quantize_with_nthread(
        batches,
        cuts,
        cols_per_block,
        crate::source::DEFAULT_INGEST_THREADS,
        f32::NAN,
    )
}

/// Quantize one bounded RecordBatch at a time while distributing independent
/// column blocks over an ingest-only Rayon pool.  Output columns and rows are
/// appended in canonical order, so scheduling cannot affect the bins.
pub fn quantize_with_nthread(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    cuts: &BinCuts,
    cols_per_block: usize,
    nthread: usize,
    missing: f32,
) -> anyhow::Result<(Vec<ColumnBlock>, usize)> {
    assert!(cols_per_block > 0, "cols_per_block 必须 > 0");
    let n_features = cuts.n_feats();
    let groups = partition_features(n_features, cols_per_block);

    // 每个块先攒成"按特征分段"的 Vec,行数未知所以只能边走边 push。
    // 列主序的好处在这里也成立:同一个特征的 bin 是连续追加的。
    let mut per_block: Vec<Vec<Vec<Bin>>> =
        groups.iter().map(|g| vec![Vec::new(); g.len()]).collect();
    let mut n_rows = 0usize;
    let pool = crate::threading::build_pool(nthread)
        .map_err(|e| anyhow::anyhow!("Arrow quantize ingest 线程池创建失败:{e}"))?;

    for batch in batches {
        let batch = batch?;
        anyhow::ensure!(
            batch.num_columns() == n_features,
            "batch 有 {} 列,cuts 是按 {n_features} 列建的",
            batch.num_columns()
        );
        n_rows += batch.num_rows();

        pool.install(|| {
            per_block
                .par_iter_mut()
                .zip(groups.par_iter())
                .try_for_each(|(block, feats)| -> anyhow::Result<()> {
                    for (local, &f) in feats.iter().enumerate() {
                        let dst = &mut block[local];
                        dst.reserve(batch.num_rows());
                        for_each_feature_f32(&batch, f as usize, missing, |value| {
                            // null, NaN and the `missing` sentinel share MISSING_BIN.
                            dst.push(if value.is_nan() {
                                MISSING_BIN
                            } else {
                                cuts.find_bin(f, value)
                            });
                        })?;
                    }
                    Ok(())
                })
        })?;
    }

    let blocks = groups
        .into_iter()
        .zip(per_block)
        .map(|(feat_ids, cols)| {
            let mut data = Vec::with_capacity(feat_ids.len() * n_rows);
            for col in cols {
                debug_assert_eq!(col.len(), n_rows, "某一列的行数和别的列对不上");
                data.extend_from_slice(&col);
            }
            ColumnBlock {
                feat_ids,
                data,
                n_rows,
            }
        })
        .collect();

    Ok((blocks, n_rows))
}

/// Quantize into final column-major blocks when the first scan already
/// established the exact row count. This avoids per-column growth buffers and
/// the final full-width concatenation copy.
fn quantize_with_nthread_known_rows(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    cuts: &BinCuts,
    cols_per_block: usize,
    n_rows: usize,
    nthread: usize,
    missing: f32,
) -> anyhow::Result<Vec<ColumnBlock>> {
    let allocation_profile = crate::source::openprof::start("bins allocate");
    assert!(cols_per_block > 0, "cols_per_block 必须 > 0");
    let groups = partition_features(cuts.n_feats(), cols_per_block);
    let mut blocks = groups
        .iter()
        .map(|features| {
            let len = features
                .len()
                .checked_mul(n_rows)
                .ok_or_else(|| anyhow::anyhow!("量化块大小溢出 usize"))?;
            Ok(ColumnBlock {
                feat_ids: features.clone(),
                data: vec![MISSING_BIN; len],
                n_rows,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    drop(allocation_profile);
    let _scan_profile = crate::source::openprof::start("bins scan");
    let pool = crate::threading::build_pool(nthread)
        .map_err(|e| anyhow::anyhow!("Arrow quantize ingest 线程池创建失败:{e}"))?;
    let mut row_offset = 0usize;

    let mut batches = batches;
    loop {
        let next = {
            let _p = crate::source::openprof::sub("bins read+decode");
            batches.next()
        };
        let Some(batch) = next else { break };
        let batch = batch?;
        anyhow::ensure!(
            batch.num_columns() == cuts.n_feats(),
            "batch 有 {} 列,cuts 是按 {} 列建的",
            batch.num_columns(),
            cuts.n_feats()
        );
        let batch_rows = batch.num_rows();
        let row_end = row_offset
            .checked_add(batch_rows)
            .ok_or_else(|| anyhow::anyhow!("输入行数溢出 usize"))?;
        anyhow::ensure!(
            row_end <= n_rows,
            "第二遍扫描行数超过第一遍:{row_end} > {n_rows}"
        );

        let _transform_profile = crate::source::openprof::sub("bins transform");
        pool.install(|| {
            blocks.par_iter_mut().zip(groups.par_iter()).try_for_each(
                |(block, features)| -> anyhow::Result<()> {
                    for (local, &feature) in features.iter().enumerate() {
                        let start = local * n_rows + row_offset;
                        let output = &mut block.data[start..start + batch_rows];
                        let mut index = 0usize;
                        for_each_feature_f32(&batch, feature as usize, missing, |value| {
                            output[index] = if value.is_nan() {
                                MISSING_BIN
                            } else {
                                cuts.find_bin(feature, value)
                            };
                            index += 1;
                        })?;
                        debug_assert_eq!(index, batch_rows);
                    }
                    Ok(())
                },
            )
        })?;
        row_offset = row_end;
    }
    anyhow::ensure!(
        row_offset == n_rows,
        "两遍扫描行数不一致:第一遍 {n_rows},第二遍 {row_offset}"
    );
    Ok(blocks)
}

/// Arrow 来的数据源。布局和 `DenseSource` 完全一致 —— 训练侧看不出
/// 数据是从内存矩阵来的还是从 Parquet 流进来的。
///
/// 这一版**量化结果全驻内存**(u8 的 bin,比原始 f32 小 4 倍)。
/// 按需从对象存储加载列块是 `ParquetBlockSource` 的事。
pub struct ArrowSource {
    cuts: BinCuts,
    blocks: Vec<ColumnBlock>,
    n_rows: usize,
    n_features: usize,
}

impl ArrowSource {
    /// 两遍扫描:先建草图定边界,再量化。
    ///
    /// `batches` 是个工厂而不是迭代器 —— 要扫两遍,而 RecordBatch 的
    /// 迭代器通常是一次性的(Parquet reader、网络流都是)。让调用方
    /// 提供"再来一遍"的能力,比我们把所有 batch 缓存下来诚实得多:
    /// 缓存就等于把整个数据集拉进内存,正是这个项目要避免的。
    pub fn new<I, F>(batches: F, max_bins: u32, cols_per_block: usize) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        Self::new_with_nthread(
            batches,
            max_bins,
            cols_per_block,
            crate::source::DEFAULT_INGEST_THREADS,
        )
    }

    pub fn new_with_nthread<I, F>(
        mut batches: F,
        max_bins: u32,
        cols_per_block: usize,
        nthread: usize,
    ) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        Self::new_with_phase_threads(|_| batches(), max_bins, cols_per_block, nthread, nthread)
    }

    pub(crate) fn new_with_phase_threads<I, F>(
        batches: F,
        max_bins: u32,
        cols_per_block: usize,
        cuts_threads: usize,
        quantize_threads: usize,
    ) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut(bool) -> anyhow::Result<I>,
    {
        Self::new_typed_with_phase_threads(
            batches,
            max_bins,
            cols_per_block,
            cuts_threads,
            quantize_threads,
            &[],
            f32::NAN,
        )
    }

    /// `new_with_phase_threads` with declared categorical features.
    pub(crate) fn new_typed_with_phase_threads<I, F>(
        mut batches: F,
        max_bins: u32,
        cols_per_block: usize,
        cuts_threads: usize,
        quantize_threads: usize,
        categorical: &[bool],
        missing: f32,
    ) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut(bool) -> anyhow::Result<I>,
    {
        let (cuts, n_rows) = build_cuts_and_count_typed(
            batches(false)?,
            max_bins as usize,
            cuts_threads,
            false,
            categorical,
            missing,
        )?;
        let blocks = quantize_with_nthread_known_rows(
            batches(true)?,
            &cuts,
            cols_per_block,
            n_rows,
            quantize_threads,
            missing,
        )?;
        let n_features = cuts.n_feats();
        Ok(Self {
            cuts,
            blocks,
            n_rows,
            n_features,
        })
    }

    /// 用给定的 cuts 量化,只扫一遍。
    ///
    /// **验证集走这条**(和 `DenseSource::with_cuts` 同理):自己重新
    /// 建草图会让同一个特征值在训练/验证里落进不同的 bin,模型直接
    /// 失效,而且跑起来不报错,只是指标莫名其妙地差。
    pub fn with_cuts<I, F>(batches: F, cuts: BinCuts, cols_per_block: usize) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        Self::with_cuts_and_nthread(
            batches,
            cuts,
            cols_per_block,
            crate::source::DEFAULT_INGEST_THREADS,
        )
    }

    pub fn with_cuts_and_nthread<I, F>(
        mut batches: F,
        cuts: BinCuts,
        cols_per_block: usize,
        nthread: usize,
    ) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        Self::from_batches_with_cuts_and_nthread(
            batches()?,
            cuts,
            cols_per_block,
            nthread,
            f32::NAN,
        )
    }

    /// `with_cuts_and_nthread` with the XGBoost `missing` sentinel.
    pub(crate) fn with_cuts_and_nthread_missing<I, F>(
        mut batches: F,
        cuts: BinCuts,
        cols_per_block: usize,
        nthread: usize,
        missing: f32,
    ) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        Self::from_batches_with_cuts_and_nthread(batches()?, cuts, cols_per_block, nthread, missing)
    }

    pub(crate) fn from_batches_with_cuts_and_nthread(
        batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
        cuts: BinCuts,
        cols_per_block: usize,
        nthread: usize,
        missing: f32,
    ) -> anyhow::Result<Self> {
        let (blocks, n_rows) =
            quantize_with_nthread(batches, &cuts, cols_per_block, nthread, missing)?;
        let n_features = cuts.n_feats();
        Ok(Self {
            cuts,
            blocks,
            n_rows,
            n_features,
        })
    }

    pub(crate) fn from_batches_with_cuts_and_nthread_known_rows(
        batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
        cuts: BinCuts,
        cols_per_block: usize,
        n_rows: usize,
        nthread: usize,
        missing: f32,
    ) -> anyhow::Result<Self> {
        let blocks = quantize_with_nthread_known_rows(
            batches,
            &cuts,
            cols_per_block,
            n_rows,
            nthread,
            missing,
        )?;
        let n_features = cuts.n_feats();
        Ok(Self {
            cuts,
            blocks,
            n_rows,
            n_features,
        })
    }

    pub fn into_cuts(self) -> BinCuts {
        self.cuts
    }

    /// 某一行某个特征的 bin。测试和调试用。
    pub fn bin_at(&self, row: usize, feat: FeatId) -> Bin {
        for b in &self.blocks {
            if let Some(local) = b.feat_ids.iter().position(|&f| f == feat) {
                return b.column(local)[row];
            }
        }
        panic!("特征 {feat} 不在任何列块里");
    }
}

impl crate::train::BlockSource for ArrowSource {
    fn n_blocks(&self) -> usize {
        self.blocks.len()
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

    /// 纯元数据,不碰数据。
    fn block_features(&self, i: usize) -> anyhow::Result<Vec<FeatId>> {
        self.blocks
            .get(i)
            .map(|b| b.feat_ids.clone())
            .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在,共 {} 个", self.blocks.len()))
    }

    /// 直接从常驻的块里切一列出来。
    ///
    /// **必须覆盖 trait 的默认实现** —— 默认实现是「`block(i)` 拿所有权
    /// 再切一列」,对全量驻留的 source 就是「为了 1 列克隆整块」。
    /// HIGGS 上 28 列 × 1050 万行 = 每次克隆 294MB,行重分区因此吃掉
    /// 单轮一半的时间(profile 抓出来的,见 CLAUDE.md「6.9× 花在哪」)。
    fn feature_column(&self, feat: FeatId) -> anyhow::Result<Vec<Bin>> {
        for b in &self.blocks {
            if let Some(local) = b.feat_ids.iter().position(|&f| f == feat) {
                return Ok(b.column(local).to_vec());
            }
        }
        anyhow::bail!("特征 {feat} 不在任何列块里,共 {} 个特征", self.n_features)
    }

    /// 驻留的块直接借出列,零拷贝(见 trait 文档)。
    fn has_resident_columns(&self) -> bool {
        true
    }

    fn resident_feature_column(&self, feat: FeatId) -> Option<&[Bin]> {
        self.blocks.iter().find_map(|b| {
            b.feat_ids
                .iter()
                .position(|&x| x == feat)
                .map(|local| b.column(local))
        })
    }

    /// 量化结果全量持有,直接借出去,**零拷贝**。ArrowSource 收的是
    /// "喂进来的 batch",没有回头再读的能力,所以做不了按需加载 ——
    /// 那是 `ParquetBlockSource` 的事(它能按 column chunk 重新读)。
    fn with_block(&self, i: usize, f: &mut dyn FnMut(&ColumnBlock)) -> anyhow::Result<()> {
        let block = self
            .blocks
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在,一共 {} 个", self.blocks.len()))?;
        f(block);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::DenseSource;
    use crate::train::BlockSource as _;
    use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    /// 3 列 × 6 行,拆成两个 batch;第 2 列带 null。
    fn batches() -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, false),
            Field::new("b", DataType::Float64, true),
            Field::new("c", DataType::Int64, false),
        ]));
        let mk = |a: Vec<f32>, b: Vec<Option<f64>>, c: Vec<i64>| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Float32Array::from(a)),
                    Arc::new(Float64Array::from(b)),
                    Arc::new(Int64Array::from(c)),
                ],
            )
            .unwrap()
        };
        vec![
            mk(
                vec![1.0, 2.0, 3.0],
                vec![Some(10.0), None, Some(30.0)],
                vec![7, 8, 9],
            ),
            mk(
                vec![4.0, 5.0, 6.0],
                vec![Some(40.0), Some(50.0), None],
                vec![1, 2, 3],
            ),
        ]
    }

    fn feed(bs: &[RecordBatch]) -> impl Iterator<Item = anyhow::Result<RecordBatch>> + '_ {
        bs.iter().cloned().map(Ok)
    }

    #[test]
    fn cut_sketch_is_thread_count_independent() {
        let bs = batches();
        let one = build_cuts_with_nthread(feed(&bs), 16, 1).unwrap();
        let eight = build_cuts_with_nthread(feed(&bs), 16, 8).unwrap();
        let rayon_default = build_cuts_with_nthread(feed(&bs), 16, 0).unwrap();

        assert_eq!(one.offsets, eight.offsets);
        assert_eq!(
            one.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            eight.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(one.offsets, rayon_default.offsets);
        assert_eq!(
            one.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            rayon_default
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    /// 同一份数据,从 Arrow 进和从稠密矩阵进,量化结果必须**逐位相同**。
    ///
    /// 这是换输入层最要紧的一条:阶段 1a 的位精确基线是拿 DenseSource
    /// 建的,只要这一条成立,训练那一半完全不用重验。
    #[test]
    fn arrow_and_dense_agree_bin_for_bin() {
        let bs = batches();
        let src = ArrowSource::new(|| Ok(feed(&bs)), 16, 2).unwrap();

        // 同样的数据摆成行主序喂 DenseSource,null 用 NaN
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

        assert_eq!(src.n_rows(), 6);
        assert_eq!(src.n_features(), 3);
        assert_eq!(src.n_blocks(), dense.n_blocks());
        for row in 0..6 {
            for f in 0..3u32 {
                assert_eq!(
                    src.bin_at(row, f),
                    dense.bin_at(row, f),
                    "行 {row} 特征 {f}:Arrow 和 Dense 的 bin 不一致"
                );
            }
        }
        // 分箱边界本身也要一样
        for f in 0..3u32 {
            assert_eq!(
                src.cuts().cuts_for(f),
                dense.cuts().cuts_for(f),
                "特征 {f} 的 cuts"
            );
        }
    }

    /// null 和 NaN 都要变成缺失哨兵,不能填 0 也不能丢行。
    #[test]
    fn nulls_become_the_missing_sentinel() {
        let bs = batches();
        let src = ArrowSource::new(|| Ok(feed(&bs)), 16, 3).unwrap();
        // 第 2 列的第 1 行和第 5 行是 null
        assert_eq!(src.bin_at(1, 1), MISSING_BIN);
        assert_eq!(src.bin_at(5, 1), MISSING_BIN);
        // 其余行不是缺失
        for row in [0usize, 2, 3, 4] {
            assert_ne!(src.bin_at(row, 1), MISSING_BIN, "行 {row} 不该是缺失");
        }
    }

    /// batch 的切法不该影响结果 —— 一个 batch 和三个 batch 一样。
    #[test]
    fn batch_boundaries_do_not_change_anything() {
        let bs = batches();
        let split = ArrowSource::new(|| Ok(feed(&bs)), 16, 2).unwrap();

        let merged = arrow::compute::concat_batches(&bs[0].schema(), &bs).unwrap();
        let one = vec![merged];
        let whole = ArrowSource::new(|| Ok(feed(&one)), 16, 2).unwrap();

        assert_eq!(split.n_rows(), whole.n_rows());
        for row in 0..6 {
            for f in 0..3u32 {
                assert_eq!(
                    split.bin_at(row, f),
                    whole.bin_at(row, f),
                    "行 {row} 特征 {f}"
                );
            }
        }
    }

    /// 验证集必须复用训练集的 cuts。
    #[test]
    fn with_cuts_reuses_the_given_boundaries() {
        let bs = batches();
        let train = ArrowSource::new(|| Ok(feed(&bs)), 16, 2).unwrap();
        let cuts = train.cuts().clone();

        // 只拿后半段当"验证集",自己建草图的话边界会完全不同
        let tail = vec![bs[1].clone()];
        let valid = ArrowSource::with_cuts(|| Ok(feed(&tail)), cuts, 2).unwrap();

        assert_eq!(valid.n_rows(), 3);
        for f in 0..3u32 {
            assert_eq!(valid.cuts().cuts_for(f), train.cuts().cuts_for(f));
        }
        // 同一个值在两边落进同一个 bin
        for f in 0..3u32 {
            assert_eq!(valid.bin_at(0, f), train.bin_at(3, f), "特征 {f}");
        }
    }

    #[test]
    fn non_numeric_columns_are_rejected_with_a_clear_message() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["a", "b"]))])
            .unwrap();
        let bs = vec![batch];
        let Err(err) = build_cuts_with_nthread(feed(&bs), 16, 8) else {
            panic!("字符串列应该被拒绝");
        };
        let msg = err.to_string();
        assert!(msg.contains("只支持数值列"), "报错要说人话,实际是:{msg}");
    }

    #[test]
    fn mismatched_column_counts_are_caught() {
        let bs = batches();
        let one_col = Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));
        let odd = RecordBatch::try_new(one_col, vec![Arc::new(Float32Array::from(vec![1.0f32]))])
            .unwrap();
        let mixed = vec![bs[0].clone(), odd];
        let Err(err) = build_cuts_with_nthread(feed(&mixed), 16, 8) else {
            panic!("列数不一致应该被拒绝");
        };
        assert!(err.to_string().contains("列数不一致"), "{err}");
    }

    #[test]
    fn empty_parallel_cut_input_is_rejected() {
        let empty = std::iter::empty::<anyhow::Result<RecordBatch>>();
        let Err(err) = build_cuts_with_nthread(empty, 16, 8) else {
            panic!("空输入应该被拒绝");
        };
        assert!(err.to_string().contains("一个 batch 都没有"), "{err}");
    }

    #[test]
    fn two_pass_source_rejects_changed_row_count() {
        let all = batches();
        let shorter = vec![all[0].clone()];
        let mut pass = 0usize;
        let Err(error) = ArrowSource::new(
            || {
                pass += 1;
                let batches = if pass == 1 {
                    all.clone()
                } else {
                    shorter.clone()
                };
                Ok(batches.into_iter().map(Ok))
            },
            16,
            2,
        ) else {
            panic!("两遍输入行数变化应该被拒绝");
        };
        assert!(error.to_string().contains("两遍扫描行数不一致"), "{error}");
    }
    /// The same 3 x 6 data as `batches()` but nulls spelled as a sentinel
    /// (-999 / 0 for the integer column), split over the same two batches.
    fn sentinel_batches(sentinel: f64) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, false),
            Field::new("b", DataType::Float64, false),
            Field::new("c", DataType::Int64, false),
        ]));
        let s = sentinel;
        let mk = |a: Vec<f32>, b: Vec<f64>, c: Vec<i64>| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Float32Array::from(a)),
                    Arc::new(Float64Array::from(b)),
                    Arc::new(Int64Array::from(c)),
                ],
            )
            .unwrap()
        };
        vec![
            mk(vec![1.0, 2.0, 3.0], vec![10.0, s, 30.0], vec![7, 8, 9]),
            mk(vec![4.0, 5.0, 6.0], vec![40.0, 50.0, s], vec![1, 2, 3]),
        ]
    }

    #[test]
    fn feature_decode_maps_the_sentinel_to_nan_for_every_column_type() {
        let bs = sentinel_batches(-999.0);
        let b = feature_as_f32(&bs[0], 1, -999.0).unwrap();
        assert!(b[1].is_nan() && b[0] == 10.0 && b[2] == 30.0);
        // Without a sentinel the value is an ordinary number.
        assert_eq!(feature_as_f32(&bs[0], 1, f32::NAN).unwrap()[1], -999.0);
        // Int64 column: -999 is a sentinel there too.
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int64, true)]));
        let ints = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![
                Some(5),
                Some(-999),
                None,
                Some(0),
            ]))],
        )
        .unwrap();
        let v = feature_as_f32(&ints, 0, -999.0).unwrap();
        assert_eq!((v[0], v[3]), (5.0, 0.0));
        assert!(v[1].is_nan() && v[2].is_nan());
        let zero = feature_as_f32(&ints, 0, 0.0).unwrap();
        assert!(zero[3].is_nan() && zero[1] == -999.0);
        // Labels / weights use the unrestricted visitor: the sentinel is not applied.
        assert_eq!(column_as_f32(&bs[0], 1).unwrap()[1], -999.0);
    }

    #[test]
    fn sentinel_does_not_disable_the_inf_and_max_checks() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float32, false)]));
        let mk = |v: f32| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Float32Array::from(vec![1.0, v]))],
            )
            .unwrap()
        };
        assert!(feature_as_f32(&mk(f32::INFINITY), 0, -999.0).is_err());
        assert!(feature_as_f32(&mk(f32::MAX), 0, -999.0).is_err());
        assert!(feature_as_f32(&mk(-999.0), 0, -999.0).is_ok());
    }

    #[test]
    fn sketch_and_cuts_ignore_sentinel_values() {
        for sentinel in [-999.0f64, 0.0] {
            let bs = sentinel_batches(sentinel);
            let (with, rows) =
                build_cuts_and_count_typed(feed(&bs), 16, 1, false, &[], sentinel as f32).unwrap();
            assert_eq!(rows, 6);
            // A sentinel is never a cut point or the observed minimum.
            let col_b = &with.values[with.offsets[1] as usize..with.offsets[2] as usize];
            assert!(!col_b.contains(&(sentinel as f32)), "{col_b:?}");
            // Equal to the sketch of the same data with nulls in those places.
            let nan_bs: Vec<RecordBatch> = batches();
            let (reference, _) =
                build_cuts_and_count_typed(feed(&nan_bs), 16, 1, false, &[], f32::NAN).unwrap();
            let ref_b =
                &reference.values[reference.offsets[1] as usize..reference.offsets[2] as usize];
            assert_eq!(col_b, ref_b);
        }
    }

    #[test]
    fn quantize_sends_sentinel_to_the_missing_bin_like_nan() {
        let bs = sentinel_batches(-999.0);
        let nan_bs = batches();
        let (cuts, _) =
            build_cuts_and_count_typed(feed(&nan_bs), 16, 1, false, &[], f32::NAN).unwrap();
        let (with, n) = quantize_with_nthread(feed(&bs), &cuts, 2, 1, -999.0).unwrap();
        let (without, _) = quantize_with_nthread(feed(&nan_bs), &cuts, 2, 1, f32::NAN).unwrap();
        assert_eq!(n, 6);
        let column_b = |blocks: &[ColumnBlock]| -> Vec<Bin> {
            blocks
                .iter()
                .find(|b| b.feat_ids.contains(&1))
                .map(|b| {
                    let local = b.feat_ids.iter().position(|&f| f == 1).unwrap();
                    b.column(local).to_vec()
                })
                .unwrap()
        };
        let got = column_b(&with);
        assert_eq!(got, column_b(&without));
        assert_eq!(got[1], MISSING_BIN);
        assert_eq!(got[5], MISSING_BIN);
        // The known-rows path (used by the real two-pass loaders) agrees.
        let known = quantize_with_nthread_known_rows(feed(&bs), &cuts, 2, 6, 1, -999.0).unwrap();
        assert_eq!(column_b(&known), got);
    }

    #[test]
    fn two_pass_source_with_sentinel_equals_source_with_nulls() {
        let bs = sentinel_batches(-999.0);
        let nan_bs = batches();
        let a =
            ArrowSource::new_typed_with_phase_threads(|_| Ok(feed(&bs)), 16, 2, 1, 1, &[], -999.0)
                .unwrap();
        let b = ArrowSource::new(|| Ok(feed(&nan_bs)), 16, 2).unwrap();
        // Column b matches in bins; columns a / c hold identical data.
        for row in 0..6 {
            assert_eq!(a.bin_at(row, 1), b.bin_at(row, 1), "row {row}");
        }
    }
}
