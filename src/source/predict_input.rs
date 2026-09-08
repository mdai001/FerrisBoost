//! 文件预测的读取路径:按**模型的规范 schema** 绑定列,产出原始 f32 行。
//!
//! 和训练路径的区别:训练要量化,预测不要 —— 预测走的是原始特征值。
//!
//! **绑定规则**(和模型里存的 `feature_names` 一一对应):
//!
//! * 名单非空 = **命名模式**:按列名找,输入文件的物理列顺序随便;
//!   缺任何一个必需特征都直接报错。
//! * 名单为空 = **位置模式**(无表头 CSV / numpy 训出来的模型):
//!   按位置绑,只能校验列数。
//!
//! 两条都不需要标签列:标签是训练期元信息。文件里**有**标签列也没关系,
//! 它只是没被选中的那一列 —— 但绝不会被当成特征混进去,因为选列是
//! 按名单来的,不是"除了某一列以外全都要"。

use anyhow::{bail, Context, Result};
use arrow::record_batch::RecordBatch;
use std::path::Path;
use std::sync::Arc;

use super::input::{projection_and_order, reorder, InputFormat, ResolvedInput};

/// 一次读一批,回调消费。**不把整个特征矩阵物化**:预测输出只有每行一个
/// f32,输入侧按 batch 流过去就够了。
pub fn for_each_batch<F>(
    resolved: &ResolvedInput,
    feature_names: &[String],
    n_features: usize,
    csv_has_header: bool,
    ingest_threads: usize,
    on_rows: F,
) -> Result<usize>
where
    F: FnMut(&[Vec<f32>]) -> Result<()>,
{
    let projected_features = (0..n_features).collect::<Vec<_>>();
    for_each_projected_batch(
        resolved,
        feature_names,
        n_features,
        &projected_features,
        csv_has_header,
        ingest_threads,
        on_rows,
    )
}

/// Validate the complete canonical schema, then materialize only the feature
/// ids used by the private inference representation. Canonical ids remain the
/// file/model contract; `projected_features` only defines callback row layout.
pub fn for_each_projected_batch<F>(
    resolved: &ResolvedInput,
    canonical_feature_names: &[String],
    canonical_n_features: usize,
    projected_features: &[usize],
    csv_has_header: bool,
    ingest_threads: usize,
    mut on_rows: F,
) -> Result<usize>
where
    F: FnMut(&[Vec<f32>]) -> Result<()>,
{
    if projected_features.is_empty() {
        bail!("prediction feature projection 不能为空");
    }
    if projected_features.windows(2).any(|pair| pair[0] >= pair[1])
        || projected_features
            .iter()
            .any(|&feature| feature >= canonical_n_features)
    {
        bail!("prediction feature projection 必须是有序、去重且不越界的 canonical id");
    }
    let mut total = 0usize;
    let batches: super::parallel_reader::BatchIter = if resolved.paths.len() == 1 {
        open_projected(
            &resolved.paths[0],
            resolved.format,
            canonical_feature_names,
            canonical_n_features,
            projected_features,
            csv_has_header,
            ingest_threads,
        )?
    } else {
        let paths = Arc::new(resolved.paths.clone());
        let names = Arc::new(canonical_feature_names.to_vec());
        let projected = Arc::new(projected_features.to_vec());
        let format = resolved.format;
        Box::new(super::parallel_reader::ordered_parallel_batches(
            paths.len(),
            ingest_threads,
            move |index| {
                open_projected(
                    &paths[index],
                    format,
                    &names,
                    canonical_n_features,
                    &projected,
                    csv_has_header,
                    1,
                )
            },
        ))
    };
    for batch in batches {
        let batch = batch?;
        let rows = batch_to_rows(&batch, projected_features.len())?;
        total += rows.len();
        on_rows(&rows)?;
    }
    Ok(total)
}

/// 为一个文件解出投影,并返回按规范顺序排好的 batch 流。
fn open_projected(
    path: &Path,
    format: InputFormat,
    canonical_feature_names: &[String],
    canonical_n_features: usize,
    projected_features: &[usize],
    csv_has_header: bool,
    ingest_threads: usize,
) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + 'static>> {
    match format {
        InputFormat::Parquet => {
            let names = super::parquet_source::all_column_names(path)?;
            let canonical = bind(&names, canonical_feature_names, canonical_n_features, path)?;
            let proj = projected_features
                .iter()
                .map(|&feature| canonical[feature])
                .collect::<Vec<_>>();
            let (mask, order) = projection_and_order(&proj);
            Ok(Box::new(
                super::parquet_source::raw_batches_with_nthread(path, &mask, ingest_threads)?
                    .map(move |b| b.map(|b| reorder(&b, &order))),
            ))
        }
        InputFormat::Csv => {
            let (schema, names) = super::csv_source::schema_and_names(path, csv_has_header)?;
            let canonical = bind(&names, canonical_feature_names, canonical_n_features, path)?;
            let proj = projected_features
                .iter()
                .map(|&feature| canonical[feature])
                .collect::<Vec<_>>();
            let (mask, order) = projection_and_order(&proj);
            Ok(Box::new(
                super::csv_source::raw_batches_with_nthread(
                    path,
                    schema,
                    &mask,
                    csv_has_header,
                    ingest_threads,
                )?
                .map(move |b| b.map(|b| reorder(&b, &order))),
            ))
        }
    }
}

/// 列名 → 投影下标。命名模式按名字,位置模式按位置。
fn bind(
    file_names: &[String],
    feature_names: &[String],
    n_features: usize,
    path: &Path,
) -> Result<Vec<usize>> {
    if feature_names.is_empty() {
        // 位置模式:只能校验列数。多出来的列没有语义可言 —— 位置绑定下
        // 无法判断多的是哪一列,所以直接拒,不猜。
        if file_names.len() != n_features {
            bail!(
                "{} 有 {} 列,模型是按 {} 个特征(位置模式)训的;\
                 位置模式下列数必须完全一致",
                path.display(),
                file_names.len(),
                n_features
            );
        }
        return Ok((0..n_features).collect());
    }
    let mut proj = Vec::with_capacity(feature_names.len());
    for want in feature_names {
        let at = file_names
            .iter()
            .position(|n| n == want)
            .with_context(|| format!("{} 里缺少模型需要的特征列 {want:?}", path.display()))?;
        proj.push(at);
    }
    // ⚠️ **多余的列被忽略,这是有意的**:预测文件常常还带着标签列或 id 列。
    // 安全的前提是选列**按名单**进行,所以多出来的列根本不会被读进来。
    Ok(proj)
}

/// RecordBatch → 行式 f32。缺失值保持 NaN,交给模型的缺失方向处理。
fn batch_to_rows(batch: &RecordBatch, n_features: usize) -> Result<Vec<Vec<f32>>> {
    let mut cols = Vec::with_capacity(n_features);
    for c in 0..n_features {
        cols.push(super::arrow_source::column_as_f32(batch, c)?);
    }
    let n_rows = batch.num_rows();
    let mut rows = vec![Vec::with_capacity(n_features); n_rows];
    for col in cols.iter() {
        for (r, v) in col.iter().enumerate() {
            rows[r].push(*v);
        }
    }
    Ok(rows)
}
