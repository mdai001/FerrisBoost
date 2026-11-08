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
use std::collections::HashMap;
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
    for_each_projected_contiguous_batch(
        resolved,
        canonical_feature_names,
        canonical_n_features,
        projected_features,
        csv_has_header,
        ingest_threads,
        f32::NAN,
        None,
        |data, _| {
            let rows = data
                .chunks_exact(projected_features.len())
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            on_rows(&rows)
        },
    )
}

/// Bounded, reusable row-major batches in compact feature order. The callback
/// borrows the buffer; it must finish consuming it before the next batch.
///
/// `missing` is the XGBoost `missing` sentinel: feature values equal to it are
/// handed to the callback as NaN (NaN = no sentinel), so scoring routes them
/// down the missing direction exactly as it does for nulls.
///
/// `categories` (the model's category names, indexed by canonical feature id)
/// re-codes named categorical columns (string dictionaries) by value; a row
/// holding a category the model does not know is an error.
#[allow(clippy::too_many_arguments)]
pub fn for_each_projected_contiguous_batch<F>(
    resolved: &ResolvedInput,
    canonical_feature_names: &[String],
    canonical_n_features: usize,
    projected_features: &[usize],
    csv_has_header: bool,
    ingest_threads: usize,
    missing: f32,
    categories: Option<&crate::category_names::CategoryTable>,
    mut on_rows: F,
) -> Result<usize>
where
    F: FnMut(&[f32], usize) -> Result<()>,
{
    if projected_features.is_empty() {
        bail!("prediction feature projection cannot be empty");
    }
    if projected_features.windows(2).any(|pair| pair[0] >= pair[1])
        || projected_features
            .iter()
            .any(|&feature| feature >= canonical_n_features)
    {
        bail!("prediction feature projection must be sorted, deduplicated, in-range canonical ids");
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
    let mut rows = Vec::new();
    let mut recoder = super::cat_recode::Recoder::fixed(categories);
    for batch in batches {
        let batch = recoder.recode(batch?, |c| projected_features.get(c).copied())?;
        batch_to_contiguous_rows(&batch, projected_features.len(), &mut rows)?;
        super::mask_missing_in_place(&mut rows, missing);
        total = total
            .checked_add(batch.num_rows())
            .context("prediction row count overflow")?;
        on_rows(&rows, batch.num_rows())?;
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
                "{} has {} columns but the model was trained on {} features (positional mode); \
                 in positional mode the column count must match exactly",
                path.display(),
                file_names.len(),
                n_features
            );
        }
        return Ok((0..n_features).collect());
    }
    // For small schemas, building a hash table costs more than scanning the
    // few file columns. Use a conservative guard, not an adaptive scheduler.
    if file_names.len() < 16 {
        let mut proj = Vec::with_capacity(feature_names.len());
        for want in feature_names {
            let at = file_names
                .iter()
                .position(|name| name == want)
                .with_context(|| {
                    format!(
                        "{} is missing feature column {want:?} required by the model",
                        path.display()
                    )
                })?;
            proj.push(at);
        }
        return Ok(proj);
    }
    // Keep the first file occurrence so duplicate-name behavior stays identical
    // to the former `iter().position()` lookup.
    let mut by_name: HashMap<&str, usize> = HashMap::with_capacity(file_names.len());
    for (at, name) in file_names.iter().enumerate() {
        by_name.entry(name.as_str()).or_insert(at);
    }

    let mut proj = Vec::with_capacity(feature_names.len());
    for want in feature_names {
        let at = by_name.get(want.as_str()).copied().with_context(|| {
            format!(
                "{} is missing feature column {want:?} required by the model",
                path.display()
            )
        })?;
        proj.push(at);
    }
    // ⚠️ **多余的列被忽略,这是有意的**:预测文件常常还带着标签列或 id 列。
    // 安全的前提是选列**按名单**进行,所以多出来的列根本不会被读进来。
    Ok(proj)
}

#[cfg(test)]
mod bind_tests {
    use super::bind;
    use std::path::Path;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn padded_names(values: &[&str], minimum_len: usize) -> Vec<String> {
        let mut result = names(values);
        result.extend((result.len()..minimum_len).map(|i| format!("extra-{i}")));
        result
    }

    #[test]
    fn named_mode_reorders_features_and_ignores_extra_columns() {
        let file = names(&["extra1", "c", "a", "extra2", "b"]);
        let requested = names(&["a", "b", "c"]);
        assert_eq!(
            bind(&file, &requested, 3, Path::new("input.csv")).unwrap(),
            vec![2, 4, 1]
        );
    }

    #[test]
    fn named_mode_duplicate_file_names_use_first_occurrence() {
        let file = padded_names(&["a", "b", "a", "b", "c", "a"], 16);
        let requested = names(&["a", "b", "c"]);
        assert_eq!(
            bind(&file, &requested, 3, Path::new("input.csv")).unwrap(),
            vec![0, 1, 4]
        );
    }

    #[test]
    fn named_mode_duplicate_requested_names_repeat_the_first_file_index() {
        let file = padded_names(&["x", "a", "b", "a"], 16);
        let requested = names(&["a", "a", "b", "a"]);
        assert_eq!(
            bind(&file, &requested, 4, Path::new("input.csv")).unwrap(),
            vec![1, 1, 2, 1]
        );
    }

    #[test]
    fn named_mode_reports_first_missing_requested_name_in_model_order() {
        let file = padded_names(&["a", "b"], 16);
        let requested = names(&["a", "missing-first", "missing-second"]);
        let error = bind(&file, &requested, 3, Path::new("data.csv")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "data.csv is missing feature column \"missing-first\" required by the model"
        );
    }

    #[test]
    fn named_mode_empty_file_reports_first_missing_name() {
        let requested = names(&["required", "also-required"]);
        let error = bind(&[], &requested, 2, Path::new("empty.csv")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "empty.csv is missing feature column \"required\" required by the model"
        );
    }

    #[test]
    fn positional_mode_accepts_exact_width_including_zero() {
        assert_eq!(
            bind(&[], &[], 0, Path::new("empty.csv")).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            bind(&names(&["p", "q", "r"]), &[], 3, Path::new("input.csv")).unwrap(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn positional_mode_rejects_too_few_or_too_many_columns() {
        for (file, n_features, file_columns) in [
            (names(&["a", "b"]), 3, 2),
            (names(&["a", "b", "c", "d"]), 3, 4),
        ] {
            let error = bind(&file, &[], n_features, Path::new("pos.csv")).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!(
                    "pos.csv has {file_columns} columns but the model was trained on {n_features} features (positional mode); in positional mode the column count must match exactly"
                )
            );
        }
    }

    #[test]
    fn positional_mode_preserves_duplicate_file_columns_by_position() {
        let file = names(&["a", "a"]);
        assert_eq!(
            bind(&file, &[], 2, Path::new("input.csv")).unwrap(),
            vec![0, 1]
        );
    }
}

/// Borrow numeric columns and pack cache-sized row tiles. Extra scratch is
/// O(features), not another materialized batch; fallback keeps the old errors.
/// 预测特征同样拒绝 `f32::MAX`:它是缺失/非缺失分裂的有限上界哨兵,
/// 训练和验证特征已经拒绝它(见 `arrow_source::for_each_feature_f32`)。
/// 预测时若接受,原始比较 `v < MAX` 会把它路由到和训练不同的一侧。
/// ±inf 在预测时仍按原来的规则接受。
///
/// 文件预测只解码模型用到的列,所以这里的列号是「这一批里的位置」、行号是
/// 批内行号,都不是文件里的坐标 —— 报错用这一批 schema 里的**列名**。
/// 模型不读的列不解码,那里的值影响不了任何预测。
pub(crate) fn reject_reserved_max(rows: &[f32], batch: &RecordBatch) -> Result<()> {
    let n_features = batch.num_columns().max(1);
    if let Some(i) = rows.iter().position(|&v| v == f32::MAX) {
        let name = batch.schema().field(i % n_features).name().clone();
        bail!(
            "prediction data feature column {name:?} contains f32::MAX: this value is reserved for the missing-value split boundary and is rejected in training and prediction; rescale this feature first"
        );
    }
    Ok(())
}

fn batch_to_contiguous_rows(
    batch: &RecordBatch,
    n_features: usize,
    rows: &mut Vec<f32>,
) -> Result<()> {
    use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::DataType;

    enum Column<'a> {
        F32(&'a [f32], Option<&'a NullBuffer>),
        F64(&'a [f64], Option<&'a NullBuffer>),
        I32(&'a [i32], Option<&'a NullBuffer>),
        I64(&'a [i64], Option<&'a NullBuffer>),
    }

    fn classify(batch: &RecordBatch) -> Option<Vec<Column<'_>>> {
        let mut columns = Vec::with_capacity(batch.num_columns());
        for array in batch.columns() {
            macro_rules! view {
                ($ty:ty, $variant:ident) => {{
                    let typed = array.as_any().downcast_ref::<$ty>()?;
                    if typed.values().len() != batch.num_rows() {
                        return None;
                    }
                    Column::$variant(
                        &typed.values()[..],
                        typed.nulls().filter(|nulls| nulls.null_count() != 0),
                    )
                }};
            }
            columns.push(match array.data_type() {
                DataType::Float32 => view!(Float32Array, F32),
                DataType::Float64 => view!(Float64Array, F64),
                DataType::Int32 => view!(Int32Array, I32),
                DataType::Int64 => view!(Int64Array, I64),
                _ => return None,
            });
        }
        Some(columns)
    }

    #[inline]
    fn pack_column<T: Copy>(
        tile: &mut [f32],
        width: usize,
        feature: usize,
        values: &[T],
        nulls: Option<&NullBuffer>,
        start: usize,
        cast: impl Fn(T) -> f32,
    ) {
        let output = tile[feature..].iter_mut().step_by(width);
        match nulls {
            None => {
                for (out, &value) in output.zip(values) {
                    *out = cast(value);
                }
            }
            Some(nulls) => {
                for (i, (out, &value)) in output.zip(values).enumerate() {
                    *out = if nulls.is_valid(start + i) {
                        cast(value)
                    } else {
                        f32::NAN
                    };
                }
            }
        }
    }

    if n_features == 0 || batch.num_columns() != n_features {
        bail!("prediction batch feature count mismatch");
    }
    let values = batch
        .num_rows()
        .checked_mul(n_features)
        .context("prediction batch size overflow")?;
    let Some(columns) = classify(batch) else {
        // Classify before resizing/writing so legacy error-time buffer state,
        // conversion policy and first unsupported column remain unchanged.
        batch_to_contiguous_rows_legacy(batch, n_features, rows)?;
        return reject_reserved_max(rows, batch);
    };
    rows.resize(values, 0.0);
    // Division order avoids overflow in a hypothetical very large row width.
    let tile_rows = (65_536 / n_features / std::mem::size_of::<f32>()).clamp(1, 64);
    for start in (0..batch.num_rows()).step_by(tile_rows) {
        let end = start.saturating_add(tile_rows).min(batch.num_rows());
        let tile = &mut rows[start * n_features..end * n_features];
        for (feature, column) in columns.iter().enumerate() {
            match column {
                Column::F32(data, nulls) => pack_column(
                    tile,
                    n_features,
                    feature,
                    &data[start..end],
                    *nulls,
                    start,
                    |v| v,
                ),
                Column::F64(data, nulls) => pack_column(
                    tile,
                    n_features,
                    feature,
                    &data[start..end],
                    *nulls,
                    start,
                    |v| v as f32,
                ),
                Column::I32(data, nulls) => pack_column(
                    tile,
                    n_features,
                    feature,
                    &data[start..end],
                    *nulls,
                    start,
                    |v| v as f32,
                ),
                Column::I64(data, nulls) => pack_column(
                    tile,
                    n_features,
                    feature,
                    &data[start..end],
                    *nulls,
                    start,
                    |v| v as f32,
                ),
            }
        }
    }
    reject_reserved_max(rows, batch)
}

/// One converted column at a time, preserving the original conversion contract.
fn batch_to_contiguous_rows_legacy(
    batch: &RecordBatch,
    n_features: usize,
    rows: &mut Vec<f32>,
) -> Result<()> {
    if n_features == 0 || batch.num_columns() != n_features {
        bail!("prediction batch feature count mismatch");
    }
    let values = batch
        .num_rows()
        .checked_mul(n_features)
        .context("prediction batch size overflow")?;
    rows.resize(values, 0.0);
    for c in 0..n_features {
        let column = super::arrow_source::column_as_f32_allow_inf(batch, c)?;
        for (row, value) in rows.chunks_exact_mut(n_features).zip(column) {
            row[c] = value;
        }
    }
    Ok(())
}

#[cfg(test)]
mod packing_tests {
    use super::{batch_to_contiguous_rows, batch_to_contiguous_rows_legacy};

    #[test]
    fn reserved_max_is_rejected_when_packing_prediction_batches() {
        use arrow::array::{ArrayRef, Float32Array};
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;
        let make = |second: ArrayRef| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("a", arrow::datatypes::DataType::Float32, true),
                Field::new("b", second.data_type().clone(), true),
            ]));
            let first: ArrayRef = Arc::new(Float32Array::from(vec![1.0, f32::MAX, 3.0]));
            RecordBatch::try_new(schema, vec![first, second]).unwrap()
        };
        let mut rows = Vec::new();
        // fast path (all primitive columns)
        let fast = make(Arc::new(Float32Array::from(vec![0.5, 0.5, f32::INFINITY])));
        let err = batch_to_contiguous_rows(&fast, 2, &mut rows).unwrap_err();
        assert!(err.to_string().contains(r#"feature column "a""#), "{err}");
        // +inf and the next value below MAX stay accepted
        let ok_schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            arrow::datatypes::DataType::Float32,
            true,
        )]));
        let ok = RecordBatch::try_new(
            ok_schema,
            vec![Arc::new(Float32Array::from(vec![
                f32::INFINITY,
                f32::MAX.next_down(),
            ])) as ArrayRef],
        )
        .unwrap();
        batch_to_contiguous_rows(&ok, 1, &mut rows).unwrap();
    }
    use arrow::array::{
        Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    };
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn reference(batch: &RecordBatch) -> anyhow::Result<Vec<f32>> {
        let width = batch.num_columns();
        let mut rows = vec![0.0; batch.num_rows() * width];
        for c in 0..width {
            let column = crate::source::arrow_source::column_as_f32_allow_inf(batch, c)?;
            for (row, value) in rows.chunks_exact_mut(width).zip(column) {
                row[c] = value;
            }
        }
        Ok(rows)
    }

    fn mixed_batch(n_rows: usize, n_features: usize) -> RecordBatch {
        let arrays = (0..n_features)
            .map(|c| {
                let count = n_rows + 3;
                match c % 4 {
                    0 => Arc::new(
                        Float32Array::from(
                            (0..count)
                                .map(|r| match (r + c) % 5 {
                                    0 => None,
                                    1 => Some(-0.0),
                                    2 => Some(f32::from_bits(0x7fc1_2345)),
                                    _ => Some((r + c) as f32 / 3.0),
                                })
                                .collect::<Vec<_>>(),
                        )
                        .slice(3, n_rows),
                    ) as ArrayRef,
                    1 => Arc::new(
                        Float64Array::from(
                            (0..count)
                                .map(|r| match (r + c) % 5 {
                                    0 => None,
                                    1 => Some(-0.0),
                                    2 => Some(f64::from_bits(0x7ff8_1234_5678_9abc)),
                                    _ => Some((r + c) as f64 / 3.0),
                                })
                                .collect::<Vec<_>>(),
                        )
                        .slice(3, n_rows),
                    ) as ArrayRef,
                    2 => Arc::new(
                        Int32Array::from(
                            (0..count)
                                .map(|r| match (r + c) % 4 {
                                    0 => None,
                                    1 => Some(16_777_217),
                                    2 => Some(i32::MIN),
                                    _ => Some(i32::MAX),
                                })
                                .collect::<Vec<_>>(),
                        )
                        .slice(3, n_rows),
                    ) as ArrayRef,
                    _ => Arc::new(
                        Int64Array::from(
                            (0..count)
                                .map(|r| match (r + c) % 4 {
                                    0 => None,
                                    1 => Some((1i64 << 55) + 1),
                                    2 => Some(i64::MIN),
                                    _ => Some(i64::MAX),
                                })
                                .collect::<Vec<_>>(),
                        )
                        .slice(3, n_rows),
                    ) as ArrayRef,
                }
            })
            .collect::<Vec<_>>();
        let fields = arrays
            .iter()
            .enumerate()
            .map(|(c, a)| Field::new(format!("f{c}"), a.data_type().clone(), true))
            .collect::<Vec<_>>();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    #[test]
    fn packing_preserves_numeric_bits_nulls_offsets_and_tile_boundaries() {
        let mut rows = vec![123.0; 1000];
        for (width, count) in [(28, 65), (256, 65), (2000, 9), (2000, 17), (4, 1), (28, 0)] {
            let batch = mixed_batch(count, width);
            let expected = reference(&batch).unwrap();
            batch_to_contiguous_rows(&batch, width, &mut rows).unwrap();
            assert_eq!(
                rows.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "shape {count}x{width}"
            );
        }
    }

    #[test]
    fn packing_preserves_shape_and_unsupported_type_errors() {
        let batch = mixed_batch(3, 4);
        for width in [0, 3, 5] {
            assert_eq!(
                batch_to_contiguous_rows(&batch, width, &mut Vec::new())
                    .unwrap_err()
                    .to_string(),
                "prediction batch feature count mismatch"
            );
        }
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Float32Array::from(vec![1.0, -0.0])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ];
        let fields = arrays
            .iter()
            .enumerate()
            .map(|(c, a)| Field::new(format!("f{c}"), a.data_type().clone(), false))
            .collect::<Vec<_>>();
        let unsupported = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        let expected_error = reference(&unsupported).unwrap_err().to_string();
        assert_eq!(
            batch_to_contiguous_rows(&unsupported, 2, &mut Vec::new())
                .unwrap_err()
                .to_string(),
            expected_error
        );
    }

    #[test]
    fn packing_65_non_null_mixed_rows_preserves_float_bits_and_reuses_output() {
        let rows = 65;
        let f32_values = (0..rows)
            .map(|row| match row % 7 {
                0 => f32::from_bits(0x7fc1_2345),
                1 => -0.0,
                2 => f32::INFINITY,
                3 => f32::NEG_INFINITY,
                4 => f32::from_bits(1), // smallest positive subnormal
                _ => row as f32 + 0.25,
            })
            .collect::<Vec<_>>();
        let f64_values = (0..rows)
            .map(|row| match row % 6 {
                0 => f64::from_bits(0x7ff8_1234_5678_9abc),
                1 => -0.0,
                2 => f64::INFINITY,
                3 => f64::NEG_INFINITY,
                4 => f64::from_bits(1),
                _ => row as f64 + 0.5,
            })
            .collect::<Vec<_>>();
        let i32_values = (0..rows)
            .map(|row| match row % 4 {
                0 => 16_777_217,
                1 => i32::MIN,
                2 => i32::MAX,
                _ => row as i32,
            })
            .collect::<Vec<_>>();
        let i64_values = (0..rows)
            .map(|row| match row % 4 {
                0 => (1i64 << 55) + 1,
                1 => i64::MIN,
                2 => i64::MAX,
                _ => row as i64,
            })
            .collect::<Vec<_>>();
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Float32Array::from(f32_values)),
            Arc::new(Float64Array::from(f64_values)),
            Arc::new(Int32Array::from(i32_values)),
            Arc::new(Int64Array::from(i64_values)),
        ];
        assert!(arrays.iter().all(|array| array.null_count() == 0));
        let fields = arrays
            .iter()
            .enumerate()
            .map(|(c, array)| Field::new(format!("f{c}"), array.data_type().clone(), false))
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        let expected = reference(&batch).unwrap();

        let mut output = vec![f32::from_bits(0x7fc0_4321); rows * 4];
        let allocation = output.as_ptr();
        batch_to_contiguous_rows(&batch, 4, &mut output).unwrap();
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        batch_to_contiguous_rows(&batch, 4, &mut output).unwrap();
        assert_eq!(
            output.as_ptr(),
            allocation,
            "output allocation should be reused"
        );
    }

    #[test]
    fn unsupported_string_positions_preserve_legacy_error_and_partial_buffer() {
        for unsupported_at in [0, 1, 2] {
            let arrays: Vec<ArrayRef> = (0..3)
                .map(|column| {
                    if column == unsupported_at {
                        Arc::new(StringArray::from(vec!["s0", "s1", "s2"])) as ArrayRef
                    } else {
                        Arc::new(Float32Array::from(vec![
                            column as f32 + 0.25,
                            column as f32 + 1.25,
                            column as f32 + 2.25,
                        ])) as ArrayRef
                    }
                })
                .collect();
            let fields = arrays
                .iter()
                .enumerate()
                .map(|(c, array)| Field::new(format!("f{c}"), array.data_type().clone(), false))
                .collect::<Vec<_>>();
            let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
            let initial = (0..9)
                .map(|i| f32::from_bits(0x3f00_0000 + i * 17))
                .collect::<Vec<_>>();

            let mut expected_buffer = initial.clone();
            let expected_error = batch_to_contiguous_rows_legacy(&batch, 3, &mut expected_buffer)
                .unwrap_err()
                .to_string();
            let mut actual_buffer = initial;
            let actual_error = batch_to_contiguous_rows(&batch, 3, &mut actual_buffer)
                .unwrap_err()
                .to_string();

            assert_eq!(actual_error, expected_error, "column {unsupported_at}");
            assert_eq!(
                actual_buffer
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_buffer
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "column {unsupported_at}"
            );
        }
    }
}
