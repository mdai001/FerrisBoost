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
use crate::source::Targets;

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
    let file =
        File::open(path).map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.display()))?;
    if crate::source::input::is_gzipped(path) {
        Ok(Box::new(flate2::read::GzDecoder::new(BufReader::new(file))))
    } else {
        Ok(Box::new(file))
    }
}

pub(crate) fn infer_schema(path: &Path, has_header: bool) -> anyhow::Result<SchemaRef> {
    let mut file = readable(path)?;
    let format = Format::default().with_header(has_header);
    let (schema, _) = format
        .infer_schema(&mut file, Some(INFER_ROWS))
        .map_err(|e| anyhow::anyhow!("schema inference failed for {}: {e}", path.display()))?;
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
    let schema = infer_schema(path.as_ref(), has_header)?;
    n_features_from_schema(&schema, label)
}

pub(crate) fn n_features_from_schema<'a>(
    schema: &Schema,
    targets: impl Into<Targets<'a>>,
) -> anyhow::Result<usize> {
    columns(schema, targets.into()).map(|c| c.features.len())
}

/// Resolve a positional label against the inferred CSV schema.  Arrow assigns
/// stable synthetic field names to headerless columns; the name is only an
/// internal handle used by the existing projection machinery.
pub fn positional_label_name(path: impl AsRef<Path>, label_index: usize) -> anyhow::Result<String> {
    let schema = infer_schema(path.as_ref(), false)?;
    positional_label_name_from_schema(&schema, label_index)
}

pub(crate) fn positional_label_name_from_schema(
    schema: &Schema,
    label_index: usize,
) -> anyhow::Result<String> {
    let n_columns = schema.fields().len();
    crate::ensure_input!(
        label_index < n_columns,
        "label column index {label_index} is out of range: the headerless CSV has only {n_columns} columns"
    );
    crate::ensure_input!(
        n_columns > 1,
        "headerless CSV has no feature columns besides the label column"
    );
    Ok(schema.field(label_index).name().clone())
}

/// 把推断出来的列类型收紧成训练能用的形状。
///
/// 推断出 Utf8 的列直接报错,而不是留到量化时才炸 —— 那时候的错误
/// 信息里只有列号,这里还能说出列名。Columns in `text` (categorical
/// features read as category names, see [`text_category_features`]) are
/// strings by design.
fn check_numeric(schema: &Schema, skip: &str, text: &[usize]) -> anyhow::Result<()> {
    for (c, f) in schema.fields().iter().enumerate() {
        if f.name() == skip || text.contains(&c) {
            continue;
        }
        match f.data_type() {
            DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64 => {}
            other => crate::bail_input!(
                "column {:?} was inferred as {other:?}; only numeric columns are supported, \
                 except text columns declared categorical (\"c\" in feature_types), which are \
                 read as category names; if it is actually numeric, check for invalid \
                 characters or thousands separators",
                f.name()
            ),
        }
    }
    Ok(())
}

/// How a canonical feature's CSV column is read, decided before any read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatRead {
    /// As inferred: numeric features, and categorical features with
    /// **integer** categories (re-coded by value).
    Inferred,
    /// Text, coded by a known list of string categories.
    Names,
    /// Training: a declared categorical feature without a known list. Pass 1
    /// reads it as text and decides from **every** field, not the inference
    /// sample: integer codes when all non-empty fields are numbers, category
    /// names otherwise. Pass 2 reads it as the decision says.
    Decide,
    /// Validation / prediction: a categorical feature the model has no names
    /// for. Read as text and parsed as numbers by [`CheckedCodes`], so a field
    /// that is not a number past the inference sample is a clear error.
    Codes,
}

impl CatRead {
    fn as_text(self) -> bool {
        self != CatRead::Inferred
    }
}

/// How each canonical feature is read.
///
/// * A feature with a `known` list of strings (the trained model's for a
///   validation set; for training, a continued model's, when the feature is
///   declared categorical) is text in every file, whatever the inferred type:
///   a file holding only `"1"` and `"2"` of the categories `["1", "2", "x"]`
///   must not be read as the codes 1 and 2.
/// * Training (`learn`): a declared categorical feature without a known list
///   is decided by pass 1 from all rows of all files ([`CatRead::Decide`]),
///   so one feature never mixes codes and names.
/// * Validation: a categorical feature of the model (`model_categorical`)
///   without a list is integer codes ([`CatRead::Codes`]).
fn cat_reads(
    n_features: usize,
    categorical: Option<&[bool]>,
    model_categorical: &[bool],
    known: Option<&crate::category_names::CategoryTable>,
    learn: bool,
) -> Vec<CatRead> {
    use crate::category_names::CategoryValues;
    let flag = |flags: Option<&[bool]>, f: usize| {
        flags.is_some_and(|flags| flags.get(f).copied().unwrap_or(false))
    };
    (0..n_features)
        .map(|f| match known.and_then(|t| t.get(f)) {
            // Training codes only declared features (a continued model's list
            // included); validation codes every feature with a list.
            Some(c) => {
                if matches!(c.values(), CategoryValues::Str(_)) && (!learn || flag(categorical, f))
                {
                    CatRead::Names
                } else {
                    CatRead::Inferred
                }
            }
            None if learn && flag(categorical, f) => CatRead::Decide,
            None if !learn && flag(Some(model_categorical), f) => CatRead::Codes,
            None => CatRead::Inferred,
        })
        .collect()
}

/// `schema` with the given columns re-typed (nullable). An empty field stays
/// null (a missing value or category), whatever the type.
fn with_column_types(schema: &SchemaRef, columns: &[(usize, DataType)]) -> SchemaRef {
    if columns.is_empty() {
        return Arc::clone(schema);
    }
    let fields: Vec<arrow::datatypes::Field> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(c, f)| match columns.iter().find(|(at, _)| *at == c) {
            Some((_, t)) => arrow::datatypes::Field::new(f.name(), t.clone(), true),
            None => f.as_ref().clone(),
        })
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// The schema a prediction reads `path` with, and the canonical features
/// [`CheckedCodes`] must parse. Features with string categories in the model
/// (`categories`, by canonical feature id; `canonical[f]` = its column) are
/// text; categorical features (`categorical`) without a list are text parsed
/// as codes; the rest are read as inferred.
pub(crate) fn prediction_schema(
    schema: SchemaRef,
    canonical: &[usize],
    categorical: &[bool],
    categories: Option<&crate::category_names::CategoryTable>,
) -> (SchemaRef, Vec<usize>) {
    let reads = cat_reads(canonical.len(), None, categorical, categories, false);
    let text: Vec<(usize, DataType)> = canonical
        .iter()
        .zip(&reads)
        .filter(|(_, r)| r.as_text())
        .map(|(&c, _)| (c, DataType::Utf8))
        .collect();
    let codes = (0..canonical.len())
        .filter(|&f| reads[f] == CatRead::Codes)
        .collect();
    (with_column_types(&schema, &text), codes)
}

/// Parses the text columns of categorical features without names (output
/// columns `columns`) as numbers, like a `Float64` CSV column, into `Float64`
/// columns. A field that is not a number is an error naming the file, the
/// column, the value and its data row, where the Arrow reader would only have
/// said it failed to parse.
pub(crate) struct CheckedCodes<I> {
    inner: I,
    path: PathBuf,
    columns: Arc<[usize]>,
    rows: usize,
}

impl<I> CheckedCodes<I> {
    pub(crate) fn new(inner: I, path: PathBuf, columns: Arc<[usize]>) -> Self {
        Self {
            inner,
            path,
            columns,
            rows: 0,
        }
    }
}

impl<I: Iterator<Item = anyhow::Result<RecordBatch>>> Iterator for CheckedCodes<I> {
    type Item = anyhow::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.inner.next()? {
            Ok(batch) if !self.columns.is_empty() => batch,
            other => return Some(other),
        };
        let first = self.rows;
        self.rows += batch.num_rows();
        Some(parse_codes(batch, &self.columns, &self.path, first))
    }
}

fn parse_codes(
    batch: RecordBatch,
    columns: &[usize],
    path: &Path,
    first_row: usize,
) -> anyhow::Result<RecordBatch> {
    use arrow::array::{Array, AsArray, Float64Array};
    use arrow::compute::kernels::cast_utils::Parser;
    let schema = batch.schema();
    let mut fields: Vec<arrow::datatypes::FieldRef> = schema.fields().iter().cloned().collect();
    let mut arrays = batch.columns().to_vec();
    for &c in columns {
        let name = schema.field(c).name();
        let text = arrays[c].as_string_opt::<i32>().ok_or_else(|| {
            anyhow::anyhow!("CSV categorical column {name:?} was not read as text")
        })?;
        let mut values = Vec::with_capacity(text.len());
        for i in 0..text.len() {
            if text.is_null(i) {
                values.push(None);
                continue;
            }
            let s = text.value(i);
            match arrow::datatypes::Float64Type::parse(s) {
                Some(v) => values.push(Some(v)),
                None => crate::bail_input!(
                    "{}: feature column {name:?} holds {s:?} (data row {}), which is not a \
                     number; the model has no category names for this categorical feature, so \
                     the file must hold its integer category codes",
                    path.display(),
                    first_row + i + 1
                ),
            }
        }
        arrays[c] = Arc::new(Float64Array::from(values));
        fields[c] = Arc::new(arrow::datatypes::Field::new(name, DataType::Float64, true));
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
        arrays,
    )?)
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

/// 一个 CSV schema 里各列的角色:特征 / 标签 / 可选权重。
struct ColumnRoles {
    features: Vec<usize>,
    label: usize,
    weight: Option<usize>,
    names: Vec<String>,
}

/// Read a whole query-id column from every file (in the given order, which is the
/// order the dataset concatenates them in). A second pass over the text with a
/// one-column projection; the features are not parsed again.
pub(crate) fn read_u64_column(
    paths: &[PathBuf],
    has_header: bool,
    name: &str,
) -> anyhow::Result<Vec<u64>> {
    let mut out = Vec::new();
    for path in paths {
        let schema = infer_schema(path, has_header)?;
        let col = schema
            .fields()
            .iter()
            .position(|f| f.name() == name)
            .ok_or_else(|| crate::input_err!("{} has no column {name:?}", path.display()))?;
        for batch in batch_reader(path, schema, has_header, vec![col])? {
            out.extend(crate::source::arrow_source::column_as_u64(
                &batch?, 0, name,
            )?);
        }
    }
    Ok(out)
}

fn columns(schema: &Schema, targets: Targets<'_>) -> anyhow::Result<ColumnRoles> {
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let find = |want: &str| {
        names.iter().position(|n| *n == want).ok_or_else(|| {
            crate::input_err!("file has no column {want:?}; available columns: {names:?}")
        })
    };
    if let Some(q) = targets.qid {
        find(q)?;
    }
    let label_col = find(targets.label)?;
    let weight_col = targets.weight.map(find).transpose()?;
    let (feat_cols, feat_names) = targets.select_features(&names)?;
    Ok(ColumnRoles {
        features: feat_cols,
        label: label_col,
        weight: weight_col,
        names: feat_names,
    })
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

pub fn open_many_with_header_and_nthread<'a>(
    paths: &[PathBuf],
    label: impl Into<Targets<'a>>,
    cuts: Option<BinCuts>,
    max_bins: u32,
    cols_per_block: usize,
    has_header: bool,
    nthread: usize,
) -> anyhow::Result<Dataset> {
    open_many_impl(
        paths,
        label.into(),
        cuts,
        max_bins,
        BlockWidth::Fixed(cols_per_block),
        has_header,
        nthread,
        None,
    )
}

/// Block width of a CSV open: fixed up front, or chosen from the exact
/// aggregate row count that the first (cut-building) pass counts, before the
/// second pass builds the column blocks. The second form lets a caller plan
/// memory from the real row count of input that has none before open.
pub(crate) enum BlockWidth<'a> {
    Fixed(usize),
    FromRowCount(&'a mut dyn FnMut(u64) -> anyhow::Result<usize>),
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn open_many_with_header_and_nthread_prepared<'a>(
    paths: &[PathBuf],
    label: impl Into<Targets<'a>>,
    cuts: Option<BinCuts>,
    max_bins: u32,
    width: BlockWidth<'_>,
    has_header: bool,
    nthread: usize,
    first_schema: Option<SchemaRef>,
) -> anyhow::Result<Dataset> {
    open_many_impl(
        paths,
        label.into(),
        cuts,
        max_bins,
        width,
        has_header,
        nthread,
        first_schema,
    )
}

#[allow(clippy::too_many_arguments)]
fn open_many_impl(
    paths: &[PathBuf],
    targets: Targets<'_>,
    cuts: Option<BinCuts>,
    max_bins: u32,
    width: BlockWidth<'_>,
    has_header: bool,
    nthread: usize,
    first_schema: Option<SchemaRef>,
) -> anyhow::Result<Dataset> {
    anyhow::ensure!(!paths.is_empty(), "no input files");
    let label = targets.label;
    let schema_pool = crate::threading::build_pool(nthread)
        .map_err(|e| anyhow::anyhow!("failed to create the CSV schema ingest thread pool: {e}"))?;
    let schema_profile = crate::source::openprof::start("schema");
    let schemas = schema_pool.install(|| {
        paths
            .par_iter()
            .enumerate()
            .map(|(index, path)| match (index, first_schema.as_ref()) {
                (0, Some(schema)) => Ok(Arc::clone(schema)),
                _ => infer_schema(path, has_header),
            })
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    drop(schema_profile);
    let mut feat_proj = Vec::with_capacity(paths.len());
    let mut label_proj = Vec::with_capacity(paths.len());
    let mut weight_proj = Vec::with_capacity(paths.len());
    let mut feature_names: Option<Vec<String>> = None;

    for (p, schema) in paths.iter().zip(&schemas) {
        let roles = columns(schema, targets)?;
        let (cols, label_col, names) = (roles.features, roles.label, roles.names);
        if let Some(w) = roles.weight {
            weight_proj.push(w);
        }
        match &feature_names {
            None => {
                feature_names = Some(names);
                feat_proj.push(cols);
            }
            Some(canon) if has_header => {
                crate::ensure_input!(
                    names.len() == canon.len(),
                    "schemas differ across files: {} has {} feature columns, {} has {}",
                    paths[0].display(),
                    canon.len(),
                    p.display(),
                    names.len()
                );
                let mut mapped = Vec::with_capacity(canon.len());
                for want in canon {
                    let at = names.iter().position(|n| n == want).ok_or_else(|| {
                        crate::input_err!(
                            "schemas differ across files: {} has no column {want:?}",
                            p.display()
                        )
                    })?;
                    mapped.push(cols[at]);
                }
                feat_proj.push(mapped);
            }
            Some(canon) => {
                crate::ensure_input!(
                    names.len() == canon.len(),
                    "positional schemas differ across files: {} has {} feature columns, {} has {}",
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
    let feature_names = feature_names.expect("non-empty");

    // Categorical features read as text are re-typed in every file before
    // any read: names are coded by value, decided columns are decided by
    // pass 1, unnamed codes of a model are parsed by `CheckedCodes`.
    let known = match &cuts {
        Some(c) => c.categories().map(|t| &**t),
        None => targets.categories,
    };
    let model_categorical: Vec<bool> = cuts
        .as_ref()
        .map(|c| c.categorical_flags().to_vec())
        .unwrap_or_default();
    let reads = cat_reads(
        feature_names.len(),
        targets.categorical,
        &model_categorical,
        known,
        cuts.is_none(),
    );
    let typed = |text_type: &dyn Fn(usize) -> Option<DataType>| -> Vec<SchemaRef> {
        schemas
            .iter()
            .zip(&feat_proj)
            .map(|(schema, cols)| {
                let types: Vec<(usize, DataType)> = cols
                    .iter()
                    .enumerate()
                    .filter_map(|(f, &c)| text_type(f).map(|t| (c, t)))
                    .collect();
                with_column_types(schema, &types)
            })
            .collect()
    };
    for (schema, cols) in schemas.iter().zip(&feat_proj) {
        let text_cols: Vec<usize> = cols
            .iter()
            .zip(&reads)
            .filter_map(|(&c, r)| r.as_text().then_some(c))
            .collect();
        check_numeric(schema, label, &text_cols)?;
    }
    let decide: Vec<bool> = reads.iter().map(|&r| r == CatRead::Decide).collect();
    let codes: Arc<[usize]> = (0..reads.len())
        .filter(|&f| reads[f] == CatRead::Codes)
        .collect();
    let first_pass_schemas = typed(&|f| reads[f].as_text().then_some(DataType::Utf8));

    let (reader_threads, transform_threads) = crate::source::pipeline_threads(nthread);
    let owned = paths.to_vec();
    let (sc, fp) = (first_pass_schemas.clone(), feat_proj.clone());
    let no_codes: Arc<[usize]> = Arc::from(Vec::new());
    let feats = || -> anyhow::Result<crate::source::parallel_reader::BatchIter> {
        Ok(dataset_reader(
            owned.clone(),
            sc.clone(),
            fp.clone(),
            has_header,
            reader_threads,
            no_codes.clone(),
        ))
    };
    let (cuts, known_rows) = match cuts {
        Some(cuts) => (cuts, None),
        None if targets.weight.is_some() => {
            // 加权分箱:权重列接在特征后面一起扫(CSV 本来就要读完整记录)。
            let _profile = crate::source::openprof::start("cuts");
            let proj = fp
                .iter()
                .zip(&weight_proj)
                .map(|(f, &w)| {
                    let mut cols = f.clone();
                    cols.push(w);
                    cols
                })
                .collect();
            let (cuts, n_rows) = crate::source::arrow_source::build_cuts_and_count_deciding(
                dataset_reader(
                    owned.clone(),
                    sc.clone(),
                    proj,
                    has_header,
                    reader_threads,
                    no_codes.clone(),
                ),
                max_bins as usize,
                transform_threads,
                true,
                targets.categorical.unwrap_or(&[]),
                &decide,
                targets.missing,
                targets.categories,
            )?;
            (cuts, Some(n_rows))
        }
        None => {
            let _profile = crate::source::openprof::start("cuts");
            let (cuts, n_rows) = crate::source::arrow_source::build_cuts_and_count_deciding(
                feats()?,
                max_bins as usize,
                transform_threads,
                false,
                targets.categorical.unwrap_or(&[]),
                &decide,
                targets.missing,
                targets.categories,
            )?;
            (cuts, Some(n_rows))
        }
    };

    // The first pass has counted the exact rows of every file: the block width
    // can be chosen now, before the second pass builds the blocks.
    let cols_per_block = match width {
        BlockWidth::Fixed(c) => c,
        BlockWidth::FromRowCount(choose) => {
            let n_rows = known_rows.ok_or_else(|| {
                anyhow::anyhow!(
                    "a CSV block width chosen from the row count needs the cut-building pass; it cannot be combined with given cuts"
                )
            })?;
            choose(n_rows as u64)?
        }
    };

    // Pass 1 has decided each decided column from every field: names are
    // text, codes are numbers (every field parses as one). The rest read as
    // in pass 1.
    let schemas = typed(&|f| match reads[f] {
        CatRead::Inferred => None,
        CatRead::Names | CatRead::Codes => Some(DataType::Utf8),
        CatRead::Decide => Some(if cuts.categories().is_some_and(|t| t.get(f).is_some()) {
            DataType::Utf8
        } else {
            DataType::Float64
        }),
    });

    // CSV must scan complete records even for a one-column projection. Read
    // the label beside features during the quantization pass, then expose a
    // cheap feature-only RecordBatch view to ArrowSource. This removes the
    // former third full-file scan without changing canonical feature order.
    // 权重列(若有)跟在标签后面,同一遍扫描里读出来。
    let combined_proj = feat_proj
        .into_iter()
        .zip(label_proj)
        .enumerate()
        .map(|(i, (mut features, label))| {
            features.push(label);
            if let Some(&w) = weight_proj.get(i) {
                features.push(w);
            }
            features
        })
        .collect::<Vec<_>>();
    let mut labels = Vec::new();
    let mut weight_values = Vec::new();
    let batches = FeatureLabelBatches::new(
        dataset_reader(
            paths.to_vec(),
            schemas,
            combined_proj,
            has_header,
            reader_threads,
            codes,
        ),
        feature_names.len(),
        &mut labels,
        targets.weight.map(|_| &mut weight_values),
    );
    let source = {
        let _profile = crate::source::openprof::start("quantize+label");
        match known_rows {
            Some(n_rows) => ArrowSource::from_batches_with_cuts_and_nthread_known_rows(
                batches,
                cuts,
                cols_per_block,
                n_rows,
                transform_threads,
                targets.missing,
            )?,
            None => ArrowSource::from_batches_with_cuts_and_nthread(
                batches,
                cuts,
                cols_per_block,
                transform_threads,
                targets.missing,
            )?,
        }
    };
    Dataset::assemble(
        source,
        labels,
        if has_header {
            feature_names
        } else {
            Vec::new()
        },
        label,
        targets.weight.map(|w| (weight_values, w)),
    )
}

/// Batches of every file in order, columns in each projection's order.
/// `codes` = output columns parsed by [`CheckedCodes`] (empty = none).
fn dataset_reader(
    paths: Vec<PathBuf>,
    schemas: Vec<SchemaRef>,
    projections: Vec<Vec<usize>>,
    has_header: bool,
    nthread: usize,
    codes: Arc<[usize]>,
) -> crate::source::parallel_reader::BatchIter {
    if nthread <= 1 {
        return Box::new(chained_csv(paths, schemas, projections, has_header, codes));
    }
    if paths.len() == 1 {
        let (mask, order) = crate::source::input::projection_and_order(&projections[0]);
        let path = paths[0].clone();
        return Box::new(CheckedCodes::new(
            parallel_csv_chunks(
                path.clone(),
                schemas[0].clone(),
                has_header,
                mask,
                order,
                nthread,
            ),
            path,
            codes,
        ));
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
            Ok(Box::new(CheckedCodes::new(
                iter.map(move |batch| {
                    batch.map(|batch| crate::source::input::reorder(&batch, &order))
                }),
                path,
                codes.clone(),
            )))
        },
    ))
}

/// Combined CSV feature+label batches are reordered to canonical features
/// followed by the label. Keep the label values and return a zero-copy Arrow
/// view containing only feature arrays for quantization.
struct FeatureLabelBatches<'a> {
    batches: crate::source::parallel_reader::BatchIter,
    n_features: usize,
    feature_projection: Vec<usize>,
    labels: &'a mut Vec<f32>,
    /// `Some` = 标签后面还跟着一列权重。
    weights: Option<&'a mut Vec<f32>>,
}

impl<'a> FeatureLabelBatches<'a> {
    fn new(
        batches: crate::source::parallel_reader::BatchIter,
        n_features: usize,
        labels: &'a mut Vec<f32>,
        weights: Option<&'a mut Vec<f32>>,
    ) -> Self {
        Self {
            batches,
            n_features,
            feature_projection: (0..n_features).collect(),
            labels,
            weights,
        }
    }
}

impl Iterator for FeatureLabelBatches<'_> {
    type Item = anyhow::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.batches.next()? {
            Ok(batch) => batch,
            Err(error) => return Some(Err(error)),
        };
        let extra = 1 + usize::from(self.weights.is_some());
        if batch.num_columns() != self.n_features + extra {
            return Some(Err(anyhow::anyhow!(
                "CSV combined batch has {} columns, expected {} features + {extra} label/weight columns",
                batch.num_columns(),
                self.n_features,
            )));
        }
        self.labels.reserve(batch.num_rows());
        if let Err(error) =
            crate::source::arrow_source::for_each_column_f32(&batch, self.n_features, |value| {
                self.labels.push(value)
            })
        {
            return Some(Err(error));
        }
        if let Some(weights) = self.weights.as_deref_mut() {
            weights.reserve(batch.num_rows());
            if let Err(error) = crate::source::arrow_source::for_each_column_f32(
                &batch,
                self.n_features + 1,
                |value| weights.push(value),
            ) {
                return Some(Err(error));
            }
        }
        Some(batch.project(&self.feature_projection).map_err(Into::into))
    }
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
                crate::bail_input!("{} has no CSV header", path.display());
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
    codes: Arc<[usize]>,
) -> impl Iterator<Item = anyhow::Result<RecordBatch>> {
    paths
        .into_iter()
        .zip(schemas)
        .zip(projections)
        .flat_map(move |((p, sc), cols)| {
            // 同 parquet:投影之后必须按规范顺序重排。
            let (mask, order) = crate::source::input::projection_and_order(&cols);
            match batch_reader(&p, sc, has_header, mask) {
                Ok(it) => Box::new(CheckedCodes::new(
                    it.map(move |b| b.map(|b| crate::source::input::reorder(&b, &order))),
                    p,
                    codes.clone(),
                ))
                    as Box<dyn Iterator<Item = anyhow::Result<RecordBatch>>>,
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        })
}

/// schema + 全部列名(含标签列)。预测绑定要用。
pub fn schema_and_names(
    path: impl AsRef<Path>,
    has_header: bool,
) -> anyhow::Result<(SchemaRef, Vec<String>)> {
    let schema = infer_schema(path.as_ref(), has_header)?;
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
        assert_eq!(
            ds.source.n_features(),
            3,
            "the label column is not a feature"
        );
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
                    "row {row} feature {f}"
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
        assert_eq!(ds.source.bin_at(1, 1), crate::types::MISSING_BIN as u32);
        assert_eq!(ds.source.bin_at(5, 1), crate::types::MISSING_BIN as u32);
        for row in [0usize, 2, 3, 4] {
            assert_ne!(ds.source.bin_at(row, 1), crate::types::MISSING_BIN as u32);
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
            panic!("string column should be rejected")
        };
        let msg = e.to_string();
        assert!(msg.contains("city"), "error should name the column: {msg}");
        assert!(msg.contains("only numeric columns are supported"), "{msg}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn declared_text_columns_learn_category_names_in_row_order() {
        use crate::category_names::CategoryValues;
        // File a's "city" fields are all integers, file b's are text: the
        // column is text in both, and the list follows the rows.
        let a = write("names_a", "city,target,x\n2,0,1\n1,1,2\n,0,3\n2,1,4\n");
        let b = write("names_b", "x,city,target\n5,Oslo,0\n6,1,1\n7,Lima,0\n");
        let flags = [true, false];
        let targets = Targets::from("target").with_categorical(&flags);
        let paths = vec![a.clone(), b.clone()];
        for nthread in [1, 3] {
            let ds = open_many_with_header_and_nthread(&paths, targets, None, 16, 2, true, nthread)
                .unwrap();
            let table = ds.source.cuts().categories().expect("names").clone();
            let strs = |v: &[&str]| CategoryValues::Str(v.iter().map(|s| s.to_string()).collect());
            assert_eq!(
                table.get(0).unwrap().values(),
                &strs(&["2", "1", "Oslo", "Lima"])
            );
            assert!(table.get(1).is_none());
            let bins: Vec<u32> = (0..7).map(|r| ds.source.bin_at(r, 0)).collect();
            let m = crate::types::MISSING_BIN as u32;
            assert_eq!(bins, vec![0, 1, m, 0, 2, 1, 3]);

            // A validation file of integers only is read as text and coded by value.
            let v = write("names_v", "city,target,x\n1,0,1\nLima,1,2\n");
            let valid = open_many_with_header_and_nthread(
                std::slice::from_ref(&v),
                targets.with_features(&ds.feature_names),
                Some(ds.source.cuts().clone()),
                0,
                2,
                true,
                nthread,
            )
            .unwrap();
            assert_eq!(valid.source.bin_at(0, 0), 1);
            assert_eq!(valid.source.bin_at(1, 0), 3);
            let unseen = write("names_u", "city,target,x\nRome,0,1\n");
            let err = open_many_with_header_and_nthread(
                std::slice::from_ref(&unseen),
                targets.with_features(&ds.feature_names),
                Some(ds.source.cuts().clone()),
                0,
                2,
                true,
                nthread,
            )
            .err()
            .expect("unseen category")
            .to_string();
            assert!(
                err.contains("\"city\"") && err.contains("\"Rome\""),
                "{err}"
            );
        }
        // Without names an all-integer categorical column keeps its codes.
        let ds = open_many_with_header_and_nthread(
            std::slice::from_ref(&a),
            targets,
            None,
            16,
            2,
            true,
            1,
        )
        .unwrap();
        assert!(ds.source.cuts().categories().is_none());
        // Undeclared text is rejected by name.
        let err = open_many(&paths, "target", None, 16, 2)
            .err()
            .expect("undeclared text")
            .to_string();
        assert!(
            err.contains("city") && err.contains("only numeric columns"),
            "{err}"
        );
    }

    #[test]
    fn missing_label_column_lists_what_is_there() {
        let path = write("no_label", BODY);
        let Err(e) = open(&path, "nope", 16, 2) else {
            panic!("should fail")
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
        assert_eq!(
            ds.source.n_rows(),
            4,
            "the first row must not be consumed as a header"
        );
        assert_eq!(ds.source.n_features(), 3);
        assert_eq!(ds.labels, vec![0.0, 1.0, 0.0, 1.0]);
        assert!(
            ds.feature_names.is_empty(),
            "a headerless model must be positional"
        );
        assert!(positional_label_name(&path, 4).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prepared_schema_matches_regular_open() {
        let path = write("prepared_schema", BODY);
        let schema = infer_schema(&path, true).unwrap();
        let prepared = open_many_with_header_and_nthread_prepared(
            std::slice::from_ref(&path),
            "target",
            None,
            16,
            BlockWidth::Fixed(2),
            true,
            4,
            Some(schema),
        )
        .unwrap();
        let regular = open(&path, "target", 16, 2).unwrap();

        assert_eq!(prepared.labels, regular.labels);
        assert_eq!(prepared.feature_names, regular.feature_names);
        assert_eq!(prepared.source.n_rows(), regular.source.n_rows());
        for feature in 0..prepared.source.n_features() as u32 {
            assert_eq!(
                prepared.source.cuts().cuts_for(feature),
                regular.source.cuts().cuts_for(feature)
            );
            for row in 0..prepared.source.n_rows() {
                assert_eq!(
                    prepared.source.bin_at(row, feature),
                    regular.source.bin_at(row, feature)
                );
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn block_width_from_the_aggregate_row_count_of_the_first_pass() {
        let a = write("width_rows_a", BODY);
        let b = write("width_rows_b", BODY);
        let paths = vec![a.clone(), b.clone()];
        let mut seen = Vec::new();
        let mut choose = |rows: u64| -> anyhow::Result<usize> {
            seen.push(rows);
            Ok(2)
        };
        let planned = open_many_with_header_and_nthread_prepared(
            &paths,
            "target",
            None,
            16,
            BlockWidth::FromRowCount(&mut choose),
            true,
            2,
            None,
        )
        .unwrap();
        assert_eq!(seen, vec![12], "the width is chosen once, from all files");
        let fixed = open_many(&paths, "target", None, 16, 2).unwrap();
        assert_eq!(planned.source.n_blocks(), fixed.source.n_blocks());
        assert_eq!(planned.labels, fixed.labels);
        for feature in 0..fixed.source.n_features() as u32 {
            for row in 0..fixed.source.n_rows() {
                assert_eq!(
                    planned.source.bin_at(row, feature),
                    fixed.source.bin_at(row, feature)
                );
            }
        }

        // A failed choice fails the open; given cuts have no first pass to count.
        let mut fail = |_: u64| -> anyhow::Result<usize> { anyhow::bail!("no plan fits") };
        let err = match open_many_with_header_and_nthread_prepared(
            &paths,
            "target",
            None,
            16,
            BlockWidth::FromRowCount(&mut fail),
            true,
            2,
            None,
        ) {
            Ok(_) => panic!("a failed width choice must fail the open"),
            Err(e) => e,
        };
        assert!(format!("{err:#}").contains("no plan fits"));
        let mut never = |_: u64| -> anyhow::Result<usize> { unreachable!() };
        assert!(open_many_with_header_and_nthread_prepared(
            &paths,
            "target",
            Some(fixed.source.cuts().clone()),
            16,
            BlockWidth::FromRowCount(&mut never),
            true,
            2,
            None,
        )
        .is_err());
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
    }
}
