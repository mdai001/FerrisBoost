//! Python glue for `base_margin` (XGBoost's per-row starting margin).
//!
//! A margin is an O(rows x outputs) vector, like labels and weights: it lives in
//! a flat array next to the data and never inside the column-blocked feature
//! storage. Array inputs carry it as an array; file inputs name the column(s)
//! holding it, read once as plain f32 columns (a single column per output, so
//! `multi:softprob` takes `num_class` column names).
//!
//! The numerics of using it live elsewhere: training starts every row's
//! prediction from it (`train::TrainConfig::base_margin`), prediction and SHAP
//! accumulate on top of it instead of on the intercept.

use super::*;
use crate::source::input::{InputFormat, ResolvedInput};

/// Normalize `base_margin` through the Python layer to a C-contiguous
/// `rows x k` float32 matrix (accepts `(n,)` for one output, `(n, k)`, lists,
/// pandas / pyarrow columns).
pub(super) fn margin_matrix<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    k: usize,
) -> PyResult<PyReadonlyArray2<'py, f32>> {
    py.import("ferrisboost")?
        .getattr("_as_base_margin")?
        .call1((value, k))?
        .extract::<PyReadonlyArray2<'py, f32>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Column names for a file `base_margin`: a string (one output) or a list of
/// `k` strings.
pub(super) fn margin_column_names(value: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<String>> {
    let names: Vec<String> = if let Ok(name) = value.extract::<String>() {
        vec![name]
    } else if let Ok(list) = value.extract::<Vec<String>>() {
        list
    } else {
        return Err(PyValueError::new_err(
            "for file inputs base_margin must be a column name (a list of num_class names for multiclass)",
        ));
    };
    if names.len() != k {
        return Err(PyValueError::new_err(format!(
            "base_margin names {} columns, the model has {k} outputs",
            names.len()
        )));
    }
    let mut seen = std::collections::HashSet::new();
    if let Some(dup) = names.iter().find(|n| !seen.insert(n.as_str())) {
        return Err(PyValueError::new_err(format!(
            "duplicate base_margin column {dup:?}"
        )));
    }
    Ok(names)
}

/// One whole f32 column of a Parquet / CSV file.
fn read_column(
    format: InputFormat,
    path: &std::path::Path,
    name: &str,
    header: bool,
) -> anyhow::Result<Vec<f32>> {
    match format {
        InputFormat::Parquet => crate::source::parquet_source::read_f32_column(path, name),
        InputFormat::Csv => {
            let (schema, names) = crate::source::csv_source::schema_and_names(path, header)?;
            let index = names.iter().position(|n| n == name).with_context(|| {
                format!("{} has no base_margin column {name:?}", path.display())
            })?;
            let mut out = Vec::new();
            for batch in crate::source::csv_source::raw_batches(path, schema, &[index], header)? {
                out.extend(crate::source::arrow_source::column_as_f32(&batch?, 0)?);
            }
            Ok(out)
        }
    }
}

/// Read the margin column(s) of every resolved file, in file order, into
/// row-major `rows x k` values. Every column must be finite and have the same
/// number of rows.
pub(super) fn read_margin_columns(
    resolved: &ResolvedInput,
    names: &[String],
    header: bool,
) -> anyhow::Result<Vec<f32>> {
    let k = names.len();
    let mut per_output: Vec<Vec<f32>> = vec![Vec::new(); k];
    for path in &resolved.paths {
        let mut rows_here = None;
        for (c, name) in names.iter().enumerate() {
            let col = read_column(resolved.format, path, name, header)?;
            anyhow::ensure!(
                col.iter().all(|v| v.is_finite()),
                "base_margin column {name:?} contains NaN or inf (file {})",
                path.display()
            );
            match rows_here {
                None => rows_here = Some(col.len()),
                Some(n) => anyhow::ensure!(
                    n == col.len(),
                    "base_margin columns have different row counts (file {})",
                    path.display()
                ),
            }
            per_output[c].extend(col);
        }
    }
    if k == 1 {
        return Ok(per_output.pop().expect("one output"));
    }
    let n = per_output[0].len();
    let mut rows = vec![0.0f32; n * k];
    for (c, col) in per_output.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            rows[i * k + c] = v;
        }
    }
    Ok(rows)
}

impl PyModel {
    /// Resolve the `base_margin` argument of a predict call into row-major
    /// `rows x num_outputs` values. Arrays are checked against the data's row
    /// count here; file margins are checked against the rows actually read.
    pub(super) fn resolve_predict_margin<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        base_margin: &Bound<'py, PyAny>,
        header: bool,
    ) -> PyResult<Vec<f32>> {
        let k = self.inner.num_outputs();
        if let Some(specs) = path_specs_of(data)? {
            let names = margin_column_names(base_margin, k)?;
            let resolved = crate::source::input::resolve(&specs)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            self.check_file_schema(&resolved, header)?;
            // The margin column sits next to the features, so the file must be
            // bound by name; a positional model scores files of exactly its
            // feature width and has nowhere to put a margin column.
            if self.inner.feature_names.is_empty() {
                return Err(PyValueError::new_err(
                    "a base_margin column in a prediction file needs a model with feature names (columns are \
                     bound by name); this model is positional, pass base_margin as an array",
                ));
            }
            if let Some(clash) = names.iter().find(|n| self.inner.feature_names.contains(n)) {
                return Err(PyValueError::new_err(format!(
                    "the base_margin column {clash:?} is also a feature name of the model"
                )));
            }
            return py
                .detach(|| read_margin_columns(&resolved, &names, header))
                .map_err(|e| PyValueError::new_err(format!("{e:#}")));
        }
        let matrix = margin_matrix(py, base_margin, k)?;
        let n_rows = array_input(py, data)?.shape()[0];
        if matrix.shape()[0] != n_rows {
            return Err(PyValueError::new_err(format!(
                "base_margin has {} rows, the data has {n_rows} rows",
                matrix.shape()[0]
            )));
        }
        Ok(matrix.as_array().iter().copied().collect())
    }
}

/// Validate a `(rows, k)` training margin and lay it out class-major (`k x
/// rows`), the layout the training loop and the device prediction buffer use.
pub(super) fn class_major_margin(
    matrix: &PyReadonlyArray2<'_, f32>,
    rows: usize,
    k: usize,
    what: &str,
) -> PyResult<Vec<f32>> {
    let shape = matrix.shape();
    if shape[0] != rows || shape[1] != k {
        return Err(PyValueError::new_err(format!(
            "{what} has shape ({}, {}), expected ({rows}, {k})",
            shape[0], shape[1]
        )));
    }
    let row_major: Vec<f32> = matrix.as_array().iter().copied().collect();
    Ok(crate::train::class_major_from_rows(&row_major, rows, k))
}

/// Feature names of a continued model. A model that already has names keeps
/// them when the new data is positional (numpy); two named sides must agree
/// exactly, in order, because the new trees index features by position.
pub(super) fn continued_feature_names(
    prior: Option<&PyModel>,
    names: Vec<String>,
) -> PyResult<Vec<String>> {
    let Some(prior) = prior else {
        return Ok(names);
    };
    let old = &prior.inner.feature_names;
    if names.is_empty() {
        return Ok(old.clone());
    }
    if !old.is_empty() && *old != names {
        return Err(PyValueError::new_err(format!(
            "the feature names of the new data {names:?} differ from those of init_model {old:?} (names and order must match)"
        )));
    }
    Ok(names)
}

/// Column names of a training file's `base_margin`: names with a header, or
/// zero-based column indices for a headerless CSV (resolved to the schema's
/// positional names, like the label / weight column).
pub(super) fn file_margin_names(
    value: &Bound<'_, PyAny>,
    k: usize,
    header: bool,
    csv_schema: Option<&arrow::datatypes::Schema>,
    label: &str,
) -> PyResult<Vec<String>> {
    let names = if header {
        margin_column_names(value, k)?
    } else {
        let indices: Vec<isize> = if let Ok(i) = value.extract::<isize>() {
            vec![i]
        } else if let Ok(list) = value.extract::<Vec<isize>>() {
            list
        } else {
            return Err(PyValueError::new_err(
                "with header=False base_margin must be a zero-based column index (a list for multiclass)",
            ));
        };
        if indices.len() != k {
            return Err(PyValueError::new_err(format!(
                "base_margin gives {} column indices, the model has {k} outputs",
                indices.len()
            )));
        }
        let schema = csv_schema.ok_or_else(|| {
            PyValueError::new_err("base_margin column indices are for headerless CSV only")
        })?;
        indices
            .into_iter()
            .map(|i| {
                if i < 0 {
                    return Err(PyValueError::new_err(format!(
                        "a base_margin column index cannot be negative, got {i}"
                    )));
                }
                crate::source::csv_source::positional_label_name_from_schema(schema, i as usize)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))
            })
            .collect::<PyResult<Vec<_>>>()?
    };
    if names.iter().any(|n| n == label) {
        return Err(PyValueError::new_err(format!(
            "the base_margin column and the label column cannot be the same ({label:?})"
        )));
    }
    Ok(names)
}

/// Item `i` of an optional Python list, `None` for a missing tail or a `None`
/// entry.
pub(super) fn list_item<'py>(
    list: Option<&Bound<'py, PyList>>,
    i: usize,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match list {
        Some(l) if i < l.len() => {
            let item = l.get_item(i)?;
            Ok((!item.is_none()).then_some(item))
        }
        _ => Ok(None),
    }
}

/// The class-major starting margin of one file dataset: the explicit
/// `(rows, k)` array when supplied, else the named file column(s), else none.
pub(super) fn dataset_margin(
    supplied: Option<&Bound<'_, PyAny>>,
    resolved: &ResolvedInput,
    names: &[String],
    header: bool,
    rows: usize,
    k: usize,
    what: &str,
) -> PyResult<Option<Vec<f32>>> {
    if let Some(value) = supplied.filter(|v| !v.is_none()) {
        let matrix: PyReadonlyArray2<f32> = value.extract()?;
        return class_major_margin(&matrix, rows, k, &format!("{what} base_margin")).map(Some);
    }
    if names.is_empty() {
        return Ok(None);
    }
    let row_major = read_margin_columns(resolved, names, header)
        .map_err(|e| PyValueError::new_err(format!("{what}:{e:#}")))?;
    if row_major.len() != rows * k {
        return Err(PyValueError::new_err(format!(
            "{what} base_margin has {} rows, the data has {rows} rows",
            row_major.len() / k
        )));
    }
    Ok(Some(crate::train::class_major_from_rows(
        &row_major, rows, k,
    )))
}
