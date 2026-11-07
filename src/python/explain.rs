//! Python glue for feature importance, SHAP contributions / interactions and
//! leaf indices (`Model.get_score`, `Model.predict(pred_contribs=...)`).
//! The numerics live in `crate::explain`; this layer only resolves inputs
//! (NumPy arrays or file specs), picks the worker count and shapes the output.
//!
//! These outputs are always computed on the CPU, whatever `use_gpu` says.

use numpy::ndarray::IxDyn;

use super::*;
use crate::explain::{Explainer, Kind};

/// Which explanation `predict` was asked for (at most one).
#[derive(Clone, Copy)]
pub(super) struct ExplainRequest {
    pub contribs: bool,
    pub approx: bool,
    pub interactions: bool,
    pub leaf: bool,
}

impl ExplainRequest {
    pub(super) fn any(self) -> bool {
        self.contribs || self.interactions || self.leaf
    }

    fn kind(self) -> PyResult<Kind> {
        if [self.contribs, self.interactions, self.leaf]
            .iter()
            .filter(|&&f| f)
            .count()
            > 1
        {
            return Err(PyValueError::new_err(
                "pred_contribs, pred_interactions and pred_leaf: one type of prediction at a time",
            ));
        }
        if self.interactions {
            if self.approx {
                return Err(PyValueError::new_err(
                    "pred_interactions=True with approx_contribs=True is not supported: \
                     XGBoost 3.4.1 ignores the feature conditioning there and returns a \
                     diagonal-only matrix. Use exact interactions (approx_contribs=False).",
                ));
            }
            Ok(Kind::Interactions)
        } else if self.contribs {
            Ok(Kind::Contribs {
                approx: self.approx,
            })
        } else {
            Ok(Kind::Leaf)
        }
    }
}

/// Rough relative cost of one explained row versus a plain prediction row, so
/// small inputs stay serial under automatic worker selection.
const EXPLAIN_WORK_FACTOR: usize = 32;

impl PyModel {
    /// `Model.get_score`: feature index -> name (`f{i}` when the model has no
    /// names, as XGBoost does) and the importance of every used feature.
    pub(super) fn feature_score_dict<'py>(
        &self,
        py: Python<'py>,
        importance_type: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kind = crate::explain::Importance::parse(importance_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown importance_type {importance_type:?}, expected one of: \
                 \"weight\", \"gain\", \"cover\", \"total_gain\", \"total_cover\""
            ))
        })?;
        let out = PyDict::new(py);
        for (feature, score) in crate::explain::feature_scores(&self.inner, kind) {
            let name = match self.inner.feature_names.get(feature) {
                Some(name) => name.clone(),
                None => format!("f{feature}"),
            };
            out.set_item(name, f64::from(score))?;
        }
        Ok(out)
    }

    /// File inputs need the same schema contract as `predict_files`; this
    /// mirrors its checks (the explain path reads full canonical rows).
    fn check_explain_file_schema(
        &self,
        resolved: &crate::source::input::ResolvedInput,
        header: bool,
    ) -> PyResult<()> {
        use crate::source::input::InputFormat;
        use crate::tree::SchemaMode;
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
        if resolved.format == InputFormat::Parquet && !header {
            return Err(PyValueError::new_err("header=False 只适用于 CSV"));
        }
        if resolved.format == InputFormat::Csv {
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
        Ok(())
    }

    fn explain_output_shape(&self, kind: Kind, rows: usize, n_trees: usize) -> Vec<usize> {
        let nc = self.inner.n_features + 1;
        let k = self.inner.num_outputs();
        let mut dims = vec![rows];
        match kind {
            Kind::Leaf => dims.push(n_trees),
            Kind::Contribs { .. } => {
                if k > 1 {
                    dims.push(k);
                }
                dims.push(nc);
            }
            Kind::Interactions => {
                if k > 1 {
                    dims.push(k);
                }
                dims.push(nc);
                dims.push(nc);
            }
        }
        dims
    }

    /// `predict(pred_contribs / pred_interactions / pred_leaf)`.
    ///
    /// Shapes follow XGBoost with `strict_shape=False`: contributions
    /// `(n, F+1)` (multiclass `(n, K, F+1)`), interactions `(n, F+1, F+1)`
    /// (multiclass `(n, K, F+1, F+1)`), leaves `(n, n_trees)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn predict_explain<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        request: ExplainRequest,
        output_margin: bool,
        n_trees: Option<usize>,
        header: bool,
        predict_threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        guard(|| {
            let kind = request.kind()?;
            if output_margin {
                return Err(PyValueError::new_err(
                    "output_margin cannot be combined with pred_contribs / \
                     pred_interactions / pred_leaf: one type of prediction at a time",
                ));
            }
            let upto = n_trees
                .unwrap_or(self.inner.trees.len())
                .min(self.inner.trees.len());
            let explainer = Explainer::new(&self.inner, upto, kind).map_err(to_py_err)?;
            let work_per_row =
                prediction_work_per_row(&self.inner, upto).saturating_mul(EXPLAIN_WORK_FACTOR);
            let n_features = self.inner.n_features;

            if let Some(specs) = path_specs_of(data)? {
                let resolved = crate::source::input::resolve(&specs)
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                self.check_explain_file_schema(&resolved, header)?;
                let projected: Vec<usize> = (0..n_features).collect();
                let initially_resolved_ingest = resolved_ingest_workers(
                    self.ingest_threads,
                    resolved.format.label(),
                    &resolved.paths,
                    n_features,
                    n_features.max(1),
                );
                let (ingest_workers, predict_worker_cap, _) = prediction_pipeline_workers(
                    self.ingest_threads,
                    predict_threads,
                    initially_resolved_ingest,
                );
                let mut pool = None;
                let mut out: Vec<f32> = Vec::new();
                let rows = py
                    .detach(|| {
                        crate::source::predict_input::for_each_projected_contiguous_batch(
                            &resolved,
                            &self.inner.feature_names,
                            n_features,
                            &projected,
                            header,
                            ingest_workers,
                            |batch, n_rows| {
                                let workers = resolved_predict_workers(
                                    predict_threads,
                                    predict_worker_cap,
                                    n_rows,
                                    work_per_row,
                                );
                                out.extend(
                                    explainer.compute(kind, batch, n_rows, workers, &mut pool)?,
                                );
                                Ok(())
                            },
                        )
                    })
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
                return self.explain_array(py, out, kind, rows, upto);
            }

            let x: PyReadonlyArray2<'py, f32> = data.extract().map_err(|_| {
                PyValueError::new_err(
                    "predict 只接受文件路径(单个 / 列表 / 目录 / glob)或二维 f32 numpy 数组",
                )
            })?;
            let shape = x.shape();
            let (n_rows, cols) = (shape[0], shape[1]);
            if cols != n_features {
                return Err(PyValueError::new_err(format!(
                    "模型是按 {n_features} 个特征训的,传进来的是 {cols} 列"
                )));
            }
            let view = x.as_array();
            if let Some((r, c)) = view
                .indexed_iter()
                .find(|(_, &v)| v == f32::MAX)
                .map(|(ix, _)| ix)
            {
                return Err(PyValueError::new_err(format!(
                    "预测数据第 {r} 行第 {c} 列是 f32::MAX:这个值保留给缺失值分裂边界,训练和预测都拒绝它;请先缩放这个特征"
                )));
            }
            let gathered;
            let rows: &[f32] = if x.is_c_contiguous() {
                x.as_slice()?
            } else {
                gathered = view.iter().copied().collect::<Vec<f32>>();
                &gathered
            };
            let workers = resolved_predict_workers(
                predict_threads,
                effective_nthread(predict_threads),
                n_rows,
                work_per_row,
            );
            let mut pool = None;
            let out = py
                .detach(|| explainer.compute(kind, rows, n_rows, workers, &mut pool))
                .map_err(to_py_err)?;
            self.explain_array(py, out, kind, n_rows, upto)
        })
    }

    fn explain_array<'py>(
        &self,
        py: Python<'py>,
        out: Vec<f32>,
        kind: Kind,
        rows: usize,
        n_trees: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let dims = self.explain_output_shape(kind, rows, n_trees);
        PyArray1::from_vec(py, out)
            .reshape(IxDyn(&dims))
            .map(Bound::into_any)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
