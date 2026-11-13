//! Python backed custom evaluation metrics.
//!
//! Inputs are copied into owned NumPy arrays before calling user code. This
//! keeps the arrays valid if the callback stores them after it returns.

use anyhow::{anyhow, bail, Result};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::metrics::Metric;

/// A Python description of a custom metric. The descriptor can be reused to
/// create independent native metric adapters for different output widths.
#[pyclass(name = "CustomMetric", module = "ferrisboost._core", frozen)]
pub(crate) struct PyCustomMetric {
    name: String,
    callback: Py<PyAny>,
    greater_is_better: bool,
}

#[pymethods]
impl PyCustomMetric {
    #[new]
    #[pyo3(signature = (name, callback, *, greater_is_better=false))]
    fn new(
        py: Python<'_>,
        name: String,
        callback: Py<PyAny>,
        greater_is_better: bool,
    ) -> PyResult<Self> {
        if name.trim().is_empty() {
            return Err(PyValueError::new_err(
                "custom metric name must not be empty",
            ));
        }
        if !callback.bind(py).is_callable() {
            return Err(PyValueError::new_err(
                "custom metric callback must be callable",
            ));
        }
        Ok(Self {
            name,
            callback,
            greater_is_better,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn greater_is_better(&self) -> bool {
        self.greater_is_better
    }

    fn __copy__(&self, py: Python<'_>) -> Self {
        self.clone_descriptor(py)
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> Self {
        // The executable callback is intentionally shared, like a Python
        // function's normal deepcopy behavior. We do not serialize code.
        self.clone_descriptor(py)
    }
}

impl PyCustomMetric {
    fn clone_descriptor(&self, py: Python<'_>) -> Self {
        Self {
            name: self.name.clone(),
            callback: self.callback.clone_ref(py),
            greater_is_better: self.greater_is_better,
        }
    }

    /// Build a training metric with the model's prediction output width.
    pub(crate) fn to_metric(&self, py: Python<'_>, noutputs: usize) -> Box<dyn Metric> {
        Box::new(PythonMetric {
            name: self.name.clone(),
            callback: self.callback.clone_ref(py),
            greater_is_better: self.greater_is_better,
            noutputs,
        })
    }
}

struct PythonMetric {
    name: String,
    callback: Py<PyAny>,
    greater_is_better: bool,
    noutputs: usize,
}

impl PythonMetric {
    fn callback_result(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: Option<&[u32]>,
    ) -> Result<f32> {
        if self.noutputs == 0 {
            bail!("custom metric output width must be positive");
        }
        let expected_pred_len = label
            .len()
            .checked_mul(self.noutputs)
            .ok_or_else(|| anyhow!("custom metric prediction shape overflows usize"))?;
        if pred.len() != expected_pred_len {
            bail!(
                "custom metric {:?} received {} predictions for {} rows and {} outputs",
                self.name,
                pred.len(),
                label.len(),
                self.noutputs
            );
        }
        // Entry points validate objective-specific weight shape. Groups may also
        // accompany non-ranking objectives, whose weights remain per row.

        Python::attach(|py| {
            let prediction = PyArray1::from_vec(py, pred.to_vec());
            let prediction: Bound<'_, PyAny> = if self.noutputs == 1 {
                prediction.into_any()
            } else {
                prediction
                    .reshape([label.len(), self.noutputs])
                    .map_err(|e| anyhow::Error::new(PyErr::from(e)))?
                    .into_any()
            };
            let labels = PyArray1::from_vec(py, label.to_vec()).into_any();
            let weights = weight.map(|w| PyArray1::from_vec(py, w.to_vec()).into_any());
            let groups = group_ptr.map(|g| PyArray1::from_vec(py, g.to_vec()).into_any());

            let result = self
                .callback
                .bind(py)
                .call1((prediction, labels, weights, groups))
                .map_err(anyhow::Error::new)?;
            let value = result.extract::<f64>().map_err(|_| {
                anyhow::Error::new(PyValueError::new_err(
                    "custom metric callback must return a finite scalar",
                ))
            })?;
            if !value.is_finite() {
                return Err(anyhow::Error::new(PyValueError::new_err(
                    "custom metric callback must return a finite scalar",
                )));
            }
            let value = value as f32;
            if !value.is_finite() {
                return Err(anyhow::Error::new(PyValueError::new_err(
                    "custom metric callback result is outside the finite f32 range",
                )));
            }
            Ok(value)
        })
    }
}

impl Metric for PythonMetric {
    fn name(&self) -> &str {
        &self.name
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.try_eval_grouped(pred, label, None, None)
            .expect("Python custom metric failed through infallible Metric::eval")
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        self.try_eval_grouped(pred, label, weight, None)
            .expect("Python custom metric failed through infallible Metric::eval_weighted")
    }

    fn eval_grouped(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: Option<&[u32]>,
    ) -> f32 {
        self.try_eval_grouped(pred, label, weight, group_ptr)
            .expect("Python custom metric failed through infallible Metric::eval_grouped")
    }

    fn try_eval_grouped(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: Option<&[u32]>,
    ) -> Result<f32> {
        self.callback_result(pred, label, weight, group_ptr)
    }

    fn lower_is_better(&self) -> bool {
        !self.greater_is_better
    }
}
