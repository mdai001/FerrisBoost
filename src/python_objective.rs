//! Python backed custom objectives.
//!
//! Inputs are copied into owned NumPy arrays before calling user code. This
//! lets callbacks retain their snapshots after the call returns. Callback
//! gradients and Hessians are unweighted; native training applies sample
//! weights after this adapter returns.

use anyhow::{anyhow, Result};
use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyTupleMethods};

use crate::custom_objective::CustomObjective;
use crate::types::GradPair;

/// A Python description of a custom objective. Clones share the callback.
#[pyclass(name = "CustomObjective", module = "ferrisboost._core", frozen)]
pub(crate) struct PyCustomObjective {
    name: String,
    callback: Py<PyAny>,
}

#[pymethods]
impl PyCustomObjective {
    #[new]
    fn new(py: Python<'_>, name: String, callback: Py<PyAny>) -> PyResult<Self> {
        if name.trim().is_empty() {
            return Err(PyValueError::new_err(
                "custom objective name must not be empty",
            ));
        }
        if !callback.bind(py).is_callable() {
            return Err(PyValueError::new_err(
                "custom objective callback must be callable",
            ));
        }
        Ok(Self { name, callback })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __copy__(&self, py: Python<'_>) -> Self {
        self.clone_descriptor(py)
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> Self {
        // The executable callback is intentionally shared, as with Python
        // functions' normal deepcopy behavior.
        self.clone_descriptor(py)
    }
}

impl PyCustomObjective {
    fn clone_descriptor(&self, py: Python<'_>) -> Self {
        Self {
            name: self.name.clone(),
            callback: self.callback.clone_ref(py),
        }
    }

    pub(crate) fn to_objective(&self, py: Python<'_>) -> Box<dyn CustomObjective> {
        Box::new(PythonObjective {
            name: self.name.clone(),
            callback: self.callback.clone_ref(py),
        })
    }
}

struct PythonObjective {
    name: String,
    callback: Py<PyAny>,
}

impl CustomObjective for PythonObjective {
    fn name(&self) -> &str {
        &self.name
    }

    fn gradients(&self, raw_margin: &[f32], labels: &[f32], out: &mut [GradPair]) -> Result<()> {
        if raw_margin.len() != labels.len() || out.len() != labels.len() {
            return Err(anyhow!(
                "custom objective {:?} received inconsistent row counts: margins {}, labels {}, output {}",
                self.name,
                raw_margin.len(),
                labels.len(),
                out.len()
            ));
        }

        let row_count = labels.len();
        Python::attach(|py| {
            // Fresh owned snapshots keep Python-retained arrays independent
            // from the native training buffers.
            let margins = PyArray1::from_vec(py, raw_margin.to_vec()).into_any();
            let labels = PyArray1::from_vec(py, labels.to_vec()).into_any();
            let result = self
                .callback
                .bind(py)
                .call1((margins, labels))
                .map_err(anyhow::Error::new)?;

            let tuple = result.cast::<PyTuple>().map_err(|_| {
                anyhow::Error::new(PyValueError::new_err(
                    "custom objective callback must return a 2-tuple of float32 NumPy arrays (gradient, hessian)",
                ))
            })?;
            if tuple.len() != 2 {
                return Err(anyhow::Error::new(PyValueError::new_err(
                    "custom objective callback must return exactly two arrays: (gradient, hessian)",
                )));
            }

            let gradient = tuple
                .get_item(0)
                .and_then(|item| {
                    item.extract::<PyReadonlyArray1<'_, f32>>()
                        .map_err(PyErr::from)
                })
                .map_err(|_| {
                    anyhow::Error::new(PyValueError::new_err(
                        "custom objective gradient must be a one-dimensional float32 NumPy array",
                    ))
                })?;
            let hessian = tuple
                .get_item(1)
                .and_then(|item| {
                    item.extract::<PyReadonlyArray1<'_, f32>>()
                        .map_err(PyErr::from)
                })
                .map_err(|_| {
                    anyhow::Error::new(PyValueError::new_err(
                        "custom objective hessian must be a one-dimensional float32 NumPy array",
                    ))
                })?;

            if !gradient.is_c_contiguous() {
                return Err(anyhow::Error::new(PyValueError::new_err(
                    "custom objective gradient must be C-contiguous",
                )));
            }
            if !hessian.is_c_contiguous() {
                return Err(anyhow::Error::new(PyValueError::new_err(
                    "custom objective hessian must be C-contiguous",
                )));
            }
            if gradient.len() != row_count {
                return Err(anyhow::Error::new(PyValueError::new_err(format!(
                    "custom objective gradient must have {} rows, got {}",
                    row_count,
                    gradient.len()
                ))));
            }
            if hessian.len() != row_count {
                return Err(anyhow::Error::new(PyValueError::new_err(format!(
                    "custom objective hessian must have {} rows, got {}",
                    row_count,
                    hessian.len()
                ))));
            }

            let gradient = gradient.as_slice().map_err(|_| {
                anyhow::Error::new(PyValueError::new_err(
                    "custom objective gradient must be C-contiguous",
                ))
            })?;
            let hessian = hessian.as_slice().map_err(|_| {
                anyhow::Error::new(PyValueError::new_err(
                    "custom objective hessian must be C-contiguous",
                ))
            })?;
            for ((pair, &grad), &hess) in out.iter_mut().zip(gradient).zip(hessian) {
                *pair = GradPair { grad, hess };
            }
            Ok(())
        })
    }
}
