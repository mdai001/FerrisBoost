//! Python backed custom objectives.
//!
//! Inputs are copied into owned NumPy arrays before calling user code. This
//! lets callbacks retain their snapshots after the call returns. Callback
//! gradients and Hessians are unweighted; native training applies sample
//! weights after this adapter returns.

use anyhow::{anyhow, Result};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyTupleMethods};

use crate::custom_objective::{CustomObjective, CustomObjectiveKind};
use crate::types::GradPair;

/// A Python description of a custom objective. Clones share the callback.
#[pyclass(name = "CustomObjective", module = "ferrisboost._core", frozen)]
pub(crate) struct PyCustomObjective {
    name: String,
    callback: Py<PyAny>,
    kind: CustomObjectiveKind,
}

#[pymethods]
impl PyCustomObjective {
    #[new]
    #[pyo3(signature = (name, callback, *, kind="scalar"))]
    fn new(py: Python<'_>, name: String, callback: Py<PyAny>, kind: &str) -> PyResult<Self> {
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
        let kind = match kind {
            "scalar" => CustomObjectiveKind::Scalar,
            "multiclass" => CustomObjectiveKind::Multiclass,
            "ranking" => CustomObjectiveKind::Ranking,
            other => {
                return Err(PyValueError::new_err(format!(
                "custom objective kind must be 'scalar', 'multiclass', or 'ranking', got {other:?}"
            )))
            }
        };
        Ok(Self {
            name,
            callback,
            kind,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.kind {
            CustomObjectiveKind::Scalar => "scalar",
            CustomObjectiveKind::Multiclass => "multiclass",
            CustomObjectiveKind::Ranking => "ranking",
        }
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
            kind: self.kind,
        }
    }

    pub(crate) fn to_objective(&self, py: Python<'_>) -> Box<dyn CustomObjective> {
        Box::new(PythonObjective {
            name: self.name.clone(),
            callback: self.callback.clone_ref(py),
            kind: self.kind,
        })
    }
}

struct PythonObjective {
    name: String,
    callback: Py<PyAny>,
    kind: CustomObjectiveKind,
}

impl CustomObjective for PythonObjective {
    fn kind(&self) -> CustomObjectiveKind {
        self.kind
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn gradients(&self, raw_margin: &[f32], labels: &[f32], out: &mut [GradPair]) -> Result<()> {
        anyhow::ensure!(
            self.kind == CustomObjectiveKind::Scalar,
            "custom objective {:?} is kind {:?}, not scalar",
            self.name,
            self.kind
        );
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

    fn gradients_multiclass(
        &self,
        raw_margin: &[f32],
        labels: &[f32],
        n_outputs: usize,
        out: &mut [GradPair],
    ) -> Result<()> {
        anyhow::ensure!(
            self.kind == CustomObjectiveKind::Multiclass,
            "custom objective {:?} is kind {:?}, not multiclass",
            self.name,
            self.kind
        );
        anyhow::ensure!(n_outputs > 0 && raw_margin.len() == labels.len() * n_outputs
            && out.len() == raw_margin.len(),
            "custom objective {:?} received inconsistent multiclass shapes: margins {}, labels {}, outputs {}, output {}",
            self.name, raw_margin.len(), labels.len(), n_outputs, out.len());

        let n = labels.len();
        Python::attach(|py| {
            // The native contract is class-major; expose conventional (rows, classes).
            // Flatten and reshape to avoid allocating one Vec per row.
            let mut row_major = vec![0.0; raw_margin.len()];
            for i in 0..n {
                for k in 0..n_outputs {
                    row_major[i * n_outputs + k] = raw_margin[k * n + i];
                }
            }
            let margins = PyArray1::from_vec(py, row_major)
                .reshape([n, n_outputs])
                .map_err(|e| anyhow::Error::new(PyErr::from(e)))?
                .into_any();
            let labels = PyArray1::from_vec(py, labels.to_vec()).into_any();
            let result = self
                .callback
                .bind(py)
                .call1((margins, labels))
                .map_err(anyhow::Error::new)?;
            extract_pair(&result, &[n, n_outputs], "multiclass", out, true)
        })
    }

    fn gradients_ranking(
        &self,
        raw_margin: &[f32],
        labels: &[f32],
        group_ptr: &[u32],
        out: &mut [GradPair],
    ) -> Result<()> {
        anyhow::ensure!(
            self.kind == CustomObjectiveKind::Ranking,
            "custom objective {:?} is kind {:?}, not ranking",
            self.name,
            self.kind
        );
        anyhow::ensure!(raw_margin.len() == labels.len() && out.len() == labels.len(),
            "custom objective {:?} received inconsistent ranking shapes: margins {}, labels {}, output {}",
            self.name, raw_margin.len(), labels.len(), out.len());
        let n = labels.len();
        Python::attach(|py| {
            let margins = PyArray1::from_vec(py, raw_margin.to_vec()).into_any();
            let labels = PyArray1::from_vec(py, labels.to_vec()).into_any();
            let groups = PyArray1::from_vec(py, group_ptr.to_vec()).into_any();
            let result = self
                .callback
                .bind(py)
                .call1((margins, labels, groups))
                .map_err(anyhow::Error::new)?;
            extract_pair(&result, &[n], "ranking", out, false)
        })
    }
}

/// Extract and validate the derivative pair while preserving Python callback
/// exceptions as their original PyErr cause (only contract errors are replaced).
fn extract_pair<'py>(
    result: &Bound<'py, PyAny>,
    expected_shape: &[usize],
    kind: &str,
    out: &mut [GradPair],
    class_major_output: bool,
) -> Result<()> {
    let tuple = result.cast::<PyTuple>().map_err(|_| anyhow::Error::new(
        PyValueError::new_err(format!(
            "{kind} custom objective callback must return a 2-tuple of float32 NumPy arrays (gradient, hessian)"
        ))) )?;
    crate::ensure_input!(
        tuple.len() == 2,
        "{kind} custom objective callback must return exactly two arrays: (gradient, hessian)"
    );
    crate::ensure_input!(
        out.len() == expected_shape.iter().product::<usize>(),
        "{kind} custom objective output length does not match expected shape {:?}",
        expected_shape
    );
    let extract = |index: usize, what: &str, hessian: bool, out: &mut [GradPair]| -> Result<()> {
        let item = tuple.get_item(index).map_err(anyhow::Error::new)?;
        // Multiclass arrays are 2D. Scalar/ranking arrays are 1D, handled below.
        if expected_shape.len() == 1 {
            let array = item.extract::<PyReadonlyArray1<'py, f32>>().map_err(|_| {
                anyhow::Error::new(PyValueError::new_err(format!(
                    "{kind} custom objective {what} must be a one-dimensional float32 NumPy array"
                )))
            })?;
            crate::ensure_input!(
                array.is_c_contiguous(),
                "{kind} custom objective {what} must be C-contiguous"
            );
            crate::ensure_input!(
                array.shape() == expected_shape,
                "{kind} custom objective {what} must have shape {:?}, got {:?}",
                expected_shape,
                array.shape()
            );
            let values = array.as_slice().map_err(anyhow::Error::new)?;
            for (pair, &value) in out.iter_mut().zip(values) {
                if hessian {
                    pair.hess = value
                } else {
                    pair.grad = value
                }
            }
            return Ok(());
        }
        let array = item.extract::<PyReadonlyArray2<'py, f32>>().map_err(|_| {
            anyhow::Error::new(PyValueError::new_err(format!(
                "{kind} custom objective {what} must be a two-dimensional float32 NumPy array"
            )))
        })?;
        crate::ensure_input!(
            array.is_c_contiguous(),
            "{kind} custom objective {what} must be C-contiguous"
        );
        crate::ensure_input!(
            array.shape() == expected_shape,
            "{kind} custom objective {what} must have shape {:?}, got {:?}",
            expected_shape,
            array.shape()
        );
        let values = array.as_slice().map_err(anyhow::Error::new)?;
        if class_major_output {
            let n = expected_shape[0];
            let k = expected_shape[1];
            for row in 0..n {
                for class in 0..k {
                    let pair = &mut out[class * n + row];
                    if hessian {
                        pair.hess = values[row * k + class]
                    } else {
                        pair.grad = values[row * k + class]
                    }
                }
            }
        } else {
            for (pair, &value) in out.iter_mut().zip(values) {
                if hessian {
                    pair.hess = value
                } else {
                    pair.grad = value
                }
            }
        }
        Ok(())
    };
    extract(0, "gradient", false, out)?;
    extract(1, "hessian", true, out)?;
    Ok(())
}
