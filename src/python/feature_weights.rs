//! Parsing and schema resolution for per-feature sampling weights.

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyDictMethods, PyList, PyString, PyTuple};

pub(super) enum FeatureWeightsSpec {
    Positional(Vec<f32>),
    Named(Vec<(String, f32)>),
}

pub(super) fn parse(raw: &Bound<'_, PyAny>) -> PyResult<Option<FeatureWeightsSpec>> {
    if raw.is_none() {
        return Ok(None);
    }

    let numpy = raw.py().import("numpy")?;
    let numpy_bool = numpy.getattr("bool_")?;
    if let Ok(dict) = raw.cast::<PyDict>() {
        let mut named = Vec::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let name = key.extract::<String>().map_err(|_| {
                PyValueError::new_err(
                    "feature_weights dictionary keys must be feature-name strings",
                )
            })?;
            let weight = parse_weight(&value, &numpy_bool, &format!("feature_weights[{name:?}]"))?;
            named.push((name, weight));
        }
        return Ok(Some(FeatureWeightsSpec::Named(named)));
    }

    if raw.is_instance_of::<PyString>() || raw.is_instance_of::<PyBytes>() {
        return Err(PyValueError::new_err(
            "feature_weights must be a mapping, list, tuple, or one-dimensional NumPy array",
        ));
    }

    if !raw.is_instance_of::<PyList>() && !raw.is_instance_of::<PyTuple>() {
        let ndarray = numpy.getattr("ndarray")?;
        if !raw.is_instance(&ndarray)? {
            return Err(PyValueError::new_err(
                "feature_weights must be a mapping, list, tuple, or one-dimensional NumPy array",
            ));
        }
        let ndim = raw.getattr("ndim")?.extract::<usize>()?;
        if ndim != 1 {
            return Err(PyValueError::new_err(
                "feature_weights NumPy array must be one-dimensional",
            ));
        }
    }
    // Convert one item at a time: do not retain a second vector of Python
    // objects alongside the native weights.
    let mut positional = Vec::with_capacity(raw.len()?);
    for (index, value) in raw.try_iter()?.enumerate() {
        positional.push(parse_weight(
            &value?,
            &numpy_bool,
            &format!("feature_weights[{index}]"),
        )?);
    }
    if positional.is_empty() {
        return Err(PyValueError::new_err(
            "feature_weights positional vector must not be empty",
        ));
    }
    Ok(Some(FeatureWeightsSpec::Positional(positional)))
}

impl FeatureWeightsSpec {
    pub(super) fn resolve(
        &self,
        n_features: usize,
        feature_names: &[String],
    ) -> PyResult<Vec<f32>> {
        let weights = match self {
            Self::Positional(weights) => {
                if weights.len() != n_features {
                    return Err(PyValueError::new_err(format!(
                        "feature_weights has {} entries but there are {n_features} features",
                        weights.len()
                    )));
                }
                weights.clone()
            }
            Self::Named(named) => {
                if feature_names.len() != n_features || feature_names.is_empty() {
                    return Err(PyValueError::new_err(
                        "feature_weights dictionary requires a complete nonempty feature-name schema",
                    ));
                }
                let mut seen = HashSet::with_capacity(feature_names.len());
                for name in feature_names {
                    if !seen.insert(name.as_str()) {
                        return Err(PyValueError::new_err(format!(
                            "feature-name schema contains duplicate name {name:?}"
                        )));
                    }
                }
                let by_name: std::collections::HashMap<&str, f32> = named
                    .iter()
                    .map(|(name, weight)| (name.as_str(), *weight))
                    .collect();
                if let Some((name, _)) =
                    named.iter().find(|(name, _)| !seen.contains(name.as_str()))
                {
                    return Err(PyValueError::new_err(format!(
                        "feature_weights contains unknown feature name {name:?}"
                    )));
                }
                feature_names
                    .iter()
                    .map(|name| by_name.get(name.as_str()).copied().unwrap_or(1.0))
                    .collect()
            }
        };

        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(PyValueError::new_err(
                "feature_weights values must be finite and non-negative float32 values",
            ));
        }
        if !weights.iter().any(|weight| *weight > 0.0) {
            return Err(PyValueError::new_err(
                "feature_weights must contain at least one positive value",
            ));
        }
        Ok(weights)
    }
}

fn parse_weight(
    value: &Bound<'_, PyAny>,
    numpy_bool: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<f32> {
    if value.is_instance_of::<PyBool>() || value.is_instance(numpy_bool)? {
        return Err(PyValueError::new_err(format!(
            "{what} must be a numeric, non-boolean value"
        )));
    }
    let weight = value.extract::<f64>().map_err(|_| {
        PyValueError::new_err(format!("{what} must be a numeric, non-boolean value"))
    })?;
    if !weight.is_finite() || weight < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{what} must be finite and non-negative"
        )));
    }
    let weight = weight as f32;
    if !weight.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{what} is outside the finite float32 range"
        )));
    }
    Ok(weight)
}
