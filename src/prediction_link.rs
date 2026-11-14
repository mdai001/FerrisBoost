//! Optional prediction transforms for custom objectives.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// A transform applied to raw margins when a custom objective model is scored.
/// Named links store an identifier only; the executable provider is supplied by
/// the caller and is not serialized with a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionLink {
    Identity,
    Sigmoid,
    Exp,
    Softmax,
    Named(String),
}

impl PredictionLink {
    const NAMED_PREFIX: &'static str = "named:";

    /// Validate and encode the stable JSON attribute representation.
    pub(crate) fn attribute_value(&self, n_outputs: usize) -> Result<String> {
        self.validate(n_outputs)?;
        Ok(match self {
            Self::Identity => "identity".to_owned(),
            Self::Sigmoid => "sigmoid".to_owned(),
            Self::Exp => "exp".to_owned(),
            Self::Softmax => "softmax".to_owned(),
            Self::Named(name) => format!("{}{name}", Self::NAMED_PREFIX),
        })
    }

    /// Parse the stable JSON attribute representation.
    pub(crate) fn from_attribute(value: &str, n_outputs: usize) -> Result<Self> {
        let link = match value {
            "identity" => Self::Identity,
            "sigmoid" => Self::Sigmoid,
            "exp" => Self::Exp,
            "softmax" => Self::Softmax,
            value if value.starts_with(Self::NAMED_PREFIX) => {
                Self::Named(value[Self::NAMED_PREFIX.len()..].to_owned())
            }
            _ => crate::bail_input!("unsupported ferrisboost prediction link {value:?}"),
        };
        link.validate(n_outputs)?;
        Ok(link)
    }

    /// Check link-specific constraints for the model output width.
    pub fn validate(&self, n_outputs: usize) -> Result<()> {
        crate::ensure_input!(
            n_outputs > 0,
            "prediction link requires at least one output"
        );
        if let Self::Softmax = self {
            crate::ensure_input!(
                n_outputs >= 2,
                "softmax prediction link requires at least two outputs"
            );
        }
        if let Self::Named(name) = self {
            crate::ensure_input!(
                !name.trim().is_empty(),
                "named prediction link name must not be empty"
            );
        }
        Ok(())
    }

    /// Apply this link to a row-major `n_rows × n_outputs` margin array.
    /// The output shape is unchanged. Named links require an executable
    /// provider, so this method returns an explicit error for them.
    pub fn apply_rows(&self, values: &mut [f32], n_outputs: usize) -> Result<()> {
        self.validate(n_outputs)?;
        crate::ensure_input!(
            values.len() % n_outputs == 0,
            "prediction link input length {} is not divisible by n_outputs {n_outputs}",
            values.len()
        );
        match self {
            Self::Identity => {}
            Self::Sigmoid => values
                .iter_mut()
                .for_each(|value| *value = crate::types::Objective::Logistic.transform(*value)),
            Self::Exp => values
                .iter_mut()
                .for_each(|value| *value = (*value).exp()),
            Self::Softmax => values
                .chunks_exact_mut(n_outputs)
                .for_each(crate::objective::softmax_in_place),
            Self::Named(name) => crate::bail_input!(
                "named prediction link {name:?} has no installed native provider; use raw margins or a provider-aware prediction API"
            ),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::PredictionLink;

    #[test]
    fn built_in_links_transform_rows_in_place() {
        let mut identity = [-2.0, 3.0];
        PredictionLink::Identity
            .apply_rows(&mut identity, 1)
            .unwrap();
        assert_eq!(identity, [-2.0, 3.0]);

        let mut sigmoid = [0.0, 1.0];
        PredictionLink::Sigmoid.apply_rows(&mut sigmoid, 1).unwrap();
        assert_eq!(sigmoid[0].to_bits(), 0.5f32.to_bits());
        assert_eq!(
            sigmoid[1].to_bits(),
            crate::types::Objective::Logistic.transform(1.0).to_bits()
        );

        let mut exp = [0.0, 1.0];
        PredictionLink::Exp.apply_rows(&mut exp, 1).unwrap();
        assert_eq!(exp[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(exp[1].to_bits(), 1.0f32.exp().to_bits());

        let mut softmax = [0.0, 1.0, 1.0, 0.0];
        PredictionLink::Softmax.apply_rows(&mut softmax, 2).unwrap();
        assert_eq!(softmax.len(), 4);
        assert!((softmax[0] + softmax[1] - 1.0).abs() < 1e-6);
        assert!((softmax[2] + softmax[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn shape_and_named_provider_errors_are_explicit() {
        assert!(PredictionLink::Softmax.validate(1).is_err());
        assert!(PredictionLink::Identity.apply_rows(&mut [0.0], 0).is_err());
        assert!(PredictionLink::Identity
            .apply_rows(&mut [0.0, 1.0, 2.0], 2)
            .is_err());
        assert!(PredictionLink::Named("custom".into())
            .apply_rows(&mut [0.0], 1)
            .unwrap_err()
            .to_string()
            .contains("no installed native provider"));
        assert!(PredictionLink::Named("  ".into()).validate(1).is_err());
    }

    #[test]
    fn attribute_encoding_distinguishes_named_links() {
        let named = PredictionLink::Named("exp".into());
        assert_eq!(named.attribute_value(1).unwrap(), "named:exp");
        assert_eq!(
            PredictionLink::from_attribute("named:exp", 1).unwrap(),
            named
        );
        assert!(PredictionLink::from_attribute("unknown", 1).is_err());
    }
}
