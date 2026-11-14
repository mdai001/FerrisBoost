//! Batched, training-only custom derivatives.
//!
//! The caller fills unweighted derivatives. Training validates them, applies
//! row weights once, then uses the existing Newton split and leaf rules.
//! Implementations are not serialized; models retain only their stable name.

use crate::types::GradPair;

/// Specifies the objective metadata and gradient batch contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CustomObjectiveKind {
    #[default]
    Scalar,
    Multiclass,
    Ranking,
}

impl CustomObjectiveKind {
    pub(crate) fn for_objective(objective: crate::types::Objective) -> Option<Self> {
        match objective {
            crate::types::Objective::SquaredError => Some(Self::Scalar),
            crate::types::Objective::Softprob { .. } | crate::types::Objective::Softmax { .. } => {
                Some(Self::Multiclass)
            }
            crate::types::Objective::Rank { .. } => Some(Self::Ranking),
            _ => None,
        }
    }
}

pub trait CustomObjective: Send + Sync {
    fn kind(&self) -> CustomObjectiveKind {
        CustomObjectiveKind::Scalar
    }

    /// Stable nonempty identifier, also checked when continuing a saved model.
    fn name(&self) -> &str;

    /// Fill every output with unweighted gradient and nonnegative Hessian.
    /// Inputs contain all training rows and current raw margins, before sampling.
    fn gradients(
        &self,
        raw_margin: &[f32],
        labels: &[f32],
        out: &mut [GradPair],
    ) -> anyhow::Result<()> {
        let _ = (raw_margin, labels, out);
        crate::bail_input!(
            "custom objective {:?} does not supply scalar derivatives",
            self.name()
        )
    }

    /// Fill class-major diagonal derivatives. A complete round's raw margins
    /// are provided before updating any class tree.
    fn gradients_multiclass(
        &self,
        _raw_margin: &[f32],
        _labels: &[f32],
        _n_outputs: usize,
        _out: &mut [GradPair],
    ) -> anyhow::Result<()> {
        crate::bail_input!(
            "custom objective {:?} does not supply multiclass derivatives",
            self.name()
        )
    }

    /// Fill unweighted row derivatives for the supplied query offsets.
    /// Training applies each query's weight to all rows in that query.
    fn gradients_ranking(
        &self,
        _raw_margin: &[f32],
        _labels: &[f32],
        _group_ptr: &[u32],
        _out: &mut [GradPair],
    ) -> anyhow::Result<()> {
        crate::bail_input!(
            "custom objective {:?} does not supply ranking derivatives",
            self.name()
        )
    }
}
