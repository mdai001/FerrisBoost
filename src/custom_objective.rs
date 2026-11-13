//! Batched, training-only custom derivatives for scalar regression.
//!
//! The caller fills unweighted derivatives. Training validates them, applies
//! row weights once, then uses the existing Newton split and leaf rules.
//! Implementations are not serialized; models retain only their stable name.

use crate::types::GradPair;

pub trait CustomObjective: Send + Sync {
    /// Stable nonempty identifier, also checked when continuing a saved model.
    fn name(&self) -> &str;

    /// Fill every output with unweighted gradient and nonnegative Hessian.
    /// Inputs contain all training rows and current raw margins, before sampling.
    fn gradients(
        &self,
        raw_margin: &[f32],
        labels: &[f32],
        out: &mut [GradPair],
    ) -> anyhow::Result<()>;
}
