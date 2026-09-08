//! Prediction-only feature compaction.
//!
//! Canonical feature ids belong to the training/serialized model. This module
//! clones the model and remaps only that private inference copy, so save/load
//! and XGBoost interchange can never observe compact ids.

use anyhow::{bail, Context, Result};

use crate::tree::Model;

pub(crate) struct CompactInferenceModel {
    model: Model,
    canonical_features: Vec<usize>,
}

impl CompactInferenceModel {
    pub(crate) fn new(canonical: &Model) -> Result<Self> {
        if canonical.n_features == 0 {
            bail!("prediction 模型必须至少有一个 feature");
        }

        let mut canonical_features = canonical
            .trees
            .iter()
            .flat_map(|tree| tree.nodes.iter())
            .filter(|node| !node.is_leaf)
            .map(|node| node.feat as usize)
            .collect::<Vec<_>>();
        canonical_features.sort_unstable();
        canonical_features.dedup();
        for &feature in &canonical_features {
            if feature >= canonical.n_features {
                bail!("prediction 模型含越界 feature {feature}");
            }
        }

        // Readers need one projected column to discover row count even when
        // the ensemble is empty or contains root-only trees.
        if canonical_features.is_empty() {
            canonical_features.push(0);
        }

        let mut canonical_to_compact = vec![usize::MAX; canonical.n_features];
        for (compact, &feature) in canonical_features.iter().enumerate() {
            canonical_to_compact[feature] = compact;
        }

        let mut model = canonical.clone();
        for tree in &mut model.trees {
            for node in &mut tree.nodes {
                if node.is_leaf {
                    continue;
                }
                let compact = canonical_to_compact[node.feat as usize];
                node.feat = u32::try_from(compact).context("compact feature id 超过 u32")?;
            }
        }
        model.n_features = canonical_features.len();
        model.feature_names = canonical_features
            .iter()
            .filter_map(|&feature| canonical.feature_names.get(feature).cloned())
            .collect();

        Ok(Self {
            model,
            canonical_features,
        })
    }

    pub(crate) fn model(&self) -> &Model {
        &self.model
    }

    pub(crate) fn canonical_features(&self) -> &[usize] {
        &self.canonical_features
    }

    pub(crate) fn n_features(&self) -> usize {
        self.canonical_features.len()
    }

    pub(crate) fn is_identity(&self, canonical_width: usize) -> bool {
        self.canonical_features.len() == canonical_width
            && self
                .canonical_features
                .iter()
                .enumerate()
                .all(|(index, &feature)| index == feature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Node, Tree};
    use crate::types::Objective;

    #[test]
    fn remapping_is_private_and_prediction_is_bit_exact() {
        let mut root = Node::leaf(0.0, 1.0);
        root.is_leaf = false;
        root.feat = 3;
        root.split_cond = 1.5;
        root.left = 1;
        root.right = 2;
        root.default_left = true;
        let canonical = Model {
            trees: vec![Tree {
                nodes: vec![root, Node::leaf(-0.25, 1.0), Node::leaf(0.75, 1.0)],
            }],
            base_score: 0.5,
            objective: Objective::Logistic,
            n_features: 5,
            best_iteration: None,
            best_score: None,
            feature_names: (0..5).map(|i| format!("f{i}")).collect(),
            schema_mode: Some(crate::tree::SchemaMode::Named),
        };
        let before = canonical.to_xgboost_json().unwrap();
        let compact = CompactInferenceModel::new(&canonical).unwrap();

        assert_eq!(compact.canonical_features(), &[3]);
        assert_eq!(compact.model().trees[0].nodes[0].feat, 0);
        assert_eq!(canonical.trees[0].nodes[0].feat, 3);
        assert_eq!(canonical.to_xgboost_json().unwrap(), before);
        for row in [
            vec![0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, f32::NAN, 0.0],
        ] {
            let projected = [row[3]];
            assert_eq!(
                canonical.predict_margin(&row).to_bits(),
                compact.model().predict_margin(&projected).to_bits()
            );
        }
    }
}
