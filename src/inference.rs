//! Prediction-only feature compaction.
//!
//! Canonical feature ids belong to the training/serialized model. This module
//! clones the model and remaps only that private inference copy, so save/load
//! and XGBoost interchange can never observe compact ids.

use anyhow::{bail, Context, Result};

use crate::tree::Model;

const CPU_PREDICTION_BLOCK_ROWS: usize = 64;

/// Prediction-only CPU tree. Keeping independently accessed fields in separate
/// arrays avoids pulling training-only gain/Hessian data through L1 while
/// traversing a tree.
struct CpuInferenceTree {
    features: Box<[u32]>,
    split_conditions: Box<[f32]>,
    default_left: Box<[u8]>,
    left_children: Box<[u32]>,
    right_children: Box<[u32]>,
    flags: Box<[u8]>,
    leaf_values: Box<[f32]>,
    max_depth: usize,
}

impl CpuInferenceTree {
    fn new(model: &Model, tree: &crate::tree::Tree) -> Result<Self> {
        if tree.nodes.is_empty() {
            bail!("prediction tree must contain at least one node");
        }

        for node in &tree.nodes {
            if !node.is_leaf {
                if node.feat as usize >= model.n_features {
                    bail!(
                        "CPU prediction model contains out-of-range feature {}",
                        node.feat
                    );
                }
                if node.left < 0
                    || node.right < 0
                    || node.left as usize >= tree.nodes.len()
                    || node.right as usize >= tree.nodes.len()
                {
                    bail!("CPU prediction model contains out-of-range child index");
                }
            }
        }

        let mut max_depth = 0usize;
        let mut visited = 0usize;
        let mut stack = vec![(0usize, 0usize)];
        while let Some((node_id, depth)) = stack.pop() {
            visited += 1;
            if visited > tree.nodes.len() {
                bail!("CPU prediction model contains a child cycle");
            }
            max_depth = max_depth.max(depth);
            let node = &tree.nodes[node_id];
            if !node.is_leaf {
                stack.push((node.left as usize, depth + 1));
                stack.push((node.right as usize, depth + 1));
            }
        }

        // The common depth <= 6 case gets a complete heap layout. At a given
        // level, left/right are arithmetic (2*i / 2*i+1), so prediction has no
        // leaf test and no child-pointer loads. An early leaf is extended with
        // NaN comparisons that deterministically walk right to the same value.
        if max_depth <= 6 {
            let internal_count = (1usize << max_depth) - 1;
            let mut features = vec![0u32; internal_count];
            let mut split_conditions = vec![f32::NAN; internal_count];
            let mut default_left = vec![0u8; internal_count];
            let mut leaf_values = vec![0.0f32; 1usize << max_depth];
            let mut stack = vec![(0usize, 0usize, 0usize)];
            while let Some((node_id, depth, level_index)) = stack.pop() {
                let node = &tree.nodes[node_id];
                if depth == max_depth {
                    debug_assert!(node.is_leaf);
                    leaf_values[level_index] = node.leaf_value;
                    continue;
                }
                let heap_index = (1usize << depth) - 1 + level_index;
                if node.is_leaf {
                    stack.push((node_id, depth + 1, 2 * level_index + 1));
                } else {
                    features[heap_index] = node.feat;
                    split_conditions[heap_index] = node.split_cond;
                    default_left[heap_index] = u8::from(node.default_left);
                    stack.push((node.left as usize, depth + 1, 2 * level_index));
                    stack.push((node.right as usize, depth + 1, 2 * level_index + 1));
                }
            }
            return Ok(Self {
                features: features.into_boxed_slice(),
                split_conditions: split_conditions.into_boxed_slice(),
                default_left: default_left.into_boxed_slice(),
                left_children: Box::new([]),
                right_children: Box::new([]),
                flags: Box::new([]),
                leaf_values: leaf_values.into_boxed_slice(),
                max_depth,
            });
        }

        // Rare deep trees retain the compact dynamic traversal. Expanding a
        // sparse depth-14 tree into a complete heap would waste substantial
        // memory for no guaranteed prediction benefit.
        let mut features = Vec::with_capacity(tree.nodes.len());
        let mut split_conditions = Vec::with_capacity(tree.nodes.len());
        let mut left_children = Vec::with_capacity(tree.nodes.len());
        let mut right_children = Vec::with_capacity(tree.nodes.len());
        let mut flags = Vec::with_capacity(tree.nodes.len());
        let mut leaf_values = Vec::with_capacity(tree.nodes.len());
        for node in &tree.nodes {
            features.push(node.feat);
            split_conditions.push(node.split_cond);
            left_children.push(if node.is_leaf { 0 } else { node.left as u32 });
            right_children.push(if node.is_leaf { 0 } else { node.right as u32 });
            flags.push(u8::from(node.is_leaf) | (u8::from(node.default_left) << 1));
            leaf_values.push(node.leaf_value);
        }
        Ok(Self {
            features: features.into_boxed_slice(),
            split_conditions: split_conditions.into_boxed_slice(),
            default_left: Box::new([]),
            left_children: left_children.into_boxed_slice(),
            right_children: right_children.into_boxed_slice(),
            flags: flags.into_boxed_slice(),
            leaf_values: leaf_values.into_boxed_slice(),
            max_depth,
        })
    }

    #[inline]
    fn add_array_block<const ANY_MISSING: bool, const DEPTH: usize>(
        &self,
        row_major: &[f32],
        n_features: usize,
        margins: &mut [f32],
    ) {
        let mut node_ids = [0u32; CPU_PREDICTION_BLOCK_ROWS];
        for depth in 0..DEPTH {
            let first_node = (1usize << depth) - 1;
            for (row, node_id) in node_ids[..margins.len()].iter_mut().enumerate() {
                let node = first_node + *node_id as usize;
                let value = row_major[row * n_features + self.features[node] as usize];
                let go_left = if ANY_MISSING && value.is_nan() {
                    self.default_left[node] != 0
                } else {
                    value < self.split_conditions[node]
                };
                *node_id = 2 * *node_id + u32::from(!go_left);
            }
        }
        for (margin, &node_id) in margins.iter_mut().zip(&node_ids) {
            *margin += self.leaf_values[node_id as usize];
        }
    }

    #[inline]
    fn add_dynamic_block<const ANY_MISSING: bool>(
        &self,
        row_major: &[f32],
        n_features: usize,
        margins: &mut [f32],
    ) {
        debug_assert!(margins.len() <= CPU_PREDICTION_BLOCK_ROWS);
        let mut node_ids = [0u32; CPU_PREDICTION_BLOCK_ROWS];
        for _ in 0..self.max_depth {
            for (row, node_id) in node_ids[..margins.len()].iter_mut().enumerate() {
                let node = *node_id as usize;
                let flags = self.flags[node];
                if flags & 1 != 0 {
                    continue;
                }
                let value = row_major[row * n_features + self.features[node] as usize];
                let go_left = if ANY_MISSING && value.is_nan() {
                    flags & 2 != 0
                } else {
                    value < self.split_conditions[node]
                };
                *node_id = if go_left {
                    self.left_children[node]
                } else {
                    self.right_children[node]
                };
            }
        }
        for (margin, &node_id) in margins.iter_mut().zip(&node_ids) {
            *margin += self.leaf_values[node_id as usize];
        }
    }

    #[inline]
    fn add_block<const ANY_MISSING: bool>(
        &self,
        row_major: &[f32],
        n_features: usize,
        margins: &mut [f32],
    ) {
        debug_assert!(margins.len() <= CPU_PREDICTION_BLOCK_ROWS);
        match self.max_depth {
            0 => self.add_array_block::<ANY_MISSING, 0>(row_major, n_features, margins),
            1 => self.add_array_block::<ANY_MISSING, 1>(row_major, n_features, margins),
            2 => self.add_array_block::<ANY_MISSING, 2>(row_major, n_features, margins),
            3 => self.add_array_block::<ANY_MISSING, 3>(row_major, n_features, margins),
            4 => self.add_array_block::<ANY_MISSING, 4>(row_major, n_features, margins),
            5 => self.add_array_block::<ANY_MISSING, 5>(row_major, n_features, margins),
            6 => self.add_array_block::<ANY_MISSING, 6>(row_major, n_features, margins),
            _ => self.add_dynamic_block::<ANY_MISSING>(row_major, n_features, margins),
        }
    }
}

/// Cached CPU inference representation. Rows are processed in small blocks so
/// each tree's split arrays remain hot while it scores multiple rows. Tree
/// accumulation order for every row is unchanged.
pub(crate) struct CpuInferenceModel {
    trees: Box<[CpuInferenceTree]>,
    base_margin: f32,
    n_features: usize,
    needs_missing_check: bool,
}

impl CpuInferenceModel {
    pub(crate) fn new(model: &Model) -> Result<Self> {
        if model.n_features == 0 {
            bail!("prediction model must contain at least one feature");
        }
        let trees = model
            .trees
            .iter()
            .map(|tree| CpuInferenceTree::new(model, tree))
            .collect::<Result<Vec<_>>>()?;
        let needs_missing_check = model
            .trees
            .iter()
            .flat_map(|tree| &tree.nodes)
            .any(|node| !node.is_leaf && node.default_left);
        Ok(Self {
            trees: trees.into_boxed_slice(),
            base_margin: crate::train::base_margin(model.objective, model.base_score),
            n_features: model.n_features,
            needs_missing_check,
        })
    }

    pub(crate) fn predict_margins(
        &self,
        row_major: &[f32],
        n_features: usize,
        n_trees: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if n_features != self.n_features {
            bail!(
                "CPU prediction input has {n_features} features, expected {}",
                self.n_features
            );
        }
        let expected = out
            .len()
            .checked_mul(n_features)
            .context("CPU prediction input size overflow")?;
        if row_major.len() != expected {
            bail!(
                "CPU prediction input contains {} values, expected {} rows x {n_features} = {expected}",
                row_major.len(),
                out.len()
            );
        }
        out.fill(self.base_margin);
        let upto = n_trees.min(self.trees.len());
        for (rows, margins) in row_major
            .chunks(CPU_PREDICTION_BLOCK_ROWS * n_features)
            .zip(out.chunks_mut(CPU_PREDICTION_BLOCK_ROWS))
        {
            let any_missing = self.needs_missing_check && rows.iter().any(|value| value.is_nan());
            for tree in &self.trees[..upto] {
                if any_missing {
                    tree.add_block::<true>(rows, n_features, margins);
                } else {
                    tree.add_block::<false>(rows, n_features, margins);
                }
            }
        }
        Ok(())
    }
}

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

    #[test]
    fn blocked_cpu_prediction_matches_row_wise_traversal_bit_exactly() {
        let mut shallow = Tree::default();
        let shallow_root = shallow.push_leaf(0.0, 1.0);
        shallow.split_leaf(shallow_root, 3, 1.5, true, 0.0, -0.25, 1.0, 0.75, 1.0);

        let mut uneven = Tree::default();
        let uneven_root = uneven.push_leaf(0.0, 1.0);
        let (_, right) = uneven.split_leaf(uneven_root, 1, -0.5, false, 0.0, 0.125, 1.0, -0.5, 1.0);
        uneven.split_leaf(right, 4, 2.0, true, 0.0, 0.625, 1.0, -0.875, 1.0);

        // Depth 7 intentionally exercises the non-expanded fallback used for
        // trees deeper than the cache-friendly heap layout.
        let mut deep = Tree::default();
        let mut next = deep.push_leaf(0.0, 1.0);
        for depth in 0..7 {
            let (_, right) = deep.split_leaf(
                next,
                0,
                -10.0,
                false,
                0.0,
                depth as f32 * 0.03125,
                1.0,
                0.0,
                1.0,
            );
            next = right;
        }

        let model = Model {
            trees: vec![shallow, uneven, deep],
            base_score: 0.3,
            objective: Objective::Logistic,
            n_features: 5,
            best_iteration: None,
            best_score: None,
            feature_names: (0..5).map(|i| format!("f{i}")).collect(),
            schema_mode: Some(crate::tree::SchemaMode::Named),
        };
        let predictor = CpuInferenceModel::new(&model).unwrap();
        let patterns = [
            [0.0, -1.0, 0.0, 1.0, 3.0],
            [0.0, 1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 0.0, f32::NAN, f32::NAN],
            [0.0, f32::NAN, 0.0, 1.5, 4.0],
            [-20.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let rows: Vec<[f32; 5]> = (0..130).map(|row| patterns[row % patterns.len()]).collect();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();

        for n_trees in 0..=model.trees.len() + 1 {
            let mut actual = vec![f32::NAN; rows.len()];
            predictor
                .predict_margins(&flat, model.n_features, n_trees, &mut actual)
                .unwrap();
            for (row, &actual) in rows.iter().zip(&actual) {
                assert_eq!(
                    actual.to_bits(),
                    model.predict_margin_upto(row, n_trees).to_bits()
                );
            }
        }
    }
}
