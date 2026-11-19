//! Path-local feature eligibility for interaction constraints.
//!
//! Constraints are represented as overlapping groups of global feature IDs.
//! Each path keeps the groups that contain every feature split on that path,
//! stored as indices into the immutable normalized rules, plus a sorted
//! eligible-feature vector for fast membership checks and composition with
//! column sampling. A child retains only groups compatible with its complete
//! ancestor path, not merely groups containing its parent's split feature.

use crate::types::FeatId;
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum InteractionConstraintError {
    #[error("interaction constraint group {group_index} is empty")]
    EmptyGroup { group_index: usize },
    #[error("feature ID {feature} in interaction constraint group {group_index} is out of range for {n_features} features")]
    FeatureOutOfRange {
        group_index: usize,
        feature: FeatId,
        n_features: usize,
    },
}

/// Immutable, normalized interaction rules. `None` means interaction
/// constraints are disabled; `Some` contains sorted unique groups and includes
/// singleton groups for every otherwise-unlisted feature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InteractionConstraints {
    n_features: usize,
    groups: Option<Vec<Vec<FeatId>>>,
}

/// State for one node's root-to-node path. Child creation never mutates this
/// value, so sibling branches remain independent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InteractionState {
    eligible: Vec<FeatId>,
    active_groups: Vec<usize>,
}

impl InteractionConstraints {
    /// Normalize caller-provided groups. An absent or empty outer list disables
    /// constraints. Duplicate feature IDs and duplicate groups are removed.
    pub(crate) fn new(
        n_features: usize,
        groups: Option<Vec<Vec<FeatId>>>,
    ) -> Result<Self, InteractionConstraintError> {
        let Some(groups) = groups.filter(|groups| !groups.is_empty()) else {
            return Ok(Self {
                n_features,
                groups: None,
            });
        };

        let mut normalized = Vec::with_capacity(groups.len() + n_features);
        let mut listed = vec![false; n_features];
        for (group_index, group) in groups.into_iter().enumerate() {
            if group.is_empty() {
                return Err(InteractionConstraintError::EmptyGroup { group_index });
            }
            for &feature in &group {
                if feature as usize >= n_features {
                    return Err(InteractionConstraintError::FeatureOutOfRange {
                        group_index,
                        feature,
                        n_features,
                    });
                }
                listed[feature as usize] = true;
            }
            let unique: BTreeSet<_> = group.into_iter().collect();
            normalized.push(unique.into_iter().collect::<Vec<_>>());
        }
        normalized.sort_unstable();
        normalized.dedup();
        for (feature, was_listed) in listed.into_iter().enumerate() {
            if !was_listed {
                normalized.push(vec![feature as FeatId]);
            }
        }
        normalized.sort_unstable();

        Ok(Self {
            n_features,
            groups: Some(normalized),
        })
    }

    pub(crate) fn root(&self) -> InteractionState {
        let eligible = (0..self.n_features as FeatId).collect();
        let active_groups = self
            .groups
            .as_ref()
            .map(|groups| (0..groups.len()).collect())
            .unwrap_or_default();
        InteractionState {
            eligible,
            active_groups,
        }
    }

    /// Return the descendant state after splitting on `feature`, or `None` if
    /// the feature is outside the parent's eligible set.
    pub(crate) fn child_after_split(
        &self,
        parent: &InteractionState,
        feature: FeatId,
    ) -> Option<InteractionState> {
        if parent.eligible.binary_search(&feature).is_err() {
            return None;
        }
        let Some(groups) = &self.groups else {
            return Some(parent.clone());
        };

        let active_groups: Vec<_> = parent
            .active_groups
            .iter()
            .copied()
            .filter(|&index| groups[index].binary_search(&feature).is_ok())
            .collect();
        debug_assert!(!active_groups.is_empty());
        let eligible = active_groups
            .iter()
            .flat_map(|&index| groups[index].iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Some(InteractionState {
            eligible,
            active_groups,
        })
    }
}

impl InteractionState {
    pub(crate) fn eligible(&self) -> &[FeatId] {
        &self.eligible
    }

    pub(crate) fn allows(&self, feature: FeatId) -> bool {
        self.eligible.binary_search(&feature).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn rules(n: usize, groups: &[&[FeatId]]) -> InteractionConstraints {
        InteractionConstraints::new(n, Some(groups.iter().map(|group| group.to_vec()).collect()))
            .unwrap()
    }

    #[test]
    fn overlapping_groups_preserve_alternatives_then_narrow() {
        let constraints = rules(3, &[&[0, 1], &[1, 2]]);
        let root = constraints.root();
        assert_eq!(root.eligible(), &[0, 1, 2]);

        let after_one = constraints.child_after_split(&root, 1).unwrap();
        assert_eq!(after_one.eligible(), &[0, 1, 2]);
        let after_zero = constraints.child_after_split(&after_one, 0).unwrap();
        assert_eq!(after_zero.eligible(), &[0, 1]);
        assert!(!after_zero.allows(2));
        assert!(constraints.child_after_split(&after_zero, 2).is_none());
    }

    #[test]
    fn triangular_overlap_requires_one_group_to_cover_the_full_path() {
        let constraints = rules(3, &[&[0, 1], &[1, 2], &[0, 2]]);
        let root = constraints.root();
        let after_zero = constraints.child_after_split(&root, 0).unwrap();
        assert_eq!(after_zero.eligible(), &[0, 1, 2]);

        let after_zero_then_one = constraints.child_after_split(&after_zero, 1).unwrap();
        assert_eq!(after_zero_then_one.eligible(), &[0, 1]);
        assert!(!after_zero_then_one.allows(2));
        assert!(constraints
            .child_after_split(&after_zero_then_one, 2)
            .is_none());
    }

    #[test]
    fn recursive_paths_match_independent_set_oracle() {
        let constraints = rules(4, &[&[0, 1], &[1, 2]]);
        let original_groups = vec![
            BTreeSet::from([0, 1]),
            BTreeSet::from([1, 2]),
            BTreeSet::from([3]), // omitted feature is an implicit singleton
        ];
        fn oracle_eligible(original_groups: &[BTreeSet<FeatId>], path: &[FeatId]) -> Vec<FeatId> {
            if path.is_empty() {
                return original_groups
                    .iter()
                    .flat_map(|group| group.iter().copied())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
            }
            original_groups
                .iter()
                .filter(|group| path.iter().all(|feature| group.contains(feature)))
                .flat_map(|group| group.iter().copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        }
        fn visit_from_original_groups(
            constraints: &InteractionConstraints,
            state: &InteractionState,
            original_groups: &[BTreeSet<FeatId>],
            path: &mut Vec<FeatId>,
            depth: usize,
        ) {
            assert_eq!(state.eligible(), oracle_eligible(original_groups, path));
            if depth == 3 {
                return;
            }
            for feature in 0..constraints.n_features as FeatId {
                let actual = constraints.child_after_split(state, feature);
                let expected =
                    oracle_eligible(original_groups, &[path.as_slice(), &[feature]].concat());
                assert_eq!(actual.is_some(), !expected.is_empty());
                if let Some(actual) = actual {
                    path.push(feature);
                    visit_from_original_groups(
                        constraints,
                        &actual,
                        original_groups,
                        path,
                        depth + 1,
                    );
                    path.pop();
                }
            }
        }
        visit_from_original_groups(
            &constraints,
            &constraints.root(),
            &original_groups,
            &mut Vec::new(),
            0,
        );
    }

    #[test]
    fn disjoint_groups_and_repeated_feature_behavior() {
        let constraints = rules(3, &[&[0], &[1, 2]]);
        let root = constraints.root();
        let after_zero = constraints.child_after_split(&root, 0).unwrap();
        assert_eq!(after_zero.eligible(), &[0]);
        assert_eq!(
            constraints
                .child_after_split(&after_zero, 0)
                .unwrap()
                .eligible(),
            &[0]
        );
        assert!(constraints.child_after_split(&after_zero, 1).is_none());
    }

    #[test]
    fn absent_and_empty_outer_constraints_are_unrestricted() {
        for constraints in [
            InteractionConstraints::new(3, None).unwrap(),
            InteractionConstraints::new(3, Some(vec![])).unwrap(),
        ] {
            let root = constraints.root();
            let child = constraints.child_after_split(&root, 2).unwrap();
            assert_eq!(child.eligible(), &[0, 1, 2]);
        }
    }

    #[test]
    fn rejected_split_does_not_mutate_parent_or_sibling_state() {
        let constraints = rules(3, &[&[0, 1], &[1, 2]]);
        let root = constraints.root();
        let left = constraints.child_after_split(&root, 0).unwrap();
        let right = constraints.child_after_split(&root, 2).unwrap();
        assert_eq!(root.eligible(), &[0, 1, 2]);
        assert_eq!(left.eligible(), &[0, 1]);
        assert_eq!(right.eligible(), &[1, 2]);
        assert!(constraints.child_after_split(&left, 2).is_none());
        assert_eq!(right.eligible(), &[1, 2]);
    }

    #[test]
    fn normalization_sorts_deduplicates_and_adds_unlisted_singletons() {
        let normalized =
            InteractionConstraints::new(4, Some(vec![vec![2, 1, 2], vec![1, 2], vec![0, 1]]))
                .unwrap();
        assert_eq!(
            normalized.groups,
            Some(vec![vec![0, 1], vec![1, 2], vec![3]])
        );
    }

    #[test]
    fn rejects_empty_groups_and_out_of_range_ids() {
        assert_eq!(
            InteractionConstraints::new(2, Some(vec![vec![]])).unwrap_err(),
            InteractionConstraintError::EmptyGroup { group_index: 0 }
        );
        assert_eq!(
            InteractionConstraints::new(2, Some(vec![vec![2]])).unwrap_err(),
            InteractionConstraintError::FeatureOutOfRange {
                group_index: 0,
                feature: 2,
                n_features: 2,
            }
        );
    }
}
