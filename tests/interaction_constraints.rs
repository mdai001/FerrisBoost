//! Public training coverage for path-based interaction constraints.

#[cfg(feature = "cuda")]
use ferrisboost::train::train_gpu;
#[cfg(feature = "cuda")]
use ferrisboost::types::GpuMath;
use ferrisboost::{
    comm::Local,
    source::DenseSource,
    train::{train, TrainConfig},
    types::{Objective, TrainParams},
    Model,
};

const FEATURES: usize = 3;
// All eight binary combinations, in row-major order.
const DATA: [f32; 24] = [
    0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1.,
];
const LABELS: [f32; 8] = [0., 1., 2., 3., 4., 5., 6., 7.];
const UNLISTED_FEATURE_LABELS: [f32; 8] = [0., 1., 0., 1., 0., 1., 0., 1.];

fn params() -> TrainParams {
    TrainParams {
        n_rounds: 1,
        max_depth: 3,
        max_bin: 8,
        learning_rate: 0.5,
        lambda: 0.0,
        min_child_weight: 0.0,
        nthread: 1,
        cols_per_block: 1,
        ..Default::default()
    }
}

fn fit(labels: &[f32], params: TrainParams) -> Model {
    let source = DenseSource::from_row_major(
        &DATA,
        labels.len(),
        FEATURES,
        params.max_bin,
        params.cols_per_block,
    );
    let config = TrainConfig::new(params, Objective::SquaredError);
    train(&source, labels, &[], &config, &mut [], &Local).unwrap()
}

fn model_bytes(model: &Model) -> Vec<u8> {
    model.to_xgboost_json().unwrap().into_bytes()
}

/// Verify each internal node against the complete root-to-node feature path.
/// Unlisted features are represented by implicit singleton groups, matching
/// the public contract exercised by these tests.
fn assert_paths_fit_groups(model: &Model, groups: &[&[u32]]) {
    let mut normalized: Vec<Vec<u32>> = groups.iter().map(|group| group.to_vec()).collect();
    for feature in 0..FEATURES as u32 {
        if !normalized.iter().any(|group| group.contains(&feature)) {
            normalized.push(vec![feature]);
        }
    }
    for tree in &model.trees {
        fn visit(
            tree: &ferrisboost::tree::Tree,
            node_id: usize,
            path: &mut Vec<u32>,
            groups: &[Vec<u32>],
        ) {
            let node = &tree.nodes[node_id];
            if node.is_leaf {
                return;
            }
            path.push(node.feat);
            assert!(
                groups
                    .iter()
                    .any(|group| path.iter().all(|feature| group.contains(feature))),
                "no constraint group covers split path {path:?}"
            );
            visit(tree, node.left as usize, path, groups);
            visit(tree, node.right as usize, path, groups);
            path.pop();
        }
        if !tree.nodes.is_empty() {
            visit(tree, 0, &mut Vec::new(), &normalized);
        }
    }
}

fn split_features(model: &Model) -> Vec<u32> {
    let mut features: Vec<_> = model
        .trees
        .iter()
        .flat_map(|tree| tree.nodes.iter())
        .filter(|node| !node.is_leaf)
        .map(|node| node.feat)
        .collect();
    features.sort_unstable();
    features.dedup();
    features
}

fn leaf_count(tree: &ferrisboost::tree::Tree) -> usize {
    tree.nodes.iter().filter(|node| node.is_leaf).count()
}

#[test]
fn disjoint_groups_allow_useful_splits() {
    let mut constrained = params();
    constrained.interaction_constraints = vec![vec![0, 1], vec![2]];
    let model = fit(&LABELS, constrained);
    assert_paths_fit_groups(&model, &[&[0, 1], &[2]]);
    assert!(
        !split_features(&model).is_empty(),
        "constraint must not stop all splits"
    );
}

#[test]
fn unlisted_feature_is_a_useful_singleton_group() {
    let mut constrained = params();
    constrained.interaction_constraints = vec![vec![0, 1]];
    let model = fit(&UNLISTED_FEATURE_LABELS, constrained);
    assert_paths_fit_groups(&model, &[&[0, 1]]);
    assert!(
        !split_features(&model).is_empty(),
        "constraint must not stop all splits"
    );
    assert!(
        split_features(&model).contains(&2),
        "the useful unlisted feature should be allowed as a singleton interaction"
    );
}

#[test]
fn triangular_overlaps_validate_complete_paths_and_still_split() {
    let groups: [&[u32]; 3] = [&[0, 1], &[1, 2], &[0, 2]];
    let mut constrained = params();
    constrained.interaction_constraints = groups.iter().map(|group| group.to_vec()).collect();
    let model = fit(&LABELS, constrained);
    assert_paths_fit_groups(&model, &groups);
    assert!(
        !split_features(&model).is_empty(),
        "constraint must not stop all splits"
    );
}

#[test]
fn disabled_and_full_group_constraints_preserve_default_model_bytes() {
    let baseline = fit(&LABELS, params());

    let mut explicitly_disabled = params();
    explicitly_disabled.interaction_constraints = vec![];
    let disabled = fit(&LABELS, explicitly_disabled);
    assert_eq!(model_bytes(&disabled), model_bytes(&baseline));

    let mut one_full_group = params();
    one_full_group.interaction_constraints = vec![vec![0, 1, 2]];
    let full = fit(&LABELS, one_full_group);
    assert_eq!(model_bytes(&full), model_bytes(&baseline));
}

#[test]
fn invalid_groups_fail_before_training() {
    for (groups, expected) in [
        (vec![vec![]], "empty"),
        (vec![vec![FEATURES as u32]], "out of range"),
    ] {
        let source = DenseSource::from_row_major(&DATA, LABELS.len(), FEATURES, 8, 1);
        let mut invalid = params();
        invalid.interaction_constraints = groups;
        let config = TrainConfig::new(invalid, Objective::SquaredError);
        let error = train(&source, &LABELS, &[], &config, &mut [], &Local).unwrap_err();
        assert!(
            error.to_string().to_lowercase().contains(expected),
            "{error}"
        );
    }
}

#[test]
fn nested_colsample_is_block_width_independent_with_constraints() {
    let groups: [&[u32]; 2] = [&[0, 1], &[1, 2]];
    let mut narrow = params();
    narrow.interaction_constraints = groups.iter().map(|group| group.to_vec()).collect();
    narrow.colsample_bytree = 0.8;
    narrow.colsample_bylevel = 0.75;
    narrow.colsample_bynode = 0.75;
    narrow.seed = 41;

    let mut wide = narrow.clone();
    wide.cols_per_block = FEATURES;
    let narrow_model = fit(&LABELS, narrow);
    let wide_model = fit(&LABELS, wide);
    assert_eq!(model_bytes(&narrow_model), model_bytes(&wide_model));
    assert_paths_fit_groups(&narrow_model, &groups);
}

#[test]
fn max_leaves_cap_holds_with_interaction_constraints() {
    let groups: [&[u32]; 2] = [&[0, 1], &[1, 2]];
    let mut constrained = params();
    constrained.interaction_constraints = groups.iter().map(|group| group.to_vec()).collect();
    constrained.max_leaves = 3;
    let model = fit(&LABELS, constrained);
    assert_paths_fit_groups(&model, &groups);
    assert!(!split_features(&model).is_empty());
    for tree in &model.trees {
        assert!(leaf_count(tree) <= 3);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn constrained_gpu_exact_and_fast_preserve_paths_across_residency_and_cache() {
    let groups: [&[u32]; 2] = [&[0, 1], &[1, 2]];
    let bytes = |model: &Model| serde_json::to_vec(model).unwrap();
    let run_params = |gpu_math, resident_blocks, gpu_split_cache| TrainParams {
        n_rounds: 3,
        max_depth: 3,
        max_leaves: 3,
        max_bin: 8,
        learning_rate: 0.5,
        lambda: 0.0,
        min_child_weight: 0.0,
        nthread: 1,
        cols_per_block: 1,
        interaction_constraints: groups.iter().map(|group| group.to_vec()).collect(),
        gpu_math,
        resident_blocks: Some(resident_blocks),
        gpu_split_cache,
        ..Default::default()
    };
    let source = DenseSource::from_row_major(&DATA, LABELS.len(), FEATURES, 8, 1);

    let cpu_config = TrainConfig::new(
        run_params(GpuMath::Exact, 0, false),
        Objective::SquaredError,
    );
    let cpu = train(&source, &LABELS, &[], &cpu_config, &mut [], &Local).unwrap();
    assert!(
        !split_features(&cpu).is_empty(),
        "constrained training should split"
    );
    assert_paths_fit_groups(&cpu, &groups);
    for tree in &cpu.trees {
        assert!(leaf_count(tree) <= 3);
    }

    for resident_blocks in [0, 1, 3] {
        for gpu_split_cache in [false, true] {
            let exact_config = TrainConfig::new(
                run_params(GpuMath::Exact, resident_blocks, gpu_split_cache),
                Objective::SquaredError,
            );
            let exact =
                train_gpu(0, &source, &LABELS, &[], &exact_config, &mut [], &Local).unwrap();
            assert_eq!(
                bytes(&exact),
                bytes(&cpu),
                "exact GPU mismatch at resident_blocks={resident_blocks}, gpu_split_cache={gpu_split_cache}"
            );

            let run_fast = || {
                let config = TrainConfig::new(
                    run_params(GpuMath::Fast, resident_blocks, gpu_split_cache),
                    Objective::SquaredError,
                );
                train_gpu(0, &source, &LABELS, &[], &config, &mut [], &Local).unwrap()
            };
            let fast = run_fast();
            let fast_again = run_fast();
            assert_eq!(
                bytes(&fast),
                bytes(&fast_again),
                "fast GPU repeat mismatch at resident_blocks={resident_blocks}, gpu_split_cache={gpu_split_cache}"
            );
            assert!(
                !split_features(&fast).is_empty(),
                "fast constrained training should split"
            );
            assert_paths_fit_groups(&fast, &groups);
            for tree in &fast.trees {
                assert!(leaf_count(tree) <= 3);
            }
        }
    }
}
