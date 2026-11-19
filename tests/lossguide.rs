use ferrisboost::{
    comm::Local,
    source::DenseSource,
    train::{train, train_weighted_with_grow_policy, TrainConfig},
    types::{GrowPolicy, Objective, TrainParams},
};

fn fit(
    values: &[f32],
    labels: &[f32],
    max_depth: u32,
    max_leaves: u32,
    grow_policy: GrowPolicy,
) -> ferrisboost::Model {
    let source = DenseSource::from_row_major(values, labels.len(), 1, 16, 1);
    let params = TrainParams {
        n_rounds: 1,
        max_depth,
        max_leaves,
        max_bin: 16,
        learning_rate: 0.3,
        lambda: 0.0,
        min_child_weight: 0.0,
        nthread: 1,
        ..Default::default()
    };
    let mut config = TrainConfig::new(params, Objective::SquaredError);
    config.base_score = Some(0.0);
    config.quiet = true;
    train_weighted_with_grow_policy(
        &source,
        labels,
        None,
        &[],
        &config,
        &mut [],
        &Local,
        grow_policy,
    )
    .unwrap()
}

fn leaves(model: &ferrisboost::Model) -> usize {
    model.trees[0]
        .nodes
        .iter()
        .filter(|node| node.is_leaf)
        .count()
}

const X8: [f32; 8] = [0., 1., 2., 3., 4., 5., 6., 7.];

#[test]
fn depthwise_is_the_default_growth_policy() {
    let labels = [0., 0., 1., 1., 10., 10., 14., 20.];
    let source = DenseSource::from_row_major(&X8, labels.len(), 1, 16, 1);
    let params = TrainParams {
        n_rounds: 1,
        max_depth: 8,
        max_bin: 16,
        learning_rate: 0.3,
        lambda: 0.0,
        min_child_weight: 0.0,
        nthread: 1,
        ..Default::default()
    };
    let mut config = TrainConfig::new(params, Objective::SquaredError);
    config.base_score = Some(0.0);
    config.quiet = true;

    let default = train(&source, &labels, &[], &config, &mut [], &Local).unwrap();
    let explicit = train_weighted_with_grow_policy(
        &source,
        &labels,
        None,
        &[],
        &config,
        &mut [],
        &Local,
        GrowPolicy::Depthwise,
    )
    .unwrap();
    let default_bytes = default.to_xgboost_json().unwrap();
    let explicit_bytes = explicit.to_xgboost_json().unwrap();
    assert_eq!(default_bytes.as_bytes(), explicit_bytes.as_bytes());
}

#[test]
fn lossguide_selects_highest_gain_eligible_leaf_under_cap() {
    // The root separates the low and high halves. Both children have a useful
    // split, but the right child's gain is much larger than the left's. With a
    // three-leaf cap, best-first growth must spend the extra split on the right.
    let y = [0., 0., 1., 1., 10., 10., 14., 20.];
    let model = fit(&X8, &y, 8, 3, GrowPolicy::LossGuide);
    let tree = &model.trees[0];
    assert_eq!(leaves(&model), 3);
    assert!(!tree.nodes[0].is_leaf);
    assert!(
        tree.nodes[1].is_leaf,
        "lower-gain left candidate should stay a leaf"
    );
    assert!(
        !tree.nodes[2].is_leaf,
        "higher-gain right candidate should be selected"
    );
}

#[test]
fn exact_gain_ties_choose_lower_node_id() {
    // Symmetric child distributions give identical split gains. Node 1 (left)
    // must win the deterministic tie against node 2 (right).
    let y = [0., 0., 1., 1., 3., 3., 4., 4.];
    let model = fit(&X8, &y, 8, 3, GrowPolicy::LossGuide);
    let tree = &model.trees[0];
    assert_eq!(leaves(&model), 3);
    assert!(!tree.nodes[1].is_leaf);
    assert!(tree.nodes[2].is_leaf);
}

#[test]
fn depth_zero_and_one_leaf_cap_both_make_root_only_trees() {
    let y = [0., 0., 1., 1., 3., 3., 4., 4.];
    for (depth, cap) in [(0, 0), (8, 1)] {
        let model = fit(&X8, &y, depth, cap, GrowPolicy::LossGuide);
        assert_eq!(leaves(&model), 1);
        assert_eq!(model.trees[0].nodes.len(), 1);
    }
}

#[test]
fn zero_leaf_cap_is_unlimited_when_depth_allows_growth() {
    let x: Vec<f32> = (0..32).map(|v| v as f32).collect();
    let y: Vec<f32> = (0..32).map(|v| ((v / 4) * (v / 4)) as f32).collect();
    let unlimited = fit(&x, &y, 8, 0, GrowPolicy::LossGuide);
    let capped = fit(&x, &y, 8, 3, GrowPolicy::LossGuide);
    assert!(leaves(&unlimited) > 3, "max_leaves=0 must not impose a cap");
    assert_eq!(leaves(&capped), 3);
}

#[test]
fn lossguide_compares_candidates_at_different_depths() {
    let y = [0., 0., 1., 1., 10., 10., 14., 20.];
    let model = fit(&X8, &y, 8, 4, GrowPolicy::LossGuide);
    let nodes = &model.trees[0].nodes;
    assert_eq!(leaves(&model), 4);
    assert!(nodes[1].is_leaf, "shallower, lower-gain candidate");
    assert!(!nodes[3].is_leaf, "deeper, higher-gain candidate");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn cuda_lossguide_exact_matches_cpu_topology_and_fast_uses_best_first_growth() {
    use ferrisboost::train::train_gpu_weighted_with_grow_policy;
    use ferrisboost::types::GpuMath;

    let y = [0., 0., 1., 1., 10., 10., 14., 20.];
    let src = DenseSource::from_row_major(&X8, y.len(), 1, 16, 1);
    let params = TrainParams {
        n_rounds: 1,
        max_depth: 8,
        max_leaves: 4,
        max_bin: 16,
        learning_rate: 0.3,
        lambda: 0.0,
        min_child_weight: 0.0,
        nthread: 1,
        ..Default::default()
    };
    let mut config = TrainConfig::new(params, Objective::SquaredError);
    config.base_score = Some(0.0);
    config.quiet = true;
    let cpu = train_weighted_with_grow_policy(
        &src,
        &y,
        None,
        &[],
        &config,
        &mut [],
        &Local,
        GrowPolicy::LossGuide,
    )
    .unwrap();

    for math in [GpuMath::Exact, GpuMath::Fast] {
        config.params.gpu_math = math;
        let gpu = train_gpu_weighted_with_grow_policy(
            0,
            &src,
            &y,
            None,
            &[],
            &config,
            &mut [],
            &Local,
            GrowPolicy::LossGuide,
        )
        .unwrap();
        let nodes = &gpu.trees[0].nodes;
        assert_eq!(leaves(&gpu), 4, "math={math:?}");
        assert!(!nodes[0].is_leaf, "root must split, math={math:?}");
        assert!(
            nodes[1].is_leaf,
            "lower-gain shallow candidate, math={math:?}"
        );
        assert!(!nodes[2].is_leaf, "higher-gain candidate, math={math:?}");
        assert!(
            !nodes[3].is_leaf,
            "best candidate at a deeper level, math={math:?}"
        );

        if math == GpuMath::Exact {
            assert_eq!(
                cpu.to_xgboost_json().unwrap(),
                gpu.to_xgboost_json().unwrap(),
                "exact CUDA loss-guided model should match CPU"
            );
        }
    }

    config.params.max_leaves = 1;
    config.params.gpu_math = GpuMath::Exact;
    let gpu = train_gpu_weighted_with_grow_policy(
        0,
        &src,
        &y,
        None,
        &[],
        &config,
        &mut [],
        &Local,
        GrowPolicy::LossGuide,
    )
    .unwrap();
    assert_eq!(leaves(&gpu), 1);
    assert_eq!(gpu.trees[0].nodes.len(), 1);
}
