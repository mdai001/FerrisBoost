use ferrisboost::{
    comm::Local,
    source::DenseSource,
    train::{train_weighted_with_grow_policy, TrainConfig},
    types::{GrowPolicy, Objective, TrainParams},
    Model,
};

const N_FEATURES: usize = 3;
const X: [f32; 24] = [
    0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1.,
];
const Y: [f32; 8] = [-23., -15., -17., -9., 11., 13., 19., 21.];
const PREDICTIONS: [f32; 8] = [-19., -19., -13., -13., 12., 12., 20., 20.];

fn config() -> TrainConfig {
    let params = TrainParams {
        n_rounds: 1,
        max_depth: 2,
        max_bin: 16,
        learning_rate: 1.0,
        lambda: 0.0,
        min_child_weight: 1.0,
        cols_per_block: 1,
        nthread: 1,
        ..Default::default()
    };
    let mut config = TrainConfig::new(params, Objective::SquaredError);
    config.base_score = Some(0.0);
    config.quiet = true;
    config
}

fn fit_with_policy(
    x: &[f32],
    y: &[f32],
    n_features: usize,
    config: &TrainConfig,
    grow_policy: GrowPolicy,
) -> Model {
    let source = DenseSource::from_row_major(
        x,
        y.len(),
        n_features,
        config.params.max_bin,
        config.params.cols_per_block,
    );
    train_weighted_with_grow_policy(&source, y, None, &[], config, &mut [], &Local, grow_policy)
        .unwrap()
}

fn fit(x: &[f32], y: &[f32], n_features: usize, config: &TrainConfig) -> Model {
    fit_with_policy(x, y, n_features, config, GrowPolicy::Symmetric)
}

/// Walk children instead of relying on the order nodes are stored in.
fn assert_levels(model: &Model, features: &[u32]) {
    let nodes = &model.trees[0].nodes;
    let mut frontier = vec![0];
    let mut visited = vec![false; nodes.len()];

    for &feature in features {
        let first = &nodes[frontier[0]];
        let mut children = Vec::with_capacity(2 * frontier.len());
        for id in frontier {
            assert!(!visited[id], "repeated node {id}");
            visited[id] = true;
            let node = &nodes[id];
            assert!(!node.is_leaf, "early leaf at node {id}");
            assert_eq!(node.feat, feature);
            assert_eq!(node.split_cond.to_bits(), first.split_cond.to_bits());
            assert_eq!(node.cats, first.cats);
            assert_eq!(node.default_left, first.default_left);
            children.push(usize::try_from(node.left).unwrap());
            children.push(usize::try_from(node.right).unwrap());
        }
        frontier = children;
    }

    for id in frontier {
        assert!(!visited[id], "repeated leaf {id}");
        visited[id] = true;
        assert!(nodes[id].is_leaf, "unexpected split at node {id}");
        assert_eq!((nodes[id].left, nodes[id].right), (-1, -1));
    }
    assert!(visited.into_iter().all(|seen| seen), "unreachable nodes");
}

fn predictions(model: &Model) -> Vec<f32> {
    X.chunks_exact(N_FEATURES)
        .map(|row| model.predict(row))
        .collect()
}

#[test]
fn symmetric_maximizes_gain_over_the_level_and_round_trips() {
    // The left child's best feature is 2 (64 vs 36). The right child favors
    // feature 1 (64 vs 4), so the shared winner is feature 1 (100 vs 68).
    let depthwise = config();
    let model = fit_with_policy(&X, &Y, N_FEATURES, &depthwise, GrowPolicy::Depthwise);
    let nodes = &model.trees[0].nodes;
    assert_eq!(nodes[0].feat, 0);
    assert_eq!(nodes[nodes[0].left as usize].feat, 2);
    assert_eq!(nodes[nodes[0].right as usize].feat, 1);

    let model = fit(&X, &Y, N_FEATURES, &config());
    assert_levels(&model, &[0, 1]);
    assert_eq!(predictions(&model), PREDICTIONS);
    let json = model.to_xgboost_json().unwrap();
    let loaded = Model::from_xgboost_json(&json).unwrap();
    assert_levels(&loaded, &[0, 1]);
    assert_eq!(predictions(&loaded), PREDICTIONS);
}

#[test]
fn symmetric_applies_regularization_and_gamma_to_the_level() {
    let mut config = config();
    // One child gains only 36. The level still pays for both splits at gamma
    // 40, and loses at gamma 51 (joint gain 100).
    config.params.gamma = 40.0;
    assert_levels(&fit(&X, &Y, N_FEATURES, &config), &[0, 1]);
    config.params.gamma = 51.0;
    assert_levels(&fit(&X, &Y, N_FEATURES, &config), &[0]);

    // L2 makes the left child's improvement negative, while the sum over
    // both children is still positive for feature 1.
    config.params.gamma = 0.0;
    config.params.lambda = 0.2;
    let model = fit(&X, &Y, N_FEATURES, &config);
    assert_levels(&model, &[0, 1]);
    let nodes = &model.trees[0].nodes;
    assert!(nodes[nodes[0].left as usize].gain < 0.0);
}

#[test]
fn symmetric_checks_child_weight_in_every_node() {
    let mut config = config();
    // With the final row removed, each remaining feature splits the right
    // node into children of sizes 2 and 1, but splits the left into 2 and 2.
    let labels = &Y[..7];
    let data = &X[..labels.len() * N_FEATURES];
    assert_levels(&fit(data, labels, N_FEATURES, &config), &[0, 1]);
    config.params.min_child_weight = 2.0;
    assert_levels(&fit(data, labels, N_FEATURES, &config), &[0]);
}

#[test]
fn symmetric_respects_monotone_and_interaction_constraints() {
    let mut config = config();
    config.params.interaction_constraints = vec![vec![0, 2], vec![1]];
    assert_levels(&fit(&X, &Y, N_FEATURES, &config), &[0, 2]);

    // Feature 1 increases on the left of the root and decreases on the
    // right. The increasing constraint must reject it across the level.
    let labels = [-23., -15., -17., -9., 19., 21., 11., 13.];
    config.params.interaction_constraints.clear();
    assert_levels(&fit(&X, &labels, N_FEATURES, &config), &[0, 1]);
    config.params.monotone_constraints = vec![0, 1, 0];
    assert_levels(&fit(&X, &labels, N_FEATURES, &config), &[0, 2]);
}

#[test]
fn symmetric_leaf_cap_accepts_only_complete_levels() {
    let labels: Vec<f32> = (0..32).map(|row| row as f32).collect();
    let data: Vec<f32> = (0..32)
        .flat_map(|row| (0..5).rev().map(move |bit| ((row >> bit) & 1) as f32))
        .collect();
    let mut config = config();
    config.params.max_depth = 5;
    for (cap, expected_depth) in [(0, 5), (1, 0), (3, 1), (31, 4), (32, 5)] {
        config.params.max_leaves = cap;
        assert_levels(
            &fit(&data, &labels, 5, &config),
            &[0, 1, 2, 3, 4][..expected_depth],
        );
    }
    config.params.max_leaves = 0;
    config.params.max_depth = 0;
    assert_levels(&fit(&data, &labels, 5, &config), &[]);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn cuda_symmetric_exact_matches_cpu_and_fast_keeps_the_shared_split() {
    use ferrisboost::{train::train_gpu_weighted_with_grow_policy, types::GpuMath};

    let source = DenseSource::from_row_major(&X, Y.len(), N_FEATURES, 16, 1);
    let mut config = config();
    config.params.gamma = 40.0;
    config.params.resident_blocks = Some(N_FEATURES);
    let cpu = fit_with_policy(&X, &Y, N_FEATURES, &config, GrowPolicy::Symmetric);
    assert_levels(&cpu, &[0, 1]);

    config.params.gpu_math = GpuMath::Exact;
    let exact = train_gpu_weighted_with_grow_policy(
        0,
        &source,
        &Y,
        None,
        &[],
        &config,
        &mut [],
        &Local,
        GrowPolicy::Symmetric,
    )
    .unwrap();
    assert_eq!(
        exact.to_xgboost_json().unwrap(),
        cpu.to_xgboost_json().unwrap()
    );
    assert_eq!(predictions(&exact), PREDICTIONS);

    config.params.gpu_math = GpuMath::Fast;
    let fast = train_gpu_weighted_with_grow_policy(
        0,
        &source,
        &Y,
        None,
        &[],
        &config,
        &mut [],
        &Local,
        GrowPolicy::Symmetric,
    )
    .unwrap();
    assert_levels(&fast, &[0, 1]);
    for (actual, expected) in predictions(&fast).into_iter().zip(PREDICTIONS) {
        approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn cuda_symmetric_exact_matches_cpu_for_quantized_numeric_frontiers() {
    use ferrisboost::{train::train_gpu_weighted_with_grow_policy, types::GpuMath, BinCuts};

    const FEATURES: usize = 4;
    let mut x = Vec::new();
    let mut y = Vec::new();
    for group in 0..2 {
        let (offset, effect) = if group == 0 { (-64., 4.) } else { (64., 12.) };
        for value in 0..72 {
            let missing = value >= 64;
            // In the left node missing rows favor the low side. The right
            // node's preference for the high side wins on aggregate.
            let high = if missing { group == 1 } else { value >= 32 };
            x.extend([
                group as f32,
                if missing { f32::NAN } else { value as f32 },
                if high { 1. } else { 0. },
                if high { 0. } else { 1. },
            ]);
            y.push(offset + if high { effect } else { -effect });
        }
    }

    // Feature 1 has several valid cuts and two common missing directions.
    // Features 2 and 3 encode the local best partition but the interaction
    // and monotone constraints, respectively, disallow them at depth one.
    let cuts = BinCuts::new(
        vec![1., 8., 16., 24., 32., 40., 48., 56., 1., 1.],
        vec![0, 1, 8, 9, 10],
    );
    let mut config = config();
    config.params.min_child_weight = 8.0;
    config.params.monotone_constraints = vec![1, 1, 0, 1];
    config.params.interaction_constraints = vec![vec![0, 1, 3], vec![2]];
    config.params.device_quantize = Some(true);
    config.params.gpu_math = GpuMath::Exact;

    for constrained in [true, false] {
        if !constrained {
            config.params.monotone_constraints.clear();
            config.params.interaction_constraints.clear();
            for row in x.chunks_exact_mut(FEATURES) {
                row[2..].fill(0.0);
            }
        }
        let source = DenseSource::with_cuts(&x, y.len(), FEATURES, cuts.clone(), 1);
        let cpu = train_weighted_with_grow_policy(
            &source,
            &y,
            None,
            &[],
            &config,
            &mut [],
            &Local,
            GrowPolicy::Symmetric,
        )
        .unwrap();
        assert_levels(&cpu, &[0, 1]);
        let nodes = &cpu.trees[0].nodes;
        let split = &nodes[nodes[0].left as usize];
        assert_eq!(split.split_cond, 32.0);
        assert!(!split.default_left);
        assert_eq!(cpu.predict(&[0., f32::NAN, 0., 1.]), -61.6);
        assert_eq!(cpu.predict(&[1., f32::NAN, 1., 0.]), 76.0);

        for resident in [FEATURES, 2, 0] {
            config.params.resident_blocks = Some(resident);
            let gpu = train_gpu_weighted_with_grow_policy(
                0,
                &source,
                &y,
                None,
                &[],
                &config,
                &mut [],
                &Local,
                GrowPolicy::Symmetric,
            )
            .unwrap();
            assert_eq!(
                cpu.to_xgboost_json().unwrap(),
                gpu.to_xgboost_json().unwrap()
            );
            for row in x.chunks_exact(FEATURES) {
                assert_eq!(cpu.predict(row).to_bits(), gpu.predict(row).to_bits());
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn cuda_symmetric_falls_back_with_prebinned_wide_columns_or_a_small_batch() {
    use ferrisboost::{train::train_gpu_weighted_with_grow_policy, BlockSource};

    let x: Vec<f32> = (0..1024)
        .flat_map(|i| {
            [
                i as f32,
                ((i >> 3) & 1) as f32,
                ((i >> 2) & 1) as f32,
                ((i >> 1) & 1) as f32,
                (i & 1) as f32,
            ]
        })
        .collect();
    let y: Vec<f32> = (0..1024).map(|i| (i % 16) as f32).collect();
    let mut config = config();
    config.params.max_depth = 4;
    config.params.resident_blocks = Some(5);
    for (source_bins, batch) in [(512, 16), (16, 2)] {
        let source = DenseSource::from_row_major(&x, y.len(), 5, source_bins, 1);
        if source_bins == 512 {
            assert!(source.cuts().n_bins(0) > 255);
        }
        config.params.hist_nodes_per_batch = Some(batch);
        let cpu = fit_with_policy(&x, &y, 5, &config, GrowPolicy::Symmetric);
        assert_levels(&cpu, &[1, 2, 3, 4]);
        let gpu = train_gpu_weighted_with_grow_policy(
            0,
            &source,
            &y,
            None,
            &[],
            &config,
            &mut [],
            &Local,
            GrowPolicy::Symmetric,
        )
        .unwrap();
        assert_eq!(
            cpu.to_xgboost_json().unwrap(),
            gpu.to_xgboost_json().unwrap()
        );
    }
}
