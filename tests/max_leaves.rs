use ferrisboost::metrics::Metric;
use ferrisboost::{
    columns::BinningStrategy,
    comm::Local,
    source::DenseSource,
    train::{train, EvalSet, TrainConfig},
    types::{Objective, TrainParams},
    Model,
};
#[cfg(feature = "cuda")]
use ferrisboost::{train::train_gpu, types::GpuMath};
use std::sync::{Arc, Mutex};

const SEPARABLE: [f32; 8] = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0];

fn fit(
    values: &[f32],
    labels: &[f32],
    max_leaves: u32,
    max_depth: u32,
    n_rounds: usize,
) -> (Model, DenseSource) {
    fit_with_lambda(values, labels, max_leaves, max_depth, n_rounds, 1.0)
}

fn fit_with_lambda(
    values: &[f32],
    labels: &[f32],
    max_leaves: u32,
    max_depth: u32,
    n_rounds: usize,
    lambda: f32,
) -> (Model, DenseSource) {
    let source = DenseSource::from_row_major(values, labels.len(), 1, 16, 1);
    let params = TrainParams {
        n_rounds,
        max_depth,
        max_leaves,
        max_bin: 8,
        learning_rate: 0.3,
        lambda,
        min_child_weight: 0.0,
        nthread: 1,
        ..Default::default()
    };
    let mut config = TrainConfig::new(params, Objective::SquaredError);
    config.base_score = Some(1.5);
    config.quiet = true;
    let model = train(&source, labels, &[], &config, &mut [], &Local).unwrap();
    (model, source)
}

fn leaf_count(model: &Model, tree_index: usize) -> usize {
    model.trees[tree_index]
        .nodes
        .iter()
        .filter(|node| node.is_leaf)
        .count()
}

struct CaptureMargins(Arc<Mutex<Vec<Vec<f32>>>>);

impl Metric for CaptureMargins {
    fn name(&self) -> &str {
        "capture_margins"
    }

    fn eval(&self, pred: &[f32], _label: &[f32]) -> f32 {
        self.0.lock().unwrap().push(pred.to_vec());
        0.0
    }
}

#[test]
fn leaf_caps_select_depthwise_in_node_order() {
    let labels = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0];

    let (one, _) = fit(&SEPARABLE, &labels, 1, 6, 1);
    assert_eq!(leaf_count(&one, 0), 1);
    assert!(one.trees[0].nodes[0].is_leaf);

    let (two, _) = fit(&SEPARABLE, &labels, 2, 6, 1);
    assert_eq!(leaf_count(&two, 0), 2);
    assert!(!two.trees[0].nodes[0].is_leaf);
    assert!(two.trees[0].nodes[1].is_leaf && two.trees[0].nodes[2].is_leaf);

    // Both depth-one nodes have a useful split. With room for one more leaf,
    // deterministic depthwise node order must split the left node first.
    let (three, _) = fit(&SEPARABLE, &labels, 3, 6, 1);
    let tree = &three.trees[0];
    assert_eq!(leaf_count(&three, 0), 3);
    assert!(!tree.nodes[0].is_leaf);
    assert!(
        !tree.nodes[1].is_leaf,
        "left depth-one node should be selected first"
    );
    assert!(
        tree.nodes[2].is_leaf,
        "right depth-one node should remain a leaf"
    );
}

#[test]
fn zero_and_nonbinding_leaf_limits_match_bit_for_bit() {
    let labels = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0];
    let (unlimited, _) = fit(&SEPARABLE, &labels, 0, 6, 3);
    let (nonbinding, _) = fit(&SEPARABLE, &labels, 64, 6, 3);
    assert_eq!(
        unlimited.to_xgboost_json().unwrap(),
        nonbinding.to_xgboost_json().unwrap()
    );
    for &value in &SEPARABLE {
        assert_eq!(
            unlimited.predict_margin(&[value]).to_bits(),
            nonbinding.predict_margin(&[value]).to_bits()
        );
    }
}

#[test]
fn depth_limit_and_root_only_depth_take_precedence() {
    let labels = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0];
    let (depth_one, _) = fit(&SEPARABLE, &labels, 3, 1, 1);
    assert_eq!(leaf_count(&depth_one, 0), 2);
    assert!(depth_one.trees[0].nodes[1].is_leaf && depth_one.trees[0].nodes[2].is_leaf);

    let (root_only, _) = fit(&SEPARABLE, &labels, 8, 0, 1);
    assert_eq!(leaf_count(&root_only, 0), 1);
    assert_eq!(root_only.trees[0].nodes.len(), 1);
}

#[test]
fn every_round_respects_cap_and_training_predictions_follow_saved_trees() {
    let labels = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0];
    let source = DenseSource::from_row_major(&SEPARABLE, labels.len(), 1, 16, 1);
    let captured = Arc::new(Mutex::new(Vec::new()));
    let params = TrainParams {
        n_rounds: 4,
        max_depth: 6,
        max_leaves: 3,
        max_bin: 8,
        learning_rate: 0.3,
        lambda: 1.0,
        min_child_weight: 0.0,
        nthread: 1,
        ..Default::default()
    };
    let mut config = TrainConfig::new(params, Objective::SquaredError);
    config.base_score = Some(1.5);
    config.quiet = true;
    config.eval_metrics = vec![Box::new(CaptureMargins(Arc::clone(&captured)))];
    let eval = EvalSet {
        name: "training".into(),
        source: &source,
        labels: &labels,
        weights: None,
        group_ptr: None,
    };
    let model = train(&source, &labels, &[eval], &config, &mut [], &Local).unwrap();
    assert_eq!(model.trees.len(), 4);
    assert!(model
        .trees
        .iter()
        .enumerate()
        .all(|(i, _)| leaf_count(&model, i) <= 3));

    // Compare margins captured from the actual eval loop after each round
    // against traversal of the saved model through that round's tree prefix.
    let rounds = captured.lock().unwrap();
    assert_eq!(rounds.len(), model.trees.len());
    for (round_index, observed) in rounds.iter().enumerate() {
        assert_eq!(observed.len(), labels.len());
        for (row, &value) in SEPARABLE.iter().enumerate() {
            let expected = model.predict_margin_upto(&[value], round_index + 1);
            assert_eq!(
                observed[row].to_bits(),
                expected.to_bits(),
                "round {round_index}, row {row}"
            );
        }
    }
}

#[test]
fn an_early_terminal_branch_still_counts_toward_the_total_leaf_cap() {
    let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let labels = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 11.0, 12.0];
    // lambda=0 makes the independent squared-error split-gain calculation
    // transparent: with base margin 1.5, the root isolates the four zeros
    // (gain 231.125), the right child splits [10,10] from [11,12] (gain 2.25),
    // and [11,12] can split again (gain 0.5). The left branch is terminal while
    // the right remains eligible.
    let (model, _) = fit_with_lambda(&values, &labels, 3, 8, 1, 0.0);
    let tree = &model.trees[0];
    let leaves = leaf_count(&model, 0);
    assert!(
        leaves <= 3,
        "terminal branches count toward max_leaves: {leaves}"
    );
    assert!(
        tree.nodes[1].is_leaf && !tree.nodes[2].is_leaf,
        "left branch is already terminal while right branch can continue growing"
    );
}

const CATEGORICAL_EFFECTS: [f32; 4] = [-8.0, 7.0, -3.0, 5.0];

fn categorical_rows() -> (Vec<f32>, Vec<f32>) {
    let mut values = Vec::new();
    let mut labels = Vec::new();
    for repeat in 0..4 {
        for category in 0..4 {
            for low_bit in 0..2 {
                for high_bit in 0..2 {
                    values.extend([category as f32, low_bit as f32, high_bit as f32]);
                    labels.push(
                        CATEGORICAL_EFFECTS[category]
                            + low_bit as f32 * 0.75
                            + high_bit as f32 * 0.25
                            + repeat as f32 * 0.01,
                    );
                }
            }
        }
    }
    (values, labels)
}

fn constrained_categorical_config(seed: u64, cols_per_block: usize) -> TrainConfig {
    let params = TrainParams {
        n_rounds: 2,
        max_depth: 5,
        max_leaves: 3,
        max_bin: 16,
        learning_rate: 0.4,
        lambda: 0.0,
        min_child_weight: 0.0,
        nthread: 1,
        cols_per_block,
        seed,
        colsample_bytree: 0.99,
        colsample_bylevel: 0.99,
        colsample_bynode: 0.99,
        interaction_constraints: vec![vec![0, 1], vec![2]],
        ..Default::default()
    };
    let mut config = TrainConfig::new(params, Objective::SquaredError);
    config.base_score = Some(0.0);
    config.quiet = true;
    config
}

fn categorical_source(values: &[f32], labels: &[f32], cols_per_block: usize) -> DenseSource {
    DenseSource::new(
        values,
        labels.len(),
        3,
        BinningStrategy::SketchTyped {
            max_bins: 16,
            categorical: &[true, false, false],
        },
        cols_per_block,
    )
}

fn assert_capped_and_path_constrained(model: &Model) {
    let groups: &[&[u32]] = &[&[0, 1], &[2]];
    for tree in &model.trees {
        let leaves = tree.nodes.iter().filter(|node| node.is_leaf).count();
        assert!(leaves <= 3, "tree exceeds max_leaves=3: {leaves}");

        fn visit(
            tree: &ferrisboost::tree::Tree,
            node_id: usize,
            path: &mut Vec<u32>,
            groups: &[&[u32]],
        ) {
            let node = &tree.nodes[node_id];
            if node.is_leaf {
                return;
            }
            path.push(node.feat);
            assert!(
                groups
                    .iter()
                    .any(|group| path.iter().all(|feat| group.contains(feat))),
                "interaction group does not cover split path {path:?}"
            );
            visit(tree, node.left as usize, path, groups);
            visit(tree, node.right as usize, path, groups);
            path.pop();
        }

        visit(tree, 0, &mut Vec::new(), &groups);
    }
    assert!(
        model
            .trees
            .iter()
            .flat_map(|tree| &tree.nodes)
            .any(|node| !node.is_leaf && node.cats.is_some()),
        "fixture should exercise a categorical split"
    );
}

#[test]
fn categorical_leaf_cap_respects_interactions_and_nested_colsample() {
    let (values, labels) = categorical_rows();
    // These fixed seeds select categorical feature 0 at the root after the
    // configured bytree -> bylevel -> bynode sampling. Block layout must not
    // change global feature sampling or the cap/path invariants.
    for seed in [2, 4] {
        let mut by_blocking = Vec::new();
        for cols_per_block in [1, 3] {
            let source = categorical_source(&values, &labels, cols_per_block);
            let config = constrained_categorical_config(seed, cols_per_block);
            let model = train(&source, &labels, &[], &config, &mut [], &Local).unwrap();
            assert_capped_and_path_constrained(&model);
            by_blocking.push(model.to_xgboost_json().unwrap());
        }
        assert_eq!(
            by_blocking[0], by_blocking[1],
            "global sampling and constrained capped tree should ignore block width for seed {seed}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn categorical_capped_nested_sampling_matches_cpu_exact_and_repeats_fast() {
    let (values, labels) = categorical_rows();
    let source = categorical_source(&values, &labels, 1);
    let mut cpu_config = constrained_categorical_config(2, 1);
    cpu_config.params.gpu_math = GpuMath::Exact;
    let cpu = train(&source, &labels, &[], &cpu_config, &mut [], &Local).unwrap();
    assert_capped_and_path_constrained(&cpu);

    let exact = train_gpu(0, &source, &labels, &[], &cpu_config, &mut [], &Local).unwrap();
    assert_capped_and_path_constrained(&exact);
    assert_eq!(
        cpu.to_xgboost_json().unwrap(),
        exact.to_xgboost_json().unwrap(),
        "exact GPU should match CPU with interaction masks, categorical splits, sampling, and cap"
    );

    let mut fast_config = constrained_categorical_config(2, 1);
    fast_config.params.gpu_math = GpuMath::Fast;
    let fast_a = train_gpu(0, &source, &labels, &[], &fast_config, &mut [], &Local).unwrap();
    let fast_b = train_gpu(0, &source, &labels, &[], &fast_config, &mut [], &Local).unwrap();
    assert_capped_and_path_constrained(&fast_a);
    assert_capped_and_path_constrained(&fast_b);
    assert_eq!(
        fast_a.to_xgboost_json().unwrap(),
        fast_b.to_xgboost_json().unwrap(),
        "fast GPU repeats should be deterministic"
    );
}
