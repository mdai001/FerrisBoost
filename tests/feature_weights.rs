use anyhow::Result;
use ferrisboost::{
    comm::Local,
    custom_objective::CustomObjective,
    source::DenseSource,
    train::{train, TrainConfig},
    types::{GradPair, Objective, TrainParams},
    Model,
};

const N_FEATURES: usize = 4;
const N_ROWS: usize = 64;

fn data(target_feature: Option<usize>) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(N_ROWS * N_FEATURES);
    let mut y = Vec::with_capacity(N_ROWS);
    for _ in 0..4 {
        for bits in 0..16 {
            let row: Vec<f32> = (0..N_FEATURES)
                .map(|feature| ((bits >> feature) & 1) as f32)
                .collect();
            y.push(match target_feature {
                Some(feature) => row[feature],
                None => row[0] + 2.0 * row[1] + 3.0 * row[2] + 4.0 * row[3],
            });
            x.extend(row);
        }
    }
    (x, y)
}

fn params() -> TrainParams {
    TrainParams {
        n_rounds: 3,
        max_depth: 4,
        max_bin: 8,
        learning_rate: 0.4,
        lambda: 1.0,
        min_child_weight: 0.0,
        nthread: 1,
        cols_per_block: 1,
        ..Default::default()
    }
}

fn source(x: &[f32], params: &TrainParams) -> DenseSource {
    DenseSource::from_row_major(x, N_ROWS, N_FEATURES, params.max_bin, params.cols_per_block)
}

fn fit(x: &[f32], y: &[f32], params: TrainParams) -> Model {
    let src = source(x, &params);
    let cfg = TrainConfig::new(params, Objective::SquaredError);
    train(&src, y, &[], &cfg, &mut [], &Local).unwrap()
}

fn split_features(model: &Model) -> Vec<u32> {
    let mut features: Vec<u32> = model
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

#[test]
fn empty_and_equal_positive_weights_preserve_legacy_nested_sampling() {
    let (x, y) = data(None);
    let mut baseline_params = params();
    baseline_params.colsample_bytree = 0.8;
    baseline_params.colsample_bylevel = 0.75;
    baseline_params.colsample_bynode = 0.6;
    baseline_params.seed = 23;
    let baseline = fit(&x, &y, baseline_params.clone());

    for weight in [1.0, 7.25] {
        let mut weighted_params = baseline_params.clone();
        weighted_params.feature_weights = vec![weight; N_FEATURES];
        let weighted = fit(&x, &y, weighted_params);
        assert_eq!(
            baseline.to_xgboost_json().unwrap(),
            weighted.to_xgboost_json().unwrap(),
            "equal positive weights must use the legacy uniform sequence"
        );
    }
}

#[test]
fn zero_weight_features_are_excluded_when_a_sampling_stage_is_active() {
    let (x, y) = data(Some(1));
    let mut p = params();
    p.n_rounds = 1;
    p.max_depth = 2;
    p.colsample_bytree = 0.5; // requested count is two; only feature 1 is eligible
    p.feature_weights = vec![0.0, 1.0, 0.0, 0.0];
    let model = fit(&x, &y, p);
    assert_eq!(
        split_features(&model),
        vec![1],
        "the tree should split on its sole positive-weight predictive feature"
    );
}

#[test]
fn rate_one_keeps_zero_weight_predictive_features_eligible() {
    let (x, y) = data(Some(0));
    let mut p = params();
    p.n_rounds = 1;
    p.max_depth = 1;
    p.colsample_bytree = 1.0;
    p.colsample_bylevel = 1.0;
    p.colsample_bynode = 1.0;
    p.feature_weights = vec![0.0, 1.0, 0.0, 0.0];
    let model = fit(&x, &y, p);
    assert!(
        split_features(&model).contains(&0),
        "rate=1 must pass through every feature, including zero-weight features"
    );
}

#[test]
fn weighted_sampling_repeats_across_threads_and_block_widths() {
    let (x, y) = data(None);
    let mut narrow_params = params();
    narrow_params.n_rounds = 4;
    narrow_params.max_depth = 3;
    narrow_params.colsample_bytree = 0.75;
    narrow_params.colsample_bylevel = 0.8;
    narrow_params.colsample_bynode = 0.75;
    narrow_params.feature_weights = vec![1.0, 5.0, 2.0, 3.0];
    narrow_params.seed = 47;
    let narrow = fit(&x, &y, narrow_params.clone());

    let repeated = fit(&x, &y, narrow_params.clone());
    assert_eq!(
        narrow.to_xgboost_json().unwrap(),
        repeated.to_xgboost_json().unwrap()
    );

    let mut wider_params = narrow_params;
    wider_params.nthread = 4;
    wider_params.cols_per_block = N_FEATURES;
    let wider = fit(&x, &y, wider_params);
    assert_eq!(
        narrow.to_xgboost_json().unwrap(),
        wider.to_xgboost_json().unwrap(),
        "sampling IDs must not depend on thread count or column-block width"
    );
}

fn assert_paths_fit_groups(model: &Model, groups: &[&[u32]], max_leaves: usize) {
    for tree in &model.trees {
        let leaves = tree.nodes.iter().filter(|node| node.is_leaf).count();
        assert!(leaves <= max_leaves, "tree has {leaves} leaves");
        fn visit(
            tree: &ferrisboost::tree::Tree,
            id: usize,
            path: &mut Vec<u32>,
            groups: &[&[u32]],
        ) {
            let node = &tree.nodes[id];
            if node.is_leaf {
                return;
            }
            path.push(node.feat);
            assert!(
                groups
                    .iter()
                    .any(|group| path.iter().all(|feature| group.contains(feature))),
                "split path {path:?} is outside the interaction groups"
            );
            visit(tree, node.left as usize, path, groups);
            visit(tree, node.right as usize, path, groups);
            path.pop();
        }
        visit(tree, 0, &mut Vec::new(), groups);
    }
}

#[test]
fn weighted_sampling_composes_with_interaction_constraints_and_leaf_caps() {
    let (x, y) = data(None);
    let groups: [&[u32]; 2] = [&[0, 1], &[2, 3]];
    let mut p = params();
    p.n_rounds = 3;
    p.max_depth = 4;
    p.max_leaves = 3;
    p.colsample_bytree = 0.75;
    p.colsample_bylevel = 0.8;
    p.colsample_bynode = 0.75;
    p.feature_weights = vec![1.0, 2.0, 3.0, 4.0];
    p.interaction_constraints = groups.iter().map(|group| group.to_vec()).collect();
    let model = fit(&x, &y, p);
    assert_paths_fit_groups(&model, &groups, 3);
    assert!(!split_features(&model).is_empty());
}

#[test]
fn invalid_feature_weights_are_rejected() {
    let (x, y) = data(None);
    let invalid = [
        vec![1.0, -1.0, 1.0, 1.0],
        vec![1.0, f32::NAN, 1.0, 1.0],
        vec![1.0, f32::INFINITY, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];
    for weights in invalid {
        let mut p = params();
        p.feature_weights = weights;
        let src = source(&x, &p);
        let cfg = TrainConfig::new(p, Objective::SquaredError);
        assert!(
            train(&src, &y, &[], &cfg, &mut [], &Local).is_err(),
            "invalid feature_weights must be rejected"
        );
    }
}

struct SquareObjective;
impl CustomObjective for SquareObjective {
    fn name(&self) -> &str {
        "test:square"
    }
    fn gradients(&self, margin: &[f32], labels: &[f32], out: &mut [GradPair]) -> Result<()> {
        for ((pair, &margin), &label) in out.iter_mut().zip(margin).zip(labels) {
            pair.grad = margin - label;
            pair.hess = 1.0;
        }
        Ok(())
    }
}

#[test]
fn feature_weights_coexist_with_custom_objective_api() {
    let (x, y) = data(Some(1));
    let mut p = params();
    p.n_rounds = 1;
    p.max_depth = 2;
    p.colsample_bytree = 0.5;
    p.feature_weights = vec![0.0, 1.0, 0.0, 0.0];
    let src = source(&x, &p);
    let mut cfg = TrainConfig::new(p, Objective::SquaredError);
    cfg.custom_objective = Some(Box::new(SquareObjective));
    let model = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
    assert_eq!(split_features(&model), vec![1]);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn exact_gpu_weighted_sampling_matches_cpu_across_residency_and_cache() {
    use ferrisboost::{train::train_gpu, types::GpuMath};

    let (x, y) = data(None);
    let mut cpu_params = params();
    cpu_params.n_rounds = 3;
    cpu_params.max_depth = 3;
    cpu_params.colsample_bytree = 0.75;
    cpu_params.colsample_bylevel = 0.8;
    cpu_params.colsample_bynode = 0.75;
    cpu_params.feature_weights = vec![1.0, 4.0, 2.0, 3.0];
    cpu_params.gpu_math = GpuMath::Exact;
    let cpu = fit(&x, &y, cpu_params.clone());

    for resident_blocks in [0, 1] {
        for gpu_split_cache in [false, true] {
            let mut p = cpu_params.clone();
            p.resident_blocks = Some(resident_blocks);
            p.gpu_split_cache = gpu_split_cache;
            let src = source(&x, &p);
            let cfg = TrainConfig::new(p, Objective::SquaredError);
            let gpu = train_gpu(0, &src, &y, &[], &cfg, &mut [], &Local).unwrap();
            assert_eq!(
                cpu.to_xgboost_json().unwrap(),
                gpu.to_xgboost_json().unwrap(),
                "resident_blocks={resident_blocks}, gpu_split_cache={gpu_split_cache}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn fast_gpu_weighted_sampling_repeats_without_claiming_cpu_parity() {
    use ferrisboost::{train::train_gpu, types::GpuMath};

    let (x, y) = data(None);
    let mut p = params();
    p.n_rounds = 3;
    p.max_depth = 3;
    p.colsample_bytree = 0.75;
    p.colsample_bylevel = 0.8;
    p.colsample_bynode = 0.75;
    p.feature_weights = vec![1.0, 4.0, 2.0, 3.0];
    p.gpu_math = GpuMath::Fast;
    let src = source(&x, &p);
    let cfg = TrainConfig::new(p, Objective::SquaredError);
    let first = train_gpu(0, &src, &y, &[], &cfg, &mut [], &Local).unwrap();
    let second = train_gpu(0, &src, &y, &[], &cfg, &mut [], &Local).unwrap();
    assert_eq!(
        first.to_xgboost_json().unwrap(),
        second.to_xgboost_json().unwrap(),
        "same-seed fast-GPU weighted sampling should repeat"
    );
}
