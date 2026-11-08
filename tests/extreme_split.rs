use ferrisboost::callback::RecordHistory;
use ferrisboost::columns::BinningStrategy;
use ferrisboost::train::{train, BlockSource, EvalSet, TrainConfig};
#[cfg(feature = "cuda")]
use ferrisboost::types::GpuMath;
use ferrisboost::types::{Objective, TrainParams, MISSING_BIN};
use ferrisboost::{DenseSource, Model};

const LABELS: [f32; 4] = [1.0, 1.0, 0.0, 0.0];

fn params(max_bin: u32, n_rounds: usize, learning_rate: f32) -> TrainParams {
    TrainParams {
        n_rounds,
        max_depth: 1,
        max_bin,
        learning_rate,
        lambda: 1.0,
        min_child_weight: 0.0,
        nthread: 1,
        ..Default::default()
    }
}

fn train_one_feature(values: [f32; 4], max_bin: u32) -> (Model, DenseSource) {
    let source = DenseSource::from_row_major(&values, 4, 1, max_bin, 1);
    let mut config = TrainConfig::new(params(max_bin, 1, 1.0), Objective::SquaredError);
    config.base_score = Some(0.5);
    let model = train(
        &source,
        &LABELS,
        &[],
        &config,
        &mut [],
        &ferrisboost::comm::Local,
    )
    .unwrap();
    (model, source)
}

fn assert_binary_split(model: &Model, positive: f32, negative: f32) {
    assert_eq!(model.trees.len(), 1);
    let tree = &model.trees[0];
    assert_eq!(tree.nodes.len(), 3);
    assert!(!tree.nodes[0].is_leaf, "expected a root split");
    assert!(tree.nodes[1].is_leaf && tree.nodes[2].is_leaf);
    let root = &tree.nodes[0];
    assert_eq!((root.left, root.right), (1, 2));
    assert!((root.gain - (2.0 / 3.0)).abs() < 1e-6);
    assert_eq!(root.sum_hess, 4.0);
    assert!((tree.nodes[1].sum_hess - 2.0).abs() < 1e-6);
    assert!((tree.nodes[2].sum_hess - 2.0).abs() < 1e-6);
    assert!((model.predict_margin(&[positive]) - 0.833_333_3).abs() < 1e-6);
    assert!((model.predict_margin(&[negative]) - 0.16666666).abs() < 1e-6);
}

#[test]
fn exact_f32_max_is_rejected_by_dense_source_entry_points() {
    let values = [f32::MAX, f32::MAX, f32::NAN, f32::NAN];
    assert!(std::panic::catch_unwind(|| DenseSource::from_row_major(&values, 4, 1, 1, 1)).is_err());
    assert!(std::panic::catch_unwind(|| {
        DenseSource::new(&values, 4, 1, BinningStrategy::Sketch { max_bins: 1 }, 1)
    })
    .is_err());

    let cuts = ferrisboost::BinCuts::new(vec![0.0], vec![0, 1]);
    assert!(std::panic::catch_unwind(|| {
        DenseSource::new(&values, 4, 1, BinningStrategy::Provided(&cuts), 1)
    })
    .is_err());
    assert!(
        std::panic::catch_unwind(|| { DenseSource::with_cuts(&values, 4, 1, cuts, 1) }).is_err()
    );
}

#[test]
fn predecessor_of_f32_max_and_missing_split_with_one_bin_and_roundtrip() {
    let below_max = f32::from_bits(f32::MAX.to_bits() - 1);
    let (model, source) = train_one_feature([below_max, below_max, f32::NAN, f32::NAN], 1);

    assert_eq!(source.bin_at(0, 0), source.bin_at(1, 0));
    assert_eq!(source.bin_at(2, 0), MISSING_BIN);
    assert_eq!(source.cuts().n_bins(0), 1);
    assert_binary_split(&model, below_max, f32::NAN);

    let json = model.to_xgboost_json().unwrap();
    let loaded = Model::from_xgboost_json(&json).unwrap();
    for value in [below_max, f32::NAN] {
        assert_eq!(
            model.predict_margin(&[value]).to_bits(),
            loaded.predict_margin(&[value]).to_bits()
        );
    }
    assert_eq!(json, loaded.to_xgboost_json().unwrap());
}

#[test]
fn f32_min_and_missing_remain_separable() {
    let (model, source) = train_one_feature([f32::MIN, f32::MIN, f32::NAN, f32::NAN], 1);
    assert_eq!(source.cuts().n_bins(0), 1);
    assert_binary_split(&model, f32::MIN, f32::NAN);
}

#[test]
fn multiround_eval_uses_training_cuts_for_extreme_finite_values() {
    let below_max = f32::from_bits(f32::MAX.to_bits() - 1);
    let data = [below_max, below_max, f32::NAN, f32::NAN];
    let training = DenseSource::from_row_major(&data, 4, 1, 1, 1);
    let evaluation = DenseSource::with_cuts(&data, 4, 1, training.cuts().clone(), 1);
    let mut config = TrainConfig::new(params(1, 3, 0.5), Objective::SquaredError);
    config.base_score = Some(0.5);
    let history = RecordHistory::default();
    let handle = history.handle();
    let mut callbacks: Vec<Box<dyn ferrisboost::Callback>> = vec![Box::new(history)];
    let eval = EvalSet {
        name: "validation".into(),
        source: &evaluation,
        labels: &LABELS,
        weights: None,
        group_ptr: None,
    };
    let model = train(
        &training,
        &LABELS,
        &[eval],
        &config,
        &mut callbacks,
        &ferrisboost::comm::Local,
    )
    .unwrap();

    let rounds = handle.lock().unwrap();
    assert_eq!(rounds.len(), 3);
    assert!(rounds
        .iter()
        .all(|round| { round.get("validation", "rmse").is_some_and(f32::is_finite) }));
    drop(rounds);

    let finite_margin = model.predict_margin(&[below_max]);
    let missing_margin = model.predict_margin(&[f32::NAN]);
    assert!(finite_margin > missing_margin);
    assert_eq!(model.trees.len(), 3);
    let loaded = Model::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap();
    for value in [below_max, f32::NAN] {
        assert_eq!(
            model.predict_margin(&[value]).to_bits(),
            loaded.predict_margin(&[value]).to_bits()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_exact_matches_cpu_and_fast_routes_supported_upper_finite_values() {
    let below_max = f32::from_bits(f32::MAX.to_bits() - 1);
    let data = [below_max, below_max, f32::NAN, f32::NAN];
    let source = DenseSource::from_row_major(&data, 4, 1, 1, 1);

    let mut exact_params = params(1, 1, 1.0);
    exact_params.gpu_math = GpuMath::Exact;
    let mut exact_config = TrainConfig::new(exact_params, Objective::Logistic);
    exact_config.base_score = Some(0.5);
    let cpu = train(
        &source,
        &LABELS,
        &[],
        &exact_config,
        &mut [],
        &ferrisboost::comm::Local,
    )
    .unwrap();
    let gpu_exact = ferrisboost::train::train_gpu(
        0,
        &source,
        &LABELS,
        &[],
        &exact_config,
        &mut [],
        &ferrisboost::comm::Local,
    )
    .unwrap();
    assert_eq!(
        cpu.to_xgboost_json().unwrap(),
        gpu_exact.to_xgboost_json().unwrap()
    );

    let mut fast_params = params(1, 1, 1.0);
    fast_params.gpu_math = GpuMath::Fast;
    let mut fast_config = TrainConfig::new(fast_params, Objective::Logistic);
    fast_config.base_score = Some(0.5);
    let gpu_fast = ferrisboost::train::train_gpu(
        0,
        &source,
        &LABELS,
        &[],
        &fast_config,
        &mut [],
        &ferrisboost::comm::Local,
    )
    .unwrap();
    let finite_margin = gpu_fast.predict_margin(&[below_max]);
    let missing_margin = gpu_fast.predict_margin(&[f32::NAN]);
    assert!(finite_margin.is_finite() && missing_margin.is_finite());
    assert!(finite_margin > missing_margin);
}
