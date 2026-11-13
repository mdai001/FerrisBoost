use anyhow::{bail, Result};
#[cfg(feature = "cuda")]
use ferrisboost::train::train_gpu;
use ferrisboost::{
    comm::Local,
    custom_objective::CustomObjective,
    source::DenseSource,
    train::{train, train_weighted, TrainConfig},
    types::{GradPair, Objective, TrainParams},
    Model,
};
use std::sync::{Arc, Mutex};

const X: [f32; 12] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0];
const Y: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];

fn params(rounds: usize) -> TrainParams {
    TrainParams {
        n_rounds: rounds,
        max_depth: 3,
        max_bin: 8,
        learning_rate: 0.3,
        lambda: 1.0,
        min_child_weight: 0.0,
        nthread: 1,
        subsample: 0.75,
        seed: 17,
        ..Default::default()
    }
}

fn source() -> DenseSource {
    DenseSource::from_row_major(&X, Y.len(), 1, 8, 1)
}

#[derive(Clone)]
struct Square {
    name: &'static str,
    calls: Option<Arc<Mutex<Vec<Vec<f32>>>>>,
}
impl Square {
    fn new(name: &'static str) -> Self {
        Self { name, calls: None }
    }
}
impl CustomObjective for Square {
    fn name(&self) -> &str {
        self.name
    }
    fn gradients(&self, margins: &[f32], labels: &[f32], out: &mut [GradPair]) -> Result<()> {
        if let Some(calls) = &self.calls {
            calls.lock().unwrap().push(margins.to_vec());
        }
        for ((pair, &margin), &label) in out.iter_mut().zip(margins).zip(labels) {
            pair.grad = margin - label;
            pair.hess = 1.0;
        }
        Ok(())
    }
}

struct Fixed {
    name: &'static str,
    grad: f32,
    hess: f32,
    fail: bool,
}
impl CustomObjective for Fixed {
    fn name(&self) -> &str {
        self.name
    }
    fn gradients(&self, _m: &[f32], _y: &[f32], out: &mut [GradPair]) -> Result<()> {
        if self.fail {
            bail!("deliberate custom objective failure");
        }
        out.fill(GradPair {
            grad: self.grad,
            hess: self.hess,
        });
        Ok(())
    }
}

struct PartialOutput;
impl CustomObjective for PartialOutput {
    fn name(&self) -> &str {
        "test:partial-output"
    }
    fn gradients(&self, _m: &[f32], _y: &[f32], out: &mut [GradPair]) -> Result<()> {
        if let Some(first) = out.first_mut() {
            *first = GradPair {
                grad: 0.0,
                hess: 1.0,
            };
        }
        Ok(())
    }
}

fn config(rounds: usize, base: Option<f32>) -> TrainConfig {
    let mut c = TrainConfig::new(params(rounds), Objective::SquaredError);
    c.base_score = base;
    c.quiet = true;
    c
}
fn custom_config(
    rounds: usize,
    base: Option<f32>,
    objective: impl CustomObjective + 'static,
) -> TrainConfig {
    let mut c = config(rounds, base);
    c.custom_objective = Some(Box::new(objective));
    c
}
fn custom_model(
    rounds: usize,
    base: Option<f32>,
    objective: impl CustomObjective + 'static,
) -> Model {
    train(
        &source(),
        &Y,
        &[],
        &custom_config(rounds, base, objective),
        &mut [],
        &Local,
    )
    .unwrap()
}

#[test]
fn squared_error_callback_matches_builtin_with_weights_and_subsampling() {
    let src = source();
    let weights = [1.0, 0.5, 1.0, 2.0, 1.0, 0.5, 1.5, 1.0, 0.75, 1.0, 2.0, 1.0];
    let builtin = train_weighted(
        &src,
        &Y,
        Some(&weights),
        &[],
        &config(4, Some(0.7)),
        &mut [],
        &Local,
    )
    .unwrap();
    let custom = train_weighted(
        &src,
        &Y,
        Some(&weights),
        &[],
        &custom_config(4, Some(0.7), Square::new("test:square")),
        &mut [],
        &Local,
    )
    .unwrap();
    assert_eq!(builtin.trees.len(), 4);
    assert_eq!(custom.trees.len(), 4);
    let builtin_json: serde_json::Value =
        serde_json::from_str(&builtin.to_xgboost_json().unwrap()).unwrap();
    let mut custom_json: serde_json::Value =
        serde_json::from_str(&custom.to_xgboost_json().unwrap()).unwrap();
    custom_json["learner"]["attributes"]
        .as_object_mut()
        .unwrap()
        .remove("ferrisboost_custom_objective");
    assert_eq!(builtin_json, custom_json);
    for &x in &X {
        assert_eq!(
            builtin.predict_margin(&[x]).to_bits(),
            custom.predict_margin(&[x]).to_bits()
        );
    }
}

#[test]
fn callback_runs_once_per_round_on_all_rows_with_updated_margins() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = custom_model(
        4,
        Some(0.0),
        Square {
            name: "test:capture",
            calls: Some(Arc::clone(&calls)),
        },
    );
    let rounds = calls.lock().unwrap();
    assert_eq!(rounds.len(), 4);
    assert!(rounds.iter().all(|m| m.len() == Y.len()));
    assert_eq!(rounds[0][0].to_bits(), 0.0f32.to_bits());
    for round in 0..rounds.len() {
        for (row, &x) in X.iter().enumerate() {
            assert_eq!(
                rounds[round][row].to_bits(),
                model.predict_margin_upto(&[x], round).to_bits()
            );
        }
    }
    assert_ne!(rounds[0], rounds[1]);
}

#[test]
fn custom_objective_defaults_base_score_to_zero() {
    let model = custom_model(1, None, Square::new("test:zero"));
    assert_eq!(model.base_score.to_bits(), 0.0f32.to_bits());
}

#[test]
fn malformed_gradients_and_callback_errors_are_reported() {
    let src = source();
    for (grad, hess) in [(f32::NAN, 1.0), (0.0, f32::INFINITY), (0.0, -1.0)] {
        let c = custom_config(
            1,
            Some(0.0),
            Fixed {
                name: "test:bad",
                grad,
                hess,
                fail: false,
            },
        );
        assert!(train(&src, &Y, &[], &c, &mut [], &Local).is_err());
    }
    let c = custom_config(
        1,
        Some(0.0),
        Fixed {
            name: "test:err",
            grad: 0.0,
            hess: 1.0,
            fail: true,
        },
    );
    let error = train(&src, &Y, &[], &c, &mut [], &Local).unwrap_err();
    assert!(
        format!("{error:#}").contains("deliberate custom objective failure"),
        "{error:#}"
    );
}

#[test]
fn partially_filled_output_and_empty_provider_name_are_rejected() {
    let partial = custom_config(1, Some(0.0), PartialOutput);
    let error = train(&source(), &Y, &[], &partial, &mut [], &Local).unwrap_err();
    assert!(format!("{error:#}").contains("must be finite"), "{error:#}");

    let empty_name = custom_config(1, Some(0.0), Square::new(""));
    let error = train(&source(), &Y, &[], &empty_name, &mut [], &Local).unwrap_err();
    assert!(
        format!("{error:#}").contains("name must not be empty"),
        "{error:#}"
    );
}

#[test]
fn zero_gradient_and_hessian_are_valid() {
    let c = custom_config(
        1,
        Some(0.0),
        Fixed {
            name: "test:zero-derivatives",
            grad: 0.0,
            hess: 0.0,
            fail: false,
        },
    );
    let model = train(&source(), &Y, &[], &c, &mut [], &Local).unwrap();
    assert_eq!(model.trees.len(), 1);
}

#[test]
fn overflow_after_applying_row_weight_is_rejected() {
    let weights = [2.0; 12];
    let c = custom_config(
        1,
        Some(0.0),
        Fixed {
            name: "test:overflow",
            grad: f32::MAX,
            hess: 1.0,
            fail: false,
        },
    );
    assert!(train_weighted(&source(), &Y, Some(&weights), &[], &c, &mut [], &Local).is_err());
}

#[test]
fn marker_round_trips_and_continuation_requires_matching_provider() {
    let prior = custom_model(2, Some(0.0), Square::new("test:square-v1"));
    let json = prior.to_xgboost_json().unwrap();
    let doc: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        doc["learner"]["attributes"]["ferrisboost_custom_objective"],
        "test:square-v1"
    );
    let loaded = Model::from_xgboost_json(&json).unwrap();
    assert_eq!(loaded.to_xgboost_json().unwrap(), json);

    let mut blank_marker = doc.clone();
    blank_marker["learner"]["attributes"]["ferrisboost_custom_objective"] =
        serde_json::Value::String(String::new());
    let error = Model::from_xgboost_json(&blank_marker.to_string()).unwrap_err();
    assert!(
        error.to_string().contains("name must not be empty"),
        "{error}"
    );

    let margins: Vec<f32> = X.iter().map(|&x| prior.predict_margin(&[x])).collect();
    let continue_with = |objective: Option<Box<dyn CustomObjective>>| {
        let mut c = config(1, Some(prior.base_score));
        c.base_margin = Some(margins.clone());
        c.prior_model = Some(prior.clone());
        c.custom_objective = objective;
        train(&source(), &Y, &[], &c, &mut [], &Local)
    };
    assert!(continue_with(None)
        .unwrap_err()
        .to_string()
        .to_lowercase()
        .contains("custom objective"));
    assert!(continue_with(Some(Box::new(Square::new("test:square-v2"))))
        .unwrap_err()
        .to_string()
        .to_lowercase()
        .contains("custom objective"));
    let joined = continue_with(Some(Box::new(Square::new("test:square-v1")))).unwrap();
    assert_eq!(joined.trees.len(), prior.trees.len() + 1);
    let uninterrupted = custom_model(3, Some(0.0), Square::new("test:square-v1"));
    assert_eq!(
        joined.to_xgboost_json().unwrap(),
        uninterrupted.to_xgboost_json().unwrap()
    );
    assert_eq!(
        prior.slice_rounds(0, 1).custom_objective,
        prior.custom_objective
    );
    assert_eq!(prior.class_view(0).custom_objective, prior.custom_objective);
}

#[test]
fn adding_custom_objective_to_builtin_prior_is_rejected() {
    let src = source();
    let prior = train(&src, &Y, &[], &config(1, Some(0.0)), &mut [], &Local).unwrap();
    let margins: Vec<f32> = X.iter().map(|&x| prior.predict_margin(&[x])).collect();
    let mut c = custom_config(1, Some(prior.base_score), Square::new("test:square"));
    c.base_margin = Some(margins);
    c.prior_model = Some(prior);
    assert!(train(&src, &Y, &[], &c, &mut [], &Local)
        .unwrap_err()
        .to_string()
        .to_lowercase()
        .contains("custom objective"));
}

#[test]
fn other_objective_and_scale_pos_weight_are_rejected() {
    let src = source();
    let mut logistic = TrainConfig::new(params(1), Objective::Logistic);
    logistic.base_score = Some(0.5);
    logistic.custom_objective = Some(Box::new(Square::new("test:square")));
    assert!(train(&src, &Y, &[], &logistic, &mut [], &Local).is_err());

    let mut scaled = custom_config(1, Some(0.0), Square::new("test:square"));
    scaled.params.scale_pos_weight = 2.0;
    assert!(train(&src, &Y, &[], &scaled, &mut [], &Local).is_err());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn exact_gpu_matches_cpu_for_custom_objective_over_multiple_rounds() {
    use ferrisboost::types::GpuMath;
    let src = source();
    let mut c = custom_config(3, Some(0.25), Square::new("test:square"));
    c.params.gpu_math = GpuMath::Exact;
    let cpu = train(&src, &Y, &[], &c, &mut [], &Local).unwrap();
    let gpu = train_gpu(0, &src, &Y, &[], &c, &mut [], &Local).unwrap();
    assert_eq!(
        cpu.to_xgboost_json().unwrap(),
        gpu.to_xgboost_json().unwrap()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn fast_gpu_rejects_custom_objective_before_training() {
    use ferrisboost::types::GpuMath;
    let mut c = custom_config(1, Some(0.0), Square::new("test:square"));
    c.params.gpu_math = GpuMath::Fast;
    let error = train_gpu(0, &source(), &Y, &[], &c, &mut [], &Local).unwrap_err();
    assert!(error
        .to_string()
        .to_lowercase()
        .contains("custom_objective"));
}
