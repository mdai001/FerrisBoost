use ferrisboost::callback::{Callback, RecordHistory, RoundMetrics};
use ferrisboost::metrics::{LogLoss, Metric, Rmse};
use ferrisboost::{
    comm::Local,
    source::DenseSource,
    train::{train, EvalSet, TrainConfig},
    types::{Objective, TrainParams},
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

const FEATURES: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
const LABELS: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

fn source() -> DenseSource {
    DenseSource::from_row_major(&FEATURES, FEATURES.len(), 1, 16, 1)
}

fn config(eval_metrics: Vec<Box<dyn Metric>>) -> TrainConfig {
    let params = TrainParams {
        n_rounds: 1,
        max_depth: 1,
        max_bin: 8,
        nthread: 1,
        ..Default::default()
    };
    TrainConfig {
        eval_metrics,
        quiet: true,
        base_score: Some(0.5),
        ..TrainConfig::new(params, Objective::Logistic)
    }
}

struct FailingMetric(&'static str);

impl Metric for FailingMetric {
    fn name(&self) -> &str {
        self.0
    }

    fn eval(&self, _pred: &[f32], _label: &[f32]) -> f32 {
        // Training must call the fallible hook, never this compatibility API.
        unreachable!("fallible metric must not be evaluated through eval")
    }

    fn try_eval_grouped(
        &self,
        _pred: &[f32],
        _label: &[f32],
        _weight: Option<&[f32]>,
        _group_ptr: Option<&[u32]>,
    ) -> anyhow::Result<f32> {
        anyhow::bail!("deliberate failure from {}", self.name())
    }
}

struct SuccessfulMetric;

impl Metric for SuccessfulMetric {
    fn name(&self) -> &str {
        "successful"
    }

    fn eval(&self, _pred: &[f32], _label: &[f32]) -> f32 {
        1.25
    }
}

struct CountCalls(Arc<AtomicUsize>);

impl Callback for CountCalls {
    fn after_iteration(&mut self, _metrics: &RoundMetrics) -> bool {
        self.0.fetch_add(1, Ordering::SeqCst);
        true
    }
}

#[test]
fn fallible_metric_error_returns_from_train_before_callbacks() {
    let src = source();
    let eval = EvalSet {
        name: "valid".into(),
        source: &src,
        labels: &LABELS,
        weights: None,
        group_ptr: None,
    };
    let cfg = config(vec![Box::new(FailingMetric("fails"))]);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut callbacks: Vec<Box<dyn Callback>> = vec![Box::new(CountCalls(Arc::clone(&calls)))];

    let error = train(&src, &LABELS, &[eval], &cfg, &mut callbacks, &Local).unwrap_err();

    assert!(error.to_string().contains("deliberate failure from fails"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn later_metric_failure_publishes_no_partial_history_or_callbacks() {
    let src = source();
    let eval = EvalSet {
        name: "valid".into(),
        source: &src,
        labels: &LABELS,
        weights: None,
        group_ptr: None,
    };
    let cfg = config(vec![
        Box::new(SuccessfulMetric),
        Box::new(FailingMetric("second metric")),
    ]);
    let recorder = RecordHistory::new();
    let history = recorder.handle();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut callbacks: Vec<Box<dyn Callback>> =
        vec![Box::new(recorder), Box::new(CountCalls(Arc::clone(&calls)))];

    let error = train(&src, &LABELS, &[eval], &cfg, &mut callbacks, &Local).unwrap_err();

    assert!(error
        .to_string()
        .contains("deliberate failure from second metric"));
    assert!(history.lock().unwrap().is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn builtin_metrics_keep_their_existing_successful_scores() {
    let src = source();
    let eval = EvalSet {
        name: "valid".into(),
        source: &src,
        labels: &LABELS,
        weights: None,
        group_ptr: None,
    };
    let cfg = config(vec![Box::new(LogLoss), Box::new(Rmse)]);
    let recorder = RecordHistory::new();
    let history = recorder.handle();
    let mut callbacks: Vec<Box<dyn Callback>> = vec![Box::new(recorder)];

    let model = train(&src, &LABELS, &[eval], &cfg, &mut callbacks, &Local).unwrap();

    let predictions: Vec<f32> = FEATURES
        .iter()
        .map(|&feature| model.predict_margin(&[feature]))
        .collect();
    let recorded = history.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].entries.len(), 2);
    assert_eq!(
        recorded[0].get("valid", "logloss").unwrap().to_bits(),
        LogLoss.eval(&predictions, &LABELS).to_bits()
    );
    assert_eq!(
        recorded[0].get("valid", "rmse").unwrap().to_bits(),
        Rmse.eval(&predictions, &LABELS).to_bits()
    );
}

#[test]
fn default_fallible_hook_delegates_to_existing_grouped_evaluation() {
    let metric = Rmse;
    let predictions = [0.5, 1.5, 2.5];
    let labels = [0.0, 2.0, 3.0];
    let weights = [1.0, 2.0, 3.0];

    let previous = metric.eval_grouped(&predictions, &labels, Some(&weights), None);
    let fallible = metric
        .try_eval_grouped(&predictions, &labels, Some(&weights), None)
        .unwrap();

    assert_eq!(fallible.to_bits(), previous.to_bits());
}
