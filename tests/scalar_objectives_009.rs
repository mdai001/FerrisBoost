use ferrisboost::{metrics, types::Objective};

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[test]
fn raw_logistic_gradient_and_identity_link_match_hand_cases() {
    let objective = Objective::LogisticRaw;
    assert_eq!(objective.name(), "binary:logitraw");
    assert_eq!(Objective::from_name(objective.name()), Some(objective));
    assert_eq!(metrics::default_metric(objective).name(), "logloss");
    assert!(objective.transform_is_identity());

    for (margin, label) in [(-2.0, 0.0), (-0.5, 1.0), (0.0, 0.25), (1.5, 0.75)] {
        let pair = objective.gradient(margin, label);
        let probability = sigmoid(margin);
        assert!((pair.grad - (probability - label)).abs() < 1e-7);
        assert!((pair.hess - probability * (1.0 - probability)).abs() < 1e-7);
        assert_eq!(objective.transform(margin), margin);
        assert_eq!(objective.base_margin(margin), margin);
    }
    assert!(objective
        .validate_labels(&[0.0, 0.25, 0.75, 1.0], "train")
        .is_ok());
    for label in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
        assert!(objective.validate_labels(&[label], "train").is_err());
    }
}

#[test]
fn squared_log_error_matches_independent_loss_differences() {
    let objective = Objective::SquaredLogError;
    assert_eq!(objective.name(), "reg:squaredlogerror");
    assert_eq!(Objective::from_name(objective.name()), Some(objective));
    assert_eq!(metrics::default_metric(objective).name(), "rmsle");
    assert!(objective.transform_is_identity());

    // Independent loss: 1/2 * (ln(1 + pred) - ln(1 + label))^2.
    // These points stay comfortably above the prediction floor and away from
    // the Hessian clamp, so central differences should recover both derivatives.
    let loss = |pred: f64, label: f64| {
        let delta = (1.0 + pred).ln() - (1.0 + label).ln();
        0.5 * delta * delta
    };
    let step = 1e-3;
    for (pred, label) in [(-0.4f32, 0.2f32), (0.1, 1.4), (0.8, 0.5), (2.0, 3.0)] {
        let p = f64::from(pred);
        let y = f64::from(label);
        let center = loss(p, y);
        let plus = loss(p + step, y);
        let minus = loss(p - step, y);
        let expected_grad = (plus - minus) / (2.0 * step);
        let expected_hess = (plus - 2.0 * center + minus) / (step * step);
        let pair = objective.gradient(pred, label);
        assert!((f64::from(pair.grad) - expected_grad).abs() < 2e-5);
        assert!((f64::from(pair.hess) - expected_hess).abs() < 2e-4);
        assert_eq!(objective.transform(pred), pred);
        assert_eq!(objective.base_margin(pred), pred);
    }
    assert!(objective
        .validate_labels(&[-0.999, 0.0, 2.0], "train")
        .is_ok());
    for label in [-1.0, -1.01, f32::NAN, f32::INFINITY] {
        assert!(objective.validate_labels(&[label], "train").is_err());
    }
}

#[test]
fn hinge_gradient_respects_margin_kinks_and_zero_hessian_floor() {
    let objective = Objective::Hinge;
    assert_eq!(objective.name(), "binary:hinge");
    assert_eq!(Objective::from_name(objective.name()), Some(objective));
    assert_eq!(metrics::default_metric(objective).name(), "error");
    assert!(!objective.transform_is_identity());

    let active_positive = objective.gradient(0.5, 1.0);
    assert_eq!(active_positive.grad, -1.0);
    assert_eq!(active_positive.hess, 1.0);
    let active_negative = objective.gradient(-0.5, 0.0);
    assert_eq!(active_negative.grad, 1.0);
    assert_eq!(active_negative.hess, 1.0);

    // Strictly below the unit margin is active; equality is already inactive.
    for (margin, label) in [(1.0, 1.0), (-1.0, 0.0), (2.0, 1.0), (-2.0, 0.0)] {
        let pair = objective.gradient(margin, label);
        assert_eq!(pair.grad, 0.0);
        assert_eq!(pair.hess, f32::MIN_POSITIVE);
    }
    assert_eq!(objective.transform(0.0), 0.0);
    assert_eq!(objective.transform(0.01), 1.0);
    assert_eq!(objective.transform(-0.01), 0.0);
    assert_eq!(objective.base_margin(-0.5), -0.5);
    assert!(objective.validate_labels(&[0.0, 1.0], "train").is_ok());
    for label in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
        assert!(objective.validate_labels(&[label], "train").is_err());
    }
}

#[test]
fn logitraw_threshold_metric_uses_probabilities() {
    let metric = ferrisboost::metrics::for_objective(
        "error@0.7",
        ferrisboost::types::Objective::LogisticRaw,
    )
    .unwrap();
    assert_eq!(metric.eval(&[0.8, 1.2], &[0.0, 1.0]), 0.0);
}
