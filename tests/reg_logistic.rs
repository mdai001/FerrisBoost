use ferrisboost::{metrics, types::Objective};

#[test]
fn fractional_label_derivatives_match_independent_loss_differences() {
    // Bernoulli cross entropy in margin space: softplus(z) - y*z.
    let loss = |z: f64, y: f64| z.exp().ln_1p() - y * z;
    let step = 1e-3;
    for margin in [-3.0f32, -0.5, 0.0, 1.25, 3.0] {
        for label in [0.0f32, 0.2, 0.75, 1.0] {
            let z = f64::from(margin);
            let y = f64::from(label);
            let center = loss(z, y);
            let plus = loss(z + step, y);
            let minus = loss(z - step, y);
            let expected_grad = (plus - minus) / (2.0 * step);
            let expected_hess = (plus - 2.0 * center + minus) / (step * step);
            let pair = Objective::LogisticRegression.gradient(margin, label);
            assert!((f64::from(pair.grad) - expected_grad).abs() < 2e-7);
            assert!((f64::from(pair.hess) - expected_hess).abs() < 2e-7);
        }
    }
}

#[test]
fn identity_default_metric_and_domain_are_probability_regression() {
    let obj = Objective::LogisticRegression;
    assert_eq!(obj.name(), "reg:logistic");
    assert_eq!(Objective::from_name(obj.name()), Some(obj));
    assert_eq!(
        Objective::from_json(obj.name(), &obj.json_params(2.0)),
        Some(obj)
    );
    assert_eq!(metrics::default_metric(obj).name(), "rmse");
    assert_eq!(
        metrics::default_metric(Objective::Logistic).name(),
        "logloss"
    );
    assert_eq!(obj.device_code(), Objective::Logistic.device_code());
    assert!(obj
        .validate_labels(&[0.0, 0.25, 0.75, 1.0], "train")
        .is_ok());
    for value in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(obj.validate_labels(&[value], "train").is_err());
        assert!(obj.validate_base_score(value).is_err());
    }
    for score in [0.0, 0.2, 0.8, 1.0] {
        obj.validate_base_score(score).unwrap();
        assert!(obj.base_margin(score).is_finite());
        assert!((obj.transform(obj.base_margin(score)) - score).abs() < 1e-5);
    }
}
