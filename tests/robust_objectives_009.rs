use ferrisboost::{metrics, types::Objective};

#[test]
fn expectile_gradient_matches_independent_asymmetric_quadratic_loss() {
    let alpha = 0.3f32;
    let objective = Objective::Expectile { alpha };
    assert_eq!(objective.name(), "reg:expectileerror");
    assert_eq!(
        Objective::from_json(objective.name(), &objective.json_params(1.0)),
        Some(objective)
    );
    assert_eq!(metrics::default_metric(objective).name(), "expectile");
    assert!(objective.transform_is_identity());

    // Use half squared loss for the independent finite-difference oracle;
    // the reported default metric intentionally omits this half factor.
    let loss = |pred: f64, label: f64| {
        let weight = if pred >= label {
            1.0 - f64::from(alpha)
        } else {
            f64::from(alpha)
        };
        0.5 * weight * (pred - label).powi(2)
    };
    let step = 1e-3;
    for (pred, label) in [(-1.2f32, 0.4f32), (0.7, -0.3), (1.8, 0.2)] {
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
}

#[test]
fn absolute_error_uses_hand_computed_scale_and_curvature() {
    let objective = Objective::AbsoluteError;
    assert_eq!(objective.name(), "reg:absoluteerror");
    assert_eq!(Objective::from_name(objective.name()), Some(objective));
    assert_eq!(metrics::default_metric(objective).name(), "mae");
    assert!(objective.transform_is_identity());

    let preds = [0.0f32, 2.0, 1.0];
    let labels = [4.0f32, 1.0, 1.0];
    let weights = [3.0f32, 1.0, 2.0];
    // sum(w * sqrt(abs(p-y))) / sum(w) = (3*2 + 1*1 + 2*0)/6 = 7/6.
    // Delta is the square of that quantity.
    let delta = ((7.0f64 / 6.0).powi(2)) as f32;
    let scale = Objective::quantile_scale(&preds, &labels, Some(&weights));
    assert!((scale - delta).abs() < 1e-6);

    for (&pred, &label) in preds.iter().zip(&labels) {
        let residual = pred - label;
        let norm = delta.hypot(residual);
        let curvature = if norm > 0.0 { delta / norm } else { 1.0 };
        let pair = Objective::absolute_gradient(scale, pred, label);
        assert!((pair.grad - residual * curvature).abs() < 1e-7);
        assert!((pair.hess - curvature).abs() < 1e-7);
        assert_eq!(objective.transform(pred), pred);
        assert_eq!(objective.base_margin(pred), pred);
    }

    // An all-zero residual batch has scale zero; the helper's explicit
    // zero-norm branch gives a unit Hessian and zero gradient.
    assert_eq!(
        Objective::quantile_scale(&[2.0, 2.0], &[2.0, 2.0], None),
        0.0
    );
    let zero = Objective::absolute_gradient(0.0, 2.0, 2.0);
    assert_eq!(zero.grad, 0.0);
    assert_eq!(zero.hess, 1.0);
}
