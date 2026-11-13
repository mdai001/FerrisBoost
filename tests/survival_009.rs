use ferrisboost::{
    metrics,
    survival::{self, AftDistribution, CoxState},
    types::Objective,
};

fn cox_loss_unscaled(margins: &[f64], labels: &[f32], weights: &[f32]) -> f64 {
    let mut times: Vec<f32> = labels
        .iter()
        .filter(|&&label| label > 0.0)
        .map(|&label| label.abs())
        .collect();
    times.sort_by(f32::total_cmp);
    times.dedup_by(|a, b| a.to_bits() == b.to_bits());
    let mut loss = 0.0;
    for time in times {
        let mut event_mass = 0.0f64;
        let mut risk = 0.0f64;
        let mut event_log_risk = 0.0f64;
        for ((&margin, &label), &weight) in margins.iter().zip(labels).zip(weights) {
            if label.abs() >= time {
                risk += f64::from(weight) * margin.exp();
            }
            if label > 0.0 && label == time {
                event_mass += f64::from(weight);
                event_log_risk += f64::from(weight) * margin;
            }
        }
        loss += event_mass * risk.ln() - event_log_risk;
    }
    loss
}

#[test]
fn cox_breslow_gradient_hessian_match_independent_loss_differences() {
    let labels = [1.0f32, -1.5, 2.0, -3.0, 4.0];
    let margins = [-0.4f32, 0.2, 0.7, -0.3, 0.5];
    let weights = [0.5f32, 1.25, 2.0, 0.75, 1.5];
    let mut state = CoxState::new(&labels).unwrap();
    let pairs = state.gradients(&margins, &labels, Some(&weights)).unwrap();
    let margins64: Vec<f64> = margins.iter().map(|&margin| f64::from(margin)).collect();
    let event_weight: f64 = labels
        .iter()
        .zip(&weights)
        .filter(|(label, _)| **label > 0.0)
        .map(|(_, &weight)| f64::from(weight))
        .sum();
    let normalized = state.loss(&margins, &labels, Some(&weights)).unwrap();
    assert!(
        (normalized * event_weight - cox_loss_unscaled(&margins64, &labels, &weights)).abs()
            < 1e-12
    );

    let step = 1e-3f64;
    for row in 0..labels.len() {
        let mut plus = margins64.clone();
        let mut minus = margins64.clone();
        plus[row] += step;
        minus[row] -= step;
        let f_plus = cox_loss_unscaled(&plus, &labels, &weights);
        let f_center = cox_loss_unscaled(&margins64, &labels, &weights);
        let f_minus = cox_loss_unscaled(&minus, &labels, &weights);
        let grad = (f_plus - f_minus) / (2.0 * step);
        let hess = (f_plus - 2.0 * f_center + f_minus) / (step * step);
        assert!((f64::from(pairs[row].grad) - grad).abs() < 2e-4);
        assert!((f64::from(pairs[row].hess) - hess).abs() < 2e-3);
    }
}

#[test]
fn cox_ties_use_grouped_breslow_and_are_permutation_and_shift_invariant() {
    let labels = [2.0f32, 2.0, -2.0, 4.0, -5.0];
    let margins = [-0.25f32, 0.375, 0.75, -0.5, 0.125];
    let weights = [1.0f32, 0.5, 1.5, 2.0, 0.75];
    let mut state = CoxState::new(&labels).unwrap();
    let got_loss = state.loss(&margins, &labels, Some(&weights)).unwrap();
    let event_weight: f64 = labels
        .iter()
        .zip(&weights)
        .filter(|(label, _)| **label > 0.0)
        .map(|(_, &weight)| f64::from(weight))
        .sum();
    let margins64: Vec<f64> = margins.iter().map(|&margin| f64::from(margin)).collect();
    assert!(
        (got_loss * event_weight - cox_loss_unscaled(&margins64, &labels, &weights)).abs() < 1e-12
    );

    let pairs = state.gradients(&margins, &labels, Some(&weights)).unwrap();
    let shifted: Vec<f32> = margins.iter().map(|&margin| margin + 3.25).collect();
    let shifted_pairs = state.gradients(&shifted, &labels, Some(&weights)).unwrap();
    let shifted_loss = state.loss(&shifted, &labels, Some(&weights)).unwrap();
    assert!((got_loss - shifted_loss).abs() < 1e-12);
    for (a, b) in pairs.iter().zip(&shifted_pairs) {
        assert!((a.grad - b.grad).abs() < 2e-7);
        assert!((a.hess - b.hess).abs() < 2e-7);
    }

    let order = [3usize, 0, 4, 2, 1];
    let permute = |values: &[f32]| order.map(|i| values[i]);
    let permuted_labels = permute(&labels);
    let permuted_margins = permute(&margins);
    let permuted_weights = permute(&weights);
    let mut permuted_state = CoxState::new(&permuted_labels).unwrap();
    let permuted_pairs = permuted_state
        .gradients(&permuted_margins, &permuted_labels, Some(&permuted_weights))
        .unwrap();
    let permuted_loss = permuted_state
        .loss(&permuted_margins, &permuted_labels, Some(&permuted_weights))
        .unwrap();
    assert!((got_loss - permuted_loss).abs() < 1e-12);
    for (new_row, &old_row) in order.iter().enumerate() {
        assert!((pairs[old_row].grad - permuted_pairs[new_row].grad).abs() < 2e-7);
        assert!((pairs[old_row].hess - permuted_pairs[new_row].hess).abs() < 2e-7);
    }
}

#[test]
fn cox_all_censored_and_zero_weight_inputs_have_zero_loss_and_gradients() {
    let margins = [-1.0f32, 0.0, 1.0];
    let censored = [-1.0f32, -2.0, -3.0];
    let mut state = CoxState::new(&censored).unwrap();
    assert_eq!(state.loss(&margins, &censored, None).unwrap(), 0.0);
    assert!(state
        .gradients(&margins, &censored, None)
        .unwrap()
        .iter()
        .all(|p| p.grad == 0.0 && p.hess == 0.0));

    let labels = [1.0f32, -2.0, 3.0];
    let weights = [0.0f32; 3];
    let mut state = CoxState::new(&labels).unwrap();
    assert_eq!(state.loss(&margins, &labels, Some(&weights)).unwrap(), 0.0);
    assert!(state
        .gradients(&margins, &labels, Some(&weights))
        .unwrap()
        .iter()
        .all(|p| p.grad == 0.0 && p.hess == 0.0));
}

fn aft_cdf(distribution: AftDistribution, z: f64) -> f64 {
    match distribution {
        AftDistribution::Normal => 0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2)),
        AftDistribution::Logistic => 1.0 / (1.0 + (-z).exp()),
        AftDistribution::Extreme => 1.0 - (-z.exp()).exp(),
    }
}

fn aft_pdf(distribution: AftDistribution, z: f64) -> f64 {
    match distribution {
        AftDistribution::Normal => (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt(),
        AftDistribution::Logistic => {
            let cdf = aft_cdf(distribution, z);
            cdf * (1.0 - cdf)
        }
        AftDistribution::Extreme => (z - z.exp()).exp(),
    }
}

fn aft_independent_loss(
    distribution: AftDistribution,
    scale: f64,
    margin: f64,
    lower: f64,
    upper: f64,
) -> f64 {
    if lower == upper {
        let z = (lower.ln() - margin) / scale;
        -(aft_pdf(distribution, z) / (scale * lower)).ln()
    } else {
        let cdf_upper = if upper.is_infinite() {
            1.0
        } else {
            aft_cdf(distribution, (upper.ln() - margin) / scale)
        };
        let cdf_lower = if lower == 0.0 {
            0.0
        } else {
            aft_cdf(distribution, (lower.ln() - margin) / scale)
        };
        -(cdf_upper - cdf_lower).ln()
    }
}

#[test]
fn aft_exact_and_interval_loss_gradients_match_independent_finite_differences() {
    let margin = 0.2f32;
    let scale = 0.8f32;
    for distribution in [
        AftDistribution::Normal,
        AftDistribution::Logistic,
        AftDistribution::Extreme,
    ] {
        let objective = Objective::Aft {
            distribution,
            scale,
        };
        assert_eq!(metrics::default_metric(objective).name(), "aft-nloglik");
        assert!(!objective.transform_is_identity());
        assert!((objective.transform(margin) - margin.exp()).abs() < 2e-7);
        assert!((objective.base_margin(0.5) - 0.5f32.ln()).abs() < 2e-7);

        for (lower, upper) in [
            (1.7f32, 1.7f32),
            (0.5, 2.0),
            (0.0, 1.4),
            (1.0, f32::INFINITY),
        ] {
            let (loss, pair) =
                survival::aft_loss_and_gradient(distribution, scale, margin, lower, upper);
            let step = 1e-3f64;
            let center = aft_independent_loss(
                distribution,
                f64::from(scale),
                f64::from(margin),
                f64::from(lower),
                f64::from(upper),
            );
            let plus = aft_independent_loss(
                distribution,
                f64::from(scale),
                f64::from(margin) + step,
                f64::from(lower),
                f64::from(upper),
            );
            let minus = aft_independent_loss(
                distribution,
                f64::from(scale),
                f64::from(margin) - step,
                f64::from(lower),
                f64::from(upper),
            );
            let grad = (plus - minus) / (2.0 * step);
            let hess = (plus - 2.0 * center + minus) / (step * step);
            assert!((loss - center).abs() < 1e-8);
            assert!((f64::from(pair.grad) - grad).abs() < 2e-4);
            assert!((f64::from(pair.hess) - hess).abs() < 2e-3);
        }
    }
}

#[test]
fn aft_bounds_validation_rejects_invalid_ranges_and_nonpositive_exact_time() {
    assert!(survival::validate_bounds(&[0.0, 1.0], &[f32::INFINITY, 1.0], 2).is_ok());
    assert!(survival::validate_bounds(&[2.0], &[f32::INFINITY], 1).is_ok());
    for (lower, upper) in [
        (vec![-0.1], vec![1.0]),
        (vec![f32::NAN], vec![1.0]),
        (vec![0.0], vec![0.0]),
        (vec![2.0], vec![1.0]),
        (vec![0.0], vec![f32::NAN]),
        (vec![0.0], vec![-1.0]),
    ] {
        assert!(survival::validate_bounds(&lower, &upper, 1).is_err());
    }
    assert!(survival::validate_bounds(&[0.0], &[1.0], 2).is_err());
}

#[test]
fn survival_objectives_preserve_names_link_functions_and_default_metrics() {
    let cox = Objective::Cox;
    assert_eq!(cox.name(), "survival:cox");
    assert_eq!(cox.transform(0.0), 1.0);
    assert_eq!(cox.base_margin(1.0), 0.0);
    assert_eq!(Objective::from_name(cox.name()), Some(cox));
    assert_eq!(metrics::default_metric(cox).name(), "cox-nloglik");

    for distribution in [
        AftDistribution::Normal,
        AftDistribution::Logistic,
        AftDistribution::Extreme,
    ] {
        let aft = Objective::Aft {
            distribution,
            scale: 1.0,
        };
        assert_eq!(aft.name(), "survival:aft");
        assert_eq!(
            Objective::from_json(aft.name(), &aft.json_params(1.0)),
            Some(aft)
        );
    }
}

#[test]
fn cox_centered_risk_arithmetic_is_invariant_to_huge_common_margin_shifts() {
    let labels = [1.0f32, -1.5, 2.0, -3.0, 4.0];
    let weights = [0.5f32, 1.25, 2.0, 0.75, 1.5];
    let centered = [0.0f32; 5];
    let huge = [1.0e30f32; 5];
    let mut centered_state = CoxState::new(&labels).unwrap();
    let mut huge_state = CoxState::new(&labels).unwrap();
    let centered_pairs = centered_state
        .gradients(&centered, &labels, Some(&weights))
        .unwrap();
    let huge_pairs = huge_state
        .gradients(&huge, &labels, Some(&weights))
        .unwrap();
    assert_eq!(
        centered_state
            .loss(&centered, &labels, Some(&weights))
            .unwrap(),
        huge_state.loss(&huge, &labels, Some(&weights)).unwrap()
    );
    for (a, b) in centered_pairs.iter().zip(&huge_pairs) {
        assert!((a.grad - b.grad).abs() < 1e-6);
        assert!((a.hess - b.hess).abs() < 1e-6);
    }
}

#[test]
fn cox_zero_weight_rows_match_removing_those_rows() {
    let labels = [1.0f32, -1.5, 2.0, -3.0, 4.0, -5.0];
    let margins = [-0.3f32, 1.0e30, 0.7, -0.1, 1.0e30, 0.9];
    let weights = [1.0f32, 0.0, 0.75, 1.25, 0.0, 2.0];
    let keep = [0usize, 2, 3, 5];
    let kept_labels = keep.map(|i| labels[i]);
    let kept_margins = keep.map(|i| margins[i]);
    let kept_weights = keep.map(|i| weights[i]);
    let mut zeroed_state = CoxState::new(&labels).unwrap();
    let mut kept_state = CoxState::new(&kept_labels).unwrap();
    let zeroed = zeroed_state
        .gradients(&margins, &labels, Some(&weights))
        .unwrap();
    let kept = kept_state
        .gradients(&kept_margins, &kept_labels, Some(&kept_weights))
        .unwrap();
    let zeroed_loss = zeroed_state
        .loss(&margins, &labels, Some(&weights))
        .unwrap();
    let kept_loss = kept_state
        .loss(&kept_margins, &kept_labels, Some(&kept_weights))
        .unwrap();
    assert!((zeroed_loss - kept_loss).abs() < 1e-12);
    for (i, &row) in keep.iter().enumerate() {
        assert!((zeroed[row].grad - kept[i].grad).abs() < 1e-7);
        assert!((zeroed[row].hess - kept[i].hess).abs() < 1e-7);
    }
}

#[test]
fn aft_extreme_scales_margins_and_interval_shapes_keep_bounded_derivatives() {
    let bounds = [
        (1.0f32, 1.0f32),
        (0.0, 1.5),
        (0.8, f32::INFINITY),
        (1.0, f32::from_bits(1.0f32.to_bits() + 1)),
        (0.0, f32::INFINITY),
    ];
    for distribution in [
        AftDistribution::Normal,
        AftDistribution::Logistic,
        AftDistribution::Extreme,
    ] {
        for scale in [1e-20f32, 1.0, 1e20] {
            for margin in [-1e20f32, -100.0, 0.0, 100.0, 1e20] {
                for (lower, upper) in bounds {
                    let (loss, pair) =
                        survival::aft_loss_and_gradient(distribution, scale, margin, lower, upper);
                    assert!(
                        loss.is_finite() && loss <= 27.631_022,
                        "invalid loss: distribution={distribution:?}, scale={scale}, margin={margin}, bounds=({lower}, {upper}), loss={loss}"
                    );
                    if lower != upper {
                        assert!(
                            loss >= -1e-12,
                            "interval NLL must be nonnegative: distribution={distribution:?}, scale={scale}, margin={margin}, bounds=({lower}, {upper}), loss={loss}"
                        );
                    }
                    assert!(
                        pair.grad.is_finite() && pair.grad.abs() <= 15.0,
                        "invalid gradient: distribution={distribution:?}, scale={scale}, margin={margin}, bounds=({lower}, {upper}), grad={} ",
                        pair.grad
                    );
                    assert!(
                        pair.hess.is_finite() && (1e-16..=15.0).contains(&pair.hess),
                        "invalid Hessian: distribution={distribution:?}, scale={scale}, margin={margin}, bounds=({lower}, {upper}), hess={}",
                        pair.hess
                    );
                    if lower == 0.0 && upper.is_infinite() {
                        assert_eq!(loss, 0.0);
                        assert_eq!(pair.grad, 0.0);
                    }
                }
            }
        }
    }
}

#[test]
fn aft_distribution_and_scale_json_round_trip_and_invalid_scales_are_rejected() {
    for distribution in [
        AftDistribution::Normal,
        AftDistribution::Logistic,
        AftDistribution::Extreme,
    ] {
        for scale in [0.25f32, 1.0, 2.5] {
            let objective = Objective::Aft {
                distribution,
                scale,
            };
            assert_eq!(
                Objective::from_json(objective.name(), &objective.json_params(1.0)),
                Some(objective)
            );
        }
    }
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(Objective::Aft {
            distribution: AftDistribution::Normal,
            scale,
        }
        .validate_params()
        .is_err());
    }
}
