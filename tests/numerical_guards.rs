use ferrisboost::{
    comm::Local,
    source::DenseSource,
    train::{train, TrainConfig},
    types::{GradPair, GradQuantizer, Objective, TrainParams},
};

fn source_and_labels() -> (DenseSource, Vec<f32>) {
    let data = [0.0f32, 1.0, 2.0, 3.0];
    let source = DenseSource::from_row_major(&data, 4, 1, 16, 1);
    (source, vec![1.0, 2.0, 3.0, 4.0])
}

fn config(objective: Objective, base_score: f32) -> TrainConfig {
    let params = TrainParams {
        n_rounds: 1,
        max_depth: 1,
        max_bin: 4,
        nthread: 1,
        min_child_weight: 0.0,
        ..Default::default()
    };
    let mut cfg = TrainConfig::new(params, objective);
    cfg.base_score = Some(base_score);
    cfg.quiet = true;
    cfg
}

fn gamma_case() -> (Objective, f32) {
    (Objective::Gamma, 1.0e-40)
}

fn poisson_case() -> (Objective, f32) {
    (
        Objective::Poisson {
            max_delta_step: 100.0,
        },
        1.0,
    )
}

#[test]
fn cpu_training_rejects_nonfinite_gamma_and_poisson_gradients() {
    let (source, labels) = source_and_labels();

    let (objective, base_score) = gamma_case();
    let err = train(
        &source,
        &labels,
        &[],
        &config(objective, base_score),
        &mut [],
        &Local,
    )
    .expect_err("tiny positive Gamma base_score must not panic or train on infinite gradients");
    assert!(
        format!("{err:#}").to_lowercase().contains("finite"),
        "{err:#}"
    );

    let (objective, base_score) = poisson_case();
    let err = train(
        &source,
        &labels,
        &[],
        &config(objective, base_score),
        &mut [],
        &Local,
    )
    .expect_err("large Poisson max_delta_step must reject Hessian overflow");
    assert!(
        format!("{err:#}").to_lowercase().contains("finite"),
        "{err:#}"
    );
}

#[test]
fn quantizer_rejects_nonfinite_pairs_and_invalid_bounds() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            GradQuantizer::try_new(
                &[GradPair {
                    grad: invalid,
                    hess: 1.0
                }],
                1
            )
            .is_err(),
            "nonfinite gradient {invalid:?} was accepted"
        );
        assert!(
            GradQuantizer::try_new(
                &[GradPair {
                    grad: 1.0,
                    hess: invalid
                }],
                1
            )
            .is_err(),
            "nonfinite Hessian {invalid:?} was accepted"
        );
    }

    for (grad, hess) in [
        (-1.0, 1.0),
        (1.0, -1.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, 1.0),
        (1.0, f64::NEG_INFINITY),
    ] {
        assert!(
            GradQuantizer::try_from_bounds(grad, hess, 4).is_err(),
            "invalid bounds ({grad:?}, {hess:?}) were accepted"
        );
    }
}

#[test]
fn finite_quantization_and_tiny_normal_values_keep_their_reference_behavior() {
    let pairs = [
        GradPair {
            grad: 1.5,
            hess: 0.125,
        },
        GradPair {
            grad: -2.0,
            hess: 0.25,
        },
    ];
    let q = GradQuantizer::try_new(&pairs, 4).expect("finite gradients should quantize");
    // Independent reference: floor(log2((2^62 / 4) / max_abs)) per channel.
    assert_eq!(q.grad_scale(), 576_460_752_303_423_488.0); // 2^59
    assert_eq!(q.hess_scale(), 4_611_686_018_427_387_904.0); // 2^62
    assert_eq!(q.to_fixed(pairs[0]).grad, 864_691_128_455_135_232);
    assert_eq!(q.to_fixed(pairs[0]).hess, 576_460_752_303_423_488);
    assert_eq!(q.to_fixed(pairs[1]).grad, -1_152_921_504_606_846_976);
    assert_eq!(q.to_fixed(pairs[1]).hess, 1_152_921_504_606_846_976);

    let tiny = GradPair {
        grad: f32::MIN_POSITIVE,
        hess: f32::MIN_POSITIVE,
    };
    let tiny_q = GradQuantizer::try_new(&[tiny], 1)
        .expect("finite positive normal values remain valid quantizer inputs");
    assert!(tiny_q.grad_scale().is_finite() && tiny_q.grad_scale() > 0.0);
    assert!(tiny_q.hess_scale().is_finite() && tiny_q.hess_scale() > 0.0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_training_rejects_gamma_and_poisson_overflow_for_all_math_paths() {
    use ferrisboost::{
        backend::cuda::{GpuTrainCtx, SharedHistogramConfig},
        subsample::{row_selected, threshold_for},
        train::train_gpu,
        types::{GpuMath, GradPair},
    };

    // Keep every CUDA attempt in this single test so GPU execution stays serialized.
    let (source, labels) = source_and_labels();
    let (gamma, gamma_base_score) = gamma_case();
    let (poisson, poisson_base_score) = poisson_case();
    for (objective, base_score, name) in [
        (gamma, gamma_base_score, "Gamma"),
        (poisson, poisson_base_score, "Poisson"),
    ] {
        for (math, device_quantize) in [
            (GpuMath::Exact, true),
            (GpuMath::Exact, false),
            (GpuMath::Fast, true),
        ] {
            let mut cfg = config(objective, base_score);
            cfg.params.gpu_math = math;
            cfg.params.device_quantize = Some(device_quantize);
            cfg.params.resident_blocks = Some(0);
            let result = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local);
            let err = result.expect_err(&format!(
                "CUDA accepted overflowing {name} gradients ({math:?}, device_quantize={device_quantize})"
            ));
            assert!(
                format!("{err:#}").to_lowercase().contains("finite"),
                "expected numerical finite-value error, got: {err:#}"
            );
        }
    }

    // The sampled upload validates selected rows while allowing invalid values
    // in rows that the device masks to zero before its max scan.
    let n_rows = 64usize;
    let threshold = threshold_for(0.5);
    let seed = 41u64;
    let tree = 0u64;
    let selected = (0..n_rows)
        .find(|&row| row_selected(row as u64, threshold, seed, tree))
        .expect("fixture must contain a selected row");
    let unselected = (0..n_rows)
        .find(|&row| !row_selected(row as u64, threshold, seed, tree))
        .expect("fixture must contain an unselected row");
    let context = GpuTrainCtx::new(
        0,
        n_rows,
        1,
        16,
        1,
        4,
        SharedHistogramConfig {
            runtime_log: false,
            device_quantize: Some(true),
            resident_blocks: Some(0),
            hist_streams: Some(1),
            ..Default::default()
        },
        false,
    )
    .expect("CUDA quantization context should initialize");

    let mut selected_invalid = vec![
        GradPair {
            grad: 1.0,
            hess: 1.0
        };
        n_rows
    ];
    selected_invalid[selected].grad = f32::NAN;
    let err = context
        .upload_and_quantize_gpair_sampled(&selected_invalid, threshold, seed, tree)
        .err()
        .expect("selected nonfinite gradient must be rejected");
    assert!(
        format!("{err:#}").to_lowercase().contains("finite"),
        "expected selected-row numerical error, got: {err:#}"
    );

    let mut unselected_invalid = vec![
        GradPair {
            grad: 1.0,
            hess: 1.0
        };
        n_rows
    ];
    unselected_invalid[unselected].grad = f32::NAN;
    context
        .upload_and_quantize_gpair_sampled(&unselected_invalid, threshold, seed, tree)
        .expect("unselected nonfinite gradient should be masked before bounds selection");
}
