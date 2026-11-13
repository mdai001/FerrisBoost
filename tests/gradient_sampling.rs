use anyhow::Result;
use ferrisboost::{
    comm::Local,
    custom_objective::CustomObjective,
    source::DenseSource,
    train::{train, train_weighted, TrainConfig},
    types::{GradPair, Objective, SamplingMethod, TrainParams},
    Model,
};
use std::sync::{Arc, Mutex};

const ROWS: usize = 96;
const FEATURES: usize = 3;

fn fixture() -> (DenseSource, Vec<f32>, Vec<f32>, Vec<f32>) {
    let data = (0..ROWS)
        .flat_map(|row| {
            (0..FEATURES).map(move |feature| {
                if (row * 17 + feature * 29) % 71 == 0 {
                    f32::NAN
                } else {
                    ((row * (feature * 2 + 7) + feature * 13) % 101) as f32 / 101.0
                }
            })
        })
        .collect::<Vec<_>>();
    let labels = (0..ROWS)
        .map(|row| {
            let x0 = ((row * 7) % 23) as f32 / 23.0;
            let x1 = ((row * 11 + 3) % 19) as f32 / 19.0;
            if (row * 13 + row / 3) % 31 < 14 {
                0.75 + x0 - 0.2 * x1
            } else {
                -0.4 + 0.3 * x0 + x1
            }
        })
        .collect::<Vec<_>>();
    let weights = (0..ROWS)
        .map(|row| [0.25, 0.5, 1.0, 1.5, 2.0][(row * 7 + row / 5) % 5])
        .collect::<Vec<_>>();
    let source = DenseSource::from_row_major(&data, ROWS, FEATURES, 16, 1);
    (source, data, labels, weights)
}

fn params(rounds: usize, sampling_method: SamplingMethod) -> TrainParams {
    TrainParams {
        n_rounds: rounds,
        max_depth: 3,
        max_bin: 16,
        learning_rate: 0.25,
        lambda: 1.0,
        min_child_weight: 0.0,
        subsample: 0.3,
        sampling_method,
        seed: 41,
        nthread: 1,
        ..Default::default()
    }
}

fn config(rounds: usize, sampling_method: SamplingMethod) -> TrainConfig {
    let mut config = TrainConfig::new(params(rounds, sampling_method), Objective::SquaredError);
    config.base_score = Some(0.1);
    config.quiet = true;
    config
}

fn model_bytes(model: &Model) -> Vec<u8> {
    serde_json::to_vec(model).unwrap()
}

#[test]
fn explicit_uniform_is_the_default_and_gradient_full_rate_is_uniform() {
    let (source, _, labels, weights) = fixture();
    let mut default_params = params(4, SamplingMethod::Uniform);
    default_params.sampling_method = SamplingMethod::default();
    let explicit_params = params(4, SamplingMethod::Uniform);
    let default_config = TrainConfig {
        params: default_params,
        ..config(4, SamplingMethod::Uniform)
    };
    let explicit_uniform = TrainConfig {
        params: explicit_params,
        ..config(4, SamplingMethod::Uniform)
    };
    let default_model = train_weighted(
        &source,
        &labels,
        Some(&weights),
        &[],
        &default_config,
        &mut [],
        &Local,
    )
    .unwrap();
    let explicit_model = train_weighted(
        &source,
        &labels,
        Some(&weights),
        &[],
        &explicit_uniform,
        &mut [],
        &Local,
    )
    .unwrap();
    assert_eq!(model_bytes(&default_model), model_bytes(&explicit_model));

    for (subsample, seed) in [(1.0, 41), (1.0, 991), (1.0, 7)] {
        let mut uniform = params(4, SamplingMethod::Uniform);
        uniform.subsample = subsample;
        uniform.seed = seed;
        let mut gradient = uniform.clone();
        gradient.sampling_method = SamplingMethod::GradientBased;
        let mut uniform_config = config(4, SamplingMethod::Uniform);
        uniform_config.params = uniform;
        let mut gradient_config = config(4, SamplingMethod::GradientBased);
        gradient_config.params = gradient;
        let uniform_model = train_weighted(
            &source,
            &labels,
            Some(&weights),
            &[],
            &uniform_config,
            &mut [],
            &Local,
        )
        .unwrap();
        let gradient_model = train_weighted(
            &source,
            &labels,
            Some(&weights),
            &[],
            &gradient_config,
            &mut [],
            &Local,
        )
        .unwrap();
        assert_eq!(
            model_bytes(&gradient_model),
            model_bytes(&uniform_model),
            "full-rate or weighted sampling mismatch at subsample={subsample}, seed={seed}"
        );
    }
}

#[test]
fn gradient_sampling_is_seeded_and_independent_of_thread_and_block_layout() {
    let (source, _, labels, weights) = fixture();
    let run = |seed, nthread, cols_per_block| {
        let mut config = config(5, SamplingMethod::GradientBased);
        config.params.seed = seed;
        config.params.nthread = nthread;
        config.params.cols_per_block = cols_per_block;
        train_weighted(
            &source,
            &labels,
            Some(&weights),
            &[],
            &config,
            &mut [],
            &Local,
        )
        .unwrap()
    };

    let baseline = run(123, 1, 1);
    let repeated = run(123, 1, 1);
    let parallel = run(123, 4, 2);
    let different_seed = run(124, 1, 1);
    assert_eq!(model_bytes(&baseline), model_bytes(&repeated));
    assert_eq!(model_bytes(&baseline), model_bytes(&parallel));
    assert_ne!(model_bytes(&baseline), model_bytes(&different_seed));
}

#[derive(Clone)]
struct CaptureSquare(Arc<Mutex<Vec<Vec<f32>>>>);

impl CustomObjective for CaptureSquare {
    fn name(&self) -> &str {
        "test:gradient-sampling-capture-square"
    }

    fn gradients(&self, margins: &[f32], labels: &[f32], out: &mut [GradPair]) -> Result<()> {
        self.0.lock().unwrap().push(margins.to_vec());
        for ((pair, &margin), &label) in out.iter_mut().zip(margins).zip(labels) {
            pair.grad = margin - label;
            pair.hess = 1.0;
        }
        Ok(())
    }
}

#[test]
fn custom_gradient_sampling_updates_margins_for_every_row_each_round() {
    let (source, data, labels, _) = fixture();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut config = config(5, SamplingMethod::GradientBased);
    config.custom_objective = Some(Box::new(CaptureSquare(Arc::clone(&calls))));
    let model = train(&source, &labels, &[], &config, &mut [], &Local).unwrap();
    let rounds = calls.lock().unwrap();
    assert_eq!(rounds.len(), 5);
    assert!(rounds.iter().all(|margins| margins.len() == ROWS));
    for (round, margins) in rounds.iter().enumerate() {
        for row in 0..ROWS {
            assert_eq!(
                margins[row].to_bits(),
                model
                    .predict_margin_upto(&data[row * FEATURES..(row + 1) * FEATURES], round)
                    .to_bits(),
                "callback round {round}, row {row}"
            );
        }
    }
    assert_ne!(rounds[0], rounds[1]);
}

#[test]
fn gradient_sampling_continuation_matches_uninterrupted_training() {
    let (source, data, labels, weights) = fixture();
    let mut prior_config = config(2, SamplingMethod::GradientBased);
    prior_config.custom_objective = Some(Box::new(CaptureSquare(Arc::new(Mutex::new(Vec::new())))));
    let prior = train_weighted(
        &source,
        &labels,
        Some(&weights),
        &[],
        &prior_config,
        &mut [],
        &Local,
    )
    .unwrap();
    let margins = (0..ROWS)
        .map(|row| prior.predict_margin(&data[row * FEATURES..(row + 1) * FEATURES]))
        .collect::<Vec<_>>();
    let mut continuation_config = config(2, SamplingMethod::GradientBased);
    continuation_config.custom_objective =
        Some(Box::new(CaptureSquare(Arc::new(Mutex::new(Vec::new())))));
    continuation_config.base_margin = Some(margins);
    continuation_config.prior_model = Some(prior);
    let continued = train_weighted(
        &source,
        &labels,
        Some(&weights),
        &[],
        &continuation_config,
        &mut [],
        &Local,
    )
    .unwrap();

    let mut full_config = config(4, SamplingMethod::GradientBased);
    full_config.custom_objective = Some(Box::new(CaptureSquare(Arc::new(Mutex::new(Vec::new())))));
    let uninterrupted = train_weighted(
        &source,
        &labels,
        Some(&weights),
        &[],
        &full_config,
        &mut [],
        &Local,
    )
    .unwrap();
    assert_eq!(continued.trees.len(), 4);
    assert_eq!(model_bytes(&continued), model_bytes(&uninterrupted));
}

#[test]
fn sampling_method_rejects_unknown_string() {
    let error = "gradient-biased"
        .parse::<SamplingMethod>()
        .expect_err("unknown sampling method string must be rejected");
    assert!(error.contains("sampling_method"), "{error}");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn fast_gpu_full_rate_gradient_sampling_is_uniform_and_repeatable() {
    use ferrisboost::{train::train_gpu, types::GpuMath};
    let (source, _, labels, _) = fixture();
    let mut gradient_config = config(4, SamplingMethod::GradientBased);
    gradient_config.params.subsample = 1.0;
    gradient_config.params.gpu_math = GpuMath::Fast;
    let mut uniform_config = config(4, SamplingMethod::Uniform);
    uniform_config.params.subsample = 1.0;
    uniform_config.params.gpu_math = GpuMath::Fast;

    let run = |config: &TrainConfig| {
        train_gpu(0, &source, &labels, &[], config, &mut [], &Local).unwrap()
    };
    let gradient = run(&gradient_config);
    let repeated = run(&gradient_config);
    let uniform = run(&uniform_config);
    assert_eq!(model_bytes(&gradient), model_bytes(&repeated));
    assert_eq!(model_bytes(&gradient), model_bytes(&uniform));
}

#[cfg(feature = "cuda")]
#[test]
fn one_tree_gpu_diagnostic_rejects_gradient_sampling() {
    use ferrisboost::train::train_gpu_one_tree;
    let (source, _, labels, _) = fixture();
    let config = config(1, SamplingMethod::GradientBased);
    let error = train_gpu_one_tree(0, &source, &labels, &config, &Local).unwrap_err();
    assert!(
        format!("{error:#}").contains("sampling_method"),
        "{error:#}"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and serialized GPU lock"]
fn exact_gpu_gradient_sampling_matches_weighted_cpu_for_quantizer_modes() {
    use ferrisboost::{train::train_gpu_weighted, types::GpuMath};

    // Keep the complete configuration matrix in one ignored test so device
    // comparisons remain serialized: host/device quantization × resident blocks.
    let (source, _, labels, weights) = fixture();
    let mut cpu_config = config(4, SamplingMethod::GradientBased);
    cpu_config.params.gpu_math = GpuMath::Exact;
    let cpu = train_weighted(
        &source,
        &labels,
        Some(&weights),
        &[],
        &cpu_config,
        &mut [],
        &Local,
    )
    .unwrap();
    for device_quantize in [true, false] {
        for resident_blocks in [0, 1] {
            let mut gpu_config = config(4, SamplingMethod::GradientBased);
            gpu_config.params.gpu_math = GpuMath::Exact;
            gpu_config.params.device_quantize = Some(device_quantize);
            gpu_config.params.resident_blocks = Some(resident_blocks);
            let run = || {
                train_gpu_weighted(
                    0,
                    &source,
                    &labels,
                    Some(&weights),
                    &[],
                    &gpu_config,
                    &mut [],
                    &Local,
                )
                .unwrap()
            };
            let gpu = run();
            let repeated = run();
            assert_eq!(
                model_bytes(&gpu),
                model_bytes(&cpu),
                "CPU/GPU model mismatch: device_quantize={device_quantize}, resident_blocks={resident_blocks}"
            );
            assert_eq!(
                model_bytes(&gpu),
                model_bytes(&repeated),
                "GPU repeat mismatch: device_quantize={device_quantize}, resident_blocks={resident_blocks}"
            );
        }
    }
}
