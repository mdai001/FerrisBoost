#![cfg(feature = "cuda")]
//! `gpu_math = "fast"` with `reg:quantileerror`: the per-round global scale is a
//! fixed-shape device reduction, so training is deterministic from run to run
//! (but not byte-identical to the CPU).

use ferrisboost::{
    backend::cuda::{GpuTrainCtx, SharedHistogramConfig},
    comm::Local,
    source::DenseSource,
    train::{train_gpu, train_gpu_weighted, TrainConfig},
    types::{GpuMath, GradPair, GradPairFixed, GradQuantizer, Objective, TrainParams},
};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / (1u64 << 24) as f32
}

/// Heteroscedastic continuous target so the quantiles differ from the mean.
fn fixture(n_rows: usize, n_feats: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut st = seed;
    let mut x = Vec::with_capacity(n_rows * n_feats);
    let mut y = Vec::with_capacity(n_rows);
    let mut w = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let row: Vec<f32> = (0..n_feats).map(|_| lcg(&mut st) * 4.0 - 2.0).collect();
        let noise = lcg(&mut st) + lcg(&mut st) - 1.0;
        let a = row[0];
        let b = row[1 % n_feats];
        let c = row[2 % n_feats];
        y.push(a * 1.5 + (b * 2.0).sin() + noise * (0.5 + c.abs()));
        w.push(0.25 + 1.75 * lcg(&mut st));
        x.extend(row);
    }
    (x, y, w)
}

fn quantile_params(math: GpuMath, rounds: usize) -> TrainParams {
    TrainParams {
        n_rounds: rounds,
        max_depth: 4,
        max_bin: 64,
        nthread: 1,
        gpu_math: math,
        ..Default::default()
    }
}

#[test]
fn fast_quantile_is_deterministic_and_calibrated() {
    let n_rows = 20_000usize;
    let n_feats = 5usize;
    let (x, y, w) = fixture(n_rows, n_feats, 7);
    let source = DenseSource::from_row_major(&x, n_rows, n_feats, 64, n_feats);
    let bytes = |m: &ferrisboost::tree::Model| serde_json::to_vec(m).unwrap();
    let coverage = |m: &ferrisboost::tree::Model| {
        let below = (0..n_rows)
            .filter(|&i| y[i] <= m.predict_margin(&x[i * n_feats..(i + 1) * n_feats]))
            .count();
        below as f32 / n_rows as f32
    };
    for alpha in [0.1f32, 0.5, 0.9] {
        for weighted in [false, true] {
            let what = format!("alpha={alpha} weighted={weighted}");
            let obj = Objective::Quantile { alpha };
            let weights = weighted.then_some(w.as_slice());
            let run = |math: GpuMath| {
                let cfg = TrainConfig::new(quantile_params(math, 25), obj);
                train_gpu_weighted(0, &source, &y, weights, &[], &cfg, &mut [], &Local).unwrap()
            };
            let fast = run(GpuMath::Fast);
            let fast_again = run(GpuMath::Fast);
            assert_eq!(
                bytes(&fast),
                bytes(&fast_again),
                "{what}: fast not deterministic"
            );
            let exact = run(GpuMath::Exact);
            let (cf, ce) = (coverage(&fast), coverage(&exact));
            assert!((cf - alpha).abs() < 0.06, "{what}: fast coverage {cf}");
            assert!((cf - ce).abs() < 0.03, "{what}: fast {cf} vs exact {ce}");
        }
    }
}

#[test]
fn fast_quantile_with_row_sampling_is_deterministic() {
    let n_rows = 6_000usize;
    let n_feats = 4usize;
    let (x, y, _) = fixture(n_rows, n_feats, 11);
    let source = DenseSource::from_row_major(&x, n_rows, n_feats, 32, n_feats);
    let run = || {
        let mut p = quantile_params(GpuMath::Fast, 6);
        p.subsample = 0.6;
        p.seed = 5;
        let cfg = TrainConfig::new(p, Objective::Quantile { alpha: 0.7 });
        let m = train_gpu(0, &source, &y, &[], &cfg, &mut [], &Local).unwrap();
        serde_json::to_vec(&m).unwrap()
    };
    assert_eq!(run(), run());
}

/// Round 0 of the device path (zero delta) against the host formulas. The device
/// scale sums in a different order than the host's serial f64 loop and uses CUDA
/// `tanhf`, so the quantized gradients agree to within rounding, not bit for bit.
#[test]
fn fast_quantile_first_round_matches_host_formulas() {
    // 100: fewer rows than threads; 70_001: several grid-stride iterations.
    for n_rows in [100usize, 70_001] {
        for weighted in [false, true] {
            let (_, y, w) = fixture(n_rows, 1, 3);
            let weights = weighted.then_some(w.as_slice());
            let alpha = 0.8f32;
            let obj = Objective::Quantile { alpha };
            let margin = obj.base_margin(obj.estimate_base_score(&y, weights));
            let config = SharedHistogramConfig {
                runtime_log: false,
                gpu_math: GpuMath::Fast,
                fast_weights: weighted,
                ..Default::default()
            };
            let ctx = GpuTrainCtx::new(0, n_rows, 1, 19, 1, 8, config, false).unwrap();
            ctx.init_fast_math(&y, weights, std::slice::from_ref(&margin))
                .unwrap();

            let preds = vec![margin; n_rows];
            let scale = Objective::quantile_scale(&preds, &y, weights);
            let host: Vec<GradPair> = (0..n_rows)
                .map(|i| {
                    let g = Objective::quantile_gradient(alpha, scale, preds[i], y[i]);
                    match weights {
                        Some(w) => GradPair {
                            grad: g.grad * w[i],
                            hess: g.hess * w[i],
                        },
                        None => g,
                    }
                })
                .collect();
            let hq = GradQuantizer::new(&host, n_rows);
            let host_sum = host.iter().fold(GradPairFixed::default(), |mut acc, &g| {
                let f = hq.to_fixed(g);
                acc.grad += f.grad;
                acc.hess += f.hess;
                acc
            });
            let (dq, dsum, _) = ctx
                .fast_gradient_and_quantize(obj, u64::MAX, 0, 0)
                .unwrap();
            let what = format!("n_rows={n_rows} weighted={weighted}");
            assert_eq!(dq.grad_scale(), hq.grad_scale(), "{what}: grad scale");
            assert_eq!(dq.hess_scale(), hq.hess_scale(), "{what}: hess scale");
            // Fixed-point units are huge (the scale is a large power of two), so
            // compare relatively: tanhf and the summation order differ by ~1e-7
            // per row, and the sums cancel to far less.
            let close = |a: i64, b: i64| (a - b).abs() as f64 <= 1e-6 * (b.abs() as f64) + 1.0;
            assert!(
                close(dsum.grad, host_sum.grad) && close(dsum.hess, host_sum.hess),
                "{what}: device {dsum:?} vs host {host_sum:?}"
            );
            // Repeating the call (delta is still zero) reproduces it exactly.
            let (dq2, dsum2, _) = ctx
                .fast_gradient_and_quantize(obj, u64::MAX, 0, 0)
                .unwrap();
            assert_eq!(dq2.grad_scale(), dq.grad_scale());
            assert_eq!(dsum2, dsum, "{what}: repeat differs");
        }
    }
}

/// The quantile reduction scratch is a train-level allocation charged in the
/// fast-math budget item; exact does not pay for it.
#[test]
fn fast_quantile_scratch_is_budgeted() {
    let n_rows = 5000usize;
    let ctx_for = |math: GpuMath| {
        let config = SharedHistogramConfig {
            runtime_log: false,
            gpu_math: math,
            ..Default::default()
        };
        GpuTrainCtx::new(0, n_rows, 4, 19, 1, 8, config, false).unwrap()
    };
    let fast = ctx_for(GpuMath::Fast);
    let exact = ctx_for(GpuMath::Exact);
    // 256 blocks x (sum, weight) + one scale, f64 each.
    let scratch = (256 * 2 + 1) * 8;
    assert_eq!(fast.budget().fast_math_bytes, n_rows * 8 + scratch);
    assert_eq!(exact.budget().fast_math_bytes, 0);
    assert_eq!(
        fast.budget().total() - exact.budget().total(),
        n_rows * 8 + scratch
    );
}
