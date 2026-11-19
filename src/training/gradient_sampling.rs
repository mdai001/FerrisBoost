//! Host importance sampling shared by CPU and CUDA exact.
//! Fixed exponent buckets bound a safeguarded scalar solve without sorting or
//! retaining row-sized score/probability arrays. Uniform sampling is separate.

use crate::types::GradPair;
const GRID: u64 = 1 << 53;
const SCORE_LAMBDA: f64 = 0.1;

#[derive(Clone, Copy, Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
}
#[derive(Clone, Copy, Default)]
struct Bucket {
    sum: Sum,
    count: u64,
}

#[inline]
fn score(pair: &GradPair) -> f64 {
    let g = f64::from(pair.grad);
    let h = f64::from(pair.hess);
    (g * g + SCORE_LAMBDA * h * h).sqrt()
}

fn threshold(pairs: &[GradPair], rate: f32) -> anyhow::Result<Option<f64>> {
    // 2048 * 24 bytes on supported 64-bit platforms, independent of row count.
    let mut buckets = vec![Bucket::default(); 2048];
    let mut positive = 0u64;
    let mut min_bin = 2047usize;
    let mut max_bin = 0usize;
    let mut maximum = 0.0f64;
    for (row, pair) in pairs.iter().enumerate() {
        crate::ensure_input!(pair.grad.is_finite() && pair.hess.is_finite() && pair.hess >= 0.0,
            "gradient_based sampling needs finite gradients and nonnegative finite Hessians (row {row})");
        let s = score(pair);
        if s == 0.0 {
            continue;
        }
        let bin = (s.to_bits() >> 52) as usize;
        buckets[bin].sum.add(s);
        buckets[bin].count += 1;
        positive += 1;
        min_bin = min_bin.min(bin);
        max_bin = max_bin.max(bin);
        maximum = maximum.max(s);
    }
    let target = f64::from(rate) * pairs.len() as f64;
    if positive == 0 || target >= positive as f64 {
        return Ok(None);
    }
    let mut total = Sum::default();
    for bucket in &buckets {
        total.add(bucket.sum.value);
    }
    let unconstrained = total.value / target;
    if unconstrained >= maximum {
        return Ok(Some(unconstrained));
    }

    // At a power-of-two boundary every bucket is entirely on one side,
    // so its sum/count gives the exact probability mass up to f64 rounding.
    let mut below = Sum::default();
    let mut above = positive;
    let mut lo = f64::from_bits((min_bin as u64) << 52);
    let mut hi = f64::from_bits(((max_bin + 1) as u64) << 52);
    for bin in min_bin..=max_bin + 1 {
        let boundary = f64::from_bits((bin as u64) << 52);
        let residual = (above as f64 - target) + below.value / boundary;
        if residual <= 0.0 {
            hi = boundary;
            break;
        }
        lo = boundary;
        below.add(buckets[bin].sum.value);
        above -= buckets[bin].count;
    }
    drop(buckets);
    // Start above the root: active-set Newton steps then descend toward it.
    // A valid Newton step need not be interrupted by bisection while converging.
    hi = hi.min(unconstrained);
    let mut tau = hi;
    for iteration in 0..128 {
        let mut unsaturated = Sum::default();
        let mut saturated = 0u64;
        for pair in pairs {
            let s = score(pair);
            if s >= tau {
                saturated += 1;
            } else {
                unsaturated.add(s);
            }
        }
        let residual = (saturated as f64 - target) + unsaturated.value / tau;
        if residual.abs() <= 16.0 * f64::EPSILON * target.max(1.0) {
            return Ok(Some(tau));
        }
        if residual > 0.0 {
            lo = tau;
        } else {
            hi = tau;
        }
        let midpoint = lo + (hi - lo) * 0.5;
        if midpoint <= lo || midpoint >= hi {
            return Ok(Some(midpoint));
        }
        let remaining = target - saturated as f64;
        let candidate = unsaturated.value / remaining;
        // Keep the alternating safeguard after the initial Newton steps so
        // pathological active sets still receive at least 60 bisections.
        tau = if (iteration < 8 || iteration % 2 == 0)
            && remaining > 0.0
            && candidate > lo
            && candidate < hi
        {
            candidate
        } else {
            midpoint
        };
    }
    Ok(Some(lo + (hi - lo) * 0.5))
}

pub(crate) fn sample(
    pairs: &mut [GradPair],
    rate: f32,
    seed: u64,
    tree: u64,
) -> anyhow::Result<usize> {
    crate::ensure_input!(
        rate.is_finite() && rate > 0.0 && rate <= 1.0,
        "subsample must be in (0, 1]"
    );
    if rate == 1.0 {
        return Ok(pairs.len());
    }
    let Some(tau) = threshold(pairs, rate)? else {
        return Ok(pairs
            .iter()
            .filter(|p| p.grad != 0.0 || p.hess != 0.0)
            .count());
    };
    let mut kept = 0usize;
    for (row, pair) in pairs.iter_mut().enumerate() {
        let s = score(pair);
        if s == 0.0 {
            continue;
        }
        let probability = (s / tau).min(1.0);
        // Use the actual discrete inclusion probability for reweighting.
        // The minimum slot preserves support even for extremely small scores.
        let cutoff = ((probability * GRID as f64) as u64).clamp(1, GRID);
        if crate::subsample::row_hash(row as u64, seed, tree) >> 11 < cutoff {
            if cutoff < GRID {
                let p = cutoff as f64 / GRID as f64;
                pair.grad = (f64::from(pair.grad) / p) as f32;
                pair.hess = (f64::from(pair.hess) / p) as f32;
                crate::ensure_input!(
                    pair.grad.is_finite() && pair.hess.is_finite(),
                    "gradient_based reweighting exceeds finite float32 range at row {row}"
                );
            }
            kept += 1;
        } else {
            *pair = GradPair {
                grad: 0.0,
                hess: 0.0,
            };
        }
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pair(g: f32) -> GradPair {
        GradPair { grad: g, hess: 0.0 }
    }

    fn sorted_oracle(pairs: &[GradPair], rate: f32) -> Option<f64> {
        let mut scores: Vec<f64> = pairs
            .iter()
            .map(score)
            .filter(|value| *value > 0.0)
            .collect();
        scores.sort_by(f64::total_cmp);
        let target = f64::from(rate) * pairs.len() as f64;
        if scores.is_empty() || target >= scores.len() as f64 {
            return None;
        }

        let mut prefix = Sum::default();
        for unsaturated in 1..=scores.len() {
            prefix.add(scores[unsaturated - 1]);
            let denominator = target - (scores.len() - unsaturated) as f64;
            if denominator <= 0.0 {
                continue;
            }
            let tau = prefix.value / denominator;
            if tau >= scores[unsaturated - 1]
                && (unsaturated == scores.len() || tau <= scores[unsaturated])
            {
                return Some(tau);
            }
        }
        None
    }

    fn assert_threshold_matches_sorted_oracle(pairs: &[GradPair], rate: f32, label: &str) {
        let target = f64::from(rate) * pairs.len() as f64;
        let expected = sorted_oracle(pairs, rate);
        let actual = threshold(pairs, rate).unwrap();
        match (expected, actual) {
            (None, None) => {}
            (Some(expected), Some(actual)) => {
                let scale = expected.abs().max(f64::MIN_POSITIVE);
                assert!(
                    (actual - expected).abs() <= 1e-11 * scale,
                    "{label}: rate={rate}, threshold {actual} != {expected}"
                );
                let mut mass = Sum::default();
                for pair in pairs {
                    mass.add((score(pair) / actual).min(1.0));
                }
                assert!(
                    (mass.value - target).abs() <= 1e-11 * target.max(1.0),
                    "{label}: rate={rate}, mass {} != target {target}",
                    mass.value
                );
            }
            (expected, actual) => panic!(
                "{label}: rate={rate}, oracle returned {expected:?}, threshold returned {actual:?}"
            ),
        }
    }

    #[derive(Clone, Copy)]
    struct TestRng(u64);
    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut value = self.0;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
            value ^ (value >> 31)
        }

        fn finite_pair(&mut self, row: usize) -> GradPair {
            if row % 11 == 0 {
                return GradPair {
                    grad: 0.0,
                    hess: 0.0,
                };
            }
            // Repeated patterns deliberately create ties alongside random magnitudes.
            if row % 7 == 0 {
                let tied = [
                    GradPair {
                        grad: 0.5,
                        hess: 0.25,
                    },
                    GradPair {
                        grad: -2.0,
                        hess: 1.0,
                    },
                    GradPair {
                        grad: 0.0,
                        hess: 0.75,
                    },
                ];
                return tied[(row / 7) % tied.len()];
            }
            let exponent = (self.next() % 261) as i32 - 140;
            let magnitude =
                2.0f32.powi(exponent) * (0.5 + (self.next() as u32 as f32 / u32::MAX as f32));
            let grad = if self.next() & 1 == 0 {
                magnitude
            } else {
                -magnitude
            };
            let hess = if self.next() & 3 == 0 {
                0.0
            } else {
                2.0f32.powi(exponent) * (self.next() as u32 as f32 / u32::MAX as f32)
            };
            GradPair { grad, hess }
        }
    }

    #[test]
    fn threshold_matches_sorted_oracle_for_deterministic_random_pairs() {
        let rates = [
            0.001f32,
            0.1,
            0.6,
            0.99,
            f32::from_bits(1.0f32.to_bits() - 1),
        ];
        for &n in &[0usize, 1, 2, 31, 257, 2049] {
            let mut rng = TestRng(0x0090_5eed ^ n as u64);
            let pairs: Vec<_> = (0..n).map(|row| rng.finite_pair(row)).collect();
            let positive_pairs: Vec<_> = pairs
                .iter()
                .copied()
                .filter(|pair| score(pair) > 0.0)
                .collect();
            for &rate in &rates {
                assert_threshold_matches_sorted_oracle(&pairs, rate, "deterministic random pairs");
                assert_threshold_matches_sorted_oracle(
                    &positive_pairs,
                    rate,
                    "deterministic random positive-only pairs",
                );
            }
        }
    }

    #[test]
    fn threshold_matches_sorted_oracle_for_float32_extremes_ties_and_adjacent_values() {
        let one = 1.0f32;
        let below_one = f32::from_bits(one.to_bits() - 1);
        let above_one = f32::from_bits(one.to_bits() + 1);
        let tiny = f32::from_bits(1);
        let pairs = [
            GradPair {
                grad: 0.0,
                hess: 0.0,
            },
            GradPair {
                grad: tiny,
                hess: 0.0,
            },
            GradPair {
                grad: -tiny,
                hess: tiny,
            },
            GradPair {
                grad: below_one,
                hess: 0.0,
            },
            GradPair {
                grad: one,
                hess: 0.0,
            },
            GradPair {
                grad: above_one,
                hess: 0.0,
            },
            GradPair {
                grad: one,
                hess: 0.0,
            },
            GradPair {
                grad: -one,
                hess: 0.0,
            },
            GradPair {
                grad: f32::MAX,
                hess: f32::MAX,
            },
            GradPair {
                grad: -f32::MAX,
                hess: f32::MAX,
            },
        ];
        for &rate in &[
            0.001f32,
            0.1,
            0.6,
            0.99,
            f32::from_bits(1.0f32.to_bits() - 1),
        ] {
            assert_threshold_matches_sorted_oracle(&pairs, rate, "float32 extremes and ties");
        }
    }

    #[test]
    fn threshold_matches_exhaustive_sorted_oracle() {
        for weights in [
            vec![1.0, 2.0, 10.0, 100.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1e-30, 1e-20, 1.0, 1e20, 1e30],
            vec![1.0; 23],
        ] {
            let pairs: Vec<_> = weights.iter().map(|g| pair(*g)).collect();
            for rate in [
                0.001f32,
                0.1,
                0.3,
                0.5,
                0.9,
                f32::from_bits(1.0f32.to_bits() - 1),
            ] {
                let target = f64::from(rate) * pairs.len() as f64;
                let mut sorted: Vec<f64> = weights
                    .iter()
                    .map(|g| f64::from(*g))
                    .filter(|g| *g > 0.0)
                    .collect();
                sorted.sort_by(f64::total_cmp);
                let result = threshold(&pairs, rate).unwrap();
                if target >= sorted.len() as f64 {
                    assert!(result.is_none());
                    continue;
                }
                let mut oracle = None;
                for count in 1..=sorted.len() {
                    let remaining = target - (sorted.len() - count) as f64;
                    if remaining <= 0.0 {
                        continue;
                    }
                    let tau = sorted[..count].iter().sum::<f64>() / remaining;
                    if tau >= sorted[count - 1] && (count == sorted.len() || tau <= sorted[count]) {
                        oracle = Some(tau);
                        break;
                    }
                }
                let actual = result.unwrap();
                let expected = oracle.unwrap();
                assert!(
                    (actual - expected).abs() <= 1e-11 * expected,
                    "{weights:?} rate={rate}: {actual} != {expected}"
                );
                let mass: f64 = pairs.iter().map(|p| (score(p) / actual).min(1.0)).sum();
                assert!((mass - target).abs() <= 1e-11 * target.max(1.0));
            }
        }
    }
    #[test]
    fn equal_scores_reweight_by_actual_uniform_probability() {
        let mut pairs = vec![
            GradPair {
                grad: 2.0,
                hess: 3.0
            };
            100
        ];
        sample(&mut pairs, 0.25, 17, 3).unwrap();
        for (row, p) in pairs.iter().enumerate() {
            let keep = crate::subsample::row_selected(row as u64, 1u64 << 51, 17, 3);
            assert_eq!(
                (p.grad, p.hess),
                if keep { (8.0, 12.0) } else { (0.0, 0.0) }
            );
        }
    }
    #[test]
    fn zeros_and_rate_one_are_safe() {
        let mut pairs = vec![
            pair(0.0),
            pair(-3.0),
            GradPair {
                grad: 0.0,
                hess: 2.0,
            },
        ];
        let before: Vec<_> = pairs
            .iter()
            .map(|p| (p.grad.to_bits(), p.hess.to_bits()))
            .collect();
        sample(&mut pairs, 1.0, 4, 5).unwrap();
        assert_eq!(
            before,
            pairs
                .iter()
                .map(|p| (p.grad.to_bits(), p.hess.to_bits()))
                .collect::<Vec<_>>()
        );
        let mut zeros = vec![pair(0.0); 20];
        assert_eq!(sample(&mut zeros, 0.1, 3, 4).unwrap(), 0);
        assert!(sample(
            &mut [GradPair {
                grad: 0.0,
                hess: -1.0
            }],
            0.5,
            0,
            0
        )
        .is_err());
        assert!(sample(&mut [pair(f32::INFINITY)], 0.5, 0, 0).is_err());
    }
    #[test]
    fn sampled_derivatives_preserve_expectation_across_seeds() {
        let input = [pair(-1.0), pair(2.0), pair(4.0), pair(12.0)];
        let rate = 0.4;
        let tau = threshold(&input, rate).unwrap().unwrap();
        let trials = 8192;
        let mut totals = [0.0f64; 4];
        let mut counts = 0usize;
        for seed in 0..trials {
            let mut sampled = input;
            counts += sample(&mut sampled, rate, seed, 7).unwrap();
            for (sum, value) in totals.iter_mut().zip(sampled) {
                *sum += f64::from(value.grad);
            }
        }
        let mut expected_count = 0.0;
        let mut count_variance = 0.0;
        for row in 0..input.len() {
            let cutoff =
                (((score(&input[row]) / tau).min(1.0) * GRID as f64) as u64).clamp(1, GRID);
            let p = cutoff as f64 / GRID as f64;
            expected_count += p;
            count_variance += p * (1.0 - p);
            let g = f64::from(input[row].grad);
            let standard_error = (g * g * (1.0 / p - 1.0) / trials as f64).sqrt();
            assert!((totals[row] / trials as f64 - g).abs() <= 6.0 * standard_error + 1e-6);
        }
        assert!(
            (counts as f64 / trials as f64 - expected_count).abs()
                <= 6.0 * (count_variance / trials as f64).sqrt()
        );
    }

    #[test]
    fn float32_extremes_and_reweight_overflow_are_explicit() {
        let tiny = f32::from_bits(1);
        let pairs = [
            pair(tiny),
            GradPair {
                grad: 0.0,
                hess: tiny,
            },
            pair(f32::MAX),
        ];
        let tau = threshold(&pairs, 0.5).unwrap().unwrap();
        assert!(tau.is_finite() && tau > 0.0);
        let mass: f64 = pairs.iter().map(|p| (score(p) / tau).min(1.0)).sum();
        assert!((mass - 1.5).abs() < 1e-12);
        let seed = (0..100)
            .find(|seed| crate::subsample::row_hash(0, *seed, 0) >> 11 < GRID / 2)
            .unwrap();
        let error = sample(&mut [pair(f32::MAX)], 0.5, seed, 0).unwrap_err();
        assert!(error.to_string().contains("float32 range"));
    }
}
