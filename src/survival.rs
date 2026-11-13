//! Native survival losses. Bounds are separate targets, never feature columns.
pub use crate::types::AftDistribution;
use crate::types::GradPair;

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let high = a.max(b);
    high + (a.min(b) - high).exp().ln_1p()
}

/// Cache the time ordering once per fit. Per-round risk scratch is reused.
/// Ties use grouped Breslow risk sets, independent of event/censor ordering.
pub struct CoxState {
    order: Vec<u32>,
    log_risk: Vec<f64>,
    shift: f64,
}
impl CoxState {
    pub fn new(labels: &[f32]) -> anyhow::Result<Self> {
        crate::ensure_input!(
            labels.len() <= u32::MAX as usize && labels.iter().all(|v| v.is_finite()),
            "Cox labels must be finite signed times"
        );
        let mut order: Vec<u32> = (0..labels.len() as u32).collect();
        order.sort_unstable_by(|a, b| {
            labels[*a as usize]
                .abs()
                .total_cmp(&labels[*b as usize].abs())
                .then(a.cmp(b))
        });
        Ok(Self {
            order,
            log_risk: vec![0.0; labels.len()],
            shift: 0.0,
        })
    }
    fn risks(
        &mut self,
        margin: &[f32],
        labels: &[f32],
        weight: Option<&[f32]>,
    ) -> anyhow::Result<()> {
        crate::ensure_input!(
            margin.len() == self.order.len() && labels.len() == margin.len(),
            "Cox target length mismatch"
        );
        crate::ensure_input!(
            margin.iter().all(|v| v.is_finite()),
            "Cox margins must be finite"
        );
        if let Some(w) = weight {
            crate::ensure_input!(
                w.len() == margin.len() && w.iter().all(|v| v.is_finite() && *v >= 0.0),
                "Cox weights must be finite and nonnegative"
            );
        }
        self.shift = margin
            .iter()
            .enumerate()
            .filter(|(i, _)| weight.is_none_or(|w| w[*i] > 0.0))
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut end = self.order.len();
        let mut risk = f64::NEG_INFINITY;
        while end > 0 {
            let mut begin = end - 1;
            let time = labels[self.order[begin] as usize].abs();
            while begin > 0 && labels[self.order[begin - 1] as usize].abs() == time {
                begin -= 1;
            }
            for &row in &self.order[begin..end] {
                let i = row as usize;
                let w = f64::from(weight.map_or(1.0, |w| w[i]));
                if w > 0.0 {
                    risk = log_add(risk, (f64::from(margin[i]) - self.shift) + w.ln());
                }
            }
            self.log_risk[begin] = risk;
            end = begin;
        }
        Ok(())
    }
    pub fn gradients(
        &mut self,
        margin: &[f32],
        labels: &[f32],
        weight: Option<&[f32]>,
    ) -> anyhow::Result<Vec<GradPair>> {
        self.risks(margin, labels, weight)?;
        let mut out = vec![GradPair::default(); margin.len()];
        let (mut a, mut b) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut begin = 0;
        while begin < self.order.len() {
            let mut end = begin + 1;
            let time = labels[self.order[begin] as usize].abs();
            while end < self.order.len() && labels[self.order[end] as usize].abs() == time {
                end += 1;
            }
            let event_mass: f64 = self.order[begin..end]
                .iter()
                .map(|&r| {
                    let i = r as usize;
                    if labels[i] > 0.0 {
                        f64::from(weight.map_or(1.0, |w| w[i]))
                    } else {
                        0.0
                    }
                })
                .sum();
            if event_mass > 0.0 {
                a = log_add(a, event_mass.ln() - self.log_risk[begin]);
                b = log_add(b, event_mass.ln() - 2.0 * self.log_risk[begin]);
            }
            for &row in &self.order[begin..end] {
                let i = row as usize;
                let w = f64::from(weight.map_or(1.0, |w| w[i]));
                if w == 0.0 {
                    continue;
                }
                let q = (f64::from(margin[i]) - self.shift) + w.ln();
                let first = (q + a).exp();
                let second = (2.0 * q + b).exp();
                let grad = first - if labels[i] > 0.0 { w } else { 0.0 };
                out[i] = GradPair {
                    grad: grad as f32,
                    hess: (first - second).max(0.0) as f32,
                };
                crate::ensure_input!(
                    out[i].grad.is_finite() && out[i].hess.is_finite(),
                    "Cox derivatives exceed float32 range"
                );
            }
            begin = end;
        }
        Ok(out)
    }
    pub fn loss(
        &mut self,
        margin: &[f32],
        labels: &[f32],
        weight: Option<&[f32]>,
    ) -> anyhow::Result<f64> {
        self.risks(margin, labels, weight)?;
        let (mut total, mut events) = (0.0, 0.0);
        let mut begin = 0;
        while begin < self.order.len() {
            let mut end = begin + 1;
            let time = labels[self.order[begin] as usize].abs();
            while end < self.order.len() && labels[self.order[end] as usize].abs() == time {
                end += 1;
            }
            for &row in &self.order[begin..end] {
                let i = row as usize;
                let w = f64::from(weight.map_or(1.0, |w| w[i]));
                if labels[i] > 0.0 && w > 0.0 {
                    total += w * (self.log_risk[begin] - (f64::from(margin[i]) - self.shift));
                    events += w;
                }
            }
            begin = end;
        }
        Ok(if events > 0.0 { total / events } else { 0.0 })
    }
}

pub fn validate_bounds(lower: &[f32], upper: &[f32], n: usize) -> anyhow::Result<()> {
    crate::ensure_input!(
        lower.len() == n && upper.len() == n,
        "AFT lower/upper bounds must match the row count"
    );
    for (i, (&l, &u)) in lower.iter().zip(upper).enumerate() {
        crate::ensure_input!(l.is_finite() && l>=0.0 && !u.is_nan() && u>0.0 && l<=u,
            "invalid AFT bounds at row {i}: require 0 <= lower <= upper, upper > 0; only upper may be +inf");
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Jet {
    v: f64,
    d: f64,
    dd: f64,
}
fn sigmoid(z: f64) -> f64 {
    let e = (-z.abs()).exp();
    if z >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}
fn normal_cdf(z: f64) -> Jet {
    if z < -12.0 {
        let x = -z;
        let inv = 1.0 / x;
        let q = inv * inv;
        let series = 1.0 - q + 3.0 * q * q - 15.0 * q.powi(3) + 105.0 * q.powi(4);
        Jet {
            v: -0.5 * z * z - x.ln() - 0.9189385332046727 + series.ln(),
            d: x + inv - 2.0 * inv.powi(3) + 10.0 * inv.powi(5) - 74.0 * inv.powi(7)
                + 706.0 * inv.powi(9),
            dd: -1.0 + q - 6.0 * q * q + 50.0 * q.powi(3) - 518.0 * q.powi(4) + 6354.0 * q.powi(5),
        }
    } else {
        let v = (0.5 * libm::erfc(-z / std::f64::consts::SQRT_2)).ln();
        let d = (-0.5 * z * z - 0.9189385332046727 - v).exp();
        Jet {
            v,
            d,
            dd: -d * (z + d),
        }
    }
}
fn cdf(distribution: AftDistribution, z: f64) -> Jet {
    match distribution {
        AftDistribution::Normal => normal_cdf(z),
        AftDistribution::Logistic => {
            let a = sigmoid(-z);
            Jet {
                v: -softplus(-z),
                d: a,
                dd: -a * sigmoid(z),
            }
        }
        AftDistribution::Extreme => {
            if z < -30.0 {
                let e = z.exp();
                return Jet {
                    v: z - 0.5 * e,
                    d: 1.0 - 0.5 * e,
                    dd: -0.5 * e,
                };
            }
            let e = z.min(700.0).exp();
            if e > 745.0 {
                return Jet {
                    v: 0.0,
                    d: 0.0,
                    dd: 0.0,
                };
            }
            let v = (-(-e).exp_m1()).ln();
            let d = if e > 50.0 {
                e * (-e).exp()
            } else {
                e / e.exp_m1()
            };
            Jet {
                v,
                d,
                dd: d * (1.0 - e - d),
            }
        }
    }
}
fn survival(distribution: AftDistribution, z: f64) -> Jet {
    if distribution == AftDistribution::Extreme {
        let e = z.min(700.0).exp();
        Jet {
            v: -e,
            d: -e,
            dd: -e,
        }
    } else {
        let a = cdf(distribution, -z);
        Jet {
            v: a.v,
            d: -a.d,
            dd: a.dd,
        }
    }
}
fn density(distribution: AftDistribution, z: f64) -> Jet {
    match distribution {
        AftDistribution::Normal => Jet {
            v: -0.5 * z * z - 0.9189385332046727,
            d: -z,
            dd: -1.0,
        },
        AftDistribution::Logistic => Jet {
            v: -softplus(z) - softplus(-z),
            d: 1.0 - 2.0 * sigmoid(z),
            dd: -2.0 * sigmoid(z) * sigmoid(-z),
        },
        AftDistribution::Extreme => {
            let e = z.min(700.0).exp();
            Jet {
                v: z - e,
                d: 1.0 - e,
                dd: -e,
            }
        }
    }
}
fn difference(big: Jet, small: Jet) -> Option<Jet> {
    let delta = small.v - big.v;
    if !(delta < 0.0) {
        return None;
    }
    let a = delta.exp();
    let denominator = -delta.exp_m1();
    let d = (big.d - a * small.d) / denominator;
    let dd = (big.dd - a * small.dd) / denominator - a * ((big.d - small.d) / denominator).powi(2);
    let result = Jet {
        v: big.v + denominator.ln(),
        d,
        dd,
    };
    (result.v.is_finite() && result.d.is_finite() && result.dd.is_finite()).then_some(result)
}

/// Analytic AFT likelihood derivatives; clipping is a tree-fitting safeguard.
/// Log-CDF arithmetic preserves censored tail probabilities without subtraction
/// of two rounded values near one. The reported likelihood uses a 1e-12 floor.
pub fn aft_loss_and_gradient(
    distribution: AftDistribution,
    scale: f32,
    margin: f32,
    lower: f32,
    upper: f32,
) -> (f64, GradPair) {
    let sigma = f64::from(scale);
    let m = f64::from(margin);
    let l = f64::from(lower);
    let u = f64::from(upper);
    let jet = if l == u {
        let mut a = density(distribution, (l.ln() - m) / sigma);
        a.v -= sigma.ln() + l.ln();
        a
    } else if l == 0.0 && u.is_infinite() {
        Jet {
            v: 0.0,
            d: 0.0,
            dd: 0.0,
        }
    } else if l == 0.0 {
        cdf(distribution, (u.ln() - m) / sigma)
    } else if u.is_infinite() {
        survival(distribution, (l.ln() - m) / sigma)
    } else {
        let log_l = l.ln();
        let log_u = u.ln();
        let zl = (log_l - m) / sigma;
        let zu = (log_u - m) / sigma;
        let upper_cdf = cdf(distribution, zu);
        let exact = if upper_cdf.v < -std::f64::consts::LN_2 {
            difference(upper_cdf, cdf(distribution, zl))
        } else {
            difference(survival(distribution, zl), survival(distribution, zu))
        };
        exact.unwrap_or_else(|| {
            // When the two z scores round to one value, integrate the local
            // density over the width computed before subtracting the margin.
            let mut a = density(distribution, ((log_l + log_u) * 0.5 - m) / sigma);
            a.v += ((log_u - log_l) / sigma).ln();
            a
        })
    };
    let grad = (jet.d / sigma).clamp(-15.0, 15.0) as f32;
    let hess = (-jet.dd / (sigma * sigma)).clamp(1e-16, 15.0) as f32;
    ((-jet.v).min(27.631021115928547), GradPair { grad, hess })
}

pub struct CoxMetric;
impl crate::metrics::Metric for CoxMetric {
    fn name(&self) -> &str {
        "cox-nloglik"
    }
    fn supports_weights(&self) -> bool {
        true
    }
    fn eval(&self, p: &[f32], y: &[f32]) -> f32 {
        self.eval_weighted(p, y, None)
    }
    fn try_eval_grouped(
        &self,
        p: &[f32],
        y: &[f32],
        w: Option<&[f32]>,
        _: Option<&[u32]>,
    ) -> anyhow::Result<f32> {
        Ok(CoxState::new(y)?.loss(p, y, w)? as f32)
    }
    fn eval_weighted(&self, p: &[f32], y: &[f32], w: Option<&[f32]>) -> f32 {
        CoxState::new(y)
            .and_then(|mut s| s.loss(p, y, w))
            .map_or(f32::NAN, |v| v as f32)
    }
}

pub struct AftMetric {
    pub distribution: AftDistribution,
    pub scale: f32,
    pub accuracy: bool,
}
impl crate::metrics::Metric for AftMetric {
    fn name(&self) -> &str {
        if self.accuracy {
            "interval-regression-accuracy"
        } else {
            "aft-nloglik"
        }
    }
    fn supports_weights(&self) -> bool {
        true
    }
    fn supports_intervals(&self) -> bool {
        true
    }
    fn lower_is_better(&self) -> bool {
        !self.accuracy
    }
    fn eval(&self, _: &[f32], _: &[f32]) -> f32 {
        f32::NAN
    }
    fn try_eval_interval(
        &self,
        p: &[f32],
        lo: &[f32],
        hi: &[f32],
        w: Option<&[f32]>,
    ) -> anyhow::Result<f32> {
        validate_bounds(lo, hi, p.len())?;
        crate::ensure_input!(
            p.iter().all(|v| v.is_finite()),
            "AFT margins must be finite"
        );
        if let Some(w) = w {
            crate::ensure_input!(
                w.len() == p.len() && w.iter().all(|v| v.is_finite() && *v >= 0.0),
                "AFT weights must be finite and nonnegative"
            );
        }
        let (mut sum, mut total) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let weight = f64::from(w.map_or(1.0, |w| w[i]));
            let value = if self.accuracy {
                f64::from(
                    f64::from(p[i]).exp() >= f64::from(lo[i])
                        && f64::from(p[i]).exp() <= f64::from(hi[i]),
                )
            } else {
                aft_loss_and_gradient(self.distribution, self.scale, p[i], lo[i], hi[i]).0
            };
            sum += weight * value;
            total += weight;
        }
        Ok(if total > 0.0 {
            (sum / total) as f32
        } else {
            0.0
        })
    }
}
