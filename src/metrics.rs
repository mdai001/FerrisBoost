//! 评估指标。
//!
//! 命名沿用 XGBoost 的写法("rmse"、"logloss"、"error"、"auc"),这样用户
//! 看参数就知道是什么,不用重新学一套。
//!
//! ⚠️ metric 收到的是 **raw margin**(链接函数之前)。需要概率的指标
//! 自己过 sigmoid;`for_objective` 负责按 objective 选对变换。

use crate::types::Objective;

pub trait Metric: Send + Sync {
    /// XGBoost 风格的名字,会出现在日志和 evals_result 的 key 里。
    fn name(&self) -> &str;

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32;

    /// 能否做加权评估。自定义 metric 默认不支持;训练入口在验证集带权重时
    /// 先检查这一项,不会把权重静默丢掉。
    fn supports_weights(&self) -> bool {
        false
    }

    /// 加权评估(XGBoost:`sum(w·loss) / sum(w)`)。`None` 走 `eval`。
    ///
    /// 默认实现只接受 `None`;支持权重的 metric 必须同时覆盖这个方法和
    /// `supports_weights`。
    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        match weight {
            None => self.eval(pred, label),
            Some(_) => panic!("metric {:?} 不支持样本权重", self.name()),
        }
    }

    /// 值越小越好?early stopping 靠这个判断方向。
    fn lower_is_better(&self) -> bool {
        true
    }
}

pub struct Rmse;

impl Metric for Rmse {
    fn name(&self) -> &str {
        "rmse"
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        assert_eq!(pred.len(), label.len(), "rmse 的预测和标签长度必须一致");
        assert!(!pred.is_empty(), "rmse 不能评估空数据");
        let sum: f64 = pred
            .iter()
            .zip(label)
            .map(|(&p, &y)| {
                let d = (p - y) as f64;
                d * d
            })
            .sum();
        (sum / pred.len() as f64).sqrt() as f32
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        let Some(w) = weight else {
            return self.eval(pred, label);
        };
        assert_eq!(pred.len(), w.len(), "rmse 的预测和权重长度必须一致");
        let (sum, wsum) = weighted_sums(pred, label, w, |p, y| {
            let d = (p - y) as f64;
            d * d
        });
        // XGBoost: `wsum == 0 ? sqrt(esum) : sqrt(esum / wsum)`
        final_ratio(sum, wsum).sqrt() as f32
    }
}

/// XGBoost 的 `GetFinal`:权重全 0 时返回 `esum`(此时它也是 0),不出 NaN。
fn final_ratio(sum: f64, wsum: f64) -> f64 {
    if wsum == 0.0 {
        sum
    } else {
        sum / wsum
    }
}

/// `(sum(w·loss), sum(w))`,f64 累加。
fn weighted_sums(
    pred: &[f32],
    label: &[f32],
    weight: &[f32],
    loss: impl Fn(f32, f32) -> f64,
) -> (f64, f64) {
    assert_eq!(pred.len(), label.len(), "预测和标签长度必须一致");
    assert!(!pred.is_empty(), "不能评估空数据");
    pred.iter()
        .zip(label)
        .zip(weight)
        .fold((0.0, 0.0), |(s, ws), ((&p, &y), &w)| {
            (s + w as f64 * loss(p, y), ws + w as f64)
        })
}

pub struct LogLoss;

impl Metric for LogLoss {
    fn name(&self) -> &str {
        "logloss"
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        assert_eq!(pred.len(), label.len(), "logloss 的预测和标签长度必须一致");
        assert!(!pred.is_empty(), "logloss 不能评估空数据");
        const EPS: f64 = 1e-15;
        let sum: f64 = pred
            .iter()
            .zip(label)
            .map(|(&raw, &y)| {
                // pred 是原始分数,需要过 sigmoid
                let p = (1.0 / (1.0 + (-raw as f64).exp())).clamp(EPS, 1.0 - EPS);
                let y = y as f64;
                -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
            })
            .sum();
        (sum / pred.len() as f64) as f32
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        let Some(w) = weight else {
            return self.eval(pred, label);
        };
        assert_eq!(pred.len(), w.len(), "logloss 的预测和权重长度必须一致");
        let (sum, wsum) = weighted_sums(pred, label, w, logloss_row);
        final_ratio(sum, wsum) as f32
    }
}

fn logloss_row(raw: f32, y: f32) -> f64 {
    const EPS: f64 = 1e-15;
    let p = (1.0 / (1.0 + (-raw as f64).exp())).clamp(EPS, 1.0 - EPS);
    let y = y as f64;
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

/// XGBoost 的 sigmoid 在 f32 里算;阈值类指标(error)和排序类指标(auc)
/// 都看变换后的 f32 概率,所以这里也用 f32,免得 0.5 附近判断不一致。
fn sigmoid_f32(raw: f32) -> f32 {
    1.0f32 / (1.0f32 + (-raw).exp())
}

/// `binary:logistic` 下的 rmse:XGBoost 在**概率**上算,不是在 margin 上。
pub struct ProbRmse;

impl Metric for ProbRmse {
    fn name(&self) -> &str {
        "rmse"
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.eval_weighted(pred, label, Some(&vec![1.0; pred.len()]))
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        let ones;
        let w = match weight {
            Some(w) => w,
            None => {
                ones = vec![1.0; pred.len()];
                &ones
            }
        };
        assert_eq!(pred.len(), w.len(), "rmse 的预测和权重长度必须一致");
        let (sum, wsum) = weighted_sums(pred, label, w, |raw, y| {
            let d = (y - sigmoid_f32(raw)) as f64;
            d * d
        });
        final_ratio(sum, wsum).sqrt() as f32
    }
}

/// 二分类错误率,阈值 0.5(XGBoost `error`):`p > 0.5 ? 1 - y : y`。
pub struct ClassError;

impl Metric for ClassError {
    fn name(&self) -> &str {
        "error"
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.eval_weighted(pred, label, Some(&vec![1.0; pred.len()]))
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        let ones;
        let w = match weight {
            Some(w) => w,
            None => {
                ones = vec![1.0; pred.len()];
                &ones
            }
        };
        assert_eq!(pred.len(), w.len(), "error 的预测和权重长度必须一致");
        let (sum, wsum) = weighted_sums(pred, label, w, |raw, y| {
            (if sigmoid_f32(raw) > 0.5 { 1.0 - y } else { y }) as f64
        });
        final_ratio(sum, wsum) as f32
    }
}

/// 二分类 ROC AUC(XGBoost `auc` 的 binary 分支)。
///
/// 按概率降序扫一遍;**相同概率的样本合成一段**,段与段之间用梯形面积
/// 累加,最后除以 `sum(w·y) · sum(w·(1-y))`。只有一类样本时返回 NaN
/// (XGBoost 同样返回 NaN 并打 warning)。
pub struct Auc;

impl Metric for Auc {
    fn name(&self) -> &str {
        "auc"
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.eval_weighted(pred, label, None)
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        assert_eq!(pred.len(), label.len(), "auc 的预测和标签长度必须一致");
        if let Some(w) = weight {
            assert_eq!(pred.len(), w.len(), "auc 的预测和权重长度必须一致");
        }
        if pred.is_empty() {
            return f32::NAN;
        }
        let p: Vec<f32> = pred.iter().map(|&r| sigmoid_f32(r)).collect();
        let mut idx: Vec<usize> = (0..p.len()).collect();
        // 稳定排序,降序。NaN 概率不应出现(raw margin 有限)。
        idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
        let wt = |i: usize| weight.map_or(1.0f32, |w| w[i]);
        let trapezoid = |x0: f64, x1: f64, y0: f64, y1: f64| (x0 - x1).abs() * (y0 + y1) * 0.5;

        let first = idx[0];
        let (y, w) = (label[first], wt(first));
        let mut fp = ((1.0 - y) * w) as f64;
        let mut tp = (y * w) as f64;
        let (mut fp_prev, mut tp_prev) = (0.0f64, 0.0f64);
        let mut area = 0.0f64;
        for k in 1..idx.len() {
            let (i, prev) = (idx[k], idx[k - 1]);
            if p[i] != p[prev] {
                area += trapezoid(fp_prev, fp, tp_prev, tp);
                fp_prev = fp;
                tp_prev = tp;
            }
            let (y, w) = (label[i], wt(i));
            fp += ((1.0 - y) * w) as f64;
            tp += (y * w) as f64;
        }
        area += trapezoid(fp_prev, fp, tp_prev, tp);
        if fp <= 0.0 || tp <= 0.0 {
            return f32::NAN;
        }
        (area / (fp * tp)).min(1.0) as f32
    }

    fn lower_is_better(&self) -> bool {
        false
    }
}

/// objective 决定的默认指标,和 XGBoost 一致。
pub fn default_metric(obj: Objective) -> Box<dyn Metric> {
    match obj {
        Objective::SquaredError => Box::new(Rmse),
        Objective::Logistic => Box::new(LogLoss),
    }
}

/// XGBoost `eval_metric` 名字 → metric,按 objective 选对预测变换。
///
/// `logloss` / `error` / `auc` 是概率空间的二分类指标,只接
/// `binary:logistic`;`rmse` 两个 objective 都接(二分类时和 XGBoost
/// 一样在概率上算)。不认识或组合不支持时返回说明原因的错误。
pub fn for_objective(name: &str, obj: Objective) -> Result<Box<dyn Metric>, String> {
    match (name, obj) {
        ("rmse", Objective::SquaredError) => Ok(Box::new(Rmse)),
        ("rmse", Objective::Logistic) => Ok(Box::new(ProbRmse)),
        ("logloss", Objective::Logistic) => Ok(Box::new(LogLoss)),
        ("error", Objective::Logistic) => Ok(Box::new(ClassError)),
        ("auc", Objective::Logistic) => Ok(Box::new(Auc)),
        ("logloss" | "error" | "auc", Objective::SquaredError) => Err(format!(
            "eval_metric {name:?} 是二分类指标,需要 objective=\"binary:logistic\""
        )),
        _ => Err(format!(
            "eval_metric {name:?} 不支持,现在只有 rmse / logloss / error / auc"
        )),
    }
}

pub fn by_name(name: &str) -> Option<Box<dyn Metric>> {
    match name {
        "rmse" => Some(Box::new(Rmse)),
        "logloss" => Some(Box::new(LogLoss)),
        "error" => Some(Box::new(ClassError)),
        "auc" => Some(Box::new(Auc)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    #[test]
    fn auc_perfect_inverse_and_ties() {
        let y = [0.0, 0.0, 1.0, 1.0];
        let good: Vec<f32> = [0.1, 0.2, 0.8, 0.9].iter().map(|&p| logit(p)).collect();
        assert_eq!(Auc.eval(&good, &y), 1.0);
        let bad: Vec<f32> = good.iter().rev().copied().collect();
        assert_eq!(Auc.eval(&bad, &y), 0.0);
        // 全部同分:一段梯形 = 0.5
        assert_eq!(Auc.eval(&[0.0; 4], &y), 0.5);
    }

    #[test]
    fn auc_matches_pairwise_definition_with_weights() {
        // AUC = P(score_pos > score_neg) + 0.5 P(=),按权重对计数。
        let p = [0.9f32, 0.7, 0.7, 0.4, 0.3, 0.2];
        let y = [1.0f32, 0.0, 1.0, 1.0, 0.0, 0.0];
        let w = [1.0f32, 2.0, 0.5, 3.0, 1.0, 4.0];
        let raw: Vec<f32> = p.iter().map(|&v| logit(v)).collect();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..6 {
            for j in 0..6 {
                if y[i] == 1.0 && y[j] == 0.0 {
                    let pw = (w[i] * w[j]) as f64;
                    let (si, sj) = (sigmoid_f32(raw[i]), sigmoid_f32(raw[j]));
                    num += pw
                        * if si > sj {
                            1.0
                        } else if si == sj {
                            0.5
                        } else {
                            0.0
                        };
                    den += pw;
                }
            }
        }
        let got = Auc.eval_weighted(&raw, &y, Some(&w));
        assert!(
            (got as f64 - num / den).abs() < 1e-6,
            "{got} vs {}",
            num / den
        );
        assert!(Auc.eval(&raw[..1], &y[..1]).is_nan(), "单一类别返回 NaN");
        assert!(!Auc.lower_is_better());
    }

    #[test]
    fn error_uses_strict_half_threshold() {
        let y = [1.0, 0.0, 1.0, 0.0];
        // p = 0.5 不算正类(严格 >)
        let raw = [0.0f32, 0.0, 2.0, 2.0];
        assert_eq!(ClassError.eval(&raw, &y), 0.5);
        let w = [1.0f32, 3.0, 1.0, 1.0];
        // 错的是 row0 (w=1) 和 row3 (w=1):2/6
        assert!((ClassError.eval_weighted(&raw, &y, Some(&w)) - 2.0 / 6.0).abs() < 1e-7);
    }

    #[test]
    fn zero_total_weight_returns_esum_like_xgboost() {
        let z = [0.0f32; 3];
        assert_eq!(
            Rmse.eval_weighted(&[1.0, 2.0, 3.0], &[0.0; 3], Some(&z)),
            0.0
        );
        assert_eq!(
            LogLoss.eval_weighted(&[1.0, 2.0, 3.0], &[0.0; 3], Some(&z)),
            0.0
        );
    }

    #[test]
    fn metric_objective_combinations() {
        assert_eq!(
            for_objective("rmse", Objective::Logistic).unwrap().name(),
            "rmse"
        );
        assert!(for_objective("auc", Objective::SquaredError).is_err());
        assert!(for_objective("mae", Objective::Logistic).is_err());
        // 二分类 rmse 在概率空间:raw=0 → p=0.5,label=1 → rmse=0.5
        let m = for_objective("rmse", Objective::Logistic).unwrap();
        assert_eq!(m.eval(&[0.0], &[1.0]), 0.5);
    }
}
