//! 评估指标。
//!
//! 命名沿用 XGBoost 的写法("rmse"、"logloss"),这样用户看参数
//! 就知道是什么,不用重新学一套。第一版只做和两个 objective
//! 对应的两个 —— auc 需要排序,分布式聚合也麻烦,放后面。

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
        (sum / wsum).sqrt() as f32
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
        (sum / wsum) as f32
    }
}

fn logloss_row(raw: f32, y: f32) -> f64 {
    const EPS: f64 = 1e-15;
    let p = (1.0 / (1.0 + (-raw as f64).exp())).clamp(EPS, 1.0 - EPS);
    let y = y as f64;
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

/// objective 决定的默认指标,和 XGBoost 一致。
pub fn default_metric(obj: Objective) -> Box<dyn Metric> {
    match obj {
        Objective::SquaredError => Box::new(Rmse),
        Objective::Logistic => Box::new(LogLoss),
    }
}

pub fn by_name(name: &str) -> Option<Box<dyn Metric>> {
    match name {
        "rmse" => Some(Box::new(Rmse)),
        "logloss" => Some(Box::new(LogLoss)),
        _ => None,
    }
}
