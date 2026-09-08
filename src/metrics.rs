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
