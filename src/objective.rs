//! 每个 objective 的全部行为集中在这里:名字、梯度、链接函数、base_score
//! 变换、标签校验、模型 JSON 里的参数块、GPU fast 路径的编号。
//!
//! 以前这些分散在 train / tree / python / metrics / CUDA ctx 各写一份
//! `match`,加一个 objective 要改七八处,漏一处就是「训得出来但对不上」。
//! 现在加 objective 只改这个文件(外加 metrics 的默认指标)。

use std::collections::BTreeMap;

use crate::types::{GradPair, Objective};

/// XGBoost 的 sigmoid(f32)。训练梯度、预测变换共用同一个表达式,
/// 保证两边逐位一致。
#[inline]
fn sigmoid(m: f32) -> f32 {
    1.0 / (1.0 + (-m).exp())
}

impl Objective {
    /// 全部支持的 objective,用于错误提示和测试遍历。
    pub const ALL: [Objective; 2] = [Objective::SquaredError, Objective::Logistic];

    /// XGBoost 的 objective 名字(写进模型 JSON、出现在报错里)。
    pub fn name(self) -> &'static str {
        match self {
            Objective::SquaredError => "reg:squarederror",
            Objective::Logistic => "binary:logistic",
        }
    }

    /// 从 XGBoost 名字解析。`reg:linear` 是老版本别名,只在读模型时出现。
    pub fn from_name(name: &str) -> Option<Objective> {
        match name {
            "reg:squarederror" | "reg:linear" => Some(Objective::SquaredError),
            "binary:logistic" => Some(Objective::Logistic),
            _ => None,
        }
    }

    /// 报错用:「a、b」形式的全部名字。
    pub fn supported_names() -> String {
        Self::ALL
            .iter()
            .map(|o| o.name())
            .collect::<Vec<_>>()
            .join("、")
    }

    /// base_score(变换**之后**的空间)→ raw margin。
    ///
    /// 训练循环里累加的 predictions 是 raw margin(链接函数**之前**的值),
    /// 但 base_score 和 XGBoost 一样是在变换**之后**的空间里给的。
    /// 二分类忘了取 logit 的话,起点就从 0.0 变成 0.5,第一棵树的梯度
    /// 全错,而且错得很小 —— 训得出来,就是对不上。
    pub fn base_margin(self, base_score: f32) -> f32 {
        match self {
            Objective::SquaredError => base_score,
            // logit(0.5) = 0.0
            Objective::Logistic => {
                assert!(
                    base_score > 0.0 && base_score < 1.0,
                    "binary:logistic 的 base_score 是概率,必须在 (0, 1) 里,给的是 {base_score}"
                );
                (base_score / (1.0 - base_score)).ln()
            }
        }
    }

    /// 单行的梯度(未乘权重)。`gradient_bounds` 和量化那一趟共用它,
    /// 保证两遍算的是同一个函数 —— 各写一遍的话,求上界用的和真正累加的
    /// 可能不一致,而那会让 scale 偏小、定点溢出。
    #[inline]
    pub fn gradient(self, pred: f32, label: f32) -> GradPair {
        match self {
            Objective::SquaredError => GradPair {
                grad: pred - label,
                hess: 1.0,
            },
            Objective::Logistic => {
                let s = sigmoid(pred);
                GradPair {
                    grad: s - label,
                    hess: (s * (1.0 - s)).max(1e-16),
                }
            }
        }
    }

    /// raw margin → 预测值(XGBoost `PredTransform`)。
    #[inline]
    pub fn transform(self, margin: f32) -> f32 {
        match self {
            Objective::SquaredError => margin,
            Objective::Logistic => sigmoid(margin),
        }
    }

    /// 预测变换是不是恒等(恒等时批量路径可以整段跳过)。
    pub fn transform_is_identity(self) -> bool {
        matches!(self, Objective::SquaredError)
    }

    /// 标签校验。`what` 是报错里的数据集描述,比如「训练集」「验证集 \"v\"」。
    pub fn validate_labels(self, labels: &[f32], what: &str) -> anyhow::Result<()> {
        match self {
            Objective::SquaredError => Ok(()),
            Objective::Logistic => {
                anyhow::ensure!(
                    labels.iter().all(|&v| (0.0..=1.0).contains(&v)),
                    "{what}的 binary:logistic 标签必须在 [0, 1]"
                );
                Ok(())
            }
        }
    }

    /// base_score 的定义域校验(变换之后的空间)。
    pub fn validate_base_score(self, base_score: f32) -> anyhow::Result<()> {
        match self {
            Objective::SquaredError => Ok(()),
            Objective::Logistic => {
                anyhow::ensure!(
                    base_score.is_finite() && base_score > 0.0 && base_score < 1.0,
                    "binary:logistic 的 base_score 必须是 (0, 1) 内的有限概率"
                );
                Ok(())
            }
        }
    }

    /// GPU `gpu_math="fast"` 梯度 kernel 里的 objective 编号
    /// (`histogram.cu` 的 `apply_pred_grad_quantize`)。两边必须同步改。
    pub fn device_code(self) -> u32 {
        match self {
            Objective::SquaredError => 0,
            Objective::Logistic => 1,
        }
    }

    /// 模型 JSON 里 objective 下面的参数块。XGBoost 加载时会找它:
    /// 两个现有 objective 都挂在 `reg_loss_param` 下面。
    pub fn json_params(self) -> BTreeMap<String, serde_json::Value> {
        let mut inner = serde_json::Map::new();
        inner.insert(
            "scale_pos_weight".into(),
            serde_json::Value::String("1".into()),
        );
        let mut rest = BTreeMap::new();
        rest.insert(
            "reg_loss_param".to_string(),
            serde_json::Value::Object(inner),
        );
        rest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for o in Objective::ALL {
            assert_eq!(Objective::from_name(o.name()), Some(o));
        }
        assert_eq!(
            Objective::from_name("reg:linear"),
            Some(Objective::SquaredError)
        );
        assert_eq!(Objective::from_name("rank:pairwise"), None);
    }

    #[test]
    fn transform_inverts_base_margin() {
        for o in Objective::ALL {
            for bs in [0.1f32, 0.5, 0.9] {
                let back = o.transform(o.base_margin(bs));
                assert!((back - bs).abs() < 1e-6, "{o:?} {bs} -> {back}");
            }
        }
    }

    #[test]
    fn device_codes_are_distinct() {
        let codes: Vec<u32> = Objective::ALL.iter().map(|o| o.device_code()).collect();
        assert_eq!(codes, vec![0, 1]);
    }
}
