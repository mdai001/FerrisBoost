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
/// XGBoost `kRtEps`。
const K_RT_EPS: f32 = 1e-6;

/// XGBoost `ObjFunction::DefaultBaseScore()`:无法估计时的回退值。
pub const DEFAULT_BASE_SCORE: f32 = 0.5;

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
    ///
    /// 逐位照搬 XGBoost 3.4.1 `ProbToMargin`:logistic 先夹到
    /// `[kRtEps, 1 − kRtEps]`(kRtEps = 1e-6,标签全 0 或全 1 时估出的
    /// 0 / 1 也能用),再用 `common::Logit(x) = −logf(1/x − 1)` ——
    /// 和 `ln(x/(1−x))` 数学上相等,但 f32 最后一位不一定相同。
    pub fn base_margin(self, base_score: f32) -> f32 {
        match self {
            Objective::SquaredError => base_score,
            // logit(0.5) = 0.0
            Objective::Logistic => {
                assert!(
                    (0.0..=1.0).contains(&base_score),
                    "binary:logistic 的 base_score 是概率,必须在 [0, 1] 里,给的是 {base_score}"
                );
                let x = base_score.clamp(K_RT_EPS, 1.0 - K_RT_EPS);
                -(1.0f32 / x - 1.0).ln()
            }
        }
    }

    /// 不给 `base_score` 时从训练标签估计(XGBoost 3.4.1 `InitEstimation`)。
    ///
    /// 两个现有 objective 都走 `FitInterceptGlmLike`:(加权)标签均值。
    /// 算术逐位照搬 `common::SampleMean` / `WeightedSampleMean`:
    /// f64 里累加 `y_i / n`(加权时 `y_i / Σw · w_i`),最后截成 f32。
    /// XGBoost 按 OpenMP 线程分段求和,我们固定按行号串行累加 ——
    /// 与 XGBoost `nthread=1` 逐位相同,而且不随线程数变化
    /// (我们的不变量;XGBoost 多线程时 f64 尾数可能差一点,截到 f32 后
    /// 几乎总是同一个数)。
    ///
    /// 权重全为 0 时 XGBoost 直接报错;我们退回默认 0.5,保证全零权重的
    /// 退化输入仍能训练(之前的审计修复依赖这一点)。
    pub fn estimate_base_score(self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        let mean = match weights {
            None => {
                let n = labels.len() as f64;
                labels.iter().fold(0.0f64, |acc, &y| acc + y as f64 / n)
            }
            Some(w) => {
                let sum_w = w.iter().fold(0.0f64, |acc, &v| acc + v as f64);
                if sum_w <= 0.0 {
                    return DEFAULT_BASE_SCORE;
                }
                labels
                    .iter()
                    .zip(w)
                    .fold(0.0f64, |acc, (&y, &wi)| acc + y as f64 / sum_w * wi as f64)
            }
        };
        mean as f32
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
                // XGBoost `CheckIntercept`:闭区间,标签全相同的退化情形可以是 0 或 1。
                anyhow::ensure!(
                    base_score.is_finite() && (0.0..=1.0).contains(&base_score),
                    "binary:logistic 的 base_score 必须是 [0, 1] 内的有限概率"
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
