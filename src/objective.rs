//! 每个 objective 的全部行为集中在这里:名字、梯度、链接函数、base_score
//! 变换与估计、标签校验、模型 JSON 里的参数块、GPU fast 路径的编号。
//!
//! 以前这些分散在 train / tree / python / metrics / CUDA ctx 各写一份
//! `match`,加一个 objective 要改七八处,漏一处就是「训得出来但对不上」。
//! 现在加 objective 只改这个文件(外加 metrics 的默认指标和 CUDA fast kernel)。
//!
//! 数值逐位照搬 XGBoost 3.4.1:`src/objective/regression_obj.cu`、
//! `regression_loss.h`、`init_estimation.cc`、`src/common/stats.cc`。
//! 所有梯度都是 f32 运算,和 XGBoost 一样。

use std::collections::BTreeMap;

use crate::types::{GradPair, Objective};

/// XGBoost `kRtEps`。
const K_RT_EPS: f32 = 1e-6;

/// XGBoost `ObjFunction::DefaultBaseScore()`:无法估计时的回退值。
pub const DEFAULT_BASE_SCORE: f32 = 0.5;

/// `count:poisson` 的 `max_delta_step` 默认值(XGBoost `PoissonRegressionParam`)。
pub const POISSON_DEFAULT_MAX_DELTA_STEP: f32 = 0.7;
/// `reg:tweedie` 的 `tweedie_variance_power` 默认值。
pub const TWEEDIE_DEFAULT_VARIANCE_POWER: f32 = 1.5;
/// `reg:pseudohubererror` 的 `huber_slope` 默认值。
pub const HUBER_DEFAULT_SLOPE: f32 = 1.0;

/// XGBoost 的 sigmoid(f32)。训练梯度、预测变换共用同一个表达式,
/// 保证两边逐位一致。
#[inline]
fn sigmoid(m: f32) -> f32 {
    1.0 / (1.0 + (-m).exp())
}

/// `sum(y/n)` 或 `sum(y/Σw·w)`,f64 串行累加 —— XGBoost
/// `SampleMean` / `WeightedSampleMean` 在 nthread=1 时的算术。
/// 权重全 0 时返回 `None`。
fn label_mean(labels: &[f32], weights: Option<&[f32]>) -> Option<f64> {
    match weights {
        None => {
            let n = labels.len() as f64;
            Some(labels.iter().fold(0.0f64, |acc, &y| acc + y as f64 / n))
        }
        Some(w) => {
            let sum_w = w.iter().fold(0.0f64, |acc, &v| acc + v as f64);
            if sum_w <= 0.0 {
                return None;
            }
            Some(
                labels
                    .iter()
                    .zip(w)
                    .fold(0.0f64, |acc, (&y, &wi)| acc + y as f64 / sum_w * wi as f64),
            )
        }
    }
}

fn json_block(block: &str, key: &str, value: String) -> BTreeMap<String, serde_json::Value> {
    let mut inner = serde_json::Map::new();
    inner.insert(key.into(), serde_json::Value::String(value));
    let mut rest = BTreeMap::new();
    rest.insert(block.to_string(), serde_json::Value::Object(inner));
    rest
}

/// 从模型 JSON 的参数块里读一个 f32(XGBoost 存成字符串)。
fn json_param(rest: &BTreeMap<String, serde_json::Value>, block: &str, key: &str) -> Option<f32> {
    match rest.get(block)?.get(key)? {
        serde_json::Value::String(s) => s.trim().parse().ok(),
        serde_json::Value::Number(n) => n.as_f64().map(|x| x as f32),
        _ => None,
    }
}

impl Objective {
    /// 全部支持的 objective(参数取默认值),用于报错和测试遍历。
    pub const ALL: [Objective; 6] = [
        Objective::SquaredError,
        Objective::Logistic,
        Objective::Poisson {
            max_delta_step: POISSON_DEFAULT_MAX_DELTA_STEP,
        },
        Objective::Gamma,
        Objective::Tweedie {
            variance_power: TWEEDIE_DEFAULT_VARIANCE_POWER,
        },
        Objective::PseudoHuber {
            slope: HUBER_DEFAULT_SLOPE,
        },
    ];

    /// XGBoost 的 objective 名字(写进模型 JSON、出现在报错里)。
    pub fn name(self) -> &'static str {
        match self {
            Objective::SquaredError => "reg:squarederror",
            Objective::Logistic => "binary:logistic",
            Objective::Poisson { .. } => "count:poisson",
            Objective::Gamma => "reg:gamma",
            Objective::Tweedie { .. } => "reg:tweedie",
            Objective::PseudoHuber { .. } => "reg:pseudohubererror",
            Objective::Softprob { .. } => "multi:softprob",
            Objective::Softmax { .. } => "multi:softmax",
        }
    }

    /// 从 XGBoost 名字解析,参数取默认值。`reg:linear` 是老版本别名,
    /// 只在读模型时出现。
    pub fn from_name(name: &str) -> Option<Objective> {
        match name {
            "reg:linear" => return Some(Objective::SquaredError),
            // num_class 由 `with_params` / 模型 JSON 填
            "multi:softprob" => return Some(Objective::Softprob { num_class: 0 }),
            "multi:softmax" => return Some(Objective::Softmax { num_class: 0 }),
            _ => {}
        }
        Self::ALL.into_iter().find(|o| o.name() == name)
    }

    /// 报错用:「a、b」形式的全部名字。
    pub fn supported_names() -> String {
        Self::ALL
            .iter()
            .map(|o| o.name())
            .chain(["multi:softprob", "multi:softmax"])
            .collect::<Vec<_>>()
            .join("、")
    }

    /// 每行几个输出(多分类是 `num_class`,其余 1)。
    pub fn num_outputs(self) -> usize {
        match self {
            Objective::Softprob { num_class } | Objective::Softmax { num_class } => {
                num_class as usize
            }
            _ => 1,
        }
    }

    /// 是不是多分类(每轮 `num_class` 棵树、预测按行做 softmax)。
    pub fn is_multiclass(self) -> bool {
        matches!(self, Objective::Softprob { .. } | Objective::Softmax { .. })
    }

    /// 带上用户参数。`max_delta_step` 是用户**显式**给的值(`None` = 没给):
    /// 和 XGBoost 一样,同一个参数名同时喂给树(默认 0)和 Poisson 的
    /// Hessian 稳定项(默认 0.7)。
    pub fn with_params(
        self,
        max_delta_step: Option<f32>,
        tweedie_variance_power: f32,
        huber_slope: f32,
        num_class: u32,
    ) -> Objective {
        match self {
            Objective::Poisson { .. } => Objective::Poisson {
                max_delta_step: max_delta_step.unwrap_or(POISSON_DEFAULT_MAX_DELTA_STEP),
            },
            Objective::Tweedie { .. } => Objective::Tweedie {
                variance_power: tweedie_variance_power,
            },
            Objective::PseudoHuber { .. } => Objective::PseudoHuber { slope: huber_slope },
            Objective::Softprob { .. } => Objective::Softprob { num_class },
            Objective::Softmax { .. } => Objective::Softmax { num_class },
            other => other,
        }
    }

    /// 用户没给 `max_delta_step` 时树用的值。XGBoost `learner.cc`:
    /// objective 是 `count:poisson` 且没设 `max_delta_step` 时,learner 把
    /// 配置里的 `max_delta_step` 设成 0.7 —— 树参数和 Poisson 参数**都**
    /// 拿到它。其它 objective 是树参数默认的 0。
    ///
    /// Rust API 的 `TrainParams::max_delta_step` 是显式值,不做这一步;
    /// Python 入口按这里补。
    pub fn default_tree_max_delta_step(self) -> f32 {
        match self {
            Objective::Poisson { .. } => POISSON_DEFAULT_MAX_DELTA_STEP,
            _ => 0.0,
        }
    }

    /// 参数合法性(XGBoost 的 `set_range` / `CHECK`)。
    pub fn validate_params(self) -> anyhow::Result<()> {
        match self {
            Objective::Poisson { max_delta_step } => anyhow::ensure!(
                max_delta_step.is_finite() && max_delta_step >= 0.0,
                "count:poisson 的 max_delta_step 必须是非负有限值"
            ),
            Objective::Tweedie { variance_power } => anyhow::ensure!(
                (1.0..2.0).contains(&variance_power),
                "tweedie_variance_power 必须在 [1, 2),给的是 {variance_power}"
            ),
            Objective::PseudoHuber { slope } => anyhow::ensure!(
                slope.is_finite() && slope != 0.0,
                "huber_slope 必须是非 0 的有限值"
            ),
            Objective::Softprob { num_class } | Objective::Softmax { num_class } => {
                anyhow::ensure!(
                    num_class >= 2,
                    "{} 需要 num_class ≥ 2,给的是 {num_class}",
                    self.name()
                )
            }
            _ => {}
        }
        Ok(())
    }

    /// `scale_pos_weight` 只对 XGBoost `RegLossObj` 家族生效
    /// (squarederror、logistic、gamma);其它 objective 忽略它。
    pub fn uses_scale_pos_weight(self) -> bool {
        matches!(
            self,
            Objective::SquaredError | Objective::Logistic | Objective::Gamma
        )
    }

    /// base_score(变换**之后**的空间)→ raw margin(XGBoost `ProbToMargin`)。
    ///
    /// logistic 先夹到 `[kRtEps, 1 − kRtEps]` 再用 `common::Logit(x) =
    /// −logf(1/x − 1)`(和 `ln(x/(1−x))` 数学上相等,但 f32 最后一位不一定
    /// 相同);log 链接(poisson / gamma / tweedie)是 `logf(x)`。
    pub fn base_margin(self, base_score: f32) -> f32 {
        match self {
            // 多分类的 base_score 本来就在 margin 空间(XGBoost 的
            // ProbToMargin 是恒等)。
            Objective::SquaredError
            | Objective::PseudoHuber { .. }
            | Objective::Softprob { .. }
            | Objective::Softmax { .. } => base_score,
            // logit(0.5) = 0.0
            Objective::Logistic => {
                assert!(
                    (0.0..=1.0).contains(&base_score),
                    "binary:logistic 的 base_score 是概率,必须在 [0, 1] 里,给的是 {base_score}"
                );
                let x = base_score.clamp(K_RT_EPS, 1.0 - K_RT_EPS);
                -(1.0f32 / x - 1.0).ln()
            }
            Objective::Poisson { .. } | Objective::Gamma | Objective::Tweedie { .. } => {
                base_score.ln()
            }
        }
    }

    /// 不给 `base_score` 时从训练标签估计(XGBoost 3.4.1 `InitEstimation`)。
    ///
    /// `grad_weights` 是已经折进 `scale_pos_weight` 的梯度权重(没有就是样本
    /// 权重)。按 objective 分两种:
    /// - `FitInterceptGlmLike`(加权标签均值):squarederror / logistic / gamma
    ///   在 `scale_pos_weight == 1` 时,poisson / tweedie 永远;
    /// - `FitIntercept`(一步 Newton):pseudohuber 永远,RegLossObj 家族在
    ///   `|scale_pos_weight − 1| > kRtEps` 时。
    pub fn initial_base_score(
        self,
        labels: &[f32],
        weights: Option<&[f32]>,
        grad_weights: Option<&[f32]>,
        scale_pos_weight: f32,
    ) -> f32 {
        let newton = match self {
            Objective::PseudoHuber { .. } => true,
            o if o.uses_scale_pos_weight() => (scale_pos_weight - 1.0).abs() > K_RT_EPS,
            _ => false,
        };
        if newton {
            self.newton_base_score(labels, grad_weights)
        } else {
            self.estimate_base_score(labels, weights)
        }
    }

    /// 加权标签均值(`FitInterceptGlmLike`)。均值按行号串行累加,和
    /// XGBoost `nthread=1` 逐位相同且不随线程数变化。权重全为 0 时 XGBoost
    /// 报错;我们退回 0.5,保证全零权重的退化输入仍能训练。
    pub fn estimate_base_score(self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        label_mean(labels, weights).map_or(DEFAULT_BASE_SCORE, |m| m as f32)
    }

    /// XGBoost `FitIntercept::InitEstimation`(一步 Newton):在 margin = 0
    /// 处求梯度(乘梯度权重),f64 求和,`w = −ΣG / max(ΣH, kRtEps)` 截成
    /// f32,再 `PredTransform` 回到 base_score 的空间。按行号串行求和
    /// (= XGBoost nthread=1)。
    pub fn newton_base_score(self, labels: &[f32], grad_weights: Option<&[f32]>) -> f32 {
        let (mut sg, mut sh) = (0.0f64, 0.0f64);
        for (i, &y) in labels.iter().enumerate() {
            let g = self.gradient(0.0, y);
            let (g, h) = match grad_weights {
                Some(w) => (g.grad * w[i], g.hess * w[i]),
                None => (g.grad, g.hess),
            };
            sg += g as f64;
            sh += h as f64;
        }
        let w = (-sg / sh.max(K_RT_EPS as f64)) as f32;
        self.transform(w)
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
            // PoissonRegression::GetGradient
            Objective::Poisson { max_delta_step } => GradPair {
                grad: pred.exp() - label,
                hess: (pred + max_delta_step).exp(),
            },
            // RegLossObj<GammaDeviance>:p = exp(margin),1 − y/p,y/p
            Objective::Gamma => {
                let p = pred.exp();
                GradPair {
                    grad: 1.0 - label / p,
                    hess: label / p,
                }
            }
            // TweedieRegression::GetGradient
            Objective::Tweedie {
                variance_power: rho,
            } => {
                let a = ((1.0 - rho) * pred).exp();
                let b = ((2.0 - rho) * pred).exp();
                GradPair {
                    grad: -label * a + b,
                    hess: -label * (1.0 - rho) * a + (2.0 - rho) * b,
                }
            }
            Objective::Softprob { .. } | Objective::Softmax { .. } => {
                unreachable!("多分类梯度按行算,走 Objective::softmax_gradients")
            }
            // PseudoHuberRegression::GetGradient
            Objective::PseudoHuber { slope } => {
                let z = pred - label;
                let slope2 = slope * slope;
                let scale_sqrt = (1.0 + z * z / slope2).sqrt();
                let scale = slope2 + z * z;
                GradPair {
                    grad: z / scale_sqrt,
                    hess: slope2 / (scale * scale_sqrt),
                }
            }
        }
    }

    /// raw margin → 预测值(XGBoost `PredTransform`)。
    #[inline]
    pub fn transform(self, margin: f32) -> f32 {
        match self {
            // 多分类的变换是按行的(softmax / argmax),在 `transform_row` 里;
            // 单个 margin 原样返回。
            Objective::SquaredError
            | Objective::PseudoHuber { .. }
            | Objective::Softprob { .. }
            | Objective::Softmax { .. } => margin,
            Objective::Logistic => sigmoid(margin),
            Objective::Poisson { .. } | Objective::Gamma | Objective::Tweedie { .. } => {
                margin.exp()
            }
        }
    }

    /// 预测变换是不是恒等(恒等时批量路径可以整段跳过)。
    pub fn transform_is_identity(self) -> bool {
        matches!(
            self,
            Objective::SquaredError
                | Objective::PseudoHuber { .. }
                | Objective::Softprob { .. }
                | Objective::Softmax { .. }
        )
    }

    /// 标签校验。`what` 是报错里的数据集描述,比如「训练集」「验证集 \"v\"」。
    pub fn validate_labels(self, labels: &[f32], what: &str) -> anyhow::Result<()> {
        let name = self.name();
        match self {
            Objective::SquaredError | Objective::PseudoHuber { .. } => {}
            Objective::Logistic => anyhow::ensure!(
                labels.iter().all(|&v| (0.0..=1.0).contains(&v)),
                "{what}的 {name} 标签必须在 [0, 1]"
            ),
            Objective::Poisson { .. } | Objective::Tweedie { .. } => anyhow::ensure!(
                labels.iter().all(|&v| v >= 0.0),
                "{what}的 {name} 标签必须非负"
            ),
            Objective::Gamma => anyhow::ensure!(
                labels.iter().all(|&v| v > 0.0),
                "{what}的 {name} 标签必须为正"
            ),
            Objective::Softprob { num_class } | Objective::Softmax { num_class } => {
                anyhow::ensure!(
                    labels
                        .iter()
                        .all(|&v| v >= 0.0 && v < num_class as f32 && v.fract() == 0.0),
                    "{what}的 {name} 标签必须是 [0, num_class={num_class}) 里的整数"
                )
            }
        }
        Ok(())
    }

    /// base_score 的定义域校验(变换之后的空间)。
    pub fn validate_base_score(self, base_score: f32) -> anyhow::Result<()> {
        let name = self.name();
        match self {
            Objective::SquaredError | Objective::PseudoHuber { .. } => {}
            Objective::Softprob { .. } | Objective::Softmax { .. } => {
                anyhow::ensure!(base_score.is_finite(), "{name} 的 base_score 必须是有限值")
            }
            // XGBoost `CheckIntercept`:闭区间,标签全相同的退化情形可以是 0 或 1。
            Objective::Logistic => anyhow::ensure!(
                base_score.is_finite() && (0.0..=1.0).contains(&base_score),
                "{name} 的 base_score 必须是 [0, 1] 内的有限概率"
            ),
            // log 链接:0 会变成 −inf margin。
            Objective::Poisson { .. } | Objective::Gamma | Objective::Tweedie { .. } => {
                anyhow::ensure!(
                    base_score.is_finite() && base_score > 0.0,
                    "{name} 的 base_score 必须为正(标签全为 0 时请显式给一个正的 base_score)"
                )
            }
        }
        Ok(())
    }

    /// GPU `gpu_math="fast"` 梯度 kernel 里的 objective 编号
    /// (`histogram.cu` 的 `apply_pred_grad_quantize`)。两边必须同步改。
    pub fn device_code(self) -> u32 {
        match self {
            Objective::SquaredError => 0,
            Objective::Logistic => 1,
            Objective::Poisson { .. } => 2,
            Objective::Gamma => 3,
            Objective::Tweedie { .. } => 4,
            Objective::PseudoHuber { .. } => 5,
            // 多分类不走这个单输出 kernel:fast 用 softmax_row_stats /
            // softmax_class_gradient / add_class_delta(见 histogram.cu)。
            Objective::Softprob { .. } | Objective::Softmax { .. } => u32::MAX,
        }
    }

    /// fast kernel 的 objective 参数(没有参数的 objective 传 0)。
    pub fn device_param(self) -> f32 {
        match self {
            Objective::Poisson { max_delta_step } => max_delta_step,
            Objective::Tweedie { variance_power } => variance_power,
            Objective::PseudoHuber { slope } => slope,
            _ => 0.0,
        }
    }

    /// 模型 JSON 里 objective 下面的参数块(XGBoost `SaveConfig`)。
    /// XGBoost 加载时会按名字去找它,缺了会报错。
    ///
    /// RegLossObj 家族写训练时的实际 `scale_pos_weight`(最短 f32 表示,
    /// 和其它参数块一致);其它 objective 不带这个参数。
    pub fn json_params(self, scale_pos_weight: f32) -> BTreeMap<String, serde_json::Value> {
        match self {
            Objective::SquaredError | Objective::Logistic | Objective::Gamma => json_block(
                "reg_loss_param",
                "scale_pos_weight",
                scale_pos_weight.to_string(),
            ),
            Objective::Poisson { max_delta_step } => json_block(
                "poisson_regression_param",
                "max_delta_step",
                max_delta_step.to_string(),
            ),
            Objective::Tweedie { variance_power } => json_block(
                "tweedie_regression_param",
                "tweedie_variance_power",
                variance_power.to_string(),
            ),
            Objective::PseudoHuber { slope } => {
                json_block("pseudo_huber_param", "huber_slope", slope.to_string())
            }
            Objective::Softprob { num_class } | Objective::Softmax { num_class } => json_block(
                "softmax_multiclass_param",
                "num_class",
                num_class.to_string(),
            ),
        }
    }

    /// 读模型时恢复 `reg_loss_param.scale_pos_weight`(缺了就是 1)。
    pub fn scale_pos_weight_from_json(rest: &BTreeMap<String, serde_json::Value>) -> f32 {
        json_param(rest, "reg_loss_param", "scale_pos_weight").unwrap_or(1.0)
    }

    /// 读模型时按 JSON 参数块恢复 objective 参数(缺了就用默认值)。
    pub fn from_json(name: &str, rest: &BTreeMap<String, serde_json::Value>) -> Option<Objective> {
        Some(match Self::from_name(name)? {
            Objective::Poisson { max_delta_step } => Objective::Poisson {
                max_delta_step: json_param(rest, "poisson_regression_param", "max_delta_step")
                    .unwrap_or(max_delta_step),
            },
            Objective::Tweedie { variance_power } => Objective::Tweedie {
                variance_power: json_param(
                    rest,
                    "tweedie_regression_param",
                    "tweedie_variance_power",
                )
                .unwrap_or(variance_power),
            },
            Objective::PseudoHuber { slope } => Objective::PseudoHuber {
                slope: json_param(rest, "pseudo_huber_param", "huber_slope").unwrap_or(slope),
            },
            Objective::Softprob { .. } => Objective::Softprob {
                num_class: json_param(rest, "softmax_multiclass_param", "num_class")? as u32,
            },
            Objective::Softmax { .. } => Objective::Softmax {
                num_class: json_param(rest, "softmax_multiclass_param", "num_class")? as u32,
            },
            other => other,
        })
    }
}

// ---------------------------------------------------------------- 多分类
//
// XGBoost 3.4.1 `SoftmaxMultiClassObj`(`src/objective/multiclass_obj.cu`),
// `one_output_per_tree`:每轮 K 棵树,第 k 棵拟合第 k 类的梯度列。

impl Objective {
    /// 一轮的 softmax 梯度。`class_preds[k][i]` 是第 i 行第 k 类的 raw margin,
    /// `out[k][i]` 收第 k 类的梯度(已乘权重)。逐位照搬 `GetGradient`:
    ///
    /// - `wmax` 从 `FLT_MIN`(最小正规格化数,不是 −∞!)起取 `fmaxf`;
    /// - `wsum` 在 f64 里累加 f32 的 `expf(m − wmax)`;
    /// - `p = expf(m − wmax) / (float)wsum`;
    /// - `h = max(2·p·(1−p)·w, 1e-16)` —— eps 在乘权重**之后**;
    /// - `g = (p − [y == k])·w`。
    pub fn softmax_gradients(
        self,
        class_preds: &[Vec<f32>],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [Vec<GradPair>],
    ) {
        use rayon::prelude::*;
        let k_count = class_preds.len();
        debug_assert_eq!(k_count, self.num_outputs());
        let n = labels.len();
        // 按行并行,每行写 K 个类各自的列;行之间没有归约,结果与线程数无关。
        let per_row: Vec<Vec<GradPair>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut wmax = f32::MIN_POSITIVE;
                for preds in class_preds {
                    wmax = preds[i].max(wmax);
                }
                let mut wsum = 0.0f64;
                for preds in class_preds {
                    wsum += (preds[i] - wmax).exp() as f64;
                }
                let label = labels[i];
                let wt = weights.map_or(1.0f32, |w| w[i]);
                (0..k_count)
                    .map(|k| {
                        let mut p = (class_preds[k][i] - wmax).exp() / wsum as f32;
                        let h = (2.0 * p * (1.0 - p) * wt).max(1e-16);
                        if label == k as f32 {
                            p -= 1.0;
                        }
                        GradPair {
                            grad: p * wt,
                            hess: h,
                        }
                    })
                    .collect()
            })
            .collect();
        for (k, col) in out.iter_mut().enumerate() {
            col.clear();
            col.extend(per_row.iter().map(|r| r[k]));
        }
    }

    /// 默认的每类 base margin(XGBoost `SoftmaxMultiClassObj::InitEstimation`):
    /// 加权类别频率 → `log(freq + kRtEps)` → 减去均值。算术:
    /// 计数在 f32 里串行累加;`freq = (float)(count · (1/Σw))`;
    /// 均值按 `0 + v0/K + v1/K + …`(f32)。
    pub fn initial_class_margins(self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
        let k_count = self.num_outputs();
        let mut bins = vec![0.0f32; k_count];
        for (i, &y) in labels.iter().enumerate() {
            bins[y as usize] += weights.map_or(1.0f32, |w| w[i]);
        }
        let sum_w = match weights {
            None => labels.len() as f64,
            Some(w) => w.iter().fold(0.0f64, |acc, &v| acc + v as f64),
        };
        if sum_w < K_RT_EPS as f64 {
            // XGBoost 在这里 CHECK 失败;我们退回全 0(均匀先验),
            // 和单输出时「全零权重退回默认值」一致。
            return vec![0.0; k_count];
        }
        let inv = 1.0 / sum_w;
        let logs: Vec<f32> = bins
            .iter()
            .map(|&b| ((b as f64 * inv) as f32 + K_RT_EPS).ln())
            .collect();
        let n = k_count as f32;
        let mean = logs.iter().fold(0.0f32, |acc, &v| acc + v / n);
        logs.iter().map(|&v| v - mean).collect()
    }

    /// 一行 K 个 margin → 预测值(XGBoost `PredTransform`)。`softprob` 原地
    /// 变成概率(`common::Softmax`:`wmax` 取真最大值、f64 累加 `expf`、再除);
    /// `softmax` 返回最大值的下标(并列取第一个),概率留在 `row` 里不用。
    pub fn transform_row(self, row: &mut [f32]) -> Option<f32> {
        match self {
            Objective::Softprob { .. } => {
                softmax_in_place(row);
                None
            }
            Objective::Softmax { .. } => Some(argmax(row) as f32),
            _ => None,
        }
    }
}

/// XGBoost `common::Softmax`(用于预测和多分类指标)。
pub fn softmax_in_place(row: &mut [f32]) {
    let mut wmax = row[0];
    for &v in &row[1..] {
        wmax = v.max(wmax);
    }
    let mut wsum = 0.0f64;
    for v in row.iter_mut() {
        *v = (*v - wmax).exp();
        wsum += *v as f64;
    }
    let d = wsum as f32;
    for v in row.iter_mut() {
        *v /= d;
    }
}

/// XGBoost `common::FindMaxIndex`:第一个最大值的下标。
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (k, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for o in Objective::ALL {
            assert_eq!(Objective::from_name(o.name()), Some(o));
            assert_eq!(Objective::from_json(o.name(), &o.json_params(1.0)), Some(o));
        }
        assert_eq!(
            Objective::from_name("reg:linear"),
            Some(Objective::SquaredError)
        );
        assert_eq!(Objective::from_name("rank:pairwise"), None);
        let t = Objective::Tweedie {
            variance_power: 1.25,
        };
        assert_eq!(Objective::from_json(t.name(), &t.json_params(1.0)), Some(t));
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
        assert_eq!(codes, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn poisson_max_delta_step_defaults_only_when_unset() {
        let p = Objective::from_name("count:poisson").unwrap();
        assert_eq!(
            p.with_params(None, 1.5, 1.0, 0),
            Objective::Poisson {
                max_delta_step: 0.7
            }
        );
        assert_eq!(
            p.with_params(Some(0.0), 1.5, 1.0, 0),
            Objective::Poisson {
                max_delta_step: 0.0
            }
        );
    }
}
