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

use crate::rank::{PairMethod, RankKind, RankParams, NOT_SET};
use crate::types::{GradPair, Objective, QuantileAlphas};

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
/// `reg:quantileerror` 的内部平滑常数(XGBoost 3.4.1 `quantile_obj.cu` 的
/// `kSmoothingScale` / `kMinSurrogateRatio`,都不是用户参数)。
const QUANTILE_SMOOTHING_SCALE: f32 = 0.04;
const QUANTILE_MIN_SURROGATE_RATIO: f32 = 3.0e-4;

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
    pub const ALL: [Objective; 7] = [
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
        // alpha 在 `ALL` 里只是占位(XGBoost 没有默认值,必须由用户给)。
        Objective::Quantile {
            alphas: QuantileAlphas::single(0.5),
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
            Objective::Quantile { .. } => "reg:quantileerror",
            Objective::Rank { kind, .. } => kind.name(),
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
            "rank:ndcg" | "rank:pairwise" | "rank:map" => {
                let kind = match name {
                    "rank:ndcg" => RankKind::Ndcg,
                    "rank:pairwise" => RankKind::Pairwise,
                    _ => RankKind::Map,
                };
                return Some(Objective::Rank {
                    kind,
                    params: RankParams::default(),
                });
            }
            _ => {}
        }
        Self::ALL.into_iter().find(|o| o.name() == name)
    }

    /// 报错用:「a、b」形式的全部名字。
    pub fn supported_names() -> String {
        Self::ALL
            .iter()
            .map(|o| o.name())
            .chain([
                "multi:softprob",
                "multi:softmax",
                "rank:ndcg",
                "rank:pairwise",
                "rank:map",
            ])
            .collect::<Vec<_>>()
            .join("、")
    }

    /// 每行几个输出(多分类是 `num_class`,其余 1)。
    pub fn num_outputs(self) -> usize {
        match self {
            Objective::Softprob { num_class } | Objective::Softmax { num_class } => {
                num_class as usize
            }
            // 多个 alpha = 多目标:每个 alpha 一棵树(XGBoost `num_target`)。
            Objective::Quantile { alphas } => alphas.len(),
            _ => 1,
        }
    }

    /// 是不是多分类(每轮 `num_class` 棵树、预测按行做 softmax)。
    pub fn is_multiclass(self) -> bool {
        matches!(self, Objective::Softprob { .. } | Objective::Softmax { .. })
    }

    /// `reg:quantileerror`:梯度依赖每轮全体行预测归约出的标量 scale。
    pub fn is_quantile(self) -> bool {
        matches!(self, Objective::Quantile { .. })
    }

    /// 多个分位数(多目标 `reg:quantileerror`)。
    pub fn is_multi_quantile(self) -> bool {
        matches!(self, Objective::Quantile { alphas } if alphas.len() > 1)
    }

    /// `quantile_alpha` 列表(只有 `reg:quantileerror` 有)。
    pub fn quantile_alphas(self) -> Option<QuantileAlphas> {
        match self {
            Objective::Quantile { alphas } => Some(alphas),
            _ => None,
        }
    }

    /// 单个分位数的 `reg:quantileerror`(梯度 / metric 的逐目标 objective)。
    pub fn quantile(alpha: f32) -> Objective {
        Objective::Quantile {
            alphas: QuantileAlphas::single(alpha),
        }
    }

    /// `rank:*`: gradients are computed per query group (`crate::rank`).
    pub fn is_ranking(self) -> bool {
        matches!(self, Objective::Rank { .. })
    }

    /// The ranking kind and parameters, if this is a ranking objective.
    pub fn rank(self) -> Option<(RankKind, RankParams)> {
        match self {
            Objective::Rank { kind, params } => Some((kind, params)),
            _ => None,
        }
    }

    /// 带上 `quantile_alpha` 列表(只对 `reg:quantileerror` 生效)。
    pub fn with_quantile_alphas(self, alphas: &[f32]) -> anyhow::Result<Objective> {
        Ok(match self {
            Objective::Quantile { .. } => Objective::Quantile {
                alphas: QuantileAlphas::new(alphas)?,
            },
            other => other,
        })
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
            // XGBoost `QuantileLossParam::Validate`:每个值在 [0, 1],且升序(可相等)。
            Objective::Quantile { alphas } => {
                let a = alphas.as_slice();
                for &alpha in a {
                    anyhow::ensure!(
                        alpha.is_finite() && (0.0..=1.0).contains(&alpha),
                        "quantile_alpha 必须在 [0, 1] 内,给的是 {alpha}"
                    );
                }
                anyhow::ensure!(
                    a.windows(2).all(|w| w[0] <= w[1]),
                    "quantile_alpha 必须按升序排列(XGBoost 同样要求),给的是 {a:?}"
                );
            }
            Objective::Softprob { num_class } | Objective::Softmax { num_class } => {
                anyhow::ensure!(
                    num_class >= 2,
                    "{} 需要 num_class ≥ 2,给的是 {num_class}",
                    self.name()
                )
            }
            Objective::Rank { params, .. } => params.validate()?,
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
            | Objective::Quantile { .. }
            | Objective::Rank { .. }
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
        if self.is_ranking() {
            unreachable!(
                "ranking base score needs the query groups: use crate::rank::estimate_base_score"
            );
        }
        if let Objective::Quantile { alphas } = self {
            // XGBoost `QuantileRegression::InitEstimation`:标签的 alpha 分位数
            // (无权重:线性插值;有权重:阶梯函数)。权重是样本权重本身。
            return quantile_intercepts(&alphas.as_slice()[..1], labels, weights)[0];
        }
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
            Objective::Quantile { .. } => {
                unreachable!("quantile 梯度依赖每轮的全局 scale,走 Objective::quantile_gradient")
            }
            Objective::Rank { .. } => {
                unreachable!("ranking gradients are per query group: use crate::rank::RankGradient")
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
            | Objective::Quantile { .. }
            | Objective::Rank { .. }
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
                | Objective::Quantile { .. }
                | Objective::Rank { .. }
                | Objective::Softprob { .. }
                | Objective::Softmax { .. }
        )
    }

    /// 标签校验。`what` 是报错里的数据集描述,比如「训练集」「验证集 \"v\"」。
    pub fn validate_labels(self, labels: &[f32], what: &str) -> anyhow::Result<()> {
        let name = self.name();
        match self {
            Objective::SquaredError
            | Objective::PseudoHuber { .. }
            | Objective::Quantile { .. } => {}
            Objective::Rank { kind, params } => match kind {
                RankKind::Ndcg => {
                    crate::rank::check_ndcg_labels(params.ndcg_exp_gain, labels, what)?
                }
                RankKind::Map => crate::rank::check_binary_labels(name, labels, what)?,
                RankKind::Pairwise => {}
            },
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
            Objective::Quantile { .. }
            | Objective::Rank { .. }
            | Objective::Softprob { .. }
            | Objective::Softmax { .. } => {
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
            // quantile 不走单趟 `apply_delta_and_gradient`:梯度要先对全体行归约一个
            // scale,fast 用专门的三个 kernel(`quantile_apply_delta_partials` →
            // `quantile_reduce_scale` → `quantile_gradient_rows`,alpha 经
            // `device_param` 传入)。这个编号只用来在 host 侧分派,kernel 不读它。
            Objective::Quantile { .. } => 6,
            // Ranking has its own device path (or none): never the single-output kernel.
            Objective::Rank { .. } => u32::MAX,
        }
    }

    /// fast kernel 的 objective 参数(没有参数的 objective 传 0)。
    pub fn device_param(self) -> f32 {
        match self {
            Objective::Poisson { max_delta_step } => max_delta_step,
            Objective::Tweedie { variance_power } => variance_power,
            Objective::PseudoHuber { slope } => slope,
            // 单输出 fast kernel 的 alpha;多个 alpha 每个目标单独传(见
            // `Objective::quantile_alphas`)。
            Objective::Quantile { alphas } => alphas.as_slice()[0],
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
            // XGBoost 存的是 ParamArray<float> 的 JSON 数组字符串。
            Objective::Quantile { alphas } => json_block(
                "quantile_loss_param",
                "quantile_alpha",
                format!(
                    "[{}]",
                    alphas
                        .as_slice()
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            ),
            Objective::Softprob { num_class } | Objective::Softmax { num_class } => json_block(
                "softmax_multiclass_param",
                "num_class",
                num_class.to_string(),
            ),
            Objective::Rank { params, .. } => rank_json_params(&params),
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
            Objective::Quantile { .. } => Objective::Quantile {
                alphas: quantile_alphas_from_json(rest)?,
            },
            Objective::Softprob { .. } => Objective::Softprob {
                num_class: json_param(rest, "softmax_multiclass_param", "num_class")? as u32,
            },
            Objective::Softmax { .. } => Objective::Softmax {
                num_class: json_param(rest, "softmax_multiclass_param", "num_class")? as u32,
            },
            Objective::Rank { kind, .. } => Objective::Rank {
                kind,
                params: rank_params_from_json(rest)?,
            },
            other => other,
        })
    }
}

// ---------------------------------------------------------------- ranking
//
// `lambdarank_param` block of the objective JSON (XGBoost `ToJson(LambdaRankParam)`):
// every value is a string; an unset `lambdarank_num_pair_per_sample` is 4294967295.

fn bool_str(v: bool) -> String {
    (v as u8).to_string()
}

fn rank_json_params(p: &RankParams) -> BTreeMap<String, serde_json::Value> {
    let mut inner = serde_json::Map::new();
    let mut put = |k: &str, v: String| {
        inner.insert(k.into(), serde_json::Value::String(v));
    };
    put("lambdarank_bias_norm", p.bias_norm.to_string());
    put("lambdarank_normalization", bool_str(p.normalization));
    put(
        "lambdarank_num_pair_per_sample",
        p.num_pair_per_sample.unwrap_or(NOT_SET).to_string(),
    );
    put("lambdarank_pair_method", p.pair_method.name().to_string());
    put(
        "lambdarank_score_normalization",
        bool_str(p.score_normalization),
    );
    put("lambdarank_unbiased", bool_str(p.unbiased));
    put("ndcg_exp_gain", bool_str(p.ndcg_exp_gain));
    let mut rest = BTreeMap::new();
    rest.insert(
        "lambdarank_param".to_string(),
        serde_json::Value::Object(inner),
    );
    rest
}

fn rank_params_from_json(rest: &BTreeMap<String, serde_json::Value>) -> Option<RankParams> {
    let mut p = RankParams::default();
    let Some(block) = rest.get("lambdarank_param") else {
        return Some(p);
    };
    let text = |k: &str| -> Option<String> {
        match block.get(k)? {
            serde_json::Value::String(s) => Some(s.trim().to_string()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(bool_str(*b)),
            _ => None,
        }
    };
    let flag = |k: &str, default: bool| -> Option<bool> {
        match text(k) {
            None => Some(default),
            Some(t) => match t.as_str() {
                "1" | "true" | "True" => Some(true),
                "0" | "false" | "False" => Some(false),
                _ => None,
            },
        }
    };
    if let Some(m) = text("lambdarank_pair_method") {
        p.pair_method = match m.as_str() {
            "topk" => PairMethod::TopK,
            "mean" => PairMethod::Mean,
            _ => return None,
        };
    }
    if let Some(n) = text("lambdarank_num_pair_per_sample") {
        let k: u64 = n.parse().ok()?;
        p.num_pair_per_sample =
            (k != u64::from(NOT_SET)).then(|| k.min(u64::from(u32::MAX - 1)) as u32);
    }
    p.normalization = flag("lambdarank_normalization", true)?;
    p.score_normalization = flag("lambdarank_score_normalization", true)?;
    p.ndcg_exp_gain = flag("ndcg_exp_gain", true)?;
    p.unbiased = flag("lambdarank_unbiased", false)?;
    if let Some(b) = text("lambdarank_bias_norm") {
        p.bias_norm = b.parse().ok()?;
    }
    Some(p)
}

// ---------------------------------------------------------------- quantile
//
// XGBoost 3.4.1 `QuantileRegression`(`src/objective/quantile_obj.cu`):
// 平滑 pinball 损失 + MM 曲率。每轮先算一个全局 scale
//
//     S = (Σ w·sqrtf(|pred − label|) / Σ w)²
//
// (f32 逐行乘积、f64 串行累加 = XGBoost `nthread=1` 的算术),再逐行
//
//     x = r / (0.04·S),  g = w·S/2·(tanh x + 1 − 2α),  h = w/(2·0.04)·max(tanh x / x, 3e-4)
//
// S ≤ 0 或 w == 0 时 (g, h) = (0, 0)。

/// 解析 `quantile_loss_param.quantile_alpha`("[0.9]"、"0.9"、"(0.9)"、
/// "[1E-1,5E-1,9E-1]"、JSON 数组)。
fn quantile_alphas_from_json(rest: &BTreeMap<String, serde_json::Value>) -> Option<QuantileAlphas> {
    let raw = rest.get("quantile_loss_param")?.get("quantile_alpha")?;
    let values: Vec<f32> = match raw {
        serde_json::Value::String(s) => s
            .trim()
            .trim_start_matches(['[', '('])
            .trim_end_matches([']', ')'])
            .split(',')
            .map(|t| t.trim().parse::<f32>().ok())
            .collect::<Option<Vec<_>>>()?,
        serde_json::Value::Number(n) => vec![n.as_f64()? as f32],
        serde_json::Value::Array(a) => a
            .iter()
            .map(|v| v.as_f64().map(|v| v as f32))
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    QuantileAlphas::new(&values).ok()
}

/// XGBoost `common::Quantile` / `WeightedQuantile`(CPU 路径,`InitEstimation`),
/// 每个 alpha 一个截距。标签只排一次。
fn quantile_intercepts(alphas: &[f32], labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
    let n = labels.len();
    if n == 0 {
        return vec![DEFAULT_BASE_SCORE; alphas.len()];
    }
    // 稳定排序(并列保持原顺序),和 XGBoost `StableSort` 一致。
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&l, &r| labels[l].total_cmp(&labels[r]));
    let val = |i: usize| labels[idx[i]];
    // 加权:累计权重只算一次。
    let cdf: Option<Vec<f32>> = weights.map(|w| {
        let mut cdf = vec![0.0f32; n];
        cdf[0] = w[idx[0]];
        for i in 1..n {
            cdf[i] = cdf[i - 1] + w[idx[i]];
        }
        cdf
    });
    alphas
        .iter()
        .map(|&alpha| {
            let alpha = f64::from(alpha);
            match &cdf {
                None => {
                    let nf = n as f64;
                    if alpha <= 1.0 / (nf + 1.0) {
                        return val(0);
                    }
                    if alpha >= nf / (nf + 1.0) {
                        return val(n - 1);
                    }
                    let x = alpha * (nf + 1.0);
                    let k = x.floor() - 1.0;
                    let d = (x - 1.0) - k;
                    let v0 = val(k as usize);
                    let v1 = val(k as usize + 1);
                    // `v0 + d * (v1 - v0)`:f32 差,f64 乘加,最后收成 f32。
                    (f64::from(v0) + d * f64::from(v1 - v0)) as f32
                }
                Some(cdf) => {
                    let thresh = (f64::from(cdf[n - 1]) * alpha) as f32;
                    // lower_bound:第一个 cdf >= thresh。
                    let at = cdf.partition_point(|&c| c < thresh).min(n - 1);
                    val(at)
                }
            }
        })
        .collect()
}

impl Objective {
    /// 多目标 `reg:quantileerror` 的每目标截距(XGBoost `InitEstimation` 里
    /// `quantiles(t)`:第 t 个 alpha 的标签分位数;单个 alpha 就是标量截距)。
    pub fn initial_quantile_margins(self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
        match self {
            Objective::Quantile { alphas } => {
                quantile_intercepts(alphas.as_slice(), labels, weights)
            }
            _ => unreachable!("initial_quantile_margins 只用于 reg:quantileerror"),
        }
    }

    /// 每轮的全局 scale `S`(`reg:quantileerror` 专用)。`preds` 是 raw margin
    /// (恒等链接,即预测值)。串行 f64 累加:结果与线程数无关,并与 XGBoost
    /// `nthread=1` 逐位相同。
    pub fn quantile_scale(preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f32 {
        let mut stat = 0.0f64;
        let mut sum_w = 0.0f64;
        match weights {
            None => {
                for (&p, &y) in preds.iter().zip(labels) {
                    stat += f64::from((p - y).abs().sqrt());
                }
                sum_w = preds.len() as f64;
            }
            Some(w) => {
                for ((&p, &y), &wi) in preds.iter().zip(labels).zip(w) {
                    stat += f64::from(wi * (p - y).abs().sqrt());
                    sum_w += f64::from(wi);
                }
            }
        }
        if sum_w.abs() < 1e-6 {
            return 0.0;
        }
        let root_mean = stat / sum_w;
        (root_mean * root_mean) as f32
    }

    /// 单行的 quantile 梯度(**未乘**样本权重)。`scale` 来自
    /// [`Objective::quantile_scale`];scale 不为正时返回 (0, 0)。
    #[inline]
    pub fn quantile_gradient(alpha: f32, scale: f32, pred: f32, label: f32) -> GradPair {
        if !(scale > 0.0) {
            return GradPair {
                grad: 0.0,
                hess: 0.0,
            };
        }
        let residual = pred - label;
        let x = residual / (QUANTILE_SMOOTHING_SCALE * scale);
        let tanh_x = x.tanh();
        let ratio = if x == 0.0 { 1.0 } else { tanh_x / x };
        let ratio = ratio.max(QUANTILE_MIN_SURROGATE_RATIO);
        GradPair {
            grad: 0.5 * scale * (tanh_x + 1.0 - 2.0 * alpha),
            hess: 0.5 / QUANTILE_SMOOTHING_SCALE * ratio,
        }
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
            // 多分位数:每行升序排列,防止分位数交叉(XGBoost
            // `QuantileRegression::PredTransform` / `EvalTransform`)。
            Objective::Quantile { .. } => {
                sort_row_ascending(row);
                None
            }
            _ => None,
        }
    }

    /// `n × K` 行主序的 raw margin → 预测值(`transform_row` 的批量版)。
    /// `softmax` 返回每行的类别下标(`n` 个值);其它原地变换并返回 `None`。
    pub fn transform_rows(self, margins: &mut [f32], k: usize) -> Option<Vec<f32>> {
        use rayon::prelude::*;
        match self {
            Objective::Softmax { .. } => Some(
                margins
                    .par_chunks_exact(k)
                    .map(|row| argmax(row) as f32)
                    .collect(),
            ),
            Objective::Softprob { .. } => {
                margins.par_chunks_exact_mut(k).for_each(softmax_in_place);
                None
            }
            Objective::Quantile { .. } if k > 1 => {
                margins.par_chunks_exact_mut(k).for_each(sort_row_ascending);
                None
            }
            _ => None,
        }
    }
}

/// XGBoost 的逐行插入排序(`>` 比较,NaN 的相对位置与它一致)。
pub fn sort_row_ascending(row: &mut [f32]) {
    for i in 1..row.len() {
        let value = row[i];
        let mut pos = i;
        while pos > 0 && row[pos - 1] > value {
            row[pos] = row[pos - 1];
            pos -= 1;
        }
        row[pos] = value;
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
        assert!(Objective::from_name("rank:pairwise").unwrap().is_ranking());
        assert_eq!(Objective::from_name("rank:bogus"), None);
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
        let codes: Vec<u32> = Objective::ALL
            .iter()
            .filter(|o| !o.is_multiclass())
            .map(|o| o.device_code())
            .collect();
        assert_eq!(codes, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(
            Objective::quantile(0.3).device_param(),
            0.3,
            "quantile 的 alpha 经 device_param 传给 fast kernel"
        );
    }

    #[test]
    fn quantile_intercept_matches_xgboost_rules() {
        let one = |a: f32, y: &[f32], w: Option<&[f32]>| quantile_intercepts(&[a], y, w)[0];
        // n = 4:alpha <= 1/5 取最小值,>= 4/5 取最大值,否则线性插值。
        let y = [4.0f32, 1.0, 3.0, 2.0];
        assert_eq!(one(0.1, &y, None), 1.0);
        assert_eq!(one(0.9, &y, None), 4.0);
        // x = 0.5 * 5 = 2.5,k = 1,d = 0.5:2 + 0.5 * (3 - 2)
        assert_eq!(one(0.5, &y, None), 2.5);
        // 加权:阶梯函数,cdf = [1, 2, 3, 4],thresh = 2 → 第二个。
        let w = [1.0f32; 4];
        assert_eq!(one(0.5, &y, Some(&w)), 2.0);
        // 多个 alpha:每个等于单独算,标签只排一次。
        let alphas = [0.1f32, 0.5, 0.9];
        assert_eq!(quantile_intercepts(&alphas, &y, None), [1.0, 2.5, 4.0]);
        assert_eq!(quantile_intercepts(&alphas, &y, Some(&w)), [1.0, 2.0, 4.0]);
    }

    #[test]
    fn quantile_alphas_validate_parse_and_round_trip_json() {
        let o = Objective::from_name("reg:quantileerror")
            .unwrap()
            .with_quantile_alphas(&[0.1, 0.5, 0.9])
            .unwrap();
        assert_eq!(o.num_outputs(), 3);
        assert!(o.is_multi_quantile());
        o.validate_params().unwrap();
        // JSON 往返:写成 "[0.1,0.5,0.9]",读回同一个列表。
        assert_eq!(Objective::from_json(o.name(), &o.json_params(1.0)), Some(o));
        // XGBoost 的写法 "[1E-1,5E-1,9E-1]" 也读得懂。
        let mut rest = o.json_params(1.0);
        rest.insert(
            "quantile_loss_param".into(),
            serde_json::json!({"quantile_alpha": "[1E-1,5E-1,9E-1]"}),
        );
        assert_eq!(Objective::from_json(o.name(), &rest), Some(o));
        // 升序、范围、个数。
        let bad = |a: &[f32]| {
            Objective::quantile(0.5)
                .with_quantile_alphas(a)
                .and_then(|o| o.validate_params())
        };
        assert!(bad(&[0.9, 0.1]).is_err());
        assert!(bad(&[0.1, 1.5]).is_err());
        assert!(bad(&[]).is_err());
        assert!(bad(&[f32::NAN]).is_err());
        assert!(bad(&[0.5, 0.5]).is_ok(), "XGBoost 允许重复(只要求非降)");
        assert!(bad(&vec![0.5; crate::types::MAX_QUANTILE_ALPHAS + 1]).is_err());
    }

    #[test]
    fn multi_quantile_rows_are_sorted_like_xgboost_pred_transform() {
        let o = Objective::quantile(0.5)
            .with_quantile_alphas(&[0.1, 0.5, 0.9])
            .unwrap();
        let mut m = vec![3.0f32, 1.0, 2.0, 0.0, 5.0, 4.0];
        assert!(o.transform_rows(&mut m, 3).is_none());
        assert_eq!(m, [1.0, 2.0, 3.0, 0.0, 4.0, 5.0]);
        let mut row = [2.0f32, 1.0, 3.0];
        assert!(o.transform_row(&mut row).is_none());
        assert_eq!(row, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn quantile_zero_scale_gives_zero_gradient() {
        let g = Objective::quantile_gradient(0.3, 0.0, 1.0, 2.0);
        assert_eq!((g.grad, g.hess), (0.0, 0.0));
        // 预测恰好等于标签:scale = 0。
        assert_eq!(
            Objective::quantile_scale(&[1.0, 2.0], &[1.0, 2.0], None),
            0.0
        );
        assert!(Objective::quantile_scale(&[0.0, 0.0], &[1.0, 4.0], None) > 0.0);
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
