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

    /// Weighted evaluation with query groups (`group_ptr` = `n_groups + 1` row
    /// offsets). Only ranking metrics use the groups, and for them `weight` is
    /// per group. Every other metric ignores `group_ptr`.
    fn eval_grouped(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        _group_ptr: Option<&[u32]>,
    ) -> f32 {
        self.eval_weighted(pred, label, weight)
    }

    /// Reject labels the metric cannot score (e.g. non-binary labels for `map`).
    fn check_labels(&self, _labels: &[f32], _what: &str) -> anyhow::Result<()> {
        Ok(())
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
pub(crate) fn sigmoid_f32(raw: f32) -> f32 {
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

    /// With query groups XGBoost scores every group on its own
    /// ([`crate::rank::GroupAuc`]); `weight` is then per group.
    fn eval_grouped(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: Option<&[u32]>,
    ) -> f32 {
        match group_ptr {
            Some(g) => crate::rank::GroupAuc::new(crate::rank::GroupAucKind::Roc, true)
                .score(pred, label, weight, g) as f32,
            None => self.eval_weighted(pred, label, weight),
        }
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

/// 二分类 PR AUC(XGBoost `aucpr` 的 binary 分支,`BinaryPRAUC`)。
///
/// 排序和分段(相同概率合成一段)与 `auc` 相同,但每段面积用 XGBoost 的
/// `CalcDeltaPRAUC`:段内按线性插值的 FP/TP 曲线对 precision-recall 积分
/// (闭式解,含 `log` 项)。除数恒为 1,所以直接返回累加面积。
/// 只有一类样本(`sum(w·y) <= 0` 或 `sum(w·(1-y)) <= 0`)时返回 NaN。
pub struct AucPr;

/// XGBoost `detail::CalcDeltaPRAUC`,全程 f64,运算顺序照抄。
pub(crate) fn delta_pr_auc(fp_prev: f64, fp: f64, tp_prev: f64, tp: f64, total_pos: f64) -> f64 {
    let pr_prev = tp_prev / total_pos;
    let pr = tp / total_pos;
    let (a, b) = if tp == tp_prev {
        (1.0f64, 0.0f64)
    } else {
        let h = (fp - fp_prev) / (tp - tp_prev);
        (h + 1.0, (fp_prev - h * tp_prev) / total_pos)
    };
    if b != 0.0 {
        (pr - pr_prev - b / a * ((a * pr + b).ln() - (a * pr_prev + b).ln())) / a
    } else {
        (pr - pr_prev) / a
    }
}

impl Metric for AucPr {
    fn name(&self) -> &str {
        "aucpr"
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.eval_weighted(pred, label, None)
    }

    fn supports_weights(&self) -> bool {
        true
    }

    /// With query groups: per-group PR-AUC ([`crate::rank::GroupAuc`]).
    fn eval_grouped(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: Option<&[u32]>,
    ) -> f32 {
        match group_ptr {
            Some(g) => crate::rank::GroupAuc::new(crate::rank::GroupAucKind::Pr, true)
                .score(pred, label, weight, g) as f32,
            None => self.eval_weighted(pred, label, weight),
        }
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        assert_eq!(pred.len(), label.len(), "aucpr 的预测和标签长度必须一致");
        if let Some(w) = weight {
            assert_eq!(pred.len(), w.len(), "aucpr 的预测和权重长度必须一致");
        }
        if pred.is_empty() {
            return f32::NAN;
        }
        let wt = |i: usize| weight.map_or(1.0f32, |w| w[i]);
        // total_pos / total_neg:f32 乘积再累加进 f64(与 XGBoost 一致)。
        let (mut total_pos, mut total_neg) = (0.0f64, 0.0f64);
        for (i, &y) in label.iter().enumerate() {
            let w = wt(i);
            total_pos += (w * y) as f64;
            total_neg += (w * (1.0f32 - y)) as f64;
        }
        if total_pos <= 0.0 || total_neg <= 0.0 {
            return f32::NAN;
        }
        let p: Vec<f32> = pred.iter().map(|&r| sigmoid_f32(r)).collect();
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));

        let first = idx[0];
        let (y, w) = (label[first], wt(first));
        let mut fp = (1.0 - y as f64) * w as f64;
        let mut tp = (y * w) as f64;
        let (mut fp_prev, mut tp_prev) = (0.0f64, 0.0f64);
        let mut area = 0.0f64;
        for k in 1..idx.len() {
            let (i, prev) = (idx[k], idx[k - 1]);
            if p[i] != p[prev] {
                area += delta_pr_auc(fp_prev, fp, tp_prev, tp, total_pos);
                fp_prev = fp;
                tp_prev = tp;
            }
            let (y, w) = (label[i], wt(i));
            fp += ((1.0f32 - y) * w) as f64;
            tp += (y * w) as f64;
        }
        area += delta_pr_auc(fp_prev, fp, tp_prev, tp, total_pos);
        if fp <= 0.0 || tp <= 0.0 {
            area = 0.0;
        }
        area.min(1.0) as f32
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
        Objective::Poisson { .. } => Box::new(Elementwise::new(Ew::PoissonNll, obj)),
        Objective::Gamma => Box::new(Elementwise::new(Ew::GammaDeviance, obj)),
        Objective::Tweedie { variance_power } => {
            Box::new(Elementwise::new(Ew::TweedieNll(variance_power), obj))
        }
        Objective::PseudoHuber { slope } => Box::new(Elementwise::new(Ew::Mphe(slope), obj)),
        Objective::Quantile { alphas } if alphas.len() > 1 => {
            Box::new(MultiQuantile::new(alphas.as_slice()))
        }
        Objective::Quantile { alphas } => Box::new(Elementwise::new(
            Ew::Quantile(alphas.as_slice()[0]),
            obj,
        )),
        Objective::Softprob { num_class } | Objective::Softmax { num_class } => {
            Box::new(MultiClass::new(MultiKind::LogLoss, num_class as usize))
        }
        Objective::Rank { kind, params } => {
            Box::new(crate::rank::RankMetric::default_for(kind, &params))
        }
    }
}

/// XGBoost `eval_metric` 名字 → metric,按 objective 选对预测变换。
///
/// `logloss` / `error` / `auc` / `aucpr` 是概率空间的二分类指标,只接
/// `binary:logistic`;`rmse` 两个 objective 都接(二分类时和 XGBoost
/// 一样在概率上算)。不认识或组合不支持时返回说明原因的错误。
pub fn for_objective(name: &str, obj: Objective) -> Result<Box<dyn Metric>, String> {
    // Ranking metrics (`ndcg`, `map`, `pre`, with `@k` / `-`) need query groups,
    // so they pair with the ranking objectives and nothing else.
    if let Objective::Rank { params, .. } = obj {
        // `auc` / `aucpr` of a ranking objective are scored per query group.
        match name {
            "auc" => {
                return Ok(Box::new(crate::rank::GroupAuc::new(
                    crate::rank::GroupAucKind::Roc,
                    false,
                )))
            }
            "aucpr" => {
                return Ok(Box::new(crate::rank::GroupAuc::new(
                    crate::rank::GroupAucKind::Pr,
                    false,
                )))
            }
            _ => {}
        }
        return crate::rank::RankMetric::parse(name, params.ndcg_exp_gain)
            .map(|m| Box::new(m) as Box<dyn Metric>)
            .map_err(|e| {
                format!(
                    "{e}; {} supports the ranking metrics ndcg / ndcg@k / ndcg@k- / map / map@k / pre@k / auc / aucpr",
                    obj.name()
                )
            });
    }
    if crate::rank::is_rank_metric_name(name) {
        return Err(format!(
            "eval_metric {name:?} 不支持:它是排序指标,需要排序 objective \
             (rank:ndcg / rank:pairwise / rank:map)和 query groups"
        ));
    }
    // 多分位数(多目标)只接 quantile:XGBoost 的其它逐行指标要求预测和标签等长。
    if let Objective::Quantile { alphas } = obj {
        if alphas.len() > 1 {
            return match name {
                "quantile" => Ok(Box::new(MultiQuantile::new(alphas.as_slice()))),
                _ => Err(format!(
                    "eval_metric {name:?} 不能用于多个 quantile_alpha 的 reg:quantileerror:\
                     预测是 n × {} 而标签是 n;多输出模型只支持 \"quantile\"",
                    alphas.len()
                )),
            };
        }
    }
    // 多分类只接 mlogloss / merror(XGBoost 对其它指标会因为预测长度不符报错)。
    if obj.is_multiclass() {
        let k = obj.num_outputs();
        return match name {
            "mlogloss" => Ok(Box::new(MultiClass::new(MultiKind::LogLoss, k))),
            "merror" => Ok(Box::new(MultiClass::new(MultiKind::Error, k))),
            _ => Err(format!(
                "eval_metric {name:?} 不能用于 {};多分类支持 mlogloss / merror",
                obj.name()
            )),
        };
    }
    if matches!(name, "mlogloss" | "merror") {
        return Err(format!(
            "eval_metric {name:?} 是多分类指标,需要 objective=\"multi:softprob\" 或 \"multi:softmax\""
        ));
    }
    let binary = matches!(obj, Objective::Logistic);
    // 新加的逐行指标:和 XGBoost 一样在 objective 变换后的预测上算。
    if let Some(kind) = Ew::parse(name, obj)? {
        if matches!(kind, Ew::Error(_)) && !binary {
            return Err(format!(
                "eval_metric {name:?} 是二分类指标,需要 objective=\"binary:logistic\""
            ));
        }
        return Ok(Box::new(Elementwise::new(kind, obj)));
    }
    match (name, obj) {
        ("rmse", Objective::SquaredError) => Ok(Box::new(Rmse)),
        ("rmse", Objective::Logistic) => Ok(Box::new(ProbRmse)),
        ("rmse", _) => Ok(Box::new(Elementwise::new(Ew::Rmse, obj))),
        ("logloss", Objective::Logistic) => Ok(Box::new(LogLoss)),
        ("error", Objective::Logistic) => Ok(Box::new(ClassError)),
        ("auc", Objective::Logistic) => Ok(Box::new(Auc)),
        ("aucpr", Objective::Logistic) => Ok(Box::new(AucPr)),
        ("logloss" | "error" | "auc" | "aucpr", _) => Err(format!(
            "eval_metric {name:?} 是二分类指标,需要 objective=\"binary:logistic\""
        )),
        _ => Err(format!(
            "eval_metric {name:?} 不支持,现在有 {SUPPORTED_METRICS}"
        )),
    }
}

/// 报错里列出的全部 metric 名字。
pub const SUPPORTED_METRICS: &str =
    "rmse / mae / mape / mphe / logloss / error / error@t / auc / aucpr / \
poisson-nloglik / gamma-nloglik / gamma-deviance / tweedie-nloglik@rho / quantile / mlogloss / merror";

pub fn by_name(name: &str) -> Option<Box<dyn Metric>> {
    match name {
        "rmse" => Some(Box::new(Rmse)),
        "logloss" => Some(Box::new(LogLoss)),
        "error" => Some(Box::new(ClassError)),
        "auc" => Some(Box::new(Auc)),
        "aucpr" => Some(Box::new(AucPr)),
        _ => None,
    }
}

// ---------------------------------------------------------------- 逐行指标
//
// XGBoost `EvalEWiseBase` 家族(`src/metric/elementwise_metric.cu`):
// 每行在 **f32** 里算 `residue = EvalRow(label, transform(margin))`,
// 再 `residue *= w`(f32),最后在 f64 里求和、`GetFinal(esum, wsum)`。
// 逐行公式逐位照搬;求和按行号串行(XGBoost 的线程分段只影响 f64 尾数)。

// XGBoost `common::LogGamma(float)` 用平台的 `lgammaf`。我们不链接平台 libm 的
// `lgammaf`:新 glibc(2.43)给它加了新的符号版本,链接后 wheel 就只能在
// glibc ≥ 2.43 上加载,manylinux 标签也打不上。改用纯 Rust 的 `libm`(musl
// 移植),Linux / Windows 结果一致;和 XGBoost 的差别在 f32 最后一位以内。

/// 逐行指标的种类。带参数的(error@t、tweedie-nloglik@rho、mphe 的 slope)
/// 存解析后的值。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ew {
    Rmse,
    Mae,
    Mape,
    /// `mphe`,参数是 `huber_slope`(objective 是 pseudohuber 时取它的,否则 1)。
    Mphe(f32),
    PoissonNll,
    GammaDeviance,
    GammaNll,
    /// `tweedie-nloglik@rho`
    TweedieNll(f32),
    /// `error@t`(t ≠ 0.5 时名字带阈值)
    Error(f32),
    /// `quantile`,参数是 objective 的 `quantile_alpha`(pinball 损失)。
    Quantile(f32),
}

impl Ew {
    /// 解析新加的 metric 名字;不是这一族的返回 `Ok(None)`。
    fn parse(name: &str, obj: Objective) -> Result<Option<Ew>, String> {
        let (base, param) = match name.split_once('@') {
            Some((b, p)) => (b, Some(p)),
            None => (name, None),
        };
        let num = |p: &str| -> Result<f32, String> {
            p.trim()
                .parse::<f32>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| format!("eval_metric {name:?} 的参数 {p:?} 不是有限数"))
        };
        let no_param = |kind: Ew| -> Result<Option<Ew>, String> {
            match param {
                None => Ok(Some(kind)),
                Some(_) => Err(format!("eval_metric {name:?} 不接受 @ 参数")),
            }
        };
        match base {
            "mae" => no_param(Ew::Mae),
            "mape" => no_param(Ew::Mape),
            "mphe" => no_param(Ew::Mphe(match obj {
                Objective::PseudoHuber { slope } => slope,
                _ => crate::objective::HUBER_DEFAULT_SLOPE,
            })),
            "quantile" => match obj {
                Objective::Quantile { alphas } => no_param(Ew::Quantile(alphas.as_slice()[0])),
                _ => Err(
                    "eval_metric \"quantile\" 需要 objective=\"reg:quantileerror\"(它的 alpha \
                     来自 quantile_alpha)"
                        .into(),
                ),
            },
            "poisson-nloglik" => no_param(Ew::PoissonNll),
            "gamma-deviance" => no_param(Ew::GammaDeviance),
            "gamma-nloglik" => no_param(Ew::GammaNll),
            "tweedie-nloglik" => {
                let rho =
                    match param {
                        Some(p) => num(p)?,
                        None => return Err(
                            "tweedie-nloglik 要写成 tweedie-nloglik@rho,比如 tweedie-nloglik@1.5"
                                .into(),
                        ),
                    };
                if !(1.0..2.0).contains(&rho) {
                    return Err(format!("tweedie-nloglik 的 rho 必须在 [1, 2),给的是 {rho}"));
                }
                Ok(Some(Ew::TweedieNll(rho)))
            }
            "error" => match param {
                Some(p) => Ok(Some(Ew::Error(num(p)?))),
                None => Ok(None), // 普通 error 走原来的 ClassError
            },
            _ => Ok(None),
        }
    }

    /// XGBoost 的 `Name()`。
    fn name(self) -> String {
        match self {
            Ew::Rmse => "rmse".into(),
            Ew::Mae => "mae".into(),
            Ew::Mape => "mape".into(),
            Ew::Mphe(_) => "mphe".into(),
            Ew::PoissonNll => "poisson-nloglik".into(),
            Ew::GammaDeviance => "gamma-deviance".into(),
            Ew::GammaNll => "gamma-nloglik".into(),
            Ew::TweedieNll(rho) => format!("tweedie-nloglik@{}", xgb_float(rho)),
            Ew::Quantile(_) => "quantile".into(),
            Ew::Error(0.5) => "error".into(),
            Ew::Error(t) => format!("error@{}", xgb_float(t)),
        }
    }

    /// `EvalRow(label, pred)`,f32,逐位照搬。
    #[inline]
    fn row(self, y: f32, p: f32) -> f32 {
        match self {
            Ew::Rmse => {
                let d = y - p;
                d * d
            }
            Ew::Mae => (y - p).abs(),
            Ew::Mape => ((y - p) / y).abs(),
            Ew::Mphe(slope) => {
                let a = y - p;
                let r = a / slope;
                slope * slope * ((1.0 + r * r).sqrt() - 1.0)
            }
            Ew::PoissonNll => {
                let py = p.max(1e-16);
                let lg = libm::lgammaf(y + 1.0);
                lg + py - py.ln() * y
            }
            Ew::GammaDeviance => {
                let (pt, lb) = (p + 1e-6, y + 1e-6);
                (pt / lb).ln() + lb / pt - 1.0
            }
            Ew::GammaNll => {
                let py = p.max(1e-6);
                let theta: f32 = (-1.0f64 / py as f64) as f32;
                let b = -(-theta).ln();
                -(y * theta - b)
            }
            Ew::TweedieNll(rho) => {
                let a = y * ((1.0 - rho) * p.ln()).exp() / (1.0 - rho);
                let b = ((2.0 - rho) * p.ln()).exp() / (2.0 - rho);
                -a + b
            }
            Ew::Error(t) => {
                if p > t {
                    1.0 - y
                } else {
                    y
                }
            }
            // XGBoost `QuantileError`:`(a·sign·d) − (1−a)·(1−sign)·d`,d = y − p。
            Ew::Quantile(a) => {
                let d = y - p;
                let sign = f32::from(d >= 0.0);
                (a * sign * d) - (1.0 - a) * (1.0 - sign) * d
            }
        }
    }

    /// `GetFinal(esum, wsum)`。
    fn finish(self, esum: f64, wsum: f64) -> f64 {
        match self {
            Ew::Rmse => {
                if wsum == 0.0 {
                    esum.sqrt()
                } else {
                    (esum / wsum).sqrt()
                }
            }
            Ew::GammaDeviance => 2.0 * esum / if wsum <= 0.0 { 1e-6 } else { wsum },
            _ => {
                if wsum == 0.0 {
                    esum
                } else {
                    esum / wsum
                }
            }
        }
    }
}

/// XGBoost 用 `std::ostream << float` 拼 metric 名字(默认 6 位有效数字,
/// 去掉尾随 0):1.5 → "1.5",0.7 → "0.7"。
fn xgb_float(v: f32) -> String {
    let s = format!("{:.6}", v as f64);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    // 6 位**有效数字**:整数部分超过 1 位时再截;这里的参数都 < 10,够用。
    s.to_string()
}

/// 逐行指标:objective 的预测变换 + `Ew` 的逐行公式。
pub struct Elementwise {
    kind: Ew,
    link: Objective,
    name: String,
}

impl Elementwise {
    pub fn new(kind: Ew, link: Objective) -> Self {
        Self {
            kind,
            link,
            name: kind.name(),
        }
    }

    fn sums(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> (f64, f64) {
        assert_eq!(
            pred.len(),
            label.len(),
            "{} 的预测和标签长度必须一致",
            self.name
        );
        assert!(!pred.is_empty(), "{} 不能评估空数据", self.name);
        let (mut esum, mut wsum) = (0.0f64, 0.0f64);
        for (i, (&m, &y)) in pred.iter().zip(label).enumerate() {
            let wt = weight.map_or(1.0f32, |w| w[i]);
            let residue = self.kind.row(y, self.link.transform(m)) * wt;
            esum += residue as f64;
            wsum += wt as f64;
        }
        (esum, wsum)
    }
}

impl Metric for Elementwise {
    fn name(&self) -> &str {
        &self.name
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.eval_weighted(pred, label, None)
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        if let Some(w) = weight {
            assert_eq!(
                pred.len(),
                w.len(),
                "{} 的预测和权重长度必须一致",
                self.name
            );
        }
        let (esum, wsum) = self.sums(pred, label, weight);
        self.kind.finish(esum, wsum) as f32
    }
}

// ---------------------------------------------------------------- 多分位数指标
//
// XGBoost `QuantileError`(`elementwise_metric.cu`),`quantile_alpha` 有多个值时:
// 预测先按行升序排(`EvalTransform` = `PredTransform`,防交叉),然后对全部
// (行, alpha) 对求和 —— 行主序 `i = row·K + a`,`loss·w` 在 f64 里累加,权重每个
// 对各算一次,所以结果是各 alpha 加权平均损失的均值。

pub struct MultiQuantile {
    alphas: Vec<f32>,
}

impl MultiQuantile {
    pub fn new(alphas: &[f32]) -> Self {
        Self {
            alphas: alphas.to_vec(),
        }
    }
}

impl Metric for MultiQuantile {
    fn name(&self) -> &str {
        "quantile"
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.eval_weighted(pred, label, None)
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        let k = self.alphas.len();
        assert_eq!(
            pred.len(),
            label.len() * k,
            "quantile 的预测长度必须是 n × quantile_alpha 个数"
        );
        assert!(!label.is_empty(), "quantile 不能评估空数据");
        let mut row = vec![0.0f32; k];
        let (mut esum, mut wsum) = (0.0f64, 0.0f64);
        for (i, &y) in label.iter().enumerate() {
            let w = weight.map_or(1.0f32, |w| w[i]);
            row.copy_from_slice(&pred[i * k..(i + 1) * k]);
            crate::objective::sort_row_ascending(&mut row);
            for (&a, &p) in self.alphas.iter().zip(&row) {
                let d = y - p;
                let sign = f32::from(d >= 0.0);
                let loss = (a * sign * d) - (1.0 - a) * (1.0 - sign) * d;
                esum += (loss * w) as f64;
                wsum += w as f64;
            }
        }
        if wsum == 0.0 {
            esum as f32
        } else {
            (esum / wsum) as f32
        }
    }
}

// ---------------------------------------------------------------- 多分类指标
//
// XGBoost `EvalMultiLogLoss` / `EvalMatchError`(`multiclass_metric.cu`):
// 预测先按行 softmax(`EvalTransform`),每行 `EvalRow(label, probs) * w`
// 在 f64 里累加,`GetFinal = esum / wsum`。`pred` 是**按行**排的
// `n × K` raw margin。

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MultiKind {
    LogLoss,
    Error,
}

pub struct MultiClass {
    kind: MultiKind,
    num_class: usize,
}

impl MultiClass {
    pub fn new(kind: MultiKind, num_class: usize) -> Self {
        Self { kind, num_class }
    }
}

impl Metric for MultiClass {
    fn name(&self) -> &str {
        match self.kind {
            MultiKind::LogLoss => "mlogloss",
            MultiKind::Error => "merror",
        }
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.eval_weighted(pred, label, None)
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        let k = self.num_class;
        assert_eq!(
            pred.len(),
            label.len() * k,
            "{} 的预测长度必须是 n × num_class",
            self.name()
        );
        assert!(!label.is_empty(), "{} 不能评估空数据", self.name());
        let mut row = vec![0.0f32; k];
        let (mut esum, mut wsum) = (0.0f64, 0.0f64);
        for (i, &y) in label.iter().enumerate() {
            let w = weight.map_or(1.0f32, |w| w[i]);
            row.copy_from_slice(&pred[i * k..(i + 1) * k]);
            crate::objective::softmax_in_place(&mut row);
            let c = y as usize;
            let residue: f32 = match self.kind {
                MultiKind::LogLoss => {
                    let eps = 1e-16f32;
                    if row[c] > eps {
                        -row[c].ln()
                    } else {
                        -eps.ln()
                    }
                }
                MultiKind::Error => f32::from(crate::objective::argmax(&row) != c),
            };
            esum += (residue * w) as f64;
            wsum += w as f64;
        }
        (esum / wsum) as f32
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
    fn aucpr_matches_xgboost_reference_values() {
        // 来自 XGBoost 3.4.1 对同一模型的 aucpr(三段并列的预测)。
        let raw = [
            -0.0f32,
            -1.617_205_3,
            -0.0,
            1.409_715_3,
            -0.0,
            -1.617_205_3,
            1.409_715_3,
            -0.0,
            -0.0,
            -1.617_205_3,
            -1.617_205_3,
            1.409_715_3,
        ];
        let y = [1.0f32, 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1.];
        let w = [1.0f32, 2., 0.5, 3., 1., 4., 2., 1., 0.5, 1.5, 2., 1.];
        let weighted = AucPr.eval_weighted(&raw, &y, Some(&w)) as f64;
        assert!((weighted - 0.970_779_804_451_6).abs() < 1e-6, "{weighted}");
        let plain = AucPr.eval(&raw, &y) as f64;
        assert!((plain - 0.917_699_510_367_1).abs() < 1e-6, "{plain}");
        assert!(!AucPr.lower_is_better());
        assert_eq!(AucPr.name(), "aucpr");
    }

    #[test]
    fn aucpr_perfect_ties_and_degenerate_inputs() {
        let y = [0.0f32, 0.0, 1.0, 1.0];
        let good: Vec<f32> = [0.1f32, 0.2, 0.8, 0.9].iter().map(|&p| logit(p)).collect();
        assert!((AucPr.eval(&good, &y) - 1.0).abs() < 1e-6);
        // 全部并列:一段,precision 恒为正类占比 0.5
        assert!((AucPr.eval(&[0.0; 4], &y) - 0.5).abs() < 1e-6);
        // 权重整体缩放不改变结果
        let w = [2.0f32; 4];
        assert!((AucPr.eval_weighted(&good, &y, Some(&w)) - 1.0).abs() < 1e-6);
        // 只有一类 / 空数据 → NaN
        assert!(AucPr.eval(&good[..2], &y[..2]).is_nan());
        assert!(AucPr.eval(&[0.0; 2], &[1.0, 1.0]).is_nan());
        assert!(AucPr.eval(&[], &[]).is_nan());
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
    fn multi_quantile_metric_is_the_mean_of_per_alpha_losses_on_sorted_rows() {
        let alphas = [0.2f32, 0.8];
        let obj = Objective::quantile(0.5).with_quantile_alphas(&alphas).unwrap();
        let m = for_objective("quantile", obj).unwrap();
        assert_eq!(m.name(), "quantile");
        assert_eq!(default_metric(obj).name(), "quantile");
        // Row 1 has crossing margins (3 > 1): the metric sorts it to [1, 3] first.
        let pred = [0.0f32, 2.0, 3.0, 1.0, 5.0, 6.0];
        let label = [1.0f32, 2.0, 7.0];
        let w = [1.0f32, 2.0, 0.5];
        let sorted = [[0.0f32, 2.0], [1.0, 3.0], [5.0, 6.0]];
        let (mut e, mut ws) = (0.0f64, 0.0f64);
        for (i, row) in sorted.iter().enumerate() {
            for (&a, &p) in alphas.iter().zip(row) {
                let single = Elementwise::new(Ew::Quantile(a), Objective::quantile(a));
                let loss = single.eval_weighted(&[p], &[label[i]], Some(&[1.0])) * w[i];
                e += f64::from(loss);
                ws += f64::from(w[i]);
            }
        }
        let got = m.eval_weighted(&pred, &label, Some(&w));
        assert!((f64::from(got) - e / ws).abs() < 1e-6, "{got} vs {}", e / ws);
        // Other elementwise metrics need prediction and label of equal length.
        let err = for_objective("mae", obj).err().unwrap();
        assert!(err.contains("quantile"), "{err}");
        assert!(m.supports_weights() && m.lower_is_better());
    }

    #[test]
    fn metric_objective_combinations() {
        assert_eq!(
            for_objective("rmse", Objective::Logistic).unwrap().name(),
            "rmse"
        );
        assert!(for_objective("auc", Objective::SquaredError).is_err());
        // 逐行指标和 XGBoost 一样接受任何 objective(在变换后的预测上算)
        assert_eq!(
            for_objective("mae", Objective::Logistic).unwrap().name(),
            "mae"
        );
        assert!(for_objective("ndcg", Objective::Logistic).is_err());
        assert!(for_objective("error@0.7", Objective::SquaredError).is_err());
        assert_eq!(
            for_objective("error@0.7", Objective::Logistic)
                .unwrap()
                .name(),
            "error@0.7"
        );
        assert!(for_objective("tweedie-nloglik", Objective::Gamma).is_err());
        assert_eq!(
            for_objective("tweedie-nloglik@1.2", Objective::Gamma)
                .unwrap()
                .name(),
            "tweedie-nloglik@1.2"
        );
        let e = for_objective("aucpr", Objective::SquaredError)
            .err()
            .unwrap();
        assert!(e.contains("binary:logistic"), "{e}");
        assert_eq!(
            for_objective("aucpr", Objective::Logistic).unwrap().name(),
            "aucpr"
        );
        assert_eq!(by_name("aucpr").unwrap().name(), "aucpr");
        // 二分类 rmse 在概率空间:raw=0 → p=0.5,label=1 → rmse=0.5
        let m = for_objective("rmse", Objective::Logistic).unwrap();
        assert_eq!(m.eval(&[0.0], &[1.0]), 0.5);
    }
}
