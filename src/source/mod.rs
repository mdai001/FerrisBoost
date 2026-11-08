//! 数据源 —— 这个项目最实际的卖点所在。
//!
//! 目标:S3 上的 Parquet 直接进训练,不落地、不转 libsvm、
//! 内存不翻倍。现在的常见流程是 S3 -> pandas -> numpy -> DMatrix,
//! 中间内存翻两三倍,几百 GB 的数据光这步就装不下。

pub mod arrow_source;
pub mod csv_source;
pub mod dense;
pub mod input;
pub mod openprof;
pub(crate) mod parallel_reader;
mod parquet_io;
pub mod parquet_source;
pub mod predict_input;

pub use dense::DenseSource;

/// 文件里不当特征用的列:标签,以及可选的样本权重列(XGBoost `weight`)。
///
/// 两者都从特征里排除 —— 拿权重当特征训练,树会照着它分裂。
/// `&str` 自动转成「只有标签」,所以只认标签的老调用点不用改。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Targets<'a> {
    pub label: &'a str,
    pub weight: Option<&'a str>,
    /// Optional query-id column (ranking). Excluded from the features like
    /// `label` and `weight`; read separately, never binned.
    pub qid: Option<&'a str>,
    /// 给了就**按列名**取特征,并排成这个顺序(验证集要和训练集的
    /// 特征一一对上)。`None` 时按文件里的物理顺序。
    pub features: Option<&'a [String]>,
    /// Categorical flag per feature, in the resolved feature order (label and
    /// weight excluded). `None` = none declared; Arrow dictionary columns are
    /// categorical regardless.
    pub categorical: Option<&'a [bool]>,
    /// XGBoost `missing`: a feature value equal to this is treated exactly like
    /// NaN / null. NaN (the default) means no extra sentinel. Applies to feature
    /// columns only, never to the label or weight column.
    pub missing: f32,
    /// File columns holding a per-row starting margin (`base_margin`, one per
    /// model output). Like the label and weight they are excluded from the
    /// features; they are read separately as plain f32 columns.
    pub base_margin: Option<&'a [String]>,
}

impl<'a> Targets<'a> {
    pub fn new(label: &'a str, weight: Option<&'a str>) -> anyhow::Result<Self> {
        crate::ensure_input!(
            weight != Some(label),
            "weight 列和 label 列不能是同一列 {label:?}"
        );
        Ok(Self {
            label,
            weight,
            qid: None,
            features: None,
            categorical: None,
            missing: f32::NAN,
            base_margin: None,
        })
    }

    /// Name the `base_margin` column(s): excluded from the features.
    pub fn with_base_margin(self, names: &'a [String]) -> Self {
        Self {
            base_margin: (!names.is_empty()).then_some(names),
            ..self
        }
    }

    /// Exclude `qid` (a query-id column) from the features.
    pub fn with_qid(self, qid: Option<&'a str>) -> Self {
        Self { qid, ..self }
    }

    /// Treat feature values equal to `missing` as missing (see the field docs).
    pub fn with_missing(self, missing: f32) -> Self {
        Self { missing, ..self }
    }

    /// Declare categorical features, one flag per feature in the resolved
    /// feature order. An empty slice means none declared.
    pub fn with_categorical(self, flags: &'a [bool]) -> Self {
        Self {
            categorical: (!flags.is_empty()).then_some(flags),
            ..self
        }
    }

    /// 要求文件里的特征列正好是 `names`(按名字绑定,顺序随 `names`)。
    pub fn with_features(self, names: &'a [String]) -> Self {
        Self {
            features: Some(names),
            ..self
        }
    }

    /// 从文件 schema 里挑出特征列。`columns` 是 schema 的全部列名(物理顺序)。
    ///
    /// 列名重复直接报错:按名字绑定时第二个同名列永远绑不到,
    /// 静默用第一个只会训出错模型。
    pub(crate) fn select_features(
        &self,
        columns: &[&str],
    ) -> anyhow::Result<(Vec<usize>, Vec<String>)> {
        let mut seen = std::collections::HashSet::new();
        if let Some(dup) = columns.iter().find(|n| !seen.insert(**n)) {
            crate::bail_input!("文件里有重复的列名 {dup:?};列名必须唯一");
        }
        let mut idx = Vec::new();
        let mut names = Vec::new();
        for (i, n) in columns.iter().enumerate() {
            if !self.excludes(n) {
                idx.push(i);
                names.push(n.to_string());
            }
        }
        crate::ensure_input!(
            !idx.is_empty(),
            "除了标签列 {:?} 之外没有别的列了",
            self.label
        );
        let Some(want) = self.features else {
            return Ok((idx, names));
        };
        let missing: Vec<&str> = want
            .iter()
            .filter(|w| !names.contains(w))
            .map(String::as_str)
            .collect();
        let extra: Vec<&str> = names
            .iter()
            .filter(|n| !want.contains(n))
            .map(String::as_str)
            .collect();
        crate::ensure_input!(
            missing.is_empty() && extra.is_empty(),
            "特征列和训练集对不上(按列名绑定):缺少 {missing:?},多出 {extra:?}"
        );
        let idx = want
            .iter()
            .map(|w| idx[names.iter().position(|n| n == w).expect("checked")])
            .collect();
        Ok((idx, want.to_vec()))
    }

    /// 这一列是标签、权重、query id 或 base_margin(不进特征)。
    pub fn excludes(&self, name: &str) -> bool {
        name == self.label
            || self.weight == Some(name)
            || self.qid == Some(name)
            || self
                .base_margin
                .is_some_and(|cols| cols.iter().any(|c| c == name))
    }
}

impl<'a> From<&'a str> for Targets<'a> {
    fn from(label: &'a str) -> Self {
        Self {
            label,
            weight: None,
            qid: None,
            features: None,
            categorical: None,
            missing: f32::NAN,
            base_margin: None,
        }
    }
}

impl<'a> From<&'a String> for Targets<'a> {
    fn from(label: &'a String) -> Self {
        Self {
            label: label.as_str(),
            weight: None,
            qid: None,
            features: None,
            categorical: None,
            missing: f32::NAN,
            base_margin: None,
        }
    }
}

/// Validate a user-supplied `missing` sentinel: NaN (no sentinel) or a finite
/// float32 below the reserved `f32::MAX`. +/-inf and `f32::MAX` are rejected:
/// features may never hold them, so a sentinel equal to one would silently
/// change which inputs are errors.
pub fn validate_missing(missing: f64) -> anyhow::Result<f32> {
    if missing.is_nan() {
        return Ok(f32::NAN);
    }
    let narrowed = missing as f32;
    crate::ensure_input!(
        narrowed.is_finite() && narrowed != f32::MAX,
        "missing={missing} is not allowed: it must be NaN or a finite float32 value \
         (+/-inf and f32::MAX are reserved and rejected in features, so they cannot be \
         the missing marker; pick a different sentinel)"
    );
    Ok(narrowed)
}

/// Dense feature values with the `missing` sentinel mapped to NaN.
///
/// Borrows when nothing needs changing (NaN `missing`, or no value equals it);
/// otherwise returns one corrected copy, never touching the caller's buffer.
pub fn missing_as_nan(data: &[f32], missing: f32) -> std::borrow::Cow<'_, [f32]> {
    use std::borrow::Cow;
    if missing.is_nan() || !data.contains(&missing) {
        return Cow::Borrowed(data);
    }
    Cow::Owned(
        data.iter()
            .map(|&v| if v == missing { f32::NAN } else { v })
            .collect(),
    )
}

/// In-place variant for buffers the reader owns.
pub fn mask_missing_in_place(data: &mut [f32], missing: f32) {
    if missing.is_nan() {
        return;
    }
    for v in data {
        if *v == missing {
            *v = f32::NAN;
        }
    }
}

/// 校验从文件读出的权重列:不能有缺失、不能是负数(XGBoost 同样拒绝)。
pub(crate) fn check_weight_column(weights: &[f32], name: &str) -> anyhow::Result<()> {
    if let Some(i) = weights.iter().position(|w| w.is_nan()) {
        crate::bail_input!("权重列 {name:?} 第 {i} 行是缺失值;样本权重不能缺失");
    }
    if let Some(i) = weights.iter().position(|w| !w.is_finite() || *w < 0.0) {
        crate::bail_input!(
            "权重列 {name:?} 第 {i} 行是 {};样本权重必须是有限的非负数",
            weights[i]
        );
    }
    Ok(())
}

/// Public file-ingest concurrency default: zero means system-adaptive auto.
/// It remains independent of training `nthread`.
pub const DEFAULT_INGEST_THREADS: usize = 0;

/// Divide one resolved concurrency budget between an upstream decoder and a
/// downstream column transform. Their worker pools run concurrently.
pub(crate) fn pipeline_threads(total: usize) -> (usize, usize) {
    let total = if total == 0 {
        crate::threading::effective_threads(0)
    } else {
        total
    };
    match total.max(1) {
        1 => (1, 1),
        n => (n.div_ceil(2), n / 2),
    }
}

/// Stable, metadata-adaptive Parquet allocation. Long ordered tasks cannot
/// benefit from a large lookahead of blocked readers. Keep their reader window
/// small, and reclaim unavailable reader/transform slots without adding workers
/// or buffers beyond the original half-budget reader limit.
/// A reader count of one means inline decoding, not a concurrent worker.
pub(crate) fn parquet_pipeline_threads(
    total: usize,
    reader_tasks: usize,
    batches_per_task: usize,
    transform_units: usize,
    feature_width: usize,
) -> (usize, usize) {
    let total = if total == 0 {
        crate::threading::effective_threads(0)
    } else {
        total
    }
    .max(1);
    let half = total.div_ceil(2);
    // The reduced reader window paid off on dense wide scans, but did not
    // improve the narrow regression workload. Preserve its existing split.
    let lookahead = if feature_width >= 256 && batches_per_task > 2 {
        2
    } else {
        half
    };
    let readers = half.min(reader_tasks.max(1)).min(lookahead).max(1);
    let transform_budget = if readers == 1 { total } else { total - readers };
    (readers, transform_budget.min(transform_units.max(1)).max(1))
}

#[cfg(test)]
mod pipeline_tests {
    use super::parquet_pipeline_threads;

    #[test]
    fn parquet_allocation_respects_budget_work_and_reader_memory_bound() {
        for total in 1usize..=64 {
            for tasks in [0, 1, 2, 3, 17, 1000] {
                for batches in [0, 1, 2, 3, 1000] {
                    for units in [0, 1, 2, 3, 2000] {
                        for width in [0, 1, 28, 255, 256, 2000] {
                            let (readers, transforms) =
                                parquet_pipeline_threads(total, tasks, batches, units, width);
                            assert!(readers >= 1 && readers <= total.div_ceil(2));
                            assert!(readers <= tasks.max(1));
                            assert!(transforms >= 1 && transforms <= units.max(1));
                            assert!(transforms <= total);
                            if readers > 1 {
                                assert!(readers + transforms <= total);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn parquet_allocation_adapts_to_task_length_and_phase_work() {
        assert_eq!(parquet_pipeline_threads(8, 21, 62, 28, 28), (4, 4));
        assert_eq!(parquet_pipeline_threads(8, 38, 2, 2000, 2000), (4, 4));
        assert_eq!(parquet_pipeline_threads(8, 19, 3, 2000, 2000), (2, 6));
        assert_eq!(parquet_pipeline_threads(8, 19, 3, 16, 2000), (2, 6));
        assert_eq!(parquet_pipeline_threads(8, 19, 3, 2000, 255), (4, 4));
        assert_eq!(parquet_pipeline_threads(8, 19, 3, 2000, 256), (2, 6));
        assert_eq!(parquet_pipeline_threads(8, 19, 2, 2000, 2000), (4, 4));
        assert_eq!(parquet_pipeline_threads(8, 38, 2, 2, 2000), (4, 2));
        assert_eq!(parquet_pipeline_threads(8, 1, 62, 28, 28), (1, 8));
        assert_eq!(parquet_pipeline_threads(2, 38, 2, 2000, 2000), (1, 2));
        assert_eq!(parquet_pipeline_threads(1, 38, 2, 2000, 2000), (1, 1));
        let auto = parquet_pipeline_threads(0, 38, 2, 2000, 2000);
        let resolved =
            parquet_pipeline_threads(crate::threading::effective_threads(0), 38, 2, 2000, 2000);
        assert_eq!(auto, resolved);
    }

    #[test]
    fn parquet_allocation_arithmetic_handles_maximum_thread_budget() {
        for (tasks, batches, units, width) in [
            (usize::MAX, usize::MAX, usize::MAX, 255),
            (usize::MAX, usize::MAX, usize::MAX, 256),
            (0, 0, 0, 2000),
        ] {
            let (readers, transforms) =
                parquet_pipeline_threads(usize::MAX, tasks, batches, units, width);
            assert!(readers >= 1 && readers <= usize::MAX.div_ceil(2));
            assert!(readers <= tasks.max(1));
            assert!(transforms >= 1 && transforms <= units.max(1));
            if readers > 1 {
                assert!(readers.checked_add(transforms).is_some());
                assert!(transforms <= usize::MAX - readers);
            }
        }
    }
}

// 设计要点:
//
// 1. Arrow RecordBatch 是唯一输入接口。这样 Parquet、S3、DuckDB、
//    Polars、任何数据库的 ADBC 驱动全部自动支持,零拷贝。
//
// 2. Parquet 本来就是列存,按列取一组 column chunk 正是它的最优
//    访问模式 —— S3 range read 直接拿需要的字节。行分批要读全部列
//    再丢弃,IO 放大好几倍。列分块在这里不只是「能」,是「更快」。
//
// 3. Parquet footer 里有每个 column chunk 的 min/max 和 null count。
//    分箱的第一遍扫描可以用这些统计做初始估计,省掉一整轮 IO。
//    这个优化在行分块下拿不到。

#[cfg(test)]
mod targets_tests {
    use super::Targets;

    #[test]
    fn select_features_binds_by_name_and_rejects_mismatch() {
        let cols = ["label", "b", "w", "a", "c"];
        let t = Targets::new("label", Some("w")).unwrap();
        // a qid column is excluded from the features too
        let q = Targets::new("label", Some("w"))
            .unwrap()
            .with_qid(Some("b"));
        assert_eq!(
            q.select_features(&cols).unwrap(),
            (vec![3, 4], vec!["a".into(), "c".into()])
        );
        assert_eq!(
            t.select_features(&cols).unwrap(),
            (vec![1, 3, 4], vec!["b".into(), "a".into(), "c".into()])
        );
        let want: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        assert_eq!(
            t.with_features(&want).select_features(&cols).unwrap(),
            (vec![3, 1, 4], want.clone())
        );
        let err = t
            .with_features(&want)
            .select_features(&["label", "a", "b", "zz", "w"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("\"c\"") && err.contains("\"zz\""), "{err}");
        let err = t
            .select_features(&["label", "a", "a"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("重复"), "{err}");
    }
}

#[cfg(test)]
mod missing_tests {
    use super::{mask_missing_in_place, missing_as_nan, validate_missing};
    use std::borrow::Cow;

    #[test]
    fn validate_accepts_nan_and_finite_float32() {
        assert!(validate_missing(f64::NAN).unwrap().is_nan());
        assert_eq!(validate_missing(-999.0).unwrap(), -999.0);
        assert_eq!(validate_missing(0.0).unwrap(), 0.0);
        assert_eq!(validate_missing(f32::MIN as f64).unwrap(), f32::MIN);
    }

    #[test]
    fn validate_rejects_inf_f32_max_and_overflow() {
        for bad in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f32::MAX as f64,
            1e39,
            -1e39,
        ] {
            let err = validate_missing(bad).unwrap_err().to_string();
            assert!(err.contains("missing"), "{bad}: {err}");
        }
    }

    #[test]
    fn missing_as_nan_borrows_unless_a_value_matches() {
        let data = [1.0f32, -999.0, f32::NAN, 2.0];
        assert!(matches!(missing_as_nan(&data, f32::NAN), Cow::Borrowed(_)));
        assert!(matches!(missing_as_nan(&data, 7.0), Cow::Borrowed(_)));
        let mapped = missing_as_nan(&data, -999.0);
        assert!(matches!(mapped, Cow::Owned(_)));
        assert_eq!(mapped[0], 1.0);
        assert!(mapped[1].is_nan() && mapped[2].is_nan());
        assert_eq!(mapped[3], 2.0);
        assert_eq!(data[1], -999.0, "input untouched");
    }

    #[test]
    fn negative_zero_matches_a_zero_sentinel() {
        // IEEE: -0.0 == 0.0, as in XGBoost's `value == missing`.
        let mapped = missing_as_nan(&[-0.0f32, 1.0], 0.0);
        assert!(mapped[0].is_nan() && mapped[1] == 1.0);
    }

    #[test]
    fn in_place_mask_respects_nan_sentinel() {
        let mut data = vec![3.0f32, -1.0, -1.0, 4.0];
        mask_missing_in_place(&mut data, f32::NAN);
        assert_eq!(data, vec![3.0, -1.0, -1.0, 4.0]);
        mask_missing_in_place(&mut data, -1.0);
        assert!(data[1].is_nan() && data[2].is_nan() && data[0] == 3.0 && data[3] == 4.0);
    }
}
