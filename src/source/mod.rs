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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Targets<'a> {
    pub label: &'a str,
    pub weight: Option<&'a str>,
    /// 给了就**按列名**取特征,并排成这个顺序(验证集要和训练集的
    /// 特征一一对上)。`None` 时按文件里的物理顺序。
    pub features: Option<&'a [String]>,
    /// Categorical flag per feature, in the resolved feature order (label and
    /// weight excluded). `None` = none declared; Arrow dictionary columns are
    /// categorical regardless.
    pub categorical: Option<&'a [bool]>,
}

impl<'a> Targets<'a> {
    pub fn new(label: &'a str, weight: Option<&'a str>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            weight != Some(label),
            "weight 列和 label 列不能是同一列 {label:?}"
        );
        Ok(Self {
            label,
            weight,
            features: None,
            categorical: None,
        })
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
            anyhow::bail!("文件里有重复的列名 {dup:?};列名必须唯一");
        }
        let mut idx = Vec::new();
        let mut names = Vec::new();
        for (i, n) in columns.iter().enumerate() {
            if !self.excludes(n) {
                idx.push(i);
                names.push(n.to_string());
            }
        }
        anyhow::ensure!(
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
        anyhow::ensure!(
            missing.is_empty() && extra.is_empty(),
            "特征列和训练集对不上(按列名绑定):缺少 {missing:?},多出 {extra:?}"
        );
        let idx = want
            .iter()
            .map(|w| idx[names.iter().position(|n| n == w).expect("checked")])
            .collect();
        Ok((idx, want.to_vec()))
    }

    /// 这一列是标签或权重(不进特征)。
    pub fn excludes(&self, name: &str) -> bool {
        name == self.label || self.weight == Some(name)
    }
}

impl<'a> From<&'a str> for Targets<'a> {
    fn from(label: &'a str) -> Self {
        Self {
            label,
            weight: None,
            features: None,
            categorical: None,
        }
    }
}

impl<'a> From<&'a String> for Targets<'a> {
    fn from(label: &'a String) -> Self {
        Self {
            label: label.as_str(),
            weight: None,
            features: None,
            categorical: None,
        }
    }
}

/// 校验从文件读出的权重列:不能有缺失、不能是负数(XGBoost 同样拒绝)。
pub(crate) fn check_weight_column(weights: &[f32], name: &str) -> anyhow::Result<()> {
    if let Some(i) = weights.iter().position(|w| w.is_nan()) {
        anyhow::bail!("权重列 {name:?} 第 {i} 行是缺失值;样本权重不能缺失");
    }
    if let Some(i) = weights.iter().position(|w| !w.is_finite() || *w < 0.0) {
        anyhow::bail!(
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
