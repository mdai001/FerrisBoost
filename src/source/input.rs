//! 公开输入规格 → 一组确定顺序的文件。
//!
//! **Python 只递路径,发现和校验都在 Rust。** 一个 spec 可以是:
//! 单个文件、目录、glob(`*` / `?`),或者以上几种的列表。
//!
//! 三条规矩:
//!
//! 1. **顺序必须确定。** 目录和 glob 的结果按路径字典序排,否则同一份数据
//!    在两台机器上会得到不同的行顺序,模型也就不可复现 —— 而这个项目的
//!    核心不变量之一就是可复现。
//! 2. **空结果是错误,不是空数据集。** 拼错的 glob 静默训出一个空模型,
//!    比报错糟糕得多。
//! 3. **格式必须一致。** 一次训练里混 parquet 和 csv 没有合理语义,直接拒。

use anyhow::{bail, Context, Result};
use arrow::record_batch::RecordBatch;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// 支持的输入格式。`.csv.gz` 是 CSV 的压缩形式,不是第三种格式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFormat {
    Parquet,
    Csv,
}

impl InputFormat {
    /// 从后缀认格式。`None` = 不是我们认识的数据文件(目录扫描时用来跳过)。
    pub fn of(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        if name.ends_with(".parquet") || name.ends_with(".pq") {
            Some(Self::Parquet)
        } else if name.ends_with(".csv") || name.ends_with(".csv.gz") {
            Some(Self::Csv)
        } else {
            None
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Parquet => "parquet",
            Self::Csv => "csv",
        }
    }
}

/// 这个文件是不是 gzip 压缩的。
pub fn is_gzipped(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.to_ascii_lowercase().ends_with(".gz"))
        .unwrap_or(false)
}

/// 解析结果:确定顺序的文件列表 + 统一格式。
#[derive(Clone, Debug)]
pub struct ResolvedInput {
    pub paths: Vec<PathBuf>,
    pub format: InputFormat,
}

/// 把一组 spec 展开成确定顺序的文件列表。
pub fn resolve(specs: &[String]) -> Result<ResolvedInput> {
    if specs.is_empty() {
        bail!("没有给任何输入路径");
    }
    // BTreeSet 同时负责去重和排序 —— 顺序确定这条不能靠调用方自觉。
    let mut found: BTreeSet<PathBuf> = BTreeSet::new();
    for spec in specs {
        let expanded = expand_one(spec)?;
        if expanded.is_empty() {
            bail!("{spec:?} 没有匹配到任何 .parquet / .csv / .csv.gz 文件");
        }
        found.extend(expanded);
    }
    let paths: Vec<PathBuf> = found.into_iter().collect();

    // 格式必须一致:混着训没有合理语义。
    let mut format = None;
    for p in &paths {
        let f = InputFormat::of(p)
            .with_context(|| format!("认不出 {} 的格式,支持 .parquet / .csv / .csv.gz", p.display()))?;
        match format {
            None => format = Some(f),
            Some(prev) if prev != f => bail!(
                "一次训练不能混用不同格式:同时出现了 {} 和 {}",
                prev.label(),
                f.label()
            ),
            _ => {}
        }
    }
    Ok(ResolvedInput { paths, format: format.expect("非空") })
}

/// 单个 spec:glob → 目录 → 普通文件。
fn expand_one(spec: &str) -> Result<Vec<PathBuf>> {
    let path = Path::new(spec);
    if spec.contains('*') || spec.contains('?') {
        return expand_glob(spec);
    }
    if path.is_dir() {
        return list_dir(path);
    }
    if !path.exists() {
        bail!("输入文件不存在:{spec}");
    }
    // 显式给出的单文件必须是认识的格式 —— 这里不静默跳过。
    InputFormat::of(path)
        .with_context(|| format!("认不出 {spec} 的格式,支持 .parquet / .csv / .csv.gz"))?;
    Ok(vec![path.to_path_buf()])
}

/// 目录:只收**直接子项**里认识的数据文件,不递归。
///
/// 不递归是刻意的:递归会把 `_SUCCESS`、临时目录、嵌套分区一并卷进来,
/// 而那需要一套分区语义,不属于 v0.0.1。
fn list_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("读不了目录 {}", dir.display()))? {
        let p = entry?.path();
        if p.is_file() && InputFormat::of(&p).is_some() {
            out.push(p);
        }
    }
    Ok(out)
}

/// 只支持**最后一段**带 `*` / `?` 的 glob(`data/*.parquet`),
/// 不支持 `**` 递归 —— 递归要配一套分区语义,v0.0.1 不做。
fn expand_glob(spec: &str) -> Result<Vec<PathBuf>> {
    let path = Path::new(spec);
    let pattern = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("glob 的最后一段为空")?;
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    if dir.to_string_lossy().contains('*') || dir.to_string_lossy().contains('?') {
        bail!("只支持文件名部分带 glob(如 data/*.parquet),目录部分不支持:{spec}");
    }
    if !dir.is_dir() {
        bail!("glob 的目录部分不存在:{}", dir.display());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("读不了目录 {}", dir.display()))? {
        let p = entry?.path();
        let name = match p.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if p.is_file() && glob_match(pattern, name) {
            out.push(p);
        }
    }
    Ok(out)
}

/// `*`(任意多字符)和 `?`(单字符)的匹配。刻意不引 glob crate:
/// 这里只需要这两个通配符,而依赖是要长期维护的。
fn glob_match(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    // 经典的双指针回溯,O(len) 摊还,不递归。
    let (mut pi, mut ni) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ni < n.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = pi;
            mark = ni;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ni = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}


/// ⚠️ **投影只是"选哪些列",不是"按什么顺序给"。** Arrow 无论你传的下标
/// 是什么顺序,返回的列都按文件里的物理顺序排。所以按名字绑定必须分两步:
/// 先用**升序去重**的下标去投影,再按模型的规范顺序把列重排回来。
///
/// 少了第二步,列序不同的文件会**静默**算出错误的预测 —— 不报错,只是
/// 每个特征都读到了别人的值。这正是测试里"重排列后结果不一致"抓到的东西。
pub fn projection_and_order(proj: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut mask: Vec<usize> = proj.to_vec();
    mask.sort_unstable();
    mask.dedup();
    // order[i] = 模型第 i 个特征在投影结果里的位置。
    let order = proj
        .iter()
        .map(|c| mask.iter().position(|m| m == c).expect("mask 由 proj 构造"))
        .collect();
    (mask, order)
}

/// 按 `order` 把投影后的列重排成模型的规范顺序。
pub fn reorder(batch: &RecordBatch, order: &[usize]) -> RecordBatch {
    let cols: Vec<_> = order.iter().map(|&i| batch.column(i).clone()).collect();
    let fields: Vec<_> = order
        .iter()
        .map(|&i| batch.schema().field(i).clone())
        .collect();
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(fields));
    RecordBatch::try_new(schema, cols).expect("重排只是换列顺序,长度和类型都不变")
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_the_usual_shapes() {
        assert!(glob_match("*.parquet", "a.parquet"));
        assert!(glob_match("part-*.csv", "part-001.csv"));
        assert!(glob_match("d?.csv", "d1.csv"));
        assert!(glob_match("*", "anything"));
        assert!(!glob_match("*.parquet", "a.csv"));
        assert!(!glob_match("d?.csv", "d12.csv"));
        // 星号可以吃掉零个字符
        assert!(glob_match("a*b", "ab"));
        assert!(glob_match("*a*b*", "xxayybzz"));
    }

    #[test]
    fn format_is_recognised_including_csv_gz() {
        assert_eq!(InputFormat::of(Path::new("x.parquet")), Some(InputFormat::Parquet));
        assert_eq!(InputFormat::of(Path::new("x.pq")), Some(InputFormat::Parquet));
        assert_eq!(InputFormat::of(Path::new("x.csv")), Some(InputFormat::Csv));
        assert_eq!(InputFormat::of(Path::new("x.CSV.GZ")), Some(InputFormat::Csv));
        assert_eq!(InputFormat::of(Path::new("x.txt")), None);
        assert!(is_gzipped(Path::new("x.csv.gz")));
        assert!(!is_gzipped(Path::new("x.csv")));
    }

    #[test]
    fn missing_input_is_an_error_not_an_empty_dataset() {
        let err = resolve(&["/nonexistent/nope.parquet".into()]).unwrap_err();
        assert!(err.to_string().contains("不存在"), "{err}");
    }

    #[test]
    fn empty_spec_list_is_rejected() {
        assert!(resolve(&[]).is_err());
    }
}
