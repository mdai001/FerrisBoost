//! 和 XGBoost CPU hist 的**位精确**回归基线。
//!
//! ⚠️ 这些测试断言的是**位相同**,不是数值接近。如果它们变红,
//! **不要放宽容差** —— 说明累加语义被改动了,先搞清楚为什么。
//!
//! 位相同不是运气:直方图走定点整数累加(`types::GradPairFixed`),
//! 整数加法没有舍入误差、和顺序无关,所以我们和 XGBoost 算的是同一个
//! 确定性函数。会让它变红的典型改动:
//!
//! - 把累加改回浮点,或者改动缩放因子的算法
//! - 并行 / 乱序归约(阶段 3 的 GPU kernel 尤其要盯这条)
//! - 分裂枚举顺序、并列时的取舍规则
//! - 增益公式(比如"顺手"把 0.5 补回去)
//!
//! **前提:分箱无损。** fixture 用的是每列 8 个唯一值的低基数数据,
//! 两边的 cuts 必然一致。连续特征 + max_bin=255 时我们的 sketch 和
//! XGBoost 的不一样,端到端不会位相同 —— 那种情况要用
//! `BinningStrategy::Provided` 注入它的 cuts 才谈得上位精确对拍,
//! 见 `internal-docs/tools/compare_continuous.py`。
//!
//! 基准值由 `internal-docs/tools/gen_fixtures.py` 用 xgboost 3.4.1 生成,存成小端
//! f32 裸二进制。测试本身不碰 Python。

use std::fs;
use std::path::{Path, PathBuf};

use ferrisboost::comm::Local;
use ferrisboost::source::DenseSource;
use ferrisboost::train::{train, TrainConfig};
use ferrisboost::types::{Objective, TrainParams};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("读不了 {}:{e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} 不是 f32 的整数倍", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Case {
    name: String,
    n_rows: usize,
    n_features: usize,
    objective: Objective,
    params: TrainParams,
    base_score: f32,
}

fn load_index() -> Vec<Case> {
    let raw = fs::read_to_string(fixtures_dir().join("index.json"))
        .expect("fixture 不在?跑 uv run python internal-docs/tools/gen_fixtures.py 重新生成");
    let idx: serde_json::Value = serde_json::from_str(&raw).unwrap();
    idx["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().unwrap().to_string(),
            n_rows: c["n_rows"].as_u64().unwrap() as usize,
            n_features: c["n_features"].as_u64().unwrap() as usize,
            objective: match c["objective"].as_str().unwrap() {
                "binary:logistic" => Objective::Logistic,
                "reg:squarederror" => Objective::SquaredError,
                other => panic!("fixture 里有没实现的 objective:{other}"),
            },
            params: TrainParams {
                n_rounds: c["n_rounds"].as_u64().unwrap() as usize,
                max_depth: c["max_depth"].as_u64().unwrap() as u32,
                max_bin: c["max_bin"].as_u64().unwrap() as u32,
                learning_rate: c["eta"].as_f64().unwrap() as f32,
                lambda: c["lambda"].as_f64().unwrap() as f32,
                gamma: c["gamma"].as_f64().unwrap() as f32,
                min_child_weight: c["min_child_weight"].as_f64().unwrap() as f32,
                cols_per_block: c["cols_per_block"].as_u64().unwrap() as usize,
                ..Default::default()
            },
            base_score: c["base_score"].as_f64().unwrap() as f32,
        })
        .collect()
}

/// 训一遍,返回每行的 raw margin。
fn margins(case: &Case) -> Vec<f32> {
    margins_with_blocks(case, case.params.cols_per_block)
}

/// 指定列块大小训一遍。列分块是这个项目的核心结构,`cols_per_block`
/// 只该影响内存和并行粒度,**不该影响结果**。
fn margins_with_blocks(case: &Case, cols_per_block: usize) -> Vec<f32> {
    margins_with(case, cols_per_block, 1)
}

/// 指定列块大小和线程数训一遍。
fn margins_with(case: &Case, cols_per_block: usize, nthread: usize) -> Vec<f32> {
    let dir = fixtures_dir();
    let x = read_f32(&dir.join(format!("{}.x", case.name)));
    let y = read_f32(&dir.join(format!("{}.y", case.name)));
    assert_eq!(x.len(), case.n_rows * case.n_features);
    assert_eq!(y.len(), case.n_rows);

    let src = DenseSource::from_row_major(
        &x,
        case.n_rows,
        case.n_features,
        case.params.max_bin,
        cols_per_block,
    );
    let params = TrainParams { cols_per_block, nthread, ..case.params.clone() };
    let cfg = TrainConfig {
        base_score: case.base_score,
        ..TrainConfig::new(params, case.objective)
    };
    let model = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
    (0..case.n_rows)
        .map(|r| model.predict_margin(&x[r * case.n_features..(r + 1) * case.n_features]))
        .collect()
}

/// 逐位比。**不要**换成 abs_diff < eps —— 这个测试的全部意义就在于
/// 精确相等,用容差断言等于没锁。
fn assert_bit_exact(case: &Case, got: &[f32]) {
    let want = read_f32(&fixtures_dir().join(format!("{}.margin", case.name)));
    assert_eq!(got.len(), want.len(), "{}:行数对不上", case.name);
    for (row, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{name} 第 {row} 行:我们 {g} ({gb:#010x}),xgboost {w} ({wb:#010x})。\n\
             这是位精确基线 —— 别放宽容差,先查累加语义是不是被改了。",
            name = case.name,
            gb = g.to_bits(),
            wb = w.to_bits(),
        );
    }
}

/// A 表:4 个 seed × 有无缺失 × 两个 objective,depth 3 / 10 轮。
#[test]
fn bit_exact_vs_xgboost_matrix() {
    let cases: Vec<Case> = load_index().into_iter().filter(|c| c.n_rows == 200).collect();
    assert_eq!(cases.len(), 16, "A 表应该有 16 个配置");
    for case in &cases {
        assert_bit_exact(case, &margins(case));
    }
}

/// 深树 + 多轮 + 大数据。小 case 抓不到的累积性偏差在这里冒头:
/// 5000 行 × depth 8 × 50 轮,每棵树几百个节点,任何一处累加语义
/// 变了都会在某一轮翻掉某个近似并列的分裂,然后一路放大。
#[test]
fn bit_exact_vs_xgboost_deep_and_long() {
    let cases: Vec<Case> = load_index().into_iter().filter(|c| c.n_rows == 5000).collect();
    assert_eq!(cases.len(), 2);
    for case in &cases {
        assert_bit_exact(case, &margins(case));
    }
}

/// 顺序无关:同一份数据换一个行顺序喂直方图,结果必须**位相同**。
///
/// 这是定点累加最直接的性质,也是阶段 3 敢并行归约的依据。浮点做不到
/// 这一点,所以它变红基本等于"有人把累加改回浮点了"。
#[test]
fn bit_exact_histogram_is_row_order_independent() {
    use ferrisboost::hist;
    use ferrisboost::types::{GradPair, GradPairFixed, GradQuantizer, RowId, MISSING_BIN};

    let n_rows = 512usize;
    let n_feats = 3usize;
    let feat_offsets = vec![0u32, 4, 9, 16];
    let width = 16usize;

    // 确定性的伪随机数据,不引入 rand 依赖
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut block = vec![0u8; n_feats * n_rows];
    for f in 0..n_feats {
        let n_bins = (feat_offsets[f + 1] - feat_offsets[f]) as u32;
        for row in 0..n_rows {
            block[f * n_rows + row] = if next() % 100 < 15 {
                MISSING_BIN
            } else {
                (next() % n_bins) as u8
            };
        }
    }
    let float: Vec<GradPair> = (0..n_rows)
        .map(|_| GradPair {
            grad: (next() % 20_000) as f32 / 10_000.0 - 1.0,
            hess: 0.05 + (next() % 1000) as f32 / 5000.0,
        })
        .collect();
    let q = GradQuantizer::new(&float, n_rows);
    let gpair: Vec<GradPairFixed> = float.iter().map(|&g| q.to_fixed(g)).collect();

    let ascending: Vec<RowId> = (0..n_rows as RowId).collect();
    let mut shuffled = ascending.clone();
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, (next() as usize) % (i + 1));
    }
    assert_ne!(ascending, shuffled, "洗牌没生效,这个测试就没意义了");

    let mut a = vec![GradPairFixed::default(); width];
    let mut b = vec![GradPairFixed::default(); width];
    hist::build(&block, n_rows, n_feats, &feat_offsets, &gpair, Some(&ascending), &mut a);
    hist::build(&block, n_rows, n_feats, &feat_offsets, &gpair, Some(&shuffled), &mut b);
    assert_eq!(a, b, "换行顺序结果就变了 —— 累加不再是顺序无关的");

    assert_eq!(
        hist::node_sum(&gpair, Some(&ascending)),
        hist::node_sum(&gpair, Some(&shuffled)),
    );
}

/// 列块怎么切,结果都必须**一模一样** —— 而且一样到位。
///
/// 这条守的是整个项目最核心的那个结构假设:列分块是内存和并行的单位,
/// 不是算法的一部分。`cols_per_block` 从 1(每个特征一个块)到大于特征
/// 总数(全塞一个块)扫一遍,都得和 XGBoost 的基准逐位相同。
///
/// 具体会挂在这里的东西:
///
/// - `BinCuts::block_hist_offsets` 的局部偏移算错(块内下标当全局用,
///   或者反过来)—— 只在块数 > 1 且特征数不能整除时才暴露
/// - `reduce_best` 的并列取舍规则和块内不一致:那样并列的候选会因为
///   "被分到哪个块"而胜负不同,换个 cols_per_block 就训出另一棵树
/// - 跨块归约时把 Comm 那一步跳过去
#[test]
fn bit_exact_vs_xgboost_across_column_block_sizes() {
    let cases = load_index();
    // 5 个特征:1 是每列一块,3 除不尽(3 + 2),5 正好一块,64 是一块装下
    let block_sizes = [1usize, 2, 3, 4, 5, 7, 64];

    for case in cases.iter().filter(|c| c.n_rows == 200) {
        for &cols in &block_sizes {
            let got = margins_with_blocks(case, cols);
            let want = read_f32(&fixtures_dir().join(format!("{}.margin", case.name)));
            for (row, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{name} cols_per_block={cols} 第 {row} 行:我们 {g},xgboost {w}。\n\
                     列块的切法影响了结果 —— 分块只该改内存和并行粒度,不该改算法。",
                    name = case.name,
                );
            }
        }
    }
}

/// 块内特征数不能整除时,偏移表最容易出错。单独钉一条:
/// 5 个特征切成 3 + 2,两个块的直方图宽度不同,`feat_offsets` 是块内
/// 局部编号,拿全局偏移去索引就会越界或者串味。
#[test]
fn bit_exact_uneven_last_block() {
    let case = load_index()
        .into_iter()
        .find(|c| c.name == "a_s0_missing_logistic")
        .expect("fixture 少了 a_s0_missing_logistic");
    // 5 个特征、每块 3 个 → 第二块只有 2 个
    let uneven = margins_with_blocks(&case, 3);
    let even = margins_with_blocks(&case, 5);
    for (row, (a, b)) in uneven.iter().zip(&even).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "第 {row} 行:3+2 和 5 分块结果不同");
    }
}

/// 线程数不影响结果 —— 位相同,不是"差不多"。
///
/// 这是定点累加最实在的一次兑现:直方图按列块并行累加,块内是整数加法
/// (与顺序无关),块间归约按块序固定,所以 1 线程和 16 线程算出来的
/// 是同一棵树。浮点直方图做不到这一点,当年只能靠"行按升序遍历"来
/// 保证可复现,一并行就没了。
///
/// 也是阶段 4 多卡的预演:那时换成"每卡一组列 + 候选 allreduce",
/// 要的是同一条性质。
#[test]
fn bit_exact_vs_xgboost_across_thread_counts() {
    let cases = load_index();
    // cols_per_block = 1 时 5 个特征 5 个块,线程数能真的用上
    for case in cases.iter().filter(|c| c.n_rows == 200) {
        let want = read_f32(&fixtures_dir().join(format!("{}.margin", case.name)));
        for nthread in [1usize, 2, 4, 8] {
            let got = margins_with(case, 1, nthread);
            for (row, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{name} nthread={nthread} 第 {row} 行:我们 {g},xgboost {w}。\n\
                     线程数影响了结果 —— 并行归约的顺序泄进了树里。",
                    name = case.name,
                );
            }
        }
    }
}
