//! `colsample_bytree` 在 **source 这一层**的部分列读取。
//!
//! 这条线的意义是:只在 H2D 之前挑列是不够的 —— 那样传输量降了、
//! **读取量没降**。`read 32 列 → pack 3 列 → H2D 3 列` 不能宣称
//! source 流量降到 3/32。所以这里直接钉两件事:
//!
//! 1. 部分读取的字节**和整块读取的对应列逐字节相同**;
//! 2. `source_bytes_read()` 确实只涨了选中列的字节数。
use ferrisboost::source::parquet_source::{ParquetBlockSource, QuantizedCacheOptions};
use ferrisboost::train::{BlockSource, BlockSourceExt};

/// 仓库里没有 `tempfile` 依赖,不为一个测试引一个。
struct TmpDir(std::path::PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fb_colsample_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 造一个小 parquet,并**强制落盘**(`budget_bytes = Some(0)`),
/// 这样走的是 spill cache —— 那正是 streaming 的形状。
fn spilled_source(dir: &std::path::Path) -> ParquetBlockSource {
    let n_rows = 500usize;
    let n_feats = 12usize;
    let path = dir.join("t.parquet");
    {
        use arrow::array::Float32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;
        let mut fields: Vec<Field> = (0..n_feats)
            .map(|f| Field::new(format!("f{f}"), DataType::Float32, false))
            .collect();
        fields.push(Field::new("label", DataType::Float32, false));
        let schema = Arc::new(Schema::new(fields));
        let mut cols: Vec<arrow::array::ArrayRef> = (0..n_feats)
            .map(|f| {
                Arc::new(Float32Array::from(
                    (0..n_rows).map(|r| ((r * 7 + f * 13) % 29) as f32).collect::<Vec<_>>(),
                )) as arrow::array::ArrayRef
            })
            .collect();
        cols.push(Arc::new(Float32Array::from(
            (0..n_rows).map(|r| (r % 2) as f32).collect::<Vec<_>>(),
        )));
        let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
        let file = std::fs::File::create(&path).unwrap();
        let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    let opts = QuantizedCacheOptions {
        budget_bytes: Some(0), // 强制 spill 到磁盘
        path: Some(dir.join("cache")),
    };
    ParquetBlockSource::open_with_cache(&path, "label", 32, 4, opts).unwrap().0
}

#[test]
fn partial_column_read_matches_full_block_and_reads_fewer_bytes() {
    let dir = TmpDir::new("partial");
    let src = spilled_source(dir.path());
    assert!(src.n_blocks() >= 2, "需要多个列块才有意义");

    // 先整块读一次(顺便让 checksum 通过验证 —— 部分读取要求块已验过)。
    let full: Vec<u8> = src.with_block_ret(0, |b| b.data.clone()).unwrap();
    let feats: Vec<_> = src.block_features(0).unwrap();
    let n_rows = src.n_rows();

    let before = src.source_bytes_read().expect("spill source 必须统计读入字节");
    // 只读第 1 列(局部下标 1)。
    let cols = vec![1usize];
    let (got, got_feats) = src
        .with_block_ret(0, |_| ())
        .map(|_| ())
        .and_then(|_| {
            let mut out = (Vec::new(), Vec::new());
            src.with_block_cols(0, &cols, &mut |b| {
                out = (b.data.clone(), b.feat_ids.clone());
            })?;
            Ok(out)
        })
        .unwrap();
    let after = src.source_bytes_read().unwrap();

    // 1. 内容必须和整块里对应的那一段逐字节相同。
    assert_eq!(got, &full[n_rows..2 * n_rows], "部分读取的列内容必须一致");
    assert_eq!(got_feats, vec![feats[1]], "feat_ids 必须是选中的那一列");

    // 2. 真的只读了选中列的字节 —— 这一趟包含一次整块读(上面的 with_block_ret)
    //    加一次单列读,所以增量应当是 整块 + 单列,而不是两次整块。
    let full_len = (feats.len() * n_rows) as u64;
    let delta = after - before;
    assert_eq!(
        delta,
        full_len + n_rows as u64,
        "增量应是一次整块 + 一次单列;实际 {delta},整块 {full_len},单列 {n_rows}"
    );
}

#[test]
fn selecting_every_column_is_equivalent_to_a_full_read() {
    let dir = TmpDir::new("all");
    let src = spilled_source(dir.path());
    let full: Vec<u8> = src.with_block_ret(0, |b| b.data.clone()).unwrap();
    let all: Vec<usize> = (0..src.block_features(0).unwrap().len()).collect();
    let mut got = Vec::new();
    src.with_block_cols(0, &all, &mut |b| got = b.data.clone()).unwrap();
    assert_eq!(got, full);
}

/// 端到端:`colsample_bytree` 必须让 **spill source 真的少读字节**。
///
/// 这是整条线的落点 —— 前面所有步骤(确定性采样、块级跳过、压实、
/// 部分列读取)加起来,只有在这里体现成「读入字节下降」才算兑现。
/// 如果只在 H2D 之前挑列,这个断言会失败。
#[test]
fn colsample_reduces_source_bytes_on_a_spilled_cache() {
    use ferrisboost::train::{train, TrainConfig};
    use ferrisboost::comm::Local;

    let measure = |rate: f32| -> (u64, usize) {
        let dir = TmpDir::new("bytes");
        let src = spilled_source(dir.path());
        let n_rows = src.n_rows();
        let labels: Vec<f32> = (0..n_rows).map(|r| (r % 2) as f32).collect();
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 3,
            nthread: 1,
            colsample_bytree: rate,
            seed: 5,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        train(&src, &labels, &[], &cfg, &mut [], &Local).unwrap();
        (src.source_bytes_read().unwrap(), src.n_blocks())
    };

    let (full, nblocks) = measure(1.0);
    let (sampled, _) = measure(0.25);
    assert!(nblocks >= 2, "需要多块才有意义");
    assert!(full > 0, "整块训练必须读到字节");
    assert!(
        sampled < full,
        "colsample 必须真的减少 source 读入字节:full {full} vs sampled {sampled}"
    );
    // 打印出来,方便人工核对稀释程度(块粒度决定它达不到理想比例)。
    println!("source bytes: colsample=1.0 -> {full}, colsample=0.25 -> {sampled} ({:.0}%)",
        100.0 * sampled as f64 / full as f64);
}


/// 多轮摊薄曲线:checksum 首触的整块读是**一次性**的,轮数越多摊得越薄。
///
/// 57%(3 轮)不是稳态 —— 这个用例把它和 1 / 5 / 20 轮放在一起,
/// 让「一次性开销」和「稳态比例」分开看,而不是拿一个轮数的数字当结论。
#[test]
fn source_byte_ratio_converges_as_rounds_amortize_first_touch() {
    use ferrisboost::comm::Local;
    use ferrisboost::train::{train, TrainConfig};

    let measure = |rate: f32, rounds: usize| -> u64 {
        let dir = TmpDir::new("amort");
        let src = spilled_source(dir.path());
        let n_rows = src.n_rows();
        let labels: Vec<f32> = (0..n_rows).map(|r| (r % 2) as f32).collect();
        let params = ferrisboost::types::TrainParams {
            n_rounds: rounds,
            max_depth: 3,
            nthread: 1,
            colsample_bytree: rate,
            seed: 5,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        train(&src, &labels, &[], &cfg, &mut [], &Local).unwrap();
        src.source_bytes_read().unwrap()
    };

    println!("rounds  colsample=1.0   colsample=0.25   ratio");
    let mut ratios = Vec::new();
    for &r in &[1usize, 5, 20] {
        let full = measure(1.0, r);
        let samp = measure(0.25, r);
        let ratio = 100.0 * samp as f64 / full as f64;
        ratios.push(ratio);
        println!("{r:>6}  {full:>13}   {samp:>14}   {ratio:>5.1}%");
    }
    // 轮数变多,比例必须单调下降(首触的整块读被摊薄)。
    assert!(
        ratios[2] <= ratios[1] && ratios[1] <= ratios[0],
        "比例应随轮数下降:{ratios:?}"
    );
}
