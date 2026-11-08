//! `max_bin` above 255: features with more bins are stored as 16-bit columns,
//! only in their own blocks; every other block stays 8-bit.
//!
//! Pinned here:
//! - mixed-width grouping (wide features never share a block with 8-bit ones);
//! - models are independent of `cols_per_block` and `nthread` with wide blocks
//!   interleaved between 8-bit blocks (cross-block tie rule);
//! - Dense ≡ Parquet in memory ≡ Parquet spilled ≡ reused 16-bit cache;
//! - all-8-bit data keeps the version-2 cache format;
//! - wide training data is learned at full resolution (thresholds beyond the
//!   8-bit range), and the GPU refuses 16-bit columns instead of misreading them.
use ferrisboost::bins::BinWidth;
use ferrisboost::columns::BinningStrategy;
use ferrisboost::comm::Local;
use ferrisboost::source::parquet_source::{ParquetBlockSource, QuantizedCacheOptions};
use ferrisboost::source::DenseSource;
use ferrisboost::train::{train, BlockSource, BlockSourceExt, TrainConfig};
use ferrisboost::types::{Objective, TrainParams};

struct TmpDir(std::path::PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "fb_wide_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const N_ROWS: usize = 3000;
/// Distinct values per feature: wide (> 255) features interleaved with 8-bit
/// ones, so width-aware grouping puts a 16-bit block between 8-bit blocks.
const DISTINCT: [usize; 7] = [8, 600, 40, 300, 1000, 5, 257];

/// Row-major lossless data (every feature has at most 1000 distinct values,
/// below `max_bin = 1024`, so every sketch gives the same cuts) with a few NaNs.
fn data() -> (Vec<f32>, Vec<f32>) {
    let n_feats = DISTINCT.len();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut x = vec![0f32; N_ROWS * n_feats];
    for r in 0..N_ROWS {
        for (f, &d) in DISTINCT.iter().enumerate() {
            x[r * n_feats + f] = if next() % 97 == 0 {
                f32::NAN
            } else {
                (next() % d) as f32 * 0.25 - 3.0
            };
        }
    }
    let y = (0..N_ROWS)
        .map(|r| {
            let row = &x[r * n_feats..(r + 1) * n_feats];
            let v = |f: usize| if row[f].is_nan() { 0.0 } else { row[f] };
            // Thresholds deep inside the wide features' ranges.
            let s = (v(1) > 70.0) as u8 as f32 + (v(4) > 200.0) as u8 as f32 * 0.5
                - (v(3) > 40.0) as u8 as f32
                + 0.1 * v(0);
            (s > 0.6) as u8 as f32
        })
        .collect();
    (x, y)
}

fn params(nthread: usize, cols_per_block: usize) -> TrainParams {
    TrainParams {
        n_rounds: 6,
        max_depth: 5,
        max_bin: 1024,
        learning_rate: 0.3,
        nthread,
        cols_per_block,
        ..TrainParams::default()
    }
}

fn model_json(src: &dyn BlockSource, y: &[f32], p: TrainParams) -> String {
    let cfg = TrainConfig::new(p, Objective::Logistic);
    let model = train(src, y, &[], &cfg, &mut [], &Local).unwrap();
    model.to_xgboost_json().unwrap()
}

fn block_widths(src: &dyn BlockSource) -> Vec<(Vec<u32>, BinWidth)> {
    (0..src.n_blocks())
        .map(|b| {
            src.with_block_ret(b, |blk| (blk.feat_ids.clone(), blk.width()))
                .unwrap()
        })
        .collect()
}

fn dense(x: &[f32], cpb: usize, nthread: usize) -> DenseSource {
    DenseSource::new_with_nthread(
        x,
        N_ROWS,
        DISTINCT.len(),
        BinningStrategy::Sketch { max_bins: 1024 },
        cpb,
        nthread,
    )
}

#[test]
fn wide_features_get_16_bit_blocks_and_others_stay_8_bit() {
    let (x, _) = data();
    let src = dense(&x, 2, 1);
    for (f, &d) in DISTINCT.iter().enumerate() {
        // Lossless: one bin per value that occurs (a few of the 600/1000
        // values may not be drawn).
        let n = src.cuts().n_bins(f as u32);
        assert!(
            n <= d && n * 10 >= d * 9,
            "feature {f}: {n} bins for {d} values"
        );
        assert_eq!(n > 255, d > 255, "feature {f}");
    }
    let blocks = block_widths(&src);
    assert_eq!(
        blocks,
        vec![
            (vec![0, 2], BinWidth::U8),
            (vec![1, 3], BinWidth::U16),
            (vec![4, 6], BinWidth::U16),
            (vec![5], BinWidth::U8),
        ]
    );
    // max_bin <= 255 keeps every block 8-bit and the contiguous grouping.
    let narrow = DenseSource::new(
        &x,
        N_ROWS,
        DISTINCT.len(),
        BinningStrategy::Sketch { max_bins: 255 },
        2,
    );
    assert!(block_widths(&narrow)
        .iter()
        .all(|(_, w)| *w == BinWidth::U8));
    assert_eq!(narrow.block_features(0).unwrap(), vec![0, 1]);
}

#[test]
fn wide_models_do_not_depend_on_blocking_or_threads() {
    let (x, y) = data();
    let reference = model_json(&dense(&x, 1, 1), &y, params(1, 1));
    for (cpb, nthread) in [(2, 1), (3, 4), (8, 4), (2, 3)] {
        let got = model_json(&dense(&x, cpb, nthread), &y, params(nthread, cpb));
        assert_eq!(got, reference, "cols_per_block={cpb} nthread={nthread}");
    }
    // The wide features are split beyond the 8-bit range: at least one
    // threshold on a > 255-bin feature lies above the 255th distinct value.
    let v: serde_json::Value = serde_json::from_str(&reference).unwrap();
    let trees = v["learner"]["gradient_booster"]["model"]["trees"]
        .as_array()
        .unwrap();
    let mut high = false;
    for t in trees {
        let ids = t["split_indices"].as_array().unwrap();
        let conds = t["split_conditions"].as_array().unwrap();
        let left = t["left_children"].as_array().unwrap();
        for ((id, c), l) in ids.iter().zip(conds).zip(left) {
            if l.as_i64() != Some(-1) && [1, 3, 4].contains(&id.as_u64().unwrap()) {
                high |= c.as_f64().unwrap() > 255.0 * 0.25 - 3.0;
            }
        }
    }
    assert!(high, "no split used a bin above the 8-bit range");
}

fn write_parquet(path: &std::path::Path, x: &[f32], y: &[f32]) {
    use arrow::array::{ArrayRef, Float32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    let n_feats = DISTINCT.len();
    let mut fields: Vec<Field> = (0..n_feats)
        .map(|f| Field::new(format!("f{f}"), DataType::Float32, true))
        .collect();
    fields.push(Field::new("label", DataType::Float32, false));
    let schema = Arc::new(Schema::new(fields));
    let mut cols: Vec<ArrayRef> = (0..n_feats)
        .map(|f| {
            Arc::new(Float32Array::from(
                (0..N_ROWS).map(|r| x[r * n_feats + f]).collect::<Vec<_>>(),
            )) as ArrayRef
        })
        .collect();
    cols.push(Arc::new(Float32Array::from(y.to_vec())));
    let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn manifest(dir: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap()
}

#[test]
fn parquet_memory_spill_and_reused_wide_cache_match_dense() {
    let (x, y) = data();
    let tmp = TmpDir::new("parquet");
    let path = tmp.0.join("t.parquet");
    write_parquet(&path, &x, &y);
    let cpb = 2;
    let reference = model_json(&dense(&x, cpb, 1), &y, params(1, cpb));

    let (mem, labels) = ParquetBlockSource::open(&path, "label", 1024, cpb).unwrap();
    assert_eq!(labels, y);
    assert_eq!(block_widths(&mem), block_widths(&dense(&x, cpb, 1)));
    assert_eq!(model_json(&mem, &y, params(1, cpb)), reference, "in memory");

    let cache = tmp.0.join("cache");
    let opts = || QuantizedCacheOptions {
        budget_bytes: Some(0),
        path: Some(cache.clone()),
    };
    let (spill, _) =
        ParquetBlockSource::open_with_cache(&path, "label", 1024, cpb, opts()).unwrap();
    assert_eq!(model_json(&spill, &y, params(2, cpb)), reference, "spilled");
    // Partial column reads of a 16-bit block agree with the whole block.
    let (feats, whole) = spill
        .with_block_ret(1, |b| (b.feat_ids.clone(), b.data.clone()))
        .unwrap();
    assert_eq!(whole.width(), BinWidth::U16);
    let mut part = None;
    spill
        .with_block_cols(1, &[1], &mut |b| {
            part = Some((b.feat_ids.clone(), b.data.clone()))
        })
        .unwrap();
    let (part_feats, part_data) = part.unwrap();
    assert_eq!(part_feats, vec![feats[1]]);
    assert_eq!(
        part_data.as_u16().unwrap(),
        &whole.as_u16().unwrap()[N_ROWS..2 * N_ROWS]
    );
    assert_eq!(
        spill.feature_column(feats[1]).unwrap().as_u16().unwrap(),
        &whole.as_u16().unwrap()[N_ROWS..2 * N_ROWS]
    );

    let m = manifest(&cache);
    assert_eq!(m["version"], 3);
    let widths: Vec<u64> = m["block_widths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(widths, vec![1, 2, 2, 1]);
    // Block files hold `feats × rows × width` bytes.
    let len = std::fs::metadata(cache.join("block-1.bin")).unwrap().len();
    assert_eq!(len as usize, 2 * N_ROWS * 2);
    drop(spill);

    let (reused, _) =
        ParquetBlockSource::open_with_cache(&path, "label", 1024, cpb, opts()).unwrap();
    assert_eq!(model_json(&reused, &y, params(1, cpb)), reference, "reused");
    drop(reused);

    // A manifest whose widths disagree with its cuts is rejected.
    let mut bad = m.clone();
    bad["block_widths"] = serde_json::json!([1, 1, 2, 1]);
    std::fs::write(
        cache.join("manifest.json"),
        serde_json::to_vec(&bad).unwrap(),
    )
    .unwrap();
    let err = ParquetBlockSource::open_with_cache(&path, "label", 1024, cpb, opts())
        .err()
        .expect("width mismatch must fail");
    assert!(format!("{err:#}").contains("width"), "{err:#}");
    // A version-2 manifest cannot claim 8-bit blocks for wide cuts either.
    let mut v2 = m;
    v2["version"] = serde_json::json!(2);
    v2.as_object_mut().unwrap().remove("block_widths");
    std::fs::write(
        cache.join("manifest.json"),
        serde_json::to_vec(&v2).unwrap(),
    )
    .unwrap();
    assert!(ParquetBlockSource::open_with_cache(&path, "label", 1024, cpb, opts()).is_err());
}

#[test]
fn all_8_bit_caches_keep_version_2() {
    let (x, y) = data();
    let tmp = TmpDir::new("v2");
    let path = tmp.0.join("t.parquet");
    write_parquet(&path, &x, &y);
    let cache = tmp.0.join("cache");
    let opts = || QuantizedCacheOptions {
        budget_bytes: Some(0),
        path: Some(cache.clone()),
    };
    let (src, _) = ParquetBlockSource::open_with_cache(&path, "label", 255, 2, opts()).unwrap();
    assert!(block_widths(&src).iter().all(|(_, w)| *w == BinWidth::U8));
    drop(src);
    let m = manifest(&cache);
    assert_eq!(m["version"], 2);
    assert!(m.get("block_widths").is_none());
    let len = std::fs::metadata(cache.join("block-0.bin")).unwrap().len();
    assert_eq!(len as usize, 2 * N_ROWS);
    ParquetBlockSource::open_with_cache(&path, "label", 255, 2, opts()).unwrap();
}

#[test]
fn max_bin_above_the_16_bit_tier_is_rejected() {
    let (x, y) = data();
    let src = dense(&x, 2, 1);
    let cfg = TrainConfig::new(
        TrainParams {
            max_bin: 65_536,
            ..params(1, 2)
        },
        Objective::Logistic,
    );
    let err = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap_err();
    assert!(format!("{err:#}").contains("max_bin"), "{err:#}");
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_refuses_16_bit_columns() {
    let (x, y) = data();
    let src = dense(&x, 2, 1);
    let cfg = TrainConfig::new(params(1, 2), Objective::Logistic);
    let err = ferrisboost::train::train_gpu(0, &src, &y, &[], &cfg, &mut [], &Local)
        .err()
        .expect("GPU training with 16-bit columns must fail");
    assert!(format!("{err:#}").contains("255 bins"), "{err:#}");
}
