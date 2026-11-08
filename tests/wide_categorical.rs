//! High-cardinality categorical features: a categorical feature whose largest
//! code is above 253 is stored as a 16-bit column in its own block; 8-bit
//! categorical and numeric features keep their 8-bit blocks.
//!
//! Pinned here:
//! - width-aware grouping for categorical codes (254 codes stay 8-bit);
//! - models are independent of `cols_per_block` and `nthread`;
//! - splits use category sets beyond the 8-bit range;
//! - Dense ≡ Parquet dictionary column in memory ≡ spilled 16-bit cache (v3)
//!   ≡ reused cache;
//! - codes above 65533 are rejected, and the GPU refuses 16-bit categorical
//!   columns up front.
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
            "fb_widecat_{tag}_{}_{}",
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

const N_ROWS: usize = 6000;
/// (levels, categorical) per feature: wide categorical features between 8-bit
/// numeric and categorical ones.
const FEATS: [(usize, bool); 6] = [
    (8, false),
    (1000, true),
    (12, true),
    (40, false),
    (300, true),
    (254, true),
];
const FLAGS: [bool; 6] = [false, true, true, false, true, true];

fn rng(seed: u64) -> impl FnMut() -> usize {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    }
}

/// Row-major data; the largest code of each categorical feature occurs.
fn data() -> (Vec<f32>, Vec<f32>) {
    let n_feats = FEATS.len();
    let mut next = rng(0x5eed);
    let mut x = vec![0f32; N_ROWS * n_feats];
    for r in 0..N_ROWS {
        for (f, &(d, _)) in FEATS.iter().enumerate() {
            x[r * n_feats + f] = if r < d {
                r as f32
            } else if next() % 89 == 0 {
                f32::NAN
            } else {
                (next() % d) as f32
            };
        }
    }
    // Category effects: pseudo-random per code, so a good split needs a set
    // that mixes codes from the whole range.
    let mut eff = rng(0xe77);
    let effects: Vec<Vec<f32>> = FEATS
        .iter()
        .map(|&(d, _)| {
            (0..d)
                .map(|_| (eff() % 1000) as f32 / 500.0 - 1.0)
                .collect()
        })
        .collect();
    let mut noise = rng(0x401e);
    let y = (0..N_ROWS)
        .map(|r| {
            let row = &x[r * n_feats..(r + 1) * n_feats];
            let mut s = 0.1 * if row[0].is_nan() { 0.0 } else { row[0] };
            for f in [1usize, 2, 4] {
                if !row[f].is_nan() {
                    s += effects[f][row[f] as usize];
                }
            }
            s + (noise() % 1000) as f32 / 2000.0
        })
        .collect();
    (x, y)
}

fn params(nthread: usize, cols_per_block: usize) -> TrainParams {
    TrainParams {
        n_rounds: 5,
        max_depth: 5,
        learning_rate: 0.3,
        nthread,
        cols_per_block,
        ..TrainParams::default()
    }
}

fn model_json(src: &dyn BlockSource, y: &[f32], p: TrainParams) -> String {
    let cfg = TrainConfig::new(p, Objective::SquaredError);
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
        FEATS.len(),
        BinningStrategy::SketchTyped {
            max_bins: 255,
            categorical: &FLAGS,
        },
        cpb,
        nthread,
    )
}

#[test]
fn wide_categorical_features_get_16_bit_blocks() {
    let (x, _) = data();
    let src = dense(&x, 2, 1);
    for (f, &(d, cat)) in FEATS.iter().enumerate() {
        assert_eq!(src.cuts().is_categorical(f as u32), cat);
        if cat {
            assert_eq!(src.cuts().n_bins(f as u32), d, "feature {f}");
        }
    }
    assert_eq!(
        block_widths(&src),
        vec![
            (vec![0, 2], BinWidth::U8),
            (vec![1, 4], BinWidth::U16),
            (vec![3, 5], BinWidth::U8),
        ]
    );
}

#[test]
fn wide_categorical_models_do_not_depend_on_blocking_or_threads() {
    let (x, y) = data();
    let reference = model_json(&dense(&x, 1, 1), &y, params(1, 1));
    for (cpb, nthread) in [(2, 1), (3, 4), (8, 4)] {
        let got = model_json(&dense(&x, cpb, nthread), &y, params(nthread, cpb));
        assert_eq!(got, reference, "cols_per_block={cpb} nthread={nthread}");
    }
    // Some split on the 1000-category feature sends a code above 255 right.
    let v: serde_json::Value = serde_json::from_str(&reference).unwrap();
    let mut high = false;
    for t in v["learner"]["gradient_booster"]["model"]["trees"]
        .as_array()
        .unwrap()
    {
        let nodes = t["categories_nodes"].as_array().unwrap();
        let segs = t["categories_segments"].as_array().unwrap();
        let sizes = t["categories_sizes"].as_array().unwrap();
        let cats = t["categories"].as_array().unwrap();
        let ids = t["split_indices"].as_array().unwrap();
        for ((n, s), z) in nodes.iter().zip(segs).zip(sizes) {
            let (n, s, z) = (
                n.as_u64().unwrap() as usize,
                s.as_u64().unwrap() as usize,
                z.as_u64().unwrap() as usize,
            );
            if ids[n].as_u64() == Some(1) {
                high |= cats[s..s + z].iter().any(|c| c.as_u64().unwrap() > 255);
            }
        }
    }
    assert!(
        high,
        "no categorical split used a code above the 8-bit range"
    );
}

fn write_parquet(path: &std::path::Path, x: &[f32], y: &[f32]) {
    use arrow::array::{ArrayRef, DictionaryArray, Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    let n_feats = FEATS.len();
    let mut fields = Vec::new();
    let mut cols: Vec<ArrayRef> = Vec::new();
    for (f, &(d, cat)) in FEATS.iter().enumerate() {
        let col: Vec<f32> = (0..N_ROWS).map(|r| x[r * n_feats + f]).collect();
        if cat {
            // An integer dictionary column (categorical without declaration):
            // dictionary values are the codes, in shuffled order.
            let values: Vec<i32> = (0..d as i32).rev().collect();
            let keys: Vec<Option<i32>> = col
                .iter()
                .map(|&v| (!v.is_nan()).then(|| d as i32 - 1 - v as i32))
                .collect();
            let arr = DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(keys),
                Arc::new(Int32Array::from(values)),
            )
            .unwrap();
            fields.push(Field::new(
                format!("f{f}"),
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int32)),
                true,
            ));
            cols.push(Arc::new(arr));
        } else {
            fields.push(Field::new(format!("f{f}"), DataType::Float32, true));
            cols.push(Arc::new(Float32Array::from(col)));
        }
    }
    fields.push(Field::new("label", DataType::Float32, false));
    cols.push(Arc::new(Float32Array::from(y.to_vec())));
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

#[test]
fn parquet_dictionary_columns_and_the_16_bit_cache_match_dense() {
    let (x, y) = data();
    let tmp = TmpDir::new("parquet");
    let path = tmp.0.join("t.parquet");
    write_parquet(&path, &x, &y);
    let cpb = 2;
    let reference = model_json(&dense(&x, cpb, 1), &y, params(1, cpb));

    let (mem, labels) = ParquetBlockSource::open(&path, "label", 255, cpb).unwrap();
    assert_eq!(labels, y);
    assert_eq!(block_widths(&mem), block_widths(&dense(&x, cpb, 1)));
    assert_eq!(model_json(&mem, &y, params(1, cpb)), reference, "in memory");

    let cache = tmp.0.join("cache");
    let opts = || QuantizedCacheOptions {
        budget_bytes: Some(0),
        path: Some(cache.clone()),
    };
    let (spill, _) = ParquetBlockSource::open_with_cache(&path, "label", 255, cpb, opts()).unwrap();
    assert_eq!(model_json(&spill, &y, params(2, cpb)), reference, "spilled");
    drop(spill);
    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(cache.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["version"], 3);
    assert_eq!(m["block_widths"], serde_json::json!([1, 2, 1]));
    let (reused, _) =
        ParquetBlockSource::open_with_cache(&path, "label", 255, cpb, opts()).unwrap();
    assert_eq!(model_json(&reused, &y, params(1, cpb)), reference, "reused");
}

#[test]
fn codes_above_the_16_bit_limit_are_rejected() {
    let max = ferrisboost::categorical::MAX_CAT_CODE_WIDE;
    assert_eq!(max, 65_533);
    let x = vec![0.0, 1.0, max as f32, 2.0];
    let y = vec![0.0, 1.0, 0.5, 0.2];
    let ok = DenseSource::new(
        &x,
        4,
        1,
        BinningStrategy::SketchTyped {
            max_bins: 255,
            categorical: &[true],
        },
        1,
    );
    assert_eq!(ok.cuts().n_bins(0), max as usize + 1);
    assert_eq!(ok.cuts().bin_width(0), BinWidth::U16);
    let model = train(
        &ok,
        &y,
        &[],
        &TrainConfig::new(params(1, 1), Objective::SquaredError),
        &mut [],
        &Local,
    )
    .unwrap();
    assert!(model.to_xgboost_json().is_ok());
    let bad = vec![0.0, 1.0, max as f32 + 1.0, 2.0];
    let err = std::panic::catch_unwind(|| {
        DenseSource::new(
            &bad,
            4,
            1,
            BinningStrategy::SketchTyped {
                max_bins: 255,
                categorical: &[true],
            },
            1,
        )
    })
    .err()
    .expect("code 65534 must be rejected");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("65533"), "{msg}");
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_refuses_16_bit_categorical_columns() {
    let (x, y) = data();
    let src = dense(&x, 2, 1);
    let cfg = TrainConfig::new(params(1, 2), Objective::SquaredError);
    let err = ferrisboost::train::train_gpu(0, &src, &y, &[], &cfg, &mut [], &Local)
        .err()
        .expect("GPU training with 16-bit categorical columns must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("16-bit") && msg.contains("categor"), "{msg}");
}
