#![cfg(feature = "cuda")]
//! Categorical (and nested column sampling) split enumeration inside the
//! resident GPU split cache. `gpu_math = "exact"` must stay byte-identical to
//! the CPU, and the device path (not the host fallback) must actually run.

use ferrisboost::{
    columns::BinningStrategy,
    comm::Local,
    source::DenseSource,
    train::{prof, train_gpu_weighted, train_weighted, BlockSource, TrainConfig},
    tree::Model,
    types::{GpuMath, Objective, TrainParams},
};
use std::sync::atomic::Ordering;

// The prof counters are process-wide; tests that read them run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Data {
    x: Vec<f32>,
    y: Vec<f32>,
    w: Vec<f32>,
    n_features: usize,
    flags: Vec<bool>,
}

/// Two numeric columns, then one categorical column per entry of `levels`.
/// The label depends on a random per-category effect, so partition splits win;
/// 6% of the categorical codes and 4% of the numeric values are missing.
fn make(n: usize, levels: &[u32], classes: u32, seed: u64) -> Data {
    let mut rng = Rng(seed | 1);
    let nf = 2 + levels.len();
    let effects: Vec<Vec<f32>> = levels
        .iter()
        .map(|&l| (0..l).map(|_| rng.unit() * 2.0 - 1.0).collect())
        .collect();
    let mut x = Vec::with_capacity(n * nf);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for _ in 0..n {
        let a = rng.unit();
        let b = rng.unit();
        let mut z = a - 0.5 * b;
        x.push(if rng.below(25) == 0 { f32::NAN } else { a });
        x.push(if rng.below(25) == 0 { f32::NAN } else { b });
        for (k, &l) in levels.iter().enumerate() {
            let code = rng.below(l as u64) as usize;
            z += effects[k][code];
            x.push(if rng.below(16) == 0 { f32::NAN } else { code as f32 });
        }
        z += 0.2 * (rng.unit() - 0.5);
        y.push(if classes > 1 {
            ((z * 1.7 + 3.0).max(0.0) as u32 % classes) as f32
        } else {
            z
        });
        w.push(0.5 + rng.unit());
    }
    let mut flags = vec![false, false];
    flags.extend(levels.iter().map(|_| true));
    Data {
        x,
        y,
        w,
        n_features: nf,
        flags,
    }
}

fn source(d: &Data, cols: usize) -> DenseSource {
    DenseSource::new(
        &d.x,
        d.y.len(),
        d.n_features,
        BinningStrategy::SketchTyped {
            max_bins: 255,
            categorical: &d.flags,
        },
        cols,
    )
}

fn params() -> TrainParams {
    TrainParams {
        n_rounds: 4,
        max_depth: 6,
        max_bin: 255,
        nthread: 2,
        cols_per_block: 2,
        gpu_math: GpuMath::Exact,
        seed: 11,
        ..Default::default()
    }
}

fn cfg(p: TrainParams, objective: Objective) -> TrainConfig {
    let mut c = TrainConfig::new(p, objective);
    c.quiet = true;
    c
}

fn json(m: &Model) -> String {
    m.to_xgboost_json().unwrap()
}

struct Run {
    model: Model,
    /// (node, categorical feature) pairs enumerated on the device.
    cat_device: u64,
    /// Nodes evaluated by the device split cache.
    device_nodes: u64,
}

fn run_gpu(d: &Data, c: &TrainConfig, weighted: bool) -> Run {
    let src = source(d, c.params.cols_per_block);
    prof::reset();
    let model = train_gpu_weighted(
        0,
        &src,
        &d.y,
        weighted.then_some(d.w.as_slice()),
        &[],
        c,
        &mut [],
        &Local,
    )
    .unwrap();
    Run {
        model,
        cat_device: prof::GPU_SPLIT_CAT_EVALUATED.load(Ordering::Relaxed),
        device_nodes: prof::GPU_SPLIT_EVALUATED_NODES.load(Ordering::Relaxed),
    }
}

fn run_cpu(d: &Data, c: &TrainConfig, weighted: bool) -> Model {
    let src = source(d, c.params.cols_per_block);
    train_weighted(
        &src,
        &d.y,
        weighted.then_some(d.w.as_slice()),
        &[],
        c,
        &mut [],
        &Local,
    )
    .unwrap()
}

/// GPU exact (device split cache) == CPU byte for byte, and the categorical
/// device kernels ran.
fn assert_exact(d: &Data, c: &TrainConfig, weighted: bool, what: &str) -> Run {
    let cpu = run_cpu(d, c, weighted);
    // `max_cat_threshold = 1` scans no prefix, so such a model has no categorical split.
    if c.params.max_cat_threshold > 1 {
        assert!(cpu.has_categorical_splits(), "{what}: data must reward categorical splits");
    }
    let gpu = run_gpu(d, c, weighted);
    assert_eq!(json(&cpu), json(&gpu.model), "{what}: GPU != CPU");
    assert!(
        gpu.device_nodes > 0 && gpu.cat_device > 0,
        "{what}: device split cache not used (nodes {}, categorical {})",
        gpu.device_nodes,
        gpu.cat_device
    );
    gpu
}

/// Keep up to `n` physical blocks resident (clamped to the block count).
fn resident(p: &mut TrainParams, d: &Data, n: usize) {
    p.resident_blocks = Some(n.min(d.n_features.div_ceil(p.cols_per_block)));
}

#[test]
fn onehot_and_partition_regimes_match_cpu_on_the_device_path() {
    let _g = SERIAL.lock().unwrap();
    // 3 and 2 levels -> one-hot; 7 / 12 / 40 / 250 -> partition.
    for (levels, tag) in [
        (vec![3u32, 2], "one-hot"),
        (vec![7], "7 levels"),
        (vec![12], "12 levels"),
        (vec![40], "40 levels"),
        (vec![250], "250 levels"),
        (vec![3, 7, 12, 40, 250, 2], "mixed"),
    ] {
        let d = make(5000, &levels, 1, 3 + levels.len() as u64);
        let mut p = params();
        resident(&mut p, &d, 4);
        assert_exact(&d, &cfg(p, Objective::SquaredError), false, tag);
    }
}

#[test]
fn weights_missing_regularization_and_thresholds_match_cpu() {
    let _g = SERIAL.lock().unwrap();
    let d = make(4000, &[5, 12, 40], 1, 21);
    let variants: Vec<(&str, Box<dyn Fn(&mut TrainParams)>)> = vec![
        ("default", Box::new(|_| {})),
        ("max_cat_to_onehot=1", Box::new(|p| p.max_cat_to_onehot = 1)),
        ("max_cat_to_onehot=8", Box::new(|p| p.max_cat_to_onehot = 8)),
        ("max_cat_threshold=1", Box::new(|p| p.max_cat_threshold = 1)),
        ("max_cat_threshold=3", Box::new(|p| p.max_cat_threshold = 3)),
        ("max_cat_threshold=64", Box::new(|p| p.max_cat_threshold = 64)),
        (
            "lambda/alpha/mds",
            Box::new(|p| {
                p.lambda = 3.0;
                p.alpha = 0.5;
                p.max_delta_step = 0.4;
            }),
        ),
        (
            "mcw/gamma",
            Box::new(|p| {
                p.min_child_weight = 4.0;
                p.gamma = 0.3;
            }),
        ),
        (
            "monotone numerics",
            Box::new(|p| p.monotone_constraints = vec![1, -1, 0, 0, 0]),
        ),
        (
            "monotone + alpha",
            Box::new(|p| {
                p.monotone_constraints = vec![-1, 1, 0, 0, 0];
                p.alpha = 0.2;
            }),
        ),
        (
            "subsample/colsample",
            Box::new(|p| {
                p.subsample = 0.7;
                p.colsample_bytree = 0.8;
            }),
        ),
    ];
    for (tag, tweak) in &variants {
        for weighted in [false, true] {
            let mut p = params();
            resident(&mut p, &d, 2);
            tweak(&mut p);
            assert_exact(
                &d,
                &cfg(p, Objective::SquaredError),
                weighted,
                &format!("{tag} weighted={weighted}"),
            );
        }
    }
}

#[test]
fn multiclass_and_logistic_match_cpu() {
    let _g = SERIAL.lock().unwrap();
    let d = make(4000, &[6, 30], 4, 5);
    let mut p = params();
    p.n_rounds = 3;
    resident(&mut p, &d, 2);
    assert_exact(
        &d,
        &cfg(p, Objective::Softprob { num_class: 4 }),
        true,
        "softprob",
    );
    let mut d2 = make(4000, &[9, 3], 2, 8);
    for v in &mut d2.y {
        *v = v.clamp(0.0, 1.0);
    }
    let mut p = params();
    resident(&mut p, &d2, 3);
    assert_exact(&d2, &cfg(p, Objective::Logistic), false, "logistic");
}

#[test]
fn cache_capacity_batching_and_streaming_match_cpu() {
    let _g = SERIAL.lock().unwrap();
    let d = make(6000, &[12, 40, 7, 3], 1, 77);
    let base = cfg(params(), Objective::SquaredError);
    let cpu = run_cpu(&d, &base, false);
    assert!(cpu.has_categorical_splits());
    // Small caps overflow the cache (histogram fallback for the spilled nodes),
    // 1 disables it, 16 is the default; resident 0 is pure streaming (host
    // enumeration); 2 of 3 blocks is hybrid.
    for cap in [1usize, 2, 3, 16] {
        for resident_blocks in [0usize, 2, 3] {
            let mut p = params();
            p.hist_nodes_per_batch = Some(cap);
            resident(&mut p, &d, resident_blocks);
            let c = cfg(p, Objective::SquaredError);
            let run = run_gpu(&d, &c, false);
            assert_eq!(
                json(&cpu),
                json(&run.model),
                "cap {cap} resident {resident_blocks}"
            );
            if cap >= 2 && resident_blocks >= 2 {
                assert!(
                    run.cat_device > 0,
                    "cap {cap} resident {resident_blocks}: device path unused"
                );
            }
            if cap == 1 || resident_blocks == 0 {
                assert_eq!(
                    run.cat_device, 0,
                    "cap {cap} resident {resident_blocks}: expected the host fallback"
                );
            }
        }
    }
}

#[test]
fn explicit_cache_off_uses_the_host_fallback_and_matches() {
    let _g = SERIAL.lock().unwrap();
    let d = make(3000, &[12, 5], 1, 9);
    let mut p = params();
    p.gpu_split_cache = false;
    resident(&mut p, &d, 2);
    let c = cfg(p, Objective::SquaredError);
    let cpu = run_cpu(&d, &c, false);
    let run = run_gpu(&d, &c, false);
    assert_eq!(json(&cpu), json(&run.model));
    assert_eq!(run.cat_device, 0);
    assert_eq!(run.device_nodes, 0);
}

#[test]
fn fast_math_categorical_device_path_is_deterministic() {
    let _g = SERIAL.lock().unwrap();
    let d = make(4000, &[12, 40], 1, 31);
    let mut p = params();
    p.gpu_math = GpuMath::Fast;
    resident(&mut p, &d, 2);
    let c = cfg(p, Objective::SquaredError);
    let a = run_gpu(&d, &c, false);
    let b = run_gpu(&d, &c, false);
    assert!(a.model.has_categorical_splits() && a.cat_device > 0);
    assert_eq!(json(&a.model), json(&b.model));
}

#[test]
fn nested_column_sampling_uses_the_device_cache_and_matches_cpu() {
    let _g = SERIAL.lock().unwrap();
    // Categorical and numeric-only data, level-only, node-only and both.
    for (levels, tag) in [(vec![12u32, 7], "categorical"), (vec![], "numeric")] {
        let d = make(4000, &levels, 1, 41);
        for (bylevel, bynode) in [(0.6f32, 1.0f32), (1.0, 0.5), (0.7, 0.6)] {
            for tree in [1.0f32, 0.8] {
                let mut p = params();
                p.n_rounds = 5;
                p.colsample_bylevel = bylevel;
                p.colsample_bynode = bynode;
                p.colsample_bytree = tree;
                resident(&mut p, &d, 2);
                let c = cfg(p, Objective::SquaredError);
                let cpu = run_cpu(&d, &c, false);
                let gpu = run_gpu(&d, &c, false);
                assert_eq!(
                    json(&cpu),
                    json(&gpu.model),
                    "{tag} level {bylevel} node {bynode} tree {tree}"
                );
                assert!(
                    gpu.device_nodes > 0,
                    "{tag} level {bylevel} node {bynode}: nested sampling fell back to the host"
                );
                if !levels.is_empty() {
                    assert!(gpu.cat_device > 0);
                }
            }
        }
    }
}

#[test]
fn source_blocks_cover_categorical_and_numeric_columns() {
    // The fixtures above rely on several physical blocks (cols_per_block = 2).
    let d = make(300, &[12, 40, 7, 3], 1, 1);
    let src = source(&d, 2);
    assert_eq!(src.n_blocks(), 3);
    assert!(src.cuts().has_categorical());
}

/// Categories with identical statistics tie exactly (stable weight order, first
/// candidate wins, forward before backward, lowest feature first).
fn tied_data(n: usize, levels: &[u32]) -> Data {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let nf = 1 + levels.len();
    for i in 0..n {
        x.push((i % 7) as f32);
        let mut z = 0.0f32;
        for (k, &l) in levels.iter().enumerate() {
            // Cycle through categories so each one holds the same rows.
            let code = ((i / (k + 1)) as u32 % l) as f32;
            z += if (code as u32) % 3 == 0 { 1.0 } else { 0.0 };
            x.push(if i % 29 == 0 { f32::NAN } else { code });
        }
        y.push(z);
    }
    let mut flags = vec![false];
    flags.extend(levels.iter().map(|_| true));
    Data {
        w: vec![1.0; n],
        x,
        y,
        n_features: nf,
        flags,
    }
}

#[test]
fn exactly_tied_categories_resolve_like_the_cpu() {
    let _g = SERIAL.lock().unwrap();
    for levels in [vec![3u32], vec![12, 12], vec![6, 9, 3], vec![40, 250]] {
        let d = tied_data(4200, &levels);
        for onehot in [4u32, 1, 300] {
            let mut p = params();
            p.cols_per_block = 1;
            p.max_cat_to_onehot = onehot;
            resident(&mut p, &d, 8);
            assert_exact(
                &d,
                &cfg(p, Objective::SquaredError),
                false,
                &format!("ties {levels:?} onehot<{onehot}"),
            );
        }
    }
}
