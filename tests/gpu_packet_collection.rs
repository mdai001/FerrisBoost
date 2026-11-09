#![cfg(feature = "cuda")]

use ferrisboost::{
    comm::Local,
    source::DenseSource,
    train::{train, train_gpu, BlockSource, TrainConfig},
    types::{GpuMath, Objective, TrainParams, MISSING_BIN},
};

const N_ROWS: usize = 4096;
const N_FEATURES: usize = 50;
const COLS_PER_BLOCK: usize = 16;

static GPU_PACKET_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fixture() -> (DenseSource, Vec<f32>) {
    let data: Vec<f32> = (0..N_ROWS)
        .flat_map(|row| {
            (0..N_FEATURES).map(move |feature| {
                // The first eight features encode independent bits. The label
                // below is their mean, so both children at successive levels
                // continue to contain a useful split through depth eight.
                // Remaining features add distractors and missing values.
                if feature < 8 {
                    if (row * 37 + feature * 19) % 257 == 0 {
                        f32::NAN
                    } else {
                        ((row >> feature) & 1) as f32
                    }
                } else if (row * 23 + feature * 41) % 89 == 0 {
                    f32::NAN
                } else {
                    let mut value = (row as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add((feature as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
                    value ^= value >> 30;
                    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    value ^= value >> 27;
                    ((value >> 32) & 1) as f32
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..N_ROWS)
        .map(|row| {
            (0..8)
                .map(|feature| ((row >> feature) & 1) as f32)
                .sum::<f32>()
                / 8.0
        })
        .collect();
    let source = DenseSource::from_row_major(&data, N_ROWS, N_FEATURES, 255, COLS_PER_BLOCK);
    (source, labels)
}

fn bytes(model: &ferrisboost::tree::Model) -> Vec<u8> {
    serde_json::to_vec(model).unwrap()
}

fn max_nonleaf_width(model: &ferrisboost::tree::Model) -> usize {
    model
        .trees
        .iter()
        .map(|tree| {
            let mut level = vec![0usize];
            let mut maximum = 0;
            while !level.is_empty() {
                let mut next = Vec::new();
                let mut width = 0;
                for index in level {
                    let node = &tree.nodes[index];
                    if !node.is_leaf {
                        width += 1;
                        next.push(node.left as usize);
                        next.push(node.right as usize);
                    }
                }
                maximum = maximum.max(width);
                level = next;
            }
            maximum
        })
        .max()
        .unwrap_or(0)
}

fn params(
    hist_nodes_per_batch: Option<usize>,
    resident_blocks: Option<usize>,
    gpu_split_cache: Option<bool>,
    subsample: f32,
    colsample_bytree: f32,
) -> TrainParams {
    let mut params = TrainParams {
        // Multiple trees prove that cache generations reset/reuse without
        // carrying stale parent packets into the next tree.
        n_rounds: 3,
        max_depth: 8,
        min_child_weight: 1.0,
        nthread: 1,
        subsample,
        colsample_bytree,
        seed: 73,
        hist_nodes_per_batch,
        resident_blocks,
        gpu_math: GpuMath::Exact,
        ..Default::default()
    };
    if let Some(enabled) = gpu_split_cache {
        params.gpu_split_cache = enabled;
    }
    params
}

#[test]
fn packet_collection_is_exact_across_windows_spills_and_cache_paths() {
    let _guard = GPU_PACKET_TEST_LOCK.lock().unwrap();
    let (source, labels) = fixture();

    // 50 features at width 16 yields four physical blocks [16, 16, 16, 2].
    // The narrow final block and missing values exercise physical-to-logical
    // feature mapping while packet batches are collected.
    assert_eq!(source.n_blocks(), 4);
    assert_eq!(source.n_features(), N_FEATURES);
    let mut found_missing = false;
    source
        .with_block(0, &mut |block| {
            found_missing = block.data.as_u8().unwrap().contains(&MISSING_BIN)
        })
        .unwrap();
    assert!(
        found_missing,
        "fixture must quantize NaNs to the missing bin"
    );

    let cpu_cfg = TrainConfig::new(
        params(None, None, Some(false), 1.0, 1.0),
        Objective::SquaredError,
    );
    let cpu = train(&source, &labels, &[], &cpu_cfg, &mut [], &Local).unwrap();
    assert!(
        max_nonleaf_width(&cpu) > 32,
        "fixture must grow past a two-generation 16-node cache window"
    );
    let expected = bytes(&cpu);

    // These caps exercise a one-node fallback, capacities below and above the
    // packet-window boundary, and the production default cap. The width
    // assertion above guarantees that every cached run eventually overflows.
    for cap in [1usize, 2, 3, 4, 16] {
        let cfg = TrainConfig::new(
            params(Some(cap), Some(2), Some(true), 1.0, 1.0),
            Objective::SquaredError,
        );
        let got = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
        assert_eq!(
            bytes(&got),
            expected,
            "packet collection changed the exact model at cache cap {cap}"
        );
    }

    // The public defaults, cache-off fallback, streaming, and partial residency
    // all share the same exact result. Two resident blocks out of four selects
    // the hybrid resident/streaming path.
    let default_cfg = TrainConfig::new(params(None, None, None, 1.0, 1.0), Objective::SquaredError);
    let default_model = train_gpu(0, &source, &labels, &[], &default_cfg, &mut [], &Local).unwrap();
    assert_eq!(
        bytes(&default_model),
        expected,
        "default cache path changed the model"
    );

    for (cache, resident) in [
        (Some(false), Some(2)),
        (Some(true), Some(0)),
        (Some(true), Some(2)),
        (Some(true), Some(4)),
    ] {
        let cfg = TrainConfig::new(
            params(Some(3), resident, cache, 1.0, 1.0),
            Objective::SquaredError,
        );
        let got = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
        assert_eq!(
            bytes(&got),
            expected,
            "cache={cache:?}, resident={resident:?}"
        );
    }

    // Sampling changes both selected rows and selected logical columns. Keep
    // the same seed on CPU/GPU and compare exact serialized models through the
    // packet overflow path, with and without residency.
    let sampled_params = params(Some(3), Some(2), Some(false), 0.6, 0.6);
    let sampled_cpu_cfg = TrainConfig::new(sampled_params.clone(), Objective::SquaredError);
    let sampled_cpu = train(&source, &labels, &[], &sampled_cpu_cfg, &mut [], &Local).unwrap();
    for (cache, resident) in [
        (Some(false), Some(2)),
        (Some(true), Some(0)),
        (Some(true), Some(2)),
        (Some(true), Some(4)),
    ] {
        let cfg = TrainConfig::new(
            params(Some(3), resident, cache, 0.6, 0.6),
            Objective::SquaredError,
        );
        let got = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
        assert_eq!(
            bytes(&got),
            bytes(&sampled_cpu),
            "sampled packet path changed the exact model: cache={cache:?}, resident={resident:?}"
        );
    }

    let mut fast_params = params(None, Some(2), Some(true), 0.6, 0.6);
    fast_params.gpu_math = GpuMath::Fast;
    let fast_cfg = TrainConfig::new(fast_params, Objective::SquaredError);
    let fast_first = train_gpu(0, &source, &labels, &[], &fast_cfg, &mut [], &Local).unwrap();
    let fast_repeat = train_gpu(0, &source, &labels, &[], &fast_cfg, &mut [], &Local).unwrap();
    assert_eq!(
        bytes(&fast_first),
        bytes(&fast_repeat),
        "sampled fast packet collection must remain deterministic across trees"
    );
    let mut fast_uncached_params = params(None, Some(2), Some(false), 0.6, 0.6);
    fast_uncached_params.gpu_math = GpuMath::Fast;
    let fast_uncached_cfg = TrainConfig::new(fast_uncached_params, Objective::SquaredError);
    let fast_uncached = train_gpu(
        0,
        &source,
        &labels,
        &[],
        &fast_uncached_cfg,
        &mut [],
        &Local,
    )
    .unwrap();
    assert_eq!(
        bytes(&fast_first),
        bytes(&fast_uncached),
        "sampled fast split cache must match its exact host fallback"
    );
}
