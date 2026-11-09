#![cfg(feature = "cuda")]

use ferrisboost::{
    backend::cuda::{
        build_histogram, build_histogram_shared, build_histogram_shared_with_config,
        partition_rows_gpu, GpuRowSpan, GpuTrainCtx, SharedHistogramConfig, SharedHistogramLayout,
    },
    comm::Local,
    hist,
    source::DenseSource,
    train::{
        train, train_gpu, train_gpu_one_tree, train_gpu_weighted, train_weighted, BlockSource,
        TrainConfig,
    },
    types::{GradPairFixed, RowId, MISSING_BIN},
};

/// Counts payload borrows while preserving DenseSource's metadata behavior.
struct CountedDenseSource {
    inner: DenseSource,
    reads: std::sync::Mutex<Vec<usize>>,
}

impl CountedDenseSource {
    fn new(inner: DenseSource) -> Self {
        let reads = vec![0; inner.n_blocks()];
        Self {
            inner,
            reads: std::sync::Mutex::new(reads),
        }
    }

    fn read_counts(&self) -> Vec<usize> {
        self.reads.lock().unwrap().clone()
    }
}

impl BlockSource for CountedDenseSource {
    fn n_blocks(&self) -> usize {
        self.inner.n_blocks()
    }

    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    fn cuts(&self) -> &ferrisboost::columns::BinCuts {
        self.inner.cuts()
    }

    fn with_block(
        &self,
        i: usize,
        f: &mut dyn FnMut(&ferrisboost::columns::ColumnBlock),
    ) -> anyhow::Result<()> {
        *self
            .reads
            .lock()
            .unwrap()
            .get_mut(i)
            .ok_or_else(|| anyhow::anyhow!("block index out of range: {i}"))? += 1;
        self.inner.with_block(i, f)
    }

    fn feature_column(
        &self,
        feat: ferrisboost::types::FeatId,
    ) -> anyhow::Result<ferrisboost::bins::BinData> {
        self.inner.feature_column(feat)
    }

    fn block_features(&self, i: usize) -> anyhow::Result<Vec<ferrisboost::types::FeatId>> {
        self.inner.block_features(i)
    }
}

struct HistogramTestDir(std::path::PathBuf);

impl HistogramTestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "fb_gpu_resident_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for HistogramTestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_parquet_resident_fixture(path: &std::path::Path, n_rows: usize, n_feats: usize) {
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let mut fields: Vec<Field> = (0..n_feats)
        .map(|feat| Field::new(format!("f{feat}"), DataType::Float32, false))
        .collect();
    fields.push(Field::new("label", DataType::Float32, false));
    let schema = Arc::new(Schema::new(fields));
    let mut columns: Vec<arrow::array::ArrayRef> = (0..n_feats)
        .map(|feat| {
            Arc::new(Float32Array::from(
                (0..n_rows)
                    .map(|row| {
                        if (row * 19 + feat * 31) % 97 == 0 {
                            f32::NAN
                        } else {
                            ((row * 17 + feat * 29 + row * feat) % 251) as f32 / 251.0
                        }
                    })
                    .collect::<Vec<_>>(),
            )) as arrow::array::ArrayRef
        })
        .collect();
    columns.push(Arc::new(Float32Array::from(
        (0..n_rows)
            .map(|row| {
                if (row * 11 + row / 5) % 23 < 11 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>(),
    )));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn assert_cpu_global_shared_equal(
    bins: &[u8],
    n_rows: usize,
    n_feats: usize,
    offsets: &[u32],
    gpair: &[GradPairFixed],
    rows: Option<&[RowId]>,
) {
    let mut cpu = vec![GradPairFixed::default(); offsets[n_feats] as usize];
    hist::build(bins, n_rows, n_feats, offsets, gpair, rows, &mut cpu);
    let global = build_histogram(0, bins, n_rows, n_feats, offsets, gpair, rows).unwrap();
    let shared = build_histogram_shared(0, bins, n_rows, n_feats, offsets, gpair, rows).unwrap();
    assert_eq!(global, cpu, "CPU/global-atomic GPU 必须逐位相同");
    assert_eq!(shared, cpu, "CPU/shared-memory GPU 必须逐位相同");
    assert_eq!(shared, global, "两个 GPU 实现必须逐位相同");
}

#[test]
fn resident_block_reuse_preserves_training_and_avoids_repeated_source_reads() {
    let n_rows = 640usize;
    let n_feats = 70usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 19 + feat * 31) % 97 == 0 {
                    f32::NAN
                } else {
                    ((row * 17 + feat * 29 + row * feat) % 251) as f32 / 251.0
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| {
            if (row * 11 + row / 5) % 23 < 11 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    // [24, 24, 22]: the final physical block is narrower than the others.
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 255, 24);
    assert_eq!(source.n_blocks(), 3);
    let bytes = |m: &ferrisboost::tree::Model| serde_json::to_vec(m).unwrap();

    let cpu_params = ferrisboost::types::TrainParams {
        n_rounds: 4,
        max_depth: 5,
        nthread: 1,
        subsample: 0.6,
        colsample_bytree: 0.6,
        seed: 73,
        gpu_math: ferrisboost::types::GpuMath::Exact,
        ..Default::default()
    };
    let cpu_cfg = TrainConfig::new(cpu_params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cpu_cfg, &mut [], &Local).unwrap();

    // Rebuild the counted wrapper per run so every vector starts at zero.
    for gpu_split_cache in [false, true] {
        for resident_blocks in [0usize, 1, 3] {
            let run_source = CountedDenseSource::new(DenseSource::from_row_major(
                &data, n_rows, n_feats, 255, 24,
            ));
            let params = ferrisboost::types::TrainParams {
                n_rounds: 4,
                max_depth: 5,
                nthread: 1,
                subsample: 0.6,
                colsample_bytree: 0.6,
                seed: 73,
                resident_blocks: Some(resident_blocks),
                gpu_split_cache,
                gpu_math: ferrisboost::types::GpuMath::Exact,
                ..Default::default()
            };
            let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
            let gpu = train_gpu(0, &run_source, &labels, &[], &cfg, &mut [], &Local).unwrap();
            let reads = run_source.read_counts();
            assert_eq!(
                bytes(&gpu),
                bytes(&cpu),
                "resident_blocks={resident_blocks}, gpu_split_cache={gpu_split_cache} changed the exact GPU model"
            );

            match resident_blocks {
                0 => assert!(
                    reads.iter().all(|&count| count > 1),
                    "streaming should reread each selected physical block across rounds: {reads:?}"
                ),
                1 => {
                    assert!(reads[0] <= 1, "first resident block was reread: {reads:?}");
                    assert!(
                        reads[1..].iter().all(|&count| count > 1),
                        "nonresident blocks should use the streaming path: {reads:?}"
                    );
                }
                3 => assert!(
                    reads.iter().all(|&count| count <= 1) && reads.iter().sum::<usize>() <= 3,
                    "full residency should read each physical block only for initial upload: {reads:?}"
                ),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn resident_block_session_reuses_bins_and_preserves_metadata_validation() {
    let n_rows = 24usize;
    let n_feats = 5usize;
    let feat_offsets = [0u32, 3, 7, 10, 14, 19];
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            let width = (feat_offsets[feat + 1] - feat_offsets[feat]) as usize;
            (0..n_rows).map(move |row| ((row * 3 + feat * 5) % width) as u8)
        })
        .collect();
    let gpair: Vec<GradPairFixed> = (0..n_rows)
        .map(|row| GradPairFixed {
            grad: (row as i64 % 7) - 3,
            hess: (row as i64 % 5) + 1,
        })
        .collect();
    let config = SharedHistogramConfig {
        resident_blocks: Some(1),
        ..Default::default()
    };
    let ctx = GpuTrainCtx::new(0, n_rows, n_feats, 19, 2, 8, config, false).unwrap();
    ctx.upload_gpair(&gpair).unwrap();
    let rows = ctx.begin_tree_rows().unwrap();

    assert!(
        ctx.block_session_resident(0, n_rows, n_feats, &feat_offsets, None)
            .unwrap()
            .is_none(),
        "a resident lookup before the first payload upload must miss"
    );
    assert!(
        ctx.block_session_resident(1, n_rows, n_feats, &feat_offsets, None)
            .unwrap()
            .is_none(),
        "a physical block outside the resident range must miss"
    );

    let mut payload_hist = Vec::new();
    {
        let mut session = ctx
            .block_session(0, &bins, n_rows, n_feats, &feat_offsets, None)
            .unwrap();
        session.histogram_device(rows, &mut payload_hist).unwrap();
    }
    let mut resident_hist = Vec::new();
    let resident_timing;
    {
        let mut session = ctx
            .block_session_resident(0, n_rows, n_feats, &feat_offsets, None)
            .unwrap()
            .expect("uploaded resident block should hit without host bins");
        session.histogram_device(rows, &mut resident_hist).unwrap();
        resident_timing = session.timing();
    }
    assert_eq!(
        resident_hist, payload_hist,
        "resident metadata lookup changed the histogram"
    );
    assert_eq!(resident_timing.block_h2d_calls, 0);
    assert_eq!(resident_timing.block_h2d_bytes, 0);

    let oversized_offsets = [0u32; 7];
    let capacity_error = ctx
        .block_session_resident(0, 0, n_feats + 1, &oversized_offsets, None)
        .err()
        .expect("warm resident lookup beyond its offsets capacity must be rejected");
    assert!(
        capacity_error.to_string().contains("metadata capacity"),
        "unexpected oversized metadata error: {capacity_error}"
    );

    let row_error = ctx
        .block_session_resident(0, n_rows - 1, n_feats, &feat_offsets, None)
        .err()
        .expect("resident row-count mismatch must be rejected");
    assert!(row_error.to_string().contains("resident block shape"));
    let short_offsets = &feat_offsets[..n_feats];
    assert!(
        ctx.block_session_resident(0, n_rows, n_feats - 1, short_offsets, None)
            .is_err(),
        "resident feature-count mismatch must be rejected"
    );
    assert!(
        ctx.block_session_resident(0, n_rows, n_feats, &[0, 3, 7, 6, 14, 19], None)
            .is_err(),
        "resident lookup must retain monotonic-offset validation"
    );
    assert!(
        ctx.block_session_resident(0, n_rows, n_feats, &feat_offsets, Some(&[0, 2, 2]))
            .is_err(),
        "resident lookup must retain selected-feature ordering validation"
    );

    let mut post_error_hist = Vec::new();
    let post_error_timing;
    {
        let mut session = ctx
            .block_session_resident(0, n_rows, n_feats, &feat_offsets, None)
            .unwrap()
            .expect("invalid metadata requests must not evict the warm resident payload");
        session
            .histogram_device(rows, &mut post_error_hist)
            .unwrap();
        post_error_timing = session.timing();
    }
    assert_eq!(post_error_hist, payload_hist);
    assert_eq!(post_error_timing.block_h2d_calls, 0);
    assert_eq!(post_error_timing.block_h2d_bytes, 0);
}

#[test]
fn resident_block_session_reuses_exact_metadata_and_refreshes_selected_mapping() {
    let n_rows = 24usize;
    let n_feats = 5usize;
    let offsets = [0u32, 2, 4, 6, 8, 10];
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            (0..n_rows).map(move |row| match feat {
                0 => 0,
                1 => 1,
                _ => ((row + feat) % 2) as u8,
            })
        })
        .collect();
    let gpair = vec![GradPairFixed { grad: 1, hess: 1 }; n_rows];
    let config = SharedHistogramConfig {
        resident_blocks: Some(1),
        ..Default::default()
    };
    let ctx = GpuTrainCtx::new(0, n_rows, n_feats, 20, 2, 8, config, false).unwrap();
    ctx.upload_gpair(&gpair).unwrap();
    let rows = ctx.begin_tree_rows().unwrap();
    let compact_column = ctx.supports_compact_column_histogram();
    let use_batched = compact_column;

    let run = |selected: Option<&[usize]>, offsets: &[u32]| {
        let mut histogram = Vec::new();
        let timing;
        {
            let mut session = ctx
                .block_session_resident(0, n_rows, n_feats, offsets, selected)
                .unwrap()
                .expect("resident payload should remain available");
            if use_batched {
                let spans = [rows];
                let mut outputs = [&mut histogram];
                session
                    .histogram_device_batch(&spans, &mut outputs)
                    .unwrap();
            } else {
                assert!(selected.is_none());
                session.histogram_device(rows, &mut histogram).unwrap();
            }
            timing = session.timing();
        }
        (histogram, timing)
    };

    // Seed resident physical block 0 once with its full-width payload. Later
    // sessions borrow these bins while varying only histogram metadata.
    let mut initial_histogram = Vec::new();
    let initial_timing;
    {
        let mut session = ctx
            .block_session(0, &bins, n_rows, n_feats, &offsets, None)
            .unwrap();
        if use_batched {
            let spans = [rows];
            let mut outputs = [&mut initial_histogram];
            session
                .histogram_device_batch(&spans, &mut outputs)
                .unwrap();
        } else {
            session
                .histogram_device(rows, &mut initial_histogram)
                .unwrap();
        }
        initial_timing = session.timing();
    }
    assert_eq!(initial_timing.block_h2d_calls, 2);
    assert_eq!(
        initial_timing.block_h2d_bytes,
        (bins.len() + offsets.len() * 4) as u64
    );

    // Nonresident physical blocks can share a temporary streaming slot, but
    // must each upload their own payload. This catches accidentally treating a
    // reused slot's prior physical identity as a resident cache hit.
    let mut streaming_results = Vec::new();
    for block_index in [1usize, 2] {
        let streaming_bins = if block_index == 1 {
            bins.clone()
        } else {
            bins.iter().map(|bin| 1 - bin).collect::<Vec<_>>()
        };
        let mut histogram = Vec::new();
        let timing;
        {
            let mut session = ctx
                .block_session(
                    block_index,
                    &streaming_bins,
                    n_rows,
                    n_feats,
                    &offsets,
                    None,
                )
                .unwrap();
            if use_batched {
                let spans = [rows];
                let mut outputs = [&mut histogram];
                session
                    .histogram_device_batch(&spans, &mut outputs)
                    .unwrap();
            } else {
                session.histogram_device(rows, &mut histogram).unwrap();
            }
            timing = session.timing();
        }
        assert_eq!(timing.block_h2d_calls, 2);
        assert_eq!(
            timing.block_h2d_bytes,
            (bins.len() + offsets.len() * 4) as u64
        );
        streaming_results.push(histogram);
    }
    assert_ne!(streaming_results[0], streaming_results[1]);

    // An exact full-width metadata hit must issue no metadata copy in any
    // supported histogram layout.
    let (_, full_hit_timing) = run(None, &offsets);
    assert_eq!(full_hit_timing.block_h2d_calls, 0);
    assert_eq!(full_hit_timing.block_h2d_bytes, 0);

    let selected_a = [0usize, 2];
    let selected_b = [1usize, 3];
    let expected_compact = |selected: &[usize]| {
        let compact_bins: Vec<u8> = selected
            .iter()
            .flat_map(|&feat| bins[feat * n_rows..(feat + 1) * n_rows].iter().copied())
            .collect();
        let compact_offsets = [0u32, 2, 4];
        let mut expected = vec![GradPairFixed::default(); 4];
        hist::build(
            &compact_bins,
            n_rows,
            selected.len(),
            &compact_offsets,
            &gpair,
            None,
            &mut expected,
        );
        expected
    };
    if compact_column {
        // First use uploads compact offsets and the selected-local map.
        let (first, first_timing) = run(Some(&selected_a), &[0, 2, 4]);
        assert_eq!(first_timing.block_h2d_calls, 2);
        assert_eq!(first_timing.block_h2d_bytes, 3 * 4 + 2 * 4);
        assert_eq!(first, expected_compact(&selected_a));
        let (same, same_timing) = run(Some(&selected_a), &[0, 2, 4]);
        assert_eq!(same, first);
        assert_eq!(same_timing.block_h2d_calls, 0);
        assert_eq!(same_timing.block_h2d_bytes, 0);

        // Logical width and offsets stay identical, while physical selected IDs
        // change. The map must refresh and produce a different histogram.
        let (remapped, remap_timing) = run(Some(&selected_b), &[0, 2, 4]);
        assert_eq!(remap_timing.block_h2d_calls, 1);
        assert_eq!(remap_timing.block_h2d_bytes, 2 * 4);
        assert_eq!(remapped, expected_compact(&selected_b));
        assert_ne!(
            remapped, first,
            "changed selected features reused a stale map"
        );

        // A valid offset change uploads on a warm resident payload; returning
        // to the sampled metadata covers sampled → full → sampled transitions.
        let changed_offsets = [0u32, 3, 6, 9, 12, 15];
        let (changed, changed_timing) = run(None, &changed_offsets);
        assert_eq!(changed_timing.block_h2d_calls, 1);
        assert_eq!(
            changed_timing.block_h2d_bytes,
            changed_offsets.len() as u64 * 4
        );
        assert_eq!(changed.len(), 15);
        let (sampled_again, sampled_timing) = run(Some(&selected_b), &[0, 2, 4]);
        assert_eq!(sampled_timing.block_h2d_calls, 2);
        assert_eq!(sampled_timing.block_h2d_bytes, 3 * 4 + 2 * 4);
        assert_eq!(sampled_again, remapped);
    } else {
        // The legacy bucket-profile column-major ABI has no compact map and
        // ignores selected-local, so cover full-width metadata only.
        let changed_offsets = [0u32, 3, 6, 9, 12, 15];
        let (changed, changed_timing) = run(None, &changed_offsets);
        assert_eq!(changed_timing.block_h2d_calls, 1);
        assert_eq!(
            changed_timing.block_h2d_bytes,
            changed_offsets.len() as u64 * 4
        );
        assert_eq!(changed.len(), 15);
        let (restored, restored_timing) = run(None, &offsets);
        assert_eq!(restored_timing.block_h2d_calls, 1);
        assert_eq!(restored_timing.block_h2d_bytes, offsets.len() as u64 * 4);
        assert_eq!(restored, initial_histogram);
    }
}

#[test]
fn resident_disk_blocks_are_read_once_and_corruption_fails_before_upload() {
    let dir = HistogramTestDir::new();
    let parquet_path = dir.0.join("training.parquet");
    let cache_path = dir.0.join("quantized-cache");
    let n_rows = 384usize;
    let n_feats = 30usize;
    write_parquet_resident_fixture(&parquet_path, n_rows, n_feats);
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| {
            if (row * 11 + row / 5) % 23 < 11 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let options = ferrisboost::source::parquet_source::QuantizedCacheOptions {
        budget_bytes: Some(0),
        path: Some(cache_path.clone()),
    };
    let open_source = || {
        ferrisboost::source::parquet_source::ParquetBlockSource::open_with_cache(
            &parquet_path,
            "label",
            255,
            16,
            options.clone(),
        )
        .unwrap()
        .0
    };

    let cpu_source = open_source();
    assert_eq!(cpu_source.cache_kind(), "disk");
    let params = ferrisboost::types::TrainParams {
        n_rounds: 2,
        max_depth: 5,
        nthread: 1,
        subsample: 0.6,
        colsample_bytree: 0.6,
        seed: 73,
        gpu_math: ferrisboost::types::GpuMath::Exact,
        ..Default::default()
    };
    let cpu_cfg = TrainConfig::new(params.clone(), ferrisboost::types::Objective::Logistic);
    let cpu = train(&cpu_source, &labels, &[], &cpu_cfg, &mut [], &Local).unwrap();

    let gpu_source = open_source();
    let n_blocks = gpu_source.n_blocks();
    assert_eq!(n_blocks, 2);
    let gpu_params = ferrisboost::types::TrainParams {
        resident_blocks: Some(n_blocks),
        ..params.clone()
    };
    let gpu_cfg = TrainConfig::new(gpu_params, ferrisboost::types::Objective::Logistic);
    let gpu = train_gpu(0, &gpu_source, &labels, &[], &gpu_cfg, &mut [], &Local).unwrap();
    assert_eq!(
        serde_json::to_vec(&gpu).unwrap(),
        serde_json::to_vec(&cpu).unwrap(),
        "fully resident spill-cache training changed the exact model"
    );
    assert_eq!(
        gpu_source.cache_read_count(),
        n_blocks,
        "each disk-backed resident block should be fetched only for its initial upload"
    );
    assert_eq!(
        gpu_source.source_bytes(),
        (n_rows * n_feats) as u64,
        "resident partitioning should not read disk-backed feature payloads after the first histogram"
    );

    // Open the cache first, then mutate one payload without changing its length.
    // The first resident upload must verify the checksum before handing bytes to CUDA.
    let corrupt_source = open_source();
    let payload_path = cache_path.join("block-0.bin");
    let mut payload = std::fs::read(&payload_path).unwrap();
    assert!(!payload.is_empty());
    payload[0] ^= 1;
    std::fs::write(&payload_path, &payload).unwrap();
    assert_eq!(
        std::fs::metadata(&payload_path).unwrap().len(),
        payload.len() as u64
    );

    let corrupt_params = ferrisboost::types::TrainParams {
        resident_blocks: Some(corrupt_source.n_blocks()),
        ..params
    };
    let corrupt_cfg = TrainConfig::new(corrupt_params, ferrisboost::types::Objective::Logistic);
    let error = train_gpu(
        0,
        &corrupt_source,
        &labels,
        &[],
        &corrupt_cfg,
        &mut [],
        &Local,
    )
    .expect_err("corrupted first-use disk block must fail checksum validation");
    assert!(
        format!("{error:#}").contains("payload checksum mismatch"),
        "unexpected corruption error: {error:#}"
    );
}

#[test]
/// `partition_scan_blocks` 是**单 block 多线程的分块 scan**:tile 内做
/// Hillis-Steele,tile 之间用 carry 串起来。**carry 那条路只有在
/// `n_blocks > blockDim.x`(1024)时才会走到**,而其它 partition 测试的夹具
/// 只有几十到几千行 —— `n_blocks` 个位数,永远是单 tile。
///
/// 所以这里专门造一个跨过阈值的规模:partition 的 row block 是 512 行,
/// 要 `n_blocks > 1024` 就需要 > 524,288 行。取 1,200,000 行 →
/// n_blocks = 2344 → 3 个 tile,carry 被真正累加两次。
///
/// (这条测试的由来:`node_sum` 并行分支当初就是因为所有夹具都低于阈值而
/// 完全没有 CI 覆盖。凡是「够大才启用」的快路径,都要有跨过阈值的测试。)
fn gpu_partition_scan_spans_multiple_scan_tiles() {
    let n_rows = 1_200_000usize;
    // 逆序行下标:顺便钉住 stable 顺序不依赖输入是否升序。
    let rows: Vec<RowId> = (0..n_rows as RowId).rev().collect();
    // 让左右都不是平凡的全有/全无,并掺入缺失。
    let bins: Vec<u8> = (0..n_rows)
        .map(|row| {
            if row % 97 == 0 {
                MISSING_BIN
            } else {
                (row % 7) as u8
            }
        })
        .collect();
    for missing_left in [false, true] {
        let (left, right) =
            partition_rows_gpu(0, &bins, n_rows, 0, &rows, 3, missing_left, 512).unwrap();
        let goes_left = |&row: &RowId| {
            let bin = bins[row as usize];
            if bin == MISSING_BIN {
                missing_left
            } else {
                (bin as u32) < 3
            }
        };
        let want_left: Vec<_> = rows.iter().filter(|r| goes_left(r)).copied().collect();
        let want_right: Vec<_> = rows.iter().filter(|r| !goes_left(r)).copied().collect();
        assert_eq!(left.len() + right.len(), n_rows, "行数必须守恒");
        assert!(left.len() > 1000 && right.len() > 1000, "两侧都要非平凡");
        assert_eq!(left, want_left, "跨 scan tile 的左侧顺序必须和串行参考一致");
        assert_eq!(
            right, want_right,
            "跨 scan tile 的右侧顺序必须和串行参考一致"
        );
    }
}

#[test]
fn gpu_partition_matches_stable_cpu_order() {
    let n_rows = 37usize;
    let n_feats = 2usize;
    let rows: Vec<RowId> = (0..n_rows as RowId).rev().collect();
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            (0..n_rows).map(move |row| {
                if row % 11 == 0 {
                    MISSING_BIN
                } else {
                    ((row + feat * 2) % 7) as u8
                }
            })
        })
        .collect();
    for missing_left in [false, true] {
        let (left, right) =
            partition_rows_gpu(0, &bins, n_rows, 1, &rows, 3, missing_left, 8).unwrap();
        let goes_left = |&row: &RowId| {
            let bin = bins[n_rows + row as usize];
            if bin == MISSING_BIN {
                missing_left
            } else {
                (bin as u32) < 3
            }
        };
        let want_left: Vec<_> = rows.iter().filter(|row| goes_left(row)).copied().collect();
        let want_right: Vec<_> = rows.iter().filter(|row| !goes_left(row)).copied().collect();
        assert_eq!(left, want_left);
        assert_eq!(right, want_right);
        let (left_again, right_again) =
            partition_rows_gpu(0, &bins, n_rows, 1, &rows, 3, missing_left, 8).unwrap();
        assert_eq!(left_again, left);
        assert_eq!(right_again, right);
    }
}

#[test]
fn training_partition_keeps_stable_rows_on_device() {
    let n_rows = 1_037usize;
    let col: Vec<u8> = (0..n_rows)
        .map(|row| {
            if row % 17 == 0 {
                MISSING_BIN
            } else {
                (row % 7) as u8
            }
        })
        .collect();
    let ctx = GpuTrainCtx::new(
        0,
        n_rows,
        1,
        8,
        1,
        512,
        SharedHistogramConfig::default(),
        false,
    )
    .unwrap();
    let root = ctx.begin_tree_rows().unwrap();
    let token = {
        let mut session = ctx.partition_session(&col).unwrap();
        session.partition(root, 3, true).unwrap()
    };
    // partition 现在只 launch;左右行数由整层一次 D2H 取回。
    let (left, right) = ctx.resolve_partitions().unwrap()[token];
    ctx.finish_partition_level().unwrap();
    let got_left = ctx.download_rows(left).unwrap();
    let got_right = ctx.download_rows(right).unwrap();
    let want_left: Vec<_> = (0..n_rows as RowId)
        .filter(|&row| col[row as usize] == MISSING_BIN || col[row as usize] < 3)
        .collect();
    let want_right: Vec<_> = (0..n_rows as RowId)
        .filter(|&row| col[row as usize] != MISSING_BIN && col[row as usize] >= 3)
        .collect();
    assert_eq!(got_left, want_left);
    assert_eq!(got_right, want_right);
}

#[test]
fn gpu_one_tree_streaming_matches_cpu_model() {
    let n_rows = 200usize;
    let n_feats = 5usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 3) % 19) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 11) % 23 < 9 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 16, 2);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 1,
        max_depth: 1,
        nthread: 1,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let gpu = train_gpu_one_tree(0, &source, &labels, &cfg, &Local).unwrap();
    assert_eq!(
        serde_json::to_vec(&gpu).unwrap(),
        serde_json::to_vec(&cpu).unwrap(),
        "流式 GPU 单树模型必须与 CPU 逐字节相同"
    );
}

#[test]
fn gpu_multilevel_training_matches_cpu_model() {
    let n_rows = 200usize;
    let n_feats = 5usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 5 + feat * 7) % 23) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 13) % 29 < 12 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 16, 2);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 2,
        max_depth: 2,
        nthread: 1,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let gpu = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    assert_eq!(
        serde_json::to_vec(&gpu).unwrap(),
        serde_json::to_vec(&cpu).unwrap()
    );
}

/// 训练级资源复用的验收:多轮 × 多层 × 多列块 × 带缺失值。
///
/// 盯的是资源生命周期改造**一个数字都不能动**。这些形状专门去踩复用
/// 逻辑里容易出错的地方:
///
/// - 多个列块在两套 stream-local bins buffer 间复用(不能跨 stream 覆写)
/// - 一个列块内多个活跃节点共用一次 H2D(块只上传一次)
/// - depth 0 的 identity 行索引在块之间复用、但每轮要重新装
/// - partition 的输出 buffer 跨节点复用(只读写各自那一段)
/// - 行数不是 threads_per_block 的整数倍,tail launch 必须走到
#[test]
fn gpu_training_reuses_resources_without_changing_the_model() {
    let n_rows = 1_037usize; // 刻意不是 512 的倍数,逼出 tail launch
    let n_feats = 13usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 7 + feat) % 17 == 0 {
                    f32::NAN // 缺失值:默认方向要在 GPU/CPU 两边一致
                } else {
                    ((row * 11 + feat * 5) % 41) as f32
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 19) % 37 < 15 { 1.0 } else { 0.0 })
        .collect();
    // 4 个列块 × 4 层 × 3 轮:bins buffer 被覆写 48 次以上。
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 4);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 3,
        max_depth: 4,
        max_bin: 32,
        cols_per_block: 4,
        nthread: 1,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let gpu = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    assert!(
        gpu.trees.iter().any(|t| t.nodes.len() > 1),
        "这个夹具应该真的分裂"
    );
    assert_eq!(
        gpu.to_xgboost_json().unwrap(),
        cpu.to_xgboost_json().unwrap(),
        "复用 device buffer 之后模型仍必须和 CPU 逐字节相同"
    );
}

/// 同时覆盖 200 行尾部、不同特征 bin 宽、缺失值和根节点 null row_idx。
#[test]
fn gpu_histogram_matches_cpu_for_all_rows() {
    let n_rows = 200usize;
    let widths = [2u32, 5, 1, 7, 3];
    let mut offsets = vec![0u32];
    for width in widths {
        offsets.push(offsets.last().unwrap() + width);
    }

    let mut bins = Vec::with_capacity(n_rows * widths.len());
    for (feat, &width) in widths.iter().enumerate() {
        for row in 0..n_rows {
            let bin = if (row * 11 + feat * 7) % 29 == 0 {
                MISSING_BIN
            } else {
                ((row * 3 + feat) % width as usize) as u8
            };
            bins.push(bin);
        }
    }
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| GradPairFixed {
            grad: (row as i64 % 17) - 8,
            hess: (row as i64 % 13) + 1,
        })
        .collect();

    assert_cpu_global_shared_equal(&bins, n_rows, widths.len(), &offsets, &gpair, None);
}

/// 子集长度和内容都与物理列 stride 不同,守住 FFI 中两者不能混用。
#[test]
fn gpu_histogram_matches_cpu_for_row_subset() {
    let n_rows = 200usize;
    let n_feats = 5usize;
    let offsets = [0u32, 2, 7, 8, 15, 18];
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            let width = (offsets[feat + 1] - offsets[feat]) as usize;
            (0..n_rows).map(move |row| {
                if (row + feat * 5) % 31 == 0 {
                    MISSING_BIN
                } else {
                    ((row * 7 + feat * 3) % width) as u8
                }
            })
        })
        .collect();
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| GradPairFixed {
            grad: 101 - row as i64,
            hess: row as i64 * 2 + 3,
        })
        .collect();
    let rows: Vec<RowId> = (0..n_rows as RowId)
        .filter(|row| row % 3 == 1 || row % 17 == 0)
        .collect();

    assert_cpu_global_shared_equal(&bins, n_rows, n_feats, &offsets, &gpair, Some(&rows));
}

#[test]
fn gpu_histogram_handles_single_feature_single_row() {
    let bins = [2u8];
    let offsets = [0u32, 3];
    let gpair = [GradPairFixed { grad: -7, hess: 11 }];
    assert_cpu_global_shared_equal(&bins, 1, 1, &offsets, &gpair, None);

    let missing = [MISSING_BIN];
    let rows = [0u32];
    assert_cpu_global_shared_equal(&missing, 1, 1, &offsets, &gpair, Some(&rows));
}

/// Reuse one histogram slot while active D2H lengths grow, shrink, then grow
/// again. The batches include empty spans between non-empty spans so their
/// output slices must remain zero while the neighboring histograms are fresh.
#[test]
fn reused_histogram_slot_overwrites_growing_and_shrinking_d2h_slices() {
    let n_rows = 128usize;
    let config = SharedHistogramConfig {
        runtime_log: false,
        threads_per_block: 128,
        hist_streams: Some(1),
        hist_nodes_per_batch: Some(4),
        resident_blocks: Some(0),
        ..SharedHistogramConfig::default()
    };
    let ctx = GpuTrainCtx::new(0, n_rows, 4, 8, 1, 128, config, false).unwrap();
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| GradPairFixed {
            grad: (row as i64 % 17) - 8,
            hess: (row as i64 % 11) + 1,
        })
        .collect();
    ctx.upload_gpair(&gpair).unwrap();
    ctx.begin_tree_rows().unwrap();

    let make_bins = |offsets: &[u32]| {
        let n_feats = offsets.len() - 1;
        (0..n_feats)
            .flat_map(|feat| {
                let width = (offsets[feat + 1] - offsets[feat]) as usize;
                (0..n_rows).map(move |row| {
                    if (row + feat * 13) % 29 == 0 {
                        MISSING_BIN
                    } else {
                        ((row * 7 + feat * 3) % width) as u8
                    }
                })
            })
            .collect::<Vec<_>>()
    };
    let check_batch = |offsets: &[u32], spans: &[GpuRowSpan]| {
        let bins = make_bins(offsets);
        let n_feats = offsets.len() - 1;
        let mut outputs: Vec<Vec<GradPairFixed>> = (0..spans.len())
            .map(|_| {
                vec![
                    GradPairFixed {
                        grad: 91,
                        hess: -37
                    };
                    9
                ]
            })
            .collect();
        let (batched, timing) = {
            let mut batched = false;
            let timing;
            {
                let mut session = ctx
                    .block_session(0, &bins, n_rows, n_feats, offsets, None)
                    .unwrap();
                if session.hist_nodes_batch_capacity() > 1 || session.requires_batched_hist() {
                    batched = true;
                    let mut out_refs: Vec<_> = outputs.iter_mut().collect();
                    session
                        .histogram_device_batch(spans, &mut out_refs)
                        .unwrap();
                } else {
                    // Bucket profiling disables the batched kernel; exercise the
                    // same slot sequence through the single-node fallback there.
                    for (span, out) in spans.iter().zip(&mut outputs) {
                        session.histogram_device(*span, out).unwrap();
                    }
                }
                timing = session.timing();
            }
            (batched, timing)
        };
        let n_hist = *offsets.last().unwrap() as usize;
        let nonempty = spans.iter().filter(|span| span.len > 0).count();
        let expected_calls = if batched {
            usize::from(nonempty > 0)
        } else {
            nonempty
        };
        let expected_bytes = if batched && nonempty > 0 {
            n_hist * spans.len() * std::mem::size_of::<GradPairFixed>()
        } else {
            n_hist * nonempty * std::mem::size_of::<GradPairFixed>()
        };
        assert_eq!(timing.d2h_calls, expected_calls as u64);
        assert_eq!(timing.d2h_bytes, expected_bytes as u64);
        for (index, (&span, got)) in spans.iter().zip(&outputs).enumerate() {
            let rows: Vec<RowId> = (span.offset..span.offset + span.len)
                .map(|row| row as RowId)
                .collect();
            let mut want = vec![GradPairFixed::default(); *offsets.last().unwrap() as usize];
            hist::build(
                &bins,
                n_rows,
                n_feats,
                offsets,
                &gpair,
                Some(rows.as_slice()),
                &mut want,
            );
            assert_eq!(got, &want, "batch node {index}, span={span:?}");
        }
    };

    // Active D2H size: 6 bins × 4 nodes, then 1 × 1, then 3 × 4.
    // Empty spans are deliberately interspersed with non-empty nodes.
    check_batch(
        &[0, 2, 4, 6],
        &[
            GpuRowSpan {
                offset: 0,
                len: 128,
            },
            GpuRowSpan { offset: 0, len: 0 },
            GpuRowSpan { offset: 0, len: 64 },
            GpuRowSpan { offset: 64, len: 0 },
        ],
    );
    check_batch(
        &[0, 2, 4, 6],
        &[
            GpuRowSpan { offset: 0, len: 0 },
            GpuRowSpan { offset: 32, len: 0 },
            GpuRowSpan { offset: 64, len: 0 },
            GpuRowSpan { offset: 96, len: 0 },
        ],
    );
    check_batch(
        &[0, 1],
        &[GpuRowSpan {
            offset: 0,
            len: 128,
        }],
    );
    check_batch(
        &[0, 1, 3],
        &[
            GpuRowSpan { offset: 0, len: 32 },
            GpuRowSpan { offset: 32, len: 0 },
            GpuRowSpan {
                offset: 32,
                len: 32,
            },
            GpuRowSpan {
                offset: 64,
                len: 64,
            },
        ],
    );

    // Also cover the single-node D2H call site through the same reused slot.
    for offsets in [&[0, 2, 4, 6][..], &[0, 1][..], &[0, 1, 3][..]] {
        let bins = make_bins(offsets);
        let n_feats = offsets.len() - 1;
        let span = GpuRowSpan {
            offset: 0,
            len: n_rows,
        };
        let mut got = vec![
            GradPairFixed {
                grad: 91,
                hess: -37
            };
            9
        ];
        let timing;
        {
            let mut session = ctx
                .block_session(0, &bins, n_rows, n_feats, offsets, None)
                .unwrap();
            session.histogram_device(span, &mut got).unwrap();
            timing = session.timing();
        }
        let mut want = vec![GradPairFixed::default(); *offsets.last().unwrap() as usize];
        hist::build(&bins, n_rows, n_feats, offsets, &gpair, None, &mut want);
        assert_eq!(got, want, "single-node offsets={offsets:?}");
        assert_eq!(timing.d2h_calls, 1);
        assert_eq!(
            timing.d2h_bytes,
            (want.len() * std::mem::size_of::<GradPairFixed>()) as u64
        );
    }

    // A zero-row scalar call has no D2H, but must replace stale output with a
    // correctly sized zero histogram before the next reuse of this slot.
    let offsets = [0, 2, 4];
    let bins = make_bins(&offsets);
    let mut got = vec![
        GradPairFixed {
            grad: 91,
            hess: -37
        };
        9
    ];
    let timing;
    {
        let mut session = ctx
            .block_session(0, &bins, n_rows, 2, &offsets, None)
            .unwrap();
        session
            .histogram_device(GpuRowSpan { offset: 0, len: 0 }, &mut got)
            .unwrap();
        timing = session.timing();
    }
    assert_eq!(got, vec![GradPairFixed::default(); 4]);
    assert_eq!(timing.d2h_calls, 0);
    assert_eq!(timing.d2h_bytes, 0);
}

/// 30 × 255 bins 需要 122,400 bytes,超过测试路径配置的单-block
/// shared-memory budget,保证 host 真的跨多个 feature tile 启动。
/// 257 行也保证第二个 row block 带尾部。
#[test]
fn shared_histogram_tiles_features_and_rows() {
    let n_rows = 257usize;
    let n_feats = 30usize;
    let width = 255u32;
    let offsets: Vec<u32> = (0..=n_feats).map(|feat| feat as u32 * width).collect();
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            (0..n_rows).map(move |row| {
                if (row * 13 + feat * 17) % 41 == 0 {
                    MISSING_BIN
                } else {
                    ((row * 7 + feat * 11) % width as usize) as u8
                }
            })
        })
        .collect();
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| GradPairFixed {
            grad: (row as i64 % 31) - 15,
            hess: row as i64 * 3 + 1,
        })
        .collect();

    assert_cpu_global_shared_equal(&bins, n_rows, n_feats, &offsets, &gpair, None);

    let mut cpu = vec![GradPairFixed::default(); offsets[n_feats] as usize];
    hist::build(&bins, n_rows, n_feats, &offsets, &gpair, None, &mut cpu);
    for layout in SharedHistogramLayout::ALL {
        let config = SharedHistogramConfig {
            runtime_log: false,
            resident_blocks: None,
            gpu_memory_budget: None,
            hist_streams: None,
            device_quantize: None,
            hist_nodes_per_batch: None,
            gpu_split_cache: false,
            gpu_math: ferrisboost::types::GpuMath::Exact,
            fast_weights: false,
            n_outputs: 1,
            multi_quantile: false,
            has_categorical: false,
            node_feature_masks: false,
            fast_rank: None,
            threads_per_block: 64,
            shared_memory_budget_bytes: Some(16 * 1024),
            layout,
        };
        let (configured, profile) = build_histogram_shared_with_config(
            0, &bins, n_rows, n_feats, &offsets, &gpair, None, config,
        )
        .unwrap();
        assert_eq!(configured, cpu, "{layout:?} 必须逐位相同");
        assert!(profile.feature_tiles > 1);
        assert_eq!(profile.threads_per_block, 64);
        assert_eq!(profile.shared_memory_budget_bytes, 16 * 1024);
        assert_eq!(profile.layout, layout);
        assert!(profile.active_blocks_per_sm > 0);
        assert!(profile.theoretical_occupancy > 0.0);
        assert!(profile.theoretical_occupancy <= 1.0);
    }
}

/// 逼出 **32-bit 拆分累加的进位路径**。
///
/// `Interleaved16Atomic32` 把一次 64-bit shared atomicAdd 拆成两次 32-bit
/// 加一次手动进位。**进位只在低 32 位溢出时才发生**,而上面那个 layout
/// 夹具的 grad 只有 -15..15、hess 不到 800,每个 bin 的和离 2^32 差着好几
/// 个数量级 —— 也就是说它**从来没走到过进位那一支**,全绿也证明不了什么。
///
/// 这里按真实定点量级构造:`pow2_scale` 的预算是 `2^62 / n_rows`,
/// 所以单值量级本来就在 2^38 附近。再让大量行落进同一个 bin,低 32 位会
/// 反复溢出;负 grad 的二补数低位接近 0xFFFFFFFF,更是每加一次就进位。
#[test]
fn atomic32_layout_is_bit_exact_when_low_word_overflows() {
    let n_rows = 4_096usize;
    let n_feats = 8usize;
    let width = 16u32; // 故意窄:行多、bin 少,单个 bin 累加上千次
    let offsets: Vec<u32> = (0..=n_feats).map(|feat| feat as u32 * width).collect();
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            (0..n_rows).map(move |row| {
                if (row * 3 + feat) % 97 == 0 {
                    MISSING_BIN
                } else {
                    ((row + feat) % width as usize) as u8
                }
            })
        })
        .collect();
    // 量级贴着真实定点预算(2^62 / n_rows);混合正负,并特意让一部分值的
    // 低 32 位紧贴 0xFFFFFFFF,保证进位分支被反复走到。
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| {
            let big = 1i64 << 38;
            let grad = match row % 4 {
                0 => big + row as i64,
                1 => -(big + row as i64),
                2 => 0xFFFF_FFF0i64 - (row as i64 % 8), // 低位贴着 u32 上界
                _ => -(0xFFFF_FFFFi64) + row as i64,
            };
            GradPairFixed {
                grad,
                hess: big + (row as i64 * 7),
            }
        })
        .collect();

    let mut cpu = vec![GradPairFixed::default(); offsets[n_feats] as usize];
    hist::build(&bins, n_rows, n_feats, &offsets, &gpair, None, &mut cpu);
    // 先确认夹具真的会溢出低 32 位,否则这个测试等于没测。
    let overflowed = cpu
        .iter()
        .any(|g| (g.grad as u64) > u32::MAX as u64 || (g.hess as u64) > u32::MAX as u64);
    assert!(overflowed, "夹具没有让低 32 位溢出,进位分支没被覆盖");

    for layout in SharedHistogramLayout::ALL {
        let config = SharedHistogramConfig {
            runtime_log: false,
            resident_blocks: None,
            gpu_memory_budget: None,
            hist_streams: None,
            device_quantize: None,
            hist_nodes_per_batch: None,
            gpu_split_cache: false,
            gpu_math: ferrisboost::types::GpuMath::Exact,
            fast_weights: false,
            n_outputs: 1,
            multi_quantile: false,
            has_categorical: false,
            node_feature_masks: false,
            fast_rank: None,
            threads_per_block: 256,
            shared_memory_budget_bytes: Some(16 * 1024),
            layout,
        };
        let (got, _) = build_histogram_shared_with_config(
            0, &bins, n_rows, n_feats, &offsets, &gpair, None, config,
        )
        .unwrap();
        assert_eq!(got, cpu, "{layout:?} 在低位溢出时必须仍然逐位相同");
    }
}

/// multi-node histogram batching 必须**跨过批容量边界**。
///
/// 深度 5 的一层有 16 个节点,配上 `hist_nodes_per_batch = 4` 就会切成 4 批 ——
/// 这条路径(`todo.chunks(cap)` 走多于一次)和单批路径是不同的代码,
/// 单批测试覆盖不到批与批之间的 `hist_out` 复用和 `blk_ptr` 重建。
///
/// 同时钉住三件事:批容量不改变模型、和 CPU 逐字节相同、
/// 关闭合并(`hist_nodes_per_batch = 1`)与打开合并结果相同。
///
/// ⚠️ 「按规模阈值启用的 fast path 必须有跨过阈值的测试」是这个仓库
/// 反复吃过亏的一条规则,不要因为默认值现在是 16 就删掉这个用例。
#[test]
fn gpu_multi_node_batching_crosses_the_batch_boundary() {
    let n_rows = 4000usize;
    let n_feats = 6usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 11 + feat * 17) % 41) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 7) % 31 < 14 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 3);

    let build = |hist_nodes_per_batch: Option<usize>| {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 2,
            // depth 5 => 最深一层 16 个节点,cap=4 时一定要分 4 批。
            max_depth: 5,
            nthread: 1,
            hist_nodes_per_batch,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        serde_json::to_vec(&train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap())
            .unwrap()
    };

    let cpu_params = ferrisboost::types::TrainParams {
        n_rounds: 2,
        max_depth: 5,
        nthread: 1,
        ..Default::default()
    };
    let cpu_cfg = TrainConfig::new(cpu_params, ferrisboost::types::Objective::Logistic);
    let cpu = serde_json::to_vec(&train(&source, &labels, &[], &cpu_cfg, &mut [], &Local).unwrap())
        .unwrap();

    let off = build(Some(1));
    let split = build(Some(4)); // 16 个节点 / 4 = 4 批,跨边界
    let single = build(Some(16)); // 一批装下

    assert_eq!(off, cpu, "关闭合并时必须和 CPU 逐字节相同");
    assert_eq!(split, cpu, "分成多批时必须和 CPU 逐字节相同");
    assert_eq!(single, cpu, "单批时必须和 CPU 逐字节相同");
}

/// `gpu_math = "fast"` 的契约:**它不承诺和 CPU 逐字节相同**,所以不能用
/// 逐字节断言。但「不保证位精确」不等于「什么都不保证」,这里钉住三件事:
///
/// 1. **同一档内确定性**:两次 fast 运行必须逐字节相同。丢掉这一条就没法
///    做 A/B,也没法复现 bug。
/// 2. **树结构不变**:HIGGS 上实测 20 棵树的 split feature 和 split
///    condition 与 exact 完全一致 —— `expf` 的 4 ULP 差异没有翻转任何一次
///    分裂决策。这里在小夹具上钉住同样的性质。
/// 3. **叶子值只在末位漂移**:实测最大相对差 9.8e-6。
///
/// ⚠️ 第 2 条在别的数据上**可能**不成立(接近并列的分裂会翻转),所以断言
/// 写在这个固定夹具上,不是当成普遍定律。
#[test]
fn gpu_math_fast_keeps_structure_and_stays_deterministic() {
    let n_rows = 3000usize;
    let n_feats = 6usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 13 + feat * 29) % 37) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 17) % 23 < 11 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 3);

    let run = |math: ferrisboost::types::GpuMath| {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 4,
            nthread: 1,
            gpu_math: math,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap()
    };

    let exact = run(ferrisboost::types::GpuMath::Exact);
    let fast = run(ferrisboost::types::GpuMath::Fast);
    let fast_again = run(ferrisboost::types::GpuMath::Fast);

    // 1. fast 自身必须确定性。
    assert_eq!(
        serde_json::to_vec(&fast).unwrap(),
        serde_json::to_vec(&fast_again).unwrap(),
        "gpu_math=fast 两次运行必须逐字节相同"
    );

    // 2. 树结构必须一致。
    assert_eq!(exact.trees.len(), fast.trees.len());
    for (te, tf) in exact.trees.iter().zip(fast.trees.iter()) {
        assert_eq!(te.nodes.len(), tf.nodes.len(), "节点数不同");
        for (ne, nf) in te.nodes.iter().zip(tf.nodes.iter()) {
            assert_eq!(ne.is_leaf, nf.is_leaf, "叶子/内点结构不同");
            if !ne.is_leaf {
                assert_eq!(ne.feat, nf.feat, "分裂特征不同");
                assert_eq!(ne.split_cond, nf.split_cond, "分裂阈值不同");
                assert_eq!(ne.default_left, nf.default_left, "缺失方向不同");
                assert_eq!((ne.left, ne.right), (nf.left, nf.right), "子节点连接不同");
            }
        }
    }

    // 3. 叶子值只允许末位漂移。
    for (te, tf) in exact.trees.iter().zip(fast.trees.iter()) {
        for (ne, nf) in te.nodes.iter().zip(tf.nodes.iter()) {
            if ne.is_leaf {
                let d = (ne.leaf_value - nf.leaf_value).abs();
                let rel = d / ne.leaf_value.abs().max(1e-12);
                assert!(
                    rel < 1e-3,
                    "叶子值相对差 {rel:.3e} 太大:exact {} vs fast {}",
                    ne.leaf_value,
                    nf.leaf_value
                );
            }
        }
    }
}

/// A wide physical block must preserve every feature and match the CPU model.
#[test]
fn wide_blocks_keep_all_features_on_gpu() {
    let n_rows = 3000usize;
    // One wide block holds all 64 features.
    let n_feats = 64usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 13) % 43) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 11) % 19 < 9 { 1.0 } else { 0.0 })
        .collect();
    // cols_per_block = 64 => 一个列块就装下全部 64 个特征。
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 64, 3);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 2,
        max_depth: 4,
        nthread: 1,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let gpu = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    assert_eq!(
        serde_json::to_vec(&gpu).unwrap(),
        serde_json::to_vec(&cpu).unwrap(),
        "宽列块下 GPU 必须和 CPU 逐字节相同"
    );
}

/// Every GPU training shape uses column-major blocks, including narrow blocks.
#[test]
fn narrow_blocks_use_the_column_major_layout() {
    for resident_blocks in [Some(0), Some(2)] {
        let config = SharedHistogramConfig {
            resident_blocks,
            hist_streams: Some(1),
            ..Default::default()
        };
        // Eight features per block is also a valid narrow column-major block.
        let ctx = GpuTrainCtx::new(0, 4096, 8, 64, 2, 8, config, false).unwrap();
        let path = ctx.path_summary();
        assert!(path.contains("layout_gpu=col_major"), "{path}");
        assert!(!path.contains("layout_gpu=row_major"), "{path}");
        assert!(!path.contains("rm_stage_per_slot="), "{path}");
    }
}

/// Runtime uses the checked-in PTX rather than compiling the `.cu` source.
/// This fingerprint check catches source/PTX drift that could otherwise make
/// tests exercise an older histogram kernel.
///
/// mtime 在 clean checkout 里没有语义,所以 PTX 生成器嵌入源文件的稳定
/// FNV-1a64;测试比较内容指纹,不再靠 checkout 恰好先写了哪个文件。
#[test]
fn checked_in_ptx_is_not_older_than_its_source() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/backend/cuda");
    let source = std::fs::read(dir.join("histogram.cu")).expect("histogram.cu");
    let mut hash = 0xcbf29ce484222325u64;
    for byte in source {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    let ptx = std::fs::read_to_string(dir.join("histogram.ptx")).expect("histogram.ptx");
    let marker = format!("// FerrisBoost source FNV-1a64: {hash:016x}");
    assert!(
        ptx.lines().any(|line| line == marker),
        "histogram.ptx 不是由当前 histogram.cu 生成的。请运行 scripts/build_ptx.sh"
    );
    assert_eq!(
        ptx.lines().find(|line| line.starts_with(".version ")),
        Some(".version 8.0"),
        "发布 PTX 必须保持在 CUDA 12.0 的 PTX ISA 8.0"
    );
    assert_eq!(
        ptx.lines().find(|line| line.starts_with(".target ")),
        Some(".target sm_75"),
        "发布 PTX 必须继续支持 Turing / RTX 20 系列"
    );
    // PTX 经 include_str! 进入发布二进制:`.file` 不得带构建机的绝对路径
    // (曾经泄漏 CUDA toolkit 的临时解压目录)。
    for line in ptx.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(".file") {
            let path = rest.split('"').nth(1).unwrap_or("");
            assert!(
                !path.starts_with('/') && !path.contains(":\\") && !path.contains(":/"),
                "histogram.ptx 含绝对 .file 路径: {trimmed}。请运行 scripts/build_ptx.sh"
            );
        }
    }
}

/// `colsample_bytree` 的跨后端契约。
///
/// FerrisBoost 的采样序列是自己的(见 `colsample` 模块的说明:XGBoost 自己
/// 的 CPU/GPU 就选不出同一批特征),所以这里钉的**不是**和 XGBoost 逐位相同,
/// 而是 FerrisBoost 自己的承诺:
///
/// 1. `colsample_bytree = 1.0` 与不设它完全一样(不能因为加了这条路径就漂);
/// 2. **CPU 与 GPU 用同一个采样集** —— exact 模式下模型仍然逐字节相同;
/// 3. 同 seed 可复现,换 seed 会变;
/// 4. 换 `cols_per_block` 不改变选中的全局特征 id(所以模型也不变)。
///
/// 第 4 条尤其重要:采样是在**全局特征 id** 上做的,如果误写成块内下标,
/// 换一个 `cols_per_block` 就会静默训出另一个模型。
#[test]
fn colsample_bytree_is_backend_and_blocking_invariant() {
    let n_rows = 3000usize;
    let n_feats = 40usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 13) % 47) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 11) % 17 < 8 { 1.0 } else { 0.0 })
        .collect();

    // ⚠️ 签名是 (data, n_rows, n_features, **max_bins**, **cols_per_block**)。
    // 之前把这两个参数写反,变成在改量化而不是改分块 —— 测试因此"失败",
    // 但那是测试自己的 bug,不是采样的。
    let train_with = |rate: f32, seed: u64, cols_per_block: usize, on_gpu: bool| -> Vec<u8> {
        let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, cols_per_block);
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 4,
            nthread: 1,
            colsample_bytree: rate,
            seed,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        let model = if on_gpu {
            train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        } else {
            train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        };
        serde_json::to_vec(&model).unwrap()
    };

    // 1. rate = 1.0 必须和默认路径完全一致。
    assert_eq!(
        train_with(1.0, 0, 8, false),
        train_with(1.0, 7, 8, false),
        "colsample=1.0 时 seed 不该影响任何东西"
    );

    // 2. CPU 与 GPU 必须用同一个采样集 —— exact 下模型逐字节相同。
    let cpu = train_with(0.3, 42, 8, false);
    let gpu = train_with(0.3, 42, 8, true);
    assert_eq!(cpu, gpu, "CPU 与 GPU 必须使用同一个采样集");

    // 3. 同 seed 可复现;换 seed 要变。
    assert_eq!(cpu, train_with(0.3, 42, 8, false));
    assert_ne!(cpu, train_with(0.3, 43, 8, false), "换 seed 应该换采样集");

    // 4. 换 cols_per_block 不改变全局采样集,因此模型不变。
    assert_eq!(
        cpu,
        train_with(0.3, 42, 5, false),
        "cols_per_block 不得影响选中哪些全局特征"
    );
    assert_eq!(cpu, train_with(0.3, 42, 40, false));

    // 极小比例仍然至少一个特征,并且能正常训练。
    let tiny = train_with(0.001, 42, 8, false);
    assert!(!tiny.is_empty());
    assert_eq!(tiny, train_with(0.001, 42, 13, false));
}

/// **改变显存几何只允许改变时间,不允许改变模型。**
///
/// 三档物理块宽:显式窄、规划器自动解出的、以及一个更宽的合法值。
/// exact 模式下三者必须**逐字节相同** —— 块宽是内存计划的属性,
/// 不是训练语义的一部分。
///
/// 同时钉住:`colsample` 选中的**全局特征 id 不随块宽变化**
/// (采样在全局 id 上做;写成块内下标的话,换块宽就会静默训出另一个模型)。
#[test]
fn streaming_block_width_changes_timing_not_the_model() {
    use ferrisboost::gpu_mem_plan::{resolve_gpu_streaming_cols_per_block, StreamingMemModel};

    let n_rows = 4000usize;
    let n_feats = 32usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 5 + feat * 11) % 53) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 13) % 23 < 11 { 1.0 } else { 0.0 })
        .collect();

    // 规划器在这个形状上会解出什么宽度(用一个宽裕的预算,让它按块数目标走)。
    let model = StreamingMemModel::new(n_rows as u64, 32, 1, 16, true, false);
    let plan = resolve_gpu_streaming_cols_per_block(
        None,
        n_feats as u64,
        4 * 1024 * 1024 * 1024,
        6 * 1024 * 1024 * 1024,
        model,
    )
    .unwrap();
    let auto_w = plan.resolved_cols_per_block as usize;
    assert!(auto_w >= 1 && auto_w <= n_feats, "auto 宽度要合法:{auto_w}");

    let run = |cols_per_block: usize, rate: f32| -> Vec<u8> {
        let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, cols_per_block);
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 4,
            nthread: 1,
            colsample_bytree: rate,
            seed: 9,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        serde_json::to_vec(&train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap())
            .unwrap()
    };

    // 三档宽度:窄 / auto / 更宽。
    let narrow = 4usize;
    let wider = n_feats; // 单块,合法的另一端
    for rate in [1.0f32, 0.5] {
        let a = run(narrow, rate);
        let b = run(auto_w, rate);
        let c = run(wider, rate);
        assert_eq!(
            a, b,
            "colsample={rate}:窄({narrow}) 与 auto({auto_w}) 必须逐字节相同"
        );
        assert_eq!(
            a, c,
            "colsample={rate}:窄({narrow}) 与宽({wider}) 必须逐字节相同"
        );
    }
}

/// `subsample` 的核心不变量与跨后端一致性。
///
/// > **采样决定哪些行「塑造」这棵树,不决定哪些行「收到」这棵树。**
///
/// 钉三件事:
/// 1. `subsample = 1.0` 与不设它**逐字节相同**(不能因为加了这条路径就漂);
/// 2. CPU 与 GPU 在同一个 seed 下选出**同一批行** —— exact 模式下模型逐字节相同。
///    ⚠️ 这条最关键:host 在 f32 gpair 上置零,device 在 gradient kernel 里置零,
///    两条完全不同的代码路径必须选出同一批行(靠同一个整数哈希 + 同一个阈值);
/// 3. **所有行都拿到预测** —— 未选中的行不是"不参与这棵树"。
///    如果采样掩码被误当成预测掩码,预测里会出现一批没被这棵树更新过的行,
///    模型质量会塌,而且不会有任何报错。
#[test]
fn subsample_shapes_the_tree_without_masking_predictions() {
    let n_rows = 4000usize;
    let n_feats = 12usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 13) % 41) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 11) % 19 < 9 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 4);

    let run = |rate: f32, on_gpu: bool| {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 4,
            nthread: 1,
            subsample: rate,
            seed: 21,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        if on_gpu {
            train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        } else {
            train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        }
    };
    let bytes = |m: &ferrisboost::tree::Model| serde_json::to_vec(m).unwrap();

    // 1. rate = 1.0 必须和默认路径完全一致。
    let base = run(1.0, false);
    let default_params = ferrisboost::types::TrainParams {
        n_rounds: 3,
        max_depth: 4,
        nthread: 1,
        seed: 21,
        ..Default::default()
    };
    let default_cfg = TrainConfig::new(default_params, ferrisboost::types::Objective::Logistic);
    let untouched = train(&source, &labels, &[], &default_cfg, &mut [], &Local).unwrap();
    assert_eq!(
        bytes(&base),
        bytes(&untouched),
        "subsample=1.0 必须是原来的路径"
    );

    for rate in [0.5f32, 0.25] {
        // 2. CPU 与 GPU 必须选出同一批行 → 模型逐字节相同。
        let cpu = run(rate, false);
        let gpu = run(rate, true);
        assert_eq!(
            bytes(&cpu),
            bytes(&gpu),
            "subsample={rate}:CPU 与 GPU 必须选出同一批行"
        );

        // 3. 采样必须真的改变了树(否则上面两条都是空的)。
        assert_ne!(
            bytes(&cpu),
            bytes(&base),
            "subsample={rate} 应该训出和全量不同的树"
        );

        // 4. **所有行都收到预测**:没有任何一行落在"这棵树没更新过"的状态。
        //    用一个恒等检查:预测必须全部有限,且不是全体相同的常数
        //    (常数意味着树退化 / 大批行没被更新)。
        let preds: Vec<f32> = (0..n_rows)
            .map(|r| cpu.predict(&data[r * n_feats..(r + 1) * n_feats]))
            .collect();
        assert_eq!(preds.len(), n_rows, "每一行都必须有预测");
        assert!(preds.iter().all(|p| p.is_finite()), "预测不能有 NaN/Inf");
        let first = preds[0];
        assert!(
            preds.iter().any(|p| (p - first).abs() > 1e-9),
            "subsample={rate}:预测不该是常数 —— 那说明大批行没拿到这棵树的更新"
        );
    }
}

/// Uneven physical block widths must not change the trained model when the
/// final, narrower block is resident.
#[test]
fn resident_narrow_last_block_keeps_the_model_bit_exact() {
    let n_rows = 3000usize;
    // 30 不能被 16 整除:块宽 16 → [16, 14],最后一块更窄。
    let n_feats = 30usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 13) % 61) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 17) % 29 < 14 { 1.0 } else { 0.0 })
        .collect();

    let run = |cols_per_block: usize, resident: usize| -> Vec<u8> {
        let source = DenseSource::from_row_major(&data, n_rows, n_feats, 64, cols_per_block);
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 5,
            nthread: 1,
            resident_blocks: Some(resident),
            seed: 4,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        serde_json::to_vec(&train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap())
            .unwrap()
    };

    // 16 → [16, 14]:第 2 块比第 1 块窄。
    let streaming = run(16, 0);
    let partial = run(16, 1); // 只常驻等宽的第一块
    let full = run(16, 2); // 连那块更窄的也常驻 —— 曾经在这里读偏
    assert_eq!(streaming, partial, "部分常驻不该改变模型");
    assert_eq!(
        streaming, full,
        "the narrower final resident block must match pure streaming"
    );

    // Equal-width blocks provide a second residency-invariance case.
    let even_streaming = run(15, 0); // 30 / 15 = [15, 15]
    let even_full = run(15, 2);
    assert_eq!(even_streaming, even_full, "等宽切分下全常驻本来就该一致");
}

/// The resident split cache is an exact, bounded alternative to the existing
/// host split path. Cover wide and narrow column-major blocks and both objectives with
/// missing values, row and column sampling, tree resets, and depth beyond the
/// two-generation cache capacity. The 40-column first block always selects the
/// column-major path; the 16+14 split makes the final resident block narrower
/// than the first.
#[test]
fn gpu_split_cache_matches_cpu_for_block_shapes_and_objectives() {
    let n_rows = 768usize;
    let fixtures = [(70usize, 40usize), (30usize, 16usize)];
    let bytes = |m: &ferrisboost::tree::Model| serde_json::to_vec(m).unwrap();

    let run = |source: &DenseSource,
               labels: &[f32],
               objective: ferrisboost::types::Objective,
               gpu: bool,
               cache: bool,
               hist_nodes_per_batch: Option<usize>,
               subsample: f32,
               colsample_bytree: f32|
     -> ferrisboost::tree::Model {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 8,
            nthread: 1,
            subsample,
            colsample_bytree,
            seed: 73,
            hist_nodes_per_batch,
            resident_blocks: Some(2),
            gpu_split_cache: cache,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, objective);
        if gpu {
            train_gpu(0, source, labels, &[], &cfg, &mut [], &Local).unwrap()
        } else {
            train(source, labels, &[], &cfg, &mut [], &Local).unwrap()
        }
    };

    for (n_feats, cols_per_block) in fixtures {
        let data: Vec<f32> = (0..n_rows)
            .flat_map(|row| {
                (0..n_feats).map(move |feat| {
                    if (row * 17 + feat * 29) % 89 == 0 {
                        f32::NAN
                    } else {
                        ((row * 13 + feat * 31 + row * feat) % 251) as f32 / 251.0
                    }
                })
            })
            .collect();
        let source = DenseSource::from_row_major(&data, n_rows, n_feats, 255, cols_per_block);
        let objectives = [
            (
                ferrisboost::types::Objective::Logistic,
                (0..n_rows)
                    .map(|row| {
                        if (row * 11 + row / 7) % 23 < 11 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<f32>>(),
            ),
            (
                ferrisboost::types::Objective::SquaredError,
                (0..n_rows)
                    .map(|row| ((row * 7) % 101) as f32 / 100.0)
                    .collect::<Vec<f32>>(),
            ),
        ];

        for (objective, labels) in objectives {
            let cpu = run(&source, &labels, objective, false, false, Some(2), 0.6, 0.6);
            let baseline_gpu = run(&source, &labels, objective, true, false, Some(2), 0.6, 0.6);
            let cached_gpu = run(&source, &labels, objective, true, true, Some(2), 0.6, 0.6);
            assert_eq!(
                bytes(&baseline_gpu),
                bytes(&cpu),
                "uncached GPU mismatch: {n_feats} features, {objective:?}"
            );
            assert_eq!(
                bytes(&cached_gpu),
                bytes(&cpu),
                "cached GPU mismatch: {n_feats} features, {objective:?}"
            );
            assert_eq!(
                bytes(&cached_gpu),
                bytes(&baseline_gpu),
                "cache changed model: {n_feats} features, {objective:?}"
            );

            // None selects the production default cap (16). At least two
            // nodes are eligible together in this fixed, grown tree shape.
            for (subsample, colsample_bytree) in [(0.6, 0.6), (1.0, 1.0)] {
                let cpu = run(
                    &source,
                    &labels,
                    objective,
                    false,
                    false,
                    None,
                    subsample,
                    colsample_bytree,
                );
                let baseline_gpu = run(
                    &source,
                    &labels,
                    objective,
                    true,
                    false,
                    None,
                    subsample,
                    colsample_bytree,
                );
                let cached_gpu = run(
                    &source,
                    &labels,
                    objective,
                    true,
                    true,
                    None,
                    subsample,
                    colsample_bytree,
                );
                assert!(max_nonleaf_width(&cached_gpu) > 1,
                    "default cap must exercise multi-node levels: features={n_feats}, {objective:?}, sampling=({subsample}, {colsample_bytree})");
                assert_eq!(
                    bytes(&baseline_gpu),
                    bytes(&cpu),
                    "uncached default-cap GPU mismatch: {n_feats} features, {objective:?}"
                );
                assert_eq!(
                    bytes(&cached_gpu),
                    bytes(&cpu),
                    "cached default-cap GPU mismatch: {n_feats} features, {objective:?}"
                );
                assert_eq!(
                    bytes(&cached_gpu),
                    bytes(&baseline_gpu),
                    "default-cap cache changed model: {n_feats} features, {objective:?}"
                );
            }
        }
    }
}

fn max_nonleaf_width(model: &ferrisboost::tree::Model) -> usize {
    model
        .trees
        .iter()
        .map(|tree| {
            let mut level = vec![0usize];
            let mut max_width = 0;
            while !level.is_empty() {
                let mut next = Vec::new();
                let mut nonleaf_width = 0;
                for index in level {
                    let node = &tree.nodes[index];
                    if !node.is_leaf {
                        nonleaf_width += 1;
                        next.push(node.left as usize);
                        next.push(node.right as usize);
                    }
                }
                max_width = max_width.max(nonleaf_width);
                level = next;
            }
            max_width
        })
        .max()
        .unwrap_or(0)
}

/// Capacity pressure, residency choices, and host fallbacks must preserve the
/// same exact model. Depth 8 grows past the cache's two-node capacity; cap 1
/// disables cache storage, and zero residency exercises the streaming path.
// Preserve the deterministic hash mask used to seed missing values.
#[allow(clippy::manual_is_multiple_of)]
#[test]
fn gpu_split_cache_capacity_and_residency_fallbacks_are_exact() {
    let n_rows = 768usize;
    let n_feats = 30usize;
    let cols_per_block = 16usize; // [16, 14], so full residency includes the narrow tail.
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 19 + feat * 23) % 83 == 0 {
                    f32::NAN
                } else {
                    ((row * 11 + feat * 37 + row * feat) % 251) as f32 / 251.0
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| {
            if (row * 5 + row / 9) % 29 < 14 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 255, cols_per_block);
    let bytes = |m: &ferrisboost::tree::Model| serde_json::to_vec(m).unwrap();

    let run = |cache: bool,
               resident: usize,
               cap: Option<usize>,
               nthread: usize,
               gpu_math: ferrisboost::types::GpuMath|
     -> ferrisboost::tree::Model {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 8,
            nthread,
            subsample: 0.6,
            colsample_bytree: 0.6,
            seed: 91,
            hist_nodes_per_batch: cap,
            resident_blocks: Some(resident),
            gpu_split_cache: cache,
            gpu_math,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap()
    };

    let exact = ferrisboost::types::GpuMath::Exact;
    let baseline_model = run(false, 2, Some(2), 1, exact);
    assert!(max_nonleaf_width(&baseline_model) > 2,
        "fallback fixture must have more than two nonleaf parents at one depth, creating four or more children");
    let baseline = bytes(&baseline_model);
    let no_resident = bytes(&run(true, 0, Some(2), 1, exact));
    let partial_resident = bytes(&run(true, 1, Some(2), 1, exact));
    let full_resident = bytes(&run(true, 2, Some(2), 1, exact));
    let cap_one_fallback = bytes(&run(true, 2, Some(1), 1, exact));
    let four_threads = bytes(&run(true, 2, Some(2), 4, exact));
    assert_eq!(
        no_resident, baseline,
        "cache enabled with no resident blocks changed the model"
    );
    assert_eq!(
        partial_resident, baseline,
        "partial residency changed the model"
    );
    assert_eq!(
        full_resident, baseline,
        "full residency or capacity overflow changed the model"
    );
    assert_eq!(
        cap_one_fallback,
        bytes(&run(false, 2, Some(1), 1, exact)),
        "capacity-one fallback changed the model"
    );
    assert_eq!(four_threads, baseline, "nthread=4 changed the cached model");

    // Exercise overflow of the production default (16-node) cache with every
    // row and column eligible. This tree shape must produce a level wider than
    // the cache; beyond that capacity the existing host split path remains the
    // exact fallback.
    let full_n_rows = 4_096usize;
    let full_n_feats = 30usize;
    let full_data: Vec<f32> = (0..full_n_rows)
        .flat_map(|row| {
            (0..full_n_feats).map(move |feat| {
                let mut hash = (row as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add((feat as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
                hash ^= hash >> 30;
                hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                hash ^= hash >> 27;
                hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
                hash ^= hash >> 31;
                if hash % 83 == 0 {
                    f32::NAN
                } else {
                    ((hash >> 32) % 251) as f32
                }
            })
        })
        .collect();
    let full_labels: Vec<f32> = (0..full_n_rows)
        .map(|row| {
            let mut hash = (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            hash ^= hash >> 30;
            hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            hash ^= hash >> 27;
            hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
            hash ^= hash >> 31;
            (hash & 1) as f32
        })
        .collect();
    let full_source = DenseSource::from_row_major(&full_data, full_n_rows, full_n_feats, 255, 16);
    let full_sample = |cache: bool| {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 8,
            min_child_weight: 1.0,
            nthread: 1,
            subsample: 1.0,
            colsample_bytree: 1.0,
            seed: 147,
            resident_blocks: Some(2),
            gpu_split_cache: cache,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        train_gpu(0, &full_source, &full_labels, &[], &cfg, &mut [], &Local).unwrap()
    };
    let default_cap_baseline = full_sample(false);
    let default_cap_cached = full_sample(true);
    assert!(
        max_nonleaf_width(&default_cap_cached) > 16,
        "full-sample fallback fixture must exceed the default 16-node cache"
    );
    assert_eq!(
        bytes(&default_cap_cached),
        bytes(&default_cap_baseline),
        "default-capacity overflow fallback changed the full-sample model"
    );

    let fast = ferrisboost::types::GpuMath::Fast;
    let fast_baseline = bytes(&run(false, 2, None, 1, fast));
    let fast_cached = bytes(&run(true, 2, None, 1, fast));
    let fast_cached_repeat = bytes(&run(true, 2, None, 1, fast));
    assert_eq!(
        fast_cached, fast_baseline,
        "fast cached path changed the baseline model"
    );
    assert_eq!(
        fast_cached_repeat, fast_cached,
        "fast cached path must remain deterministic"
    );
}

/// Enabling the resident split cache by default must preserve the established
/// uncached model for the same deterministic sampled training fixture.
#[test]
fn gpu_split_cache_default_preserves_exact_and_fast_models() {
    use ferrisboost::types::{GpuMath, Objective, TrainParams};

    assert!(
        TrainParams::default().gpu_split_cache,
        "the untouched training parameter default must enable the split cache"
    );

    let n_rows = 768usize;
    let n_feats = 30usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 19 + feat * 23) % 83 == 0 {
                    f32::NAN
                } else {
                    ((row * 11 + feat * 37 + row * feat) % 251) as f32 / 251.0
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| {
            if (row * 5 + row / 9) % 29 < 14 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 255, 16);
    assert_eq!(
        source.n_blocks(),
        2,
        "fixture should use physical blocks [16, 14]"
    );

    let run = |gpu: bool, gpu_math: GpuMath, cache: Option<bool>| {
        let mut params = TrainParams {
            n_rounds: 3,
            max_depth: 8,
            nthread: 1,
            subsample: 0.6,
            colsample_bytree: 0.6,
            seed: 91,
            resident_blocks: Some(2),
            gpu_math,
            ..Default::default()
        };
        if let Some(enabled) = cache {
            params.gpu_split_cache = enabled;
        }
        let cfg = TrainConfig::new(params, Objective::Logistic);
        if gpu {
            train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        } else {
            train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        }
    };
    let bytes = |model: &ferrisboost::tree::Model| serde_json::to_vec(model).unwrap();

    let exact_default = run(true, GpuMath::Exact, None);
    let exact_uncached_gpu = run(true, GpuMath::Exact, Some(false));
    let exact_cpu = run(false, GpuMath::Exact, None);
    assert_eq!(bytes(&exact_default), bytes(&exact_uncached_gpu));
    assert_eq!(bytes(&exact_default), bytes(&exact_cpu));

    let fast_default = bytes(&run(true, GpuMath::Fast, None));
    let fast_default_repeat = bytes(&run(true, GpuMath::Fast, None));
    let fast_uncached = bytes(&run(true, GpuMath::Fast, Some(false)));
    assert_eq!(fast_default, fast_uncached);
    assert_eq!(
        fast_default, fast_default_repeat,
        "fast default path must remain deterministic"
    );
}

/// Cache accounting follows the allocations actually enabled by a resident
/// slot. It stays zero when streaming has no resident cache or when cap=1
/// disables the feature, and includes both parent generations and scratch at
/// cap=2.
#[test]
fn gpu_split_cache_bytes_match_allocations_and_activation_rules() {
    use ferrisboost::backend::cuda::SharedHistogramConfig;

    let n_rows = 512usize;
    let max_feats = 12usize;
    let max_hist_bins = 12 * 16;
    let n_blocks = 2usize;
    let make_ctx = |cache: bool, resident: usize, cap: usize| {
        GpuTrainCtx::new(
            0,
            n_rows,
            max_feats,
            max_hist_bins,
            n_blocks,
            128,
            SharedHistogramConfig {
                runtime_log: false,
                resident_blocks: Some(resident),
                hist_nodes_per_batch: Some(cap),
                gpu_split_cache: cache,
                ..Default::default()
            },
            false,
        )
        .unwrap()
    };

    let make_default_ctx = |resident: usize, cap: usize| {
        GpuTrainCtx::new(
            0,
            n_rows,
            max_feats,
            max_hist_bins,
            n_blocks,
            128,
            SharedHistogramConfig {
                runtime_log: false,
                resident_blocks: Some(resident),
                hist_nodes_per_batch: Some(cap),
                ..Default::default()
            },
            false,
        )
        .unwrap()
    };

    assert_eq!(make_ctx(true, 0, 2).budget().split_cache_bytes, 0);
    assert_eq!(make_ctx(true, 2, 1).budget().split_cache_bytes, 0);
    assert!(SharedHistogramConfig::default().gpu_split_cache);
    assert_eq!(make_default_ctx(0, 2).budget().split_cache_bytes, 0);
    assert_eq!(make_default_ctx(2, 1).budget().split_cache_bytes, 0);
    let expected_per_slot =
        // per feature: candidates 64/node + feats/enabled/monotone 12;
        // per node: sums 16 + metadata 16 + bounds 8, plus the packet window.
        max_hist_bins * 16 * 2 * 2 + max_feats * (2 * 64 + 12) + 2 * (40 + 64 * 2);
    let bucket_profile = matches!(
        std::env::var("FB_HIST_BUCKET_PROFILE").as_deref(),
        Ok("1") | Ok("true")
    );
    if bucket_profile {
        assert_eq!(
            make_ctx(true, 2, 2).budget().split_cache_bytes,
            0,
            "bucket profiling deliberately disables the split cache to isolate histogram buckets"
        );
        assert_eq!(make_default_ctx(2, 2).budget().split_cache_bytes, 0);
    } else {
        assert_eq!(
            make_ctx(true, 2, 2).budget().split_cache_bytes,
            2 * expected_per_slot,
            "budget must include two histogram generations, feature candidates, packets, sums, metadata, monotone bounds, and feature masks"
        );
        assert_eq!(
            make_default_ctx(2, 2).budget().split_cache_bytes,
            2 * expected_per_slot,
            "default config budget must account for the same enabled split-cache allocations"
        );
    }
}

/// 样本权重的跨后端契约:`gpu_math="exact"` 下 CPU 与 GPU 模型逐字节相同,
/// 覆盖 resident / streaming、host / device 量化、split cache 开关、行采样、
/// 列采样和两个 objective。权重含 0、小数和大于 1 的值。
#[test]
fn weighted_training_is_cpu_gpu_byte_identical() {
    let n_rows = 1_037usize;
    let n_feats = 13usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 7 + feat) % 17 == 0 {
                    f32::NAN
                } else {
                    ((row * 11 + feat * 5) % 41) as f32
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 19) % 37 < 15 { 1.0 } else { 0.0 })
        .collect();
    let weights: Vec<f32> = (0..n_rows)
        .map(|row| [0.0, 0.25, 1.0, 2.5, 1.75, 4.0][(row * 5) % 6])
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 4);
    for objective in [
        ferrisboost::types::Objective::Logistic,
        ferrisboost::types::Objective::SquaredError,
    ] {
        for (resident_blocks, device_quantize, split_cache, subsample, colsample) in [
            (None, None, true, 1.0, 1.0),
            (Some(0), None, true, 1.0, 1.0),
            (None, Some(false), true, 0.7, 1.0),
            (None, Some(true), false, 0.7, 0.6),
            (Some(0), Some(true), true, 0.8, 0.6),
        ] {
            let params = ferrisboost::types::TrainParams {
                n_rounds: 3,
                max_depth: 4,
                max_bin: 32,
                cols_per_block: 4,
                nthread: 1,
                subsample,
                colsample_bytree: colsample,
                seed: 11,
                resident_blocks,
                device_quantize,
                gpu_split_cache: split_cache,
                ..Default::default()
            };
            let cfg = TrainConfig::new(params, objective);
            let what = format!(
                "{objective:?} resident={resident_blocks:?} dq={device_quantize:?} \
                 cache={split_cache} subsample={subsample} colsample={colsample}"
            );
            let cpu = train_weighted(&source, &labels, Some(&weights), &[], &cfg, &mut [], &Local)
                .unwrap();
            let gpu = train_gpu_weighted(
                0,
                &source,
                &labels,
                Some(&weights),
                &[],
                &cfg,
                &mut [],
                &Local,
            )
            .unwrap();
            assert!(
                cpu.trees.iter().any(|t| t.nodes.len() > 1),
                "{what}: 夹具应该真的分裂"
            );
            assert_eq!(
                gpu.to_xgboost_json().unwrap(),
                cpu.to_xgboost_json().unwrap(),
                "{what}: 加权模型必须 CPU/GPU 逐字节相同"
            );
            let unweighted = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
            assert_ne!(
                unweighted.to_xgboost_json().unwrap(),
                cpu.to_xgboost_json().unwrap(),
                "{what}: 权重必须真的生效"
            );
        }
    }
}

/// all-ones 权重在 GPU 上也必须与不加权逐字节相同(含 fast 之外的默认路径)。
#[test]
fn gpu_all_ones_weights_match_unweighted() {
    let n_rows = 600usize;
    let n_feats = 6usize;
    let data: Vec<f32> = (0..n_rows * n_feats)
        .map(|i| ((i * 31) % 97) as f32)
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|r| ((r * 13) % 7 < 3) as u8 as f32)
        .collect();
    let ones = vec![1.0f32; n_rows];
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 3);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 3,
        max_depth: 3,
        max_bin: 32,
        cols_per_block: 3,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let plain = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let ones_model =
        train_gpu_weighted(0, &source, &labels, Some(&ones), &[], &cfg, &mut [], &Local).unwrap();
    assert_eq!(
        plain.to_xgboost_json().unwrap(),
        ones_model.to_xgboost_json().unwrap()
    );
}

/// `gpu_math="fast"` + 样本权重:权重常驻 device,由 objective kernel 乘进
/// grad/hess。fast 不承诺 CPU/GPU 逐字节相同,但必须:全 1 权重 ≡ 不加权、
/// 重复运行逐字节相同、权重真的生效。resident 与 streaming 两条路径都覆盖。
#[test]
fn fast_math_weights_are_deterministic_and_effective() {
    let n_rows = 1_037usize;
    let n_feats = 13usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 7 + feat) % 17 == 0 {
                    f32::NAN
                } else {
                    ((row * 11 + feat * 5) % 41) as f32
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 19) % 37 < 15 { 1.0 } else { 0.0 })
        .collect();
    let weights: Vec<f32> = (0..n_rows)
        .map(|row| [0.0, 0.25, 1.0, 2.5, 1.75, 4.0][(row * 5) % 6])
        .collect();
    let ones = vec![1.0f32; n_rows];
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 4);
    for objective in [
        ferrisboost::types::Objective::Logistic,
        ferrisboost::types::Objective::SquaredError,
    ] {
        for (resident_blocks, subsample) in [(None, 1.0), (Some(0), 0.7)] {
            let params = ferrisboost::types::TrainParams {
                n_rounds: 3,
                max_depth: 4,
                max_bin: 32,
                cols_per_block: 4,
                subsample,
                seed: 5,
                resident_blocks,
                gpu_math: ferrisboost::types::GpuMath::Fast,
                ..Default::default()
            };
            let cfg = TrainConfig::new(params, objective);
            let what = format!("{objective:?} resident={resident_blocks:?} subsample={subsample}");
            let run = |w: Option<&[f32]>| {
                train_gpu_weighted(0, &source, &labels, w, &[], &cfg, &mut [], &Local)
                    .unwrap()
                    .to_xgboost_json()
                    .unwrap()
            };
            let plain = run(None);
            assert_eq!(plain, run(Some(&ones)), "{what}: 全 1 权重必须等于不加权");
            let a = run(Some(&weights));
            assert_eq!(a, run(Some(&weights)), "{what}: fast 加权必须逐次确定");
            assert_ne!(a, plain, "{what}: 权重必须真的生效");
        }
    }
}

/// 单调约束:CPU 与 GPU(exact)模型逐字节相同,覆盖 GPU 缓存 split kernel
/// (`hist_nodes_per_batch = 16`)、host 回退(缓存关、streaming)、采样和权重。
#[test]
fn monotone_training_is_cpu_gpu_byte_identical() {
    let n_rows = 1_211usize;
    let n_feats = 9usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 5 + feat) % 23 == 0 {
                    f32::NAN
                } else {
                    ((row * 13 + feat * 7) % 37) as f32
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| {
            let x0 = ((row * 13) % 37) as f32;
            let x1 = ((row * 13 + 7) % 37) as f32;
            if x0 - 0.6 * x1 + ((row * 31) % 11) as f32 > 12.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let weights: Vec<f32> = (0..n_rows)
        .map(|row| [0.5, 1.0, 2.0, 0.0, 3.0][(row * 3) % 5])
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 3);
    let constraints = vec![1i8, -1, 0, 1, 0, 0, -1, 0, 1];
    for objective in [
        ferrisboost::types::Objective::Logistic,
        ferrisboost::types::Objective::SquaredError,
    ] {
        for (resident_blocks, split_cache, nodes, subsample, colsample, weighted) in [
            (None, true, Some(16), 1.0, 1.0, false),
            (None, true, Some(16), 0.8, 0.7, true),
            (Some(0), true, Some(16), 1.0, 1.0, true),
            (None, false, Some(16), 1.0, 1.0, false),
            (None, true, Some(2), 1.0, 0.7, false),
            (None, true, Some(1), 0.8, 1.0, true),
        ] {
            let params = ferrisboost::types::TrainParams {
                n_rounds: 4,
                max_depth: 6,
                max_bin: 32,
                cols_per_block: 3,
                nthread: 1,
                subsample,
                colsample_bytree: colsample,
                seed: 5,
                resident_blocks,
                hist_nodes_per_batch: nodes,
                gpu_split_cache: split_cache,
                monotone_constraints: constraints.clone(),
                ..Default::default()
            };
            let cfg = TrainConfig::new(params.clone(), objective);
            let w = weighted.then_some(weights.as_slice());
            let what = format!(
                "{objective:?} resident={resident_blocks:?} cache={split_cache} nodes={nodes:?} \
                 subsample={subsample} colsample={colsample} weighted={weighted}"
            );
            let cpu = train_weighted(&source, &labels, w, &[], &cfg, &mut [], &Local).unwrap();
            let gpu =
                train_gpu_weighted(0, &source, &labels, w, &[], &cfg, &mut [], &Local).unwrap();
            assert!(
                cpu.trees.iter().any(|t| t.nodes.len() > 3),
                "{what}: 夹具应该长出多层"
            );
            assert_eq!(
                gpu.to_xgboost_json().unwrap(),
                cpu.to_xgboost_json().unwrap(),
                "{what}: 带约束模型必须 CPU/GPU 逐字节相同"
            );
            let free = TrainConfig::new(
                ferrisboost::types::TrainParams {
                    monotone_constraints: Vec::new(),
                    ..params
                },
                objective,
            );
            let unconstrained =
                train_weighted(&source, &labels, w, &[], &free, &mut [], &Local).unwrap();
            assert_ne!(
                unconstrained.to_xgboost_json().unwrap(),
                cpu.to_xgboost_json().unwrap(),
                "{what}: 约束应该真的改变模型"
            );
        }
    }
}

// ------------------------------------------------------------ 多分类 fast

fn multiclass_fixture(
    n_rows: usize,
    n_feats: usize,
    k: usize,
) -> (DenseSource, Vec<f32>, Vec<f32>) {
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 7 + feat) % 29 == 0 {
                    f32::NAN
                } else {
                    ((row * 13 + feat * 11) % 41) as f32
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| {
            let s = ((row * 13) % 41) + ((row * 5) % 7);
            (s * k / 48).min(k - 1) as f32
        })
        .collect();
    let weights: Vec<f32> = (0..n_rows)
        .map(|row| [0.5, 1.0, 2.0, 0.25, 3.0][(row * 3) % 5])
        .collect();
    (
        DenseSource::from_row_major(&data, n_rows, n_feats, 32, 3),
        labels,
        weights,
    )
}

/// 多分类 fast 的梯度 kernel 对拍 host `softmax_gradients`:同一份初始 margin 下,
/// 每个类的量化 scale 必须相同(pow2,对 ULP 级差异不敏感),root sum 只差
/// 量化舍入。同时钉住行采样(键 = 树序号)与 host 选同一批行。
#[test]
fn gpu_multiclass_fast_gradient_matches_host_softmax() {
    use ferrisboost::types::{GradPair, GradQuantizer, Objective};
    let k = 5usize;
    let n_rows = 3000usize;
    let (_, labels, weights) = multiclass_fixture(n_rows, 4, k);
    let obj = Objective::Softprob {
        num_class: k as u32,
    };
    let margins: Vec<f32> = (0..k).map(|c| 0.3 * c as f32 - 0.7).collect();
    let config = SharedHistogramConfig {
        runtime_log: false,
        gpu_math: ferrisboost::types::GpuMath::Fast,
        fast_weights: true,
        n_outputs: k,
        fast_rank: None,
        ..Default::default()
    };
    let ctx = GpuTrainCtx::new(0, n_rows, 4, 19, 2, 8, config, false).unwrap();
    ctx.init_fast_math(&labels, Some(&weights), &margins)
        .unwrap();
    // 初始化后 device prediction 是 class-major 的 K 份 base margin。
    let mut dev = vec![0.0f32; n_rows * k];
    ctx.sync_prediction_to_host(&mut dev).unwrap();
    for c in 0..k {
        assert!(dev[c * n_rows..(c + 1) * n_rows]
            .iter()
            .all(|&v| v == margins[c]));
    }

    let class_preds: Vec<Vec<f32>> = margins.iter().map(|&m| vec![m; n_rows]).collect();
    let mut host: Vec<Vec<GradPair>> = vec![Vec::new(); k];
    obj.softmax_gradients(&class_preds, &labels, Some(&weights), &mut host);

    ctx.fast_multiclass_begin_round().unwrap();
    for (rate, class) in [(1.0f32, 0usize), (1.0, 3), (0.6, 2)] {
        let tree = (7 * k + class) as u64;
        let threshold = ferrisboost::subsample::threshold_for(rate);
        let mut col = host[class].clone();
        if rate < 1.0 {
            for (i, g) in col.iter_mut().enumerate() {
                if !ferrisboost::subsample::row_selected(i as u64, threshold, 11, tree) {
                    *g = GradPair {
                        grad: 0.0,
                        hess: 0.0,
                    };
                }
            }
        }
        let hq = GradQuantizer::new(&col, n_rows);
        let host_sum = col.iter().fold(GradPairFixed::default(), |mut acc, &g| {
            let f = hq.to_fixed(g);
            acc.grad += f.grad;
            acc.hess += f.hess;
            acc
        });
        let (dq, dsum, _) = ctx
            .fast_multiclass_gradient_and_quantize(class, threshold, 11, tree)
            .unwrap();
        assert_eq!(dq.grad_scale(), hq.grad_scale(), "class {class} grad scale");
        assert_eq!(dq.hess_scale(), hq.hess_scale(), "class {class} hess scale");
        let tol = n_rows as i64; // 每行最多差一个定点单位
        assert!(
            (dsum.grad - host_sum.grad).abs() <= tol && (dsum.hess - host_sum.hess).abs() <= tol,
            "class {class} rate {rate}: device {dsum:?} vs host {host_sum:?}"
        );
    }
    // 越界类别明确报错,单输出入口拒绝多分类。
    assert!(ctx
        .fast_multiclass_gradient_and_quantize(k, u64::MAX, 0, 0)
        .is_err());
    assert!(ctx
        .fast_gradient_and_quantize(Objective::Logistic, u64::MAX, 0, 0)
        .is_err());
}

/// fast 量化尾巴的 device scale(ilogb + pow2)在各种量级的梯度上必须和 host
/// `GradQuantizer` 逐位相同,且不触发「退回 host scale 重量化」。样本权重按
/// 2 的幂与非 2 的幂缩放,把 max|g| / max|h| 扫过很多个数量级。
#[test]
fn gpu_fast_device_scale_matches_host_quantizer_across_magnitudes() {
    use ferrisboost::types::{GradPair, GradQuantizer, Objective};
    let k = 3usize;
    let n_rows = 2500usize;
    let (_, labels, base_weights) = multiclass_fixture(n_rows, 4, k);
    let obj = Objective::Softprob {
        num_class: k as u32,
    };
    let margins: Vec<f32> = (0..k).map(|c| 0.2 * c as f32 - 0.3).collect();
    for factor in [1.0e-30f32, 3.7e-7, 0.5, 1.0, 1.37, 123.4, 6.0e8, 1.0e20] {
        let weights: Vec<f32> = base_weights.iter().map(|&w| w * factor).collect();
        let config = SharedHistogramConfig {
            runtime_log: false,
            gpu_math: ferrisboost::types::GpuMath::Fast,
            fast_weights: true,
            n_outputs: k,
            fast_rank: None,
            ..Default::default()
        };
        let ctx = GpuTrainCtx::new(0, n_rows, 4, 19, 2, 8, config, false).unwrap();
        ctx.init_fast_math(&labels, Some(&weights), &margins)
            .unwrap();
        let class_preds: Vec<Vec<f32>> = margins.iter().map(|&m| vec![m; n_rows]).collect();
        let mut host: Vec<Vec<GradPair>> = vec![Vec::new(); k];
        obj.softmax_gradients(&class_preds, &labels, Some(&weights), &mut host);
        ctx.fast_multiclass_begin_round().unwrap();
        for class in 0..k {
            let hq = GradQuantizer::new(&host[class], n_rows);
            let (dq, _, _) = ctx
                .fast_multiclass_gradient_and_quantize(class, u64::MAX, 0, 0)
                .unwrap();
            assert_eq!(
                dq.grad_scale(),
                hq.grad_scale(),
                "factor {factor} class {class}"
            );
            assert_eq!(
                dq.hess_scale(),
                hq.hess_scale(),
                "factor {factor} class {class}"
            );
        }
        assert_eq!(
            ctx.device_scale_fallbacks(),
            0,
            "factor {factor}: device scale must agree with the host quantizer without fallback"
        );
    }
}

/// 非有限梯度在 fast 的 device scale 路径上仍然报错(host 校验 max),不会量化后继续训练。
#[test]
fn gpu_fast_non_finite_gradient_is_rejected_after_device_scale() {
    use ferrisboost::types::Objective;
    let n_rows = 1500usize;
    let labels: Vec<f32> = (0..n_rows).map(|i| (i % 5) as f32).collect();
    let config = SharedHistogramConfig {
        runtime_log: false,
        gpu_math: ferrisboost::types::GpuMath::Fast,
        ..Default::default()
    };
    let ctx = GpuTrainCtx::new(0, n_rows, 4, 19, 2, 8, config, false).unwrap();
    // expf(200) overflows f32, so every poisson gradient is +inf.
    ctx.init_fast_math(&labels, None, &[200.0]).unwrap();
    let Err(err) = ctx.fast_gradient_and_quantize(
        Objective::Poisson {
            max_delta_step: 0.7,
        },
        u64::MAX,
        0,
        0,
    ) else {
        panic!("non-finite gradients must be rejected");
    };
    assert!(
        format!("{err:#}").contains("non-finite"),
        "unexpected error: {err:#}"
    );
}

/// 多分类 fast 训练:逐次确定,和 exact(= CPU 字节)接近,覆盖 K=2/3/7、权重、
/// 行/列采样、单调约束、streaming/resident 两种布局。
#[test]
fn gpu_multiclass_fast_is_deterministic_and_close_to_exact() {
    use ferrisboost::types::{GpuMath, Objective, TrainParams};
    for k in [2usize, 3, 7] {
        let n_rows = 2400usize;
        let n_feats = 6usize;
        let (source, labels, weights) = multiclass_fixture(n_rows, n_feats, k);
        for (weighted, subsample, colsample, monotone, resident) in [
            (false, 1.0f32, 1.0f32, vec![], None),
            (true, 0.75, 0.7, vec![], Some(0usize)),
            (true, 1.0, 1.0, vec![1i8, -1, 0, 0, 0, 1], None),
        ] {
            let what = format!(
                "k={k} weighted={weighted} subsample={subsample} colsample={colsample} \
                 monotone={monotone:?} resident={resident:?}"
            );
            let run = |math: GpuMath| {
                let params = TrainParams {
                    n_rounds: 4,
                    max_depth: 4,
                    max_bin: 32,
                    cols_per_block: 3,
                    nthread: 1,
                    subsample,
                    colsample_bytree: colsample,
                    monotone_constraints: monotone.clone(),
                    resident_blocks: resident,
                    seed: 6,
                    gpu_math: math,
                    ..Default::default()
                };
                let cfg = TrainConfig::new(
                    params,
                    Objective::Softprob {
                        num_class: k as u32,
                    },
                );
                train_gpu_weighted(
                    0,
                    &source,
                    &labels,
                    weighted.then_some(&weights[..]),
                    &[],
                    &cfg,
                    &mut [],
                    &Local,
                )
                .unwrap()
            };
            let exact = run(GpuMath::Exact);
            let fast = run(GpuMath::Fast);
            let again = run(GpuMath::Fast);
            assert_eq!(
                fast.to_xgboost_json().unwrap(),
                again.to_xgboost_json().unwrap(),
                "{what}: fast 必须逐次确定"
            );
            assert_eq!(fast.trees.len(), 4 * k, "{what}");
            // 所有行都收到了 delta(采样不是预测掩码):margin 和 exact 接近。
            let row = |i: usize| -> Vec<f32> {
                (0..n_feats)
                    .map(|f| {
                        if (i * 7 + f).is_multiple_of(29) {
                            f32::NAN
                        } else {
                            ((i * 13 + f * 11) % 41) as f32
                        }
                    })
                    .collect()
            };
            let mut max_diff = 0.0f32;
            for i in (0..n_rows).step_by(7) {
                let a = fast.predict_margins_upto(&row(i), usize::MAX);
                let b = exact.predict_margins_upto(&row(i), usize::MAX);
                for (x, y) in a.iter().zip(&b) {
                    max_diff = max_diff.max((x - y).abs());
                }
            }
            assert!(
                max_diff < 5e-3,
                "{what}: fast 与 exact 的 margin 差 {max_diff}"
            );
        }
    }
}

/// 预算:运行时的 `multiclass_bytes`、回显和规划器的 `with_fast_multiclass`
/// 必须是同一个数;exact 与单输出不收费。
#[test]
fn gpu_multiclass_fast_budget_matches_planner_and_echo() {
    use ferrisboost::gpu_mem_plan::StreamingMemModel;
    use ferrisboost::types::GpuMath;
    let n_rows = 5000usize;
    let ctx_for = |math: GpuMath, k: usize| {
        let config = SharedHistogramConfig {
            runtime_log: false,
            gpu_math: math,
            n_outputs: k,
            fast_rank: None,
            ..Default::default()
        };
        GpuTrainCtx::new(0, n_rows, 4, 19, 2, 8, config, false).unwrap()
    };
    let single = ctx_for(GpuMath::Fast, 1);
    assert_eq!(single.budget().multiclass_bytes, 0);
    assert!(!single.path_summary().contains("multiclass"));
    let exact = ctx_for(GpuMath::Exact, 7);
    assert_eq!(exact.budget().multiclass_bytes, 0);
    assert_eq!(
        exact.budget().total(),
        ctx_for(GpuMath::Exact, 1).budget().total()
    );
    assert!(!exact.path_summary().contains("multiclass"));

    for k in [2usize, 5, 7] {
        let ctx = ctx_for(GpuMath::Fast, k);
        let extra = n_rows * 4 * (k - 1) + n_rows * 8;
        let b = ctx.budget();
        assert_eq!(b.multiclass_bytes, extra);
        assert_eq!(b.total(), single.budget().total() + extra);
        assert_eq!(b.per_row_bytes(), single.budget().per_row_bytes() + extra);
        assert!(b.report().contains("multiclass"));
        let echo = ctx.path_summary();
        assert!(
            echo.contains(&format!("multiclass_outputs={k} multiclass_bytes={extra}")),
            "{echo}"
        );
        // 规划器用同一个公式。
        let base = StreamingMemModel::new(n_rows as u64, 32, 1, 16, true, true);
        let planned = base.with_fast_multiclass(n_rows as u64, k as u64);
        assert_eq!(planned.fixed_bytes - base.fixed_bytes, extra as u64);
    }
}

/// `base_margin` on the fast path: the resident device prediction is filled from
/// per-row margins (class-major `K x n`) exactly as given, and a constant margin
/// is the same state as the per-output initialization.
#[test]
fn gpu_fast_per_row_margins_initialize_the_device_prediction() {
    for k in [1usize, 3] {
        let n_rows = 2500usize;
        let (_, labels, weights) = multiclass_fixture(n_rows, 4, k.max(2));
        let labels: Vec<f32> = labels
            .iter()
            .map(|&y| if k == 1 { y.min(1.0) } else { y })
            .collect();
        let config = SharedHistogramConfig {
            runtime_log: false,
            gpu_math: ferrisboost::types::GpuMath::Fast,
            fast_weights: true,
            n_outputs: k,
            ..Default::default()
        };
        let make = || GpuTrainCtx::new(0, n_rows, 4, 19, 2, 8, config, false).unwrap();
        // Distinct finite values everywhere, including negative zero.
        let rows: Vec<f32> = (0..k * n_rows)
            .map(|i| {
                if i == 7 {
                    -0.0
                } else {
                    ((i * 31 % 977) as f32 - 488.0) * 0.003
                }
            })
            .collect();
        let ctx = make();
        ctx.init_fast_math_rows(&labels, Some(&weights), &rows)
            .unwrap();
        let mut back = vec![0.0f32; k * n_rows];
        ctx.sync_prediction_to_host(&mut back).unwrap();
        assert!(
            rows.iter()
                .zip(&back)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "k={k}: device prediction must equal the uploaded margins bit for bit"
        );
        // A constant per-row margin == the per-output initialization.
        let per_output: Vec<f32> = (0..k).map(|c| 0.25 * c as f32 - 0.1).collect();
        let constant: Vec<f32> = per_output
            .iter()
            .flat_map(|&m| std::iter::repeat_n(m, n_rows))
            .collect();
        let a = make();
        a.init_fast_math(&labels, Some(&weights), &per_output)
            .unwrap();
        let b = make();
        b.init_fast_math_rows(&labels, Some(&weights), &constant)
            .unwrap();
        let (mut ha, mut hb) = (vec![0.0f32; k * n_rows], vec![0.0f32; k * n_rows]);
        a.sync_prediction_to_host(&mut ha).unwrap();
        b.sync_prediction_to_host(&mut hb).unwrap();
        assert_eq!(
            ha.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            hb.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // Wrong length is an explicit error.
        assert!(make()
            .init_fast_math_rows(&labels, Some(&weights), &rows[1..])
            .is_err());
    }
}
