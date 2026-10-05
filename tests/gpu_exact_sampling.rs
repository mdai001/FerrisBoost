#![cfg(feature = "cuda")]

use ferrisboost::{
    backend::cuda::{GpuTrainCtx, SharedHistogramConfig},
    comm::Local,
    source::DenseSource,
    subsample::{row_selected, threshold_for},
    train::{train, train_gpu, TrainConfig},
    types::{GradPair, GradPairFixed, GradQuantizer, Objective, TrainParams, MISSING_BIN},
};

const ROW_GROUP: usize = 255;

fn quantization_context(n_rows: usize, n_features: usize) -> GpuTrainCtx {
    let mut config = SharedHistogramConfig {
        runtime_log: false,
        device_quantize: Some(true),
        resident_blocks: Some(0),
        ..Default::default()
    };
    config.hist_streams = Some(1);
    GpuTrainCtx::new(
        0,
        n_rows,
        n_features,
        n_features * ROW_GROUP,
        1,
        128,
        config,
        false,
    )
    .unwrap()
}

/// Give each row a unique, non-missing bin in one feature. For larger inputs,
/// each feature owns a disjoint group of at most 255 rows; the other rows are
/// missing for that feature. Summing the feature histograms then reconstructs
/// every quantized row pair through the existing public histogram path.
fn identity_histogram_layout(n_rows: usize) -> (usize, Vec<u32>, Vec<u8>) {
    let n_features = n_rows.div_ceil(ROW_GROUP);
    let mut offsets = Vec::with_capacity(n_features + 1);
    offsets.push(0u32);
    for _ in 0..n_features {
        offsets.push(*offsets.last().unwrap() + ROW_GROUP as u32);
    }

    let mut bins = vec![MISSING_BIN; n_features * n_rows];
    for feat in 0..n_features {
        let begin = feat * ROW_GROUP;
        let end = (begin + ROW_GROUP).min(n_rows);
        for row in begin..end {
            bins[feat * n_rows + row] = (row - begin) as u8;
        }
    }
    (n_features, offsets, bins)
}

fn histogram_gpair(
    ctx: &GpuTrainCtx,
    n_rows: usize,
    n_features: usize,
    offsets: &[u32],
    bins: &[u8],
) -> Vec<GradPairFixed> {
    let rows = ctx.begin_tree_rows().unwrap();
    let mut session = ctx
        .block_session(0, bins, n_rows, n_features, offsets, None)
        .unwrap();
    let mut histogram = Vec::new();
    if session.requires_batched_hist() {
        let spans = [rows];
        let mut outputs = [&mut histogram];
        session
            .histogram_device_batch(&spans, &mut outputs)
            .unwrap();
    } else {
        session.histogram_device(rows, &mut histogram).unwrap();
    }
    histogram
}

fn reconstruct_per_row(
    n_rows: usize,
    n_features: usize,
    offsets: &[u32],
    histogram: &[GradPairFixed],
) -> Vec<GradPairFixed> {
    let mut recovered = vec![GradPairFixed::default(); n_rows];
    for feat in 0..n_features {
        let begin = feat * ROW_GROUP;
        let end = (begin + ROW_GROUP).min(n_rows);
        for row in begin..end {
            recovered[row] = histogram[offsets[feat] as usize + row - begin];
        }
    }
    recovered
}

fn assert_fixed_pairs_eq(actual: &[GradPairFixed], expected: &[GradPairFixed], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    if let Some(row) = actual
        .iter()
        .zip(expected)
        .position(|(got, want)| got != want)
    {
        panic!(
            "{context}: first mismatch at row {row}: got {:?}, expected {:?}",
            actual[row], expected[row]
        );
    }
}

fn assert_same_scales(a: &GradQuantizer, b: &GradQuantizer, what: &str) {
    assert_eq!(
        a.grad_scale().to_bits(),
        b.grad_scale().to_bits(),
        "{what}: grad scale"
    );
    assert_eq!(
        a.hess_scale().to_bits(),
        b.hess_scale().to_bits(),
        "{what}: hess scale"
    );
}

fn sample_fixture(n_rows: usize) -> Vec<GradPair> {
    let mut pairs: Vec<_> = (0..n_rows)
        .map(|row| {
            // Include both signs, varied magnitudes, and signed zero.
            let grad = match row % 9 {
                0 => -3.25,
                1 => 3.25,
                2 => -0.0,
                3 => 0.0,
                4 => -1.5,
                _ => ((row * 37 % 101) as f32 - 50.0) / 16.0,
            };
            let hess = match row % 7 {
                0 => 0.0,
                1 => -0.0,
                2 => 0.125,
                3 => 0.375,
                _ => ((row * 13 % 29) as f32 + 1.0) / 64.0,
            };
            GradPair { grad, hess }
        })
        .collect();
    if n_rows > 1 {
        let bounds = GradQuantizer::new(&pairs, n_rows);
        let half_grad = (0.5 / bounds.grad_scale()) as f32;
        pairs[0].grad = half_grad;
        pairs[1].grad = -half_grad;
        // The scales are powers of two, so these exactly represent ±0.5 after
        // scaling and pin ties-away-from-zero for both signed directions.
        assert_eq!(bounds.to_fixed(pairs[0]).grad, 1);
        assert_eq!(bounds.to_fixed(pairs[1]).grad, -1);
        let half_hess = (0.5 / bounds.hess_scale()) as f32;
        pairs[0].hess = half_hess;
        pairs[1].hess = -half_hess;
        assert_eq!(bounds.to_fixed(pairs[0]).hess, 1);
        assert_eq!(bounds.to_fixed(pairs[1]).hess, -1);
    }
    pairs
}

#[test]
fn sampled_device_quantization_matches_host_mask_and_old_upload_at_boundaries() {
    let thresholds = [0, 1, (1u64 << 53) - 1, u64::MAX, threshold_for(0.6)];
    let identities = [(0u64, 0u64), (u64::MAX, u64::MAX)];

    for n_rows in [1usize, 511, 512, 513, 4097] {
        let (n_features, offsets, bins) = identity_histogram_layout(n_rows);
        let ctx = quantization_context(n_rows, n_features);
        let source_gpair = sample_fixture(n_rows);

        for &threshold in &thresholds {
            for &(seed, tree) in &identities {
                let mut host_masked: Vec<_> = source_gpair
                    .iter()
                    .enumerate()
                    .map(|(row, &pair)| {
                        if row_selected(row as u64, threshold, seed, tree) {
                            pair
                        } else {
                            GradPair {
                                grad: 0.0,
                                hess: 0.0,
                            }
                        }
                    })
                    .collect();
                let baseline_quantizer = GradQuantizer::new(&host_masked, n_rows);
                let baseline_fixed: Vec<_> = host_masked
                    .iter()
                    .map(|&pair| baseline_quantizer.to_fixed(pair))
                    .collect();

                let outlier_row = if threshold == 0 {
                    Some(0)
                } else if threshold == threshold_for(0.6) && n_rows > 1 {
                    let selected = (0..n_rows)
                        .filter(|&row| row_selected(row as u64, threshold, seed, tree))
                        .count();
                    assert!(
                        selected > 0 && selected < n_rows,
                        "partial sample must select and reject rows: n_rows={n_rows}, seed={seed}, tree={tree}"
                    );
                    Some(
                        (0..n_rows)
                            .find(|&row| !row_selected(row as u64, threshold, seed, tree))
                            .expect("partial sample has an unselected row"),
                    )
                } else {
                    None
                };
                let mut input_gpair = source_gpair.clone();
                if let Some(row) = outlier_row {
                    assert!(!row_selected(row as u64, threshold, seed, tree));
                    input_gpair[row] = GradPair {
                        grad: 1.0e20,
                        hess: -1.0e10,
                    };
                    host_masked[row] = GradPair {
                        grad: 0.0,
                        hess: 0.0,
                    };
                }
                // The deliberately extreme pair is masked in both the empty
                // and partial samples; it must not affect bounds or root sum.
                let expected_quantizer = GradQuantizer::new(&host_masked, n_rows);
                assert_same_scales(
                    &expected_quantizer,
                    &baseline_quantizer,
                    "masked outlier vs original row",
                );
                let expected_fixed: Vec<_> = host_masked
                    .iter()
                    .map(|&pair| expected_quantizer.to_fixed(pair))
                    .collect();
                assert_eq!(
                    expected_fixed, baseline_fixed,
                    "masked outlier changed fixed rows"
                );
                let expected_root =
                    expected_fixed
                        .iter()
                        .copied()
                        .fold(GradPairFixed::default(), |sum, pair| GradPairFixed {
                            grad: sum.grad.wrapping_add(pair.grad),
                            hess: sum.hess.wrapping_add(pair.hess),
                        });

                let original_input_bits: Vec<_> = input_gpair
                    .iter()
                    .map(|pair| (pair.grad.to_bits(), pair.hess.to_bits()))
                    .collect();
                let (sampled_quantizer, sampled_root, _) = ctx
                    .upload_and_quantize_gpair_sampled(&input_gpair, threshold, seed, tree)
                    .unwrap();
                assert_eq!(
                    input_gpair
                        .iter()
                        .map(|pair| (pair.grad.to_bits(), pair.hess.to_bits()))
                        .collect::<Vec<_>>(),
                    original_input_bits,
                    "sampled upload must not mutate its f32 input"
                );
                assert_same_scales(
                    &sampled_quantizer,
                    &expected_quantizer,
                    "sampled vs host mask",
                );
                assert_eq!(
                    sampled_root, expected_root,
                    "n_rows={n_rows}, threshold={threshold}, seed={seed}, tree={tree}"
                );

                let sampled_hist = histogram_gpair(&ctx, n_rows, n_features, &offsets, &bins);
                let sampled_rows = reconstruct_per_row(n_rows, n_features, &offsets, &sampled_hist);
                assert_fixed_pairs_eq(
                    &sampled_rows,
                    &expected_fixed,
                    &format!("sampled per-row gpair n_rows={n_rows}, threshold={threshold}, seed={seed}, tree={tree}"),
                );

                // The established non-sampled upload remains the reference device
                // quantizer when given the exact host-masked working gpair.
                let (old_quantizer, old_root, _) =
                    ctx.upload_and_quantize_gpair(&host_masked).unwrap();
                assert_same_scales(&sampled_quantizer, &old_quantizer, "sampled vs old upload");
                assert_eq!(sampled_root, old_root, "sampled vs old root sum");
                let old_hist = histogram_gpair(&ctx, n_rows, n_features, &offsets, &bins);
                let old_rows = reconstruct_per_row(n_rows, n_features, &offsets, &old_hist);
                assert_fixed_pairs_eq(
                    &old_rows,
                    &expected_fixed,
                    &format!("old upload per-row gpair n_rows={n_rows}, threshold={threshold}, seed={seed}, tree={tree}"),
                );
            }
        }
    }
}

#[test]
fn sampled_quantization_covers_the_gpair_grid_stride_boundary() {
    // GPAIR quantization launches 1024 blocks of 256 threads. One extra row
    // forces the first element of the grid-stride second iteration.
    let n_rows = 256usize * 1024 + 1;
    let n_features = 1usize;
    let offsets = [0u32, ROW_GROUP as u32];
    let bins: Vec<u8> = (0..n_rows)
        .map(|row| {
            if row % 97 == 0 {
                MISSING_BIN
            } else {
                (row % 251) as u8
            }
        })
        .collect();
    let source_gpair = sample_fixture(n_rows);
    let threshold = threshold_for(0.6);
    let seed = u64::MAX;
    let tree = u64::MAX;
    let selected_count = (0..n_rows)
        .filter(|&row| row_selected(row as u64, threshold, seed, tree))
        .count();
    assert!(selected_count > 0 && selected_count < n_rows);

    let host_masked: Vec<_> = source_gpair
        .iter()
        .enumerate()
        .map(|(row, &pair)| {
            if row_selected(row as u64, threshold, seed, tree) {
                pair
            } else {
                GradPair {
                    grad: 0.0,
                    hess: 0.0,
                }
            }
        })
        .collect();
    let expected_quantizer = GradQuantizer::new(&host_masked, n_rows);
    let expected_fixed: Vec<_> = host_masked
        .iter()
        .map(|&pair| expected_quantizer.to_fixed(pair))
        .collect();
    let expected_root =
        expected_fixed
            .iter()
            .copied()
            .fold(GradPairFixed::default(), |sum, pair| GradPairFixed {
                grad: sum.grad.wrapping_add(pair.grad),
                hess: sum.hess.wrapping_add(pair.hess),
            });
    let mut expected_hist = vec![GradPairFixed::default(); ROW_GROUP];
    for (row, &bin) in bins.iter().enumerate() {
        if bin != MISSING_BIN {
            let slot = &mut expected_hist[bin as usize];
            slot.grad = slot.grad.wrapping_add(expected_fixed[row].grad);
            slot.hess = slot.hess.wrapping_add(expected_fixed[row].hess);
        }
    }

    let ctx = quantization_context(n_rows, n_features);
    let (sampled_quantizer, sampled_root, _) = ctx
        .upload_and_quantize_gpair_sampled(&source_gpair, threshold, seed, tree)
        .unwrap();
    assert_same_scales(
        &sampled_quantizer,
        &expected_quantizer,
        "grid-stride sampled vs host",
    );
    assert_eq!(sampled_root, expected_root, "grid-stride sampled root");
    let sampled_hist = histogram_gpair(&ctx, n_rows, n_features, &offsets, &bins);
    assert_fixed_pairs_eq(
        &sampled_hist,
        &expected_hist,
        "grid-stride sampled histogram",
    );

    let (old_quantizer, old_root, _) = ctx.upload_and_quantize_gpair(&host_masked).unwrap();
    assert_same_scales(
        &sampled_quantizer,
        &old_quantizer,
        "grid-stride sampled vs old upload",
    );
    assert_eq!(sampled_root, old_root, "grid-stride sampled vs old root");
    let old_hist = histogram_gpair(&ctx, n_rows, n_features, &offsets, &bins);
    assert_fixed_pairs_eq(
        &old_hist,
        &expected_hist,
        "grid-stride old upload histogram",
    );
}

#[test]
fn sampled_upload_rejects_length_mismatch() {
    let ctx = quantization_context(1, 1);
    let error = match ctx.upload_and_quantize_gpair_sampled(&[], 0, u64::MAX, u64::MAX) {
        Ok(_) => panic!("length mismatch should be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("长度"));
}

#[test]
fn exact_sampling_matches_cpu_across_quantizers_and_repeats_with_missing_values() {
    let n_rows = 1024usize;
    let n_features = 4usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_features).map(move |feat| {
                if (row * 19 + feat * 31) % 83 == 0 {
                    f32::NAN
                } else {
                    ((row * 17 + feat * 29 + row * feat) % 127) as f32 / 127.0
                }
            })
        })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_features, 32, n_features);
    let objectives = [
        (
            Objective::SquaredError,
            (0..n_rows)
                .map(|row| (row % 97) as f32 / 97.0)
                .collect::<Vec<_>>(),
        ),
        (
            Objective::Logistic,
            (0..n_rows)
                .map(|row| {
                    if (row * 11 + row / 7) % 23 < 11 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
        ),
    ];
    let bytes = |model: &ferrisboost::tree::Model| serde_json::to_vec(model).unwrap();

    for (objective, labels) in objectives {
        for subsample in [0.6f32, 0.01, 1.0] {
            let selected_first_tree = (0..n_rows)
                .filter(|&row| row_selected(row as u64, threshold_for(subsample), 41, 0))
                .count();
            if subsample < 1.0 {
                assert!(selected_first_tree > 0 && selected_first_tree < n_rows);
            }
            let params = TrainParams {
                n_rounds: 3,
                max_depth: 3,
                nthread: 1,
                subsample,
                seed: 41,
                ..Default::default()
            };
            let config = TrainConfig::new(params, objective);
            let cpu = train(&source, &labels, &[], &config, &mut [], &Local).unwrap();

            for (device_quantize, resident_blocks, split_cache) in
                [(true, 1usize, true), (false, 0usize, false)]
            {
                let run_gpu = || {
                    let params = TrainParams {
                        n_rounds: 3,
                        max_depth: 3,
                        nthread: 1,
                        subsample,
                        seed: 41,
                        device_quantize: Some(device_quantize),
                        resident_blocks: Some(resident_blocks),
                        gpu_split_cache: split_cache,
                        ..Default::default()
                    };
                    let config = TrainConfig::new(params, objective);
                    train_gpu(0, &source, &labels, &[], &config, &mut [], &Local).unwrap()
                };
                let gpu = run_gpu();
                let repeated = run_gpu();
                assert_eq!(bytes(&gpu), bytes(&cpu), "CPU/GPU model mismatch: {objective:?}, subsample={subsample}, device_quantize={device_quantize}");
                assert_eq!(bytes(&gpu), bytes(&repeated), "GPU repeat changed model: {objective:?}, subsample={subsample}, device_quantize={device_quantize}");

                if subsample < 1.0 {
                    let baseline_margin = cpu.predict_margin_upto(&data[..n_features], 0);
                    let unsampled_updated = (0..n_rows).any(|row| {
                        !row_selected(row as u64, threshold_for(subsample), 41, 0)
                            && cpu
                                .predict_margin(&data[row * n_features..(row + 1) * n_features])
                                .to_bits()
                                != baseline_margin.to_bits()
                    });
                    assert!(unsampled_updated, "unsampled rows must still receive tree predictions: {objective:?}, subsample={subsample}");
                }
            }
        }
    }
}
