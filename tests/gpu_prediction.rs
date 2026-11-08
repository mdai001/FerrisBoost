#![cfg(feature = "cuda")]

use ferrisboost::backend::cuda::GpuInferenceModel;
use ferrisboost::train::{train, TrainConfig};
use ferrisboost::types::{Objective, TrainParams};
use ferrisboost::{DenseSource, Model};
use std::sync::Arc;
use std::thread;

fn fixture() -> (Model, Vec<Vec<f32>>) {
    let rows = vec![
        vec![-2.0, 0.5, f32::NAN],
        vec![-1.0, f32::NAN, 3.0],
        vec![0.0, 2.0, -4.0],
        vec![1.0, 3.0, 2.0],
        vec![2.0, -1.0, f32::NAN],
        vec![f32::NAN, 4.0, 1.0],
    ];
    let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
    let source = DenseSource::from_row_major(&flat, rows.len(), 3, 32, 2);
    let params = TrainParams {
        max_depth: 3,
        min_child_weight: 0.0,
        nthread: 1,
        ..Default::default()
    };
    let config = TrainConfig::new(params, Objective::Logistic);
    let model = train(
        &source,
        &labels,
        &[],
        &config,
        &mut [],
        &ferrisboost::comm::Local,
    )
    .unwrap();
    (model, rows)
}

fn repeated_rows(patterns: &[Vec<f32>], n_rows: usize, shift: usize) -> Vec<Vec<f32>> {
    (0..n_rows)
        .map(|row| patterns[(row + shift) % patterns.len()].clone())
        .collect()
}

fn compact_input(gpu: &GpuInferenceModel, rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| gpu.input_features().iter().map(|&feature| row[feature]))
        .collect()
}

fn assert_matches_cpu(gpu: &GpuInferenceModel, model: &Model, rows: &[Vec<f32>], n_trees: usize) {
    let input = compact_input(gpu, rows);
    let expected = rows
        .iter()
        .map(|row| model.predict_margin_upto(row, n_trees))
        .collect::<Vec<_>>();
    let actual = gpu.predict_margins(&input, rows.len(), n_trees).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (row, (want, got)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(want.to_bits(), got.to_bits(), "row {row}, trees {n_trees}");
    }
}

#[test]
fn gpu_prediction_matches_cpu_bit_for_bit_and_reuses_model() {
    let (model, rows) = fixture();
    let gpu = GpuInferenceModel::new(&model, 0).unwrap();
    let flat = rows
        .iter()
        .flat_map(|row| gpu.input_features().iter().map(|&feature| row[feature]))
        .collect::<Vec<_>>();
    assert!(!gpu.input_features().is_empty());
    assert!(gpu.input_features().len() <= model.n_features);

    for n_trees in [0, 1, model.trees.len() / 2, model.trees.len()] {
        let expected = rows
            .iter()
            .map(|row| model.predict_margin_upto(row, n_trees))
            .collect::<Vec<_>>();
        let first = gpu.predict_margins(&flat, rows.len(), n_trees).unwrap();
        let second = gpu.predict_margins(&flat, rows.len(), n_trees).unwrap();
        assert_eq!(first.len(), expected.len());
        for (row, ((want, got), again)) in expected.iter().zip(&first).zip(&second).enumerate() {
            assert_eq!(want.to_bits(), got.to_bits(), "row {row}, trees {n_trees}");
            assert_eq!(got.to_bits(), again.to_bits(), "repeat row {row}");
        }
    }
}

#[test]
fn gpu_prediction_reuses_staging_across_row_sizes_and_tree_counts() {
    let (model, patterns) = fixture();
    let gpu = GpuInferenceModel::new(&model, 0).unwrap();

    // Repeated canonical rows retain NaNs, missing-value default directions,
    // and values on both sides of the fixture's split thresholds.
    for (index, n_rows) in [0, 1, 257, 31, 65_537, 16].into_iter().enumerate() {
        let shift = index % patterns.len();
        let rows = repeated_rows(&patterns, n_rows, shift);
        for n_trees in [0, 1, model.trees.len(), model.trees.len() + 5] {
            assert_matches_cpu(&gpu, &model, &rows, n_trees);
        }
        let changed_rows = repeated_rows(&patterns, n_rows, (shift + 1) % patterns.len());
        assert_matches_cpu(&gpu, &model, &changed_rows, model.trees.len());
    }

    // A malformed call must not poison subsequent valid calls on this model.
    let one_row = repeated_rows(&patterns, 1, 2);
    let mut malformed = compact_input(&gpu, &one_row);
    malformed.pop();
    assert!(gpu
        .predict_margins(&malformed, 1, model.trees.len())
        .is_err());
    assert_matches_cpu(&gpu, &model, &one_row, model.trees.len());
}

#[test]
fn concurrent_gpu_prediction_callers_match_cpu_after_join() {
    let (model, patterns) = fixture();
    let model = Arc::new(model);
    let gpu = Arc::new(GpuInferenceModel::new(&model, 0).unwrap());
    let requests = [(257, 0, 1), (31, 3, model.trees.len() + 5)];
    let workers = requests
        .into_iter()
        .map(|(n_rows, shift, n_trees)| {
            let model = Arc::clone(&model);
            let gpu = Arc::clone(&gpu);
            let patterns = patterns.clone();
            thread::spawn(move || {
                let rows = repeated_rows(&patterns, n_rows, shift);
                for _ in 0..2 {
                    assert_matches_cpu(&gpu, &model, &rows, n_trees);
                }
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn gpu_prediction_leaf_only_model_reuses_staging() {
    let (mut model, patterns) = fixture();
    model.trees = vec![ferrisboost::tree::Tree {
        nodes: vec![ferrisboost::tree::Node::leaf(0.25, 1.0)],
    }];
    let gpu = GpuInferenceModel::new(&model, 0).unwrap();
    // Compact models retain feature 0 for file-reader row-count discovery.
    assert_eq!(gpu.input_features(), &[0]);
    for n_rows in [257, 1, 31, 0, 256] {
        let rows = repeated_rows(&patterns, n_rows, 1);
        for n_trees in [0, 1, 6] {
            assert_matches_cpu(&gpu, &model, &rows, n_trees);
        }
    }
    model.n_features = 0;
    assert!(GpuInferenceModel::new(&model, 0).is_err());
}

/// Row-wise host margin with an optional per-row start, the reference for the
/// device contract: start, then leaf values added in tree order.
fn host_margin(model: &Model, row: &[f32], n_trees: usize, init: Option<f32>) -> f32 {
    match init {
        None => model.predict_margin_upto(row, n_trees),
        Some(start) => {
            let mut acc = start;
            for tree in &model.trees[..n_trees.min(model.trees.len())] {
                acc += tree.leaf_value(row);
            }
            acc
        }
    }
}

fn assert_matches_host(
    gpu: &GpuInferenceModel,
    model: &Model,
    rows: &[Vec<f32>],
    n_trees: usize,
    init: Option<&[f32]>,
) {
    let input = compact_input(gpu, rows);
    let actual = gpu
        .predict_margins_from(&input, rows.len(), n_trees, init)
        .unwrap();
    assert_eq!(actual.len(), rows.len());
    for (i, (row, got)) in rows.iter().zip(&actual).enumerate() {
        let want = host_margin(model, row, n_trees, init.map(|m| m[i]));
        assert_eq!(want.to_bits(), got.to_bits(), "row {i}, trees {n_trees}");
    }
}

fn wide_set(categories: &[u32]) -> Box<[u32]> {
    ferrisboost::categorical::words_from_categories(categories).into_boxed_slice()
}

/// Categorical splits with bitsets of different widths (one beyond 256 bits,
/// one empty, one a single word) mixed with numeric splits, on top of the
/// trained numeric fixture's feature layout.
fn categorical_fixture() -> (Model, Vec<Vec<f32>>) {
    let (mut model, _) = fixture();
    let sets: [(Vec<u32>, bool); 3] = [
        (vec![1, 4, 33, 300, 1000, 40_000], true),
        (vec![], false),
        (vec![0, 31, 32], true),
    ];
    let mut trees = Vec::new();
    for (index, (cats, default_left)) in sets.iter().enumerate() {
        let mut tree = ferrisboost::tree::Tree::default();
        let root = tree.push_leaf(0.0, 1.0);
        let base = 0.125 * (index as f32 + 1.0);
        let (left, right) = tree.split_leaf(
            root,
            index as u32 % 2,
            0.0,
            *default_left,
            1.0,
            base,
            1.0,
            -base * 3.0,
            1.0,
        );
        tree.nodes[root].cats = Some(wide_set(cats));
        // Numeric split below the "left" branch, categorical again on the right.
        tree.split_leaf(left, 2, 0.5, index == 1, 1.0, 0.5, 1.0, -0.25, 1.0);
        tree.split_leaf(
            right,
            index as u32 % 2,
            0.0,
            !default_left,
            1.0,
            1.5,
            1.0,
            -2.0,
            1.0,
        );
        tree.nodes[right].cats = Some(wide_set(&[2, 33, 300]));
        trees.push(tree);
    }
    model.trees = trees;
    model.feature_types = vec![true, true, false];
    // Edge values: member / non-member / fractional / negative / huge / NaN /
    // beyond the stored words / at and above the 2^24 invalid threshold.
    let cat_values = [
        0.0,
        1.0,
        2.0,
        4.9,
        31.0,
        32.0,
        33.0,
        255.0,
        256.0,
        300.0,
        300.99,
        999.0,
        1000.0,
        40_000.0,
        40_000.5,
        65_535.0,
        65_536.0,
        16_777_215.0,
        16_777_216.0,
        -0.5,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut rows = Vec::new();
    for &a in &cat_values {
        for &b in &[1.0, 33.0, 301.0, f32::NAN, -3.0] {
            for &c in &[-1.0, 0.75, f32::NAN] {
                rows.push(vec![a, b, c]);
            }
        }
    }
    (model, rows)
}

#[test]
fn gpu_categorical_prediction_matches_host_bit_for_bit() {
    let (model, rows) = categorical_fixture();
    assert!(model.has_categorical_splits());
    let gpu = GpuInferenceModel::new(&model, 0).unwrap();
    for n_trees in 0..=model.trees.len() + 1 {
        assert_matches_host(&gpu, &model, &rows, n_trees, None);
    }
    // Reuse after a different batch size (staging reuse path).
    assert_matches_host(&gpu, &model, &rows[..7], model.trees.len(), None);
}

#[test]
fn gpu_prediction_with_per_row_start_matches_host_bit_for_bit() {
    // Numeric model.
    let (model, patterns) = fixture();
    let gpu = GpuInferenceModel::new(&model, 0).unwrap();
    let rows = repeated_rows(&patterns, 1000, 2);
    let init: Vec<f32> = (0..rows.len())
        .map(|i| (i as f32 * 0.37).sin() * 3.0)
        .collect();
    for n_trees in [0, 1, model.trees.len()] {
        assert_matches_host(&gpu, &model, &rows, n_trees, Some(&init));
        // Interleave with intercept-only calls on the same cached staging.
        assert_matches_host(&gpu, &model, &rows, n_trees, None);
    }
    // Wrong length is rejected and does not poison the model.
    assert!(gpu
        .predict_margins_from(&compact_input(&gpu, &rows), rows.len(), 1, Some(&init[1..]))
        .is_err());
    assert_matches_host(&gpu, &model, &rows, model.trees.len(), Some(&init));

    // Categorical model.
    let (model, rows) = categorical_fixture();
    let gpu = GpuInferenceModel::new(&model, 0).unwrap();
    let init: Vec<f32> = (0..rows.len()).map(|i| i as f32 * -0.011 + 0.3).collect();
    for n_trees in 0..=model.trees.len() {
        assert_matches_host(&gpu, &model, &rows, n_trees, Some(&init));
    }
}
