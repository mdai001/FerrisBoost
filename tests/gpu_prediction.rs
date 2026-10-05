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
