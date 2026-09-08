#![cfg(feature = "cuda")]

use ferrisboost::backend::cuda::GpuInferenceModel;
use ferrisboost::train::{train, TrainConfig};
use ferrisboost::types::{Objective, TrainParams};
use ferrisboost::{DenseSource, Model};

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
