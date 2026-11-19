//! ferrisboost — 列分块的梯度提升树。
//!
//! Column-blocked training reads Arrow, Parquet, or CSV without a DMatrix.
//! See the [usage guide](https://github.com/mdai001/FerrisBoost/blob/develop/documents/howto.md)
//! for Python examples and the public modules below for the Rust API.

pub mod backend;
#[path = "data/bins.rs"]
pub mod bins;
#[path = "training/callback.rs"]
pub mod callback;
#[path = "data/categorical.rs"]
pub mod categorical;
#[path = "data/category_names.rs"]
pub mod category_names;
#[path = "data/colsample.rs"]
pub mod colsample;
#[path = "data/columns.rs"]
pub mod columns;
#[path = "training/comm.rs"]
pub mod comm;
#[path = "training/custom_objective.rs"]
pub mod custom_objective;
pub mod error;
#[path = "model/explain.rs"]
pub mod explain;
#[path = "training/gpu_mem_plan.rs"]
pub mod gpu_mem_plan;
#[path = "training/gradient_sampling.rs"]
mod gradient_sampling;
#[path = "training/hist.rs"]
pub mod hist;
#[path = "model/inference.rs"]
mod inference;
#[path = "training/interaction.rs"]
mod interaction;
#[path = "evaluation/metrics.rs"]
pub mod metrics;
#[path = "training/objective.rs"]
pub mod objective;
#[path = "model/prediction_link.rs"]
pub mod prediction_link;
#[path = "training/rank.rs"]
pub mod rank;
#[path = "data/sketch.rs"]
pub mod sketch;
pub mod source;
#[path = "training/split.rs"]
pub mod split;
#[path = "data/subsample.rs"]
pub mod subsample;
#[path = "training/threading.rs"]
mod threading;
#[path = "training/mod.rs"]
pub mod train;
#[path = "model/tree.rs"]
pub mod tree;
pub mod types;
// Private, compile-only NVTX probes; normal builds have no NVTX dependency.
#[allow(unexpected_cfgs)]
mod diagnostic;
#[cfg(any(feature = "python", test))]
#[path = "model/model_io.rs"]
mod model_io;

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
#[path = "python/metric.rs"]
mod python_metric;
#[cfg(feature = "python")]
#[path = "python/objective.rs"]
mod python_objective;

pub use callback::{Callback, EarlyStopping, RecordHistory, VerboseEval};
pub use columns::{BinCuts, BinningStrategy};
pub use prediction_link::PredictionLink;
pub use source::DenseSource;
pub use train::{train, BlockSource, EvalSet, TrainConfig};
pub use tree::Model;

#[path = "training/survival.rs"]
pub mod survival;
