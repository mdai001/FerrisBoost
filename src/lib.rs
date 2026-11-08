//! ferrisboost — 列分块的梯度提升树。
//!
//! Column-blocked training reads Arrow, Parquet, or CSV without a DMatrix.
//! See the [usage guide](https://github.com/mdai001/FerrisBoost/blob/develop/documents/howto.md)
//! for Python examples and the public modules below for the Rust API.

pub mod backend;
pub mod bins;
pub mod callback;
pub mod categorical;
pub mod colsample;
pub mod columns;
pub mod comm;
pub mod error;
pub mod explain;
pub mod gpu_mem_plan;
pub mod hist;
mod inference;
pub mod metrics;
pub mod objective;
pub mod rank;
pub mod sketch;
pub mod source;
pub mod split;
pub mod subsample;
mod threading;
pub mod train;
pub mod tree;
pub mod types;
// Private, compile-only NVTX probes; normal builds have no NVTX dependency.
#[allow(unexpected_cfgs)]
mod diagnostic;
#[cfg(any(feature = "python", test))]
mod model_io;

#[cfg(feature = "python")]
mod python;

pub use callback::{Callback, EarlyStopping, RecordHistory, VerboseEval};
pub use columns::{BinCuts, BinningStrategy};
pub use source::DenseSource;
pub use train::{train, BlockSource, EvalSet, TrainConfig};
pub use tree::Model;
