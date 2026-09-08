//! ferrisboost — 列分块的梯度提升树。
//!
//! 见 internal-docs/ARCHITECTURE.md 了解整体设计。

pub mod types;
pub mod columns;
pub mod sketch;
pub mod hist;
pub mod colsample;
pub mod subsample;
pub mod gpu_mem_plan;
pub mod split;
pub mod tree;
pub mod comm;
pub mod metrics;
pub mod callback;
pub mod train;
pub mod source;
pub mod backend;
mod inference;
mod threading;

#[cfg(feature = "python")]
mod python;

pub use train::{train, BlockSource, TrainConfig, EvalSet};
pub use columns::{BinCuts, BinningStrategy};
pub use source::DenseSource;
pub use tree::Model;
pub use callback::{Callback, EarlyStopping, VerboseEval, RecordHistory};
