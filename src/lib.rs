//! Rust implementation of BERTScore.

pub mod tokenizer;
pub mod model;
pub mod similarity;
pub mod idf;
pub mod baseline;
pub mod pipeline;

#[cfg(feature = "python")]
pub mod python;

// Re-export main types
pub use pipeline::{BERTScorer, BERTScorerBuilder, BERTScorerConfig};
pub use similarity::BERTScoreResult;

/// Convenient alias for a result with a boxed error.
pub type Result<T> = anyhow::Result<T>;

// Python module entry point
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn rust_bert_score(_py: Python, m: &PyModule) -> PyResult<()> {
    python::_rust(_py, m)
}
