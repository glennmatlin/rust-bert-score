//! Rust implementation of BERTScore.

pub mod cli;
pub mod core;

#[cfg(feature = "python")]
pub mod python;

// Re-export main types
pub use core::{BERTScorer, BERTScorerBuilder, BERTScorerConfig, BERTScoreResult};

// Re-export modules for backward compatibility
pub use core::baseline;
pub use core::idf;
pub use core::pipeline;
pub use core::score as similarity;

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
