//! Rust implementation of BERTScore.

pub mod tokenizer;
pub mod model;
pub mod similarity;
pub mod idf;
pub mod baseline;
pub mod pipeline;

pub use pipeline::BERTScorer;
/// Convenient alias for a result with a boxed error.
pub type Result<T> = anyhow::Result<T>;
