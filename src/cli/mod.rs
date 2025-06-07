pub mod types;
mod score;
mod similarity;

pub use types::{Cli, ScoreArgs, SimilarityArgs, TokenizerSpec};
pub use score::cmd_score;
pub use similarity::cmd_similarity;
