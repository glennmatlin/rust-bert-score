use clap::Parser;
use rust_bert_score::cli::{cmd_score, cmd_similarity, Cli};

fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Cli::parse();

    // Dispatch to the appropriate command
    match args.command {
        rust_bert_score::cli::types::Command::Score(score_args) => cmd_score(score_args),
        rust_bert_score::cli::types::Command::Similarity(similarity_args) => {
            cmd_similarity(similarity_args)
        }
    }
}
