use anyhow::Result;
use std::fs;

use crate::cli::ScoreArgs;
use crate::core::{BERTScorerBuilder, choose_hf_or_pathed_tokenizer};
/// Command to compute BERTScore for candidate and reference texts
/// using the specified model and configuration.
///
/// # Arguments
/// `args` - The CLI arguments structure containing:
///  - `candidates`: Path to the file containing candidate texts.
///  - `references`: Path to the file containing reference texts.
///  - `vocab`: Path to the vocabulary file.
///  - `merges`: Optional path to the merges file for tokenization.
///  - `model_type`: The type of BERT model to use (e.g., Bert, DistilBert, etc.).
///  - `idf`: Whether to use IDF weighting.
///  - `baseline`: Whether to apply baseline rescaling.
pub fn cmd_score(args: ScoreArgs) -> Result<()> {
    // Read files
    let candidates = read_lines(&args.candidates)?;
    let references = read_lines(&args.references)?;

    if candidates.len() != references.len() {
        return Err(anyhow::anyhow!(
            "Number of candidates ({}) must equal number of references ({})",
            candidates.len(),
            references.len()
        ));
    }

    let device = tch::Device::cuda_if_available();

    // Validate tokenizer spec (either tokenizer name or vocab/merges must be specified)

    // If a pretrained tokenizer is specified, fetch vocab files instead of using local files
    let (vocab_path, merges_path) = choose_hf_or_pathed_tokenizer(&args.tokenizer)?;

    let scorer = BERTScorerBuilder::new()
        .model(args.model_type.into(), "bert-base-uncased")
        .vocab_paths(vocab_path, merges_path)
        .use_idf(args.idf)
        .device(device)
        .rescale_with_baseline(args.baseline)
        .build()?;

    // In a real implementation, we would:
    let results = scorer.score(&candidates, &references)?;
    for (i, result) in results.iter().enumerate() {
        println!(
            "Pair {}: P={:.3}, R={:.3}, F1={:.3}",
            i + 1,
            result.precision,
            result.recall,
            result.f1
        );
    }

    Ok(())
}

#[inline(always)]
fn read_lines(filename: &str) -> Result<Vec<String>> {
    let content = fs::read_to_string(filename)?;
    Ok(content.lines().map(String::from).collect())
}
