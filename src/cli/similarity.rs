use anyhow::Result;
use std::io::{self, BufRead};
use tch::Device;
use tch::Tensor;

use crate::cli::SimilarityArgs;
use crate::core::compute_bertscore;

// Does this work?

pub fn cmd_similarity(_: SimilarityArgs) -> Result<()> {
    println!("Reading embeddings from stdin...");
    println!("Expected format: Two blocks of space-separated floats, separated by a blank line");
    println!();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    // Read candidate embeddings
    let mut cand_values = Vec::new();
    let mut cand_rows = 0;
    let mut cand_cols = 0;

    for line in lines.by_ref() {
        let line = line?;
        if line.trim().is_empty() {
            break; // End of candidate embeddings
        }

        let values : Vec<f32> = line
            .split_whitespace()
            .map(|s| s.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;

        if cand_cols == 0 {
            cand_cols = values.len();
        } else if values.len() != cand_cols {
            return Err(anyhow::anyhow!("Inconsistent number of columns"));
        }

        cand_values.extend(values);
        cand_rows += 1;
    }


    // Read reference embeddings
    let mut ref_values = Vec::new();
    let mut ref_rows = 0;
    let mut ref_cols = 0;

    for line in lines.by_ref() {
        let line = line?;
        if line.trim().is_empty() {
            break; // End of reference embeddings
        }

        let values: Vec<f32> = line
            .split_whitespace()
            .map(|s| s.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;

        if ref_cols == 0 {
            ref_cols = values.len();
        } else if values.len() != ref_cols {
            return Err(anyhow::anyhow!("Inconsistent number of columns"));
        }

        ref_values.extend(values);
        ref_rows += 1;
    }

    if cand_rows == 0 || ref_rows == 0 {
        return Err(anyhow::anyhow!("No embeddings found"));
    }

    if cand_cols != ref_cols {
        return Err(anyhow::anyhow!(
            "Embedding dimensions don't match: {} vs {}",
            cand_cols,
            ref_cols
        ));
    }

    println!("Candidate embeddings: {} x {}", cand_rows, cand_cols);
    println!("Reference embeddings: {} x {}", ref_rows, ref_cols);
    println!();

    // Create tensors
    let cand_emb = Tensor::from_slice(&cand_values).view([cand_rows as i64, cand_cols as i64]);
    let ref_emb = Tensor::from_slice(&ref_values).view([ref_rows as i64, ref_cols as i64]);

    // Create masks (all valid)
    let cand_mask = Tensor::ones([cand_rows as i64], (tch::Kind::Float, Device::Cpu));
    let ref_mask = Tensor::ones([ref_rows as i64], (tch::Kind::Float, Device::Cpu));

    // Compute BERTScore
    let result = compute_bertscore(&cand_emb, &ref_emb, &cand_mask, &ref_mask, None);

    println!("BERTScore Results:");
    println!("  Precision: {:.4}", result.precision);
    println!("  Recall:    {:.4}", result.recall);
    println!("  F1:        {:.4}", result.f1);

    Ok(())
}
