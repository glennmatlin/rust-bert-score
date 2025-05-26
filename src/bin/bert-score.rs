//! Command-line interface for rust-bert-score.

use anyhow::Result;
use rust_bert_score::{
    BERTScorerBuilder,
    similarity::compute_bertscore,
};
use rust_bert::pipelines::common::ModelType;
use std::env;
use std::fs;
use std::io::{self, BufRead};
use tch::{Device, Tensor};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage(&args[0]);
        return Ok(());
    }
    
    match args[1].as_str() {
        "score" => cmd_score(&args[2..])?,
        "similarity" => cmd_similarity(&args[2..])?,
        "help" | "--help" | "-h" => print_usage(&args[0]),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage(&args[0]);
        }
    }
    
    Ok(())
}

fn print_usage(program: &str) {
    println!("Usage: {} <command> [options]", program);
    println!();
    println!("Commands:");
    println!("  score      Score candidates against references (requires model files)");
    println!("  similarity Compute similarity between embeddings from stdin");
    println!("  help       Show this help message");
    println!();
    println!("Examples:");
    println!("  # Score two sentences (requires vocab files)");
    println!("  {} score --candidates cands.txt --references refs.txt \\", program);
    println!("           --vocab vocab.txt --model-type bert");
    println!();
    println!("  # Compute similarity from embeddings");
    println!("  {} similarity < embeddings.txt", program);
}

fn cmd_score(args: &[String]) -> Result<()> {
    let mut candidates_file = None;
    let mut references_file = None;
    let mut vocab_path = None;
    let mut merges_path = None;
    let mut model_type = ModelType::Roberta;
    let mut use_idf = false;
    let mut use_baseline = false;
    
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidates" | "-c" => {
                i += 1;
                candidates_file = Some(&args[i]);
            }
            "--references" | "-r" => {
                i += 1;
                references_file = Some(&args[i]);
            }
            "--vocab" => {
                i += 1;
                vocab_path = Some(&args[i]);
            }
            "--merges" => {
                i += 1;
                merges_path = Some(&args[i]);
            }
            "--model-type" => {
                i += 1;
                model_type = match args[i].as_str() {
                    "bert" => ModelType::Bert,
                    "roberta" => ModelType::Roberta,
                    "distilbert" => ModelType::DistilBert,
                    "deberta" => ModelType::Deberta,
                    _ => {
                        eprintln!("Unknown model type: {}", args[i]);
                        return Ok(());
                    }
                };
            }
            "--idf" => use_idf = true,
            "--baseline" => use_baseline = true,
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                return Ok(());
            }
        }
        i += 1;
    }
    
    // Validate required arguments
    let candidates_file = candidates_file.ok_or_else(|| {
        anyhow::anyhow!("Missing required argument: --candidates")
    })?;
    let references_file = references_file.ok_or_else(|| {
        anyhow::anyhow!("Missing required argument: --references")
    })?;
    let vocab_path = vocab_path.ok_or_else(|| {
        anyhow::anyhow!("Missing required argument: --vocab")
    })?;
    
    // Read files
    let candidates = read_lines(candidates_file)?;
    let references = read_lines(references_file)?;
    
    if candidates.len() != references.len() {
        return Err(anyhow::anyhow!(
            "Number of candidates ({}) must equal number of references ({})",
            candidates.len(),
            references.len()
        ));
    }
    
    println!("Building BERTScorer...");
    println!("  Model type: {:?}", model_type);
    println!("  Vocab: {}", vocab_path);
    if let Some(merges) = merges_path {
        println!("  Merges: {}", merges);
    }
    println!("  IDF: {}", use_idf);
    println!("  Baseline rescaling: {}", use_baseline);
    println!();
    
    // Note: This would require actual vocab files to work
    let _scorer = BERTScorerBuilder::new()
        .model(model_type, "model-name")
        .vocab_paths(vocab_path, merges_path.map(|s| s.as_str()))
        .use_idf(use_idf)
        .rescale_with_baseline(use_baseline)
        .build();
    
    println!("Note: Full scoring requires model files and would download weights.");
    println!("This is a demonstration of the CLI interface.");
    
    // In a real implementation, we would:
    // let results = scorer.score(&candidates, &references)?;
    // for (i, result) in results.iter().enumerate() {
    //     println!("Pair {}: P={:.3}, R={:.3}, F1={:.3}", 
    //              i + 1, result.precision, result.recall, result.f1);
    // }
    
    Ok(())
}

fn cmd_similarity(_args: &[String]) -> Result<()> {
    println!("Reading embeddings from stdin...");
    println!("Expected format: Two blocks of space-separated floats, separated by a blank line");
    println!();
    
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    
    // Read candidate embeddings
    let mut cand_values = Vec::new();
    let mut cand_rows = 0;
    let mut cand_cols = 0;
    
    while let Some(line) = lines.next() {
        let line = line?;
        if line.trim().is_empty() {
            break;
        }
        
        let values: Vec<f32> = line
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
    
    while let Some(line) = lines.next() {
        let line = line?;
        if line.trim().is_empty() {
            break;
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
    let cand_emb = Tensor::from_slice(&cand_values)
        .view([cand_rows as i64, cand_cols as i64]);
    let ref_emb = Tensor::from_slice(&ref_values)
        .view([ref_rows as i64, ref_cols as i64]);
    
    // Create masks (all valid)
    let cand_mask = Tensor::ones(&[cand_rows as i64], (tch::Kind::Float, Device::Cpu));
    let ref_mask = Tensor::ones(&[ref_rows as i64], (tch::Kind::Float, Device::Cpu));
    
    // Compute BERTScore
    let result = compute_bertscore(&cand_emb, &ref_emb, &cand_mask, &ref_mask, None)?;
    
    println!("BERTScore Results:");
    println!("  Precision: {:.4}", result.precision);
    println!("  Recall:    {:.4}", result.recall);
    println!("  F1:        {:.4}", result.f1);
    
    Ok(())
}

fn read_lines(filename: &str) -> Result<Vec<String>> {
    let content = fs::read_to_string(filename)?;
    Ok(content.lines().map(String::from).collect())
}