use anyhow::Result;
use std::fs;
use serde::{Deserialize, Serialize};

use crate::cli::ScoreArgs;
use crate::core::{BERTScorerBuilder, choose_hf_or_pathed_tokenizer};

#[derive(Debug, Deserialize, Serialize)]
struct TsvRow {
    id: String,
    candidate: String,
    reference: String,
}

#[derive(Debug, Serialize)]
struct CsvOutputRow {
    id: String,
    candidate: String,
    reference: String,
    #[serde(rename = "P_rust")]
    p_rust: f32,
    #[serde(rename = "R_rust")]
    r_rust: f32,
    #[serde(rename = "F1_rust")]
    f1_rust: f32,
}
/// Command to compute BERTScore for candidate and reference texts
/// using the specified model and configuration.
///
/// Supports two input modes:
/// 1. Separate files: --candidates and --references
/// 2. TSV input: --input-tsv with id, candidate, reference columns
///
/// Supports two output modes:
/// 1. Stdout: Pretty-printed results (default)
/// 2. CSV file: --output-csv with preserved input data + score columns
pub fn cmd_score(args: ScoreArgs) -> Result<()> {
    // Parse input data from either TSV or separate files
    let (candidates, references, input_data) = if let Some(tsv_path) = &args.input_tsv {
        // Read TSV input
        let tsv_data = read_tsv_input(tsv_path)?;
        let candidates: Vec<String> = tsv_data.iter().map(|row| row.candidate.clone()).collect();
        let references: Vec<String> = tsv_data.iter().map(|row| row.reference.clone()).collect();
        (candidates, references, Some(tsv_data))
    } else {
        // Read separate files (legacy mode)
        let candidates_file = args.candidates.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Either --input-tsv or --candidates/--references must be specified")
        })?;
        let references_file = args.references.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Either --input-tsv or --candidates/--references must be specified")
        })?;
        
        let candidates = read_lines(candidates_file)?;
        let references = read_lines(references_file)?;
        
        if candidates.len() != references.len() {
            return Err(anyhow::anyhow!(
                "Number of candidates ({}) must equal number of references ({})",
                candidates.len(),
                references.len()
            ));
        }
        
        (candidates, references, None)
    };

    let device = tch::Device::cuda_if_available();

    // Validate tokenizer spec (either tokenizer name or vocab/merges must be specified)

    // If a pretrained tokenizer is specified, fetch vocab files instead of using local files
    let (vocab_path, merges_path) = choose_hf_or_pathed_tokenizer(&args.tokenizer)?;

    let mut builder = BERTScorerBuilder::new()
        .model(args.model_type.into(), &args.model_name)
        .vocab_paths(vocab_path, merges_path)
        .use_idf(args.idf)
        .device(device)
        .rescale_with_baseline(args.baseline)
        .language(&args.lang);
    
    if let Some(layer) = args.layer {
        builder = builder.num_layers(layer);
    }
    
    let scorer = builder.build()?;

    // Compute BERTScore
    let results = scorer.score(&candidates, &references)?;
    
    // Output results
    if let Some(output_path) = &args.output_csv {
        // CSV output mode
        if let Some(tsv_data) = input_data {
            write_csv_output(output_path, &tsv_data, &results)?;
            println!("✓ Results written to {}", output_path);
        } else {
            return Err(anyhow::anyhow!(
                "CSV output requires TSV input (--input-tsv). Use stdout mode for separate files."
            ));
        }
    } else {
        // Stdout mode (legacy)
        for (i, result) in results.iter().enumerate() {
            println!(
                "Pair {}: P={:.3}, R={:.3}, F1={:.3}",
                i + 1,
                result.precision,
                result.recall,
                result.f1
            );
        }
    }

    Ok(())
}

#[inline(always)]
fn read_lines(filename: &str) -> Result<Vec<String>> {
    let content = fs::read_to_string(filename)?;
    Ok(content.lines().map(String::from).collect())
}

fn read_tsv_input(filename: &str) -> Result<Vec<TsvRow>> {
    let file = fs::File::open(filename)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(file);
    
    let mut rows = Vec::new();
    for result in rdr.deserialize() {
        let row: TsvRow = result?;
        rows.push(row);
    }
    
    if rows.is_empty() {
        return Err(anyhow::anyhow!("TSV file is empty or has no valid data rows"));
    }
    
    Ok(rows)
}

fn write_csv_output(
    filename: &str,
    input_data: &[TsvRow],
    results: &[crate::BERTScoreResult],
) -> Result<()> {
    if input_data.len() != results.len() {
        return Err(anyhow::anyhow!(
            "Mismatch between input data ({}) and results ({})",
            input_data.len(),
            results.len()
        ));
    }
    
    let file = fs::File::create(filename)?;
    let mut wtr = csv::Writer::from_writer(file);
    
    for (input_row, result) in input_data.iter().zip(results.iter()) {
        let output_row = CsvOutputRow {
            id: input_row.id.clone(),
            candidate: input_row.candidate.clone(),
            reference: input_row.reference.clone(),
            p_rust: result.precision,
            r_rust: result.recall,
            f1_rust: result.f1,
        };
        wtr.serialize(&output_row)?;
    }
    
    wtr.flush()?;
    Ok(())
}
