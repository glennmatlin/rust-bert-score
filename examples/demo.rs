//! Comprehensive demo of rust-bert-score functionality.

use rust_bert::pipelines::common::ModelType;

use rust_bert_score::core::{
    IdfDict,
    compute_bertscore,
    create_scoring_mask,
    BaselineManager,
    BaselineScores,
};

// use rust_bert_score::core::idf::IdfDict;
// use rust_bert_score::core::score::{compute_bertscore, create_scoring_mask};
// use rust_bert_score::core::baseline::{BaselineManager, BaselineScores};
use rust_bert_score::{BERTScoreResult, BERTScorerBuilder, BERTScorerConfig};



// use rust_bert_score::{
//     baseline::{BaselineManager, BaselineScores},
//     idf::IdfDict,
//     similarity::{compute_bertscore, create_scoring_mask},
//     BERTScoreResult, BERTScorer, BERTScorerBuilder, BERTScorerConfig,
// };
use std::collections::HashSet;
use std::path::PathBuf;
use tch::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    println!("=== Rust BERTScore Demo ===\n");

    // Demo 1: Basic similarity computation (no model required)
    demo_similarity_computation()?;

    // Demo 2: IDF weighting
    demo_idf_weighting()?;

    // Demo 3: Baseline rescaling
    demo_baseline_rescaling()?;

    // Demo 4: Builder pattern (requires model files)
    demo_builder_pattern();

    // Demo 5: Custom configuration (requires model files)
    demo_custom_config();

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demo_similarity_computation() -> anyhow::Result<()> {
    println!("1. Basic Similarity Computation");
    println!("------------------------------------");

    // Simulate embeddings for words in two sentences
    // Candidate: "The cat sat"
    // Reference: "A cat was sitting"

    let candidate_embeddings = Tensor::from_slice2(&[
        &[0.9, 0.1, 0.0], // "The"
        &[0.1, 0.9, 0.0], // "cat"
        &[0.0, 0.1, 0.9], // "sat"
    ]);

    let reference_embeddings = Tensor::from_slice2(&[
        &[0.8, 0.2, 0.0], // "A"
        &[0.1, 0.9, 0.0], // "cat" (exact match)
        &[0.0, 0.0, 1.0], // "was"
        &[0.0, 0.2, 0.8], // "sitting" (similar to "sat")
    ]);

    // Create masks (all tokens are valid)
    let cand_mask = Tensor::ones(&[3], (tch::Kind::Float, Device::Cpu));
    let ref_mask = Tensor::ones(&[4], (tch::Kind::Float, Device::Cpu));

    // let result = compute_bertscore(
    //     &candidate_embeddings,
    //     &reference_embeddings,
    //     &cand_mask,
    //     &ref_mask,
    //     None,
    // )?;

    let result = compute_bertscore(
        &candidate_embeddings,
        &reference_embeddings,
        &cand_mask,
        &ref_mask,
        None,
    );

    println!("  Precision: {:.3}", result.precision);
    println!("  Recall:    {:.3}", result.recall);
    println!("  F1:        {:.3}", result.f1);
    println!();

    Ok(())
}

fn demo_idf_weighting() -> anyhow::Result<()> {
    println!("2. IDF Weighting");
    println!("---------------------");

    // Simulate a reference corpus
    let reference_tokens = vec![
        vec![1, 2, 3, 4],    // "the cat sat quietly"
        vec![1, 5, 6, 7],    // "the dog ran fast"
        vec![2, 8, 9, 10],   // "cat jumped very high"
        vec![1, 11, 12, 13], // "the bird flew away"
    ];

    let special_tokens = HashSet::from([0]); // 0 is padding

    let idf_dict = IdfDict::from_references(&reference_tokens, &special_tokens)?;

    // Token 1 ("the") appears in 3/4 documents - should have low IDF
    // Token 2 ("cat") appears in 2/4 documents - medium IDF
    // Token 8 ("jumped") appears in 1/4 documents - high IDF

    println!("  IDF scores:");
    println!("    Token 1 ('the'):    {:.3}", idf_dict.get_score(1));
    println!("    Token 2 ('cat'):    {:.3}", idf_dict.get_score(2));
    println!("    Token 8 ('jumped'): {:.3}", idf_dict.get_score(8));
    println!("    Token 99 (unseen):  {:.3}", idf_dict.get_score(99));
    println!();

    Ok(())
}

fn demo_baseline_rescaling() -> anyhow::Result<()> {
    println!("3. Baseline Rescaling Demo");
    println!("--------------------------");

    // Create a baseline manager with some example baselines
    let mut manager = BaselineManager::new();
    manager.add_baseline(
        "bert-base-uncased",
        "en",
        BaselineScores::new(0.85, 0.85, 0.85),
    );

    // Raw scores from BERTScore
    let raw_scores = vec![
        (0.95, 0.92, 0.935), // Good match
        (0.88, 0.87, 0.875), // Moderate match
        (0.85, 0.85, 0.85),  // Baseline level
    ];

    println!("  Raw scores → Rescaled scores:");
    for (i, &(p, r, f1)) in raw_scores.iter().enumerate() {
        if let Some((p_scaled, r_scaled, f1_scaled)) =
            manager.rescale_scores("bert-base-uncased", "en", p, r, f1)
        {
            println!(
                "    Pair {}: ({:.3}, {:.3}, {:.3}) → ({:.3}, {:.3}, {:.3})",
                i + 1,
                p,
                r,
                f1,
                p_scaled,
                r_scaled,
                f1_scaled
            );
        }
    }
    println!();

    Ok(())
}

fn demo_builder_pattern() {
    println!("4. Builder Pattern Demo (Configuration Only)");
    println!("-------------------------------------------");

    let _builder = BERTScorerBuilder::new()
        .model(ModelType::Roberta, "roberta-large")
        .language("en")
        .batch_size(32)
        .num_layers(-2) // Second-to-last layer
        .use_idf(true)
        .rescale_with_baseline(true)
        .device(Device::cuda_if_available());

    println!("  Created BERTScorer configuration:");
    println!("    Model: roberta-large");
    println!("    Language: en");
    println!("    Batch size: 32");
    println!("    Layer: -2 (second-to-last)");
    println!("    IDF weighting: enabled");
    println!("    Baseline rescaling: enabled");
    println!("    Device: {:?}", Device::cuda_if_available());

    // Note: Actual building would require vocab files:
    // let scorer = builder
    //     .vocab_paths("/path/to/vocab.json", Some("/path/to/merges.txt"))
    //     .build()?;


    println!();
}

fn demo_custom_config() {
    println!("5. Custom Configuration Demo");
    println!("----------------------------");

    let _config = BERTScorerConfig {
        model_type: ModelType::Bert,
        model_name: "bert-base-multilingual-cased".to_string(),
        language: "zh".to_string(),
        vocab_path: PathBuf::from("path/to/vocab.json"),
        merges_path: None,
        lower_case: false,
        device: Device::Cpu,
        num_layers: Some(9), // Use layer 9
        max_length: 256,
        batch_size: 16,
        use_idf: true,
        rescale_with_baseline: false,
        custom_baseline: Some(BaselineScores::new(0.84, 0.84, 0.84)),
    };

    println!("  Custom configuration for Chinese:");
    println!("    Model: bert-base-multilingual-cased");
    println!("    Language: zh");
    println!("    Layer: 9");
    println!("    Max length: 256");
    println!("    Batch size: 16");
    println!("    Custom baseline: (0.84, 0.84, 0.84)");

    // Note: Building would require actual model files:
    // let scorer = BERTScorer::new(config)?;
}

/// Example of scoring with special token handling
#[allow(dead_code)]
fn example_with_special_tokens() -> anyhow::Result<()> {
    // Example token IDs including special tokens
    let token_ids = vec![101, 2054, 2003, 1996, 3260, 102]; // [CLS] what is the cat [SEP]
    let special_token_ids = vec![101, 102, 0]; // [CLS], [SEP], [PAD]
    let length = 6; // All tokens before padding

    let _mask = create_scoring_mask(&token_ids, &special_token_ids, length);

    // The mask would be [0, 1, 1, 1, 1, 0] - excluding [CLS] and [SEP]

    Ok(())
}

/// Example of multi-reference scoring simulation
#[allow(dead_code)]
fn example_multi_reference() -> anyhow::Result<()> {
    // Simulate scoring against multiple references
    let candidate_results = vec![
        BERTScoreResult {
            precision: 0.85,
            recall: 0.82,
            f1: 0.834,
        },
        BERTScoreResult {
            precision: 0.90,
            recall: 0.88,
            f1: 0.889,
        }, // Best
        BERTScoreResult {
            precision: 0.87,
            recall: 0.84,
            f1: 0.854,
        },
    ];

    // Select best by F1
    let best = candidate_results
        .into_iter()
        .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap())
        .unwrap();

    println!("Best match: F1 = {:.3}", best.f1);

    Ok(())
}
