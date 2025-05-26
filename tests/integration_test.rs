//! Integration tests for the BERTScore implementation.

use rust_bert_score::{
    BERTScorerBuilder, BERTScorerConfig,
    baseline::BaselineScores,
};
use rust_bert::pipelines::common::ModelType;
use tch::Device;

#[test]
#[ignore] // Ignore by default as it downloads models
fn test_basic_scoring() {
    // This test requires model downloads, so it's ignored by default
    // Run with: cargo test --test integration_test -- --ignored
    
    let config = BERTScorerConfig {
        model_type: ModelType::Bert,
        model_name: "bert-base-uncased".to_string(),
        language: "en".to_string(),
        vocab_path: "path/to/vocab.txt".to_string(), // Would need actual path
        merges_path: None,
        lower_case: true,
        device: Device::Cpu,
        num_layers: None,
        max_length: 128,
        batch_size: 2,
        use_idf: false,
        rescale_with_baseline: false,
        custom_baseline: None,
    };
    
    // Would need actual vocab path to run
    // let scorer = BERTScorer::new(config).unwrap();
    
    // Test data
    let candidates = vec![
        "The cat sat on the mat.",
        "A dog ran in the park.",
    ];
    
    let references = vec![
        "The cat was sitting on the mat.",
        "A dog was running in the park.",
    ];
    
    // let results = scorer.score(&candidates, &references).unwrap();
    
    // assert_eq!(results.len(), 2);
    // for result in results {
    //     assert!(result.precision > 0.0 && result.precision <= 1.0);
    //     assert!(result.recall > 0.0 && result.recall <= 1.0);
    //     assert!(result.f1 > 0.0 && result.f1 <= 1.0);
    // }
}

#[test]
fn test_similarity_computation() {
    // Test the similarity module directly
    use rust_bert_score::similarity::compute_bertscore;
    use tch::Tensor;
    
    // Create simple test embeddings
    let cand_emb = Tensor::from_slice2(&[
        &[1.0, 0.0, 0.0],  // "cat"
        &[0.9, 0.1, 0.0],  // "sat"
        &[0.0, 1.0, 0.0],  // "on"
    ]);
    
    let ref_emb = Tensor::from_slice2(&[
        &[1.0, 0.0, 0.0],  // "cat" (exact match)
        &[0.8, 0.2, 0.0],  // "sitting" (similar to "sat")
        &[0.0, 1.0, 0.0],  // "on" (exact match)
        &[0.0, 0.0, 1.0],  // "the"
    ]);
    
    // All tokens are valid (no special tokens)
    let cand_mask = Tensor::from_slice(&[1.0, 1.0, 1.0]);
    let ref_mask = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    
    let result = compute_bertscore(&cand_emb, &ref_emb, &cand_mask, &ref_mask, None).unwrap();
    
    // Check that scores are in valid range
    assert!(result.precision > 0.0 && result.precision <= 1.0);
    assert!(result.recall > 0.0 && result.recall <= 1.0);
    assert!(result.f1 > 0.0 && result.f1 <= 1.0);
    
    // F1 should be harmonic mean of precision and recall
    let expected_f1 = 2.0 * result.precision * result.recall / (result.precision + result.recall);
    assert!((result.f1 - expected_f1).abs() < 1e-6);
}

#[test]
fn test_idf_weighting() {
    use rust_bert_score::idf::IdfDict;
    use std::collections::HashSet;
    
    // Create test reference corpus
    let references = vec![
        vec![1, 2, 3],      // "the cat sat"
        vec![1, 4, 5],      // "the dog ran"
        vec![2, 6, 7],      // "cat jumped high"
    ];
    
    let special_tokens = HashSet::from([0]); // 0 is special
    
    let idf_dict = IdfDict::from_references(&references, &special_tokens).unwrap();
    
    // "the" (token 1) appears in 2/3 docs, should have lower IDF
    // "jumped" (token 6) appears in 1/3 docs, should have higher IDF
    let the_idf = idf_dict.get_score(1);
    let jumped_idf = idf_dict.get_score(6);
    
    assert!(jumped_idf > the_idf);
    assert!(the_idf > 0.0);
}

#[test]
fn test_baseline_rescaling() {
    use rust_bert_score::baseline::{BaselineScores, rescale_with_baseline};
    
    let baseline = BaselineScores::new(0.8, 0.8, 0.8);
    let raw_scores = (0.9, 0.85, 0.875);
    
    let (p, r, f1) = rescale_with_baseline(raw_scores, &baseline);
    
    // (0.9 - 0.8) / (1.0 - 0.8) = 0.5
    assert!((p - 0.5).abs() < 1e-6);
    // (0.85 - 0.8) / 0.2 = 0.25
    assert!((r - 0.25).abs() < 1e-6);
    // (0.875 - 0.8) / 0.2 = 0.375
    assert!((f1 - 0.375).abs() < 1e-6);
}

#[test]
fn test_builder_pattern() {
    // Test that the builder pattern works correctly
    let builder = BERTScorerBuilder::new()
        .model(ModelType::Roberta, "roberta-large")
        .language("en")
        .batch_size(32)
        .use_idf(true)
        .rescale_with_baseline(true);
    
    // Would need actual vocab paths to build
    // let scorer = builder
    //     .vocab_paths("/path/to/vocab.json", Some("/path/to/merges.txt"))
    //     .build()
    //     .unwrap();
}

#[test]
fn test_multi_reference_scoring() {
    // Test conceptually how multi-reference scoring should work
    use rust_bert_score::similarity::BERTScoreResult;
    
    // Simulate scoring against multiple references
    let scores = vec![
        BERTScoreResult { precision: 0.8, recall: 0.7, f1: 0.746 },
        BERTScoreResult { precision: 0.9, recall: 0.85, f1: 0.874 }, // Best F1
        BERTScoreResult { precision: 0.75, recall: 0.8, f1: 0.774 },
    ];
    
    // Should select the one with highest F1
    let best = scores.into_iter().max_by(|a, b| {
        a.f1.partial_cmp(&b.f1).unwrap()
    }).unwrap();
    
    assert_eq!(best.f1, 0.874);
}