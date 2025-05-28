//! Comprehensive integration tests for rust-bert-score

use rust_bert_score::{
    baseline::{BaselineManager, BaselineScores},
    idf::IdfDict,
    pipeline::{BERTScorerBuilder, BERTScorerConfig},
    similarity::{BERTScoreResult, compute_bertscore, create_scoring_mask},
};
use rust_bert::pipelines::common::ModelType;
use std::collections::HashSet;
use tch::{Device, Tensor};

#[test]
fn test_bertscore_computation_pipeline() {
    // Test the core BERTScore computation with mock data
    let device = Device::Cpu;
    
    // Mock embeddings for testing
    let candidate_embeddings = Tensor::from_slice2(&[
        &[1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0],
        &[0.0, 0.0, 1.0],
    ]).to_device(device);
    
    let reference_embeddings = Tensor::from_slice2(&[
        &[1.0, 0.0, 0.0],
        &[0.5, 0.866, 0.0],
        &[0.0, 0.0, 1.0],
        &[0.0, 0.707, 0.707],
    ]).to_device(device);
    
    // Create masks (all tokens valid)
    let candidate_mask = Tensor::ones(&[3], (tch::Kind::Float, device));
    let reference_mask = Tensor::ones(&[4], (tch::Kind::Float, device));
    
    // Compute BERTScore without IDF
    let result = compute_bertscore(
        &candidate_embeddings,
        &reference_embeddings,
        &candidate_mask,
        &reference_mask,
        None,
    ).unwrap();
    
    // Verify results are reasonable
    assert!(result.precision > 0.0 && result.precision <= 1.0);
    assert!(result.recall > 0.0 && result.recall <= 1.0);
    assert!(result.f1 > 0.0 && result.f1 <= 1.0);
    
    // F1 should be harmonic mean of precision and recall
    let expected_f1 = 2.0 * result.precision * result.recall / (result.precision + result.recall);
    assert!((result.f1 - expected_f1).abs() < 1e-5);
}

#[test]
fn test_idf_weighting() {
    // Test IDF computation with multiple documents
    let references = vec![
        vec![101, 2054, 2003, 102],      // [CLS] what is [SEP]
        vec![101, 2054, 2003, 2002, 102], // [CLS] what is he [SEP]
        vec![101, 2002, 2003, 102],       // [CLS] he is [SEP]
    ];
    
    let special_tokens: HashSet<i64> = vec![101, 102].into_iter().collect();
    
    let idf_dict = IdfDict::from_references(&references, &special_tokens).unwrap();
    
    // Special tokens should have 0 weight
    assert_eq!(idf_dict.get_score(101), 0.0);
    assert_eq!(idf_dict.get_score(102), 0.0);
    
    // Common words should have lower IDF
    // "is" (2003) appears in all documents
    let is_score = idf_dict.get_score(2003);
    
    // Less common words should have higher IDF
    // "what" (2054) appears in 2/3 documents
    let what_score = idf_dict.get_score(2054);
    
    // "he" (2002) appears in 2/3 documents
    let he_score = idf_dict.get_score(2002);
    
    // Verify IDF ordering
    assert!(is_score < what_score);
    assert!(is_score < he_score);
}

#[test]
fn test_baseline_rescaling() {
    let baseline_manager = BaselineManager::with_defaults();
    
    // Test rescaling with a known baseline
    let raw_scores = (0.95, 0.93, 0.94);
    
    // Get baseline for bert-base-uncased
    let baseline = baseline_manager.get_baseline("bert-base-uncased", "en").unwrap();
    
    // Rescale scores
    let (p_rescaled, r_rescaled, f1_rescaled) = baseline.rescale(
        raw_scores.0,
        raw_scores.1,
        raw_scores.2,
    );
    
    // Rescaled scores should be different from raw scores
    assert_ne!(p_rescaled, raw_scores.0);
    assert_ne!(r_rescaled, raw_scores.1);
    assert_ne!(f1_rescaled, raw_scores.2);
    
    // Rescaled scores should still be valid (0 to 1)
    assert!(p_rescaled >= 0.0 && p_rescaled <= 1.0);
    assert!(r_rescaled >= 0.0 && r_rescaled <= 1.0);
    assert!(f1_rescaled >= 0.0 && f1_rescaled <= 1.0);
}

#[test]
fn test_scoring_mask_creation() {
    let token_ids = vec![101, 2054, 2003, 1996, 2158, 102, 0, 0];
    let special_tokens = vec![101, 102, 0]; // CLS, SEP, PAD
    let actual_length = 6; // Only first 6 tokens are real
    
    let mask = create_scoring_mask(&token_ids, &special_tokens, actual_length);
    
    let mask_values: Vec<f32> = (0..token_ids.len())
        .map(|i| f32::try_from(mask.get(i as i64)).unwrap())
        .collect();
    
    // Expected: [0, 1, 1, 1, 1, 0, 0, 0]
    assert_eq!(mask_values[0], 0.0); // CLS
    assert_eq!(mask_values[1], 1.0); // what
    assert_eq!(mask_values[2], 1.0); // is
    assert_eq!(mask_values[3], 1.0); // the
    assert_eq!(mask_values[4], 1.0); // man
    assert_eq!(mask_values[5], 0.0); // SEP
    assert_eq!(mask_values[6], 0.0); // PAD
    assert_eq!(mask_values[7], 0.0); // PAD
}

#[test]
fn test_builder_configuration() {
    // Test that builder properly configures BERTScorer
    let builder = BERTScorerBuilder::new()
        .model(ModelType::Bert, "bert-base-uncased")
        .language("en")
        .device(Device::Cpu)
        .batch_size(32)
        .use_idf(true)
        .rescale_with_baseline(true)
        .num_layers(-1);
    
    let config = builder.config.clone();
    
    assert_eq!(config.model_type, ModelType::Bert);
    assert_eq!(config.model_name, "bert-base-uncased");
    assert_eq!(config.language, "en");
    assert_eq!(config.device, Device::Cpu);
    assert_eq!(config.batch_size, 32);
    assert!(config.use_idf);
    assert!(config.rescale_with_baseline);
    assert_eq!(config.num_layers, Some(-1));
}

#[test]
fn test_multi_reference_logic() {
    // Test selecting best score from multiple references
    let results = vec![
        BERTScoreResult { precision: 0.8, recall: 0.7, f1: 0.75 },
        BERTScoreResult { precision: 0.9, recall: 0.85, f1: 0.87 },
        BERTScoreResult { precision: 0.7, recall: 0.9, f1: 0.79 },
    ];
    
    // Find best by F1
    let best = results.into_iter()
        .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap())
        .unwrap();
    
    assert_eq!(best.f1, 0.87);
    assert_eq!(best.precision, 0.9);
    assert_eq!(best.recall, 0.85);
}

#[test]
fn test_batch_processing_logic() {
    // Test that batching covers all items correctly
    let total_items = 157;
    let batch_size = 64;
    
    let mut processed = 0;
    for batch_start in (0..total_items).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(total_items);
        let batch_items = batch_end - batch_start;
        processed += batch_items;
    }
    
    assert_eq!(processed, total_items);
}

#[test]
fn test_tensor_operations() {
    // Test key tensor operations used in BERTScore
    let device = Device::Cpu;
    
    // Test L2 normalization
    let embeddings = Tensor::from_slice2(&[
        &[3.0, 4.0],    // norm = 5
        &[5.0, 12.0],   // norm = 13
    ]).to_device(device);
    
    let norms = embeddings.norm_scalaropt_dim(2.0, &[1], true);
    let normalized = &embeddings / &norms.clamp_min(1e-12);
    
    // Check first vector is normalized
    let first_norm = normalized.get(0).norm_scalaropt_dim(2.0, &[0], false);
    let first_norm_val: f64 = first_norm.double_value(&[]);
    assert!((first_norm_val - 1.0).abs() < 1e-6);
    
    // Test cosine similarity via matrix multiplication
    let a = Tensor::from_slice2(&[[1.0, 0.0], [0.0, 1.0]]).to_device(device);
    let b = Tensor::from_slice2(&[[0.707, 0.707], [1.0, 0.0]]).to_device(device);
    
    let sim = a.matmul(&b.transpose(0, 1));
    
    // sim[0,0] = dot([1,0], [0.707,0.707]) = 0.707
    let sim_00: f64 = sim.get(0).get(0).double_value(&[]);
    assert!((sim_00 - 0.707).abs() < 1e-3);
    // sim[1,1] = dot([0,1], [1,0]) = 0
    let sim_11: f64 = sim.get(1).get(1).double_value(&[]);
    assert!(sim_11.abs() < 1e-6);
}

#[cfg(test)]
mod config_tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = BERTScorerConfig::default();
        assert_eq!(config.model_type, ModelType::Roberta);
        assert_eq!(config.max_length, 512);
        assert_eq!(config.batch_size, 64);
        assert!(!config.use_idf);
        assert!(!config.rescale_with_baseline);
    }
    
    #[test]
    fn test_custom_baseline() {
        let custom_baseline = BaselineScores::new(0.85, 0.85, 0.85);
        let builder = BERTScorerBuilder::new()
            .custom_baseline(custom_baseline);
        
        let config = builder.config.clone();
        
        assert!(config.custom_baseline.is_some());
        let baseline = config.custom_baseline.unwrap();
        assert_eq!(baseline.precision, 0.85);
        assert_eq!(baseline.recall, 0.85);
        assert_eq!(baseline.f1, 0.85);
    }
}