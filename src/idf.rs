//! IDF (Inverse Document Frequency) weight computation for importance weighting.

use crate::Result;
use std::collections::{HashMap, HashSet};
use tch::{Device, Tensor};

/// IDF dictionary mapping token IDs to their IDF scores.
#[derive(Debug, Clone)]
pub struct IdfDict {
    /// Map from token ID to IDF score
    token_scores: HashMap<i64, f32>,
    /// Default IDF score for unseen tokens
    default_score: f32,
    /// Total number of documents (reference sentences)
    pub num_documents: usize,
}

impl IdfDict {
    /// Creates a new IDF dictionary from reference token sequences.
    ///
    /// # Arguments
    /// * `reference_tokens` - List of token ID sequences (one per reference sentence)
    /// * `special_token_ids` - Set of special token IDs to assign zero weight
    ///
    /// # Returns
    /// IdfDict with computed IDF scores
    pub fn from_references(
        reference_tokens: &[Vec<i64>],
        special_token_ids: &HashSet<i64>,
    ) -> Result<Self> {
        let num_documents = reference_tokens.len();
        if num_documents == 0 {
            return Ok(Self {
                token_scores: HashMap::new(),
                default_score: 0.0,
                num_documents: 0,
            });
        }
        
        // Count document frequency for each token
        let mut document_frequencies: HashMap<i64, usize> = HashMap::new();
        
        for tokens in reference_tokens {
            // Use set to count each token only once per document
            let unique_tokens: HashSet<&i64> = tokens.iter().collect();
            for &token_id in unique_tokens {
                *document_frequencies.entry(token_id).or_insert(0) += 1;
            }
        }
        
        // Compute IDF scores
        let mut token_scores = HashMap::new();
        let log_num_docs_plus_one = ((num_documents + 1) as f32).ln();
        
        for (&token_id, &doc_freq) in &document_frequencies {
            if special_token_ids.contains(&token_id) {
                // Special tokens get zero weight
                token_scores.insert(token_id, 0.0);
            } else {
                // IDF formula: log((N + 1) / (df + 1))
                let idf = log_num_docs_plus_one - ((doc_freq + 1) as f32).ln();
                token_scores.insert(token_id, idf);
            }
        }
        
        // Default score for unseen tokens
        let default_score = log_num_docs_plus_one;
        
        Ok(Self {
            token_scores,
            default_score,
            num_documents,
        })
    }
    
    /// Creates an IDF dictionary from a precomputed mapping.
    ///
    /// # Arguments
    /// * `scores` - HashMap of token ID to IDF score
    /// * `default_score` - Score for tokens not in the map
    /// * `num_documents` - Number of documents used to compute IDF
    pub fn from_precomputed(
        scores: HashMap<i64, f32>,
        default_score: f32,
        num_documents: usize,
    ) -> Self {
        Self {
            token_scores: scores,
            default_score,
            num_documents,
        }
    }
    
    /// Gets the IDF score for a token ID.
    pub fn get_score(&self, token_id: i64) -> f32 {
        // First check if it's explicitly in the scores (including special tokens with 0)
        if let Some(&score) = self.token_scores.get(&token_id) {
            score
        } else {
            // For unseen tokens, return default score
            self.default_score
        }
    }
    
    /// Converts token IDs to IDF weight tensors.
    ///
    /// # Arguments
    /// * `token_ids` - List of token IDs
    /// * `device` - Device to place the tensor on
    /// * `special_token_ids` - Set of special token IDs to assign zero weight
    ///
    /// # Returns
    /// Tensor of IDF weights with same length as token_ids
    pub fn to_weight_tensor(&self, token_ids: &[i64], special_token_ids: &HashSet<i64>, device: Device) -> Tensor {
        let weights: Vec<f32> = token_ids
            .iter()
            .map(|&id| {
                if special_token_ids.contains(&id) {
                    0.0
                } else {
                    self.get_score(id)
                }
            })
            .collect();
        
        Tensor::from_slice(&weights).to_device(device)
    }
    
    /// Creates weight tensors for a batch of token sequences.
    ///
    /// # Arguments
    /// * `batch_token_ids` - List of token ID sequences
    /// * `special_token_ids` - Set of special token IDs to assign zero weight
    /// * `device` - Device to place tensors on
    ///
    /// # Returns
    /// List of IDF weight tensors, one per sequence
    pub fn to_weight_tensors_batch(
        &self,
        batch_token_ids: &[Vec<i64>],
        special_token_ids: &HashSet<i64>,
        device: Device,
    ) -> Vec<Tensor> {
        batch_token_ids
            .iter()
            .map(|ids| self.to_weight_tensor(ids, special_token_ids, device))
            .collect()
    }
}

/// Computes IDF weights for use in BERTScore.
///
/// # Arguments
/// * `reference_tokens` - Token sequences from all reference sentences
/// * `candidate_tokens` - Token sequences from all candidate sentences
/// * `special_token_ids` - Set of special tokens to exclude
/// * `device` - Device for tensor placement
///
/// # Returns
/// Tuple of (candidate_weights, reference_weights) as lists of tensors
pub fn compute_idf_weights(
    reference_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<i64>],
    special_token_ids: &HashSet<i64>,
    device: Device,
) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    // Build IDF dictionary from references
    let idf_dict = IdfDict::from_references(reference_tokens, special_token_ids)?;
    
    // Convert to weight tensors
    let candidate_weights = idf_dict.to_weight_tensors_batch(candidate_tokens, special_token_ids, device);
    let reference_weights = idf_dict.to_weight_tensors_batch(reference_tokens, special_token_ids, device);
    
    Ok((candidate_weights, reference_weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_idf_computation() {
        // Test documents:
        // Doc 1: "the cat sat" -> [1, 2, 3]
        // Doc 2: "the dog sat" -> [1, 4, 3]
        // Doc 3: "a cat ran" -> [5, 2, 6]
        let references = vec![
            vec![1, 2, 3],
            vec![1, 4, 3],
            vec![5, 2, 6],
        ];
        
        let special_tokens = HashSet::from([0]); // 0 is special token
        
        let idf_dict = IdfDict::from_references(&references, &special_tokens).unwrap();
        
        // Token 1 ("the"): appears in 2/3 docs
        // IDF = log((3+1)/(2+1)) = log(4/3) ≈ 0.288
        assert!((idf_dict.get_score(1) - 0.288).abs() < 0.01);
        
        // Token 2 ("cat"): appears in 2/3 docs
        assert!((idf_dict.get_score(2) - 0.288).abs() < 0.01);
        
        // Token 3 ("sat"): appears in 2/3 docs
        assert!((idf_dict.get_score(3) - 0.288).abs() < 0.01);
        
        // Token 4 ("dog"): appears in 1/3 docs
        // IDF = log((3+1)/(1+1)) = log(4/2) = log(2) ≈ 0.693
        assert!((idf_dict.get_score(4) - 0.693).abs() < 0.01);
        
        // Token 7 (unseen): gets default score
        // IDF = log((3+1)/(0+1)) = log(4) ≈ 1.386
        assert!((idf_dict.get_score(7) - 1.386).abs() < 0.01);
        
        // Test weight tensor with special tokens
        let test_tokens = vec![0, 1, 2, 7];  // special, seen, seen, unseen
        let weights = idf_dict.to_weight_tensor(&test_tokens, &special_tokens, Device::Cpu);
        
        // Check that special token gets 0 weight
        assert_eq!(f32::try_from(weights.get(0)).unwrap(), 0.0);
    }
    
    #[test]
    fn test_weight_tensor_conversion() {
        let mut scores = HashMap::new();
        scores.insert(1, 0.5);
        scores.insert(2, 1.0);
        scores.insert(3, 1.5);
        
        let idf_dict = IdfDict::from_precomputed(scores, 2.0, 10);
        
        let token_ids = vec![1, 2, 3, 4]; // 4 is unseen
        let special_tokens = HashSet::new(); // No special tokens in this test
        let weights = idf_dict.to_weight_tensor(&token_ids, &special_tokens, Device::Cpu);
        
        let expected = vec![0.5, 1.0, 1.5, 2.0]; // 4 gets default score
        
        // Check each value
        for (i, &expected_val) in expected.iter().enumerate() {
            let actual_val = f32::try_from(weights.get(i as i64)).unwrap();
            assert!((actual_val - expected_val).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_empty_references() {
        let references: Vec<Vec<i64>> = vec![];
        let special_tokens = HashSet::new();
        
        let idf_dict = IdfDict::from_references(&references, &special_tokens).unwrap();
        
        assert_eq!(idf_dict.num_documents, 0);
        assert_eq!(idf_dict.get_score(1), 0.0); // Default for empty
    }
}