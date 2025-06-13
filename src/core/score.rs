//! Pairwise cosine similarity computation module.

use tch::Tensor;

/// Result of BERTScore computation for a single candidate-reference pair.
#[derive(Debug, Clone)]
pub struct BERTScoreResult {
    /// Precision score (how well candidate tokens match reference)
    pub precision: f32,
    /// Recall score (how well reference tokens match candidate)
    pub recall: f32,
    /// F1 score (harmonic mean of precision and recall)
    pub f1: f32,
}

/// Computes BERTScore metrics between candidate and reference embeddings.
///
/// # Arguments
/// * `candidate_embeddings` - Embeddings for candidate tokens (seq_len_c × hidden_dim)
/// * `reference_embeddings` - Embeddings for reference tokens (seq_len_r × hidden_dim)
/// * `candidate_mask` - Boolean mask for valid candidate tokens (excludes padding/special)
/// * `reference_mask` - Boolean mask for valid reference tokens (excludes padding/special)
/// * `idf_weights` - Optional IDF weights for tokens (if None, uniform weighting)
///
/// # Returns
/// BERTScoreResult containing precision, recall, and F1 scores
pub fn compute_bertscore(
    candidate_embeddings: &Tensor,
    reference_embeddings: &Tensor,
    candidate_mask: &Tensor,
    reference_mask: &Tensor,
    idf_weights: Option<(&Tensor, &Tensor)>,
) -> BERTScoreResult {
    // Normalize embeddings to unit length for cosine similarity
    let candidate_norm = normalize_embeddings(candidate_embeddings);
    let reference_norm = normalize_embeddings(reference_embeddings);

    // Compute pairwise cosine similarity matrix (candidate × reference)
    let similarity_matrix = compute_similarity_matrix(candidate_norm, reference_norm);

    // Apply masks to exclude special tokens and padding
    let masked_similarity = apply_masks(similarity_matrix, candidate_mask, reference_mask);

    // Greedy matching: find max similarities for precision and recall
    let (candidate_idf, reference_idf) = if let Some((cand_idf, ref_idf)) = idf_weights {
        (Some(cand_idf), Some(ref_idf))
    } else {
        (None, None)
    };

    let (precision, recall) = compute_scores(
        &masked_similarity,
        candidate_mask,
        reference_mask,
        candidate_idf,
        reference_idf,
    );

    // Compute F1 score
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    BERTScoreResult {
        precision,
        recall,
        f1,
    }
}

/// Normalizes embeddings to unit length (L2 normalization).
fn normalize_embeddings(embeddings: &Tensor) -> Tensor {
    // Compute L2 norm along hidden dimension (dim=1)
    let norms = embeddings.norm_scalaropt_dim(2.0, [1], true);
    // Add small epsilon to avoid division by zero
    let norms = norms.clamp_min(1e-12);
    // Divide embeddings by their norms
    embeddings / norms
}

/// Computes cosine similarity matrix between two sets of normalized embeddings.
fn compute_similarity_matrix(candidate_norm: Tensor, reference_norm: Tensor) -> Tensor {
    // Matrix multiplication: (seq_len_c × hidden) @ (hidden × seq_len_r) = (seq_len_c × seq_len_r)
    candidate_norm.matmul(&reference_norm.transpose(0, 1))
}

/// Applies masks to similarity matrix to exclude special tokens and padding.
fn apply_masks(
    similarity_matrix: Tensor,
    candidate_mask: &Tensor,
    reference_mask: &Tensor,
) -> Tensor {
    // Expand masks to match similarity matrix dimensions
    let cand_mask_expanded = candidate_mask.unsqueeze(1); // (seq_len_c × 1)
    let ref_mask_expanded = reference_mask.unsqueeze(0); // (1 × seq_len_r)

    // Create combined mask (seq_len_c × seq_len_r)
    let combined_mask = (&cand_mask_expanded * &ref_mask_expanded).gt(0.5); // Convert to boolean

    // Apply mask: set masked positions to -inf so they won't be selected by max
    similarity_matrix.where_self(
        &combined_mask,
        &Tensor::full_like(&similarity_matrix, f64::NEG_INFINITY).to_device(combined_mask.device()),
    )
}

/// Computes unweighted precision and recall scores.
///
/// # Arguments
/// * `masked_similarity` - Similarity matrix after applying masks
/// * `candidate_mask` - Mask for valid candidate tokens
/// * `reference_mask` - Mask for valid reference tokens
/// * `candidate_idf` - Optional IDF weights for candidate tokens
/// * `reference_idf` - Optional IDF weights for reference tokens
fn compute_scores(
    masked_similarity: &Tensor,
    candidate_mask: &Tensor,
    reference_mask: &Tensor,
    candidate_idf: Option<&Tensor>,
    reference_idf: Option<&Tensor>,
) -> (f32, f32) {
    // Apply L1 normalization to IDF weights if provided (to match Python BERTScore)
    let (normalized_cand_idf, normalized_ref_idf) = if let (Some(cand_idf), Some(ref_idf)) = (candidate_idf, reference_idf) {
        // Mask IDF weights for valid tokens only
        let masked_cand_idf = cand_idf * candidate_mask;
        let masked_ref_idf = ref_idf * reference_mask;
        
        // Sum of IDF weights for normalization
        let cand_idf_sum = masked_cand_idf.sum(tch::Kind::Float);
        let ref_idf_sum = masked_ref_idf.sum(tch::Kind::Float);
        
        // L1 normalize: divide by sum to create probability distribution
        let norm_cand_idf = if cand_idf_sum.double_value(&[]) > 0.0 {
            &masked_cand_idf / &cand_idf_sum
        } else {
            masked_cand_idf
        };
        
        let norm_ref_idf = if ref_idf_sum.double_value(&[]) > 0.0 {
            &masked_ref_idf / &ref_idf_sum
        } else {
            masked_ref_idf
        };
        
        (Some(norm_cand_idf), Some(norm_ref_idf))
    } else {
        // No IDF weights - use uniform weights (1/n for each valid token)
        let cand_count = candidate_mask.sum(tch::Kind::Float);
        let ref_count = reference_mask.sum(tch::Kind::Float);
        
        let uniform_cand = if cand_count.double_value(&[]) > 0.0 {
            candidate_mask / &cand_count
        } else {
            candidate_mask.shallow_clone()
        };
        
        let uniform_ref = if ref_count.double_value(&[]) > 0.0 {
            reference_mask / &ref_count
        } else {
            reference_mask.shallow_clone()
        };
        
        (Some(uniform_cand), Some(uniform_ref))
    };

    // Precision: for each candidate token, find max similarity with any reference token
    let max_cand_sims = masked_similarity.max_dim(1, false).0;
    
    // Apply normalized weights and mask
    let weighted_cand_sims = if let Some(ref weights) = &normalized_cand_idf {
        (&max_cand_sims * weights).where_self(
            &candidate_mask.eq(1),
            &Tensor::zeros_like(&max_cand_sims).to_device(candidate_mask.device()),
        )
    } else {
        max_cand_sims
    };
    
    // Recall: for each reference token, find max similarity with any candidate token
    let max_ref_sims = masked_similarity.max_dim(0, false).0;
    
    // Apply normalized weights and mask
    let weighted_ref_sims = if let Some(ref weights) = &normalized_ref_idf {
        (&max_ref_sims * weights).where_self(
            &reference_mask.eq(1),
            &Tensor::zeros_like(&max_ref_sims).to_device(reference_mask.device()),
        )
    } else {
        max_ref_sims
    };

    // Sum weighted similarities (weights already sum to 1.0 due to L1 normalization)
    let precision = weighted_cand_sims.sum(tch::Kind::Float).double_value(&[]) as f32;
    let recall = weighted_ref_sims.sum(tch::Kind::Float).double_value(&[]) as f32;

    (precision, recall)
}

/// Batch computation of BERTScore for multiple candidate-reference pairs.
#[cfg(any())] // Fake config that never matches, to avoid unused function warning and keep this code for future use
pub fn compute_bertscore_batch(
    embeddings: &[Tensor],       // All embeddings from model forward pass
    candidate_lengths: &[usize], // Actual lengths of candidate sequences
    reference_lengths: &[usize], // Actual lengths of reference sequences
    layer_index: usize,          // Which layer to use (e.g., -1 for last)
    _special_tokens_mask: Option<&[Vec<bool>]>, // Masks for special tokens to exclude
    _idf_weights: Option<&[Vec<f32>]>, // IDF weights for each token
    _device: Device,
) -> Result<Vec<BERTScoreResult>> {
    // Extract embeddings from specified layer
    let _layer_embeddings = &embeddings[layer_index];

    let batch_size = candidate_lengths.len();
    let mut results = Vec::with_capacity(batch_size);

    // Process each candidate-reference pair
    for i in 0..batch_size {
        // Extract candidate and reference embeddings for this pair
        let _cand_len = candidate_lengths[i];
        let _ref_len = reference_lengths[i];

        // TODO: Extract individual sequences from batched embeddings
        // This requires knowing the batch structure from tokenization
        // For now, we'll implement the core similarity computation

        // Placeholder for actual implementation
        results.push(BERTScoreResult {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
        });
    }

    Ok(results)
}

/// Creates a mask that excludes special tokens (CLS, SEP, PAD) from scoring.
///
/// # Arguments
/// * `token_ids` - Token IDs from tokenization
/// * `special_token_ids` - Set of special token IDs to exclude
/// * `length` - Actual sequence length (before padding)
///
/// # Returns
/// Boolean tensor where true indicates a token should be included in scoring
pub fn create_scoring_mask(token_ids: &[i64], special_token_ids: &[i64], length: usize) -> Tensor {
    let mut mask = vec![1.0f32; token_ids.len()];

    // Set mask to 0 for special tokens
    for (i, &token_id) in token_ids.iter().enumerate() {
        if special_token_ids.contains(&token_id) || i >= length {
            mask[i] = 0.0;
        }
    }

    Tensor::from_slice(&mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_normalize_embeddings() {
        let embeddings = Tensor::from_slice2(&[
            &[3.0, 4.0], // norm = 5.0
            &[1.0, 0.0], // norm = 1.0
            &[0.0, 0.0], // norm = 0.0 (should handle gracefully)
        ]);

        let normalized = normalize_embeddings(&embeddings);
        let expected = Tensor::from_slice2(&[&[0.6, 0.8], &[1.0, 0.0], &[0.0, 0.0]]);

        assert!((normalized - expected).abs().max().double_value(&[]) < 1e-6);
    }

    #[test]
    fn test_compute_similarity_matrix() {
        let cand = Tensor::from_slice2(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let ref_ = Tensor::from_slice2(&[&[1.0, 0.0], &[0.707, 0.707], &[0.0, 1.0]]);

        let sim_matrix = compute_similarity_matrix(cand, ref_);
        let expected = Tensor::from_slice2(&[&[1.0, 0.707, 0.0], &[0.0, 0.707, 1.0]]);

        assert!((sim_matrix - expected).abs().max().double_value(&[]) < 1e-3);
    }

    #[test]
    fn test_compute_bertscore() {
        // Simple test case with 2 candidate tokens and 3 reference tokens
        let cand_emb = Tensor::from_slice2(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let ref_emb = Tensor::from_slice2(&[
            &[1.0, 0.0],
            &[0.5, 0.866], // ~60 degree angle
            &[0.0, 1.0],
        ]);
        let cand_mask = Tensor::from_slice(&[1.0, 1.0]);
        let ref_mask = Tensor::from_slice(&[1.0, 1.0, 1.0]);

        let result = compute_bertscore(&cand_emb, &ref_emb, &cand_mask, &ref_mask, None);

        // Precision: each candidate token matches its best reference
        // Token 0 matches ref token 0 with similarity 1.0
        // Token 1 matches ref token 2 with similarity 1.0
        // Precision = (1.0 + 1.0) / 2 = 1.0

        // Recall: each reference token matches its best candidate
        // Ref token 0 matches cand token 0 with similarity 1.0
        // Ref token 1 matches cand token 0 with similarity 0.5 or cand token 1 with similarity 0.866
        // Ref token 2 matches cand token 1 with similarity 1.0
        // Recall = (1.0 + 0.866 + 1.0) / 3 = 0.955...

        assert!((result.precision - 1.0).abs() < 0.01);
        assert!((result.recall - 0.955).abs() < 0.01);
        assert!(result.f1 > 0.9 && result.f1 < 1.0);
    }
}
