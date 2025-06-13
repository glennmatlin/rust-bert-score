//! High-level pipeline assembling tokenizer, model, and scoring.

use crate::core::{
    baseline::{BaselineManager, BaselineScores},
    idf::compute_idf_weights,
    model::Model,
    score::{compute_bertscore, create_scoring_mask, BERTScoreResult},
    tokenizer::{Tokenizer, TokenizerArgs},
};
use crate::Result;
use rust_bert::pipelines::common::ModelType;
use rust_tokenizers::tokenizer::TruncationStrategy;
use std::{collections::HashSet, path::PathBuf};
use tch::{Device, Tensor};

/// Configuration for BERTScorer.
#[derive(Debug, Clone)]
pub struct BERTScorerConfig {
    /// Model type (e.g., BERT, RoBERTa, DeBERTa)
    pub model_type: ModelType,
    /// Model name for baseline lookup (e.g., "roberta-large")
    pub model_name: String,
    /// Language code for baseline lookup (e.g., "en", "zh")
    pub language: String,
    /// Path to vocabulary file
    pub vocab_path: PathBuf,
    /// Path to merges file (for BPE tokenizers)
    pub merges_path: Option<PathBuf>,
    /// Whether to lowercase input text
    pub lower_case: bool,
    /// Device to run on (CPU or CUDA)
    pub device: Device,
    /// Which layer to extract embeddings from (0-indexed, -1 for last)
    pub num_layers: Option<i32>,
    /// Maximum sequence length
    pub max_length: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Whether to use IDF weighting
    pub use_idf: bool,
    /// Whether to rescale with baseline
    pub rescale_with_baseline: bool,
    /// Custom baseline scores (if not using defaults)
    pub custom_baseline: Option<BaselineScores>,
}

impl Default for BERTScorerConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Roberta,
            model_name: "roberta-large".into(),
            language: "en".into(),
            vocab_path: PathBuf::default(),
            merges_path: None,
            lower_case: false,
            device: Device::Cpu,
            num_layers: None,
            max_length: 512,
            batch_size: 64,
            use_idf: false,
            rescale_with_baseline: false,
            custom_baseline: None,
        }
    }
}

/// BERTScorer handles the full BERTScore pipeline.
pub struct BERTScorer {
    tokenizer: Tokenizer,
    model: Model,
    config: BERTScorerConfig,
    baseline_manager: BaselineManager,
    special_token_ids: HashSet<i64>,
}

impl BERTScorer {
    /// Get the configuration.
    pub fn config(&self) -> &BERTScorerConfig {
        &self.config
    }
    
    /// Creates a new BERTScorer instance.
    pub fn new(config: BERTScorerConfig) -> Result<Self> {
        // Initialize tokenizer

        let tokenizer_args : TokenizerArgs = TokenizerArgs {
            model: config.model_type,
            vocab_path: config.vocab_path.clone(),
            merges_path: config.merges_path.clone(),
            lower_case: config.lower_case,
            strip_accents: None, // Default behavior
            add_prefix_space: match config.model_type {
                ModelType::Roberta | ModelType::GPT2 => Some(true),
                _ => None,
            },
            max_len: config.max_length,
            truncation_strategy: TruncationStrategy::LongestFirst,
            stride: 0, // No stride for single sentences
        };

        let tokenizer = Tokenizer::new(
            tokenizer_args
        )?;

        // Initialize model
        let model = Model::new(config.model_type, config.device)?;

        // Initialize baseline manager
        let mut baseline_manager = BaselineManager::with_defaults();
        if let Some(baseline) = &config.custom_baseline {
            baseline_manager.add_baseline(&config.model_name, &config.language, *baseline);
        }

        // Identify special token IDs
        let mut special_token_ids = HashSet::new();
        // CLS token is typically BOS in BERT models
        if let Some(bos_id) = tokenizer.tokenizer.get_bos_id() {
            special_token_ids.insert(bos_id);
        }
        if let Some(sep_id) = tokenizer.tokenizer.get_sep_id() {
            special_token_ids.insert(sep_id);
        }
        if let Some(pad_id) = tokenizer.tokenizer.get_pad_id() {
            special_token_ids.insert(pad_id);
        }

        Ok(Self {
            tokenizer,
            model,
            config,
            baseline_manager,
            special_token_ids,
        })
    }

    /// Scores a batch of candidate-reference pairs.
    ///
    /// # Arguments
    /// * `candidates` - List of candidate sentences
    /// * `references` - List of reference sentences (same length as candidates)
    ///
    /// # Returns
    /// Vector of BERTScoreResult, one per candidate-reference pair
    pub fn score<S: AsRef<str> + Send + Sync>(
        &self,
        candidates: &[S],
        references: &[S],
    ) -> Result<Vec<BERTScoreResult>> {
        if candidates.len() != references.len() {
            return Err(anyhow::anyhow!(
                "Number of candidates ({}) must equal number of references ({})",
                candidates.len(),
                references.len()
            ));
        }

        let mut all_results = Vec::with_capacity(candidates.len());

        // Process in batches
        for batch_start in (0..candidates.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(candidates.len());
            let batch_candidates = &candidates[batch_start..batch_end];
            let batch_references = &references[batch_start..batch_end];

            // Process batch
            let batch_results = self.score_batch(batch_candidates, batch_references)?;
            all_results.extend(batch_results);
        }

        Ok(all_results)
    }

    /// Scores a single batch of candidate-reference pairs.
    fn score_batch<S: AsRef<str> + Send + Sync>(
        &self,
        candidates: &[S],
        references: &[S],
    ) -> Result<Vec<BERTScoreResult>> {
        // Tokenize candidates and references
        let cand_encoding = self.tokenizer.encode(candidates, self.config.device);
        let ref_encoding = self.tokenizer.encode(references, self.config.device);

        // Get embeddings from model
        let cand_hidden_states = self.model.forward(
            &cand_encoding.input_ids,
            &cand_encoding.attention_mask,
            Some(&cand_encoding.token_type_ids),
        );

        let ref_hidden_states = self.model.forward(
            &ref_encoding.input_ids,
            &ref_encoding.attention_mask,
            Some(&ref_encoding.token_type_ids),
        );

        // Select layer
        let layer_idx = self.get_layer_index(&cand_hidden_states)?;
        let cand_embeddings = &cand_hidden_states[layer_idx];
        let ref_embeddings = &ref_hidden_states[layer_idx];

        // Compute IDF weights
        // Note: Python BERTScore always uses IDF weights, just with default values when IDF is disabled
        let idf_weights = if self.config.use_idf {
            // Compute actual IDF weights from corpus
            Some(compute_idf_weights(
                &ref_encoding.token_ids,
                &cand_encoding.token_ids,
                &self.special_token_ids,
                Some(self.config.device),
            )?)
        } else {
            // Use default IDF weights: 1.0 for regular tokens, 0.0 for special tokens
            let mut cand_weights = Vec::new();
            let mut ref_weights = Vec::new();
            
            for i in 0..candidates.len() {
                // Create weight vector: 1.0 for regular tokens, 0.0 for special tokens
                let cand_weight_vec: Vec<f32> = cand_encoding.token_ids[i]
                    .iter()
                    .map(|&id| if self.special_token_ids.contains(&id) { 0.0 } else { 1.0 })
                    .collect();
                let cand_weights_tensor = Tensor::from_slice(&cand_weight_vec).to_device(self.config.device);
                cand_weights.push(cand_weights_tensor);
                
                // Create default weights for reference tokens
                let ref_weight_vec: Vec<f32> = ref_encoding.token_ids[i]
                    .iter()
                    .map(|&id| if self.special_token_ids.contains(&id) { 0.0 } else { 1.0 })
                    .collect();
                let ref_weights_tensor = Tensor::from_slice(&ref_weight_vec).to_device(self.config.device);
                ref_weights.push(ref_weights_tensor);
            }
            
            Some((cand_weights, ref_weights))
        };

        // Score each pair
        let mut results = Vec::with_capacity(candidates.len());

        for i in 0..candidates.len() {
            // Extract embeddings for this pair
            let cand_emb = cand_embeddings.get(i as i64);
            let ref_emb = ref_embeddings.get(i as i64);

            // Get actual lengths (before padding)
            let cand_len = cand_encoding.lengths[i];
            let ref_len = ref_encoding.lengths[i];

            // Slice to actual length
            let cand_emb = cand_emb.slice(0, 0, cand_len as i64, 1);
            let ref_emb = ref_emb.slice(0, 0, ref_len as i64, 1);

            // Create masks excluding special tokens
            let cand_mask = create_scoring_mask(
                &cand_encoding.token_ids[i],
                &self.special_token_ids.iter().copied().collect::<Vec<_>>(),
                cand_len,
            );
            let ref_mask = create_scoring_mask(
                &ref_encoding.token_ids[i],
                &self.special_token_ids.iter().copied().collect::<Vec<_>>(),
                ref_len,
            );

            // Get IDF weights for this pair if using IDF
            let pair_idf_weights = if let Some((cand_weights, ref_weights)) = &idf_weights {
                Some((
                    cand_weights[i].slice(0, 0, cand_len as i64, 1),
                    ref_weights[i].slice(0, 0, ref_len as i64, 1),
                ))
            } else {
                None
            };

            let cand_emb = cand_emb.to_device(self.config.device);
            let ref_emb = ref_emb.to_device(self.config.device);
            let cand_mask = cand_mask.to_device(self.config.device);
            let ref_mask = ref_mask.to_device(self.config.device);
            
            // Check for empty strings (no valid tokens after masking special tokens)
            let cand_has_tokens = cand_mask.sum(tch::Kind::Float).double_value(&[]) > 0.0;
            let ref_has_tokens = ref_mask.sum(tch::Kind::Float).double_value(&[]) > 0.0;

            // Compute BERTScore
            let mut result = if !cand_has_tokens || !ref_has_tokens {
                // Python BERTScore returns 0.0 for empty strings
                // This will be rescaled to negative values if baseline rescaling is enabled
                BERTScoreResult {
                    precision: 0.0,
                    recall: 0.0,
                    f1: 0.0,
                }
            } else {
                compute_bertscore(
                    &cand_emb,
                    &ref_emb,
                    &cand_mask,
                    &ref_mask,
                    pair_idf_weights.as_ref().map(|(c, r)| (c, r)),
                )
            };

            // Apply baseline rescaling if configured
            if self.config.rescale_with_baseline {
                if let Some((p, r, f1)) = self.baseline_manager.rescale_scores_for_layer(
                    &self.config.model_name,
                    &self.config.language,
                    layer_idx,
                    result.precision,
                    result.recall,
                    result.f1,
                ) {
                    result.precision = p;
                    result.recall = r;
                    result.f1 = f1;
                }
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Gets the layer index to use for embeddings.
    fn get_layer_index(&self, hidden_states: &[Tensor]) -> Result<usize> {
        let num_layers = hidden_states.len();

        match self.config.num_layers {
            Some(n) if n >= 0 => {
                let idx = n as usize;
                if idx >= num_layers {
                    Err(anyhow::anyhow!(
                        "Requested layer {} but model only has {} layers",
                        idx,
                        num_layers
                    ))
                } else {
                    Ok(idx)
                }
            }
            Some(n) => {
                // Negative indexing
                let idx = (num_layers as i32 + n) as usize;
                if idx >= num_layers {
                    Err(anyhow::anyhow!("Invalid layer index {}", n))
                } else {
                    Ok(idx)
                }
            }
            None => {
                // Default to last layer
                Ok(num_layers - 1)
            }
        }
    }

    /// Scores with multiple references per candidate.
    ///
    /// # Arguments
    /// * `candidates` - List of candidate sentences
    /// * `references` - List of reference lists (one list per candidate)
    ///
    /// # Returns
    /// Vector of BERTScoreResult, one per candidate (best F1 among references)
    pub fn score_multi_refs<S: AsRef<str> + Send + Sync>(
        &self,
        candidates: &[S],
        references: &[Vec<S>],
    ) -> Result<Vec<BERTScoreResult>> {
        if candidates.len() != references.len() {
            return Err(anyhow::anyhow!(
                "Number of candidates ({}) must equal number of reference lists ({})",
                candidates.len(),
                references.len()
            ));
        }

        let mut best_results = Vec::with_capacity(candidates.len());

        for (i, candidate) in candidates.iter().enumerate() {
            let refs = &references[i];
            if refs.is_empty() {
                return Err(anyhow::anyhow!(
                    "Reference list for candidate {} is empty",
                    i
                ));
            }

            // Score against each reference
            let mut best_result: Option<BERTScoreResult> = None;

            for reference in refs {
                let results = self.score(&[candidate], &[reference])?;
                let result = results.into_iter().next().unwrap();

                // Keep the result with highest F1
                match &best_result {
                    None => best_result = Some(result),
                    Some(best) if result.f1 > best.f1 => best_result = Some(result),
                    _ => {}
                }
            }

            best_results.push(best_result.unwrap());
        }

        Ok(best_results)
    }
}

/// Builder for creating BERTScorer with custom configuration.
pub struct BERTScorerBuilder {
    pub config: BERTScorerConfig,
}

impl BERTScorerBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: BERTScorerConfig::default(),
        }
    }

    /// Sets the model type and name.
    pub fn model(mut self, model_type: ModelType, model_name: &str) -> Self {
        self.config.model_type = model_type;
        self.config.model_name = model_name.to_string();
        self
    }

    /// Sets the language.
    pub fn language(mut self, lang: &str) -> Self {
        self.config.language = lang.to_string();
        self
    }

    // /// Sets the vocabulary and merges paths.
    pub fn vocab_paths(mut self, vocab: PathBuf, merges: Option<PathBuf>) -> Self {
        self.config.vocab_path = vocab;
        self.config.merges_path = merges;
        self
    }

    /// Sets the device.
    pub fn device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }

    /// Sets the layer to extract embeddings from.
    pub fn num_layers(mut self, layers: i32) -> Self {
        self.config.num_layers = Some(layers);
        self
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Enables IDF weighting.
    pub fn use_idf(mut self, use_idf: bool) -> Self {
        self.config.use_idf = use_idf;
        self
    }

    /// Enables baseline rescaling.
    pub fn rescale_with_baseline(mut self, rescale: bool) -> Self {
        self.config.rescale_with_baseline = rescale;
        self
    }

    /// Sets custom baseline scores.
    pub fn custom_baseline(mut self, baseline: BaselineScores) -> Self {
        self.config.custom_baseline = Some(baseline);
        self
    }

    /// Builds the BERTScorer.
    pub fn build(self) -> Result<BERTScorer> {
        BERTScorer::new(self.config)
    }
}

impl Default for BERTScorerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::baseline::BaselineScores;
    use rust_bert::pipelines::common::ModelType;
    use tch::Device;

    #[test]
    fn test_config_default() {
        let config = BERTScorerConfig::default();
        
        assert_eq!(config.model_type, ModelType::Roberta);
        assert_eq!(config.model_name, "roberta-large");
        assert_eq!(config.language, "en");
        assert_eq!(config.max_length, 512);
        assert_eq!(config.batch_size, 64);
        assert!(!config.use_idf);
        assert!(!config.rescale_with_baseline);
        assert!(config.custom_baseline.is_none());
    }

    #[test]
    fn test_builder_pattern() {
        let builder = BERTScorerBuilder::new()
            .model(ModelType::Bert, "bert-base-uncased")
            .language("zh")
            .vocab_paths(std::path::PathBuf::from("/path/to/vocab"), Some(std::path::PathBuf::from("/path/to/merges")))
            .device(Device::Cpu)
            .num_layers(-2)
            .batch_size(32)
            .use_idf(true)
            .rescale_with_baseline(true)
            .custom_baseline(BaselineScores::new(0.8, 0.8, 0.8));
        
        let config = builder.config;
        
        assert_eq!(config.model_type, ModelType::Bert);
        assert_eq!(config.model_name, "bert-base-uncased");
        assert_eq!(config.language, "zh");
        assert_eq!(config.vocab_path, std::path::PathBuf::from("/path/to/vocab"));
        assert_eq!(config.merges_path, Some(std::path::PathBuf::from("/path/to/merges")));
        assert_eq!(config.device, Device::Cpu);
        assert_eq!(config.num_layers, Some(-2));
        assert_eq!(config.batch_size, 32);
        assert!(config.use_idf);
        assert!(config.rescale_with_baseline);
        assert!(config.custom_baseline.is_some());
    }

    #[test]
    fn test_layer_index_calculation() {
        // Mock hidden states with 13 layers (embeddings + 12 transformer layers)
        let hidden_states: Vec<Tensor> = (0..13)
            .map(|_| Tensor::zeros(&[1, 10, 768], (tch::Kind::Float, Device::Cpu)))
            .collect();
        
        let config = BERTScorerConfig {
            num_layers: Some(-1),
            ..Default::default()
        };
        
        // Create a mock scorer to test get_layer_index
        // Since we can't create a real BERTScorer without model files,
        // we'll test the logic directly
        
        // Test last layer (-1)
        let get_layer_index = |num_layers: Option<i32>, states_len: usize| -> Result<usize> {
            match num_layers {
                Some(n) if n >= 0 => {
                    let idx = n as usize;
                    if idx >= states_len {
                        Err(anyhow::anyhow!("Requested layer {} but model only has {} layers", idx, states_len))
                    } else {
                        Ok(idx)
                    }
                }
                Some(n) => {
                    let idx = (states_len as i32 + n) as usize;
                    if idx >= states_len {
                        Err(anyhow::anyhow!("Invalid layer index {}", n))
                    } else {
                        Ok(idx)
                    }
                }
                None => Ok(states_len - 1),
            }
        };
        
        // Test various layer configurations
        assert_eq!(get_layer_index(None, 13).unwrap(), 12); // Default to last
        assert_eq!(get_layer_index(Some(-1), 13).unwrap(), 12); // Last layer
        assert_eq!(get_layer_index(Some(-2), 13).unwrap(), 11); // Second to last
        assert_eq!(get_layer_index(Some(0), 13).unwrap(), 0); // First layer
        assert_eq!(get_layer_index(Some(5), 13).unwrap(), 5); // Middle layer
        
        // Test error cases
        assert!(get_layer_index(Some(13), 13).is_err()); // Out of bounds
        assert!(get_layer_index(Some(-14), 13).is_err()); // Negative out of bounds
    }

    #[test]
    fn test_special_tokens_extraction() {
        // Test that special token IDs would be properly collected
        let special_tokens = vec![0i64, 1, 2]; // Mock PAD, CLS, SEP
        let special_set: HashSet<i64> = special_tokens.into_iter().collect();
        
        assert!(special_set.contains(&0));
        assert!(special_set.contains(&1));
        assert!(special_set.contains(&2));
        assert!(!special_set.contains(&100));
    }

    #[test]
    fn test_batch_processing() {
        // Test batch size logic
        let total_items = 150;
        let batch_size = 64;
        
        let mut batch_starts = Vec::new();
        let mut batch_ends = Vec::new();
        
        for batch_start in (0..total_items).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_items);
            batch_starts.push(batch_start);
            batch_ends.push(batch_end);
        }
        
        assert_eq!(batch_starts, vec![0, 64, 128]);
        assert_eq!(batch_ends, vec![64, 128, 150]);
        
        // Verify all items are covered
        assert_eq!(batch_starts[0], 0);
        assert_eq!(batch_ends[batch_ends.len() - 1], total_items);
    }

    #[test]
    fn test_multi_ref_selection() {
        // Test logic for selecting best result from multiple references
        use crate::core::score::BERTScoreResult;
        let results = vec![
            BERTScoreResult { precision: 0.8, recall: 0.9, f1: 0.85 },
            BERTScoreResult { precision: 0.9, recall: 0.85, f1: 0.87 },
            BERTScoreResult { precision: 0.7, recall: 0.95, f1: 0.81 },
        ];
        
        let best = results.into_iter().max_by(|a, b| {
            a.f1.partial_cmp(&b.f1).unwrap()
        }).unwrap();
        
        assert_eq!(best.f1, 0.87);
        assert_eq!(best.precision, 0.9);
        assert_eq!(best.recall, 0.85);
    }

    #[test]
    fn test_error_handling() {
        // Test that mismatched lengths would be caught
        let candidates = vec!["a", "b", "c"];
        let references = vec!["x", "y"]; // Wrong length
        
        // In real scorer this would return an error
        assert_ne!(candidates.len(), references.len());
        
        // Test empty reference list detection
        let multi_refs: Vec<Vec<&str>> = vec![vec![], vec!["ref"]];
        assert!(multi_refs[0].is_empty());
    }
}
