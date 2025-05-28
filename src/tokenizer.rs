//! Sentence preprocessing and tokenization module.

use crate::Result;
use rust_bert::pipelines::common::{ModelType, TokenizerOption};
use rust_tokenizers::tokenizer::TruncationStrategy;
use std::path::Path;
use tch::{Device, Tensor};

/// Wrapper for a BERT-family tokenizer, handling tokenization, special tokens, and batching.
pub struct Tokenizer {
    pub(crate) tokenizer: TokenizerOption,
    max_len: usize,
    truncation_strategy: TruncationStrategy,
    stride: usize,
}

/// Batched tokenization outputs, including padded tensors and raw token ID lists.
pub struct EncodingResult {
    /// Batched input IDs tensor of shape (batch_size, seq_len).
    pub input_ids: Tensor,
    /// Batched attention mask tensor of shape (batch_size, seq_len).
    pub attention_mask: Tensor,
    /// Batched token type IDs tensor of shape (batch_size, seq_len).
    pub token_type_ids: Tensor,
    /// Raw token IDs per sequence (unpadded), including special tokens.
    pub token_ids: Vec<Vec<i64>>,
    /// Raw segment IDs per sequence (unpadded), typically all zeros for single sequences.
    pub segment_ids: Vec<Vec<i64>>,
    /// True length of each sequence (number of tokens before padding).
    pub lengths: Vec<usize>,
}

impl Tokenizer {
    /// Creates a new tokenizer for the specified model.
    ///
    /// # Arguments
    /// * `model`: the BERT-family model type (e.g., ModelType::Roberta).
    /// * `vocab_path`: path to the vocabulary file.
    /// * `merges_path`: optional path to merges file for BPE-based tokenizers.
    /// * `lower_case`: whether to lowercase input text.
    /// * `strip_accents`: whether to strip accents; if `None`, defaults to `lower_case` behavior.
    /// * `add_prefix_space`: whether to add a prefix space (for some tokenizers); if `None`, defaults per model.
    /// * `max_len`: maximum token length for truncation/padding.
    /// * `truncation_strategy`: strategy for handling over-long sequences.
    /// * `stride`: stride size for overflowing tokens (typically 0 for single sentences).
    pub fn new<P: AsRef<Path>>(
        model: ModelType,
        vocab_path: P,
        merges_path: Option<P>,
        lower_case: bool,
        strip_accents: Option<bool>,
        add_prefix_space: Option<bool>,
        max_len: usize,
        truncation_strategy: TruncationStrategy,
        stride: usize,
    ) -> Result<Self> {
        let vocab = vocab_path.as_ref().to_string_lossy();
        let merges = merges_path.as_ref().map(|p| p.as_ref().to_string_lossy());
        let tokenizer = TokenizerOption::from_file(
            model,
            &vocab,
            merges.as_ref().map(|s| s.as_ref()),
            lower_case,
            strip_accents,
            add_prefix_space,
        )?;
        Ok(Tokenizer {
            tokenizer,
            max_len,
            truncation_strategy,
            stride,
        })
    }

    /// Tokenizes a batch of input texts, returning padded tensors on the given device
    /// and raw token/segment ID lists for downstream processing.
    pub fn encode<S: AsRef<str> + Send + Sync>(
        &self,
        texts: &[S],
        device: Device,
    ) -> EncodingResult {
        let inputs = self
            .tokenizer
            .encode_list(texts, self.max_len, &self.truncation_strategy, self.stride);
        let pad_id = self.tokenizer.get_pad_id().unwrap_or(0);
        let mut token_ids = Vec::with_capacity(inputs.len());
        let mut segment_ids = Vec::with_capacity(inputs.len());
        let mut lengths = Vec::with_capacity(inputs.len());
        let max_len = inputs.iter().map(|i| i.token_ids.len()).max().unwrap_or(0);
        let mut id_batches = Vec::with_capacity(inputs.len());
        let mut mask_batches = Vec::with_capacity(inputs.len());
        let mut type_batches = Vec::with_capacity(inputs.len());
        for input in inputs {
            let seq_len = input.token_ids.len();
            lengths.push(seq_len);
            token_ids.push(input.token_ids.clone());
            let raw_segment: Vec<i64> = input.segment_ids.iter().map(|&s| s as i64).collect();
            segment_ids.push(raw_segment.clone());
            let mut ids = input.token_ids;
            let mut mask = vec![1; seq_len];
            let mut types = raw_segment;
            ids.resize(max_len, pad_id);
            mask.resize(max_len, 0);
            types.resize(max_len, 0);
            id_batches.push(ids);
            mask_batches.push(mask);
            type_batches.push(types);
        }
        let input_ids = Tensor::stack(
            &id_batches
                .iter()
                .map(|ids| Tensor::from_slice(ids))
                .collect::<Vec<_>>(),
            0,
        )
        .to_device(device);
        let attention_mask = Tensor::stack(
            &mask_batches
                .iter()
                .map(|mask| Tensor::from_slice(mask))
                .collect::<Vec<_>>(),
            0,
        )
        .to_device(device);
        let token_type_ids = Tensor::stack(
            &type_batches
                .iter()
                .map(|types| Tensor::from_slice(types))
                .collect::<Vec<_>>(),
            0,
        )
        .to_device(device);
        EncodingResult {
            input_ids,
            attention_mask,
            token_type_ids,
            token_ids,
            segment_ids,
            lengths,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_tokenizers::tokenizer::TruncationStrategy;
    use tch::Device;

    #[test]
    fn test_encoding_result_structure() {
        // Test that EncodingResult maintains consistent shapes
        let batch_size = 3;
        let seq_len = 10;
        
        let input_ids = Tensor::zeros(&[batch_size, seq_len], (tch::Kind::Int64, Device::Cpu));
        let attention_mask = Tensor::ones(&[batch_size, seq_len], (tch::Kind::Int64, Device::Cpu));
        let token_type_ids = Tensor::zeros(&[batch_size, seq_len], (tch::Kind::Int64, Device::Cpu));
        
        let result = EncodingResult {
            input_ids: input_ids.shallow_clone(),
            attention_mask: attention_mask.shallow_clone(),
            token_type_ids: token_type_ids.shallow_clone(),
            token_ids: vec![vec![101, 2054, 102], vec![101, 2023, 102], vec![101, 102]],
            segment_ids: vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0]],
            lengths: vec![3, 3, 2],
        };
        
        assert_eq!(result.input_ids.size(), vec![batch_size, seq_len]);
        assert_eq!(result.attention_mask.size(), vec![batch_size, seq_len]);
        assert_eq!(result.token_type_ids.size(), vec![batch_size, seq_len]);
        assert_eq!(result.token_ids.len(), batch_size as usize);
        assert_eq!(result.segment_ids.len(), batch_size as usize);
        assert_eq!(result.lengths.len(), batch_size as usize);
    }

    #[test]
    fn test_truncation_strategies() {
        // Test that all expected truncation strategies are valid
        let strategies = vec![
            TruncationStrategy::LongestFirst,
            TruncationStrategy::OnlyFirst,
            TruncationStrategy::OnlySecond,
            TruncationStrategy::DoNotTruncate,
        ];
        
        // Just verify they exist and can be used
        for strategy in strategies {
            match strategy {
                TruncationStrategy::LongestFirst => assert!(true),
                TruncationStrategy::OnlyFirst => assert!(true),
                TruncationStrategy::OnlySecond => assert!(true),
                TruncationStrategy::DoNotTruncate => assert!(true),
            }
        }
    }

    #[test]
    fn test_padding_logic() {
        // Test padding logic with mock data
        let sequences = vec![
            vec![101i64, 2054, 102],      // len=3
            vec![101, 2023, 2003, 102],   // len=4
            vec![101, 102],                // len=2
        ];
        
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap();
        assert_eq!(max_len, 4);
        
        let pad_id = 0i64;
        let mut padded_sequences = Vec::new();
        
        for seq in sequences {
            let mut padded = seq.clone();
            padded.resize(max_len, pad_id);
            padded_sequences.push(padded);
        }
        
        assert_eq!(padded_sequences[0], vec![101, 2054, 102, 0]);
        assert_eq!(padded_sequences[1], vec![101, 2023, 2003, 102]);
        assert_eq!(padded_sequences[2], vec![101, 102, 0, 0]);
    }

    #[test]
    fn test_attention_mask_generation() {
        // Test attention mask generation logic
        let sequences = vec![
            vec![101i64, 2054, 102],      // len=3
            vec![101, 2023, 2003, 102],   // len=4
            vec![101, 102],                // len=2
        ];
        
        let max_len = 4;
        let mut attention_masks = Vec::new();
        
        for seq in &sequences {
            let seq_len = seq.len();
            let mut mask = vec![1i64; seq_len];
            mask.resize(max_len, 0);
            attention_masks.push(mask);
        }
        
        assert_eq!(attention_masks[0], vec![1, 1, 1, 0]);
        assert_eq!(attention_masks[1], vec![1, 1, 1, 1]);
        assert_eq!(attention_masks[2], vec![1, 1, 0, 0]);
    }

    #[test]
    fn test_model_type_support() {
        // Verify that we support the expected model types
        use rust_bert::pipelines::common::ModelType;
        
        let supported_models = vec![
            ModelType::Bert,
            ModelType::DistilBert,
            ModelType::Roberta,
            ModelType::Albert,
            ModelType::XLMRoberta,
            ModelType::Electra,
            ModelType::Marian,
            ModelType::T5,
            ModelType::GPT2,
            ModelType::OpenAiGpt,
            ModelType::XLNet,
            ModelType::Reformer,
            ModelType::ProphetNet,
            ModelType::Longformer,
            ModelType::MBart,
            ModelType::M2M100,
            ModelType::FNet,
            ModelType::Deberta,
            ModelType::DebertaV2,
        ];
        
        // Just verify these model types exist
        assert!(!supported_models.is_empty());
    }
}