//! Sentence preprocessing and tokenization module.

use crate::Result;
use rust_bert::pipelines::common::{ModelType, TokenizerOption};
use rust_tokenizers::tokenizer::TruncationStrategy;
use std::path::Path;
use tch::{Device, Tensor};

/// Wrapper for a BERT-family tokenizer, handling tokenization, special tokens, and batching.
pub struct Tokenizer {
    tokenizer: TokenizerOption,
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