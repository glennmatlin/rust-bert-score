//! Sentence preprocessing and tokenization module.

use std::path::PathBuf;

use crate::Result;
use anyhow::Error;
use rust_bert::pipelines::common::{ModelType, TokenizerOption};
use crate::cli::types::TokenizerSpec;
use rust_tokenizers::tokenizer::TruncationStrategy;
use tch::{Device, Tensor};

use super::api::fetch_vocab_files;

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

/// Macro to create mutable vectors with a specified capacity for batched tokenization.
macro_rules! with_size_mut {
    ($cap:expr, $($name:ident), +) => {

        $(let mut $name = Vec::with_capacity($cap);)+

    };
}

/// Pads a vector in place to ensure it has at least `target_len` elements, using `pad_value`.
/// If the vector is already longer than `target_len`, it remains unchanged.
#[inline(always)]
fn pad_vec_mut<T: Clone>(vec: &mut Vec<T>, pad_value: T, target_len: usize) -> &Vec<T> {
    if vec.len() < target_len {
        vec.resize(target_len, pad_value);
    }
    vec
}

/// Pads a vector to ensure it has at least `target_len` elements, using `pad_value`.
/// If the vector is already longer than `target_len`, it remains unchanged.
#[inline(always)]
fn pad_vec<T: Clone>(vec: &[T], pad_value: T, target_len: usize) -> Vec<T> {
    let mut padded = vec.to_owned();
    if padded.len() < target_len {
        padded.resize(target_len, pad_value);
    }
    padded
}

/// Converts a list of vectors into a single batched tensor on the specified device.
/// Each vector in `vecs` should have the same length, and they will be stacked along a new dimension.
#[inline(always)]
fn to_batched_tensor<T: Copy + tch::kind::Element>(vecs: &[Vec<T>], device: Device) -> Tensor {
    Tensor::stack(
        &vecs
            .iter()
            .map(|v| Tensor::from_slice(v))
            .collect::<Vec<_>>(),
        0,
    )
    .to_device(device)
}

pub struct TokenizerArgs {
    pub model: ModelType,
    pub vocab_path: PathBuf,
    pub merges_path: Option<PathBuf>,
    pub lower_case: bool,
    pub strip_accents: Option<bool>,
    pub add_prefix_space: Option<bool>,
    pub max_len: usize,
    pub truncation_strategy: TruncationStrategy,
    pub stride: usize,
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
    pub fn new(
        args : TokenizerArgs,
    ) -> Result<Self> {
        let vocab = args.vocab_path;
        let tokenizer = TokenizerOption::from_file(
            args.model,
            vocab.to_str().ok_or(Error::msg("Invalid vocab path"))?,
            args.merges_path
                .as_ref()
                .map(|p| p.to_str().ok_or(Error::msg("Invalid merges path")))
                .transpose()?,
            args.lower_case,
            args.strip_accents,
            args.add_prefix_space,
        )?;
        Ok(Tokenizer {
            tokenizer,
            max_len: args.max_len,
            truncation_strategy: args.truncation_strategy,
            stride: args.stride,
        })
    }

    /// Tokenizes a batch of input texts, returning padded tensors on the given device
    /// and raw token/segment ID lists for downstream processing.
    pub fn encode<S: AsRef<str> + Send + Sync>(
        &self,
        texts: &[S],
        device: Device,
    ) -> EncodingResult {
        let inputs =
            self.tokenizer
                .encode_list(texts, self.max_len, &self.truncation_strategy, self.stride);
        let pad_id = self.tokenizer.get_pad_id().unwrap_or(0);

        let max_len = inputs.iter().map(|i| i.token_ids.len()).max().unwrap_or(0);

        with_size_mut!(
            inputs.len(),
            token_ids,
            segment_ids,
            lengths,
            id_batches,
            mask_batches,
            type_batches
        );

        for input in inputs {
            let seq_len = input.token_ids.len();

            let raw_segment: Vec<i64> = input.segment_ids.iter().map(|&s| s as i64).collect();

            let ids = pad_vec(&input.token_ids, pad_id, max_len);
            let mut mask = vec![1; seq_len];
            pad_vec_mut(&mut mask, 0, max_len);
            let types = pad_vec(&raw_segment, 0, max_len);

            token_ids.push(input.token_ids.clone());
            segment_ids.push(raw_segment.clone());
            lengths.push(seq_len);
            id_batches.push(ids);
            mask_batches.push(mask);
            type_batches.push(types);
        }

        let input_ids = to_batched_tensor(&id_batches, device);
        let attention_mask = to_batched_tensor(&mask_batches, device);
        let token_type_ids = to_batched_tensor(&type_batches, device);

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

pub fn choose_hf_or_pathed_tokenizer(
    spec : &TokenizerSpec,
) -> Result<(PathBuf, Option<PathBuf>)> {

    // If a pretrained tokenizer is specified, fetch vocab files instead of using local files
    if let Some(pretrained) = &spec.pretrained {
        // Fetch vocab files from Hugging Face
        let (vocab, merges) = fetch_vocab_files(pretrained)?;
        Ok((vocab, merges))
    } else if let Some(vocab) = &spec.vocab {
        // Use custom vocab file
        Ok((PathBuf::from(vocab), spec.merges.as_ref().map(PathBuf::from)))
    } else {
        Err(anyhow::anyhow!(
            "Either a pretrained tokenizer or a vocabulary file must be specified"
        ))
    }

}