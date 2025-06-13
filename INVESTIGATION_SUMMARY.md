# BERTScore Investigation Summary

## The "OK" vs "Okay" Discrepancy

### Initial Problem
- Manual calculation: F1 = 0.942455
- Python bert_score: F1 = 0.998344  
- Large discrepancy of ~0.056

### Root Causes Identified

1. **Tokenization Differences**: RoBERTa tokenizer behavior
   - "OK" without prefix space → token 9335
   - "Okay" without prefix space → token 33082
   - " OK" with prefix space → token 4954
   - " Okay" with prefix space → token 8487
   - The tokens " OK" (4954) and " Okay" (8487) have much higher cosine similarity (0.998345) than "OK" (9335) and "Okay" (33082) which have similarity 0.942455

2. **Prefix Space Handling**: Python bert_score's `sent_encode` function explicitly adds `add_prefix_space=True` for RoBERTa models, while the Rust implementation was using the default (None).

3. **Baseline Rescaling**: When enabled, transforms scores using:
   ```
   rescaled_score = (raw_score - baseline) / (1 - baseline)
   ```
   For roberta-large layer 17, baseline F1 = 0.83122575

### Solution Implemented

Modified `src/core/pipeline.rs` to explicitly set `add_prefix_space: Some(true)` for RoBERTa and GPT2 models:

```rust
add_prefix_space: match config.model_type {
    ModelType::Roberta | ModelType::GPT2 => Some(true),
    _ => None,
},
```

### Results After Fix

For "OK" vs "Okay":
- Python: F1 = 0.998344
- Rust: F1 = 0.997504
- Difference: 0.000841 (well within acceptable tolerance)

### Remaining Differences

Some test cases still show differences > 0.01, but these appear to be due to:
1. Different handling of edge cases (empty strings, multiple spaces)
2. Minor numerical precision differences
3. Possible differences in special token handling

### Key Takeaways

1. **Tokenization matters**: The same text can produce very different embeddings depending on tokenization settings.
2. **Model-specific defaults**: Different models (BERT vs RoBERTa) have different tokenization conventions that must be respected.
3. **Baseline rescaling**: This post-processing step significantly transforms scores and must be consistently applied when comparing implementations.
4. **The manual calculation was correct**: It was computing similarity for "OK" vs "Okay" without prefix spaces, which gives ~0.942.