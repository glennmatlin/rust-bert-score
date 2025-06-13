# Testing and Validation Plan for rust-bert-score

## Current Status vs. Requirements

### ✅ Completed from AGENT.md & PLAN.md:

1. **Model and Tokenizer Setup** (Step 1)
   - ✅ Multi-model support (BERT, DistilBERT, RoBERTa, DeBERTa)
   - ✅ Tokenizer integration with rust-bert
   - ✅ Device placement (CPU/GPU)

2. **Sentence Preprocessing** (Step 2)
   - ✅ Tokenization with special tokens
   - ✅ Attention masks and token type IDs
   - ✅ Batch processing support

3. **Embedding Extraction** (Step 3)
   - ✅ Layer selection (configurable)
   - ✅ Hidden states extraction
   - ✅ Special token handling

4. **Cosine Similarity Calculation** (Step 4)
   - ✅ L2 normalization
   - ✅ Pairwise similarity matrix
   - ✅ Greedy token matching

5. **Precision, Recall, F1 Calculation** (Step 5)
   - ✅ Proper averaging formulas
   - ✅ Multi-reference support
   - ✅ Edge case handling

6. **IDF Weighting** (Step 6)
   - ✅ Document frequency computation
   - ✅ Set semantics implementation
   - ✅ Precomputed dictionary support

7. **Baseline Rescaling** (Step 7)
   - ✅ Rescaling formula implementation
   - ✅ Multi-model/language support
   - ✅ TSV file loading capability

8. **Testing and Validation** (Step 8)
   - ✅ Unit tests for each component
   - ✅ Integration tests
   - ⚠️ Need validation against Python BERTScore

9. **Python Bindings** (Step 9)
   - ✅ PyO3 infrastructure
   - ⚠️ Need to build and test

## Required Testing & Validation

### 1. Model Download and Integration Testing
```bash
# Test with actual model files
# Need to obtain:
# - Vocabulary files (vocab.txt for BERT, vocab.json for RoBERTa)
# - Merges files (merges.txt for RoBERTa/GPT models)
# - Model will auto-download via rust-bert
```

### 2. Numerical Validation Against Python BERTScore
```python
# Compare outputs with original implementation
from bert_score import score as python_score
import rust_bert_score

# Test cases
candidates = [
    "The cat sat on the mat.",
    "A dog ran in the park.",
    "The weather is nice today."
]
references = [
    "The cat was sitting on the mat.",
    "A dog was running in the park.", 
    "Today the weather is nice."
]

# Python BERTScore
P_py, R_py, F1_py = python_score(candidates, references, lang='en')

# Rust BERTScore (after building)
scorer = rust_bert_score.BERTScore(...)
P_rs, R_rs, F1_rs = scorer.score(candidates, references)

# Compare results (should be within floating point tolerance)
assert all(abs(p1 - p2) < 0.001 for p1, p2 in zip(P_py, P_rs))
```

### 3. Performance Benchmarking
```rust
// Measure performance improvements
// - Tokenization speed
// - Model inference time
// - Similarity computation time
// - End-to-end scoring time
// Compare with Python implementation on various dataset sizes
```

### 4. Edge Case Testing
- Empty strings
- Very long sequences (> model max length)
- Special characters and Unicode
- Single token sentences
- Identical candidate/reference pairs

### 5. Memory Usage Testing
- Monitor memory consumption with different batch sizes
- Test GPU memory limits
- Verify proper cleanup after scoring

## Next Steps for Full Validation

### Step 1: Obtain Test Resources
```bash
# Download test vocabulary files
mkdir -p test_resources
cd test_resources

# For BERT
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt

# For RoBERTa
wget https://huggingface.co/roberta-large/resolve/main/vocab.json
wget https://huggingface.co/roberta-large/resolve/main/merges.txt
```

### Step 2: Create Validation Script
```rust
// tests/validation_test.rs
#[test]
#[ignore] // Requires model files
fn test_against_reference_scores() {
    // Load known good scores from Python BERTScore
    let reference_scores = load_reference_scores("test_data/reference_scores.json");
    
    // Run our implementation
    let scorer = BERTScorer::new(config)?;
    let results = scorer.score(&candidates, &references)?;
    
    // Compare within tolerance
    for (result, reference) in results.iter().zip(reference_scores.iter()) {
        assert!((result.precision - reference.precision).abs() < 0.001);
        assert!((result.recall - reference.recall).abs() < 0.001);
        assert!((result.f1 - reference.f1).abs() < 0.001);
    }
}
```

### Step 3: Python Binding Validation
```bash
# Build Python bindings
./build.sh

# Run Python tests
cd python
python test_basic.py

# Test against real models
python test_with_models.py
```

### Step 4: Create Comprehensive Test Suite
1. **Correctness Tests**
   - Validate tokenization matches HuggingFace
   - Verify embedding extraction
   - Check similarity computation accuracy
   - Confirm score calculations

2. **Performance Tests**
   - Batch size optimization
   - GPU vs CPU performance
   - Memory efficiency
   - Scaling with dataset size

3. **Compatibility Tests**
   - Different model architectures
   - Various languages
   - Multiple Python versions
   - Different PyTorch versions

## Known Limitations to Address

1. **Model Files**: Need actual vocabulary files to run full tests
2. **Baseline Data**: Currently using example values, need real baseline files
3. **GPU Testing**: Requires CUDA-capable device for full validation
4. **Large-scale Testing**: Need larger datasets for performance validation

## Validation Checklist

- [ ] Download vocabulary files for test models
- [ ] Create reference score dataset using Python BERTScore
- [ ] Implement numerical comparison tests
- [ ] Run performance benchmarks
- [ ] Test all supported model types
- [ ] Validate IDF computation against Python
- [ ] Test baseline rescaling with real baselines
- [ ] Build and test Python bindings
- [ ] Create end-to-end integration tests
- [ ] Document any discrepancies or limitations

## Expected Outcomes

After completing validation:
1. Numerical accuracy within 0.001 of Python implementation
2. Performance improvement of 2-5x over Python
3. Memory usage reduction of 30-50%
4. Full compatibility with existing BERTScore workflows