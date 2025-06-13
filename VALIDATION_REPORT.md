# Rust-BERTScore Validation Report

## Executive Summary

We successfully built and ran the Rust implementation of BERTScore, but validation against the Python reference implementation revealed significant discrepancies that need to be addressed before the implementations can be considered equivalent.

## Current Status: ❌ VALIDATION FAILED

### Key Metrics
- **Maximum Difference**: 1.088 (tolerance: 0.0001)  
- **Correlation**: 0.62-0.67 (target: >0.99999)
- **Failure Rate**: 100% of test cases exceed tolerance

## Infrastructure Achievements ✅

1. **Enhanced CLI with TSV/CSV Support**
   - Added `--input-tsv` flag for batch processing
   - Added `--output-csv` for structured output
   - Added `--model-name` parameter for model specification

2. **Python Environment Setup**
   - Implemented uv dependency groups (production vs benchmark)
   - Consolidated to single `pyproject.toml` in root
   - Successfully runs Python bert-score 0.3.13

3. **PyTorch/libtorch Configuration**
   - Downloaded libtorch 2.4.0 for tch 0.17.0 compatibility
   - Resolved library linking issues
   - Rust CLI successfully runs with CPU computation

4. **Validation Pipeline**
   - Automated test data generation
   - Statistical comparison framework
   - Detailed diagnostic tools

## Critical Issues Identified 🔴

### 1. Whitespace Handling
**Test Case**: "   Leading and trailing spaces   " vs "Leading and trailing spaces"
- Python: F1 = 1.0000 (perfect match)
- Rust: F1 = 0.2209 (poor match)
- **Hypothesis**: Different tokenization or special token handling

### 2. Short Text Anomalies  
**Test Cases**: "OK" vs "Okay", "Yes" vs "No"
- Python: F1 > 0.96 (high similarity)
- Rust: F1 < 0.28 (low similarity)
- **Hypothesis**: Baseline rescaling or IDF weighting issues

### 3. Negative Scores
- Python: 2/21 cases with negative scores
- Rust: 6/21 cases with negative scores
- **Hypothesis**: Baseline rescaling parameters differ

### 4. Model Configuration
- Different layer selection or aggregation
- Special token handling inconsistencies
- Possible model loading differences

## Root Cause Analysis

### Most Likely Causes (Priority Order)

1. **Baseline Rescaling Mismatch** (HIGH)
   - Rust may use different baseline values
   - Rescaling formula implementation differences
   - Edge case handling for extreme scores

2. **Tokenization Differences** (HIGH)
   - Whitespace preprocessing
   - Special token insertion/handling
   - Subword tokenization alignment

3. **IDF Computation** (MEDIUM)
   - Document frequency calculation
   - Smoothing parameters
   - Special token weighting

4. **Model Layer Selection** (MEDIUM)
   - Default layer differs between implementations
   - Aggregation method for embeddings

## Recommended Next Steps

### Phase 1: Isolate Issues (1-2 days)
1. Run without baseline rescaling (`--no-baseline`)
2. Run without IDF weighting (`--no-idf`)
3. Compare raw similarity scores
4. Test with simple examples

### Phase 2: Debug Components (3-5 days)
1. Log tokenization outputs for problematic cases
2. Compare baseline rescaling values
3. Verify model layer extraction
4. Check numerical precision

### Phase 3: Fix Implementation (1 week)
1. Align tokenization preprocessing
2. Match baseline parameters exactly
3. Ensure consistent special token handling
4. Verify all hyperparameters match

### Phase 4: Extended Validation (3 days)
1. Test on WMT16 dataset
2. Multi-language validation
3. Performance benchmarking
4. Edge case coverage

## Technical Details

### Environment
- **Python**: PyTorch 2.7.1, bert-score 0.3.13
- **Rust**: tch 0.17.0, libtorch 2.4.0
- **Model**: roberta-large
- **Settings**: IDF=true, baseline=true

### Test Dataset
- 21 carefully crafted test cases
- Includes: whitespace, punctuation, case differences, short/long texts
- Source: `data/benchmark/direct_eval_pairs.tsv`

### Validation Scripts
```bash
# Generate test data
uv run --group benchmark python scripts/benchmark/make_direct_set.py

# Run Python reference
uv run --group benchmark python scripts/benchmark/run_direct_py.py

# Run Rust implementation  
export LIBTORCH="/path/to/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
./target/release/bert-score score \
  --input-tsv data/benchmark/direct_eval_pairs.tsv \
  --output-csv reports/direct_scores_rust.csv \
  --pretrained roberta-large \
  --model-type roberta \
  --idf --baseline

# Compare results
uv run --group benchmark python scripts/benchmark/compare_direct.py
```

## Conclusion

While we've successfully created the infrastructure for comprehensive validation, the Rust implementation currently produces significantly different results than the Python reference. The differences are too large to attribute to numerical precision alone and indicate fundamental implementation differences that must be resolved.

**Current Recommendation**: Do not use rust-bert-score as a drop-in replacement for Python bert-score until these issues are resolved.

---
*Generated: December 2024*  
*Next Review: After implementation fixes*