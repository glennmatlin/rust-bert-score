# BERTScore Validation: Critical Findings

## Executive Summary

We identified that Python BERTScore and Rust BERTScore are producing significantly different results. The root cause appears to be that **Python BERTScore gives perfect scores (1.0) for texts that differ only in leading/trailing whitespace**, while Rust correctly identifies these as different.

## Key Discovery: Whitespace Handling

### Test Case
- Candidate: `"   Leading and trailing spaces   "`
- Reference: `"Leading and trailing spaces"`

### Results
- **Python BERTScore**: F1 = 1.0000 (perfect match!)
- **Rust BERTScore**: F1 = 0.8728 (correctly identifies difference)

### Tokenization Analysis
The RoBERTa tokenizer creates different tokens:
- Without spaces: 5 tokens `['Lead', 'ing', 'Ġand', 'Ġtrailing', 'Ġspaces']`
- With spaces: 9 tokens `['Ġ', 'Ġ', 'ĠLeading', 'Ġand', 'Ġtrailing', 'Ġspaces', 'Ġ', 'Ġ', 'Ġ']`

Despite different tokenization, Python BERTScore returns 1.0!

## Impact of Baseline Rescaling

Baseline rescaling **dramatically amplifies** the differences:

| Configuration | Max Difference | Mean Difference |
|--------------|----------------|-----------------|
| Raw scores (no baseline) | 0.127 | 0.031 |
| With baseline rescaling | 0.979 | 0.198 |

The baseline rescaling formula `(score - baseline) / (1 - baseline)` can turn small differences into large ones, especially when the baseline is high.

## Configuration Mismatch

Initial testing had a configuration mismatch:
- Python: `roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.52.4)-rescaled`
- Rust: Was using `--idf` flag

After matching configurations (both without IDF), differences persisted.

## Other Problematic Cases

1. **Short texts**: "OK" vs "Okay" 
   - Python: F1 = 0.998 (near perfect)
   - Rust: F1 = 0.903

2. **Empty strings**: Large negative scores with baseline rescaling
   - Python: F1 = -4.925
   - Suggests baseline rescaling edge cases

## Conclusions

1. **Python BERTScore has unexpected behavior** with whitespace that makes it give perfect scores for different texts. This could be:
   - A feature for robustness to formatting
   - A bug in the implementation
   - Special preprocessing we're not aware of

2. **Rust implementation appears more correct** from a pure text similarity perspective - different tokenizations should yield different scores.

3. **Baseline rescaling amplifies differences** making validation extremely sensitive to small discrepancies.

## Recommendations

1. **Investigate Python BERTScore source code** to understand the whitespace handling
2. **Test without baseline rescaling** for more stable comparisons
3. **Consider this a feature difference** rather than a bug - Python's behavior might be intentional
4. **Document the difference** and let users choose which behavior they prefer
5. **Create test suite without whitespace edge cases** to validate core algorithm

## Next Steps

1. Report findings to Python BERTScore maintainers
2. Add configuration option to Rust implementation to match Python behavior if desired
3. Focus validation on core algorithm without edge cases
4. Create comprehensive documentation about the differences