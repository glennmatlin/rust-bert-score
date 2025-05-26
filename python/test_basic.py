#!/usr/bin/env python3
"""
Basic test of the Python bindings.

Note: This requires the module to be built with maturin first:
    maturin develop --features python
"""

import numpy as np

try:
    from rust_bert_score import compute_bertscore_from_embeddings, BaselineManager
    print("✓ Successfully imported rust_bert_score")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("  Build with: maturin develop --features python")
    exit(1)


def test_compute_from_embeddings():
    """Test computing BERTScore from pre-computed embeddings."""
    print("\n1. Testing compute_bertscore_from_embeddings:")
    
    # Create simple test embeddings
    candidate_embeddings = np.array([
        [0.9, 0.1, 0.0],  # Token 1
        [0.1, 0.9, 0.0],  # Token 2
        [0.0, 0.1, 0.9],  # Token 3
    ], dtype=np.float32)
    
    reference_embeddings = np.array([
        [0.8, 0.2, 0.0],  # Token 1
        [0.1, 0.9, 0.0],  # Token 2 (exact match)
        [0.0, 0.0, 1.0],  # Token 3
        [0.0, 0.2, 0.8],  # Token 4 (similar to candidate token 3)
    ], dtype=np.float32)
    
    result = compute_bertscore_from_embeddings(
        candidate_embeddings,
        reference_embeddings,
    )
    
    print(f"  Precision: {result.precision:.3f}")
    print(f"  Recall:    {result.recall:.3f}")
    print(f"  F1:        {result.f1:.3f}")
    print(f"  Result repr: {result}")
    
    # Test with masks
    print("\n2. Testing with masks:")
    candidate_mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # Mask out token 3
    reference_mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)  # Mask out token 4
    
    result_masked = compute_bertscore_from_embeddings(
        candidate_embeddings,
        reference_embeddings,
        candidate_mask=candidate_mask,
        reference_mask=reference_mask,
    )
    
    print(f"  Precision: {result_masked.precision:.3f}")
    print(f"  Recall:    {result_masked.recall:.3f}")
    print(f"  F1:        {result_masked.f1:.3f}")


def test_baseline_manager():
    """Test baseline manager functionality."""
    print("\n3. Testing BaselineManager:")
    
    # Create manager with defaults
    manager = BaselineManager.with_defaults()
    
    # Add custom baseline
    manager.add_baseline("test-model", "en", 0.8, 0.8, 0.8)
    
    # Test rescaling
    rescaled = manager.rescale_scores("test-model", "en", 0.9, 0.85, 0.875)
    if rescaled:
        p, r, f1 = rescaled
        print(f"  Raw scores:      P=0.900, R=0.850, F1=0.875")
        print(f"  Rescaled scores: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    else:
        print("  No baseline found")
    
    # Test with default baseline
    rescaled_default = manager.rescale_scores("roberta-large", "en", 0.95, 0.93, 0.94)
    if rescaled_default:
        p, r, f1 = rescaled_default
        print(f"\n  RoBERTa-large baseline rescaling:")
        print(f"  Raw scores:      P=0.950, R=0.930, F1=0.940")
        print(f"  Rescaled scores: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")


def test_idf_weights():
    """Test IDF weighting in score computation."""
    print("\n4. Testing IDF weighting:")
    
    # Create embeddings
    embeddings = np.random.randn(5, 10).astype(np.float32)
    
    # Create IDF weights (common words get low weight, rare words get high weight)
    idf_weights = np.array([0.1, 0.1, 0.5, 0.8, 0.9], dtype=np.float32)
    
    result_weighted = compute_bertscore_from_embeddings(
        embeddings,
        embeddings,  # Compare with itself for simplicity
        candidate_idf_weights=idf_weights,
        reference_idf_weights=idf_weights,
    )
    
    print(f"  With IDF weights: P={result_weighted.precision:.3f}, "
          f"R={result_weighted.recall:.3f}, F1={result_weighted.f1:.3f}")
    
    # Without IDF weights should give perfect scores (comparing with itself)
    result_unweighted = compute_bertscore_from_embeddings(
        embeddings,
        embeddings,
    )
    
    print(f"  Without weights:  P={result_unweighted.precision:.3f}, "
          f"R={result_unweighted.recall:.3f}, F1={result_unweighted.f1:.3f}")


if __name__ == "__main__":
    print("=== Rust BERTScore Python Bindings Test ===")
    
    test_compute_from_embeddings()
    test_baseline_manager()
    test_idf_weights()
    
    print("\n✓ All tests completed successfully!")
    print("\nNote: Full BERTScorer test requires vocab files and would download models.")