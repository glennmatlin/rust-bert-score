#!/usr/bin/env python3
"""
Analyze how Python BERTScore handles whitespace and edge cases.
"""

import bert_score as bs
import torch
from transformers import AutoTokenizer

def test_whitespace_handling():
    """Test how BERTScore handles whitespace."""
    print("🔍 Testing Whitespace Handling in Python BERTScore")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("   Leading and trailing spaces   ", "Leading and trailing spaces"),
        ("Hello world", "Hello world"),
        ("Hello  world", "Hello world"),  # Double space
        ("Hello\tworld", "Hello world"),  # Tab
        ("Hello\nworld", "Hello world"),  # Newline
        ("", "Empty test"),  # Empty string
        ("   ", "Just spaces"),  # Only spaces
    ]
    
    # Test with and without baseline
    for use_baseline in [False, True]:
        print(f"\n{'With' if use_baseline else 'Without'} baseline rescaling:")
        print("-" * 40)
        
        for cand, ref in test_cases:
            result = bs.score(
                [cand], [ref],
                model_type="roberta-large",
                lang="en",
                rescale_with_baseline=use_baseline,
                verbose=False,
                return_hash=False
            )
            
            P, R, F1 = result
            print(f"Cand: '{cand}' | Ref: '{ref}'")
            print(f"  P={P[0]:.4f}, R={R[0]:.4f}, F1={F1[0]:.4f}")

def analyze_tokenization():
    """Analyze how the tokenizer handles whitespace."""
    print("\n\n🔍 Tokenization Analysis")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    
    test_strings = [
        "Leading and trailing spaces",
        "   Leading and trailing spaces   ",
        "OK",
        "Okay",
    ]
    
    for text in test_strings:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"\nText: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: '{decoded}'")
        print(f"Length: {len(tokens)} tokens")

def test_perfect_matches():
    """Test what gets perfect scores."""
    print("\n\n🔍 Perfect Score Analysis")
    print("=" * 60)
    
    test_pairs = [
        ("Hello world", "Hello world"),
        ("   Hello world   ", "   Hello world   "),
        ("   Hello world   ", "Hello world"),
        ("Hello world", "   Hello world   "),
    ]
    
    for cand, ref in test_pairs:
        P, R, F1 = bs.score(
            [cand], [ref],
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=False,
            verbose=False
        )
        
        print(f"'{cand}' vs '{ref}'")
        print(f"  Raw scores: P={P[0]:.6f}, R={R[0]:.6f}, F1={F1[0]:.6f}")
        
        # Check if it's a perfect match
        if F1[0] > 0.9999:
            print("  ✅ Near perfect match!")

if __name__ == "__main__":
    test_whitespace_handling()
    analyze_tokenization()
    test_perfect_matches()