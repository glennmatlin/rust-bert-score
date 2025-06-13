#!/usr/bin/env python3
"""
Debug whitespace handling in Python bert_score.
"""

from transformers import AutoTokenizer

def test_tokenization():
    """Test how tokenizer handles different whitespace."""
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    
    test_cases = [
        ("Hello world", "Normal"),
        ("   Hello world   ", "Leading/trailing spaces"),
        ("Hello   world", "Multiple spaces"),
        ("Hello\nworld", "Newline"),
        ("Hello\tworld", "Tab"),
    ]
    
    print("Tokenization Analysis:")
    print("=" * 80)
    
    for text, desc in test_cases:
        # Show what strip() does
        stripped = text.strip()
        
        # Tokenize original
        tokens_orig = tokenizer.tokenize(text)
        ids_orig = tokenizer.encode(text, add_special_tokens=False)
        
        # Tokenize stripped
        tokens_strip = tokenizer.tokenize(stripped)
        ids_strip = tokenizer.encode(stripped, add_special_tokens=False)
        
        print(f"\n{desc}:")
        print(f"  Original: '{text}'")
        print(f"  Stripped: '{stripped}'")
        print(f"  Original tokens: {tokens_orig}")
        print(f"  Stripped tokens: {tokens_strip}")
        print(f"  Same after strip? {tokens_orig == tokens_strip}")

if __name__ == "__main__":
    test_tokenization()