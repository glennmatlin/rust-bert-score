#!/usr/bin/env python3
"""Test RoBERTa tokenization behavior with and without prefix spaces."""

from transformers import AutoTokenizer

# Load RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

# Test cases
test_strings = ["OK", "Okay", " OK", " Okay"]

print("RoBERTa Tokenization Test")
print("=" * 50)

for text in test_strings:
    # Tokenize
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Also test with encode
    encoded = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"\nText: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {ids}")
    print(f"Encoded IDs: {encoded}")

# Test the actual bert_score behavior
print("\n" + "=" * 50)
print("Testing bert_score's sent_encode function")
print("=" * 50)

from bert_score.utils import sent_encode

for text in ["OK", "Okay"]:
    ids = sent_encode(tokenizer, text)
    print(f"\nText: '{text}'")
    print(f"sent_encode IDs: {ids}")
    
    # Decode to see what it actually tokenized
    decoded = tokenizer.decode(ids[1:-1])  # Skip CLS and SEP
    print(f"Decoded (without special tokens): '{decoded}'")