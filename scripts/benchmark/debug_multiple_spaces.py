#!/usr/bin/env python3
"""
Debug the "Multiple spaces" test case.
"""

from transformers import AutoTokenizer
from bert_score.utils import sent_encode

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

# Test case S0003
candidate = "Multiple   spaces   between   words"
reference = "Multiple spaces between words"

print("Debugging 'Multiple spaces' case")
print("="*60)
print(f"Candidate: '{candidate}'")
print(f"Reference: '{reference}'")

# Check stripping
print(f"\nAfter strip():")
print(f"Candidate: '{candidate.strip()}'")
print(f"Reference: '{reference.strip()}'")

# Tokenize with bert_score
cand_ids = sent_encode(tokenizer, candidate)
ref_ids = sent_encode(tokenizer, reference)

print(f"\nCandidate token IDs: {cand_ids}")
print(f"Reference token IDs: {ref_ids}")

# Decode to see tokens
cand_tokens = [tokenizer.decode([id]) for id in cand_ids]
ref_tokens = [tokenizer.decode([id]) for id in ref_ids]

print(f"\nCandidate tokens: {cand_tokens}")
print(f"Reference tokens: {ref_tokens}")

# Count tokens
print(f"\nToken counts:")
print(f"Candidate: {len(cand_ids)} tokens")
print(f"Reference: {len(ref_ids)} tokens")

# Check if the extra spaces create extra tokens
print("\nAnalysis:")
if len(cand_ids) != len(ref_ids):
    print("✗ Different number of tokens - this will affect similarity")
else:
    print("✓ Same number of tokens")
    
# Let's also check raw tokenization
print("\nRaw tokenization (no special tokens):")
cand_raw = tokenizer.tokenize(candidate)
ref_raw = tokenizer.tokenize(reference)
print(f"Candidate: {cand_raw}")
print(f"Reference: {ref_raw}")