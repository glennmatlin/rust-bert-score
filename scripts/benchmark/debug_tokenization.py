#!/usr/bin/env python3
"""
Debug tokenization differences for "OK" vs "Okay".
"""

from transformers import AutoTokenizer
from bert_score.utils import sent_encode
import torch

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

# Test texts
candidate = "OK"
reference = "Okay"

print("Tokenization Debug for 'OK' vs 'Okay'")
print("="*60)

# Show raw tokenization
print("\n1. Basic tokenize():")
cand_tokens = tokenizer.tokenize(candidate)
ref_tokens = tokenizer.tokenize(reference)
print(f"Candidate '{candidate}' → {cand_tokens}")
print(f"Reference '{reference}' → {ref_tokens}")

# Show token IDs
print("\n2. Basic encode():")
cand_ids_basic = tokenizer.encode(candidate, add_special_tokens=False)
ref_ids_basic = tokenizer.encode(reference, add_special_tokens=False)
print(f"Candidate IDs: {cand_ids_basic}")
print(f"Reference IDs: {ref_ids_basic}")

# Show with special tokens
print("\n3. encode() with special tokens:")
cand_ids_special = tokenizer.encode(candidate, add_special_tokens=True)
ref_ids_special = tokenizer.encode(reference, add_special_tokens=True)
print(f"Candidate IDs: {cand_ids_special}")
print(f"Reference IDs: {ref_ids_special}")

# Show bert_score's sent_encode
print("\n4. bert_score sent_encode():")
cand_ids_bert = sent_encode(tokenizer, candidate)
ref_ids_bert = sent_encode(tokenizer, reference)
print(f"Candidate IDs: {cand_ids_bert}")
print(f"Reference IDs: {ref_ids_bert}")

# Decode to see the actual tokens
print("\n5. Decoded tokens from sent_encode:")
cand_decoded = [tokenizer.decode([id]) for id in cand_ids_bert]
ref_decoded = [tokenizer.decode([id]) for id in ref_ids_bert]
print(f"Candidate: {cand_decoded}")
print(f"Reference: {ref_decoded}")

# Check if stripping affects it
print("\n6. Effect of strip():")
print(f"Original candidate: '{candidate}' (len={len(candidate)})")
print(f"Stripped candidate: '{candidate.strip()}' (len={len(candidate.strip())})")

# Check prefix space behavior for RoBERTa
print("\n7. RoBERTa prefix space behavior:")
# RoBERTa adds a prefix space by default
print("Without prefix space:")
ids_no_prefix = tokenizer.encode("OK", add_special_tokens=False, add_prefix_space=False)
print(f"  'OK' → {ids_no_prefix} → {[tokenizer.decode([id]) for id in ids_no_prefix]}")

print("With prefix space (default for RoBERTa):")
ids_with_prefix = tokenizer.encode("OK", add_special_tokens=False, add_prefix_space=True)
print(f"  'OK' → {ids_with_prefix} → {[tokenizer.decode([id]) for id in ids_with_prefix]}")

# Compare full pipeline
print("\n8. Full tokenization comparison:")
print(f"Candidate sent_encode: {cand_ids_bert}")
print(f"Reference sent_encode: {ref_ids_bert}")
print(f"Are they identical? {cand_ids_bert == ref_ids_bert}")

# Check if OK and Okay have the same token when lowercase
print("\n9. Case sensitivity:")
ok_lower = tokenizer.encode("ok", add_special_tokens=False, add_prefix_space=True)
okay_lower = tokenizer.encode("okay", add_special_tokens=False, add_prefix_space=True)
print(f"'ok' → {ok_lower}")
print(f"'okay' → {okay_lower}")

# Important finding
print("\n" + "="*60)
print("KEY FINDING:")
if cand_ids_bert[1:-1] == ref_ids_bert[1:-1]:  # Excluding special tokens
    print("The content tokens are IDENTICAL for 'OK' and 'Okay'!")
    print("This explains why the similarity is so high (0.998344)")
else:
    print("The content tokens are different.")
    print(f"Content tokens for 'OK': {cand_ids_bert[1:-1]}")
    print(f"Content tokens for 'Okay': {ref_ids_bert[1:-1]}")