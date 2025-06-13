#!/usr/bin/env python3
"""
Test embeddings and similarity for "OK" vs "Okay".
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Initialize
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModel.from_pretrained("roberta-large")
model.eval()

# Test inputs
candidate = "OK"
reference = "Okay"

print("Embedding Analysis for 'OK' vs 'Okay'")
print("="*60)

# Tokenize
cand_ids = tokenizer.encode(candidate, return_tensors="pt")
ref_ids = tokenizer.encode(reference, return_tensors="pt")

print(f"Candidate token IDs: {cand_ids[0].tolist()}")
print(f"Reference token IDs: {ref_ids[0].tolist()}")
print(f"Content tokens - Candidate: {cand_ids[0][1:-1].tolist()}, Reference: {ref_ids[0][1:-1].tolist()}")

# Get embeddings
with torch.no_grad():
    cand_outputs = model(cand_ids, output_hidden_states=True)
    ref_outputs = model(ref_ids, output_hidden_states=True)

# Extract layer 17 embeddings (0-indexed)
cand_emb_17 = cand_outputs.hidden_states[17][0]  # Shape: [seq_len, hidden_dim]
ref_emb_17 = ref_outputs.hidden_states[17][0]

print(f"\nEmbedding shapes:")
print(f"Candidate: {cand_emb_17.shape}")
print(f"Reference: {ref_emb_17.shape}")

# Get content token embeddings (excluding CLS and SEP)
cand_content_emb = cand_emb_17[1:-1]  # Shape: [1, hidden_dim] for "OK"
ref_content_emb = ref_emb_17[1:-1]    # Shape: [1, hidden_dim] for "Okay"

# Normalize
cand_norm = torch.nn.functional.normalize(cand_content_emb, p=2, dim=-1)
ref_norm = torch.nn.functional.normalize(ref_content_emb, p=2, dim=-1)

# Compute cosine similarity
cos_sim = torch.mm(cand_norm, ref_norm.T)
print(f"\nCosine similarity between 'OK' and 'Okay' embeddings: {cos_sim[0,0].item():.6f}")

# Also check all token similarities
print("\nFull similarity matrix (including special tokens):")
cand_all_norm = torch.nn.functional.normalize(cand_emb_17, p=2, dim=-1)
ref_all_norm = torch.nn.functional.normalize(ref_emb_17, p=2, dim=-1)
sim_matrix = torch.mm(cand_all_norm, ref_all_norm.T)
print(sim_matrix)

# Let's also check if the CLS tokens are identical
cls_sim = torch.cosine_similarity(cand_emb_17[0:1], ref_emb_17[0:1])
sep_sim = torch.cosine_similarity(cand_emb_17[-1:], ref_emb_17[-1:])
print(f"\nCLS token similarity: {cls_sim.item():.6f}")
print(f"SEP token similarity: {sep_sim.item():.6f}")

# Now compute BERTScore style
print("\n" + "="*60)
print("BERTScore-style computation:")

# With IDF weights disabled (all 1.0 except CLS/SEP which are 0)
idf_weights_cand = torch.tensor([0.0, 1.0, 0.0])  # [CLS, OK, SEP]
idf_weights_ref = torch.tensor([0.0, 1.0, 0.0])   # [CLS, Okay, SEP]

# Normalize IDF weights
idf_norm_cand = idf_weights_cand / idf_weights_cand.sum()
idf_norm_ref = idf_weights_ref / idf_weights_ref.sum()

# Greedy matching
word_precision = sim_matrix.max(dim=1)[0]  # Max for each candidate token
word_recall = sim_matrix.max(dim=0)[0]     # Max for each reference token

print(f"\nWord-level scores:")
print(f"Precision per token: {word_precision}")
print(f"Recall per token: {word_recall}")

# Weighted average
P = (word_precision * idf_norm_cand).sum()
R = (word_recall * idf_norm_ref).sum()
F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

print(f"\nBERTScore results:")
print(f"P = {P:.6f}")
print(f"R = {R:.6f}")
print(f"F1 = {F1:.6f}")

# This should explain why we're getting high scores
print("\n" + "="*60)
print("EXPLANATION:")
print("Even though 'OK' and 'Okay' are different tokens, their CLS and SEP tokens")
print("are identical (similarity = 1.0). When computing BERTScore, the greedy matching")
print("allows CLS to match with CLS and SEP to match with SEP, giving perfect scores")
print("for those positions. This inflates the overall score.")