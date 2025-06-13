#!/usr/bin/env python3
"""
Test bert_score internals to understand the 0.998344 score.
"""

import bert_score as bs
import torch
from bert_score.utils import get_bert_embedding, greedy_cos_idf, get_model, get_tokenizer
from collections import defaultdict

# Initialize
model_type = "roberta-large"
num_layers = 17  # Layer 17 for roberta-large
tokenizer = get_tokenizer(model_type)
model = get_model(model_type, num_layers, all_layers=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test case
candidates = ["OK"]
references = ["Okay"]

print("BERTScore Internal Analysis")
print("="*60)

# Create IDF dict (disabled)
idf_dict = defaultdict(lambda: 1.0)
idf_dict[tokenizer.sep_token_id] = 0
idf_dict[tokenizer.cls_token_id] = 0

print(f"CLS token ID: {tokenizer.cls_token_id}")
print(f"SEP token ID: {tokenizer.sep_token_id}")

# Get embeddings using bert_score's functions
ref_embedding, ref_masks, ref_idf = get_bert_embedding(
    references, model, tokenizer, idf_dict, device=device, all_layers=False
)
hyp_embedding, hyp_masks, hyp_idf = get_bert_embedding(
    candidates, model, tokenizer, idf_dict, device=device, all_layers=False
)

print(f"\nEmbedding shapes:")
print(f"Reference: {ref_embedding.shape}")
print(f"Hypothesis: {hyp_embedding.shape}")

print(f"\nMasks:")
print(f"Reference mask: {ref_masks}")
print(f"Hypothesis mask: {hyp_masks}")

print(f"\nIDF weights:")
print(f"Reference IDF: {ref_idf}")
print(f"Hypothesis IDF: {hyp_idf}")

# Compute scores using bert_score's greedy_cos_idf
P, R, F1 = greedy_cos_idf(
    ref_embedding,
    ref_masks,
    ref_idf,
    hyp_embedding,
    hyp_masks,
    hyp_idf,
    all_layers=False
)

print(f"\nScores from greedy_cos_idf:")
print(f"P = {P.item():.6f}")
print(f"R = {R.item():.6f}")
print(f"F1 = {F1.item():.6f}")

# Now let's manually check the embeddings
print("\n" + "="*60)
print("Manual inspection:")

# Normalize embeddings
ref_norm = ref_embedding / torch.norm(ref_embedding, dim=-1, keepdim=True)
hyp_norm = hyp_embedding / torch.norm(hyp_embedding, dim=-1, keepdim=True)

# Compute similarity matrix
sim = torch.bmm(hyp_norm, ref_norm.transpose(1, 2))[0]
print(f"\nSimilarity matrix shape: {sim.shape}")
print(f"Similarity matrix:\n{sim}")

# Apply masks
mask_matrix = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())[0]
print(f"\nMask matrix:\n{mask_matrix}")

masked_sim = sim * mask_matrix
print(f"\nMasked similarity:\n{masked_sim}")

# Check what tokens we have
from bert_score.utils import sent_encode
cand_ids = sent_encode(tokenizer, candidates[0])
ref_ids = sent_encode(tokenizer, references[0])
print(f"\nCandidate token IDs: {cand_ids}")
print(f"Reference token IDs: {ref_ids}")

# Decode tokens
cand_tokens = [tokenizer.decode([id]) for id in cand_ids]
ref_tokens = [tokenizer.decode([id]) for id in ref_ids]
print(f"Candidate tokens: {cand_tokens}")
print(f"Reference tokens: {ref_tokens}")

# Now use the official score function for comparison
print("\n" + "="*60)
print("Official bert_score result:")
P_official, R_official, F1_official = bs.score(
    candidates, references,
    model_type=model_type,
    lang="en",
    idf=False,
    rescale_with_baseline=False,
    verbose=False
)
print(f"P = {P_official[0]:.6f}")
print(f"R = {R_official[0]:.6f}")
print(f"F1 = {F1_official[0]:.6f}")

print("\n" + "="*60)
print("DEBUGGING THE DIFFERENCE:")
print(f"greedy_cos_idf F1: {F1.item():.6f}")
print(f"Official F1: {F1_official[0]:.6f}")
print(f"Difference: {abs(F1.item() - F1_official[0]):.6f}")