#!/usr/bin/env python3
"""Analyze the similarity matrix computation in detail."""

import torch
import bert_score as bs
from transformers import AutoTokenizer, AutoModel

# Test case
candidate = "OK"
reference = "Okay"

# Load model and tokenizer
model_type = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModel.from_pretrained(model_type)
model.eval()

# Get embeddings using bert_score's internal method
from bert_score.utils import get_bert_embedding

# Process through bert_score's pipeline
all_preds = [candidate]
all_targets = [reference]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get embeddings the way bert_score does it
model_embeddings = get_bert_embedding(
    all_preds, model, tokenizer, idf_dict=None,
    device=device, all_layers=False, idf=False
)

# The embeddings are returned as (batch_embeddings, idf_weights)
pred_embeddings = model_embeddings[0][0]  # First item in batch
ref_embeddings = model_embeddings[0][1]   # Reference is also in first batch

print(f"Candidate: '{candidate}'")
print(f"Reference: '{reference}'")
print(f"Pred embeddings shape: {pred_embeddings.shape}")
print(f"Ref embeddings shape: {ref_embeddings.shape}")

# Compute similarity the way bert_score does
from bert_score.utils import greedy_cos_idf

# Call the actual function used by bert_score
all_preds_embeddings = pred_embeddings.unsqueeze(0)  # Add batch dimension
all_targets_embeddings = ref_embeddings.unsqueeze(0)

# IDF weights (all 1.0 except for special tokens)
pred_idfs = torch.ones(pred_embeddings.shape[0]).to(device)
pred_idfs[0] = 0  # CLS
pred_idfs[-1] = 0  # SEP

ref_idfs = torch.ones(ref_embeddings.shape[0]).to(device)
ref_idfs[0] = 0  # CLS
ref_idfs[-1] = 0  # SEP

all_preds_idfs = pred_idfs.unsqueeze(0)
all_targets_idfs = ref_idfs.unsqueeze(0)

# Call the actual function
P, R, F1 = greedy_cos_idf(
    all_preds_embeddings, all_targets_embeddings,
    all_preds_idfs, all_targets_idfs
)

print(f"\nDirect greedy_cos_idf call:")
print(f"P={P[0]:.6f}, R={R[0]:.6f}, F1={F1[0]:.6f}")

# Compare with official score
result = bs.score(
    [candidate], [reference],
    model_type=model_type,
    lang="en",
    idf=False,
    rescale_with_baseline=False,
    verbose=False
)
P_official, R_official, F1_official = result
print(f"\nOfficial bert_score call:")
print(f"P={P_official[0]:.6f}, R={R_official[0]:.6f}, F1={F1_official[0]:.6f}")