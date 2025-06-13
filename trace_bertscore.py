#!/usr/bin/env python3
"""Trace through bert_score computation step by step."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Manually replicate bert_score's computation
def manual_bertscore(candidate, reference, model_type="roberta-large"):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type, output_hidden_states=True)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize using bert_score's method
    from bert_score.utils import sent_encode
    cand_ids = sent_encode(tokenizer, candidate)
    ref_ids = sent_encode(tokenizer, reference)
    
    print(f"Candidate: '{candidate}' -> {cand_ids}")
    print(f"Reference: '{reference}' -> {ref_ids}")
    
    # Get embeddings
    with torch.no_grad():
        cand_input = torch.tensor([cand_ids]).to(device)
        ref_input = torch.tensor([ref_ids]).to(device)
        
        cand_outputs = model(cand_input)
        ref_outputs = model(ref_input)
        
        # Get layer 17 (0-indexed, so index 16)
        cand_emb = cand_outputs.hidden_states[17][0]  # Remove batch dim
        ref_emb = ref_outputs.hidden_states[17][0]
    
    print(f"\nEmbedding shapes: {cand_emb.shape}, {ref_emb.shape}")
    
    # Create IDF weights
    cand_idf = torch.ones(len(cand_ids)).to(device)
    cand_idf[0] = 0  # CLS
    cand_idf[-1] = 0  # SEP
    
    ref_idf = torch.ones(len(ref_ids)).to(device)
    ref_idf[0] = 0  # CLS
    ref_idf[-1] = 0  # SEP
    
    # Normalize embeddings
    cand_norm = torch.nn.functional.normalize(cand_emb, p=2, dim=-1)
    ref_norm = torch.nn.functional.normalize(ref_emb, p=2, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(cand_norm, ref_norm.T)
    print(f"\nSimilarity matrix:\n{sim_matrix}")
    
    # Greedy matching
    word_precision = sim_matrix.max(dim=1)[0]
    word_recall = sim_matrix.max(dim=0)[0]
    
    print(f"\nWord precision: {word_precision}")
    print(f"Word recall: {word_recall}")
    
    # L1 normalize IDF weights
    cand_idf_norm = cand_idf / cand_idf.sum()
    ref_idf_norm = ref_idf / ref_idf.sum()
    
    print(f"\nNormalized IDF weights:")
    print(f"Candidate: {cand_idf_norm}")
    print(f"Reference: {ref_idf_norm}")
    
    # Compute scores
    P = (word_precision * cand_idf_norm).sum().item()
    R = (word_recall * ref_idf_norm).sum().item()
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    
    return P, R, F1

# Test case
candidate = "OK"
reference = "Okay"

print("Manual BERTScore computation:")
print("="*50)
P, R, F1 = manual_bertscore(candidate, reference)
print(f"\nManual scores: P={P:.6f}, R={R:.6f}, F1={F1:.6f}")

# Compare with official
import bert_score as bs
print("\n" + "="*50)
print("Official BERTScore:")
result = bs.score(
    [candidate], [reference],
    model_type="roberta-large",
    lang="en",
    idf=False,
    rescale_with_baseline=False,
    verbose=False
)
P_official, R_official, F1_official = result
print(f"Official scores: P={P_official[0]:.6f}, R={R_official[0]:.6f}, F1={F1_official[0]:.6f}")

# Try with baseline rescaling
result_baseline = bs.score(
    [candidate], [reference],
    model_type="roberta-large",
    lang="en",
    idf=False,
    rescale_with_baseline=True,
    verbose=False
)
P_baseline, R_baseline, F1_baseline = result_baseline
print(f"\nWith baseline rescaling: P={P_baseline[0]:.6f}, R={R_baseline[0]:.6f}, F1={F1_baseline[0]:.6f}")