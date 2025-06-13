#!/usr/bin/env python3
"""
Test the effect of prefix space on tokenization and similarity.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from bert_score.utils import sent_encode

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModel.from_pretrained("roberta-large")
model.eval()

print("RoBERTa Prefix Space Analysis")
print("="*60)

# Test different tokenization methods
test_words = ["OK", "Okay"]

for word in test_words:
    print(f"\nTesting: '{word}'")
    print("-"*40)
    
    # Method 1: Direct encode without special tokens
    ids1 = tokenizer.encode(word, add_special_tokens=False)
    tokens1 = [tokenizer.decode([id]) for id in ids1]
    print(f"encode(add_special_tokens=False): {ids1} → {tokens1}")
    
    # Method 2: Direct encode with special tokens
    ids2 = tokenizer.encode(word, add_special_tokens=True)
    tokens2 = [tokenizer.decode([id]) for id in ids2]
    print(f"encode(add_special_tokens=True): {ids2} → {tokens2}")
    
    # Method 3: bert_score's sent_encode
    ids3 = sent_encode(tokenizer, word)
    tokens3 = [tokenizer.decode([id]) for id in ids3]
    print(f"sent_encode(): {ids3} → {tokens3}")

# Now check the similarity between " OK" and " Okay" vs "OK" and "Okay"
print("\n" + "="*60)
print("Similarity comparison:")

# Get embeddings for both versions
with torch.no_grad():
    # With prefix space (bert_score style)
    ok_space = tokenizer.encode(" OK", return_tensors="pt")
    okay_space = tokenizer.encode(" Okay", return_tensors="pt")
    
    # Without prefix space
    ok_no_space = tokenizer.encode("OK", return_tensors="pt", add_special_tokens=True)
    okay_no_space = tokenizer.encode("Okay", return_tensors="pt", add_special_tokens=True)
    
    # Get embeddings
    ok_space_emb = model(ok_space, output_hidden_states=True).hidden_states[17][0]
    okay_space_emb = model(okay_space, output_hidden_states=True).hidden_states[17][0]
    ok_no_space_emb = model(ok_no_space, output_hidden_states=True).hidden_states[17][0]
    okay_no_space_emb = model(okay_no_space, output_hidden_states=True).hidden_states[17][0]

# Compare content token similarities
print("\nContent token IDs:")
print(f"' OK': {ok_space[0].tolist()} → content token: {ok_space[0][1:-1].tolist()}")
print(f"' Okay': {okay_space[0].tolist()} → content token: {okay_space[0][1:-1].tolist()}")
print(f"'OK': {ok_no_space[0].tolist()} → content token: {ok_no_space[0][1:-1].tolist()}")
print(f"'Okay': {okay_no_space[0].tolist()} → content token: {okay_no_space[0][1:-1].tolist()}")

# Compute similarities for content tokens only
def compute_content_similarity(emb1, emb2):
    # Extract content tokens (excluding CLS and SEP)
    content1 = emb1[1:-1]
    content2 = emb2[1:-1]
    # Normalize
    content1_norm = torch.nn.functional.normalize(content1, p=2, dim=-1)
    content2_norm = torch.nn.functional.normalize(content2, p=2, dim=-1)
    # Compute similarity
    sim = torch.mm(content1_norm, content2_norm.T)
    return sim[0, 0].item() if sim.numel() > 0 else 0.0

sim_with_space = compute_content_similarity(ok_space_emb, okay_space_emb)
sim_without_space = compute_content_similarity(ok_no_space_emb, okay_no_space_emb)

print(f"\nContent token similarities:")
print(f"' OK' vs ' Okay': {sim_with_space:.6f}")
print(f"'OK' vs 'Okay': {sim_without_space:.6f}")

# Check if " OK" and " Okay" might be the same token
print("\n" + "="*60)
print("KEY INSIGHT:")
if ok_space[0][1:-1].tolist() == okay_space[0][1:-1].tolist():
    print("' OK' and ' Okay' tokenize to the SAME content token!")
    print("This explains the 0.998344 similarity score.")
else:
    print("' OK' and ' Okay' tokenize to different tokens.")
    print(f"The high similarity ({sim_with_space:.6f}) is due to their embeddings being very similar.")