#!/usr/bin/env python3
"""
Debug tool to compare Python and Rust BERTScore computation step-by-step.
"""

import bert_score as bs
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import subprocess
import os
import json
import pandas as pd

def debug_python_bertscore(candidate, reference, use_baseline=False):
    """Debug Python BERTScore computation step by step."""
    print("="*80)
    print("PYTHON BERTSCORE DEBUG")
    print("="*80)
    print(f"Candidate: '{candidate}'")
    print(f"Reference: '{reference}'")
    print()
    
    # Initialize tokenizer and model
    model_type = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    model.eval()
    
    # Step 1: Tokenization
    print("1. TOKENIZATION")
    print("-"*40)
    
    # Show raw tokenization
    cand_tokens = tokenizer.tokenize(candidate)
    ref_tokens = tokenizer.tokenize(reference)
    print(f"Candidate tokens: {cand_tokens}")
    print(f"Reference tokens: {ref_tokens}")
    
    # Show with strip
    cand_stripped = candidate.strip()
    ref_stripped = reference.strip()
    print(f"\nAfter strip():")
    print(f"Candidate: '{cand_stripped}'")
    print(f"Reference: '{ref_stripped}'")
    
    # Encode
    from bert_score.utils import sent_encode
    cand_ids = sent_encode(tokenizer, candidate)
    ref_ids = sent_encode(tokenizer, reference)
    print(f"\nCandidate IDs: {cand_ids}")
    print(f"Reference IDs: {ref_ids}")
    
    # Step 2: Get embeddings
    print("\n2. EMBEDDINGS")
    print("-"*40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare inputs
    cand_tensor = torch.tensor([cand_ids]).to(device)
    ref_tensor = torch.tensor([ref_ids]).to(device)
    
    with torch.no_grad():
        cand_outputs = model(cand_tensor, output_hidden_states=True)
        ref_outputs = model(ref_tensor, output_hidden_states=True)
    
    # Get layer 17 embeddings (0-indexed)
    cand_emb = cand_outputs.hidden_states[17][0]  # Remove batch dim
    ref_emb = ref_outputs.hidden_states[17][0]
    
    print(f"Candidate embeddings shape: {cand_emb.shape}")
    print(f"Reference embeddings shape: {ref_emb.shape}")
    
    # Step 3: IDF weights (when IDF is disabled)
    print("\n3. IDF WEIGHTS (IDF=False)")
    print("-"*40)
    
    # Default IDF dict when IDF is disabled
    from collections import defaultdict
    idf_dict = defaultdict(lambda: 1.0)
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0
    
    cand_idf = torch.tensor([idf_dict[i] for i in cand_ids], dtype=torch.float).to(device)
    ref_idf = torch.tensor([idf_dict[i] for i in ref_ids], dtype=torch.float).to(device)
    
    print(f"CLS token ID: {tokenizer.cls_token_id}, SEP token ID: {tokenizer.sep_token_id}")
    print(f"Candidate IDF weights: {cand_idf}")
    print(f"Reference IDF weights: {ref_idf}")
    
    # Step 4: Normalization
    print("\n4. NORMALIZATION")
    print("-"*40)
    
    # L2 normalize
    cand_norm = torch.nn.functional.normalize(cand_emb, p=2, dim=-1)
    ref_norm = torch.nn.functional.normalize(ref_emb, p=2, dim=-1)
    
    print(f"First 5 values of normalized candidate: {cand_norm[0][:5]}")
    print(f"First 5 values of normalized reference: {ref_norm[0][:5]}")
    
    # Step 5: Similarity matrix
    print("\n5. SIMILARITY MATRIX")
    print("-"*40)
    
    sim_matrix = torch.matmul(cand_norm, ref_norm.T)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity matrix:\n{sim_matrix}")
    
    # Step 6: Greedy matching
    print("\n6. GREEDY MATCHING")
    print("-"*40)
    
    # Max along dimensions
    word_precision = sim_matrix.max(dim=1)[0]
    word_recall = sim_matrix.max(dim=0)[0]
    
    print(f"Word precision scores: {word_precision}")
    print(f"Word recall scores: {word_recall}")
    
    # Step 7: Weighted average
    print("\n7. WEIGHTED AVERAGE")
    print("-"*40)
    
    # Normalize IDF weights
    cand_idf_norm = cand_idf / cand_idf.sum()
    ref_idf_norm = ref_idf / ref_idf.sum()
    
    print(f"Normalized candidate IDF: {cand_idf_norm}")
    print(f"Normalized reference IDF: {ref_idf_norm}")
    
    P = (word_precision * cand_idf_norm).sum()
    R = (word_recall * ref_idf_norm).sum()
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0
    
    print(f"\nRaw scores: P={P:.6f}, R={R:.6f}, F1={F:.6f}")
    
    # Step 8: Get official scores for comparison
    print("\n8. OFFICIAL PYTHON SCORES")
    print("-"*40)
    
    result = bs.score(
        [candidate], [reference],
        model_type=model_type,
        lang="en",
        idf=False,
        rescale_with_baseline=use_baseline,
        verbose=False
    )
    P_official, R_official, F1_official = result
    print(f"Official scores: P={P_official[0]:.6f}, R={R_official[0]:.6f}, F1={F1_official[0]:.6f}")
    
    return {
        "cand_ids": cand_ids,
        "ref_ids": ref_ids,
        "cand_emb_shape": list(cand_emb.shape),
        "ref_emb_shape": list(ref_emb.shape),
        "sim_matrix": sim_matrix.cpu().numpy().tolist(),
        "P": float(P),
        "R": float(R),
        "F1": float(F),
        "P_official": float(P_official[0]),
        "R_official": float(R_official[0]),
        "F1_official": float(F1_official[0])
    }


def debug_rust_bertscore(candidate, reference):
    """Get debug info from Rust implementation."""
    print("\n" + "="*80)
    print("RUST BERTSCORE DEBUG")
    print("="*80)
    
    # Create temporary TSV file
    test_data = pd.DataFrame([{
        "id": "DEBUG001",
        "candidate": candidate,
        "reference": reference
    }])
    test_file = "temp_debug.tsv"
    test_data.to_csv(test_file, sep="\t", index=False)
    
    # Run Rust implementation
    env = os.environ.copy()
    env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
    env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    cmd = [
        "./target/release/bert-score",
        "score",
        "--input-tsv", test_file,
        "--output-csv", "temp_debug_output.csv",
        "--model-name", "roberta-large",
        "--pretrained", "roberta-large",
        "--lang", "en"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Read results
    rust_df = pd.read_csv("temp_debug_output.csv")
    
    # Clean up
    os.remove(test_file)
    os.remove("temp_debug_output.csv")
    
    print(f"Rust scores: P={rust_df['P_rust'][0]:.6f}, R={rust_df['R_rust'][0]:.6f}, F1={rust_df['F1_rust'][0]:.6f}")
    
    return {
        "P": rust_df['P_rust'][0],
        "R": rust_df['R_rust'][0],
        "F1": rust_df['F1_rust'][0]
    }


def main():
    """Debug specific test cases."""
    print("🔍 BERTScore Step-by-Step Debugger")
    print("="*80)
    
    # Test cases to debug
    test_cases = [
        ("OK", "Okay", "S0012 - Worst F1 difference"),
        ("Yes.", "No.", "S0020 - Large difference"),
        ("Hello world", "Hello world", "Identical - should be 1.0"),
        ("   Leading and trailing spaces   ", "Leading and trailing spaces", "S0008 - Whitespace"),
    ]
    
    for candidate, reference, description in test_cases:
        print(f"\n\n{'#'*80}")
        print(f"TEST CASE: {description}")
        print(f"{'#'*80}")
        
        # Debug Python
        py_debug = debug_python_bertscore(candidate, reference)
        
        # Debug Rust
        rust_debug = debug_rust_bertscore(candidate, reference)
        
        # Compare
        if rust_debug:
            print("\n" + "="*80)
            print("COMPARISON")
            print("="*80)
            print(f"Python F1: {py_debug['F1_official']:.6f}")
            print(f"Rust F1:   {rust_debug['F1']:.6f}")
            print(f"Difference: {abs(py_debug['F1_official'] - rust_debug['F1']):.6f}")
            
            # Save detailed debug info
            debug_file = f"debug_{description.split()[0].lower()}.json"
            with open(debug_file, "w") as f:
                json.dump({
                    "test_case": description,
                    "candidate": candidate,
                    "reference": reference,
                    "python": py_debug,
                    "rust": rust_debug
                }, f, indent=2)
            print(f"\nDetailed debug info saved to: {debug_file}")


if __name__ == "__main__":
    main()