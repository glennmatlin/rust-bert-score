#!/usr/bin/env python3
"""
Compare tokenization between Python (HuggingFace) and Rust implementations.
Based on Strategy 4 from EXPERIMENTS.md - Tokenization & Embedding Parity Check.
"""

import json
import sys
import numpy as np
from typing import List, Dict, Tuple
import subprocess
import os

# Test sentences covering various scenarios
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are changing NLP.",
    "¿Dónde está la biblioteca?",
    "你好，世界！",
    "Hello, world!",
    "123.45 + 678.90 = 802.35",
    "user@example.com visited https://www.example.com",
    "I love 🍕 and 🍔!",
    "CamelCaseWords and snake_case_words",
    "",  # Empty string edge case
]

def tokenize_with_python(sentences: List[str], model_name: str = "roberta-large") -> Dict[str, List]:
    """Tokenize sentences using HuggingFace transformers."""
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        results = []
        
        for sentence in sentences:
            # Get token IDs and tokens
            encoding = tokenizer(sentence, add_special_tokens=True, return_offsets_mapping=True)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
            
            results.append({
                "text": sentence,
                "token_ids": encoding["input_ids"],
                "tokens": tokens,
                "length": len(encoding["input_ids"]),
            })
        
        return results
    
    except ImportError:
        print("ERROR: transformers package not installed.")
        print("Please install: pip install transformers")
        sys.exit(1)

def tokenize_with_rust(sentences: List[str], model_name: str = "roberta-large") -> Dict[str, List]:
    """Tokenize sentences using Rust implementation."""
    # Mock implementation for now
    # In production, this would call the Rust tokenizer
    
    results = []
    for sentence in sentences:
        # Placeholder tokenization
        mock_tokens = sentence.split() if sentence else []
        mock_ids = list(range(len(mock_tokens) + 2))  # +2 for special tokens
        
        results.append({
            "text": sentence,
            "token_ids": mock_ids,
            "tokens": ["<s>"] + mock_tokens + ["</s>"],
            "length": len(mock_ids),
        })
    
    return results
    
    # Real implementation:
    """
    # Write sentences to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for sentence in sentences:
            f.write(sentence + '\n')
        input_file = f.name
    
    try:
        result = subprocess.run([
            'cargo', 'run', '--release', '--bin', 'bert-score', '--',
            'dump-tokens',
            '--model', model_name,
            '--input', input_file,
            '--format', 'json'
        ], capture_output=True, text=True, check=True)
        
        return json.loads(result.stdout)
    finally:
        os.unlink(input_file)
    """

def compare_tokenizations(py_results: List[Dict], rust_results: List[Dict]) -> Tuple[bool, List[Dict]]:
    """Compare tokenization results between Python and Rust."""
    all_match = True
    comparisons = []
    
    for py_res, rust_res in zip(py_results, rust_results):
        comparison = {
            "text": py_res["text"],
            "matches": True,
            "py_length": py_res["length"],
            "rust_length": rust_res["length"],
            "details": []
        }
        
        # Compare lengths
        if py_res["length"] != rust_res["length"]:
            comparison["matches"] = False
            comparison["details"].append(
                f"Length mismatch: Python={py_res['length']}, Rust={rust_res['length']}"
            )
            all_match = False
        
        # Compare token IDs
        py_ids = py_res["token_ids"]
        rust_ids = rust_res["token_ids"]
        
        if py_ids != rust_ids:
            comparison["matches"] = False
            comparison["details"].append("Token ID mismatch")
            
            # Find first difference
            for i, (py_id, rust_id) in enumerate(zip(py_ids, rust_ids)):
                if py_id != rust_id:
                    comparison["details"].append(
                        f"  First diff at position {i}: Python={py_id}, Rust={rust_id}"
                    )
                    break
            
            all_match = False
        
        comparisons.append(comparison)
    
    return all_match, comparisons

def generate_tokenization_report(py_results: List[Dict], 
                               rust_results: List[Dict],
                               comparisons: List[Dict]) -> str:
    """Generate detailed tokenization comparison report."""
    report = ["# Tokenization Parity Report\n"]
    report.append("## Summary\n")
    
    matches = sum(1 for c in comparisons if c["matches"])
    total = len(comparisons)
    
    if matches == total:
        report.append(f"✅ **PASSED**: All {total} sentences tokenized identically.\n")
    else:
        report.append(f"❌ **FAILED**: {total - matches}/{total} sentences have tokenization differences.\n")
    
    # Detailed comparison
    report.append("\n## Detailed Comparison\n")
    
    for i, (py_res, rust_res, comp) in enumerate(zip(py_results, rust_results, comparisons)):
        report.append(f"\n### Test Case {i + 1}\n")
        report.append(f"**Text**: `{comp['text']}`\n")
        report.append(f"**Status**: {'✅ MATCH' if comp['matches'] else '❌ MISMATCH'}\n")
        
        if not comp["matches"]:
            report.append("\n**Issues**:\n")
            for detail in comp["details"]:
                report.append(f"- {detail}\n")
        
        # Token details
        report.append("\n**Python Tokenization**:\n")
        report.append(f"- Token IDs: {py_res['token_ids']}\n")
        report.append(f"- Tokens: {py_res['tokens']}\n")
        
        report.append("\n**Rust Tokenization**:\n")
        report.append(f"- Token IDs: {rust_res['token_ids']}\n")
        report.append(f"- Tokens: {rust_res['tokens']}\n")
    
    return "".join(report)

def test_embeddings_if_available(model_name: str = "roberta-large"):
    """Test embedding extraction if both implementations support it."""
    print("\nTesting embedding extraction...")
    
    # Simple test sentence
    test_sentence = "Hello, world!"
    
    try:
        # Python embedding extraction
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        model.eval()
        
        # Tokenize and get embeddings
        inputs = tokenizer(test_sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # Get last layer embeddings
            last_hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
        
        print(f"Python embeddings shape: {last_hidden.shape}")
        print(f"First token embedding (first 5 dims): {last_hidden[0][:5].numpy()}")
        
        # TODO: Call Rust embedding extraction and compare
        # For now, just show what we would compare
        print("\nTo compare with Rust:")
        print("1. Extract embeddings from Rust implementation")
        print("2. Compare shapes")
        print("3. Compute element-wise differences")
        print("4. Ensure max absolute difference < 1e-6")
        
    except Exception as e:
        print(f"Could not test embeddings: {e}")

def main():
    """Main tokenization comparison workflow."""
    print("Tokenization Parity Check")
    print("=" * 50)
    
    model_name = "roberta-large"
    print(f"Model: {model_name}")
    print(f"Testing {len(TEST_SENTENCES)} sentences...")
    
    # Tokenize with Python
    print("\nTokenizing with Python (HuggingFace)...")
    py_results = tokenize_with_python(TEST_SENTENCES, model_name)
    
    # Tokenize with Rust
    print("Tokenizing with Rust...")
    rust_results = tokenize_with_rust(TEST_SENTENCES, model_name)
    
    # Compare
    print("\nComparing tokenizations...")
    all_match, comparisons = compare_tokenizations(py_results, rust_results)
    
    # Generate report
    report = generate_tokenization_report(py_results, rust_results, comparisons)
    
    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/tokenization_parity.md", "w") as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 50)
    if all_match:
        print("✅ PASSED: All tokenizations match exactly")
    else:
        print("❌ FAILED: Tokenization differences detected")
        
        # Show mismatches
        for i, comp in enumerate(comparisons):
            if not comp["matches"]:
                print(f"\nMismatch in sentence {i + 1}: {comp['text'][:50]}...")
                for detail in comp["details"]:
                    print(f"  {detail}")
    
    print(f"\nDetailed report saved to: reports/tokenization_parity.md")
    
    # Optional: test embeddings
    test_embeddings_if_available(model_name)
    
    sys.exit(0 if all_match else 1)

if __name__ == "__main__":
    main()