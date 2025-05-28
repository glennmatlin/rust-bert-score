#!/usr/bin/env python3
"""
Validation script to compare Rust BERTScore implementation against Python BERTScore.
Based on Strategy 3 from EXPERIMENTS.md - Direct Score Comparison on Custom Test Cases.
"""

import json
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import subprocess
import tempfile
import os

# Test sentence pairs covering various scenarios
TEST_PAIRS = [
    # Identical sentences
    ("The quick brown fox jumps over the lazy dog.", 
     "The quick brown fox jumps over the lazy dog."),
    
    # Paraphrases
    ("The cat sat on the mat.",
     "A feline was resting on the rug."),
    
    # Different word order
    ("Alice gave Bob a book.",
     "Bob was given a book by Alice."),
    
    # Partial overlap
    ("The weather is nice today.",
     "Today the weather seems pleasant."),
    
    # Completely different
    ("I love programming in Rust.",
     "The rain in Spain stays mainly in the plain."),
    
    # Long sentences
    ("The development of artificial intelligence has revolutionized many industries, from healthcare to finance, enabling more efficient processes and better decision-making through advanced algorithms and machine learning techniques.",
     "AI has transformed numerous sectors including medical care and banking by implementing sophisticated computational methods that enhance operational efficiency and improve the quality of decisions."),
    
    # Short sentences
    ("Yes.", "No."),
    
    # Numbers and special characters
    ("The price is $29.99 (30% off)!",
     "Cost: twenty-nine dollars and ninety-nine cents with 30 percent discount."),
    
    # Code-like text
    ("def hello(): print('Hello, world!')",
     "function hello() { console.log('Hello, world!'); }"),
    
    # Empty vs non-empty
    ("", "This is some text."),
    
    # Punctuation differences
    ("Hello, world!", "Hello world"),
    
    # Case differences
    ("THIS IS UPPERCASE", "this is uppercase"),
    
    # Unicode and emoji
    ("I love 🍕 and 🍔!", "I enjoy pizza and hamburgers!"),
    
    # Technical jargon
    ("The transformer architecture uses self-attention mechanisms.",
     "Self-attention is a key component of transformers."),
    
    # Negation
    ("I like this movie.", "I don't like this movie."),
]

def run_python_bertscore(candidates: List[str], references: List[str]) -> Dict:
    """Run Python BERTScore on candidate-reference pairs."""
    try:
        import bert_score
        
        # Use same settings as our Rust implementation
        P, R, F1 = bert_score.score(
            candidates,
            references,
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=False,  # Start without baseline for simplicity
            batch_size=32,
            verbose=False,
            use_fast_tokenizer=True,
        )
        
        return {
            "precision": P.numpy().tolist(),
            "recall": R.numpy().tolist(),
            "f1": F1.numpy().tolist(),
        }
    except ImportError:
        print("ERROR: bert-score package not installed.")
        print("Please install: pip install bert-score")
        sys.exit(1)

def run_rust_bertscore(candidates: List[str], references: List[str]) -> Dict:
    """Run Rust BERTScore implementation."""
    # For now, we'll create a mock response since the CLI isn't fully implemented
    # In production, this would call the actual Rust binary
    
    # Mock response matching Python structure
    num_pairs = len(candidates)
    return {
        "precision": [0.95] * num_pairs,  # Placeholder values
        "recall": [0.93] * num_pairs,
        "f1": [0.94] * num_pairs,
    }
    
    # Real implementation would be:
    """
    # Write input to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        for i, (cand, ref) in enumerate(zip(candidates, references)):
            f.write(f"{i}\t{cand}\t{ref}\n")
        input_file = f.name
    
    try:
        # Call Rust CLI
        result = subprocess.run([
            'cargo', 'run', '--release', '--bin', 'bert-score', '--',
            'score',
            '--input', input_file,
            '--model', 'roberta-large',
            '--lang', 'en',
            '--format', 'json'
        ], capture_output=True, text=True, check=True)
        
        return json.loads(result.stdout)
    finally:
        os.unlink(input_file)
    """

def compare_results(py_scores: Dict, rust_scores: Dict, tolerance: float = 1e-4) -> Tuple[bool, Dict]:
    """Compare Python and Rust results."""
    results = {
        "passed": True,
        "max_diff": {},
        "pearson_corr": {},
        "details": []
    }
    
    for metric in ["precision", "recall", "f1"]:
        py_vals = np.array(py_scores[metric])
        rust_vals = np.array(rust_scores[metric])
        
        # Maximum absolute difference
        diff = np.abs(py_vals - rust_vals)
        max_diff = np.max(diff)
        results["max_diff"][metric] = float(max_diff)
        
        # Pearson correlation
        if len(py_vals) > 1:
            corr = np.corrcoef(py_vals, rust_vals)[0, 1]
            results["pearson_corr"][metric] = float(corr)
        
        # Check tolerance
        if max_diff > tolerance:
            results["passed"] = False
            results["details"].append(f"{metric}: max diff {max_diff:.6f} exceeds tolerance {tolerance}")
    
    return results["passed"], results

def generate_report(test_pairs: List[Tuple[str, str]], 
                   py_scores: Dict, 
                   rust_scores: Dict,
                   comparison: Dict) -> str:
    """Generate a detailed comparison report."""
    report = ["# BERTScore Parity Validation Report\n"]
    report.append("## Summary\n")
    
    if comparison["passed"]:
        report.append("✅ **PASSED**: Rust implementation matches Python within tolerance.\n")
    else:
        report.append("❌ **FAILED**: Significant differences detected.\n")
    
    # Statistics
    report.append("\n## Statistics\n")
    report.append("| Metric | Max Absolute Diff | Pearson Correlation |\n")
    report.append("|--------|------------------|--------------------|\n")
    
    for metric in ["precision", "recall", "f1"]:
        max_diff = comparison["max_diff"].get(metric, 0)
        corr = comparison["pearson_corr"].get(metric, 1.0)
        report.append(f"| {metric.capitalize()} | {max_diff:.6f} | {corr:.6f} |\n")
    
    # Detailed comparison
    report.append("\n## Detailed Results\n")
    report.append("| ID | Candidate | Reference | P_py | P_rust | P_diff | R_py | R_rust | R_diff | F1_py | F1_rust | F1_diff |\n")
    report.append("|-----|-----------|-----------|------|--------|--------|------|--------|--------|-------|---------|----------|\n")
    
    for i, (cand, ref) in enumerate(test_pairs):
        # Truncate long text for display
        cand_short = (cand[:30] + "...") if len(cand) > 30 else cand
        ref_short = (ref[:30] + "...") if len(ref) > 30 else ref
        
        p_py = py_scores["precision"][i]
        p_rust = rust_scores["precision"][i]
        r_py = py_scores["recall"][i]
        r_rust = rust_scores["recall"][i]
        f1_py = py_scores["f1"][i]
        f1_rust = rust_scores["f1"][i]
        
        report.append(f"| {i:3d} | {cand_short:30s} | {ref_short:30s} | "
                     f"{p_py:.4f} | {p_rust:.4f} | {abs(p_py-p_rust):.4f} | "
                     f"{r_py:.4f} | {r_rust:.4f} | {abs(r_py-r_rust):.4f} | "
                     f"{f1_py:.4f} | {f1_rust:.4f} | {abs(f1_py-f1_rust):.4f} |\n")
    
    return "".join(report)

def main():
    """Main validation workflow."""
    print("BERTScore Parity Validation")
    print("=" * 50)
    
    # Extract candidates and references
    candidates = [pair[0] for pair in TEST_PAIRS]
    references = [pair[1] for pair in TEST_PAIRS]
    
    print(f"Testing {len(TEST_PAIRS)} sentence pairs...")
    
    # Run Python BERTScore
    print("\nRunning Python BERTScore...")
    py_scores = run_python_bertscore(candidates, references)
    
    # Run Rust BERTScore
    print("Running Rust BERTScore...")
    rust_scores = run_rust_bertscore(candidates, references)
    
    # Compare results
    print("\nComparing results...")
    passed, comparison = compare_results(py_scores, rust_scores)
    
    # Generate report
    report = generate_report(TEST_PAIRS, py_scores, rust_scores, comparison)
    
    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/parity_validation.md", "w") as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 50)
    if passed:
        print("✅ PASSED: Implementations match within tolerance")
    else:
        print("❌ FAILED: Significant differences detected")
    
    print(f"\nMax differences:")
    for metric, diff in comparison["max_diff"].items():
        print(f"  {metric}: {diff:.6f}")
    
    print(f"\nPearson correlations:")
    for metric, corr in comparison["pearson_corr"].items():
        print(f"  {metric}: {corr:.6f}")
    
    print(f"\nDetailed report saved to: reports/parity_validation.md")
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()