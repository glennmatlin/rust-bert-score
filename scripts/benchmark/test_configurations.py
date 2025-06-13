#!/usr/bin/env python3
"""
Test BERTScore across different configuration combinations.
Ensures Rust implementation matches Python across all settings.
"""

import pandas as pd
import bert_score as bs
import subprocess
import os
import sys
import numpy as np
import itertools
from typing import List, Dict, Tuple
import json

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
os.chdir(project_dir)

# Standard test sentences for configuration testing
TEST_PAIRS = [
    ("The cat sat on the mat.", "A feline was resting on the rug."),
    ("I love programming in Python.", "Python programming is my passion."),
    ("The weather is beautiful today.", "Today's weather is lovely."),
    ("Machine learning is fascinating.", "AI and ML are interesting fields."),
    ("This is a test sentence.", "This is a test sentence."),  # Identical
    ("", ""),  # Both empty
    ("Short", "This is a much longer reference sentence with many words."),
    ("Multiple   spaces   test", "Multiple spaces test"),
]

# Configuration options to test
CONFIGS = {
    "models": ["roberta-large", "bert-base-uncased", "distilbert-base-uncased"],
    "layers": {
        "roberta-large": [17, 16, 12, 8, -1],  # Best layer, one before, middle, early, last
        "bert-base-uncased": [9, 8, 6, 4, -1],
        "distilbert-base-uncased": [5, 4, 3, 2, -1],
    },
    "idf": [True, False],
    "baseline": [True, False],
    "batch_sizes": [1, 8, 32],
}

def get_model_layer_for_python(model_type: str, layer: int) -> int:
    """Convert layer index for Python bert_score (0-indexed vs 1-indexed)."""
    if layer == -1:
        # Last layer - Python uses model-specific defaults
        defaults = {
            "roberta-large": 17,
            "bert-base-uncased": 9,
            "distilbert-base-uncased": 5,
        }
        return defaults.get(model_type, layer)
    return layer

def run_python_bertscore(candidates: List[str], references: List[str],
                        model_type: str, layer: int, use_idf: bool,
                        use_baseline: bool, batch_size: int) -> Dict:
    """Run Python BERTScore with specific configuration."""
    py_layer = get_model_layer_for_python(model_type, layer)
    
    try:
        result = bs.score(
            candidates,
            references,
            model_type=model_type,
            num_layers=py_layer,
            lang="en",
            idf=use_idf,
            rescale_with_baseline=use_baseline,
            batch_size=batch_size,
            verbose=False,
            return_hash=True,
        )
        
        if len(result) == 2:
            (P, R, F1), hash_code = result
        else:
            P, R, F1 = result
            hash_code = "N/A"
            
        return {
            "P": P.numpy(),
            "R": R.numpy(),
            "F1": F1.numpy(),
            "hash": hash_code,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def run_rust_bertscore(test_file: str, output_file: str,
                      model_name: str, layer: int, use_idf: bool,
                      use_baseline: bool) -> pd.DataFrame:
    """Run Rust BERTScore with specific configuration."""
    # Map model names to rust-bert ModelType
    model_type_map = {
        "roberta-large": "roberta",
        "bert-base-uncased": "bert",
        "distilbert-base-uncased": "distilbert",
    }
    
    cmd = [
        "./target/release/bert-score",
        "score",
        "--input-tsv", test_file,
        "--output-csv", output_file,
        "--model-name", model_name,
        "--pretrained", model_name,
    ]
    
    if layer != -1:
        cmd.extend(["--layer", str(layer)])
    
    if use_idf:
        cmd.append("--idf")
    if use_baseline:
        cmd.append("--baseline")
    
    # Set up environment
    env = os.environ.copy()
    env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
    env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Rust BERTScore failed: {result.stderr}")
    
    return pd.read_csv(output_file)

def test_configuration(model: str, layer: int, use_idf: bool, 
                      use_baseline: bool, batch_size: int) -> Dict:
    """Test a specific configuration."""
    config_str = f"{model}_L{layer}_idf={use_idf}_base={use_baseline}_bs={batch_size}"
    print(f"Testing: {config_str}")
    
    # Prepare test data
    test_data = []
    for i, (cand, ref) in enumerate(TEST_PAIRS):
        test_data.append({
            "id": f"C{i:03d}",
            "candidate": cand,
            "reference": ref
        })
    
    df = pd.DataFrame(test_data)
    test_file = "temp_config_test.tsv"
    df[["id", "candidate", "reference"]].to_csv(test_file, sep="\t", index=False)
    
    # Run Python
    candidates = df["candidate"].tolist()
    references = df["reference"].tolist()
    py_result = run_python_bertscore(
        candidates, references, model, layer,
        use_idf, use_baseline, batch_size
    )
    
    if py_result["status"] == "error":
        os.remove(test_file)
        return {
            "config": config_str,
            "status": "py_error",
            "error": py_result["error"]
        }
    
    # Run Rust (batch size doesn't affect Rust CLI currently)
    try:
        rust_output = "temp_rust_config.csv"
        rust_df = run_rust_bertscore(
            test_file, rust_output, model, layer,
            use_idf, use_baseline
        )
    except Exception as e:
        os.remove(test_file)
        return {
            "config": config_str,
            "status": "rust_error",
            "error": str(e)
        }
    
    # Compare results
    diffs = []
    for i, row in df.iterrows():
        rust_row = rust_df[rust_df["id"] == row["id"]].iloc[0]
        
        p_diff = abs(py_result["P"][i] - rust_row["P_rust"])
        r_diff = abs(py_result["R"][i] - rust_row["R_rust"])
        f1_diff = abs(py_result["F1"][i] - rust_row["F1_rust"])
        
        diffs.extend([p_diff, r_diff, f1_diff])
    
    # Clean up
    os.remove(test_file)
    os.remove(rust_output)
    
    max_diff = max(diffs)
    mean_diff = np.mean(diffs)
    
    return {
        "config": config_str,
        "model": model,
        "layer": layer,
        "idf": use_idf,
        "baseline": use_baseline,
        "batch_size": batch_size,
        "status": "success",
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "passed": max_diff < 1e-6,
        "py_hash": py_result.get("hash", "N/A")
    }

def main():
    """Test all configuration combinations."""
    print("🔧 BERTScore Configuration Matrix Testing")
    print("=" * 60)
    
    results = []
    
    # Test each model separately to avoid memory issues
    for model in CONFIGS["models"]:
        print(f"\n\nTesting model: {model}")
        print("-" * 40)
        
        # Get valid layers for this model
        layers = CONFIGS["layers"][model]
        
        # Generate all combinations for this model
        combos = list(itertools.product(
            [model],
            layers,
            CONFIGS["idf"],
            CONFIGS["baseline"],
            CONFIGS["batch_sizes"]
        ))
        
        print(f"Total configurations for {model}: {len(combos)}")
        
        for i, (m, l, idf, base, bs) in enumerate(combos):
            print(f"\n[{i+1}/{len(combos)}] ", end="")
            
            try:
                result = test_configuration(m, l, idf, base, bs)
                results.append(result)
                
                if result["status"] == "success":
                    status = "✓" if result["passed"] else "✗"
                    print(f" {status} max_diff={result['max_diff']:.2e}")
                else:
                    print(f" ERROR: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f" EXCEPTION: {e}")
                results.append({
                    "config": f"{m}_L{l}_idf={idf}_base={base}_bs={bs}",
                    "status": "exception",
                    "error": str(e)
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = "reports/configuration_matrix_results.csv"
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n\n✓ Results saved to {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("CONFIGURATION MATRIX SUMMARY")
    print("="*60)
    
    success_results = results_df[results_df["status"] == "success"]
    if len(success_results) > 0:
        passed = len(success_results[success_results["passed"] == True])
        total = len(success_results)
        
        print(f"\nTotal configurations tested: {len(results)}")
        print(f"Successful runs: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {total-passed} ({(total-passed)/total*100:.1f}%)")
        
        # By model
        print("\nBy Model:")
        for model in CONFIGS["models"]:
            model_results = success_results[success_results["model"] == model]
            if len(model_results) > 0:
                m_passed = len(model_results[model_results["passed"] == True])
                m_total = len(model_results)
                print(f"  {model}: {m_passed}/{m_total} passed ({m_passed/m_total*100:.1f}%)")
        
        # Worst cases
        if total > passed:
            print("\nWorst configurations (max_diff):")
            worst = success_results.nlargest(10, "max_diff")
            for _, w in worst.iterrows():
                if not w["passed"]:
                    print(f"  {w['config']}: {w['max_diff']:.6e}")
    
    # Errors
    error_results = results_df[results_df["status"] != "success"]
    if len(error_results) > 0:
        print(f"\nErrors encountered: {len(error_results)}")
        for _, err in error_results.iterrows():
            print(f"  {err['config']}: {err['status']} - {err.get('error', 'N/A')[:100]}")

if __name__ == "__main__":
    main()