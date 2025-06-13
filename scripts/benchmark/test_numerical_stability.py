#!/usr/bin/env python3
"""
Test numerical stability and edge cases for BERTScore.
Ensures Rust implementation handles numerical edge cases correctly.
"""

import pandas as pd
import bert_score as bs
import subprocess
import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import warnings

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
os.chdir(project_dir)

# Numerical stability test cases
STABILITY_TESTS = {
    "identical": [
        # (candidate, reference, description, expected_f1)
        ("Hello world", "Hello world", "Simple identical", 1.0),
        ("The quick brown fox jumps over the lazy dog.", 
         "The quick brown fox jumps over the lazy dog.", "Pangram identical", 1.0),
        ("", "", "Both empty", 1.0),  # Should handle gracefully
        ("   ", "   ", "Both whitespace only", 1.0),
        ("🍕🍔🌮", "🍕🍔🌮", "Emojis identical", 1.0),
        ("x" * 500, "x" * 500, "Long identical", 1.0),
    ],
    
    "near_identical": [
        ("Hello world", "Hello world!", "One punctuation diff", None),
        ("The cat sat on the mat", "The cat sat on a mat", "One word diff", None),
        ("Hello World", "hello world", "Case difference only", None),
        ("Hello  world", "Hello world", "Extra space", None),
        ("1234567890", "1234567891", "One digit diff", None),
    ],
    
    "completely_different": [
        ("Hello world", "Goodbye universe", "Unrelated English", None),
        ("The sun is shining", "It's raining heavily", "Opposite meaning", None),
        ("ABC123", "XYZ789", "Different alphanumeric", None),
        ("🍕🍔🌮", "🚗🚙🚕", "Different emojis", None),
        ("English text", "中文文本", "Different languages", None),
    ],
    
    "edge_cases": [
        ("", "Non-empty text", "Empty vs non-empty", None),
        ("Non-empty text", "", "Non-empty vs empty", None),
        ("a", "b", "Single different chars", None),
        (".", "!", "Single punctuation", None),
        (" ", ".", "Space vs punctuation", None),
    ],
    
    "potential_numerical_issues": [
        ("0" * 100, "0" * 100, "Many zeros", 1.0),
        ("inf", "inf", "Text 'inf'", 1.0),
        ("nan", "nan", "Text 'nan'", 1.0),
        ("-1.23e-45", "-1.23e-45", "Scientific notation", 1.0),
        ("∞", "∞", "Infinity symbol", 1.0),
    ],
}

def run_determinism_test(candidate: str, reference: str, 
                        num_runs: int = 5) -> Dict:
    """Test if BERTScore is deterministic across multiple runs."""
    py_scores = []
    rust_scores = []
    
    # Create test file
    test_data = pd.DataFrame([{
        "id": "DET001",
        "candidate": candidate,
        "reference": reference
    }])
    test_file = "temp_determinism_test.tsv"
    test_data[["id", "candidate", "reference"]].to_csv(test_file, sep="\t", index=False)
    
    for i in range(num_runs):
        # Python
        result = bs.score(
            [candidate], [reference],
            model_type="roberta-large",
            lang="en",
            idf=False,
            rescale_with_baseline=True,
            verbose=False
        )
        P, R, F1 = result
        py_scores.append({
            "P": P[0].item(),
            "R": R[0].item(),
            "F1": F1[0].item()
        })
        
        # Rust
        rust_output = f"temp_rust_det_{i}.csv"
        cmd = [
            "./target/release/bert-score",
            "score",
            "--input-tsv", test_file,
            "--output-csv", rust_output,
            "--model-name", "roberta-large",
            "--pretrained", "roberta-large",
            "--baseline"
        ]
        
        env = os.environ.copy()
        env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
        env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode == 0:
            rust_df = pd.read_csv(rust_output)
            rust_scores.append({
                "P": rust_df["P_rust"][0],
                "R": rust_df["R_rust"][0],
                "F1": rust_df["F1_rust"][0]
            })
            os.remove(rust_output)
        else:
            rust_scores.append(None)
    
    os.remove(test_file)
    
    # Analyze variance
    py_f1s = [s["F1"] for s in py_scores]
    rust_f1s = [s["F1"] for s in rust_scores if s is not None]
    
    return {
        "py_std": np.std(py_f1s) if py_f1s else None,
        "py_range": max(py_f1s) - min(py_f1s) if py_f1s else None,
        "rust_std": np.std(rust_f1s) if rust_f1s else None,
        "rust_range": max(rust_f1s) - min(rust_f1s) if rust_f1s else None,
        "py_scores": py_scores,
        "rust_scores": rust_scores
    }

def test_numerical_stability(category: str, cases: List[Tuple], 
                           use_baseline: bool = True) -> pd.DataFrame:
    """Test numerical stability for a category of cases."""
    print(f"\n{'='*60}")
    print(f"Testing {category} (baseline={use_baseline})")
    print(f"{'='*60}")
    
    # Prepare test data
    test_data = []
    for i, case in enumerate(cases):
        if len(case) == 4:
            cand, ref, desc, expected = case
        else:
            cand, ref, desc = case
            expected = None
            
        test_data.append({
            "id": f"{category}_{i:03d}",
            "candidate": cand,
            "reference": ref,
            "description": desc,
            "expected_f1": expected
        })
    
    df = pd.DataFrame(test_data)
    test_file = f"temp_stability_{category}.tsv"
    df[["id", "candidate", "reference"]].to_csv(test_file, sep="\t", index=False)
    
    # Run Python BERTScore
    candidates = df["candidate"].tolist()
    references = df["reference"].tolist()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bs.score(
            candidates, references,
            model_type="roberta-large",
            lang="en",
            idf=False,
            rescale_with_baseline=use_baseline,
            verbose=False
        )
    P, R, F1 = result
    py_P, py_R, py_F1 = P.numpy(), R.numpy(), F1.numpy()
    
    # Run Rust BERTScore
    rust_output = f"temp_rust_stability_{category}.csv"
    cmd = [
        "./target/release/bert-score",
        "score",
        "--input-tsv", test_file,
        "--output-csv", rust_output,
        "--model-name", "roberta-large",
        "--pretrained", "roberta-large",
    ]
    if use_baseline:
        cmd.append("--baseline")
    
    env = os.environ.copy()
    env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
    env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Rust error: {result.stderr}")
        os.remove(test_file)
        return pd.DataFrame()
    
    rust_df = pd.read_csv(rust_output)
    
    # Analyze results
    results = []
    for i, row in df.iterrows():
        rust_row = rust_df[rust_df["id"] == row["id"]].iloc[0]
        
        # Check for numerical issues
        py_has_nan = np.isnan(py_F1[i])
        py_has_inf = np.isinf(py_F1[i])
        rust_has_nan = np.isnan(rust_row["F1_rust"])
        rust_has_inf = np.isinf(rust_row["F1_rust"])
        
        # Calculate differences
        if not (py_has_nan or py_has_inf or rust_has_nan or rust_has_inf):
            f1_diff = abs(py_F1[i] - rust_row["F1_rust"])
        else:
            f1_diff = float('inf')
        
        # Check expectations
        if row["expected_f1"] is not None:
            py_meets_expected = abs(py_F1[i] - row["expected_f1"]) < 1e-6
            rust_meets_expected = abs(rust_row["F1_rust"] - row["expected_f1"]) < 1e-6
        else:
            py_meets_expected = None
            rust_meets_expected = None
        
        results.append({
            "id": row["id"],
            "description": row["description"],
            "py_F1": py_F1[i],
            "rust_F1": rust_row["F1_rust"],
            "f1_diff": f1_diff,
            "py_has_nan": py_has_nan,
            "py_has_inf": py_has_inf,
            "rust_has_nan": rust_has_nan,
            "rust_has_inf": rust_has_inf,
            "expected_f1": row["expected_f1"],
            "py_meets_expected": py_meets_expected,
            "rust_meets_expected": rust_meets_expected,
            "status": "PASS" if f1_diff < 1e-6 else "FAIL"
        })
    
    # Clean up
    os.remove(test_file)
    os.remove(rust_output)
    
    return pd.DataFrame(results)

def main():
    """Run all numerical stability tests."""
    print("🔢 BERTScore Numerical Stability Testing")
    print("=" * 60)
    
    all_results = []
    
    # Test with and without baseline
    for use_baseline in [True, False]:
        config_name = "with_baseline" if use_baseline else "without_baseline"
        print(f"\n\n{'='*80}")
        print(f"Configuration: {config_name}")
        print(f"{'='*80}")
        
        for category, cases in STABILITY_TESTS.items():
            results = test_numerical_stability(category, cases, use_baseline)
            if not results.empty:
                results["config"] = config_name
                results["category"] = category
                all_results.append(results)
                
                # Summary
                failed = results[results["status"] == "FAIL"]
                numerical_issues = results[
                    results["py_has_nan"] | results["py_has_inf"] |
                    results["rust_has_nan"] | results["rust_has_inf"]
                ]
                
                print(f"\n{category}: {len(results)-len(failed)}/{len(results)} passed")
                if len(numerical_issues) > 0:
                    print(f"  ⚠️  Numerical issues detected: {len(numerical_issues)} cases")
                
                if len(failed) > 0:
                    print(f"  Failed cases:")
                    for _, fail in failed.iterrows():
                        print(f"    - {fail['description']}: diff={fail['f1_diff']:.6e}")
    
    # Determinism tests
    print(f"\n\n{'='*80}")
    print("Determinism Tests")
    print(f"{'='*80}")
    
    det_tests = [
        ("Hello world", "Hello world", "Simple text"),
        ("The quick brown fox jumps over the lazy dog", 
         "A lazy dog was jumped over by a quick brown fox", "Complex rearrangement"),
        ("", "", "Empty strings"),
    ]
    
    for cand, ref, desc in det_tests:
        print(f"\nTesting determinism for: {desc}")
        det_result = run_determinism_test(cand, ref, num_runs=5)
        
        print(f"  Python std dev: {det_result['py_std']:.2e}")
        print(f"  Python range: {det_result['py_range']:.2e}")
        print(f"  Rust std dev: {det_result['rust_std']:.2e if det_result['rust_std'] else 'N/A'}")
        print(f"  Rust range: {det_result['rust_range']:.2e if det_result['rust_range'] else 'N/A'}")
    
    # Save all results
    if all_results:
        full_results = pd.concat(all_results, ignore_index=True)
        output_file = "reports/numerical_stability_results.csv"
        os.makedirs("reports", exist_ok=True)
        full_results.to_csv(output_file, index=False)
        print(f"\n\n✓ Results saved to {output_file}")
        
        # Final summary
        print("\n" + "="*60)
        print("NUMERICAL STABILITY SUMMARY")
        print("="*60)
        
        total = len(full_results)
        passed = len(full_results[full_results["status"] == "PASS"])
        numerical_issues = len(full_results[
            full_results["py_has_nan"] | full_results["py_has_inf"] |
            full_results["rust_has_nan"] | full_results["rust_has_inf"]
        ])
        
        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {total-passed} ({(total-passed)/total*100:.1f}%)")
        print(f"Numerical issues: {numerical_issues}")
        
        # Check expected values
        expected_tests = full_results[full_results["expected_f1"].notna()]
        if len(expected_tests) > 0:
            py_correct = len(expected_tests[expected_tests["py_meets_expected"] == True])
            rust_correct = len(expected_tests[expected_tests["rust_meets_expected"] == True])
            print(f"\nExpected value tests:")
            print(f"  Python correct: {py_correct}/{len(expected_tests)}")
            print(f"  Rust correct: {rust_correct}/{len(expected_tests)}")

if __name__ == "__main__":
    main()