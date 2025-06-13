#!/usr/bin/env python3
"""
Quick verification that whitespace handling is fixed.
"""

import bert_score as bs
import subprocess
import os
import pandas as pd

# Test cases specifically for whitespace
WHITESPACE_TESTS = [
    ("   Hello world   ", "Hello world", "Leading and trailing spaces"),
    ("\tHello world\t", "Hello world", "Tabs"),
    ("Hello   world", "Hello world", "Multiple internal spaces"),
    ("Hello\nworld", "Hello world", "Newline"),
    ("   ", "", "Spaces vs empty"),
    ("", "", "Both empty"),
]

def main():
    print("🔍 Verifying Whitespace Handling Fix")
    print("=" * 60)
    
    # Create test file
    test_data = []
    for i, (cand, ref, desc) in enumerate(WHITESPACE_TESTS):
        test_data.append({
            "id": f"WS{i:03d}",
            "candidate": cand,
            "reference": ref
        })
    
    df = pd.DataFrame(test_data)
    test_file = "temp_whitespace_test.tsv"
    df.to_csv(test_file, sep="\t", index=False)
    
    # Run Python BERTScore
    print("\nPython BERTScore:")
    candidates = [t[0] for t in WHITESPACE_TESTS]
    references = [t[1] for t in WHITESPACE_TESTS]
    
    result = bs.score(
        candidates, references,
        model_type="roberta-large",
        lang="en",
        idf=False,
        rescale_with_baseline=False,  # Raw scores for clarity
        verbose=False
    )
    P, R, F1 = result
    
    # Run Rust BERTScore
    print("\nRust BERTScore:")
    rust_output = "temp_rust_whitespace.csv"
    
    env = os.environ.copy()
    env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
    env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    cmd = [
        "./target/release/bert-score",
        "score",
        "--input-tsv", test_file,
        "--output-csv", rust_output,
        "--model-name", "roberta-large",
        "--pretrained", "roberta-large",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return
    
    rust_df = pd.read_csv(rust_output)
    
    # Compare results
    print("\nComparison (Raw F1 scores):")
    print("-" * 60)
    print(f"{'Description':<30} {'Python':>10} {'Rust':>10} {'Diff':>10} {'Status':>10}")
    print("-" * 60)
    
    all_pass = True
    for i, (_, _, desc) in enumerate(WHITESPACE_TESTS):
        py_f1 = F1[i].item()
        rust_f1 = rust_df.iloc[i]["F1_rust"]
        diff = abs(py_f1 - rust_f1)
        status = "✓ PASS" if diff < 1e-6 else "✗ FAIL"
        
        if diff >= 1e-6:
            all_pass = False
        
        print(f"{desc:<30} {py_f1:>10.6f} {rust_f1:>10.6f} {diff:>10.2e} {status:>10}")
    
    # Cleanup
    os.remove(test_file)
    os.remove(rust_output)
    
    print("\n" + "="*60)
    if all_pass:
        print("✅ SUCCESS: Whitespace handling is correctly implemented!")
        print("   Rust now matches Python's behavior of stripping whitespace.")
    else:
        print("❌ FAILURE: Whitespace handling still has differences.")
        print("   Further investigation needed.")

if __name__ == "__main__":
    main()