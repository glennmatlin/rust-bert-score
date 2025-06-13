#!/usr/bin/env python3
"""
Test that the prefix space fix resolves the tokenization differences.
"""

import pandas as pd
import bert_score as bs
import subprocess
import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
os.chdir(project_dir)

# Read the test data
test_file = "data/benchmark/direct_eval_pairs.tsv"
if not os.path.exists(test_file):
    print(f"Error: {test_file} not found")
    sys.exit(1)

df = pd.read_csv(test_file, sep="\t")

# Run Python BERTScore without baseline
print("Running Python BERTScore...")
candidates = df["candidate"].tolist()
references = df["reference"].tolist()

result = bs.score(
    candidates, references,
    model_type="roberta-large",
    lang="en",
    idf=False,
    rescale_with_baseline=False,
    verbose=False
)
P_py, R_py, F1_py = result

# Run Rust BERTScore without baseline
print("Running Rust BERTScore...")
rust_output = "temp_prefix_test_output.csv"
cmd = [
    "./target/release/bert-score",
    "score",
    "--input-tsv", test_file,
    "--output-csv", rust_output,
    "--model-name", "roberta-large",
    "--pretrained", "roberta-large",
    "--lang", "en"
]

env = os.environ.copy()
env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"

result = subprocess.run(cmd, capture_output=True, text=True, env=env)
if result.returncode != 0:
    print(f"Error running Rust: {result.stderr}")
    sys.exit(1)

rust_df = pd.read_csv(rust_output)

# Compare results
print("\nComparison of F1 scores:")
print("="*80)
print(f"{'ID':<10} {'Candidate':<30} {'Reference':<30} {'Python F1':<10} {'Rust F1':<10} {'Diff':<10}")
print("-"*80)

max_diff = 0
differences = []

for i in range(len(df)):
    py_f1 = F1_py[i].item()
    rust_f1 = rust_df.iloc[i]["F1_rust"]
    diff = abs(py_f1 - rust_f1)
    max_diff = max(max_diff, diff)
    
    differences.append({
        "id": df.iloc[i]["id"],
        "candidate": df.iloc[i]["candidate"][:28],
        "reference": df.iloc[i]["reference"][:28],
        "py_f1": py_f1,
        "rust_f1": rust_f1,
        "diff": diff
    })

# Sort by difference
differences.sort(key=lambda x: x["diff"], reverse=True)

# Show top 10 differences
for d in differences[:10]:
    status = "✓" if d["diff"] < 0.01 else "✗"
    print(f"{d['id']:<10} {d['candidate']:<30} {d['reference']:<30} {d['py_f1']:<10.6f} {d['rust_f1']:<10.6f} {d['diff']:<10.6f} {status}")

print("\nSummary:")
print(f"Maximum difference: {max_diff:.6f}")
print(f"Average difference: {sum(d['diff'] for d in differences) / len(differences):.6f}")

# Count how many are within tolerance
tolerance = 0.01
within_tolerance = sum(1 for d in differences if d["diff"] < tolerance)
print(f"Within tolerance ({tolerance}): {within_tolerance}/{len(differences)} ({within_tolerance/len(differences)*100:.1f}%)")

# Specific check for OK vs Okay
ok_test = next((d for d in differences if d["id"] == "S0012"), None)
if ok_test:
    print(f"\nSpecific test - 'OK' vs 'Okay':")
    print(f"  Python: {ok_test['py_f1']:.6f}")
    print(f"  Rust:   {ok_test['rust_f1']:.6f}")
    print(f"  Diff:   {ok_test['diff']:.6f}")
    print(f"  Status: {'✓ FIXED' if ok_test['diff'] < 0.01 else '✗ NOT FIXED'}")

# Clean up
os.remove(rust_output)

print("\n" + "="*80)
if max_diff < 0.01:
    print("✓ SUCCESS: All scores are within tolerance!")
else:
    print("✗ FAIL: Some scores exceed tolerance")