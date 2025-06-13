#!/usr/bin/env python3
"""Quick test to compare Python and Rust BERTScore implementations."""

import bert_score as bs
import subprocess
import os
import pandas as pd

# Test cases
test_cases = [
    ("Hello world", "Hello world"),
    ("OK", "Okay"),
    ("   Leading and trailing spaces   ", "Leading and trailing spaces"),
    ("", "test"),  # Empty candidate
    ("test", ""),  # Empty reference
    ("", ""),      # Both empty
]

# Run Python BERTScore
print("Running Python BERTScore...")
candidates, references = zip(*test_cases)
py_results = bs.score(
    candidates, references,
    model_type="roberta-large",
    lang="en",
    idf=False,
    rescale_with_baseline=True,
    verbose=False
)
py_precision, py_recall, py_f1 = py_results

# Create test file for Rust
test_df = pd.DataFrame({
    'id': [f'TEST{i:03d}' for i in range(len(test_cases))],
    'candidate': candidates,
    'reference': references
})
test_df.to_csv('temp_test.tsv', sep='\t', index=False)

# Run Rust BERTScore
print("\nRunning Rust BERTScore...")
env = os.environ.copy()
env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"

cmd = [
    "./target/release/bert-score",
    "score",
    "--input-tsv", "temp_test.tsv",
    "--output-csv", "temp_output.csv",
    "--model-name", "roberta-large",
    "--pretrained", "roberta-large",
    "--lang", "en",
    "--baseline"  # Enable baseline rescaling to match Python
]

result = subprocess.run(cmd, capture_output=True, text=True, env=env)
if result.returncode != 0:
    print(f"Error running Rust: {result.stderr}")
    exit(1)

# Read Rust results
rust_df = pd.read_csv("temp_output.csv")

# Compare results
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)
print(f"{'Test Case':<50} {'Python F1':>10} {'Rust F1':>10} {'Diff':>10}")
print("-"*80)

for i, (cand, ref) in enumerate(test_cases):
    py_score = py_f1[i].item()
    rust_score = rust_df.iloc[i]['F1_rust']
    diff = abs(py_score - rust_score)
    
    # Format test case description
    if cand == "" and ref == "":
        desc = "Both empty"
    elif cand == "":
        desc = f"Empty candidate vs '{ref}'"
    elif ref == "":
        desc = f"'{cand}' vs empty reference"
    else:
        desc = f"'{cand[:20]}...' vs '{ref[:20]}...'" if len(cand) > 20 or len(ref) > 20 else f"'{cand}' vs '{ref}'"
    
    print(f"{desc:<50} {py_score:10.6f} {rust_score:10.6f} {diff:10.6f}")

# Clean up
os.remove('temp_test.tsv')
os.remove('temp_output.csv')

print("\nMax difference:", max(abs(py_f1[i].item() - rust_df.iloc[i]['F1_rust']) for i in range(len(test_cases))))