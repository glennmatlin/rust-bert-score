#!/usr/bin/env python3
"""Quick comparison of Python and Rust results."""

import pandas as pd
import sys

# Read both CSV files
py_df = pd.read_csv("reports/py_baseline_test.csv")
rust_df = pd.read_csv("reports/rust_baseline_test.csv")

# Compare F1 scores
print("ID       Python F1    Rust F1    Diff")
print("-" * 40)

max_diff = 0
for i in range(len(py_df)):
    py_f1 = py_df.iloc[i]["F1_py"]
    rust_f1 = rust_df.iloc[i]["F1_rust"]
    diff = abs(py_f1 - rust_f1)
    max_diff = max(max_diff, diff)
    
    status = "✓" if diff < 1e-4 else "✗"
    print(f"{py_df.iloc[i]['id']:<8} {py_f1:>10.6f} {rust_f1:>10.6f} {diff:>10.6f} {status}")

print(f"\nMax difference: {max_diff:.6f}")
print("✓ PASS" if max_diff < 1e-4 else "✗ FAIL")