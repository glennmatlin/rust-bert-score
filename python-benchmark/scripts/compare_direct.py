#!/usr/bin/env python
"""
Compare direct-set results between Python and Rust CSV outputs.
Outputs simple stats report to stdout + writes reports/direct_agreement.txt
"""
import pandas as pd
import numpy as np
import scipy.stats as ss
import sys
import textwrap
import os
from pathlib import Path

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)

# Check if both CSV files exist
py_file = "reports/direct_scores_python.csv"
rust_file = "reports/direct_scores_rust.csv"

if not os.path.exists(py_file):
    print(f"✗ Python results not found: {py_file}")
    print("  Run: python scripts/run_direct_py.py")
    sys.exit(1)

if not os.path.exists(rust_file):
    print(f"✗ Rust results not found: {rust_file}")
    print("  The Rust implementation needs to generate this file")
    sys.exit(1)

# Load results
p = pd.read_csv(py_file)
r = pd.read_csv(rust_file)

if not (p.id == r.id).all():
    sys.exit("✗ ID mismatch between CSVs")

# Calculate differences for all metrics
metrics = ["P", "R", "F1"]
stats = {}

for metric in metrics:
    py_col = f"{metric}_py"
    rust_col = f"{metric}_rust"
    
    if py_col not in p.columns or rust_col not in r.columns:
        print(f"✗ Missing column: {py_col} or {rust_col}")
        continue
    
    delta = p[py_col] - r[rust_col]
    stats[metric] = {
        "max_abs": delta.abs().max(),
        "mean_abs": delta.abs().mean(),
        "std": delta.std(),
        "pearson": ss.pearsonr(p[py_col], r[rust_col])[0],
        "spearman": ss.spearmanr(p[py_col], r[rust_col])[0],
    }

# Generate report
report_lines = [
    "DIRECT-SET AGREEMENT REPORT",
    "=" * 50,
    f"Samples: {len(p)}",
    "",
]

# Determine pass/fail with tolerance
tolerance = 1e-4
all_pass = True

for metric in metrics:
    if metric in stats:
        s = stats[metric]
        pass_str = "✓ PASS" if s["max_abs"] <= tolerance else "✗ FAIL"
        if s["max_abs"] > tolerance:
            all_pass = False
        
        report_lines.extend([
            f"\n{metric} Score:",
            f"  Max |Δ|     : {s['max_abs']:.6g} {pass_str}",
            f"  Mean |Δ|    : {s['mean_abs']:.6g}",
            f"  Std Dev     : {s['std']:.6g}",
            f"  Pearson r   : {s['pearson']:.6f}",
            f"  Spearman ρ  : {s['spearman']:.6f}",
        ])

# Overall result
report_lines.extend([
    "",
    "=" * 50,
    f"Overall: {'✓ ALL TESTS PASS' if all_pass else '✗ SOME TESTS FAIL'}",
    f"Tolerance: {tolerance}",
])

# Find worst cases
if not all_pass:
    report_lines.extend(["", "Worst Cases:"])
    for metric in metrics:
        if metric in stats:
            py_col = f"{metric}_py"
            rust_col = f"{metric}_rust"
            delta = (p[py_col] - r[rust_col]).abs()
            worst_idx = delta.idxmax()
            worst_row = p.iloc[worst_idx]
            report_lines.extend([
                f"\n{metric} worst case (ID: {worst_row['id']}):",
                f"  Python: {worst_row[py_col]:.6f}",
                f"  Rust:   {r.iloc[worst_idx][rust_col]:.6f}",
                f"  Diff:   {delta.iloc[worst_idx]:.6f}",
                f"  Text:   {worst_row['candidate'][:50]}...",
            ])

report = "\n".join(report_lines)
print(report)

# Save report
Path("reports/direct_agreement.txt").write_text(report)
print(f"\n✓ Report saved to reports/direct_agreement.txt")

# Exit with appropriate code
sys.exit(0 if all_pass else 1)