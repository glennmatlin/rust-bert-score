#!/usr/bin/env python
"""
Compare WMT16 results between Python and Rust implementations.
Analyzes system-level correlations with human judgments.
"""
import pandas as pd
import scipy.stats as ss
import numpy as np
import os
import sys
from pathlib import Path

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)

# Check for required files
human_file = "data/benchmark/wmt16/human_sys_scores.tsv"
py_file = "reports/wmt16_sys_scores_py.csv"
rust_file = "reports/wmt16_sys_scores_rust.csv"

if not os.path.exists(human_file):
    print(f"✗ Human scores not found: {human_file}")
    sys.exit(1)

if not os.path.exists(py_file):
    print(f"✗ Python results not found: {py_file}")
    print("  Run: python scripts/run_wmt16_py.py")
    sys.exit(1)

if not os.path.exists(rust_file):
    print(f"✗ Rust results not found: {rust_file}")
    print("  The Rust implementation needs to generate this file")
    sys.exit(1)

# Load data
human = pd.read_csv(human_file, sep="\t")
py = pd.read_csv(py_file)
rust = pd.read_csv(rust_file)

# Merge with human scores
py_merged = human.merge(py, on="system")
rust_merged = human.merge(rust, on="system")

# Calculate correlations
py_pearson = ss.pearsonr(py_merged["human_score"], py_merged["mean_F1_py"])[0]
py_spearman = ss.spearmanr(py_merged["human_score"], py_merged["mean_F1_py"])[0]
py_kendall = ss.kendalltau(py_merged["human_score"], py_merged["mean_F1_py"])[0]

rust_pearson = ss.pearsonr(rust_merged["human_score"], rust_merged["mean_F1_rust"])[0]
rust_spearman = ss.spearmanr(rust_merged["human_score"], rust_merged["mean_F1_rust"])[0]
rust_kendall = ss.kendalltau(rust_merged["human_score"], rust_merged["mean_F1_rust"])[0]

# Compare rankings
py_ranked = py_merged.sort_values("mean_F1_py", ascending=False)["system"].tolist()
rust_ranked = rust_merged.sort_values("mean_F1_rust", ascending=False)["system"].tolist()
ranking_agreement = py_ranked == rust_ranked

# Calculate Kendall's tau between Python and Rust rankings
if len(py_ranked) == len(rust_ranked):
    # Create rank dictionaries
    py_ranks = {sys: i for i, sys in enumerate(py_ranked)}
    rust_ranks = {sys: i for i, sys in enumerate(rust_ranked)}
    
    # Get common systems
    common_systems = set(py_ranks.keys()) & set(rust_ranks.keys())
    py_rank_vals = [py_ranks[s] for s in common_systems]
    rust_rank_vals = [rust_ranks[s] for s in common_systems]
    
    rank_correlation = ss.kendalltau(py_rank_vals, rust_rank_vals)[0]
else:
    rank_correlation = None

# Generate report
report_lines = [
    "WMT16 CORRELATION ANALYSIS",
    "=" * 50,
    "",
    "System-level correlation with human judgments:",
    "",
    "Python BERTScore:",
    f"  Pearson r  : {py_pearson:.4f}",
    f"  Spearman ρ : {py_spearman:.4f}",
    f"  Kendall τ  : {py_kendall:.4f}",
    "",
    "Rust BERTScore:",
    f"  Pearson r  : {rust_pearson:.4f}",
    f"  Spearman ρ : {rust_spearman:.4f}",
    f"  Kendall τ  : {rust_kendall:.4f}",
    "",
    "Differences:",
    f"  Δ Pearson  : {abs(py_pearson - rust_pearson):.4f}",
    f"  Δ Spearman : {abs(py_spearman - rust_spearman):.4f}",
    f"  Δ Kendall  : {abs(py_kendall - rust_kendall):.4f}",
    "",
    "=" * 50,
    "System Rankings:",
    "",
]

# Add ranking comparison
report_lines.append("Python ranking:")
for i, sys in enumerate(py_ranked[:5], 1):
    score = py_merged[py_merged["system"] == sys]["mean_F1_py"].iloc[0]
    report_lines.append(f"  {i}. {sys}: {score:.4f}")

report_lines.extend(["", "Rust ranking:"])
for i, sys in enumerate(rust_ranked[:5], 1):
    score = rust_merged[rust_merged["system"] == sys]["mean_F1_rust"].iloc[0]
    report_lines.append(f"  {i}. {sys}: {score:.4f}")

report_lines.extend([
    "",
    f"Ranking agreement: {'✓ YES' if ranking_agreement else '✗ NO'}",
])

if rank_correlation is not None:
    report_lines.append(f"Kendall τ between rankings: {rank_correlation:.4f}")

# Pass/fail criteria
tolerance = 0.002  # As specified in EXPERIMENTS.md
correlations_pass = (
    abs(py_pearson - rust_pearson) <= tolerance and
    abs(py_spearman - rust_spearman) <= tolerance and
    abs(py_kendall - rust_kendall) <= tolerance
)

report_lines.extend([
    "",
    "=" * 50,
    f"Overall: {'✓ PASS' if correlations_pass and ranking_agreement else '✗ FAIL'}",
    f"Correlation tolerance: ±{tolerance}",
])

report = "\n".join(report_lines)
print(report)

# Save report
Path("reports/wmt16_agreement.txt").write_text(report)
print(f"\n✓ Report saved to reports/wmt16_agreement.txt")

# Exit with appropriate code
sys.exit(0 if correlations_pass and ranking_agreement else 1)