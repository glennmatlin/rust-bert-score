#!/usr/bin/env python
"""
Compute BERTScore on a small ad-hoc test suite using official Python package.
Outputs: reports/direct_scores_python.csv
"""
import pandas as pd
import bert_score as bs
import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)

# Check if data file exists
data_file = "data/direct_eval_pairs.tsv"
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Please run make_direct_set.py first.")
    sys.exit(1)

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Read the test pairs
df = pd.read_csv(data_file, sep="\t")

print(f"Computing BERTScore for {len(df)} sentence pairs...")

# Compute BERTScore
P, R, F1 = bs.score(
    df["candidate"].tolist(),
    df["reference"].tolist(),
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=True,
    batch_size=32,
    verbose=True,
)

# Add scores to dataframe
df["P_py"] = P.numpy()
df["R_py"] = R.numpy()
df["F1_py"] = F1.numpy()

# Save results
output_file = "reports/direct_scores_python.csv"
df.to_csv(output_file, index=False)
print(f"✓ Python direct-set scoring complete — wrote {output_file}")

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"Mean Precision: {df['P_py'].mean():.4f}")
print(f"Mean Recall: {df['R_py'].mean():.4f}")
print(f"Mean F1: {df['F1_py'].mean():.4f}")