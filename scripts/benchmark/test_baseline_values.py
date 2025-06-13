#!/usr/bin/env python3
"""Test to verify baseline values are being applied correctly."""

import bert_score as bs
import pandas as pd
import os

# Test one specific case
candidates = ["The quick brown fox jumps over the lazy dog."]
references = ["The quick brown fox jumps over the lazy dog."]

print("Testing baseline rescaling values...")
print("="*60)

# Test without baseline
result_raw = bs.score(
    candidates, references,
    model_type="roberta-large",
    lang="en",
    idf=False,
    rescale_with_baseline=False,
    verbose=False
)
P_raw, R_raw, F1_raw = result_raw

# Test with baseline
result_rescaled = bs.score(
    candidates, references,
    model_type="roberta-large", 
    lang="en",
    idf=False,
    rescale_with_baseline=True,
    verbose=False
)
P_rescaled, R_rescaled, F1_rescaled = result_rescaled

print(f"Raw scores:      P={P_raw[0]:.6f}, R={R_raw[0]:.6f}, F1={F1_raw[0]:.6f}")
print(f"Rescaled scores: P={P_rescaled[0]:.6f}, R={R_rescaled[0]:.6f}, F1={F1_rescaled[0]:.6f}")

# Load baseline file directly
baseline_path = os.path.join(
    os.path.dirname(bs.__file__),
    "rescale_baseline/en/roberta-large.tsv"
)

baselines_df = pd.read_csv(baseline_path)
layer_17_baseline = baselines_df.iloc[17]  # 0-indexed, so layer 17 is row 17

print(f"\nBaseline values for layer 17:")
print(f"  P baseline: {layer_17_baseline['P']:.8f}")
print(f"  R baseline: {layer_17_baseline['R']:.8f}")
print(f"  F baseline: {layer_17_baseline['F']:.8f}")

# Manual calculation
P_manual = (P_raw[0].item() - layer_17_baseline['P']) / (1 - layer_17_baseline['P'])
R_manual = (R_raw[0].item() - layer_17_baseline['R']) / (1 - layer_17_baseline['R'])
F_manual = (F1_raw[0].item() - layer_17_baseline['F']) / (1 - layer_17_baseline['F'])

print(f"\nManual rescaling: P={P_manual:.6f}, R={R_manual:.6f}, F1={F_manual:.6f}")
print(f"Match Python?     P={abs(P_manual - P_rescaled[0].item()) < 1e-6}, "
      f"R={abs(R_manual - R_rescaled[0].item()) < 1e-6}, "
      f"F={abs(F_manual - F1_rescaled[0].item()) < 1e-6}")