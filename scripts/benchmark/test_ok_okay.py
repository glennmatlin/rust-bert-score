#!/usr/bin/env python3
"""
Test the specific "OK" vs "Okay" case to understand the baseline rescaling.
"""

import bert_score as bs
import pandas as pd
import subprocess
import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
os.chdir(project_dir)

print("Testing 'OK' vs 'Okay' with and without baseline rescaling")
print("="*60)

# Test case
candidate = "OK"
reference = "Okay"

# Test without baseline
result_no_baseline = bs.score(
    [candidate], [reference],
    model_type="roberta-large",
    lang="en",
    idf=False,
    rescale_with_baseline=False,
    verbose=False
)
P_raw, R_raw, F1_raw = result_no_baseline

# Test with baseline
result_with_baseline = bs.score(
    [candidate], [reference],
    model_type="roberta-large",
    lang="en",
    idf=False,
    rescale_with_baseline=True,
    verbose=False
)
P_rescaled, R_rescaled, F1_rescaled = result_with_baseline

print(f"\nPython BERTScore:")
print(f"Without baseline: P={P_raw[0]:.6f}, R={R_raw[0]:.6f}, F1={F1_raw[0]:.6f}")
print(f"With baseline:    P={P_rescaled[0]:.6f}, R={R_rescaled[0]:.6f}, F1={F1_rescaled[0]:.6f}")

# Load baseline value
baseline_path = os.path.join(
    os.path.dirname(bs.__file__),
    "rescale_baseline/en/roberta-large.tsv"
)
baselines_df = pd.read_csv(baseline_path)
layer_17_baseline = baselines_df.iloc[17]  # 0-indexed

print(f"\nBaseline values for layer 17:")
print(f"  F1 baseline: {layer_17_baseline['F']:.8f}")

# Manual calculation
F1_manual_rescaled = (F1_raw[0].item() - layer_17_baseline['F']) / (1 - layer_17_baseline['F'])
print(f"\nManual rescaling calculation:")
print(f"  ({F1_raw[0].item():.6f} - {layer_17_baseline['F']:.6f}) / (1 - {layer_17_baseline['F']:.6f}) = {F1_manual_rescaled:.6f}")
print(f"  Matches Python rescaled? {abs(F1_manual_rescaled - F1_rescaled[0].item()) < 1e-6}")

# Now test Rust implementation
print("\n" + "="*60)
print("Testing Rust implementation:")

# Create test file
test_data = pd.DataFrame([{
    "id": "OK_TEST",
    "candidate": candidate,
    "reference": reference
}])
test_file = "temp_ok_test.tsv"
test_data.to_csv(test_file, sep="\t", index=False)

# Test Rust without baseline
rust_output_no_baseline = "temp_rust_ok_no_baseline.csv"
cmd_no_baseline = [
    "./target/release/bert-score",
    "score",
    "--input-tsv", test_file,
    "--output-csv", rust_output_no_baseline,
    "--model-name", "roberta-large",
    "--pretrained", "roberta-large",
    "--lang", "en"
]

env = os.environ.copy()
env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"

result = subprocess.run(cmd_no_baseline, capture_output=True, text=True, env=env)
if result.returncode == 0:
    rust_df_no_baseline = pd.read_csv(rust_output_no_baseline)
    rust_F1_raw = rust_df_no_baseline["F1_rust"][0]
    print(f"\nRust without baseline: F1={rust_F1_raw:.6f}")
else:
    print(f"Error running Rust (no baseline): {result.stderr}")
    rust_F1_raw = None

# Test Rust with baseline
rust_output_with_baseline = "temp_rust_ok_with_baseline.csv"
cmd_with_baseline = [
    "./target/release/bert-score",
    "score",
    "--input-tsv", test_file,
    "--output-csv", rust_output_with_baseline,
    "--model-name", "roberta-large",
    "--pretrained", "roberta-large",
    "--lang", "en",
    "--baseline"
]

result = subprocess.run(cmd_with_baseline, capture_output=True, text=True, env=env)
if result.returncode == 0:
    rust_df_with_baseline = pd.read_csv(rust_output_with_baseline)
    rust_F1_rescaled = rust_df_with_baseline["F1_rust"][0]
    print(f"Rust with baseline:    F1={rust_F1_rescaled:.6f}")
else:
    print(f"Error running Rust (with baseline): {result.stderr}")
    rust_F1_rescaled = None

# Compare results
print("\n" + "="*60)
print("COMPARISON:")
print(f"Raw F1 scores:")
print(f"  Python: {F1_raw[0]:.6f}")
if rust_F1_raw is not None:
    print(f"  Rust:   {rust_F1_raw:.6f}")
    print(f"  Diff:   {abs(F1_raw[0].item() - rust_F1_raw):.6f}")

print(f"\nRescaled F1 scores:")
print(f"  Python: {F1_rescaled[0]:.6f}")
if rust_F1_rescaled is not None:
    print(f"  Rust:   {rust_F1_rescaled:.6f}")
    print(f"  Diff:   {abs(F1_rescaled[0].item() - rust_F1_rescaled):.6f}")

# Clean up
try:
    os.remove(test_file)
    os.remove(rust_output_no_baseline)
    os.remove(rust_output_with_baseline)
except:
    pass

print("\n" + "="*60)
print("CONCLUSION:")
print("The discrepancy between manual calculation (0.942455) and official")
print("Python score (0.998344) is due to baseline rescaling.")
print(f"Raw score: {F1_raw[0]:.6f} → Rescaled: {F1_rescaled[0]:.6f}")