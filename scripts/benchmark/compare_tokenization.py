#!/usr/bin/env python3
"""
Compare tokenization between Python and Rust for the "OK" vs "Okay" case.
"""

import pandas as pd
import subprocess
import os
import json

# Create test data
test_data = pd.DataFrame([
    {"id": "T001", "candidate": "OK", "reference": "Okay"},
    {"id": "T002", "candidate": " OK", "reference": " Okay"},
    {"id": "T003", "candidate": "Hello", "reference": "World"},
])

test_file = "temp_tokenization_test.tsv"
test_data.to_csv(test_file, sep="\t", index=False)

# Run Rust with debug tokens output
env = os.environ.copy()
env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"
env["RUST_LOG"] = "debug"  # Enable debug logging

cmd = [
    "./target/release/bert-score",
    "score",
    "--input-tsv", test_file,
    "--output-csv", "temp_rust_tokens.csv",
    "--model-name", "roberta-large",
    "--pretrained", "roberta-large",
    "--lang", "en"
]

print("Running Rust BERTScore to check tokenization...")
result = subprocess.run(cmd, capture_output=True, text=True, env=env)

if result.returncode == 0:
    # Read results
    rust_df = pd.read_csv("temp_rust_tokens.csv")
    print("\nRust results:")
    print(rust_df[["id", "candidate", "reference", "F1_rust"]])
    
    # Compare with expected values
    print("\nExpected similarities:")
    print("'OK' vs 'Okay' with prefix space: ~0.998")
    print("'OK' vs 'Okay' without prefix space: ~0.942")
    
    ok_result = rust_df[rust_df["id"] == "T001"]["F1_rust"].values[0]
    print(f"\nActual Rust result for 'OK' vs 'Okay': {ok_result:.6f}")
    
    if abs(ok_result - 0.998) < 0.01:
        print("✓ Rust is using prefix space (matches Python)")
    elif abs(ok_result - 0.942) < 0.01:
        print("✗ Rust is NOT using prefix space (differs from Python)")
    else:
        print("? Unexpected result")
else:
    print(f"Error: {result.stderr}")

# Clean up
try:
    os.remove(test_file)
    os.remove("temp_rust_tokens.csv")
except:
    pass