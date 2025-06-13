#!/usr/bin/env python3
"""
Test multiple BERTScore configurations to identify discrepancies.
"""

import subprocess
import os
import sys
import pandas as pd
from pathlib import Path

def run_python_config(name, idf=False, baseline=True):
    """Run Python BERTScore with specific configuration."""
    print(f"\n{'='*60}")
    print(f"Running Python BERTScore: {name}")
    print(f"{'='*60}")
    
    cmd = [
        "uv", "run", "--group", "benchmark",
        "python", "scripts/benchmark/run_direct_py.py",
        f"--output-suffix=_{name}"
    ]
    
    if idf:
        cmd.append("--idf")
    if not baseline:
        cmd.append("--no-baseline")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return False
    
    # Extract hash from output
    for line in result.stdout.split('\n'):
        if 'Hash code:' in line:
            print(f"✅ {line}")
    
    return True

def run_rust_config(name, idf=False, baseline=True):
    """Run Rust BERTScore with specific configuration."""
    print(f"\n{'='*60}")
    print(f"Running Rust BERTScore: {name}")
    print(f"{'='*60}")
    
    # Set up environment
    env = os.environ.copy()
    libtorch_path = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
    env["LIBTORCH"] = libtorch_path
    env["LD_LIBRARY_PATH"] = f"{libtorch_path}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    cmd = [
        "./target/release/bert-score", "score",
        "--input-tsv", "data/benchmark/direct_eval_pairs.tsv",
        "--output-csv", f"reports/direct_scores_rust_{name}.csv",
        "--pretrained", "roberta-large",
        "--model-type", "roberta",
        "--model-name", "roberta-large"
    ]
    
    if idf:
        cmd.append("--idf")
    if baseline:
        cmd.append("--baseline")
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return False
    
    print(f"✅ Success: {result.stdout}")
    return True

def compare_results(config_name):
    """Compare Python and Rust results for a configuration."""
    py_file = f"reports/direct_scores_python_{config_name}.csv"
    rust_file = f"reports/direct_scores_rust_{config_name}.csv"
    
    if not os.path.exists(py_file) or not os.path.exists(rust_file):
        print(f"⚠️  Missing files for {config_name}")
        return
    
    py_df = pd.read_csv(py_file)
    rust_df = pd.read_csv(rust_file)
    
    # Calculate differences
    print(f"\n📊 Comparison for {config_name}:")
    for metric in ['P', 'R', 'F1']:
        py_col = f"{metric}_py"
        rust_col = f"{metric}_rust"
        
        if py_col in py_df.columns and rust_col in rust_df.columns:
            diff = (py_df[py_col] - rust_df[rust_col]).abs()
            print(f"  {metric}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

def main():
    # Change to project root
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)
    
    print("🔍 Testing Multiple BERTScore Configurations")
    print("=" * 60)
    
    # Define configurations to test
    configs = [
        {"name": "baseline_only", "idf": False, "baseline": True},    # Default Python
        {"name": "idf_baseline", "idf": True, "baseline": True},      # What Rust was using
        {"name": "no_baseline_no_idf", "idf": False, "baseline": False},  # Raw scores
        {"name": "idf_only", "idf": True, "baseline": False},         # IDF without baseline
    ]
    
    # Run Python configurations
    print("\n🐍 Running Python BERTScore configurations...")
    for config in configs:
        if not run_python_config(**config):
            print(f"⚠️  Failed to run Python config: {config['name']}")
    
    # Run Rust configurations
    print("\n🦀 Running Rust BERTScore configurations...")
    for config in configs:
        if not run_rust_config(**config):
            print(f"⚠️  Failed to run Rust config: {config['name']}")
    
    # Compare results
    print("\n📈 Comparing Results...")
    print("=" * 60)
    for config in configs:
        compare_results(config["name"])
    
    # Summary
    print("\n📋 Summary")
    print("=" * 60)
    print("Configuration files generated:")
    for config in configs:
        name = config["name"]
        print(f"  - Python: reports/direct_scores_python_{name}.csv")
        print(f"  - Rust: reports/direct_scores_rust_{name}.csv")
    
    print("\n💡 Key Insights:")
    print("1. Compare 'baseline_only' to see if matching Python defaults helps")
    print("2. Compare 'no_baseline_no_idf' to isolate core similarity issues")
    print("3. Check if IDF computation differs between implementations")
    print("4. Check if baseline rescaling differs between implementations")

if __name__ == "__main__":
    main()