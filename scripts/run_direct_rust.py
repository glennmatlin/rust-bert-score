#!/usr/bin/env python3
"""
Generate BERTScore CSV output using Rust implementation.
This script creates the missing `reports/direct_scores_rust.csv` file.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Change to project directory  
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)
    
    # Ensure input file exists
    input_file = "data/benchmark/direct_eval_pairs.tsv"
    if not Path(input_file).exists():
        print(f"✗ Input file not found: {input_file}")
        print("  Run: uv run --group benchmark python scripts/benchmark/make_direct_set.py")
        sys.exit(1)
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Output file
    output_file = reports_dir / "direct_scores_rust.csv"
    
    print(f"Computing BERTScore using Rust implementation...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Build the CLI tool first
    print("Building Rust CLI...")
    build_result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "bert-score"],
        capture_output=True,
        text=True
    )
    
    if build_result.returncode != 0:
        print("✗ Failed to build Rust CLI:")
        print(build_result.stderr)
        sys.exit(1)
    
    # Run the Rust BERTScore command
    cmd = [
        "./target/release/bert-score",
        "score",
        "--input-tsv", str(input_file),
        "--output-csv", str(output_file),
        "--pretrained", "roberta-large",
        "--model-type", "roberta",
        "--model-name", "roberta-large",
        "--idf",
        "--baseline"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("✗ Rust BERTScore failed:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    
    print("✓ Rust BERTScore completed successfully")
    print(result.stdout)
    
    # Verify output file was created
    if output_file.exists():
        print(f"✓ Output file created: {output_file}")
        
        # Show basic stats
        with open(output_file, 'r') as f:
            lines = f.readlines()
            print(f"  Rows: {len(lines) - 1} (excluding header)")
            if len(lines) > 1:
                print(f"  Header: {lines[0].strip()}")
                print(f"  Sample: {lines[1].strip()}")
    else:
        print(f"✗ Output file not created: {output_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()