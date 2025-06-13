#!/usr/bin/env python3
"""
Diagnose differences between Python and Rust BERTScore implementations.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def main():
    print("🔍 Diagnosing BERTScore Implementation Differences")
    print("=" * 50)
    
    # Load both results
    py_df = pd.read_csv("reports/direct_scores_python.csv")
    rust_df = pd.read_csv("reports/direct_scores_rust_real.csv")
    
    # Merge on ID
    df = pd.merge(py_df, rust_df, on=['id', 'candidate', 'reference'], suffixes=('_py', '_rust'))
    
    # Calculate differences
    for metric in ['P', 'R', 'F1']:
        df[f'{metric}_diff'] = df[f'{metric}_py'] - df[f'{metric}_rust']
        df[f'{metric}_abs_diff'] = df[f'{metric}_diff'].abs()
    
    # Sort by worst F1 difference
    df_sorted = df.sort_values('F1_abs_diff', ascending=False)
    
    print("\n📊 Top 5 Worst Cases (by F1 difference):")
    print("-" * 50)
    
    for idx, row in df_sorted.head(5).iterrows():
        print(f"\nID: {row['id']}")
        print(f"Text: '{row['candidate'][:50]}...' → '{row['reference'][:50]}...'")
        print(f"Python  - P: {row['P_py']:.4f}, R: {row['R_py']:.4f}, F1: {row['F1_py']:.4f}")
        print(f"Rust    - P: {row['P_rust']:.4f}, R: {row['R_rust']:.4f}, F1: {row['F1_rust']:.4f}")
        print(f"Diff    - P: {row['P_diff']:.4f}, R: {row['R_diff']:.4f}, F1: {row['F1_diff']:.4f}")
    
    # Analyze patterns
    print("\n📈 Pattern Analysis:")
    print("-" * 50)
    
    # Check if negative scores are more problematic
    rust_negative = df[(df['P_rust'] < 0) | (df['R_rust'] < 0) | (df['F1_rust'] < 0)]
    py_negative = df[(df['P_py'] < 0) | (df['R_py'] < 0) | (df['F1_py'] < 0)]
    
    print(f"Negative scores - Python: {len(py_negative)}, Rust: {len(rust_negative)}")
    
    # Check correlation with text length
    df['cand_len'] = df['candidate'].str.len()
    df['ref_len'] = df['reference'].str.len()
    
    print(f"\nCorrelation of diff with text length:")
    print(f"  Candidate length: {df['F1_abs_diff'].corr(df['cand_len']):.3f}")
    print(f"  Reference length: {df['F1_abs_diff'].corr(df['ref_len']):.3f}")
    
    # Check special cases
    print("\n🔍 Special Cases Analysis:")
    
    # Empty or whitespace
    special_cases = df[
        (df['candidate'].str.strip() == '') | 
        (df['reference'].str.strip() == '') |
        (df['candidate'] != df['candidate'].str.strip()) |
        (df['reference'] != df['reference'].str.strip())
    ]
    
    if len(special_cases) > 0:
        print(f"Found {len(special_cases)} cases with empty/whitespace issues:")
        for idx, row in special_cases.iterrows():
            print(f"  - {row['id']}: F1 diff = {row['F1_diff']:.4f}")
    
    # Perfect scores in Python
    perfect_py = df[(df['P_py'] > 0.99) | (df['R_py'] > 0.99) | (df['F1_py'] > 0.99)]
    print(f"\nPerfect scores (>0.99) in Python: {len(perfect_py)}")
    if len(perfect_py) > 0:
        for idx, row in perfect_py.iterrows():
            print(f"  - {row['id']}: Py F1={row['F1_py']:.4f}, Rust F1={row['F1_rust']:.4f}")
    
    # Statistical summary
    print("\n📊 Statistical Summary:")
    print("-" * 50)
    for metric in ['P', 'R', 'F1']:
        print(f"\n{metric} Score:")
        print(f"  Mean diff: {df[f'{metric}_diff'].mean():.4f}")
        print(f"  Std diff:  {df[f'{metric}_diff'].std():.4f}")
        print(f"  Max diff:  {df[f'{metric}_abs_diff'].max():.4f}")
        print(f"  Correlation: {df[f'{metric}_py'].corr(df[f'{metric}_rust']):.4f}")
    
    # Save detailed analysis
    analysis_file = "reports/detailed_analysis.csv"
    df.to_csv(analysis_file, index=False)
    print(f"\n💾 Detailed analysis saved to: {analysis_file}")
    
    # Hypothesis testing
    print("\n🧪 Hypothesis Testing:")
    print("-" * 50)
    
    # Check if rescaling is the issue
    # BERTScore with baseline rescaling can produce very different results
    print("Possible causes of differences:")
    print("1. ❓ Baseline rescaling parameters differ")
    print("2. ❓ IDF computation differs") 
    print("3. ❓ Tokenization differences (especially for whitespace)")
    print("4. ❓ Model loading or layer selection differs")
    print("5. ❓ Numerical precision in similarity computation")
    
    print("\n💡 Recommendations:")
    print("1. Compare tokenization outputs for problematic cases")
    print("2. Verify baseline rescaling values match")
    print("3. Check IDF weights computation")
    print("4. Ensure same model layers are used")
    print("5. Test without IDF and baseline to isolate issues")

if __name__ == "__main__":
    main()