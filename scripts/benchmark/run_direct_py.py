#!/usr/bin/env python
"""
Compute BERTScore on a small ad-hoc test suite using official Python package.
Outputs: reports/direct_scores_python.csv
"""
import pandas as pd
import bert_score as bs
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run Python BERTScore on test data')
    parser.add_argument('--model-type', default='roberta-large', 
                        help='Model type (default: roberta-large)')
    parser.add_argument('--idf', action='store_true', 
                        help='Use IDF weighting')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Disable baseline rescaling')
    parser.add_argument('--output-suffix', default='',
                        help='Suffix for output file (e.g., _idf, _no_baseline)')
    parser.add_argument('--input-tsv', default='data/benchmark/direct_eval_pairs.tsv',
                        help='Input TSV file with candidate and reference columns')
    parser.add_argument('--output-csv', default=None,
                        help='Output CSV file (if not specified, uses reports/direct_scores_python{suffix}.csv)')
    args = parser.parse_args()

    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels: scripts/benchmark -> scripts -> root
    os.chdir(project_dir)

    # Check if data file exists
    data_file = args.input_tsv
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        sys.exit(1)

    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    # Read the test pairs
    df = pd.read_csv(data_file, sep="\t")

    print(f"Computing BERTScore for {len(df)} sentence pairs...")
    print(f"Configuration:")
    print(f"  Model: {args.model_type}")
    print(f"  IDF: {args.idf}")
    print(f"  Baseline rescaling: {not args.no_baseline}")

    # Compute BERTScore
    result = bs.score(
        df["candidate"].tolist(),
        df["reference"].tolist(),
        model_type=args.model_type,
        lang="en",
        idf=args.idf,
        rescale_with_baseline=(not args.no_baseline),
        batch_size=32,
        verbose=True,
        return_hash=True,
    )
    
    # Handle return values based on whether hash is returned
    if len(result) == 2:  # Returns (scores, hash) when return_hash=True
        (P, R, F1), hash_code = result
        print(f"\nHash code: {hash_code}")
    else:  # Returns just scores
        P, R, F1 = result
        hash_code = "N/A"

    # Add scores to dataframe
    df["P_py"] = P.numpy()
    df["R_py"] = R.numpy()
    df["F1_py"] = F1.numpy()

    # Save results
    if args.output_csv:
        output_file = args.output_csv
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_file = f"reports/direct_scores_python{args.output_suffix}.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Python direct-set scoring complete — wrote {output_file}")

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean Precision: {df['P_py'].mean():.4f}")
    print(f"Mean Recall: {df['R_py'].mean():.4f}")
    print(f"Mean F1: {df['F1_py'].mean():.4f}")

if __name__ == "__main__":
    main()