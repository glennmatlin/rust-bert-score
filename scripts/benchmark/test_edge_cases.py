#!/usr/bin/env python3
"""
Comprehensive edge case tests for BERTScore validation.
Tests various text edge cases to ensure Rust implementation matches Python exactly.
"""

import pandas as pd
import bert_score as bs
import subprocess
import os
import sys
import numpy as np
from typing import List, Tuple

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
os.chdir(project_dir)

# Test cases organized by category
EDGE_CASES = {
    "whitespace": [
        # (candidate, reference, description)
        ("Hello world", "Hello world", "Identical text"),
        ("   Hello world   ", "Hello world", "Leading and trailing spaces"),
        ("Hello   world", "Hello world", "Multiple spaces between words"),
        ("\tHello world\t", "Hello world", "Tab characters"),
        ("Hello\nworld", "Hello world", "Newline character"),
        ("Hello\r\nworld", "Hello world", "CRLF characters"),
        ("Hello\u00A0world", "Hello world", "Non-breaking space"),
        ("Hello\u2003world", "Hello world", "Em space"),
        ("   ", "", "Only spaces vs empty"),
        ("", "", "Both empty strings"),
        ("\t\n\r", "", "Mixed whitespace vs empty"),
    ],
    
    "unicode": [
        ("Hello 世界", "Hello 世界", "Chinese characters identical"),
        ("Hello 世界", "Hello world", "Chinese vs English"),
        ("café", "cafe", "Accented characters"),
        ("🍕🍔🌮", "🍕🍔🌮", "Emojis identical"),
        ("I love 🍕!", "I love pizza!", "Emoji vs text"),
        ("Привет мир", "Hello world", "Cyrillic vs English"),
        ("مرحبا بالعالم", "Hello world", "Arabic vs English"),
        ("🇺🇸🇬🇧🇫🇷", "USA UK France", "Flag emojis vs text"),
        ("½ ¼ ¾", "1/2 1/4 3/4", "Unicode fractions"),
        ("α β γ", "alpha beta gamma", "Greek letters"),
    ],
    
    "length": [
        ("a", "a", "Single character"),
        ("I", "I", "Single token"),
        (".", ".", "Single punctuation"),
        ("a" * 512, "a" * 512, "Exactly max length"),
        ("a" * 513, "a" * 513, "Over max length"),
        ("The " * 200, "The " * 200, "Very long repetitive"),
        ("a b c d e f g h i j", "a b c d e f g h i j", "Many short tokens"),
        ("supercalifragilisticexpialidocious", "supercalifragilisticexpialidocious", "Very long single word"),
        ("", "Non-empty reference", "Empty candidate"),
        ("Non-empty candidate", "", "Empty reference"),
    ],
    
    "special_chars": [
        ("Hello, world!", "Hello world", "Punctuation differences"),
        ("Hello... world?", "Hello world", "Multiple punctuation"),
        ("$100.00", "$100.00", "Currency symbols"),
        ("#hashtag @mention", "#hashtag @mention", "Social media symbols"),
        ("email@example.com", "email@example.com", "Email address"),
        ("https://www.example.com", "https://www.example.com", "URL"),
        ("C:\\Users\\file.txt", "C:\\Users\\file.txt", "Windows path"),
        ("/usr/bin/python", "/usr/bin/python", "Unix path"),
        ("(parentheses) [brackets] {braces}", "(parentheses) [brackets] {braces}", "Various brackets"),
        ("\"quotes\" 'apostrophes'", "\"quotes\" 'apostrophes'", "Quote types"),
    ],
    
    "case_sensitivity": [
        ("HELLO WORLD", "hello world", "All caps vs lowercase"),
        ("Hello World", "hello world", "Title case vs lowercase"),
        ("HeLLo WoRLd", "hello world", "Mixed case vs lowercase"),
        ("iPhone", "iphone", "Brand name casing"),
        ("PhD", "phd", "Acronym casing"),
        ("USA", "usa", "All caps acronym"),
        ("McCoy", "mccoy", "Special name casing"),
        ("O'Brien", "o'brien", "Apostrophe in name"),
    ],
    
    "html_entities": [
        ("&amp;", "&", "HTML ampersand"),
        ("&lt;div&gt;", "<div>", "HTML tags"),
        ("&quot;quoted&quot;", "\"quoted\"", "HTML quotes"),
        ("&nbsp;", " ", "HTML non-breaking space"),
        ("&copy;2024", "©2024", "HTML copyright symbol"),
        ("&hearts;", "♥", "HTML heart symbol"),
    ],
    
    "numeric": [
        ("123", "123", "Identical numbers"),
        ("123", "one two three", "Numbers vs words"),
        ("1,000,000", "1000000", "Formatted vs unformatted"),
        ("3.14159", "3.14159", "Decimal numbers"),
        ("1st 2nd 3rd", "first second third", "Ordinals"),
        ("10%", "10 percent", "Percentage symbol"),
        ("2+2=4", "2 + 2 = 4", "Math with spacing"),
    ],
    
    "edge_tokens": [
        ("[CLS]", "[CLS]", "CLS token"),
        ("[SEP]", "[SEP]", "SEP token"),
        ("[PAD]", "[PAD]", "PAD token"),
        ("[UNK]", "[UNK]", "UNK token"),
        ("[MASK]", "[MASK]", "MASK token"),
        ("##ing", "##ing", "Subword token"),
        ("Ġ", "Ġ", "RoBERTa space token"),
    ],
}

def run_python_bertscore(candidates: List[str], references: List[str], 
                        model_type: str = "roberta-large",
                        use_idf: bool = False,
                        use_baseline: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Python BERTScore on the given texts."""
    result = bs.score(
        candidates,
        references,
        model_type=model_type,
        lang="en",
        idf=use_idf,
        rescale_with_baseline=use_baseline,
        batch_size=32,
        verbose=False,
        return_hash=False,
    )
    P, R, F1 = result
    return P.numpy(), R.numpy(), F1.numpy()

def run_rust_bertscore(test_file: str, output_file: str,
                      model_name: str = "roberta-large",
                      use_idf: bool = False,
                      use_baseline: bool = True) -> pd.DataFrame:
    """Run Rust BERTScore on the test file."""
    cmd = [
        "./target/release/bert-score",
        "score",
        "--input-tsv", test_file,
        "--output-csv", output_file,
        "--model-name", model_name,
        "--pretrained", model_name,
    ]
    
    if use_idf:
        cmd.append("--idf")
    if use_baseline:
        cmd.append("--baseline")
    
    # Set up environment
    env = os.environ.copy()
    env["LIBTORCH"] = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
    env["LD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Rust command failed: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Rust BERTScore failed: {result.stderr}")
    
    return pd.read_csv(output_file)

def test_edge_cases(category: str, cases: List[Tuple[str, str, str]], 
                   config_name: str, use_idf: bool = False, 
                   use_baseline: bool = True) -> pd.DataFrame:
    """Test a category of edge cases."""
    print(f"\n{'='*60}")
    print(f"Testing {category} edge cases ({config_name})")
    print(f"{'='*60}")
    
    # Prepare test data
    test_data = []
    for i, (cand, ref, desc) in enumerate(cases):
        test_data.append({
            "id": f"{category}_{i:03d}",
            "candidate": cand,
            "reference": ref,
            "description": desc
        })
    
    df = pd.DataFrame(test_data)
    
    # Save test data
    test_file = f"temp_test_{category}.tsv"
    df[["id", "candidate", "reference"]].to_csv(test_file, sep="\t", index=False)
    
    # Run Python BERTScore
    candidates = df["candidate"].tolist()
    references = df["reference"].tolist()
    py_P, py_R, py_F1 = run_python_bertscore(
        candidates, references, 
        use_idf=use_idf, 
        use_baseline=use_baseline
    )
    
    # Run Rust BERTScore
    rust_output = f"temp_rust_{category}.csv"
    rust_df = run_rust_bertscore(
        test_file, rust_output,
        use_idf=use_idf,
        use_baseline=use_baseline
    )
    
    # Compare results
    results = []
    for i, row in df.iterrows():
        rust_row = rust_df[rust_df["id"] == row["id"]].iloc[0]
        
        p_diff = abs(py_P[i] - rust_row["P_rust"])
        r_diff = abs(py_R[i] - rust_row["R_rust"])
        f1_diff = abs(py_F1[i] - rust_row["F1_rust"])
        max_diff = max(p_diff, r_diff, f1_diff)
        
        results.append({
            "id": row["id"],
            "description": row["description"],
            "candidate": row["candidate"][:50] + "..." if len(row["candidate"]) > 50 else row["candidate"],
            "reference": row["reference"][:50] + "..." if len(row["reference"]) > 50 else row["reference"],
            "py_P": py_P[i],
            "py_R": py_R[i],
            "py_F1": py_F1[i],
            "rust_P": rust_row["P_rust"],
            "rust_R": rust_row["R_rust"],
            "rust_F1": rust_row["F1_rust"],
            "diff_P": p_diff,
            "diff_R": r_diff,
            "diff_F1": f1_diff,
            "max_diff": max_diff,
            "status": "PASS" if max_diff < 1e-6 else "FAIL"
        })
    
    # Clean up temp files
    os.remove(test_file)
    os.remove(rust_output)
    
    return pd.DataFrame(results)

def main():
    """Run all edge case tests."""
    print("🔍 BERTScore Edge Case Testing Suite")
    print("=" * 60)
    
    # Test configurations
    configs = [
        ("default", False, True),    # No IDF, with baseline
        ("idf_only", True, False),   # With IDF, no baseline
        ("idf_baseline", True, True), # Both IDF and baseline
        ("raw", False, False),       # Neither IDF nor baseline
    ]
    
    all_results = []
    
    for config_name, use_idf, use_baseline in configs:
        print(f"\n\n{'='*80}")
        print(f"Configuration: {config_name} (IDF={use_idf}, Baseline={use_baseline})")
        print(f"{'='*80}")
        
        for category, cases in EDGE_CASES.items():
            try:
                results = test_edge_cases(
                    category, cases, config_name,
                    use_idf=use_idf, 
                    use_baseline=use_baseline
                )
                results["config"] = config_name
                all_results.append(results)
                
                # Show summary
                failed = results[results["status"] == "FAIL"]
                print(f"\n{category}: {len(results)-len(failed)}/{len(results)} passed")
                
                if len(failed) > 0:
                    print(f"  Failed cases:")
                    for _, fail in failed.iterrows():
                        print(f"    - {fail['description']}: max_diff={fail['max_diff']:.6e}")
                        print(f"      Py F1={fail['py_F1']:.6f}, Rust F1={fail['rust_F1']:.6f}")
                
            except Exception as e:
                print(f"ERROR in {category}: {e}")
                continue
    
    # Save all results
    if all_results:
        full_results = pd.concat(all_results, ignore_index=True)
        output_file = "reports/edge_case_validation.csv"
        os.makedirs("reports", exist_ok=True)
        full_results.to_csv(output_file, index=False)
        print(f"\n\n✓ Full results saved to {output_file}")
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        for config_name, _, _ in configs:
            config_results = full_results[full_results["config"] == config_name]
            total = len(config_results)
            passed = len(config_results[config_results["status"] == "PASS"])
            print(f"\n{config_name}:")
            print(f"  Total tests: {total}")
            print(f"  Passed: {passed} ({passed/total*100:.1f}%)")
            print(f"  Failed: {total-passed} ({(total-passed)/total*100:.1f}%)")
            
            if total > passed:
                worst = config_results.nlargest(5, "max_diff")
                print(f"  Worst cases:")
                for _, w in worst.iterrows():
                    print(f"    - {w['description']}: {w['max_diff']:.6e}")

if __name__ == "__main__":
    main()