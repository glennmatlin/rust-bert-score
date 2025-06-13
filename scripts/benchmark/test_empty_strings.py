#!/usr/bin/env python3
"""
Test how Python bert_score handles empty strings.
"""

import bert_score as bs
import warnings
import sys

# Capture warnings
warnings.simplefilter("always")

def test_empty_strings():
    """Test various empty string scenarios."""
    test_cases = [
        ([""], [""], "Both empty"),
        ([""], ["Hello world"], "Empty candidate"),
        (["Hello world"], [""], "Empty reference"),
        (["   "], [""], "Whitespace vs empty"),
        ([""], ["   "], "Empty vs whitespace"),
        (["", "Hello"], ["", "World"], "Mixed with empty"),
        (["Hello", ""], ["World", ""], "Mixed with empty at end"),
    ]
    
    print("Testing Python bert_score with empty strings:")
    print("=" * 80)
    
    for candidates, references, desc in test_cases:
        print(f"\n{desc}:")
        print(f"  Candidates: {candidates}")
        print(f"  References: {references}")
        
        # Redirect stderr to capture warnings
        import io
        from contextlib import redirect_stderr
        
        f = io.StringIO()
        with redirect_stderr(f):
            try:
                result = bs.score(
                    candidates,
                    references,
                    model_type="roberta-large",
                    lang="en",
                    idf=False,
                    rescale_with_baseline=True,
                    verbose=False,
                    return_hash=False,
                )
                P, R, F1 = result
                
                for i in range(len(candidates)):
                    print(f"  Result {i}: P={P[i].item():.6f}, R={R[i].item():.6f}, F1={F1[i].item():.6f}")
                    
            except Exception as e:
                print(f"  ERROR: {e}")
        
        # Print any warnings
        warnings_text = f.getvalue()
        if warnings_text:
            print(f"  WARNINGS:\n{warnings_text}")

if __name__ == "__main__":
    test_empty_strings()