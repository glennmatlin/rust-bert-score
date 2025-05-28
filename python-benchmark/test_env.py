#!/usr/bin/env python
"""Test that the Python environment is set up correctly."""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import bert_score
        print("✓ bert-score imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import bert-score: {e}")
        return False
    
    try:
        import torch
        print(f"✓ torch imported successfully (version: {torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ Failed to import torch: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers imported successfully (version: {transformers.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import transformers: {e}")
        return False
    
    try:
        import pandas
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False
    
    try:
        import numpy
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False
    
    return True

def test_data_files():
    """Test that data files exist."""
    print("\nTesting data files...")
    
    import os
    
    files_to_check = [
        "data/direct_eval_pairs.tsv",
        "data/wmt16/ref.txt",
        "data/wmt16/human_sys_scores.tsv",
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    # Check for system files
    if os.path.exists("data/wmt16/sys"):
        sys_files = len(list(os.listdir("data/wmt16/sys")))
        print(f"✓ Found {sys_files} system output files")
    else:
        print("✗ data/wmt16/sys directory not found")
        all_exist = False
    
    return all_exist

def main():
    print("Python Benchmark Environment Test")
    print("=" * 50)
    
    imports_ok = test_imports()
    data_ok = test_data_files()
    
    print("\n" + "=" * 50)
    if imports_ok and data_ok:
        print("✓ Environment is ready!")
        return 0
    else:
        print("✗ Environment setup incomplete")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())