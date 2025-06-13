#!/usr/bin/env python
"""Check the setup status without requiring heavy dependencies."""

import os
import subprocess
import sys

def check_data_files():
    """Check that required data files exist."""
    print("📁 Checking data files...")
    
    checks = [
        ("data/direct_eval_pairs.tsv", "Direct evaluation pairs"),
        ("data/wmt16/ref.txt", "WMT16 reference sentences"),
        ("data/wmt16/human_sys_scores.tsv", "WMT16 human scores"),
    ]
    
    all_good = True
    for path, desc in checks:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {desc}: {path} ({size} bytes)")
        else:
            print(f"  ❌ {desc}: {path} NOT FOUND")
            all_good = False
    
    # Check system files
    sys_dir = "data/wmt16/sys"
    if os.path.exists(sys_dir):
        sys_files = [f for f in os.listdir(sys_dir) if f.endswith('.txt')]
        print(f"  ✅ WMT16 system outputs: {len(sys_files)} files")
        for f in sorted(sys_files)[:3]:
            print(f"     - {f}")
        if len(sys_files) > 3:
            print(f"     ... and {len(sys_files) - 3} more")
    else:
        print(f"  ❌ WMT16 system outputs: {sys_dir} NOT FOUND")
        all_good = False
    
    return all_good

def check_scripts():
    """Check that all scripts exist."""
    print("\n📄 Checking scripts...")
    
    scripts = [
        "scripts/make_direct_set.py",
        "scripts/run_direct_py.py",
        "scripts/run_wmt16_py.py",
        "scripts/compare_direct.py",
        "scripts/compare_wmt16.py",
        "scripts/dump_tokens_py.py",
        "scripts/fetch_wmt16.sh",
    ]
    
    all_good = True
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} NOT FOUND")
            all_good = False
    
    return all_good

def check_python_packages():
    """Check if packages are being installed."""
    print("\n📦 Checking Python packages...")
    
    try:
        # Check if uv pip list works
        result = subprocess.run(
            ["uv", "pip", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"  ℹ️  {len(lines)} packages currently installed")
            
            # Check for key packages
            key_packages = ["bert-score", "torch", "transformers", "pandas"]
            for pkg in key_packages:
                if any(pkg in line for line in lines):
                    print(f"  ✅ {pkg} is installed")
                else:
                    print(f"  ⏳ {pkg} not yet installed")
        else:
            print("  ⚠️  Could not check package list")
            
    except Exception as e:
        print(f"  ⚠️  Error checking packages: {e}")

def main():
    print("🔍 Python Benchmark Setup Status Check")
    print("=" * 50)
    
    # Check current directory
    cwd = os.getcwd()
    if "python-benchmark" not in cwd:
        print(f"⚠️  Not in python-benchmark directory: {cwd}")
        print("   Please run from python-benchmark/")
        return 1
    
    data_ok = check_data_files()
    scripts_ok = check_scripts()
    check_python_packages()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Data files: {'✅ Ready' if data_ok else '❌ Missing'}")
    print(f"  Scripts: {'✅ Ready' if scripts_ok else '❌ Missing'}")
    print("\n💡 Next steps:")
    print("  1. Wait for package installation to complete")
    print("  2. Run: uv run python test_env.py")
    print("  3. Run: uv run python scripts/run_direct_py.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())