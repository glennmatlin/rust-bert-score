#!/usr/bin/env python3
"""
Diagnose PyTorch compatibility issues between Python and Rust.
This helps identify the correct library versions needed.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔍 PyTorch Compatibility Diagnostic Tool")
    print("=" * 50)
    
    # Check Python PyTorch
    try:
        import torch
        print(f"✅ Python PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   PyTorch built with: {torch._C._GLIBCXX_USE_CXX11_ABI}")
        
        # Find library location
        torch_dir = Path(torch.__file__).parent
        lib_dir = torch_dir / "lib"
        print(f"   Library directory: {lib_dir}")
        
        # List key libraries
        if lib_dir.exists():
            libs = sorted([f.name for f in lib_dir.glob("*.so")])[:5]
            print(f"   Key libraries: {', '.join(libs)}")
    except ImportError:
        print("❌ PyTorch not found in Python environment")
        return 1
    
    print("\n" + "=" * 50)
    
    # Check Rust binary dependencies
    rust_binary = "target/release/bert-score"
    if Path(rust_binary).exists():
        print(f"🦀 Rust binary analysis: {rust_binary}")
        
        # Run ldd to check dependencies
        result = subprocess.run(["ldd", rust_binary], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            missing_libs = [line for line in lines if "not found" in line]
            
            if missing_libs:
                print("❌ Missing libraries:")
                for lib in missing_libs:
                    print(f"   {lib.strip()}")
            else:
                print("✅ All libraries found")
                
            # Check for libtorch specifically
            torch_libs = [line for line in lines if "torch" in line.lower()]
            if torch_libs:
                print("\n📚 PyTorch-related libraries:")
                for lib in torch_libs:
                    print(f"   {lib.strip()}")
        else:
            print("❌ Failed to run ldd")
    else:
        print(f"⚠️  Rust binary not found: {rust_binary}")
        print("   Run: cargo build --release --bin bert-score")
    
    print("\n" + "=" * 50)
    
    # Check tch crate expectations
    print("📦 Checking tch crate configuration...")
    cargo_toml = Path("Cargo.toml")
    if cargo_toml.exists():
        content = cargo_toml.read_text()
        if 'tch = ' in content:
            # Extract tch version
            for line in content.split('\n'):
                if line.strip().startswith('tch = '):
                    print(f"   tch version in Cargo.toml: {line.strip()}")
                    break
    
    # Suggest environment setup
    print("\n🔧 Suggested environment setup:")
    print(f"export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH")
    print(f"export LIBTORCH={lib_dir}")
    print(f"export TORCH_CUDA_VERSION=cpu  # or cu118, cu121, etc.")
    
    # Test if it works
    print("\n🧪 Testing Rust CLI with current environment...")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    
    result = subprocess.run(
        [rust_binary, "score", "--help"],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Rust CLI works with Python PyTorch libraries!")
        print("   You can now run the full validation pipeline.")
    else:
        print("❌ Rust CLI failed to run:")
        if result.stderr:
            # Show first few lines of error
            error_lines = result.stderr.strip().split('\n')[:3]
            for line in error_lines:
                print(f"   {line}")
        
        print("\n💡 Troubleshooting suggestions:")
        print("1. Version mismatch: tch crate may expect different PyTorch version")
        print("2. ABI mismatch: C++ ABI incompatibility")
        print("3. Missing dependencies: Check if all .so files are present")
        print("4. Consider using Docker or conda for isolated environment")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())