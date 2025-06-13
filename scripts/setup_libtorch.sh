#!/bin/bash
# Setup libtorch environment for rust-bert-score

echo "🔧 Setting up libtorch environment for rust-bert-score"
echo "===================================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Find Python PyTorch installation
PYTHON_TORCH_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null)

if [ -z "$PYTHON_TORCH_PATH" ]; then
    echo "❌ PyTorch not found in Python environment"
    echo "   Please run: uv sync"
    exit 1
fi

TORCH_LIB_PATH="$PYTHON_TORCH_PATH/lib"
echo "✅ Found PyTorch at: $PYTHON_TORCH_PATH"
echo "   Library path: $TORCH_LIB_PATH"

# Check PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
echo "   PyTorch version: $TORCH_VERSION"

# Export environment variables for using Python's PyTorch
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"

echo ""
echo "🚀 Environment configured!"
echo "   LIBTORCH_USE_PYTORCH=1"
echo "   LD_LIBRARY_PATH includes: $TORCH_LIB_PATH"

# Test if Rust CLI works now
RUST_BINARY="$PROJECT_DIR/target/release/bert-score"

if [ -f "$RUST_BINARY" ]; then
    echo ""
    echo "🧪 Testing Rust CLI..."
    if $RUST_BINARY score --help > /dev/null 2>&1; then
        echo "✅ Rust CLI is working!"
        echo ""
        echo "You can now run:"
        echo "  $RUST_BINARY score \\"
        echo "    --input-tsv data/benchmark/direct_eval_pairs.tsv \\"
        echo "    --output-csv reports/direct_scores_rust.csv \\"
        echo "    --pretrained roberta-large \\"
        echo "    --model-type roberta \\"
        echo "    --idf --baseline"
    else
        echo "❌ Rust CLI still not working. Error:"
        $RUST_BINARY score --help 2>&1 | head -5
        echo ""
        echo "💡 Try rebuilding with LIBTORCH_USE_PYTORCH=1:"
        echo "   export LIBTORCH_USE_PYTORCH=1"
        echo "   cargo build --release --bin bert-score"
    fi
else
    echo ""
    echo "⚠️  Rust binary not found at: $RUST_BINARY"
    echo "   Build it with:"
    echo "   export LIBTORCH_USE_PYTORCH=1"
    echo "   cargo build --release --bin bert-score"
fi

echo ""
echo "📝 To make these settings permanent, add to your shell profile:"
echo "   export LIBTORCH_USE_PYTORCH=1"
echo "   export LD_LIBRARY_PATH=\"$TORCH_LIB_PATH:\$LD_LIBRARY_PATH\""