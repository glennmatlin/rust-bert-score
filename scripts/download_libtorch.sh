#!/bin/bash
# Download pre-built libtorch 2.4.0 for tch 0.17.0 compatibility

echo "📥 Downloading libtorch 2.4.0 for rust-bert-score"
echo "===================================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LIBTORCH_DIR="$PROJECT_DIR/libtorch"

# Check if already downloaded
if [ -d "$LIBTORCH_DIR" ]; then
    echo "✅ libtorch already exists at: $LIBTORCH_DIR"
    echo "   To re-download, remove this directory first."
else
    # Download libtorch 2.4.0 CPU version
    echo "⏬ Downloading libtorch 2.4.0 (CPU)..."
    cd "$PROJECT_DIR"
    
    # URL for libtorch 2.4.0 CPU version
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip"
    
    wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip
    
    if [ $? -eq 0 ]; then
        echo "📦 Extracting libtorch..."
        unzip -q libtorch.zip
        rm libtorch.zip
        echo "✅ libtorch extracted to: $LIBTORCH_DIR"
    else
        echo "❌ Failed to download libtorch"
        exit 1
    fi
fi

# Set environment variables
export LIBTORCH="$LIBTORCH_DIR"
export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH"

echo ""
echo "🔧 Environment configured:"
echo "   LIBTORCH=$LIBTORCH"
echo "   LD_LIBRARY_PATH includes: $LIBTORCH_DIR/lib"

# Build Rust binary
echo ""
echo "🔨 Building rust-bert-score with libtorch 2.4.0..."
cd "$PROJECT_DIR"
cargo build --release --bin bert-score

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Test the binary
    echo ""
    echo "🧪 Testing Rust CLI..."
    RUST_BINARY="$PROJECT_DIR/target/release/bert-score"
    
    if $RUST_BINARY score --help > /dev/null 2>&1; then
        echo "✅ Rust CLI is working!"
        echo ""
        echo "🚀 You can now run the full validation:"
        echo "   export LIBTORCH=\"$LIBTORCH_DIR\""
        echo "   export LD_LIBRARY_PATH=\"$LIBTORCH_DIR/lib:\$LD_LIBRARY_PATH\""
        echo "   ./target/release/bert-score score \\"
        echo "     --input-tsv data/benchmark/direct_eval_pairs.tsv \\"
        echo "     --output-csv reports/direct_scores_rust.csv \\"
        echo "     --pretrained roberta-large \\"
        echo "     --model-type roberta \\"
        echo "     --idf --baseline"
    else
        echo "❌ Rust CLI test failed"
        $RUST_BINARY score --help 2>&1 | head -5
    fi
else
    echo "❌ Build failed"
fi

echo ""
echo "📝 To use this libtorch in future sessions:"
echo "   export LIBTORCH=\"$LIBTORCH_DIR\""
echo "   export LD_LIBRARY_PATH=\"$LIBTORCH_DIR/lib:\$LD_LIBRARY_PATH\""