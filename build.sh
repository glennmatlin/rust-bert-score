#!/bin/bash
# Build script for rust-bert-score with Python bindings

set -e

echo "=== Building rust-bert-score ==="

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "maturin not found. Installing with pip..."
    pip install maturin
fi

# Build the Rust library
echo "Building Rust library..."
cargo build --release

# Build Python bindings
echo "Building Python bindings..."
maturin develop --release --features python

echo "✓ Build complete!"
echo ""
echo "To test the Python bindings:"
echo "  cd python && python test_basic.py"
echo ""
echo "To use in Python:"
echo "  from rust_bert_score import BERTScore, score"