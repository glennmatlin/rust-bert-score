#!/bin/bash
# Setup environment variables for rust-bert-score development

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if libtorch exists in the project directory
if [ -d "$SCRIPT_DIR/libtorch" ]; then
    echo "✅ Using local libtorch at: $SCRIPT_DIR/libtorch"
    export LIBTORCH="$SCRIPT_DIR/libtorch"
    export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
    
# Check if libtorch exists in .venv (from PyTorch installation)
elif [ -d "$SCRIPT_DIR/.venv/lib/python3.11/site-packages/torch" ]; then
    TORCH_DIR="$SCRIPT_DIR/.venv/lib/python3.11/site-packages/torch"
    echo "✅ Using PyTorch's libtorch from .venv at: $TORCH_DIR"
    export LIBTORCH="$TORCH_DIR"
    export LD_LIBRARY_PATH="$TORCH_DIR/lib:$LD_LIBRARY_PATH"
    
# Check for system PyTorch
elif command -v python3 &> /dev/null && python3 -c "import torch; print(torch.__file__)" &> /dev/null; then
    TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
    echo "✅ Using system PyTorch's libtorch at: $TORCH_PATH"
    export LIBTORCH="$TORCH_PATH"
    export LD_LIBRARY_PATH="$TORCH_PATH/lib:$LD_LIBRARY_PATH"
else
    echo "⚠️  No libtorch found! Please run one of the following:"
    echo "   1. ./scripts/download_libtorch.sh"
    echo "   2. Install PyTorch in your Python environment"
    echo "   3. Download libtorch manually to ./libtorch/"
    return 1
fi

echo ""
echo "🔧 Environment variables set:"
echo "   LIBTORCH=$LIBTORCH"
echo "   LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""
echo "📝 To make this permanent for your shell session, run:"
echo "   source setup_env.sh"