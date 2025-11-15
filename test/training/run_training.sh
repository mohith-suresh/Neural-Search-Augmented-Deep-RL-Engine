#!/bin/bash

# Training script launcher for single GPU
# Optimized for 4GB GPU RAM

echo "=========================================="
echo "Chess CNN Training - Single GPU"
echo "=========================================="
echo ""

# Check if dataset exists
DATASET_PATH="../outputs/chess_elo1900_500K.npz"

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_PATH"
    echo "Please create the dataset first:"
    echo "  cd ../data"
    echo "  python create_dataset.py"
    exit 1
fi

echo "✓ Dataset found: $DATASET_PATH"
echo ""

# Activate virtual environment
VENV_PATH="../../venv_ee542/bin/python"

if [ ! -f "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found"
    echo "Please create venv first with: uv venv venv_ee542"
    exit 1
fi

echo "✓ Using Python: $VENV_PATH"
echo ""

# Check GPU
echo "Checking GPU availability..."
$VENV_PATH -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run training
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

$VENV_PATH train_single_gpu.py

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
