#!/bin/bash

# Setup Chess AI Environment

# Don't write Python bytecode
export PYTHONDONTWRITEBYTECODE=1

# Activate conda environment
conda activate chessai

# Clean existing cache
echo "Cleaning __pycache__ directories..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo " Environment configured"
echo "PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE"
echo "Python version: $(python3 --version)"
echo "Ready to train!"
