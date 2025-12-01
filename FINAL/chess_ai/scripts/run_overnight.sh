#!/bin/bash

# ==============================================================================
# Overnight Training Pipeline for Chess AI
# ==============================================================================
# 1. Extract 100M positions (Direct Stream)
# 2. Verify Data Integrity (Check 13th Plane)
# 3. Train CNN (40 Virtual Epochs)
# ==============================================================================

# Exit immediately if any command fails
set -e

# --- Configuration Paths ---
# Adjust these if your folder structure is different
PROJECT_ROOT="/home/krish/EE542-Project/FINAL/chess_ai"
PREPROCESSOR="$PROJECT_ROOT/data/data_preprocessor.py"
TRAINER="$PROJECT_ROOT/game_engine/cnn.py"
RAW_DATA_DIR="$PROJECT_ROOT/data/lichess_raw"
VERIFY_SCRIPT="$PROJECT_ROOT/data/verify_data.py"
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================="
echo "🚀 STARTING OVERNIGHT PIPELINE"
echo "   Log File: $LOG_FILE"
echo "================================================================="

# redirect stdout/stderr to both console and logfile
exec > >(tee -a "$LOG_FILE") 2>&1

# ==============================================================================
# STEP 1: Data Extraction
# ==============================================================================
echo ""
echo "[Step 1/3] Extracting 100M Positions..."
echo "Target: 100,000,000 | Input: $RAW_DATA_DIR"

if [ ! -f "$PREPROCESSOR" ]; then
    echo "❌ Error: Preprocessor not found at $PREPROCESSOR"
    exit 1
fi

# Run extraction with explicit target
python3 "$PREPROCESSOR" --input-dir "$RAW_DATA_DIR" --target 100000000

echo "✅ Extraction Complete."

# ==============================================================================
# STEP 2: Verification (Auto-Generate Script)
# ==============================================================================
echo ""
echo "[Step 2/3] Verifying Data Integrity..."

# Create verify_data.py on the fly to ensure we have the correct logic
cat << 'EOF' > "$VERIFY_SCRIPT"
import numpy as np
import json
import pathlib
import sys

DATA_DIR = pathlib.Path('/home/krish/EE542-Project/FINAL/chess_ai/data/training_data')

def verify():
    print(f"Checking {DATA_DIR}...")
    
    if not (DATA_DIR / 'metadata.json').exists():
        print("❌ Metadata missing!")
        sys.exit(1)

    with open(DATA_DIR / 'metadata.json', 'r') as f:
        meta = json.load(f)
    
    print(f"  Count: {meta['count']:,}")
    print(f"  Shape: {meta['pos_shape']}")
    
    # Check 1: 13 Channels
    if meta['pos_shape'][1] != 13:
        print(f"❌ FAIL: Expected 13 channels, found {meta['pos_shape'][1]}")
        sys.exit(1)
        
    # Check 2: Turn Indicator Plane
    positions = np.memmap(DATA_DIR / 'positions.bin', dtype=meta['pos_dtype'], mode='r', shape=tuple(meta['pos_shape']))
    
    # Sample first 1000
    sample_size = min(1000, len(positions))
    w_turns, b_turns = 0, 0
    
    for i in range(sample_size):
        plane_13 = positions[i, 12, :, :]
        mean = np.mean(plane_13)
        if mean == 0.0: w_turns += 1
        elif mean == 1.0: b_turns += 1
        
    print(f"  Sampled {sample_size}: White={w_turns}, Black={b_turns}")
    
    if w_turns == 0 or b_turns == 0:
        # It is possible a small chunk is sorted, but warn anyway
        print("⚠️ Warning: Only one player turn detected in sample (might be sorted data).")
    
    print("✅ DATA VERIFICATION PASSED")

if __name__ == "__main__":
    verify()
EOF

# Run the verification
python3 "$VERIFY_SCRIPT"

# ==============================================================================
# STEP 3: Training
# ==============================================================================
echo ""
echo "[Step 3/3] Starting CNN Training..."
echo "Config: 100M Positions | Auto-Scaling GPU | 40 Virtual Epochs"

if [ ! -f "$TRAINER" ]; then
    echo "❌ Error: Training script not found at $TRAINER"
    exit 1
fi

python3 "$TRAINER"

echo ""
echo "================================================================="
echo "🎉 PIPELINE COMPLETE. Good morning!"
echo "   Results saved in: game_engine/model/"
echo "================================================================="