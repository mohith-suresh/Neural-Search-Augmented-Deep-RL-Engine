import numpy as np
import json
import pathlib

# Path to where you output the data
DATA_DIR = pathlib.Path('/home/krish/EE542-Project/FINAL/chess_ai/data/training_data')

def verify():
    print(f"Checking data in {DATA_DIR}...")
    
    # 1. Check Metadata
    with open(DATA_DIR / 'metadata.json', 'r') as f:
        meta = json.load(f)
    
    print(f"\n[Metadata Check]")
    print(f"Total Positions: {meta['count']:,}")
    print(f"Shape: {meta['pos_shape']}")
    
    if meta['pos_shape'][1] == 13:
        print("✅ SUCCESS: 13 Input Channels detected (Turn Indicator is present).")
    else:
        print(f"❌ FAIL: Expected 13 channels, found {meta['pos_shape'][1]}.")
        return

    # 2. Check Binary Data (Memmap)
    positions = np.memmap(
        DATA_DIR / 'positions.bin', 
        dtype=meta['pos_dtype'], 
        mode='r', 
        shape=tuple(meta['pos_shape'])
    )
    
    # 3. Validation Logic
    print(f"\n[Content Check]")
    
    # Check a few samples
    white_turns = 0
    black_turns = 0
    
    for i in range(0, min(100000, len(positions))):
        # The 13th plane is index 12
        turn_plane = positions[i, 12, :, :]
        
        # Mean should be exactly 0.0 (White) or 1.0 (Black)
        plane_mean = np.mean(turn_plane)
        
        if plane_mean == 0.0:
            white_turns += 1
        elif plane_mean == 1.0:
            black_turns += 1
        else:
            print(f"⚠️ Warning at index {i}: 13th plane is not binary! Mean: {plane_mean}")

    print(f"Sampled 100 positions:")
    print(f"  White to move: {white_turns}")
    print(f"  Black to move: {black_turns}")
    
    if white_turns > 0 and black_turns > 0:
        print("✅ SUCCESS: Dataset contains perspectives for both players.")
    else:
        print("⚠️ WARNING: Only one player's turn detected in sample (might be normal if chunk is small/sorted).")

if __name__ == "__main__":
    verify()