# data/verify_dataset.py
"""
Verify dataset quality and show statistics
"""

import numpy as np
from pathlib import Path

def verify_dataset(npz_file):
    """Verify dataset and show statistics"""
    
    print(f"\n{'=' * 70}")
    print(f"VERIFYING: {Path(npz_file).name}")
    print('=' * 70)
    
    # Load
    data = np.load(npz_file, allow_pickle=True)
    positions = data['positions']
    moves = data['moves']
    results = data['results']
    metadata = data['metadata'].item()
    
    # Basic info
    print(f"\n📊 Dataset Info:")
    print(f"  Positions: {len(positions):,}")
    print(f"  Min ELO: {metadata['min_elo']}")
    print(f"  Time control: {metadata['time_control_min']}s+")
    print(f"  Created: {metadata['creation_date']}")
    
    # Array shapes
    print(f"\n📦 Array Shapes:")
    print(f"  Positions: {positions.shape} ({positions.dtype})")
    print(f"  Moves: {moves.shape} ({moves.dtype})")
    print(f"  Results: {results.shape} ({results.dtype})")
    
    # File size
    file_size = Path(npz_file).stat().st_size / (1024**2)
    print(f"\n💾 File Size: {file_size:.1f} MB")
    
    # Result distribution
    white = (results > 0).sum()
    draw = (results == 0).sum()
    black = (results < 0).sum()
    
    print(f"\n⚖️ Results:")
    print(f"  White wins: {white:,} ({white/len(results)*100:.1f}%)")
    print(f"  Draws: {draw:,} ({draw/len(results)*100:.1f}%)")
    print(f"  Black wins: {black:,} ({black/len(results)*100:.1f}%)")
    
    # Quality checks
    print(f"\n✅ Quality Checks:")
    print(f"  No NaN: {'✅' if not np.isnan(positions).any() else '❌'}")
    print(f"  No Inf: {'✅' if not np.isinf(positions).any() else '❌'}")
    print(f"  Pos in [0,1]: {'✅' if (positions >= 0).all() and (positions <= 1).all() else '❌'}")
    print(f"  Results in [-1,1]: {'✅' if (results >= -1).all() and (results <= 1).all() else '❌'}")
    
    # Sample
    print(f"\n🔍 Sample Position:")
    print(f"  Pieces: {np.count_nonzero(positions[0])}")
    print(f"  Move: {moves[0]}")
    print(f"  Result: {results[0]}")
    
    print('=' * 70)


if __name__ == "__main__":
    datasets = [
        "../outputs/chess_elo1900_500K.npz"
        # "../outputs/chess_elo1900_500K.npz",
        # "../outputs/chess_elo1900_3M.npz"
    ]
    
    for dataset in datasets:
        if Path(dataset).exists():
            verify_dataset(dataset)
        else:
            print(f"⚠️ Not found: {dataset}")