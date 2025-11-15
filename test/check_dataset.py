"""
Quick check of dataset move indices
"""
import numpy as np

dataset_path = 'outputs/chess_elo1900_500K.npz'

print("Loading dataset...")
data = np.load(dataset_path)

moves = data['moves']

print(f"Total positions: {len(moves):,}")
print(f"Min move index: {moves.min()}")
print(f"Max move index: {moves.max()}")
print(f"Unique moves: {len(np.unique(moves)):,}")

# Check if any moves exceed 4672
if moves.max() >= 4672:
    print(f"\n⚠️ WARNING: Some moves exceed 4672!")
    print(f"Need output size of at least: {moves.max() + 1}")
else:
    print(f"\n✓ All moves are within range [0, 4671]")
    print(f"Model output size of 4672 is correct")
