#!/usr/bin/env python3
"""
validate_chunks.py -- Robust checker for chunked chess AI dataset

- Checks shapes, dtypes, ranges, and consistency for all chunk_*.npz files
- Reports key metrics for each chunk and overall (move diversity, outcomes, piece statistics)
- Designed for datasets created by your preprocessor with multiple chunk files

Usage:
    python validate_chunks.py /path/to/data/training_data [num_chunks_expected] [positions_expected]

    Example:
    python validate_chunks.py data/training_data 40 20000000
"""

import os
import sys
import glob
import numpy as np

import collections

def main(chunk_dir, num_chunks_expected=40, positions_expected=20_000_000):
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npz")))
    print(f"Found {len(chunk_files)} chunk files in {chunk_dir}")

    if not chunk_files:
        print("ERROR: No chunk files found!")
        sys.exit(1)

    if len(chunk_files) != num_chunks_expected:
        print(f"WARNING: Expected {num_chunks_expected} chunks, found {len(chunk_files)}.")

    total_positions = 0
    all_results = []
    all_moves = []
    move_counts = collections.Counter()
    result_counts = collections.Counter()
    piece_counts = []

    for i, f in enumerate(chunk_files):
        print(f"\n--- Validating {os.path.basename(f)} ({i+1}/{len(chunk_files)}) ---")
        arr = np.load(f)

        # SHAPE AND DTYPES
        for k in ['positions', 'moves', 'results']:
            if k not in arr:
                print(f"ERROR: Missing key '{k}' in {f}")
                sys.exit(1)
        positions = arr['positions']
        moves = arr['moves']
        results = arr['results']

        N = len(positions)
        print(f"  positions: shape={positions.shape}, dtype={positions.dtype}")
        print(f"  moves:     shape={moves.shape},     dtype={moves.dtype}")
        print(f"  results:   shape={results.shape},   dtype={results.dtype}")

        if positions.dtype != np.float16 or results.dtype != np.float16 or moves.dtype != np.int32:
            print(f"  ERROR: Wrong dtype!")
            sys.exit(1)
        if positions.shape != (N, 12, 8, 8):
            print("  ERROR: positions wrong shape!")
            sys.exit(1)
        if moves.shape != (N,):
            print("  ERROR: moves wrong shape!")
            sys.exit(1)
        if results.shape != (N,):
            print("  ERROR: results wrong shape!")
            sys.exit(1)

        # NaN / INF
        for name, arr_ in [('positions', positions), ('results', results)]:
            if np.isnan(arr_).any():
                print(f"  ERROR: NaN in {name}!")
                sys.exit(1)
            if np.isinf(arr_).any():
                print(f"  ERROR: Inf in {name}!")
                sys.exit(1)

        # Moves/range
        minm, maxm = moves.min(), moves.max()
        if not ((0 <= minm) and (maxm <= 8191)):
            print(f"  ERROR: moves index out of range! Min: {minm}, Max: {maxm}")
            sys.exit(1)
        unique_moves = np.unique(moves)
        print(f"  Moves: min={minm}, max={maxm}, unique={len(unique_moves)}")

        # Results
        valid_results = {-1.0, 0.0, 1.0}
        if not set(np.unique(results)).issubset(valid_results):
            print("  ERROR: Invalid result values!")
            sys.exit(1)
        results_ct = collections.Counter(results)
        print(f"  Results: {dict(results_ct)}")

        # Pieces stats (randomly sample 20 positions for king count etc.)
        indices = np.random.choice(N, min(20, N), replace=False)
        for idx in indices:
            pos = positions[idx]
            wk = np.sum(pos[5])  # White king plane must have 1
            bk = np.sum(pos[11]) # Black king plane must have 1
            if wk != 1 or bk != 1:
                print(f'  WARNING: Unusual king count at idx={idx}: WK={wk}, BK={bk}')
            piece_counts.append(pos.sum())

        # Collect stats
        total_positions += N
        all_results.extend(list(results))
        all_moves.extend(list(moves))
        result_counts.update(list(results))
        move_counts.update(list(moves))

    # OVERALL SUMMARY
    print("\n================= OVERALL SUMMARY =================")
    print(f"Total positions: {total_positions} {'(EXPECTED)' if total_positions == positions_expected else '(MISMATCH!)'}")
    print(f"Total chunks:    {len(chunk_files)}")
    print(f"Moves: min={min(all_moves)}, max={max(all_moves)}, unique={len(set(all_moves))}")
    print(f"Results: {dict(result_counts)}")
    print("Result distribution:")
    ct = sum(result_counts.values())
    for r in [-1.0, 0.0, 1.0]:
        pr = 100*result_counts[r]/ct
        print(f"  {r:+.1f}: {result_counts[r]:,} ({pr:5.2f}%)")
    print(f"Avg total piece count per position (sampled): {np.mean(piece_counts):.2f}")
    print(f"Unique move indices overall: {len(set(all_moves))} (diversity: {100*len(set(all_moves))/8192:.2f}% of all possible)")
    print("PASSED: All chunks validated.\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("USAGE: python validate_chunks.py /path/to/data/training_data [num_chunks] [positions_expected]")
        sys.exit(1)
    chunk_dir = sys.argv[1]
    num_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    pos_expected = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000_000
    main(chunk_dir, num_chunks, pos_expected)
