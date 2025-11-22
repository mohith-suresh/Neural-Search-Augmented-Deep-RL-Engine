#!/usr/bin/env python3
"""
Memory Analysis for MCTS Self-Play
===================================

Calculates memory requirements for different configurations
on 8GB RAM system with 4GB GTX GPU.
"""

import numpy as np

class MemoryAnalyzer:
    """Analyze memory requirements for MCTS."""

    def __init__(self):
        self.available_ram = 8 * 1024  # 8GB in MB
        self.available_gpu = 4 * 1024  # 4GB in MB
        self.system_overhead = 2 * 1024  # 2GB for OS

    def calculate_mcts_tree_memory(self, num_simulations: int, branching_factor: int = 35):
        """
        Calculate memory for MCTS tree.

        Args:
            num_simulations: Number of MCTS simulations
            branching_factor: Avg legal moves per position

        Returns:
            Memory in MB
        """
        # Each node stores:
        # - visit_count: 8 bytes (int64)
        # - total_value: 8 bytes (float64)
        # - prior_prob: 4 bytes (float32)
        # - children dict: ~50 bytes overhead + 8 bytes per child
        # - board state: ~100 bytes
        # Total per node: ~200 bytes

        bytes_per_node = 200

        # Estimate total nodes
        # With UCB, tree is unbalanced - not all branches explored equally
        # Approximation: nodes ≈ num_simulations * 1.2 (accounting for reuse)
        total_nodes = num_simulations * 1.2

        total_bytes = total_nodes * bytes_per_node
        return total_bytes / (1024 * 1024)  # Convert to MB

    def calculate_selfplay_game_memory(self, avg_game_length: int = 80):
        """
        Calculate memory for one self-play game.

        Args:
            avg_game_length: Average moves per game

        Returns:
            Memory in MB
        """
        # Each position stores:
        # - position tensor: 12 * 8 * 8 * 4 bytes (float32) = 3072 bytes
        # - policy vector: 8192 * 4 bytes (float32) = 32768 bytes
        # - value: 4 bytes (float32)
        # Total per position: ~36 KB

        bytes_per_position = 36 * 1024
        total_bytes = avg_game_length * bytes_per_position
        return total_bytes / (1024 * 1024)

    def calculate_model_memory(self):
        """
        Calculate CNN model memory on GPU.

        Returns:
            Memory in MB
        """
        # ChessCNN with 10 ResNet blocks, 128 filters:
        # - Input conv: 12 * 128 * 3 * 3 ≈ 14K params
        # - 10 ResNet blocks: 128 * 128 * 3 * 3 * 2 * 10 ≈ 3M params
        # - Policy head: 128 * 32 * 1 * 1 + 2048 * 8192 ≈ 16M params
        # - Value head: 128 * 1 * 1 * 1 + 64 * 256 + 256 * 1 ≈ 17K params
        # Total: ~20M parameters * 4 bytes = 80 MB

        # Add activations, gradients (training): ~3x model size
        # Inference only: ~1.5x model size

        model_params = 20_000_000
        bytes_per_param = 4  # float32
        inference_multiplier = 1.5

        total_bytes = model_params * bytes_per_param * inference_multiplier
        return total_bytes / (1024 * 1024)

    def analyze_configurations(self):
        """Analyze different MCTS configurations."""

        print("="*80)
        print("MEMORY ANALYSIS FOR 8GB RAM + 4GB GPU")
        print("="*80)
        print(f"Available RAM: {self.available_ram} MB")
        print(f"Available GPU: {self.available_gpu} MB")
        print(f"Usable RAM (after OS): {self.available_ram - self.system_overhead} MB")
        print("="*80)

        # Model memory (GPU)
        model_mem = self.calculate_model_memory()
        print(f"\nCNN Model (GPU): {model_mem:.1f} MB")
        print(f"GPU remaining: {self.available_gpu - model_mem:.1f} MB")

        # MCTS configurations
        print("\n" + "="*80)
        print("MCTS TREE MEMORY (RAM)")
        print("="*80)

        sim_counts = [50, 100, 200, 400, 800, 1600]

        for num_sims in sim_counts:
            tree_mem = self.calculate_mcts_tree_memory(num_sims)
            print(f"{num_sims:4d} simulations: {tree_mem:6.1f} MB")

        # Self-play game memory
        print("\n" + "="*80)
        print("SELF-PLAY GAME MEMORY (RAM)")
        print("="*80)

        game_lengths = [40, 60, 80, 100, 120]

        for length in game_lengths:
            game_mem = self.calculate_selfplay_game_memory(length)
            print(f"{length:3d} moves/game: {game_mem:6.1f} MB")

        # Concurrent games analysis
        print("\n" + "="*80)
        print("MAXIMUM CONCURRENT GAMES")
        print("="*80)

        available_ram = self.available_ram - self.system_overhead

        for num_sims in [100, 200, 400]:
            tree_mem = self.calculate_mcts_tree_memory(num_sims)
            game_mem = self.calculate_selfplay_game_memory(80)
            total_per_game = tree_mem + game_mem

            max_concurrent = int(available_ram / total_per_game)

            print(f"{num_sims} sims: {max_concurrent} concurrent games "
                  f"({total_per_game:.1f} MB/game)")

        # Optimal configuration
        print("\n" + "="*80)
        print("RECOMMENDED CONFIGURATIONS")
        print("="*80)

        print("\n1. SEQUENTIAL (1 game at a time):")
        print("   - Simulations: 400-800")
        print("   - Memory: ~100-200 MB per game")
        print("   - Games/hour: 6-12 (depending on sims)")
        print("   - Total games possible: Limited only by time")

        print("\n2. PARALLEL (4 games simultaneously):")
        print("   - Simulations: 100-200 per game")
        print("   - Memory: ~40-80 MB per game = 160-320 MB total")
        print("   - Games/hour: 20-30")
        print("   - Recommended for 8GB RAM")

        print("\n3. BATCH (store games, periodic save):")
        print("   - Generate 10 games in memory")
        print("   - Save to disk, clear memory")
        print("   - Repeat")
        print("   - Can generate unlimited games")

        # GPU batch inference
        print("\n" + "="*80)
        print("GPU BATCH INFERENCE")
        print("="*80)

        batch_sizes = [1, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            # Input: batch_size * 12 * 8 * 8 * 4 bytes
            # Output: batch_size * (8192 + 1) * 4 bytes
            input_mem = batch_size * 12 * 8 * 8 * 4 / (1024 * 1024)
            output_mem = batch_size * 8193 * 4 / (1024 * 1024)
            total_batch_mem = input_mem + output_mem + model_mem

            if total_batch_mem < self.available_gpu:
                print(f"Batch {batch_size:2d}: {total_batch_mem:6.1f} MB (OK)")
            else:
                print(f"Batch {batch_size:2d}: {total_batch_mem:6.1f} MB (TOO LARGE)")


def main():
    analyzer = MemoryAnalyzer()
    analyzer.analyze_configurations()

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
For 8GB RAM + 4GB GTX GPU:

RAM Capacity:
- Can run 1 game with 800 simulations comfortably (~200 MB)
- Can run 4 games with 200 simulations in parallel (~320 MB total)
- Can run 10 games with 100 simulations in parallel (~400 MB total)

GPU Capacity:
- Model fits easily (~120 MB)
- Can batch 16-32 positions for inference (~200-400 MB)
- Leaves room for CUDA kernels and overhead

RECOMMENDATION:
- Use 200 simulations per game (good balance)
- Run 4 games in parallel (maximize throughput)
- Batch neural network inference (8-16 positions)
- Save games every 10 completions (prevent memory buildup)

Expected throughput:
- 4 parallel games × 15 minutes/game = 16 games/hour
- 24 hours = ~384 games = ~30,000 training positions
""")


if __name__ == '__main__':
    main()
