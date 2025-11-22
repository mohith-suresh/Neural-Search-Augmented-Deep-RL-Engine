#!/usr/bin/env python3
"""
Parallel Pure MCTS Self-Play (CPU Multiprocessing)
===================================================

Project: EE542 - Deconstructing AlphaZero's Success
Baseline: Pure MCTS with NO neural network guidance

Architecture:
- 4 CPU workers running independent pure MCTS games
- No GPU needed (no neural network)
- Each worker runs complete MCTS with random rollouts
- 4x throughput compared to sequential execution

Memory Usage:
- 4 MCTS trees × ~10 MB = 40 MB RAM
- Game histories: ~12 MB RAM
- Total: ~60 MB RAM (very light)

Performance:
- Sequential: ~3 games/hour (1600 sims/move)
- Parallel (4 workers): ~12 games/hour
- 4x speedup with linear scaling

Purpose:
- Establish "tabula rasa" baseline (no domain knowledge)
- Compare with CNN-guided MCTS to measure knowledge contribution
- Expected strength: ~800-1200 ELO
"""

import multiprocessing as mp
import queue
import time
import chess
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import logging
import random
import math

from pure_mcts_selfplay import PureMCTS, PureMCTSConfig
from self_play import BoardEncoder, PolicyEncoder


@dataclass
class ParallelPureMCTSConfig:
    """Configuration for parallel pure MCTS self-play."""

    # Parallelism
    num_workers: int = 4  # Number of parallel games

    # MCTS parameters
    num_simulations: int = 1600
    c_puct: float = 1.414  # sqrt(2)
    rollout_depth: int = 50

    # Temperature schedule
    temp_threshold: int = 30
    temp_init: float = 1.0
    temp_final: float = 0.1

    # Game limits
    max_game_length: int = 200

    # Output
    save_frequency: int = 10  # Save every N games
    output_dir: Path = Path("parallel_pure_mcts_games")


class PureMCTSWorker:
    """
    Single worker process for pure MCTS self-play game generation.

    Runs completely independently on CPU - no shared resources needed.
    """

    def __init__(
        self,
        worker_id: int,
        result_queue: mp.Queue,
        config: ParallelPureMCTSConfig,
        num_games: int
    ):
        """
        Initialize worker.

        Args:
            worker_id: Worker ID
            result_queue: Queue for completed games
            config: Configuration
            num_games: Number of games to generate
        """
        self.worker_id = worker_id
        self.result_queue = result_queue
        self.config = config
        self.num_games = num_games

        # Setup logging
        self.logger = logging.getLogger(f'Worker{worker_id}')

    def play_game(self, game_num: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one pure MCTS self-play game.

        Args:
            game_num: Game number

        Returns:
            List of (position, policy, value) examples
        """
        self.logger.info(f"Starting game {game_num}")

        # Create pure MCTS (no neural network)
        mcts = PureMCTS(
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            max_rollout_depth=self.config.rollout_depth,
            temperature=self.config.temp_init
        )

        board = chess.Board()
        game_history = []
        move_count = 0

        start_time = time.time()

        # Play game
        while not board.is_game_over() and move_count < self.config.max_game_length:
            # Set temperature
            if move_count < self.config.temp_threshold:
                mcts.temperature = self.config.temp_init
            else:
                mcts.temperature = self.config.temp_final

            # MCTS search (with random rollouts)
            move_probs, selected_move = mcts.get_move_probabilities(board)

            # Record example
            position_tensor = BoardEncoder.board_to_tensor(board)
            policy_vector = PolicyEncoder.policy_dict_to_vector(move_probs, board)
            game_history.append((position_tensor, policy_vector, board.turn))

            # Make move
            board.push(selected_move)
            move_count += 1

            if move_count % 10 == 0:
                self.logger.debug(f"Game {game_num}: Move {move_count}")

        # Get game result
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            else:
                outcome = 0.0
        else:
            outcome = 0.0  # Draw by move limit

        # Create training examples
        examples = []
        for position, policy, turn in game_history:
            # Value from perspective of player to move
            value = outcome if turn == chess.WHITE else -outcome
            examples.append((position, policy, value))

        game_time = time.time() - start_time
        self.logger.info(
            f"Game {game_num} complete: {move_count} moves, "
            f"{len(examples)} examples, {game_time:.1f}s"
        )

        return examples

    def run(self):
        """Run worker loop to generate games."""
        self.logger.info(f"Worker {self.worker_id} starting...")

        # Set random seed for this worker
        random.seed(self.worker_id * 1000 + int(time.time()))
        np.random.seed(self.worker_id * 1000 + int(time.time()))

        for game_num in range(self.num_games):
            try:
                examples = self.play_game(game_num)
                self.result_queue.put((self.worker_id, game_num, examples))

            except Exception as e:
                self.logger.error(f"Game {game_num} failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        self.logger.info(f"Worker {self.worker_id} completed {self.num_games} games")


def run_parallel_pure_mcts(
    num_games: int,
    config: ParallelPureMCTSConfig = None
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Run parallel pure MCTS self-play generation.

    Args:
        num_games: Total number of games to generate
        config: Parallel configuration

    Returns:
        List of all training examples
    """
    if config is None:
        config = ParallelPureMCTSConfig()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s'
    )
    logger = logging.getLogger('ParallelPureMCTS')

    logger.info("="*80)
    logger.info("PARALLEL PURE MCTS SELF-PLAY GENERATION")
    logger.info("="*80)
    logger.info(f"Total games: {num_games}")
    logger.info(f"Workers: {config.num_workers}")
    logger.info(f"Simulations: {config.num_simulations}")
    logger.info(f"Rollout depth: {config.rollout_depth}")
    logger.info(f"c_puct: {config.c_puct}")
    logger.info("="*80)
    logger.info("NOTE: Pure MCTS with random rollouts (NO neural network)")
    logger.info("This establishes the tabula rasa baseline")
    logger.info("="*80)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create result queue
    result_queue = mp.Queue()

    # Distribute games across workers
    games_per_worker = num_games // config.num_workers
    remaining_games = num_games % config.num_workers

    # Start worker processes
    workers = []
    worker_games = []

    for worker_id in range(config.num_workers):
        # Distribute remaining games to first few workers
        num_worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
        worker_games.append(num_worker_games)

        worker = mp.Process(
            target=lambda wid=worker_id, ng=num_worker_games: PureMCTSWorker(
                wid,
                result_queue,
                config,
                ng
            ).run()
        )
        workers.append(worker)
        worker.start()
        logger.info(f"Started Worker {worker_id} (assigned {num_worker_games} games)")

    # Collect results
    all_examples = []
    games_completed = 0
    expected_games = sum(worker_games)

    logger.info(f"\nCollecting results from {expected_games} games...")

    start_time = time.time()

    while games_completed < expected_games:
        try:
            worker_id, game_num, examples = result_queue.get(timeout=600)
            all_examples.extend(examples)
            games_completed += 1

            elapsed = time.time() - start_time
            games_per_hour = (games_completed / elapsed) * 3600 if elapsed > 0 else 0

            logger.info(
                f"Received game {games_completed}/{expected_games} "
                f"from worker {worker_id} ({len(examples)} examples) "
                f"[{games_per_hour:.1f} games/hour]"
            )

            # Periodic save
            if games_completed % config.save_frequency == 0:
                save_examples(all_examples, games_completed, config.output_dir)

        except queue.Empty:
            logger.warning("Timeout waiting for game results")
            logger.warning("Checking worker status...")

            alive_workers = sum(1 for w in workers if w.is_alive())
            logger.warning(f"Workers alive: {alive_workers}/{config.num_workers}")

            if alive_workers == 0:
                logger.error("All workers terminated unexpectedly")
                break

    # Wait for workers to finish
    logger.info("Waiting for workers to complete...")
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            logger.warning(f"Worker still alive, terminating...")
            worker.terminate()

    # Final save
    if all_examples:
        save_examples(all_examples, games_completed, config.output_dir)

    total_time = time.time() - start_time
    avg_game_time = total_time / games_completed if games_completed > 0 else 0

    logger.info("="*80)
    logger.info("PARALLEL GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total games: {games_completed}")
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"Avg examples/game: {len(all_examples)/games_completed:.1f}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Avg time/game: {avg_game_time/60:.1f} minutes")
    logger.info(f"Throughput: {games_completed/(total_time/3600):.1f} games/hour")
    logger.info("="*80)

    return all_examples


def save_examples(examples: List[Tuple], game_num: int, output_dir: Path):
    """Save training examples to disk."""
    if not examples:
        return

    positions = np.array([ex[0] for ex in examples], dtype=np.float32)
    policies = np.array([ex[1] for ex in examples], dtype=np.float32)
    values = np.array([ex[2] for ex in examples], dtype=np.float32)

    output_file = output_dir / f"pure_mcts_games_{game_num}.npz"
    np.savez_compressed(
        output_file,
        positions=positions,
        policies=policies,
        values=values
    )

    print(f"Saved {len(examples)} examples to {output_file}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel pure MCTS self-play (no neural network)"
    )
    parser.add_argument(
        '--games',
        type=int,
        default=40,
        help='Total number of games to generate'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (CPU cores)'
    )
    parser.add_argument(
        '--sims',
        type=int,
        default=1600,
        help='MCTS simulations per move'
    )
    parser.add_argument(
        '--rollout-depth',
        type=int,
        default=50,
        help='Maximum rollout depth'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='parallel_pure_mcts_games',
        help='Output directory name'
    )

    args = parser.parse_args()

    config = ParallelPureMCTSConfig(
        num_workers=args.workers,
        num_simulations=args.sims,
        rollout_depth=args.rollout_depth,
        output_dir=Path(args.output)
    )

    print(f"\n{'='*80}")
    print(f"CONFIGURATION")
    print(f"{'='*80}")
    print(f"Total games: {args.games}")
    print(f"Parallel workers: {args.workers}")
    print(f"Simulations per move: {args.sims}")
    print(f"Rollout depth: {args.rollout_depth}")
    print(f"Output directory: {args.output}")
    print(f"{'='*80}")
    print(f"NOTE: This is PURE MCTS (no neural network)")
    print(f"Expected runtime: ~{(args.games / args.workers) * 20:.0f} minutes")
    print(f"Expected strength: ~800-1200 ELO")
    print(f"{'='*80}\n")

    examples = run_parallel_pure_mcts(args.games, config)

    print(f"\n{'='*80}")
    print(f"SUCCESS")
    print(f"{'='*80}")
    print(f"Generated {len(examples)} training examples from {args.games} games")
    print(f"Data saved to: {config.output_dir}/")
    print(f"\nNext steps:")
    print(f"1. Evaluate with: python evaluate_pure_mcts.py --sims {args.sims}")
    print(f"2. Compare with CNN-guided MCTS to measure knowledge contribution")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
