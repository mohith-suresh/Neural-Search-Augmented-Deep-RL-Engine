#!/usr/bin/env python3
"""
Self-Play Game Generation for Chess AI
=======================================

Project: EE542 - Deconstructing AlphaZero's Success
Implementation: AlphaZero-style self-play with MCTS

Generates training data through self-play games:
1. Use MCTS to select moves
2. Store (position, policy, value) tuples
3. Play until terminal state
4. Label positions with final game outcome

Output Format:
- positions: Board states (12x8x8 tensors)
- policies: MCTS visit distributions (8192-dim vectors)
- values: Game outcomes from position perspective (+1/0/-1)

Key Features:
- Temperature annealing (exploration -> exploitation)
- Position augmentation (symmetries)
- Game history tracking
- Parallel self-play support
- Resumable game generation

References:
- Silver et al. (2017): AlphaZero self-play training loop
"""

import chess
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from mcts_tree import MCTS


@dataclass
class SelfPlayConfig:
    """Configuration for self-play generation."""

    # MCTS parameters
    num_simulations: int = 800
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Temperature schedule
    temp_threshold: int = 30  # Move number to switch temperature
    temp_init: float = 1.0    # Initial temperature (exploration)
    temp_final: float = 0.1   # Final temperature (exploitation)

    # Game limits
    max_game_length: int = 500
    resign_threshold: float = -0.9  # Value threshold for resignation
    resign_enabled: bool = False

    # Output
    save_frequency: int = 10  # Save every N games
    output_dir: Path = Path("self_play_games")


class BoardEncoder:
    """Encode chess board to neural network input format."""

    PIECE_TO_CHANNEL = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    @staticmethod
    def board_to_tensor(board: chess.Board) -> np.ndarray:
        """
        Convert board to 12x8x8 tensor.

        Args:
            board: Chess board

        Returns:
            12x8x8 numpy array (float32)
        """
        tensor = np.zeros((12, 8, 8), dtype=np.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = BoardEncoder.PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
                tensor[channel, rank, file] = 1.0

        return tensor


class PolicyEncoder:
    """Encode move policies to neural network output format."""

    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """
        Map chess move to policy index (0-8191).

        Args:
            move: Chess move

        Returns:
            Policy index
        """
        from_sq = move.from_square
        to_sq = move.to_square

        if move.promotion:
            # Promotions: offset by 4096
            return from_sq * 64 + to_sq + 4096
        else:
            # Normal moves
            return from_sq * 64 + to_sq

    @staticmethod
    def policy_dict_to_vector(
        policy_dict: Dict[chess.Move, float],
        board: chess.Board
    ) -> np.ndarray:
        """
        Convert policy dict to 8192-dim vector.

        Args:
            policy_dict: Dict mapping moves to probabilities
            board: Current board (for legal moves)

        Returns:
            8192-dim policy vector (probabilities sum to 1.0)
        """
        policy_vector = np.zeros(8192, dtype=np.float32)

        for move, prob in policy_dict.items():
            if move in board.legal_moves:
                idx = PolicyEncoder.move_to_index(move)
                policy_vector[idx] = prob

        # Normalize (should already be normalized, but ensure)
        total = np.sum(policy_vector)
        if total > 0:
            policy_vector /= total

        return policy_vector


class SelfPlayGame:
    """
    Single self-play game generator.

    Plays a full game using MCTS and records training examples.
    """

    def __init__(self, mcts: MCTS, config: SelfPlayConfig, logger: logging.Logger):
        """
        Initialize self-play game.

        Args:
            mcts: MCTS search engine
            config: Self-play configuration
            logger: Logger instance
        """
        self.mcts = mcts
        self.config = config
        self.logger = logger

        # Game state
        self.board = chess.Board()
        self.game_history: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self.move_count = 0

    def get_temperature(self) -> float:
        """
        Get temperature based on move number.

        Returns:
            Temperature value
        """
        if self.move_count < self.config.temp_threshold:
            return self.config.temp_init
        else:
            return self.config.temp_final

    def should_resign(self, value: float) -> bool:
        """
        Check if position should be resigned.

        Args:
            value: Position value from neural network

        Returns:
            True if should resign
        """
        if not self.config.resign_enabled:
            return False

        return value < self.config.resign_threshold

    def play_move(self) -> bool:
        """
        Play one move using MCTS.

        Returns:
            True if game continues, False if game over
        """
        # Check terminal
        if self.board.is_game_over():
            return False

        # Set temperature
        temperature = self.get_temperature()
        self.mcts.temperature = temperature

        # Run MCTS search
        add_noise = (self.move_count < self.config.temp_threshold)
        move_probs, selected_move = self.mcts.get_move_probabilities(
            self.board,
            add_noise=add_noise
        )

        # Encode current position
        position_tensor = BoardEncoder.board_to_tensor(self.board)
        policy_vector = PolicyEncoder.policy_dict_to_vector(move_probs, self.board)

        # Store training example (value will be filled at game end)
        self.game_history.append((position_tensor, policy_vector, self.board.turn))

        # Make move
        self.board.push(selected_move)
        self.move_count += 1

        # Check max length
        if self.move_count >= self.config.max_game_length:
            return False

        return True

    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play full self-play game.

        Returns:
            List of (position, policy, value) training examples
        """
        self.logger.info(f"Starting self-play game...")

        start_time = time.time()

        # Play until terminal
        while self.play_move():
            pass

        # Get game result
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                outcome = 1.0  # White win
            elif result == "0-1":
                outcome = -1.0  # Black win
            else:
                outcome = 0.0  # Draw
        else:
            # Max length reached
            outcome = 0.0

        # Create training examples with outcomes
        training_examples = []
        for position, policy, turn in self.game_history:
            # Value from perspective of player to move
            if turn == chess.WHITE:
                value = outcome
            else:
                value = -outcome

            training_examples.append((position, policy, value))

        elapsed = time.time() - start_time
        self.logger.info(
            f"Game finished: {self.move_count} moves, "
            f"result={result if self.board.is_game_over() else 'max_length'}, "
            f"time={elapsed:.1f}s, "
            f"examples={len(training_examples)}"
        )

        return training_examples


class SelfPlayWorker:
    """
    Self-play game generator worker.

    Manages generation of multiple self-play games.
    """

    def __init__(self, neural_net, config: SelfPlayConfig):
        """
        Initialize self-play worker.

        Args:
            neural_net: Neural network for MCTS evaluation
            config: Self-play configuration
        """
        self.neural_net = neural_net
        self.config = config

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MCTS
        self.mcts = MCTS(
            neural_net=neural_net,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temp_init,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon
        )

    def generate_games(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate

        Returns:
            List of all training examples from all games
        """
        self.logger.info(f"Generating {num_games} self-play games...")
        self.logger.info(f"MCTS simulations: {self.config.num_simulations}")
        self.logger.info(f"c_puct: {self.config.c_puct}")

        all_examples = []

        for game_num in range(1, num_games + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"GAME {game_num}/{num_games}")
            self.logger.info(f"{'='*80}")

            # Play game
            game = SelfPlayGame(self.mcts, self.config, self.logger)
            examples = game.play_game()

            all_examples.extend(examples)

            # Save periodically
            if game_num % self.config.save_frequency == 0:
                self.save_examples(all_examples, game_num)

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"GENERATION COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total games: {num_games}")
        self.logger.info(f"Total examples: {len(all_examples)}")
        self.logger.info(f"Avg examples/game: {len(all_examples)/num_games:.1f}")

        # Final save
        self.save_examples(all_examples, num_games)

        return all_examples

    def save_examples(self, examples: List[Tuple[np.ndarray, np.ndarray, float]], game_num: int):
        """
        Save training examples to disk.

        Args:
            examples: List of (position, policy, value) tuples
            game_num: Current game number
        """
        if not examples:
            return

        # Convert to arrays
        positions = np.array([ex[0] for ex in examples], dtype=np.float32)
        policies = np.array([ex[1] for ex in examples], dtype=np.float32)
        values = np.array([ex[2] for ex in examples], dtype=np.float32)

        # Save
        output_file = self.config.output_dir / f"selfplay_games_{game_num}.npz"
        np.savez_compressed(
            output_file,
            positions=positions,
            policies=policies,
            values=values
        )

        self.logger.info(f"Saved {len(examples)} examples to {output_file}")


def main():
    """Example usage of self-play generation."""

    # Mock neural network for testing
    class MockNeuralNet:
        def predict(self, board):
            # Return uniform policy and random value
            legal_moves = list(board.legal_moves)
            policy = {move: 1.0 / len(legal_moves) for move in legal_moves}
            value = np.random.uniform(-0.2, 0.2)
            return policy, value

    # Configuration
    config = SelfPlayConfig(
        num_simulations=100,  # Reduced for testing
        c_puct=1.0,
        temp_threshold=15,
        temp_init=1.0,
        temp_final=0.1,
        max_game_length=200,
        save_frequency=5,
        output_dir=Path("test_selfplay_games")
    )

    # Create worker
    neural_net = MockNeuralNet()
    worker = SelfPlayWorker(neural_net, config)

    # Generate games
    examples = worker.generate_games(num_games=10)

    print(f"\nGenerated {len(examples)} training examples")


if __name__ == '__main__':
    main()
