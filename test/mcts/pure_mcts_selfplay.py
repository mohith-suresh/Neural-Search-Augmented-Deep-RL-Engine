#!/usr/bin/env python3
"""
Pure MCTS Self-Play (NO Neural Network)
========================================

Project: EE542 - Deconstructing AlphaZero's Success
Hypothesis Testing: Pure self-play RL vs Domain Knowledge

This implementation uses PURE MCTS with:
- Random rollouts for position evaluation (no CNN)
- Uniform policy priors (no domain knowledge)
- UCB exploration only
- Self-play game generation

Purpose: Establish baseline for "tabula rasa" learning
Compare to CNN-guided MCTS to measure domain knowledge contribution

Expected Performance:
- Much weaker than CNN-guided MCTS (~800-1200 ELO)
- Requires many more simulations (1600-3200 vs 200-400)
- Slower convergence in training

Research Question:
Does domain knowledge (CNN priors) provide 70-80% of playing strength?
Answer: Compare ELO of pure MCTS vs CNN-guided MCTS
"""

import chess
import numpy as np
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class PureMCTSConfig:
    """Configuration for pure MCTS self-play."""

    # MCTS parameters
    num_simulations: int = 1600  # More sims needed without CNN
    c_puct: float = 1.414  # Standard UCT constant (sqrt(2))

    # Rollout parameters
    max_rollout_depth: int = 50  # Max moves in random rollout

    # Temperature schedule
    temp_threshold: int = 30
    temp_init: float = 1.0
    temp_final: float = 0.1

    # Game limits
    max_game_length: int = 200

    # Output
    save_frequency: int = 10
    output_dir: Path = Path("pure_mcts_games")


class PureMCTSNode:
    """
    MCTS node for pure tree search (no neural network).

    Uses random rollouts for evaluation instead of value network.
    Uses uniform priors instead of policy network.
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional['PureMCTSNode'] = None,
        move: Optional[chess.Move] = None
    ):
        """
        Initialize pure MCTS node.

        Args:
            board: Chess board position
            parent: Parent node
            move: Move from parent to this node
        """
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, 'PureMCTSNode'] = {}

        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0

        # Uniform prior (all legal moves equally likely)
        self.prior_prob = 1.0

        # Expansion state
        self.is_expanded = False

    def is_leaf(self) -> bool:
        """Check if node is leaf."""
        return not self.is_expanded

    def value(self) -> float:
        """Get mean value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float, parent_visits: Optional[int] = None) -> float:
        """
        Calculate UCB score for node selection.

        UCB = Q(s,a) + c_puct * sqrt(ln(N(s)) / N(s,a))

        This is the classic UCT formula (no policy prior).

        Args:
            c_puct: Exploration constant
            parent_visits: Parent visit count

        Returns:
            UCB score
        """
        if parent_visits is None:
            parent_visits = self.parent.visit_count if self.parent else 1

        # Avoid division by zero
        if self.visit_count == 0:
            return float('inf')  # Prioritize unvisited nodes

        # Q-value (exploitation)
        q_value = self.value()

        # UCB exploration term (classic UCT)
        exploration = c_puct * np.sqrt(np.log(parent_visits) / self.visit_count)

        return q_value + exploration

    def select_child(self, c_puct: float) -> 'PureMCTSNode':
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None

        for child in self.children.values():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self) -> None:
        """
        Expand node by creating children for all legal moves.

        No policy network - all moves get uniform prior.
        """
        if self.is_expanded:
            return

        legal_moves = list(self.board.legal_moves)

        for move in legal_moves:
            # Create child board
            child_board = self.board.copy()
            child_board.push(move)

            # Create child node (uniform prior)
            child_node = PureMCTSNode(
                board=child_board,
                parent=self,
                move=move
            )

            self.children[move] = child_node

        self.is_expanded = True

    def backpropagate(self, value: float) -> None:
        """Backpropagate value up tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Negate for opponent
            node = node.parent

    def get_visit_distribution(self, temperature: float = 1.0) -> Dict[chess.Move, float]:
        """Get move probabilities based on visit counts."""
        if not self.children:
            return {}

        moves = []
        visits = []

        for move, child in self.children.items():
            moves.append(move)
            visits.append(child.visit_count)

        visits = np.array(visits, dtype=np.float64)

        if temperature == 0:
            # Deterministic
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Apply temperature
            visits = visits ** (1.0 / temperature)
            total = np.sum(visits)
            probs = visits / total if total > 0 else np.ones_like(visits) / len(visits)

        return dict(zip(moves, probs))


class PureMCTS:
    """
    Pure MCTS with random rollouts (no neural network).

    This is "tabula rasa" learning - no domain knowledge.
    Uses random playouts to estimate position value.
    """

    def __init__(
        self,
        num_simulations: int = 1600,
        c_puct: float = 1.414,
        max_rollout_depth: int = 50,
        temperature: float = 1.0
    ):
        """
        Initialize pure MCTS.

        Args:
            num_simulations: Number of MCTS simulations
            c_puct: Exploration constant (sqrt(2) standard)
            max_rollout_depth: Max depth for random rollouts
            temperature: Temperature for move selection
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_rollout_depth = max_rollout_depth
        self.temperature = temperature

    def random_rollout(self, board: chess.Board) -> float:
        """
        Perform random rollout from position.

        Play random moves until terminal or max depth.

        Args:
            board: Starting position

        Returns:
            Game result from current player's perspective
        """
        rollout_board = board.copy()
        depth = 0

        while not rollout_board.is_game_over() and depth < self.max_rollout_depth:
            legal_moves = list(rollout_board.legal_moves)
            if not legal_moves:
                break

            # Random move
            move = random.choice(legal_moves)
            rollout_board.push(move)
            depth += 1

        # Get result
        if rollout_board.is_game_over():
            result = rollout_board.result()
            if result == "1-0":
                outcome = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                outcome = 1.0 if board.turn == chess.BLACK else -1.0
            else:
                outcome = 0.0  # Draw
        else:
            # Max depth reached - estimate as draw
            outcome = 0.0

        return outcome

    def search(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Run MCTS search from position.

        Args:
            board: Starting position

        Returns:
            Dict mapping moves to visit-based probabilities
        """
        root = PureMCTSNode(board)

        # Expand root
        root.expand()

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)

        # Return visit distribution
        return root.get_visit_distribution(self.temperature)

    def _simulate(self, root: PureMCTSNode) -> None:
        """
        Run single MCTS simulation.

        1. Selection: Traverse using UCB
        2. Expansion: Create children
        3. Rollout: Random playout to terminal
        4. Backpropagation: Update statistics

        Args:
            root: Root node
        """
        node = root
        search_path = [node]

        # 1. Selection - traverse to leaf
        while not node.is_leaf() and not node.board.is_game_over():
            node = node.select_child(self.c_puct)
            search_path.append(node)

        # 2. Terminal check
        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0":
                value = 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                value = 1.0 if node.board.turn == chess.WHITE else -1.0
            else:
                value = 0.0
        else:
            # 3. Expansion and Rollout
            node.expand()

            # Random rollout for evaluation
            value = self.random_rollout(node.board)

        # 4. Backpropagation
        for n in reversed(search_path):
            n.backpropagate(value)
            value = -value

    def get_move_probabilities(
        self,
        board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], chess.Move]:
        """
        Get move probabilities and select move.

        Args:
            board: Chess position

        Returns:
            (move_probabilities, selected_move)
        """
        move_probs = self.search(board)

        # Sample move
        if self.temperature == 0 or not move_probs:
            # Deterministic - most visited
            best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        else:
            # Stochastic
            moves = list(move_probs.keys())
            probs = list(move_probs.values())
            best_move = np.random.choice(moves, p=probs)

        return move_probs, best_move


class PureMCTSSelfPlayGame:
    """
    Self-play game using pure MCTS (no neural network).

    Generates training data from pure self-play reinforcement learning.
    """

    def __init__(self, config: PureMCTSConfig, logger: logging.Logger):
        """
        Initialize self-play game.

        Args:
            config: Configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Game state
        self.board = chess.Board()
        self.game_history: List[Tuple[np.ndarray, Dict[chess.Move, float], chess.Color]] = []
        self.move_count = 0

    def get_temperature(self) -> float:
        """Get temperature based on move number."""
        if self.move_count < self.config.temp_threshold:
            return self.config.temp_init
        else:
            return self.config.temp_final

    def encode_board(self, board: chess.Board) -> np.ndarray:
        """
        Encode board to 12x8x8 tensor.

        Args:
            board: Chess board

        Returns:
            12x8x8 numpy array
        """
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

        tensor = np.zeros((12, 8, 8), dtype=np.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
                tensor[channel, rank, file] = 1.0

        return tensor

    def encode_policy(
        self,
        move_probs: Dict[chess.Move, float],
        board: chess.Board
    ) -> np.ndarray:
        """
        Encode move probabilities to 8192-dim vector.

        Args:
            move_probs: Dict mapping moves to probabilities
            board: Chess board

        Returns:
            8192-dim policy vector
        """
        policy_vector = np.zeros(8192, dtype=np.float32)

        for move, prob in move_probs.items():
            from_sq = move.from_square
            to_sq = move.to_square

            if move.promotion:
                idx = from_sq * 64 + to_sq + 4096
            else:
                idx = from_sq * 64 + to_sq

            policy_vector[idx] = prob

        # Normalize
        total = np.sum(policy_vector)
        if total > 0:
            policy_vector /= total

        return policy_vector

    def play_move(self) -> bool:
        """
        Play one move using pure MCTS.

        Returns:
            True if game continues, False if game over
        """
        if self.board.is_game_over():
            return False

        # Set temperature
        temperature = self.get_temperature()

        # Create MCTS
        mcts = PureMCTS(
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            max_rollout_depth=50,
            temperature=temperature
        )

        # Get move
        move_probs, selected_move = mcts.get_move_probabilities(self.board)

        # Encode position and policy
        position_tensor = self.encode_board(self.board)
        policy_vector = self.encode_policy(move_probs, self.board)

        # Store training example
        self.game_history.append((position_tensor, move_probs, self.board.turn))

        # Make move
        self.board.push(selected_move)
        self.move_count += 1

        # Check limits
        if self.move_count >= self.config.max_game_length:
            return False

        return True

    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play full self-play game with pure MCTS.

        Returns:
            List of (position, policy, value) training examples
        """
        self.logger.info("Starting pure MCTS self-play game...")
        start_time = time.time()

        # Play until terminal
        while self.play_move():
            if self.move_count % 10 == 0:
                self.logger.info(f"Move {self.move_count}...")

        # Get game result
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            else:
                outcome = 0.0
        else:
            outcome = 0.0

        # Create training examples
        training_examples = []
        for position, move_probs, turn in self.game_history:
            # Value from perspective of player to move
            value = outcome if turn == chess.WHITE else -outcome

            # Encode policy
            policy_vector = self.encode_policy(move_probs, self.board)

            training_examples.append((position, policy_vector, value))

        elapsed = time.time() - start_time
        self.logger.info(
            f"Game finished: {self.move_count} moves, "
            f"result={result if self.board.is_game_over() else 'max_length'}, "
            f"time={elapsed:.1f}s, "
            f"examples={len(training_examples)}"
        )

        return training_examples


class PureMCTSSelfPlayWorker:
    """Worker for generating pure MCTS self-play games."""

    def __init__(self, config: PureMCTSConfig):
        """
        Initialize worker.

        Args:
            config: Configuration
        """
        self.config = config

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_games(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate

        Returns:
            List of all training examples
        """
        self.logger.info("="*80)
        self.logger.info("PURE MCTS SELF-PLAY (NO NEURAL NETWORK)")
        self.logger.info("="*80)
        self.logger.info(f"Games: {num_games}")
        self.logger.info(f"Simulations: {self.config.num_simulations}")
        self.logger.info(f"c_puct: {self.config.c_puct}")
        self.logger.info("="*80)

        all_examples = []

        for game_num in range(1, num_games + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"GAME {game_num}/{num_games}")
            self.logger.info(f"{'='*80}")

            # Play game
            game = PureMCTSSelfPlayGame(self.config, self.logger)
            examples = game.play_game()

            all_examples.extend(examples)

            # Save periodically
            if game_num % self.config.save_frequency == 0:
                self.save_examples(all_examples, game_num)

        self.logger.info(f"\n{'='*80}")
        self.logger.info("GENERATION COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total games: {num_games}")
        self.logger.info(f"Total examples: {len(all_examples)}")
        self.logger.info(f"Avg examples/game: {len(all_examples)/num_games:.1f}")

        # Final save
        self.save_examples(all_examples, num_games)

        return all_examples

    def save_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray, float]],
        game_num: int
    ):
        """Save training examples to disk."""
        if not examples:
            return

        positions = np.array([ex[0] for ex in examples], dtype=np.float32)
        policies = np.array([ex[1] for ex in examples], dtype=np.float32)
        values = np.array([ex[2] for ex in examples], dtype=np.float32)

        output_file = self.config.output_dir / f"pure_mcts_games_{game_num}.npz"
        np.savez_compressed(
            output_file,
            positions=positions,
            policies=policies,
            values=values
        )

        self.logger.info(f"Saved {len(examples)} examples to {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pure MCTS Self-Play (No Neural Network)"
    )
    parser.add_argument(
        '--games',
        type=int,
        default=10,
        help='Number of games to generate'
    )
    parser.add_argument(
        '--sims',
        type=int,
        default=1600,
        help='MCTS simulations per move'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pure_mcts_games',
        help='Output directory'
    )

    args = parser.parse_args()

    # Configuration
    config = PureMCTSConfig(
        num_simulations=args.sims,
        c_puct=1.414,
        max_rollout_depth=50,
        temp_threshold=30,
        max_game_length=200,
        save_frequency=5,
        output_dir=Path(args.output)
    )

    # Generate games
    worker = PureMCTSSelfPlayWorker(config)
    examples = worker.generate_games(args.games)

    print(f"\nGenerated {len(examples)} training examples from pure MCTS")


if __name__ == '__main__':
    main()
