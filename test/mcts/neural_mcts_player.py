#!/usr/bin/env python3
"""
Neural Network + MCTS Chess Player
===================================

Project: EE542 - Deconstructing AlphaZero's Success
Integration: Combines trained CNN with MCTS search

This module integrates:
1. Trained CNN (from cnn.py) for position evaluation
2. MCTS (from mcts_tree.py) for move selection
3. Policy/value prediction from neural network

Usage:
    player = NeuralMCTSPlayer(model_path="model/best_model.pth")
    move = player.select_move(board)

Components:
- NeuralNetWrapper: Adapts PyTorch CNN to MCTS interface
- NeuralMCTSPlayer: Complete chess player with MCTS
- Evaluation utilities for testing player strength
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import chess
from typing import Dict, Tuple, Optional, List
import logging

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "FINAL" / "chess_ai"))

from cnn import ChessCNN, TrainingConfig
from mcts_tree import MCTS


class BoardEncoder:
    """Encode chess board to CNN input format."""

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
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        """
        Convert board to CNN input tensor.

        Args:
            board: Chess board

        Returns:
            (1, 12, 8, 8) tensor
        """
        tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = BoardEncoder.PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
                tensor[channel, rank, file] = 1.0

        return tensor.unsqueeze(0)  # Add batch dimension


class MoveEncoder:
    """Encode/decode moves to policy indices."""

    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """Convert move to policy index (0-8191)."""
        from_sq = move.from_square
        to_sq = move.to_square

        if move.promotion:
            return from_sq * 64 + to_sq + 4096
        else:
            return from_sq * 64 + to_sq

    @staticmethod
    def index_to_move(index: int) -> chess.Move:
        """Convert policy index to move (approximate)."""
        if index >= 4096:
            # Promotion
            index -= 4096
            from_sq = index // 64
            to_sq = index % 64
            return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        else:
            # Normal move
            from_sq = index // 64
            to_sq = index % 64
            return chess.Move(from_sq, to_sq)


class NeuralNetWrapper:
    """
    Wrapper for PyTorch CNN to provide MCTS interface.

    Converts between:
    - chess.Board <-> CNN tensor input
    - CNN policy logits <-> move probability dict
    - CNN value output <-> position evaluation
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize neural net wrapper.

        Args:
            model: Trained PyTorch CNN model
            device: Device for inference ("cuda" or "cpu")
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

    @torch.no_grad()
    def predict(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """
        Predict policy and value for position.

        Args:
            board: Chess position

        Returns:
            (policy_dict, value)
            - policy_dict: {move: probability} for legal moves
            - value: Position evaluation (-1 to +1)
        """
        # Encode board
        board_tensor = self.encoder.board_to_tensor(board).to(self.device)

        # Forward pass
        policy_logits, value = self.model(board_tensor)

        # Convert to probabilities
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.item()

        # Filter to legal moves
        legal_moves = list(board.legal_moves)
        move_probs = {}

        for move in legal_moves:
            move_idx = self.move_encoder.move_to_index(move)
            move_probs[move] = policy_probs[move_idx]

        # Normalize (ensure probabilities sum to 1.0)
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        else:
            # Fallback to uniform
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {move: uniform_prob for move in legal_moves}

        return move_probs, value


class NeuralMCTSPlayer:
    """
    Complete chess player combining neural network and MCTS.

    Usage:
        player = NeuralMCTSPlayer(model_path="best_model.pth")
        move = player.select_move(board)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 0.0,
        device: str = "cuda"
    ):
        """
        Initialize neural MCTS player.

        Args:
            model_path: Path to trained model checkpoint
            model: Pre-loaded model (if model_path not provided)
            num_simulations: MCTS simulation count
            c_puct: Exploration constant
            temperature: Move selection temperature
            device: Device for inference
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Load model
        if model is None:
            if model_path is None:
                raise ValueError("Must provide either model_path or model")
            self.logger.info(f"Loading model from {model_path}")
            model = self._load_model(model_path, device)

        # Wrap model for MCTS interface
        self.neural_net = NeuralNetWrapper(model, device)

        # Initialize MCTS
        self.mcts = MCTS(
            neural_net=self.neural_net,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature
        )

        self.num_simulations = num_simulations
        self.logger.info(
            f"Player initialized: {num_simulations} sims, "
            f"c_puct={c_puct}, temp={temperature}"
        )

    def _load_model(self, model_path: str, device: str) -> nn.Module:
        """
        Load trained model from checkpoint.

        Args:
            model_path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded PyTorch model
        """
        checkpoint = torch.load(model_path, map_location=device)

        # Get config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Use default config
            config = TrainingConfig()

        # Create model
        model = ChessCNN(config)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        self.logger.info(
            f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}"
        )

        return model

    def select_move(
        self,
        board: chess.Board,
        return_probabilities: bool = False
    ) -> chess.Move:
        """
        Select best move using MCTS.

        Args:
            board: Current position
            return_probabilities: If True, return (move, move_probs)

        Returns:
            Selected move (or (move, move_probs) if return_probabilities=True)
        """
        self.logger.debug(f"Selecting move for position: {board.fen()}")

        # Run MCTS search
        move_probs, best_move = self.mcts.get_move_probabilities(
            board,
            add_noise=False
        )

        self.logger.debug(f"Selected move: {best_move}")

        if return_probabilities:
            return best_move, move_probs
        else:
            return best_move

    def get_position_evaluation(self, board: chess.Board) -> Dict:
        """
        Get detailed position evaluation.

        Args:
            board: Position to evaluate

        Returns:
            Dict with policy, value, and top moves
        """
        policy_probs, value = self.neural_net.predict(board)

        # Get top moves
        top_moves = sorted(policy_probs.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'value': value,
            'top_moves': [(move.uci(), prob) for move, prob in top_moves],
            'num_legal_moves': len(policy_probs)
        }

    def play_game(
        self,
        opponent_player,
        player_color: chess.Color = chess.WHITE,
        max_moves: int = 200
    ) -> Tuple[str, List[str]]:
        """
        Play complete game against opponent.

        Args:
            opponent_player: Opponent with select_move(board) method
            player_color: Color for this player
            max_moves: Maximum game length

        Returns:
            (result, moves) where result is "1-0", "0-1", or "1/2-1/2"
        """
        board = chess.Board()
        moves = []

        self.logger.info(f"Starting game (playing as {chess.COLOR_NAMES[player_color]})")

        while not board.is_game_over() and len(moves) < max_moves * 2:
            if board.turn == player_color:
                # This player's turn
                move = self.select_move(board)
            else:
                # Opponent's turn
                move = opponent_player.select_move(board)

            board.push(move)
            moves.append(move.uci())

        result = board.result() if board.is_game_over() else "1/2-1/2"

        self.logger.info(f"Game finished: {result} after {len(moves)} moves")

        return result, moves


class RandomPlayer:
    """Random move player for testing."""

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select random legal move."""
        return np.random.choice(list(board.legal_moves))


def main():
    """Example usage of Neural MCTS Player."""

    # Example: Load trained model and play
    model_path = "/home/krish/EE542-Project/FINAL/chess_ai/game_engine/model/best_model.pth"

    if Path(model_path).exists():
        # Create player with trained model
        player = NeuralMCTSPlayer(
            model_path=model_path,
            num_simulations=200,
            c_puct=1.0,
            temperature=0.0,
            device="cuda"
        )

        # Test position evaluation
        board = chess.Board()
        evaluation = player.get_position_evaluation(board)

        print("\nPosition Evaluation:")
        print(f"Value: {evaluation['value']:.3f}")
        print("\nTop moves:")
        for move, prob in evaluation['top_moves']:
            print(f"  {move}: {prob:.3f}")

        # Play game vs random
        print("\nPlaying vs Random opponent...")
        random_opponent = RandomPlayer()
        result, moves = player.play_game(random_opponent, chess.WHITE, max_moves=100)

        print(f"\nResult: {result}")
        print(f"Moves: {len(moves)}")
        print(f"First 10 moves: {' '.join(moves[:10])}")

    else:
        print(f"Model not found at {model_path}")
        print("Train a model first using cnn.py or cnn_colab.py")


if __name__ == '__main__':
    main()
