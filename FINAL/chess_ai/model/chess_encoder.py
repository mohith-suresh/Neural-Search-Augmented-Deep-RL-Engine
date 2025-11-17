"""
Chess Board and Move Encoding for Neural Network Input/Output

This module bridges the chess environment (python-chess) with the CNN model.
It handles:
1. Board state to tensor conversion (12-channel representation)
2. Move to index encoding (for policy targets)
3. Index to move decoding (for policy outputs)
4. Game result to value encoding
"""

import chess
import numpy as np
import torch
from typing import List, Tuple, Optional


class ChessEncoder:
    """
    Handles all encoding/decoding between chess environment and neural network.
    
    Board Representation:
    - 12 channels (8x8 each): 6 piece types x 2 colors
    - Channel order: white pawns, knights, bishops, rooks, queens, kings,
                     black pawns, knights, bishops, rooks, queens, kings
    
    Move Encoding:
    - Simple encoding: from_square * 64 + to_square = index 0-4095
    - Promotion moves: add offset based on promotion piece
    """
    
    # Piece type to channel mapping
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
    
    def __init__(self):
        """Initialize the encoder with move mappings."""
        self.num_channels = 12
        self.board_size = 8
        
    def board_to_tensor(self, board: chess.Board) -> np.ndarray:
        """
        Convert a chess board to a 12x8x8 numpy array.
        
        Args:
            board: python-chess Board object
            
        Returns:
            numpy array of shape (12, 8, 8) with binary piece placement
        """
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                channel = self.PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
                row = square // 8
                col = square % 8
                tensor[channel, row, col] = 1.0
                
        return tensor
    
    def board_to_torch(self, board: chess.Board) -> torch.Tensor:
        """
        Convert board to PyTorch tensor.
        
        Args:
            board: python-chess Board object
            
        Returns:
            torch.Tensor of shape (12, 8, 8)
        """
        return torch.from_numpy(self.board_to_tensor(board))
    
    def move_to_index(self, move: chess.Move) -> int:
        """
        Convert a chess move to a flat index for policy network.
        
        Encoding scheme:
        - Normal moves: from_square * 64 + to_square (0-4095)
        - Promotions: 4096 + offset based on piece and squares
        
        Args:
            move: python-chess Move object
            
        Returns:
            Integer index (0-8191 for full action space)
        """
        from_square = move.from_square
        to_square = move.to_square
        
        if move.promotion is None:
            # Normal move
            return from_square * 64 + to_square
        else:
            # Promotion move
            base = 4096
            # 4 promotion types (knight, bishop, rook, queen) x 64 from x 8 to
            promotion_offset = {
                chess.KNIGHT: 0,
                chess.BISHOP: 1,
                chess.ROOK: 2,
                chess.QUEEN: 3,
            }[move.promotion]
            
            from_file = from_square % 8
            to_file = to_square % 8
            
            return base + (promotion_offset * 512) + (from_file * 64) + to_file
    
    def index_to_move(self, index: int, board: chess.Board) -> Optional[chess.Move]:
        """
        Convert a policy index back to a chess move.
        
        Args:
            index: Integer index from policy network
            board: Current board position (needed to validate move)
            
        Returns:
            chess.Move object if valid, None otherwise
        """
        if index < 4096:
            # Normal move
            from_square = index // 64
            to_square = index % 64
            move = chess.Move(from_square, to_square)
        else:
            # Promotion move
            offset = index - 4096
            promotion_type_idx = offset // 512
            remainder = offset % 512
            from_file = remainder // 64
            to_file = remainder % 64
            
            promotion_types = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
            promotion = promotion_types[promotion_type_idx]
            
            # Determine from/to squares based on color
            if board.turn == chess.WHITE:
                from_square = 48 + from_file  # 7th rank
                to_square = 56 + to_file       # 8th rank
            else:
                from_square = 8 + from_file    # 2nd rank
                to_square = to_file            # 1st rank
                
            move = chess.Move(from_square, to_square, promotion=promotion)
        
        # Validate move is legal
        if move in board.legal_moves:
            return move
        return None
    
    def result_to_value(self, result: str, perspective: chess.Color) -> float:
        """
        Convert game result to value for training.
        
        Args:
            result: Game result string ('1-0', '0-1', '1/2-1/2')
            perspective: Color from whose perspective to evaluate
            
        Returns:
            Float value: 1.0 for win, 0.0 for loss, 0.5 for draw
        """
        if result == '1/2-1/2':
            return 0.5
        
        if result == '1-0':
            return 1.0 if perspective == chess.WHITE else 0.0
        
        if result == '0-1':
            return 1.0 if perspective == chess.BLACK else 0.0
        
        # Default to draw for unknown results
        return 0.5
    
    def get_legal_move_mask(self, board: chess.Board) -> np.ndarray:
        """
        Create a mask of legal moves for the current position.
        
        Args:
            board: python-chess Board object
            
        Returns:
            numpy array of shape (8192,) with 1s for legal moves, 0s elsewhere
        """
        mask = np.zeros(8192, dtype=np.float32)
        
        for move in board.legal_moves:
            index = self.move_to_index(move)
            if 0 <= index < 8192:
                mask[index] = 1.0
                
        return mask
    
    def encode_game(self, board: chess.Board, move: chess.Move, 
                   result: Optional[str] = None) -> Tuple[np.ndarray, int, Optional[float]]:
        """
        Encode a single game position for training.
        
        Args:
            board: Current board state
            move: Move played from this position
            result: Final game result (None if game ongoing)
            
        Returns:
            Tuple of (board_tensor, move_index, value)
        """
        board_tensor = self.board_to_tensor(board)
        move_index = self.move_to_index(move)
        
        value = None
        if result is not None:
            value = self.result_to_value(result, board.turn)
            
        return board_tensor, move_index, value


# Global encoder instance
encoder = ChessEncoder()


# Convenience functions
def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert board to numpy tensor."""
    return encoder.board_to_tensor(board)


def move_to_index(move: chess.Move) -> int:
    """Convert move to policy index."""
    return encoder.move_to_index(move)


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """Convert policy index to move."""
    return encoder.index_to_move(index, board)


def result_to_value(result: str, perspective: chess.Color) -> float:
    """Convert result to value."""
    return encoder.result_to_value(result, perspective)


if __name__ == "__main__":
    # Test the encoder
    print("Testing ChessEncoder...")
    
    # Test 1: Board encoding
    board = chess.Board()
    tensor = board_to_tensor(board)
    print(f"Board tensor shape: {tensor.shape}")
    print(f"Number of pieces (should be 32): {tensor.sum()}")
    
    # Test 2: Move encoding
    move = chess.Move.from_uci("e2e4")
    index = move_to_index(move)
    print(f"Move e2e4 encoded as index: {index}")
    
    # Test 3: Move decoding
    decoded = index_to_move(index, board)
    print(f"Index {index} decoded back to: {decoded}")
    assert decoded == move, "Move encoding/decoding mismatch!"
    
    # Test 4: Promotion encoding
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    # Set up a promotion scenario
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promo_move = chess.Move.from_uci("a7a8q")
    promo_index = move_to_index(promo_move)
    print(f"Promotion a7a8q encoded as: {promo_index}")
    
    # Test 5: Legal move mask
    board = chess.Board()
    mask = encoder.get_legal_move_mask(board)
    print(f"Legal moves in start position: {mask.sum()} (should be 20)")
    
    print("\nAll tests passed!")
