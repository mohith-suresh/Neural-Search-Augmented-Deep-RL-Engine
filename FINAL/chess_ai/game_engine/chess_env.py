import chess
import uuid
import numpy as np
import copy

class ChessGame:
    def __init__(self, fen=None, _board=None):
        if _board:
            self.board = _board
        else:
            self.board = chess.Board(fen) if fen else chess.Board()
            
        self.game_id = str(uuid.uuid4())
        self.moves = []       
        self._cache_legal = None  

    @property
    def turn_player(self):
        """
        Returns 1.0 for White, 0.0 for Black.
        Used by MCTS to track whose turn it is in the tree.
        """
        return 1.0 if self.board.turn == chess.WHITE else 0.0

    @property
    def is_over(self):
        return self.board.is_game_over()

    @property
    def result(self):
        return self.board.result()

    def legal_moves(self):
        """Returns list of UCI strings. Cached for MCTS speed."""
        if self._cache_legal is None:
            self._cache_legal = [move.uci() for move in self.board.legal_moves]
        return self._cache_legal

    def push(self, move_uci):
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.moves.append(move_uci)
                self._cache_legal = None # Invalidate cache
                return True
            return False
        except ValueError:
            return False

    def copy(self):
        """Optimized copy for MCTS simulation."""
        new_board = self.board.copy()
        new_game = ChessGame(_board=new_board)
        new_game.moves = self.moves.copy()
        return new_game

    def __deepcopy__(self, memo):
        return self.copy()

    def to_tensor(self):
        """
        Input for cnn.py. 
        Shape: (13, 8, 8) - Float32
        
        MATCHES TRAINING DATA ENCODING:
        - Rank 1 is at index 0 (sq // 8)
        - Rank 8 is at index 7
        """
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                offset = 0 if piece.color == chess.WHITE else 6
                idx = offset + piece_map[piece.piece_type]
                
                # --- CRITICAL FIX: MATCH TRAINING DATA ORIENTATION ---
                # Old: row = 7 - (square // 8)  <-- Visual (Top-Down)
                # New: row = square // 8        <-- Mathematical (Bottom-Up, matches training)
                row = square // 8
                col = square % 8
                
                tensor[idx][row][col] = 1.0
                
        # Channel 12: 1.0 if Black to move
        if self.board.turn == chess.BLACK:
             tensor[12, :, :] = 1.0
             
        return tensor

    def get_reward_for_turn(self, turn_val):
        """
        Returns reward from the perspective of 'turn_val'.
        If turn_val is White (1.0) and White won, returns +1.
        """
        res = self.board.result()
        if res == "1-0": return 1.0 if turn_val == 1.0 else -1.0
        if res == "0-1": return 1.0 if turn_val == 0.0 else -1.0
        return 0.0 # Draw