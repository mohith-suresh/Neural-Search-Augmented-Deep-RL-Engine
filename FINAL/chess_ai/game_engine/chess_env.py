import chess
import uuid

class ChessGame:
    def __init__(self, fen=None):
        """
        Initialize a new Chess game. Optionally from a FEN string.
        """
        self.game_id = str(uuid.uuid4())
        self.board = chess.Board(fen) if fen else chess.Board()
        self.moves = []  # List of UCI moves
        self.is_over = False
        self.result = None
        self.last_move = None

    def legal_moves(self):
        """
        Returns a list of legal moves (UCI notation).
        """
        return [move.uci() for move in self.board.legal_moves]

    def push(self, move_uci):
        """
        Make a move given in UCI notation. Updates state.
        Return True if move was legal and applied, else False.
        """
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.moves.append(move_uci)
            self.last_move = move_uci
            self.is_over = self.board.is_game_over()
            if self.is_over:
                self.result = self.board.result()
            return True
        return False

    def get_state(self):
        """
        Returns current board state info as a dict.
        """
        return {
            "game_id": self.game_id,
            "fen": self.board.fen(),
            "legal_moves": self.legal_moves(),
            "last_move": self.last_move,
            "move_count": len(self.moves),
            "is_over": self.is_over,
            "result": self.result
        }

    def reset(self, fen=None):
        """
        Resets game to starting position or from optional FEN.
        """
        self.__init__(fen)

    def random_move(self):
        """
        Makes a random legal move (for demo/testing use).
        Returns the move UCI string, or None if over.
        """
        legal_moves = self.legal_moves()
        if legal_moves:
            import random
            move_uci = random.choice(legal_moves)
            self.push(move_uci)
            return move_uci
        return None
