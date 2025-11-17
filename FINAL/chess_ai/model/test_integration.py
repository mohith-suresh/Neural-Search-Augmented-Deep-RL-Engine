"""
Test integration between chess environment and encoder.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game_engine.chess_env import ChessGame
from model.chess_encoder import ChessEncoder
import numpy as np

def test_game_encoding():
    """Test encoding a complete game."""
    encoder = ChessEncoder()
    game = ChessGame()
    
    positions = []
    moves = []
    
    print("Playing a random game and encoding positions...")
    
    while not game.is_over and len(game.moves) < 50:
        # Encode current position
        tensor = encoder.board_to_tensor(game.board)
        positions.append(tensor)
        
        # Make random move
        move_uci = game.random_move()
        if move_uci is None:
            break
            
        # Encode the move
        move_obj = game.board.peek()  # Get last move
        move_idx = encoder.move_to_index(move_obj)
        moves.append(move_idx)
    
    print(f"Game ended after {len(positions)} moves")
    print(f"Result: {game.result}")
    
    # Convert result to values
    if game.result:
        for i, pos in enumerate(positions):
            color = (i % 2 == 0)  # White if even, black if odd
            value = encoder.result_to_value(game.result, color)
            print(f"Position {i}: value from {'white' if color else 'black'} perspective = {value}")
    
    # Stack into arrays
    positions_array = np.stack(positions)
    moves_array = np.array(moves)
    
    print(f"\nFinal arrays:")
    print(f"Positions shape: {positions_array.shape}")
    print(f"Moves shape: {moves_array.shape}")
    
    return positions_array, moves_array

if __name__ == "__main__":
    test_game_encoding()
    print("\nIntegration test passed!")
