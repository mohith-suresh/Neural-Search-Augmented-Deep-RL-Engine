from game_engine.chess_env import ChessGame
game = ChessGame()
print(game.get_state())
game.push('e2e4')
print(game.get_state())
