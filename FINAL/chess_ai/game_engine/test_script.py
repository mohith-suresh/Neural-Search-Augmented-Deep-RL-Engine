import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.evaluation import StockfishEvaluator

# Config
MODEL_PATH = "game_engine/model/best_model.pth"
STOCKFISH_PATH = "/usr/games/stockfish" 

print("--- TESTING STOCKFISH COMPATIBILITY ---")

# 1. Initialize Evaluator
try:
    evaluator = StockfishEvaluator(STOCKFISH_PATH, simulations=10) # Low sims for speed
    
    # 2. Run with the new minimum Elo (1350)
    print("Attempting to run 2 games at Elo 1350...")
    elo = evaluator.evaluate(MODEL_PATH, num_games=10, stockfish_elo=1350)
    
    if elo is not None:
        print(f"\n✅ SUCCESS! Stockfish accepted the configuration. Estimated Elo: {elo}")
        print("You are ready for the overnight run.")
    else:
        print("\n❌ FAILURE. Stockfish crashed or returned None.")

except Exception as e:
    print(f"\n❌ CRASHED WITH ERROR: {e}")