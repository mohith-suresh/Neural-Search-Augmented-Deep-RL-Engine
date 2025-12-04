import torch
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from FINAL.chess_ai.game_engine.cnn_old import ChessCNN
from game_engine.chess_env import ChessGame

MODEL_PATH = "game_engine/model/best_model.pth"

def test_model():
    print(f"--- MODEL SANITY CHECK ---")
    print(f"Loading: {MODEL_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessCNN().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle state dict format
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 1. Test Starting Position
    game = ChessGame()
    tensor = torch.tensor(game.to_tensor(), dtype=torch.float32).unsqueeze(0).to(device)
    
    print("\n[Input Shape Check]")
    print(f"Expected: (1, 13, 8, 8)")
    print(f"Actual:   {tensor.shape}")
    
    with torch.no_grad():
        policy, value = model(tensor)
        
    policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
    value = value.item()
    
    print(f"\n[Model Output]")
    print(f"Value Evaluation (Win Prob): {value:.4f} (Should be near 0.0 or small +)")
    
    # 2. Decode Top 5 Moves
    print("\n[Top 5 Suggested Moves]")
    moves = []
    legal_moves = list(game.board.legal_moves)
    legal_uci = [m.uci() for m in legal_moves]
    
    # Brute-force map indices back to moves to see what the model 'likes'
    # (Using the logic from mcts.py)
    move_probs = []
    for move in legal_uci:
        src = (ord(move[0]) - 97) + (int(move[1]) - 1) * 8
        dst = (ord(move[2]) - 97) + (int(move[3]) - 1) * 8
        idx = src * 64 + dst
        # Promotion logic
        if len(move) == 5 and move[4] in ['n', 'r', 'b']:
            idx += 4096
            
        prob = policy[idx]
        move_probs.append((move, prob))
        
    # Sort by probability
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    for m, p in move_probs[:5]:
        print(f"Move: {m} | Prob: {p:.4f}")

    print("\nINTERPRETATION:")
    top_move = move_probs[0][0]
    if top_move in ['e2e4', 'd2d4', 'g1f3', 'c2c4']:
        print("✅ SUCCESS: Model recognizes standard openings.")
    else:
        print("❌ FAILURE: Model suggests weird moves. Likely Tensor Encoding Mismatch.")

if __name__ == "__main__":
    test_model()