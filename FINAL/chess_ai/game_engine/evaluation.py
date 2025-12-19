import torch
import os
import sys
import chess
import chess.engine
import json
import random
import time
import shutil
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.mcts import MCTSWorker
from game_engine.cnn import ChessCNN
from game_engine.chess_env import ChessGame
from game_engine.bayeselo_runner import BayesEloRunner

class EvalMCTS:
    """Evaluation wrapper using MCTSWorker.search_direct()"""
    
    def __init__(self, model_path, simulations=800, batch_size=8, device=None):
        self.simulations = simulations
        self.batch_size = batch_size
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FIX: Explicitly set upgraded=True to match your 256-filter model
        self.model = ChessCNN(upgraded=True).to(self.device)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[Eval] Loaded model from {model_path}")
            except Exception as e:
                print(f"[Eval] ❌ Failed to load checkpoint: {e}")
        else:
            print(f"[Eval] ⚠️ Model not found at {model_path}, using random weights")
        
        self.model.eval()
        
        # Create MCTSWorker for search (no queues needed for direct eval)
        self.worker = MCTSWorker(worker_id=0, input_queue=None, output_queue=None, simulations=simulations, batch_size=batch_size)

    def get_move(self, fen, temperature=0.0):
        """Get best move for a given FEN"""
        game = ChessGame(fen=fen)
        
        # Use direct search (bypass queues)
        best_move_uci, _ = self.worker.search_direct(
            game, 
            self.model, 
            temperature=temperature, 
            use_dirichlet=False
        )
        
        return best_move_uci

class Arena:
    """Manages matches between two models or Model vs Stockfish"""
    
    def __init__(self, champion_path, candidate_path, simulations=800, device=None):
        self.champ = EvalMCTS(champion_path, simulations, device=device)
        self.cand = EvalMCTS(candidate_path, simulations, device=device)
        
    def play_game(self, game_id, max_moves=400): # Default 400
        game = ChessGame()
        
        # Randomize colors (0=Champ White, 1=Cand White)
        cand_color = random.choice([chess.WHITE, chess.BLACK])
        
        while not game.is_over and len(game.moves) < max_moves:
            fen = game.board.fen()
            
            if game.board.turn == cand_color:
                # Candidate to move
                move = self.cand.get_move(fen, temperature=0.5 if len(game.moves) < 10 else 0.1)
            else:
                # Champion to move
                move = self.champ.get_move(fen, temperature=0.5 if len(game.moves) < 10 else 0.1)
                
            if not game.push(move):
                print(f"[Arena] ❌ Illegal move attempted: {move}")
                break
                
        # Determine result
        res = game.result
        if len(game.moves) >= max_moves:
            return "DRAW_FORCED"
        
        if res == "1-0":
            return "CAND_WIN" if cand_color == chess.WHITE else "CHAMP_WIN"
        elif res == "0-1":
            return "CAND_WIN" if cand_color == chess.BLACK else "CHAMP_WIN"
        else:
            return "DRAW"

class StockfishEvaluator:
    """Evaluates model against Stockfish"""
    
    def __init__(self, stockfish_path, simulations=800):
        self.stockfish_path = stockfish_path
        self.simulations = simulations
        
    def evaluate_with_bayeselo(self, model_path, pgn_output_path, num_games=20, stockfish_elo=1320, max_moves=400):
        print(f" [Stockfish/BayesElo] Playing {num_games} games vs Elo {stockfish_elo}...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(pgn_output_path), exist_ok=True)
        
        model = EvalMCTS(model_path, simulations=self.simulations)
        
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        engine.configure({"Skill Level": 20}) # Set skill level if needed, or use UCI_LimitStrength
        
        # Configure Elo if supported
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
        except:
            print(" [Warning] Stockfish version might not support UCI_Elo")

        results = {"wins": 0, "losses": 0, "draws": 0}
        
        with open(pgn_output_path, "w") as pgn_file:
            for i in range(num_games):
                game = ChessGame()
                model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
                
                game_header = chess.pgn.Game()
                game_header.headers["Event"] = "Model vs Stockfish"
                game_header.headers["Site"] = "Localhost"
                game_header.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                game_header.headers["Round"] = str(i+1)
                game_header.headers["White"] = "Model" if model_color == chess.WHITE else f"Stockfish {stockfish_elo}"
                game_header.headers["Black"] = f"Stockfish {stockfish_elo}" if model_color == chess.WHITE else "Model"
                
                node = game_header
                
                while not game.is_over and len(game.moves) < max_moves:
                    fen = game.board.fen()
                    
                    if game.board.turn == model_color:
                        move_uci = model.get_move(fen, temperature=0.05)
                    else:
                        result = engine.play(game.board, chess.engine.Limit(time=0.1))
                        move_uci = result.move.uci()
                    
                    game.push(move_uci)
                    node = node.add_variation(chess.Move.from_uci(move_uci))
                
                # Result handling
                res = game.result
                if len(game.moves) >= max_moves:
                    res = "1/2-1/2"
                
                game_header.headers["Result"] = res
                print(game_header, file=pgn_file, end="\n\n")
                
                # Scoring
                if res == "1/2-1/2":
                    results["draws"] += 1
                    outcome = "1/2-1/2"
                elif (res == "1-0" and model_color == chess.WHITE) or (res == "0-1" and model_color == chess.BLACK):
                    results["wins"] += 1
                    outcome = "1-0 (Model)"
                else:
                    results["losses"] += 1
                    outcome = "0-1 (SF)"
                    
                print(f"Game {i+1}/{num_games}: {outcome} | Moves: {len(game.moves)}")
        
        engine.quit()
        
        # Run BayesElo
        runner = BayesEloRunner(stockfish_elo=stockfish_elo)
        metrics = runner.run(pgn_output_path)
        return metrics

class MetricsLogger:
    @staticmethod
    def log(data):
        with open("training_log.json", "a") as f:
            f.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="game_engine/model/best_model.pth")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--elo", type=int, default=1320)
    parser.add_argument("--max_moves", type=int, default=400, help="Max moves before forced draw")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"🏆 FINAL EVALUATION: Model vs Stockfish (Limit {args.elo} Elo)")
    print(f"Games: {args.games}")
    print(f"MCTS Simulations: {args.sims}")
    print("=" * 70 + "\n")
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    evaluator = StockfishEvaluator(
        stockfish_path=args.stockfish,
        simulations=args.sims
    )
    
    pgn_path = f"game_engine/evaluation/pgn/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
    
    # FIX: Use args.max_moves (Default 400) instead of hardcoded 200
    results = evaluator.evaluate_with_bayeselo(
        model_path=args.model,
        pgn_output_path=pgn_path,
        num_games=args.games,
        stockfish_elo=args.elo,
        max_moves=args.max_moves
    )
    
    if results:
        print("\n" + "=" * 70)
        print("✅ TEST SUCCESSFUL")
        print("=" * 70)
        print(f"Your Model:     {results['model_elo']:.0f} Elo")
