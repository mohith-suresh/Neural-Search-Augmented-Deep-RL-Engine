import torch
import os
import sys
import chess
import chess.engine
import json
import random
from datetime import datetime


sys.path.append(os.getcwd())
from game_engine.mcts import MCTSWorker
from game_engine.cnn import ChessCNN
from game_engine.chess_env import ChessGame


class EvalMCTS:
    """Evaluation wrapper using MCTSWorker.search_direct()"""
    
    def __init__(self, model_path, simulations=1200, batch_size=8, device=None):
        self.simulations = simulations
        self.batch_size = batch_size
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ChessCNN().to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"WARNING: Model not found at {model_path}")
        
        self.model.eval()
        
        # Create MCTSWorker for search (no queues needed for eval)
        self.mcts = MCTSWorker(
            worker_id=0,
            input_queue=None,
            output_queue=None,
            simulations=simulations,
            batch_size=batch_size
        )
    
    def search(self, game, temperature=0.0, use_dirichlet=False):
        """
        Search using MCTSWorker.search_direct()
        
        Args:
            game: ChessGame instance
            temperature: 0.0 for greedy, >0 for sampling
            use_dirichlet: Whether to add exploration noise
        
        Returns:
            Best action (UCI string)
        """
        action, _ = self.mcts.search_direct(
            game,
            model=self.model,
            temperature=temperature,
            use_dirichlet=use_dirichlet
        )
        return action

class Arena:
    """Arena for comparing two models."""
    
    def __init__(self, candidate_path, champion_path, simulations=1200, max_moves=200):
        self.candidate = EvalMCTS(candidate_path, simulations=simulations)
        self.champion = EvalMCTS(champion_path, simulations=simulations)
        self.max_moves = max_moves
    
    def play_match(self, num_games=10, temperature=0.0, use_dirichlet=False):
        """Play match between candidate and champion."""
        wins, draws, losses, forced_draws = 0, 0, 0, 0
        
        for i in range(num_games):
            game = ChessGame()
            
            # --- OPENING VARIETY (CORRECTED) ---
            # Randomly pick ANY valid first move to ensure true variety
            legal_moves = list(game.board.legal_moves)
            random_opening_move = random.choice(legal_moves)
            # Convert the Move object to a UCI string before pushing
            game.push(random_opening_move.uci())
            
            cand_is_white = (i % 2 == 0)
            p1_label = "Cand" if cand_is_white else "Champ"
            p2_label = "Champ" if cand_is_white else "Cand"
            
            while not game.is_over and len(game.moves) < self.max_moves:
                if game.board.turn == chess.WHITE:
                    move = self.candidate.search(game, temperature=temperature, use_dirichlet=use_dirichlet) if cand_is_white else self.champion.search(game, temperature=temperature, use_dirichlet=use_dirichlet)
                else:
                    move = self.champion.search(game, temperature=temperature, use_dirichlet=use_dirichlet) if cand_is_white else self.candidate.search(game, temperature=temperature, use_dirichlet=use_dirichlet)
                
                if move is None:
                    break
                game.push(move)
            
            # Check for forced draw
            is_forced_draw = not game.is_over and len(game.moves) >= self.max_moves
            if is_forced_draw:
                print(f" [Arena] Game {i+1} ended in FORCED DRAW (Max moves {self.max_moves})")

            result = game.result
            if result == "1-0":
                if cand_is_white:
                    wins += 1
                else:
                    losses += 1
            elif result == "0-1":
                if cand_is_white:
                    losses += 1
                else:
                    wins += 1
            else: # Game is a draw
                if is_forced_draw:
                    forced_draws += 1
                else:
                    draws += 1
            
            print(f"Arena Game {i+1}: {result} ({p1_label} vs {p2_label}) | Total Moves: {len(game.moves)}")
        
        return wins, draws, losses, forced_draws

class StockfishEvaluator:
    """Evaluate model against Stockfish."""
    
    def __init__(self, stockfish_path, simulations=1200):
        self.stockfish_path = stockfish_path
        self.simulations = simulations
    
    def evaluate(self, model_path, num_games=10, stockfish_elo=1350, max_moves=200, use_dirichlet=False):
        """
        Evaluate model against Stockfish.
        
        Args:
            model_path: Path to model checkpoint
            num_games: Number of games to play
            stockfish_elo: Stockfish Elo rating
            max_moves: Maximum moves per game
            use_dirichlet: If True, add exploration noise
        
        Returns:
            (score, num_games)
        """
        agent = EvalMCTS(model_path, simulations=self.simulations)
        score = 0.0
        
        if not os.path.exists(self.stockfish_path):
            print(f"WARNING: Stockfish not found at {self.stockfish_path}")
            return 0.0, 0
        
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                engine.configure({"Skill Level": 0})
                
                for i in range(num_games):
                    game = ChessGame()
                    agent_is_white = (i % 2 == 0)
                    p1_label = "Agent" if agent_is_white else "Stockfish"
                    p2_label = "Stockfish" if agent_is_white else "Agent"
                    
                    while not game.is_over and len(game.moves) < max_moves:
                        is_agent = (game.board.turn == chess.WHITE and agent_is_white) or \
                                  (game.board.turn == chess.BLACK and not agent_is_white)
                        
                        if is_agent:
                            move = agent.search(game, temperature=0.0, use_dirichlet=use_dirichlet)
                        else:
                            try:
                                res = engine.play(game.board, chess.engine.Limit(time=0.05))
                                move = res.move.uci()
                            except:
                                break
                        
                        if move is None:
                            break
                        game.push(move)
                    
                    if not game.is_over and len(game.moves) >= max_moves:
                        print(f" [Stockfish] Game {i+1} ended in FORCED DRAW (Max moves {max_moves})")
                    
                    res = game.result
                    if res == "1-0":
                        score += 1.0 if agent_is_white else 0.0
                    elif res == "0-1":
                        score += 0.0 if agent_is_white else 1.0
                    else:
                        score += 0.5
                    
                    print(f"Stockfish Game {i+1}: {res} ({p1_label} vs {p2_label}) | Total Moves: {len(game.moves)}")
        
        except Exception as e:
            print(f"Stockfish Error: {e}")
            return 0.0, 0
        
        return score, num_games
    
class MetricsLogger:
    def __init__(self, metrics_file="game_engine/model/metrics.json"):
        self.metrics_file = metrics_file
        self.history = []
        
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.history = json.load(f)
            except:
                pass
    
    def log(self, iteration, policy_loss, value_loss, arena_win_rate, elo=None):
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "arena_win_rate": float(arena_win_rate),
            "elo": elo
        }
        self.history.append(entry)
        
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=4)
