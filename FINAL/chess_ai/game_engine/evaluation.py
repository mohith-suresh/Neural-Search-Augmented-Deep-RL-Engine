import torch
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import chess
import chess.engine
import time
from datetime import datetime

# Ensure imports work
sys.path.append(os.getcwd())

from game_engine.cnn import ChessCNN
from game_engine.chess_env import ChessGame
from game_engine.mcts import Node 

class EvalMCTS:
    def __init__(self, model_path, simulations=200, device=None):
        self.simulations = simulations
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
            print(f"WARNING: EvalMCTS could not find model at {model_path}. Using random weights.")
            
        self.model.eval()
        self.cpu = 1.0 

    def search(self, game, temperature=0.0):
        # Wrapper for simple search
        root = Node(game)
        for _ in range(self.simulations):
            node = root
            path = [node]
            while node.is_expanded():
                _, node = node.select_child(self.cpu)
                path.append(node)
            
            if node.state.is_over:
                reward = node.state.get_reward_for_turn(node.state.turn_player)
                self.backpropagate(path, reward, node.state.turn_player)
                continue
                
            tensor = torch.from_numpy(node.state.to_tensor()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p, v = self.model(tensor)
            
            node.expand(node.state.legal_moves(), p.cpu().numpy()[0])
            self.backpropagate(path, v.item(), node.state.turn_player)
            
        return root.best_action()

    def backpropagate(self, path, value, turn_perspective):
        for node in reversed(path):
            node.visit_count += 1
            if node.state.turn_player == turn_perspective:
                node.value_sum += value
            else:
                node.value_sum -= value

class Arena:
    def __init__(self, candidate_path, champion_path, simulations=200, max_moves=200):
        self.candidate_path = candidate_path
        self.champion_path = champion_path
        self.simulations = simulations
        self.max_moves = max_moves

    def play_match(self, num_games=10):
        # Instantiates models locally for thread safety in workers
        cand = EvalMCTS(self.candidate_path, self.simulations)
        champ = EvalMCTS(self.champion_path, self.simulations)
        
        wins, draws, losses = 0, 0, 0
        
        for i in range(num_games):
            game = ChessGame()
            # Alternate colors
            cand_is_white = (i % 2 == 0)
            
            p1_label = "Cand" if cand_is_white else "Champ"
            p2_label = "Champ" if cand_is_white else "Cand"

            while not game.is_over and len(game.moves) < self.max_moves:
                if game.board.turn == chess.WHITE:
                    move = cand.search(game, temperature=0.0) if cand_is_white else champ.search(game, temperature=0.0)
                else:
                    move = champ.search(game, temperature=0.0) if cand_is_white else cand.search(game, temperature=0.0)
                game.push(move)
            
            # Check for forced draw
            if not game.is_over and len(game.moves) >= self.max_moves:
                 print(f"   [Arena] Game {i+1} ended in FORCED DRAW (Max moves {self.max_moves})")
                 # Result will be '*' or similar, falling into the else (draw) block below effectively, 
                 # but we rely on game.result parsing usually.
                 # To ensure it counts as draw:
            
            result = game.result
            if result == "1-0":
                if cand_is_white: wins += 1
                else: losses += 1
            elif result == "0-1":
                if cand_is_white: losses += 1
                else: wins += 1
            else:
                draws += 1
            
            # Optional: Print result of each game
            print(f"Arena Game {i+1}: {result} ({p1_label} vs {p2_label})")
                
        return wins, draws, losses

class StockfishEvaluator:
    def __init__(self, stockfish_path, simulations=200):
        self.stockfish_path = stockfish_path
        self.simulations = simulations

    def evaluate(self, model_path, num_games=10, stockfish_elo=1350, max_moves=200):
        agent = EvalMCTS(model_path, self.simulations)
        score = 0.0
        
        if not os.path.exists(self.stockfish_path):
            return 0.0, 0

        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
                
                for i in range(num_games):
                    game = ChessGame()
                    agent_is_white = (i % 2 == 0)
                    
                    while not game.is_over and len(game.moves) < max_moves:
                        is_agent = (game.board.turn == chess.WHITE and agent_is_white) or \
                                   (game.board.turn == chess.BLACK and not agent_is_white)
                        
                        if is_agent:
                            move = agent.search(game)
                        else:
                            try:
                                res = engine.play(game.board, chess.engine.Limit(time=0.05))
                                move = res.move.uci()
                            except: break
                        game.push(move)
                    
                    if not game.is_over and len(game.moves) >= max_moves:
                        print(f"   [Stockfish] Game {i+1} ended in FORCED DRAW (Max moves {max_moves})")

                    res = game.result
                    if res == "1-0": score += 1.0 if agent_is_white else 0.0
                    elif res == "0-1": score += 0.0 if agent_is_white else 1.0
                    else: score += 0.5
                    
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
            except: pass

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