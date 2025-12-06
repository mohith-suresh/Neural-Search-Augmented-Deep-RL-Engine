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
        move, _ = self.search_with_stats(game, temperature)
        return move

    def search_with_stats(self, game, temperature=0.0):
        root = Node(game)
        root_policy_probs = {}
        
        for i in range(self.simulations):
            node = root
            search_path = [node]
            
            while node.is_expanded():
                action, node = node.select_child(self.cpu)
                search_path.append(node)
                
            if node.state.is_over:
                reward = node.state.get_reward_for_turn(node.state.turn_player)
                self.backpropagate(search_path, reward, node.state.turn_player)
                continue
            
            # Efficient Tensor conversion
            tensor = torch.from_numpy(node.state.to_tensor()).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                policy_logits, value_scalar = self.model(tensor)
            
            policy = policy_logits.cpu().numpy()[0]
            value = value_scalar.item()
            
            # Capture Root Policy on first pass
            if i == 0:
                valid_moves = node.state.legal_moves()
                temp_probs = {}
                policy_sum = 0
                for m in valid_moves:
                    src = (ord(m[0]) - 97) + (int(m[1]) - 1) * 8
                    dst = (ord(m[2]) - 97) + (int(m[3]) - 1) * 8
                    idx = src * 64 + dst
                    if len(m) == 5 and m[4] in ['n', 'r', 'b']: idx += 4096
                    
                    prob = np.exp(policy[idx]) if idx < len(policy) else 0
                    temp_probs[m] = prob
                    policy_sum += prob
                
                if policy_sum > 0:
                    sorted_p = sorted(temp_probs.items(), key=lambda x: x[1], reverse=True)
                    root_policy_probs = [(m, float(p/policy_sum)) for m, p in sorted_p[:3]]

            valid_moves = node.state.legal_moves()
            node.expand(valid_moves, policy)
            self.backpropagate(search_path, value, node.state.turn_player)
            
        visit_counts = {k: v.visit_count for k, v in root.children.items()}
        sorted_visits = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
        
        stats = {
            'visits': visit_counts,
            'model_top_3': sorted_visits[:3],
            'policy_top_3': root_policy_probs
        }
        
        if temperature == 0.0:
            return root.best_action(), stats
        else:
            actions = list(root.children.keys())
            visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
            
            if len(actions) == 0: return None, stats

            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            
            if np.isinf(np.sum(visits)): return root.best_action(), stats
            
            if np.sum(visits) == 0:
                probs = np.ones(len(visits)) / len(visits)
            else:
                probs = visits / np.sum(visits)
            
            chosen_action = np.random.choice(actions, p=probs)
            return chosen_action, stats

    def backpropagate(self, path, value, turn_perspective):
        for node in reversed(path):
            node.visit_count += 1
            if node.state.turn_player == turn_perspective:
                node.value_sum += value
            else:
                node.value_sum -= value

class Arena:
    def __init__(self, candidate_path, champion_path, simulations=200, max_moves=100):
        self.candidate_path = candidate_path
        self.champion_path = champion_path
        self.simulations = simulations
        self.max_moves = max_moves

    def play_match(self, num_games=10):
        print(f"--- ARENA: Playing {num_games} games (Candidate vs Champion) ---")
        player_candidate = EvalMCTS(self.candidate_path, self.simulations)
        
        if not os.path.exists(self.champion_path):
            print("No champion model found. Candidate wins by default.")
            return 1.0
            
        player_champion = EvalMCTS(self.champion_path, self.simulations)
        
        candidate_wins = 0
        draws = 0
        champion_wins = 0
        
        for i in range(num_games):
            if i % 2 == 0:
                white, black = player_candidate, player_champion
                p1_label, p2_label = "Cand", "Champ"
            else:
                white, black = player_champion, player_candidate
                p1_label, p2_label = "Champ", "Cand"
                
            game = ChessGame()
            moves = 0
            while not game.is_over and moves < self.max_moves:
                if game.board.turn == chess.WHITE:
                    move = white.search(game, temperature=0.0)
                else:
                    move = black.search(game, temperature=0.0)
                game.push(move)
                moves += 1
                
            result = game.result
            if result == "1-0":
                if i % 2 == 0: candidate_wins += 1
                else: champion_wins += 1
            elif result == "0-1":
                if i % 2 == 0: champion_wins += 1
                else: candidate_wins += 1
            else:
                draws += 1
            print(f"Game {i+1}/{num_games}: {result} ({p1_label} vs {p2_label})")

        total_score = candidate_wins + (draws * 0.5)
        win_rate = total_score / num_games
        print(f"--- ARENA RESULTS: Win Rate {win_rate:.2f} ---")
        return win_rate

class StockfishEvaluator:
    def __init__(self, stockfish_path, simulations=200):
        self.stockfish_path = stockfish_path
        self.simulations = simulations
        self.analysis_dir = "game_engine/analysis"
        os.makedirs(self.analysis_dir, exist_ok=True)

    def evaluate(self, model_path, num_games=10, stockfish_elo=1500):
        """
        Evaluates the agent against Stockfish and returns the adjusted win rate.
        """
        if not os.path.exists(self.stockfish_path):
            print(f"Skipping Stockfish eval: Binary not found at {self.stockfish_path}")
            # Return a special value to indicate skip/failure
            return {'win_rate': 0.0, 'adjusted_win_rate': 0.5} 

        print(f"--- STOCKFISH EVAL: {num_games} games vs Stockfish (Elo {stockfish_elo}) ---")
        agent = EvalMCTS(model_path, self.simulations)
        score = 0
        
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
                
                for i in range(num_games):
                    game = ChessGame()
                    agent_is_white = (i % 2 == 0)
                    
                    while not game.is_over:
                        is_agent_turn = (game.board.turn == chess.WHITE and agent_is_white) or \
                                        (game.board.turn == chess.BLACK and not agent_is_white)
                        
                        if is_agent_turn:
                            move = agent.search(game, temperature=0.0)
                        else:
                            result = engine.play(game.board, chess.engine.Limit(time=0.1))
                            move = result.move.uci()
                        
                        game.push(move)
                        
                    res = game.result
                    if res == "1-0":
                        score += 1 if agent_is_white else 0
                    elif res == "0-1":
                        score += 0 if agent_is_white else 1
                    else:
                        score += 0.5
                    
                    print(f"Game {i+1}: {res} (Agent {'White' if agent_is_white else 'Black'})")

        except Exception as e:
            print(f"Stockfish Engine Error: {e}")
            return {'win_rate': 0.0, 'adjusted_win_rate': 0.5}
            
        win_rate = score / num_games
        
        # --- Elo Adjustment (Bayesian smoothing) ---
        # The 1.0/2.0 additions are for Bayesian smoothing (adding 2 pseudo-games, 1 win, 1 loss)
        adjusted_score = score + 1.0
        adjusted_games = num_games + 2.0
        adjusted_win_rate = adjusted_score / adjusted_games
        
        print(f"Win Rate vs Stockfish {stockfish_elo}: {win_rate*100:.1f}%")
        
        # Return the win rates, Elo calculation moves to main.py
        return {
            'win_rate': win_rate, 
            'adjusted_win_rate': adjusted_win_rate
        }

class MetricsLogger:
    def __init__(self, metrics_file="game_engine/model/metrics.json"):
        self.metrics_file = metrics_file
        self.history = []
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                try: self.history = json.load(f)
                except: self.history = []

    def log(self, iteration, policy_loss, value_loss, arena_win_rate, elo=None):
        entry = {
            "iteration": iteration,
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "arena_win_rate": float(arena_win_rate),
            "elo": elo
        }
        self.history.append(entry)
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        try:
            self.plot_metrics()
        except Exception as e:
            print(f"Plotting failed (likely no GUI): {e}")

    def plot_metrics(self):
        if not self.history: return
        iterations = [h['iteration'] for h in self.history]
        p_loss = [h['policy_loss'] for h in self.history]
        v_loss = [h['value_loss'] for h in self.history]
        win_rates = [h['arena_win_rate'] for h in self.history]
        elos = [h['elo'] for h in self.history if h['elo'] is not None]
        elo_iters = [h['iteration'] for h in self.history if h['elo'] is not None]

        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(iterations, p_loss, label='Policy Loss', color='blue')
        plt.plot(iterations, v_loss, label='Value Loss', color='orange')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(iterations, win_rates, label='Win Rate vs Prev Best', color='green', marker='o')
        plt.axhline(y=0.55, color='r', linestyle='--', label='Threshold (0.55)')
        plt.title('Candidate Win Rate')
        plt.legend()
        plt.ylim(0, 1.0)
        plt.grid(True)
        
        if elos:
            plt.subplot(2, 2, 3)
            plt.plot(elo_iters, elos, label='Est. Elo', color='purple', marker='x')
            plt.title('Estimated Elo (vs Stockfish)')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig("game_engine/model/training_metrics.png")
        plt.close()
        print("Metrics plot updated at game_engine/model/training_metrics.png")