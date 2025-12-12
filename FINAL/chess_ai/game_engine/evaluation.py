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
from game_engine.bayeselo_runner import BayesEloRunner

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
    
    def evaluate(self, model_path, num_games=10, stockfish_elo=1350, max_moves=200, use_dirichlet=False, pgn_output_path=None):
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
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
                
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
    
    def evaluate_with_bayeselo(self, model_path, pgn_output_path, num_games=40, stockfish_elo=1350, max_moves=200):
        """Play games, save PGN, and run BayesElo for accurate rating."""
        print(f"\n{'='*70}\n📊 STOCKFISH EVALUATION - {num_games} GAMES\n{'='*70}\n")
        
        agent = EvalMCTS(model_path, simulations=self.simulations)
        pgn_games = []
        
        if not os.path.exists(self.stockfish_path):
            print(f"❌ Stockfish not found at {self.stockfish_path}")
            return None
        
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
                            move = agent.search(game, temperature=0.0, use_dirichlet=False)
                        else:
                            try:
                                res = engine.play(game.board, chess.engine.Limit(time=0.05))
                                move = res.move.uci()
                            except:
                                break
                        
                        if move is None:
                            break
                        game.push(move)
                    
                    result = game.result
                    pgn_games.append({
                        'moves': ' '.join(game.moves),
                        'result': result,
                        'white': 'Model' if agent_is_white else 'Stockfish',
                        'black': 'Stockfish' if agent_is_white else 'Model'
                    })

                    white_name = 'Model' if agent_is_white else 'Stockfish'
                    black_name = 'Stockfish' if agent_is_white else 'Model'
                    print(f"Game {i+1}/{num_games}: {result} ({white_name} vs {black_name}) | Total Moves: {len(game.moves)}")
        
        except Exception as e:
            print(f"❌ Evaluation Error: {e}")
            return None
        
        # Save PGN
        os.makedirs(os.path.dirname(pgn_output_path), exist_ok=True)
        with open(pgn_output_path, 'w') as f:
            for i, game_data in enumerate(pgn_games, 1):
                f.write(f'[Event "Model vs Stockfish"]\n')
                f.write(f'[Site "Local"]\n')
                f.write(f'[Date "{datetime.now().strftime("%Y.%m.%d")}"\n')
                f.write(f'[Round "{i}"]\n')
                f.write(f'[White "{game_data["white"]}"]\n')
                f.write(f'[Black "{game_data["black"]}"]\n')
                f.write(f'[Result "{game_data["result"]}"]\n\n')
                f.write(f'{game_data["moves"]} {game_data["result"]}\n\n')
        
        print(f"\n✅ Saved {len(pgn_games)} games to {pgn_output_path}")
        
        # Run BayesElo
        try:
            runner = BayesEloRunner(stockfish_elo=stockfish_elo)
            results = runner.run(pgn_output_path)
            
            if results:
                print(f"\n{'='*70}\n🏆 BAYESELO RESULTS\n{'='*70}")
                print(f"Model Strength:     {results['model_elo']:.0f} ± {(results['model_ci_upper']-results['model_ci_lower'])/2:.0f} Elo")
                print(f"Vs Stockfish:       {results['sf_elo']:.0f} Elo")
                print(f"Difference:         +{results['diff_elo']:.0f} Elo [{results['diff_ci_lower']:.0f}, {results['diff_ci_upper']:.0f}]")
                print(f"{'='*70}\n")
                
                wins = sum(1 for g in pgn_games if (g['white'] == 'Model' and g['result'] == '1-0') or (g['black'] == 'Model' and g['result'] == '0-1'))
                losses = sum(1 for g in pgn_games if (g['white'] == 'Model' and g['result'] == '0-1') or (g['black'] == 'Model' and g['result'] == '1-0'))
                draws = len(pgn_games) - wins - losses
                
                results['win_count'] = wins
                results['loss_count'] = losses
                results['draw_count'] = draws
                results['total_games'] = len(pgn_games)
                results['win_rate'] = wins / len(pgn_games) if pgn_games else 0
                
                print("\n" + "=" * 70)
                print("✅ TEST SUCCESSFUL")
                print("=" * 70)
                print(f"Your Model:     {results['model_elo']:.0f} Elo")
                print(f"Stockfish:      {results['sf_elo']:.0f} Elo")
                print(f"Difference:     +{results['diff_elo']:.0f} Elo")
                print(f"Win Rate:       {results['win_rate']:.1%}")
                print(f"Record:         {results['win_count']}-{results['loss_count']}-{results['draw_count']}")
                print(f"PGN Saved:      {pgn_path}")
                print("=" * 70)
                
                return results
            else:
                print(f"❌ BayesElo failed")
                return None
        
        except Exception as e:
            print(f"❌ BayesElo Error: {e}")
            import traceback
            traceback.print_exc()
            return None

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
    
    def log(self, iteration, policy_loss, value_loss, arena_win_rate, elo=None, stockfish_elo=None):
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "arena_win_rate": float(arena_win_rate),
            "elo": elo,
            "stockfish_elo": stockfish_elo
        }
        self.history.append(entry)
        
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=4)

if __name__ == "__main__":
    """
    Quick test on your laptop with minimal games
    
    Usage:
        python game_engine/evaluation.py
    """
    
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BayesElo evaluation on laptop")
    parser.add_argument("--model", default="game_engine/model/best_model.pth", help="Model path")
    parser.add_argument("--stockfish", default="/usr/games/stockfish", help="Stockfish path")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--elo", type=int, default=1350, help="Stockfish Elo")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations (REDUCED for laptop)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧪 EVALUATION TESTER - LAPTOP MODE (Minimal Games)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Stockfish: {args.stockfish} ({args.elo} Elo)")
    print(f"Games: {args.games} (REDUCED for quick test)")
    print(f"MCTS Simulations: {args.sims} (REDUCED from 1200)")
    print("=" * 70 + "\n")
    
    # Check files exist
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.stockfish):
        print(f"❌ Stockfish not found: {args.stockfish}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = StockfishEvaluator(
        stockfish_path=args.stockfish,
        simulations=args.sims
    )
    
    # Run evaluation
    pgn_path = f"game_engine/evaluation/pgn/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
    
    results = evaluator.evaluate_with_bayeselo(
        model_path=args.model,
        pgn_output_path=pgn_path,
        num_games=args.games,
        stockfish_elo=args.elo,
        max_moves=200
    )
    
    if results:
        print("\n" + "=" * 70)
        print("✅ TEST SUCCESSFUL")
        print("=" * 70)
        print(f"Your Model:     {results['model_elo']:.0f} Elo")
        print(f"Stockfish:      {results['sf_elo']:.0f} Elo")
        print(f"Difference:     +{results['diff_elo']:.0f} Elo")
        print(f"Win Rate:       {results['win_rate']:.1%}")
        print(f"Record:         {results['win_count']}-{results['loss_count']}-{results['draw_count']}")
        print(f"PGN Saved:      {pgn_path}")
        print("=" * 70)
    else:
        print("\n❌ TEST FAILED")
        sys.exit(1)
