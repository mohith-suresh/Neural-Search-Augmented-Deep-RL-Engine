import torch
import os
import sys
import chess
import chess.engine

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
    
    def play_match(self, num_games=10):
        """Play match between candidate and champion."""
        wins, draws, losses = 0, 0, 0
        
        for i in range(num_games):
            game = ChessGame()
            cand_is_white = (i % 2 == 0)
            p1_label = "Cand" if cand_is_white else "Champ"
            p2_label = "Champ" if cand_is_white else "Cand"
            
            while not game.is_over and len(game.moves) < self.max_moves:
                if game.board.turn == chess.WHITE:
                    move = self.candidate.search(game, temperature=0.0) if cand_is_white else self.champion.search(game, temperature=0.0)
                else:
                    move = self.champion.search(game, temperature=0.0) if cand_is_white else self.candidate.search(game, temperature=0.0)
                
                if move is None:
                    break
                game.push(move)
            
            if not game.is_over and len(game.moves) >= self.max_moves:
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
            else:
                draws += 1
            
            print(f"Arena Game {i+1}: {result} ({p1_label} vs {p2_label}) | Total Moves: {len(game.moves)}")
        
        return wins, draws, losses

class StockfishEvaluator:
    """Evaluate model against Stockfish."""
    
    def __init__(self, stockfish_path, simulations=1200):
        self.stockfish_path = stockfish_path
        self.simulations = simulations
    
    def evaluate(self, model_path, num_games=10, stockfish_elo=1350, 
                 max_moves=200, use_dirichlet=False):
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

if __name__ == "__main__":
    """Local testing without Stockfish"""
    print("=" * 80)
    print("LOCAL TEST: MCTSWorker.search_direct()")
    print("=" * 80)
    
    # Test with dummy model path (will warn but won't crash)
    dummy_path = "game_engine/model/best_model.pth"
    
    # Test 1: Single game
    print("\n[Test 1] Single game (5 moves)")
    try:
        eval_mcts = EvalMCTS(dummy_path, simulations=50, batch_size=8)
        game = ChessGame()
        
        for move_num in range(5):
            move = eval_mcts.search(game, temperature=0.0, use_dirichlet=False)
            if move:
                print(f"  Move {move_num+1}: {move}")
                game.push(move)
            else:
                print(f"  Move {move_num+1}: No legal move found")
                break
        print(f"  Game state: {len(game.moves)} moves played")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 2: With Dirichlet noise
    print("\n[Test 2] Same position with Dirichlet noise")
    try:
        eval_mcts = EvalMCTS(dummy_path, simulations=1200, batch_size=8)
        game = ChessGame()
        
        for move_num in range(5):
            move = eval_mcts.search(game, temperature=0.0, use_dirichlet=True)
            if move:
                print(f"  Move {move_num+1}: {move}")
                game.push(move)
            else:
                print(f"  Move {move_num+1}: No legal move found")
                break
        print(f"  Game state: {len(game.moves)} moves played")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Tests completed!")
    print("=" * 80)
