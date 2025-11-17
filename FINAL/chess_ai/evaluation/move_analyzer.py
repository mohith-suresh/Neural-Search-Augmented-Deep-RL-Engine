"""
Chess Move Quality Analyzer

Analyzes move quality using Stockfish:
- Classifies moves as excellent, good, inaccuracy, mistake, or blunder
- Calculates centipawn loss for each move
- Provides post-game analysis similar to Chess.com/Lichess
- Generates quality statistics and visualizations
"""

import chess
import chess.engine
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime


class MoveQualityAnalyzer:
    """
    Analyze chess move quality using Stockfish engine.
    """
    
    THRESHOLDS = {
        'brilliant': -10,
        'excellent': 0,
        'good': 25,
        'inaccuracy': 75,
        'mistake': 150,
        'blunder': float('inf')
    }
    
    MATE_SCORE = 10000
    
    def __init__(self, stockfish_path: str = "stockfish", 
                 analysis_time: float = 0.1,
                 analysis_depth: int = 15):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.available = True
            self.analysis_time = analysis_time
            self.analysis_depth = analysis_depth
            print(f"Move Analyzer initialized with Stockfish")
        except Exception as e:
            print(f"Warning: Could not initialize Stockfish: {e}")
            self.available = False
            self.engine = None
    
    def evaluate_position(self, board: chess.Board) -> int:
        if not self.available:
            return 0
        
        info = self.engine.analyse(
            board, 
            chess.engine.Limit(time=self.analysis_time, depth=self.analysis_depth)
        )
        
        score = info["score"].white()
        
        if score.is_mate():
            mate_in = score.mate()
            if mate_in > 0:
                return self.MATE_SCORE - mate_in
            else:
                return -self.MATE_SCORE - mate_in
        else:
            return score.score()
    
    def get_best_move(self, board: chess.Board) -> Tuple[chess.Move, int]:
        if not self.available:
            return list(board.legal_moves)[0], 0
        
        info = self.engine.analyse(
            board,
            chess.engine.Limit(time=self.analysis_time, depth=self.analysis_depth)
        )
        
        # Fixed: don't use board.peek() as fallback
        best_move = info["pv"][0] if "pv" in info else list(board.legal_moves)[0]
        score = info["score"].white()
        
        if score.is_mate():
            mate_in = score.mate()
            eval_cp = self.MATE_SCORE - abs(mate_in) if mate_in > 0 else -self.MATE_SCORE + abs(mate_in)
        else:
            eval_cp = score.score()
        
        return best_move, eval_cp
    
    def classify_move(self, centipawn_loss: int) -> str:
        if centipawn_loss < self.THRESHOLDS['brilliant']:
            return 'brilliant'
        elif centipawn_loss < self.THRESHOLDS['excellent']:
            return 'excellent'
        elif centipawn_loss < self.THRESHOLDS['good']:
            return 'good'
        elif centipawn_loss < self.THRESHOLDS['inaccuracy']:
            return 'inaccuracy'
        elif centipawn_loss < self.THRESHOLDS['mistake']:
            return 'mistake'
        else:
            return 'blunder'
    
    def analyze_move(self, board: chess.Board, move: chess.Move) -> Dict:
        if not self.available:
            return {
                'error': 'Stockfish not available',
                'move': move.uci(),
                'classification': 'unknown'
            }
        
        best_move, eval_before = self.get_best_move(board)
        board.push(move)
        eval_after = self.evaluate_position(board)
        board.pop()
        
        if board.turn == chess.WHITE:
            loss = eval_before - eval_after
        else:
            loss = eval_after - eval_before
        
        classification = self.classify_move(loss)
        is_best = (move == best_move)
        
        return {
            'move': move.uci(),
            'eval_before': eval_before,
            'eval_after': eval_after,
            'centipawn_loss': loss,
            'classification': classification,
            'is_best_move': is_best,
            'best_move': best_move.uci() if not is_best else None,
            'color': 'white' if board.turn == chess.WHITE else 'black'
        }
    
    def analyze_game(self, moves: List[str], verbose: bool = True) -> Dict:
        if not self.available:
            return {'error': 'Stockfish not available'}
        
        board = chess.Board()
        analysis = []
        
        if verbose:
            print(f"Analyzing game ({len(moves)} moves)...")
        
        for i, move_uci in enumerate(moves):
            move = chess.Move.from_uci(move_uci)
            move_analysis = self.analyze_move(board, move)
            move_analysis['move_number'] = i + 1
            move_analysis['ply'] = i
            
            analysis.append(move_analysis)
            board.push(move)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Analyzed {i + 1}/{len(moves)} moves...")
        
        stats = self._calculate_game_statistics(analysis)
        
        return {
            'moves': analysis,
            'statistics': stats,
            'num_moves': len(moves),
            'result': board.result(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_game_statistics(self, analysis: List[Dict]) -> Dict:
        stats = {
            'white': {'brilliant': 0, 'excellent': 0, 'good': 0, 'inaccuracy': 0, 'mistake': 0, 'blunder': 0},
            'black': {'brilliant': 0, 'excellent': 0, 'good': 0, 'inaccuracy': 0, 'mistake': 0, 'blunder': 0},
            'overall': {'brilliant': 0, 'excellent': 0, 'good': 0, 'inaccuracy': 0, 'mistake': 0, 'blunder': 0}
        }
        
        white_losses = []
        black_losses = []
        
        for move_data in analysis:
            color = move_data['color']
            classification = move_data['classification']
            loss = move_data['centipawn_loss']
            
            stats[color][classification] += 1
            stats['overall'][classification] += 1
            
            if color == 'white':
                white_losses.append(loss)
            else:
                black_losses.append(loss)
        
        stats['white']['avg_centipawn_loss'] = np.mean(white_losses) if white_losses else 0
        stats['black']['avg_centipawn_loss'] = np.mean(black_losses) if black_losses else 0
        stats['overall']['avg_centipawn_loss'] = np.mean(white_losses + black_losses)
        
        def calc_accuracy(color_stats):
            total = sum(v for k, v in color_stats.items() if k != 'avg_centipawn_loss')
            good_moves = color_stats['brilliant'] + color_stats['excellent'] + color_stats['good'] + color_stats['inaccuracy']
            return (good_moves / total * 100) if total > 0 else 0
        
        stats['white']['accuracy'] = calc_accuracy(stats['white'])
        stats['black']['accuracy'] = calc_accuracy(stats['black'])
        
        return stats
    
    def print_summary(self, analysis: Dict):
        stats = analysis['statistics']
        
        print("\n" + "="*70)
        print("GAME ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nTotal Moves: {analysis['num_moves']}")
        print(f"Result: {analysis['result']}")
        
        for color in ['white', 'black']:
            print(f"\n{color.upper()} Statistics:")
            print(f"  Accuracy: {stats[color]['accuracy']:.1f}%")
            print(f"  Average CP Loss: {stats[color]['avg_centipawn_loss']:.1f}")
            print(f"  Brilliant: {stats[color]['brilliant']}")
            print(f"  Excellent: {stats[color]['excellent']}")
            print(f"  Good: {stats[color]['good']}")
            print(f"  Inaccuracies: {stats[color]['inaccuracy']}")
            print(f"  Mistakes: {stats[color]['mistake']}")
            print(f"  Blunders: {stats[color]['blunder']}")
        
        print("\n" + "="*70)
    
    def close(self):
        if self.engine:
            self.engine.quit()


def analyze_game_quick(moves: List[str], save_to: str = None) -> Dict:
    analyzer = MoveQualityAnalyzer(analysis_time=0.1, analysis_depth=15)
    
    try:
        analysis = analyzer.analyze_game(moves)
        analyzer.print_summary(analysis)
        return analysis
    finally:
        analyzer.close()


if __name__ == "__main__":
    print("Testing Move Quality Analyzer...")
    print("="*70)
    
    test_moves = [
        'e2e4', 'e7e5',
        'f1c4', 'b8c6',
        'd1h5', 'g8f6',
        'h5f7'
    ]
    
    print("\nAnalyzing sample game (Scholar's Mate)...")
    analysis = analyze_game_quick(test_moves)
    
    print("\n" + "="*70)
    print("Move Analyzer Test Complete!")
