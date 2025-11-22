#!/usr/bin/env python3
"""
Pure MCTS Evaluation Script
============================

Evaluates pure MCTS (no neural network) playing strength.

Tests:
1. Playing strength vs random opponent
2. Playing strength vs greedy opponent
3. ELO estimation via Stockfish
4. Search quality metrics

Usage:
    python evaluate_pure_mcts.py --sims 1600 --games 20
"""

import chess
import chess.engine
import numpy as np
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse

from pure_mcts_selfplay import PureMCTS, PureMCTSConfig


@dataclass
class EvaluationConfig:
    """Configuration for pure MCTS evaluation."""

    # MCTS parameters
    num_simulations: int = 1600
    c_puct: float = 1.414

    # Evaluation
    num_games: int = 20
    max_moves: int = 200

    # Stockfish
    stockfish_path: str = "/usr/games/stockfish"
    stockfish_time: float = 0.1  # seconds per move
    stockfish_levels: List[int] = None  # Skill levels to test

    # Output
    output_dir: Path = Path("pure_mcts_evaluation")

    def __post_init__(self):
        if self.stockfish_levels is None:
            self.stockfish_levels = [0, 1, 2, 3, 5]


class RandomPlayer:
    """Plays random legal moves."""

    def get_move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


class GreedyPlayer:
    """Plays simple material-greedy moves."""

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    def get_move(self, board: chess.Board) -> chess.Move:
        """Select move that maximizes immediate material gain."""
        legal_moves = list(board.legal_moves)
        best_score = float('-inf')
        best_moves = []

        for move in legal_moves:
            score = 0

            # Capturing moves
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    score += self.PIECE_VALUES.get(captured_piece.piece_type, 0)

            # Promotion
            if move.promotion:
                score += self.PIECE_VALUES.get(move.promotion, 0)

            # Check bonus
            board.push(move)
            if board.is_check():
                score += 0.5
            board.pop()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves) if best_moves else random.choice(legal_moves)


class StockfishPlayer:
    """Plays using Stockfish engine."""

    def __init__(self, stockfish_path: str, skill_level: int, time_limit: float):
        """
        Initialize Stockfish player.

        Args:
            stockfish_path: Path to stockfish binary
            skill_level: Skill level 0-20
            time_limit: Time per move in seconds
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Skill Level": skill_level})
        self.time_limit = time_limit
        self.skill_level = skill_level

    def get_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(
            board,
            chess.engine.Limit(time=self.time_limit)
        )
        return result.move

    def close(self):
        self.engine.quit()


class PureMCTSEvaluator:
    """Evaluates pure MCTS playing strength."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Create MCTS player
        mcts_config = PureMCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct
        )
        self.mcts = PureMCTS(mcts_config)

    def play_game(
        self,
        white_player,
        black_player,
        max_moves: int = 200
    ) -> Tuple[str, int, List[chess.Move]]:
        """
        Play a game between two players.

        Args:
            white_player: Player for white
            black_player: Player for black
            max_moves: Maximum number of moves

        Returns:
            (result, num_moves, move_list)
            result: "1-0", "0-1", or "1/2-1/2"
        """
        board = chess.Board()
        moves = []

        while not board.is_game_over() and len(moves) < max_moves:
            if board.turn == chess.WHITE:
                move = white_player.get_move(board)
            else:
                move = black_player.get_move(board)

            board.push(move)
            moves.append(move)

        if board.is_game_over():
            result = board.result()
        else:
            result = "1/2-1/2"  # Draw by move limit

        return result, len(moves), moves

    def evaluate_vs_random(self, num_games: int = 20) -> Dict:
        """
        Evaluate MCTS vs random player.

        Args:
            num_games: Number of games to play

        Returns:
            Statistics dictionary
        """
        print(f"\n{'='*80}")
        print(f"EVALUATION: Pure MCTS vs Random Player")
        print(f"{'='*80}")
        print(f"Games: {num_games} (alternating colors)")
        print(f"MCTS simulations: {self.config.num_simulations}")

        random_player = RandomPlayer()

        results = {
            'mcts_white_wins': 0,
            'mcts_black_wins': 0,
            'draws': 0,
            'total_moves': 0,
            'game_lengths': []
        }

        for game_num in range(num_games):
            # Alternate colors
            if game_num % 2 == 0:
                print(f"\nGame {game_num+1}: MCTS(White) vs Random(Black)")
                result, num_moves, _ = self.play_game(self.mcts, random_player)

                if result == "1-0":
                    results['mcts_white_wins'] += 1
                    print(f"  Result: MCTS wins as White ({num_moves} moves)")
                elif result == "0-1":
                    print(f"  Result: Random wins as Black ({num_moves} moves)")
                else:
                    results['draws'] += 1
                    print(f"  Result: Draw ({num_moves} moves)")
            else:
                print(f"\nGame {game_num+1}: Random(White) vs MCTS(Black)")
                result, num_moves, _ = self.play_game(random_player, self.mcts)

                if result == "0-1":
                    results['mcts_black_wins'] += 1
                    print(f"  Result: MCTS wins as Black ({num_moves} moves)")
                elif result == "1-0":
                    print(f"  Result: Random wins as White ({num_moves} moves)")
                else:
                    results['draws'] += 1
                    print(f"  Result: Draw ({num_moves} moves)")

            results['total_moves'] += num_moves
            results['game_lengths'].append(num_moves)

        # Calculate statistics
        total_mcts_wins = results['mcts_white_wins'] + results['mcts_black_wins']
        win_rate = total_mcts_wins / num_games

        results['total_wins'] = total_mcts_wins
        results['win_rate'] = win_rate
        results['avg_game_length'] = results['total_moves'] / num_games

        print(f"\n{'='*80}")
        print(f"RESULTS: Pure MCTS vs Random")
        print(f"{'='*80}")
        print(f"MCTS wins (White): {results['mcts_white_wins']}")
        print(f"MCTS wins (Black): {results['mcts_black_wins']}")
        print(f"Total MCTS wins: {total_mcts_wins}/{num_games}")
        print(f"Win rate: {win_rate*100:.1f}%")
        print(f"Draws: {results['draws']}")
        print(f"Average game length: {results['avg_game_length']:.1f} moves")

        return results

    def evaluate_vs_greedy(self, num_games: int = 20) -> Dict:
        """Evaluate MCTS vs greedy material player."""
        print(f"\n{'='*80}")
        print(f"EVALUATION: Pure MCTS vs Greedy Player")
        print(f"{'='*80}")
        print(f"Games: {num_games} (alternating colors)")

        greedy_player = GreedyPlayer()

        results = {
            'mcts_white_wins': 0,
            'mcts_black_wins': 0,
            'draws': 0,
            'total_moves': 0,
            'game_lengths': []
        }

        for game_num in range(num_games):
            if game_num % 2 == 0:
                print(f"\nGame {game_num+1}: MCTS(White) vs Greedy(Black)")
                result, num_moves, _ = self.play_game(self.mcts, greedy_player)

                if result == "1-0":
                    results['mcts_white_wins'] += 1
                    print(f"  Result: MCTS wins ({num_moves} moves)")
                elif result == "0-1":
                    print(f"  Result: Greedy wins ({num_moves} moves)")
                else:
                    results['draws'] += 1
                    print(f"  Result: Draw ({num_moves} moves)")
            else:
                print(f"\nGame {game_num+1}: Greedy(White) vs MCTS(Black)")
                result, num_moves, _ = self.play_game(greedy_player, self.mcts)

                if result == "0-1":
                    results['mcts_black_wins'] += 1
                    print(f"  Result: MCTS wins ({num_moves} moves)")
                elif result == "1-0":
                    print(f"  Result: Greedy wins ({num_moves} moves)")
                else:
                    results['draws'] += 1
                    print(f"  Result: Draw ({num_moves} moves)")

            results['total_moves'] += num_moves
            results['game_lengths'].append(num_moves)

        total_mcts_wins = results['mcts_white_wins'] + results['mcts_black_wins']
        results['total_wins'] = total_mcts_wins
        results['win_rate'] = total_mcts_wins / num_games
        results['avg_game_length'] = results['total_moves'] / num_games

        print(f"\n{'='*80}")
        print(f"RESULTS: Pure MCTS vs Greedy")
        print(f"{'='*80}")
        print(f"MCTS wins: {total_mcts_wins}/{num_games}")
        print(f"Win rate: {results['win_rate']*100:.1f}%")
        print(f"Draws: {results['draws']}")

        return results

    def evaluate_vs_stockfish(self, num_games_per_level: int = 10) -> Dict:
        """Evaluate MCTS vs Stockfish at different skill levels."""
        print(f"\n{'='*80}")
        print(f"EVALUATION: Pure MCTS vs Stockfish")
        print(f"{'='*80}")

        if not Path(self.config.stockfish_path).exists():
            print(f"Stockfish not found at {self.config.stockfish_path}")
            print("Skipping Stockfish evaluation")
            return {}

        results = {}

        for skill_level in self.config.stockfish_levels:
            print(f"\n{'='*80}")
            print(f"Testing vs Stockfish Skill Level {skill_level}")
            print(f"{'='*80}")

            stockfish = StockfishPlayer(
                self.config.stockfish_path,
                skill_level,
                self.config.stockfish_time
            )

            level_results = {
                'mcts_wins': 0,
                'stockfish_wins': 0,
                'draws': 0
            }

            for game_num in range(num_games_per_level):
                if game_num % 2 == 0:
                    print(f"Game {game_num+1}: MCTS(White) vs Stockfish(Black)")
                    result, num_moves, _ = self.play_game(self.mcts, stockfish)

                    if result == "1-0":
                        level_results['mcts_wins'] += 1
                        print(f"  MCTS wins ({num_moves} moves)")
                    elif result == "0-1":
                        level_results['stockfish_wins'] += 1
                        print(f"  Stockfish wins ({num_moves} moves)")
                    else:
                        level_results['draws'] += 1
                        print(f"  Draw ({num_moves} moves)")
                else:
                    print(f"Game {game_num+1}: Stockfish(White) vs MCTS(Black)")
                    result, num_moves, _ = self.play_game(stockfish, self.mcts)

                    if result == "0-1":
                        level_results['mcts_wins'] += 1
                        print(f"  MCTS wins ({num_moves} moves)")
                    elif result == "1-0":
                        level_results['stockfish_wins'] += 1
                        print(f"  Stockfish wins ({num_moves} moves)")
                    else:
                        level_results['draws'] += 1
                        print(f"  Draw ({num_moves} moves)")

            win_rate = level_results['mcts_wins'] / num_games_per_level
            level_results['win_rate'] = win_rate

            print(f"\nResults vs Stockfish Level {skill_level}:")
            print(f"  MCTS wins: {level_results['mcts_wins']}/{num_games_per_level}")
            print(f"  Win rate: {win_rate*100:.1f}%")

            results[f'level_{skill_level}'] = level_results
            stockfish.close()

        return results

    def estimate_elo(self, stockfish_results: Dict) -> Optional[int]:
        """
        Estimate ELO rating from Stockfish results.

        Stockfish skill level to ELO mapping (approximate):
        Level 0: ~800 ELO
        Level 1: ~900 ELO
        Level 2: ~1000 ELO
        Level 3: ~1100 ELO
        Level 5: ~1300 ELO
        Level 10: ~1700 ELO
        """
        if not stockfish_results:
            return None

        SKILL_TO_ELO = {
            0: 800,
            1: 900,
            2: 1000,
            3: 1100,
            5: 1300,
            10: 1700,
            15: 2100,
            20: 2500
        }

        # Find highest level where win rate > 50%
        highest_beaten = None
        for level in sorted(self.config.stockfish_levels):
            key = f'level_{level}'
            if key in stockfish_results:
                win_rate = stockfish_results[key]['win_rate']
                if win_rate > 0.5:
                    highest_beaten = level

        if highest_beaten is not None:
            estimated_elo = SKILL_TO_ELO.get(highest_beaten, 800)
            print(f"\nEstimated ELO: ~{estimated_elo}")
            print(f"(Based on beating Stockfish level {highest_beaten})")
            return estimated_elo
        else:
            print(f"\nEstimated ELO: <{SKILL_TO_ELO[min(self.config.stockfish_levels)]}")
            return SKILL_TO_ELO[min(self.config.stockfish_levels)]

    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite."""
        print(f"\n{'='*80}")
        print(f"PURE MCTS EVALUATION")
        print(f"{'='*80}")
        print(f"MCTS simulations per move: {self.config.num_simulations}")
        print(f"c_puct: {self.config.c_puct}")
        print(f"{'='*80}")

        all_results = {}

        # Test 1: vs Random
        all_results['vs_random'] = self.evaluate_vs_random(self.config.num_games)

        # Test 2: vs Greedy
        all_results['vs_greedy'] = self.evaluate_vs_greedy(self.config.num_games)

        # Test 3: vs Stockfish
        all_results['vs_stockfish'] = self.evaluate_vs_stockfish(
            num_games_per_level=min(10, self.config.num_games)
        )

        # ELO estimation
        all_results['estimated_elo'] = self.estimate_elo(all_results['vs_stockfish'])

        # Save results
        self.save_report(all_results)

        return all_results

    def save_report(self, results: Dict):
        """Save evaluation report to file."""
        report_file = self.config.output_dir / "pure_mcts_evaluation.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PURE MCTS EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Configuration:\n")
            f.write(f"  Simulations: {self.config.num_simulations}\n")
            f.write(f"  c_puct: {self.config.c_puct}\n")
            f.write(f"  Games per opponent: {self.config.num_games}\n\n")

            f.write("="*80 + "\n")
            f.write("RESULTS\n")
            f.write("="*80 + "\n\n")

            # vs Random
            if 'vs_random' in results:
                r = results['vs_random']
                f.write(f"vs Random Player:\n")
                f.write(f"  Win rate: {r['win_rate']*100:.1f}%\n")
                f.write(f"  Wins: {r['total_wins']}/{self.config.num_games}\n")
                f.write(f"  Draws: {r['draws']}\n\n")

            # vs Greedy
            if 'vs_greedy' in results:
                r = results['vs_greedy']
                f.write(f"vs Greedy Player:\n")
                f.write(f"  Win rate: {r['win_rate']*100:.1f}%\n")
                f.write(f"  Wins: {r['total_wins']}/{self.config.num_games}\n")
                f.write(f"  Draws: {r['draws']}\n\n")

            # vs Stockfish
            if 'vs_stockfish' in results and results['vs_stockfish']:
                f.write(f"vs Stockfish:\n")
                for level in sorted(self.config.stockfish_levels):
                    key = f'level_{level}'
                    if key in results['vs_stockfish']:
                        r = results['vs_stockfish'][key]
                        f.write(f"  Level {level}: {r['win_rate']*100:.1f}% win rate\n")
                f.write("\n")

            # ELO
            if 'estimated_elo' in results and results['estimated_elo']:
                f.write(f"Estimated ELO: ~{results['estimated_elo']}\n\n")

            f.write("="*80 + "\n")

        print(f"\nReport saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pure MCTS")
    parser.add_argument('--sims', type=int, default=1600, help='MCTS simulations per move')
    parser.add_argument('--games', type=int, default=20, help='Games per opponent')
    parser.add_argument('--stockfish', type=str, default='/usr/games/stockfish', help='Path to stockfish')
    parser.add_argument('--quick', action='store_true', help='Quick evaluation (fewer games)')

    args = parser.parse_args()

    config = EvaluationConfig(
        num_simulations=args.sims,
        num_games=5 if args.quick else args.games,
        stockfish_path=args.stockfish
    )

    evaluator = PureMCTSEvaluator(config)
    results = evaluator.run_full_evaluation()

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    if 'estimated_elo' in results and results['estimated_elo']:
        print(f"Estimated ELO: ~{results['estimated_elo']}")


if __name__ == '__main__':
    main()
