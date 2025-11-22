#!/usr/bin/env python3
"""
Complete MCTS + Neural Network Evaluation Pipeline
===================================================

Project: EE542 - Deconstructing AlphaZero's Success
Full Pipeline: Train -> MCTS -> Self-Play -> Evaluate

This script runs the complete evaluation workflow:
1. Load trained CNN model
2. Test MCTS search quality
3. Generate self-play games
4. Evaluate playing strength
5. Compare different configurations
6. Generate comprehensive report

Usage:
    python run_full_evaluation.py --model path/to/model.pth
    python run_full_evaluation.py --model path/to/model.pth --quick
"""

import argparse
import sys
from pathlib import Path
import logging
import time
import json
import chess
import numpy as np

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "FINAL" / "chess_ai"))

from neural_mcts_player import NeuralMCTSPlayer, RandomPlayer
from mcts_evaluation import MCTSEvaluator, EvaluationConfig
from self_play import SelfPlayWorker, SelfPlayConfig
from evaluation.elo_rating import StockfishBenchmark


class FullEvaluationPipeline:
    """
    Complete evaluation pipeline for neural MCTS chess player.
    """

    def __init__(self, model_path: str, output_dir: Path, quick_mode: bool = False):
        """
        Initialize evaluation pipeline.

        Args:
            model_path: Path to trained model checkpoint
            output_dir: Directory for output files
            quick_mode: If True, use reduced parameters for faster testing
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.quick_mode = quick_mode

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'evaluation.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {
            'model_path': model_path,
            'quick_mode': quick_mode,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluations': {}
        }

    def run_complete_evaluation(self):
        """Run all evaluation stages."""

        self.logger.info("="*80)
        self.logger.info("COMPLETE NEURAL MCTS EVALUATION PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Model: {self.model_path}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Quick mode: {self.quick_mode}")
        self.logger.info("="*80)

        # Stage 1: Load and test model
        self.logger.info("\n[STAGE 1] Loading model and basic testing...")
        player = self._load_model()

        # Stage 2: MCTS search quality
        self.logger.info("\n[STAGE 2] Evaluating MCTS search quality...")
        self._evaluate_search_quality(player)

        # Stage 3: Playing strength
        self.logger.info("\n[STAGE 3] Evaluating playing strength...")
        self._evaluate_playing_strength(player)

        # Stage 4: Parameter sensitivity
        self.logger.info("\n[STAGE 4] Parameter sensitivity analysis...")
        self._evaluate_parameter_sensitivity(player)

        # Stage 5: Self-play generation (optional)
        if not self.quick_mode:
            self.logger.info("\n[STAGE 5] Generating self-play games...")
            self._generate_selfplay_games(player)

        # Stage 6: Stockfish benchmark
        self.logger.info("\n[STAGE 6] Stockfish ELO estimation...")
        self._estimate_elo(player)

        # Stage 7: Generate report
        self.logger.info("\n[STAGE 7] Generating final report...")
        self._generate_final_report()

        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info("="*80)

    def _load_model(self) -> NeuralMCTSPlayer:
        """Load model and run basic tests."""

        player = NeuralMCTSPlayer(
            model_path=self.model_path,
            num_simulations=100 if self.quick_mode else 400,
            c_puct=1.0,
            temperature=0.0
        )

        # Test on starting position
        board = chess.Board()
        eval_result = player.get_position_evaluation(board)

        self.results['evaluations']['starting_position'] = {
            'value': eval_result['value'],
            'top_move': eval_result['top_moves'][0][0],
            'top_move_prob': eval_result['top_moves'][0][1]
        }

        self.logger.info(f"Starting position value: {eval_result['value']:.3f}")
        self.logger.info(f"Top move: {eval_result['top_moves'][0][0]} "
                        f"({eval_result['top_moves'][0][1]:.3f})")

        return player

    def _evaluate_search_quality(self, player: NeuralMCTSPlayer):
        """Evaluate MCTS search quality metrics."""

        num_positions = 10 if self.quick_mode else 50
        positions = self._generate_test_positions(num_positions)

        search_times = []
        value_predictions = []

        for i, board in enumerate(positions):
            start_time = time.time()
            evaluation = player.get_position_evaluation(board)
            search_time = time.time() - start_time

            search_times.append(search_time)
            value_predictions.append(evaluation['value'])

            if i % 10 == 0:
                self.logger.info(f"Evaluated {i+1}/{len(positions)} positions")

        self.results['evaluations']['search_quality'] = {
            'avg_search_time': float(np.mean(search_times)),
            'std_search_time': float(np.std(search_times)),
            'avg_value': float(np.mean(value_predictions)),
            'std_value': float(np.std(value_predictions))
        }

        self.logger.info(f"Avg search time: {np.mean(search_times):.3f}s")
        self.logger.info(f"Avg value: {np.mean(value_predictions):.3f}")

    def _evaluate_playing_strength(self, player: NeuralMCTSPlayer):
        """Evaluate playing strength against baselines."""

        num_games = 10 if self.quick_mode else 30

        # vs Random
        self.logger.info(f"Playing {num_games} games vs Random...")
        random_player = RandomPlayer()

        wins = 0
        draws = 0

        for game_num in range(num_games):
            player_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
            result, moves = player.play_game(random_player, player_color, max_moves=150)

            if (result == "1-0" and player_color == chess.WHITE) or \
               (result == "0-1" and player_color == chess.BLACK):
                wins += 1
            elif result == "1/2-1/2":
                draws += 1

            if (game_num + 1) % 5 == 0:
                self.logger.info(f"Completed {game_num+1}/{num_games} games")

        win_rate = (wins + 0.5 * draws) / num_games

        self.results['evaluations']['vs_random'] = {
            'games': num_games,
            'wins': wins,
            'draws': draws,
            'losses': num_games - wins - draws,
            'win_rate': float(win_rate)
        }

        self.logger.info(f"vs Random: {win_rate:.1%} ({wins}W {draws}D "
                        f"{num_games-wins-draws}L)")

    def _evaluate_parameter_sensitivity(self, player: NeuralMCTSPlayer):
        """Test sensitivity to MCTS parameters."""

        test_positions = self._generate_test_positions(5)

        # Test different simulation counts
        sim_counts = [50, 100, 200] if self.quick_mode else [50, 100, 200, 400, 800]
        sim_results = {}

        self.logger.info("Testing simulation counts...")
        for num_sims in sim_counts:
            test_player = NeuralMCTSPlayer(
                model_path=self.model_path,
                num_simulations=num_sims,
                c_puct=1.0,
                temperature=0.0
            )

            times = []
            for board in test_positions:
                start_time = time.time()
                test_player.select_move(board)
                times.append(time.time() - start_time)

            sim_results[num_sims] = {
                'avg_time': float(np.mean(times)),
                'std_time': float(np.std(times))
            }

            self.logger.info(f"  {num_sims} sims: {np.mean(times):.3f}s")

        self.results['evaluations']['simulation_sensitivity'] = sim_results

        # Test different c_puct values
        c_puct_values = [0.5, 1.0, 2.0]
        c_puct_results = {}

        self.logger.info("Testing c_puct values...")
        for c_puct in c_puct_values:
            test_player = NeuralMCTSPlayer(
                model_path=self.model_path,
                num_simulations=100,
                c_puct=c_puct,
                temperature=0.0
            )

            # Test move consistency
            moves_selected = []
            for board in test_positions:
                move = test_player.select_move(board)
                moves_selected.append(move.uci())

            c_puct_results[c_puct] = {
                'moves': moves_selected
            }

            self.logger.info(f"  c_puct={c_puct}: moves={moves_selected[:3]}")

        self.results['evaluations']['c_puct_sensitivity'] = c_puct_results

    def _generate_selfplay_games(self, player: NeuralMCTSPlayer):
        """Generate self-play games for training data."""

        num_games = 5 if self.quick_mode else 20

        self.logger.info(f"Generating {num_games} self-play games...")

        config = SelfPlayConfig(
            num_simulations=100 if self.quick_mode else 200,
            c_puct=1.0,
            temp_threshold=15,
            temp_init=1.0,
            temp_final=0.1,
            max_game_length=200,
            save_frequency=5,
            output_dir=self.output_dir / "selfplay_games"
        )

        worker = SelfPlayWorker(player.neural_net, config)
        examples = worker.generate_games(num_games)

        self.results['evaluations']['selfplay'] = {
            'num_games': num_games,
            'total_examples': len(examples),
            'avg_examples_per_game': len(examples) / num_games
        }

        self.logger.info(f"Generated {len(examples)} training examples")

    def _estimate_elo(self, player: NeuralMCTSPlayer):
        """Estimate ELO using Stockfish benchmark."""

        try:
            benchmark = StockfishBenchmark()

            if not benchmark.available:
                self.logger.warning("Stockfish not available, skipping ELO estimation")
                return

            num_games = 5 if self.quick_mode else 10

            # Create wrapper for player
            class PlayerWrapper:
                def __init__(self, mcts_player):
                    self.player = mcts_player

                def predict_move(self, board):
                    return self.player.select_move(board)

            wrapped_player = PlayerWrapper(player)

            self.logger.info(f"Running Stockfish benchmark ({num_games} games per level)...")
            result = benchmark.estimate_elo(wrapped_player, num_games_per_level=num_games)

            if 'estimated_elo' in result:
                self.results['evaluations']['elo_estimate'] = {
                    'estimated_elo': result['estimated_elo'],
                    'method': result['method'],
                    'results_by_level': {
                        str(level): {
                            'rating': data['rating'],
                            'score_rate': data['score_rate'],
                            'wins': data['wins'],
                            'draws': data['draws'],
                            'losses': data['losses']
                        }
                        for level, data in result.get('results_by_level', {}).items()
                    }
                }

                self.logger.info(f"Estimated ELO: {result['estimated_elo']}")

        except Exception as e:
            self.logger.error(f"ELO estimation failed: {e}")

    def _generate_test_positions(self, num_positions: int) -> list:
        """Generate test positions at various depths."""

        positions = []

        for _ in range(num_positions):
            board = chess.Board()
            num_moves = np.random.randint(5, 40)

            for _ in range(num_moves):
                if board.is_game_over():
                    break
                move = np.random.choice(list(board.legal_moves))
                board.push(move)

            if not board.is_game_over():
                positions.append(board.copy())

        return positions

    def _generate_final_report(self):
        """Generate comprehensive evaluation report."""

        report_path = self.output_dir / "evaluation_report.txt"
        results_json = self.output_dir / "results.json"

        # Save JSON results
        with open(results_json, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Generate text report
        with open(report_path, 'w') as f:
            f.write("NEURAL MCTS CHESS PLAYER EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Model: {self.model_path}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Quick mode: {self.quick_mode}\n\n")

            f.write("-"*80 + "\n")
            f.write("SEARCH QUALITY\n")
            f.write("-"*80 + "\n")
            if 'search_quality' in self.results['evaluations']:
                sq = self.results['evaluations']['search_quality']
                f.write(f"Avg search time: {sq['avg_search_time']:.3f}s\n")
                f.write(f"Avg position value: {sq['avg_value']:.3f}\n\n")

            f.write("-"*80 + "\n")
            f.write("PLAYING STRENGTH\n")
            f.write("-"*80 + "\n")
            if 'vs_random' in self.results['evaluations']:
                vr = self.results['evaluations']['vs_random']
                f.write(f"vs Random: {vr['win_rate']:.1%} ")
                f.write(f"({vr['wins']}W {vr['draws']}D {vr['losses']}L)\n\n")

            if 'elo_estimate' in self.results['evaluations']:
                f.write("-"*80 + "\n")
                f.write("ELO ESTIMATION\n")
                f.write("-"*80 + "\n")
                elo = self.results['evaluations']['elo_estimate']
                f.write(f"Estimated ELO: {elo['estimated_elo']}\n")
                f.write(f"Method: {elo['method']}\n\n")

                if 'results_by_level' in elo:
                    f.write("Results by Stockfish level:\n")
                    for level, data in elo['results_by_level'].items():
                        f.write(f"  Level {level} (ELO {data['rating']}): ")
                        f.write(f"{data['score_rate']:.1%} ")
                        f.write(f"({data['wins']}W {data['draws']}D {data['losses']}L)\n")

        self.logger.info(f"Report saved to {report_path}")
        self.logger.info(f"JSON results saved to {results_json}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Complete Neural MCTS Evaluation Pipeline"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='mcts_evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode with reduced parameters'
    )

    args = parser.parse_args()

    # Verify model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    # Run pipeline
    pipeline = FullEvaluationPipeline(
        model_path=args.model,
        output_dir=Path(args.output),
        quick_mode=args.quick
    )

    pipeline.run_complete_evaluation()


if __name__ == '__main__':
    main()
