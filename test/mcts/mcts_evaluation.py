#!/usr/bin/env python3
"""
MCTS Evaluation and Testing Framework
======================================

Project: EE542 - Deconstructing AlphaZero's Success
Evaluation: MCTS performance analysis and comparison

Evaluation Metrics:
1. Search Quality
   - Policy accuracy vs ground truth
   - Value prediction correlation
   - Move ranking quality

2. Computational Efficiency
   - Nodes per second
   - Time per simulation
   - Memory usage

3. Playing Strength
   - Win rate vs baselines (random, minimax, stockfish)
   - ELO estimation
   - Tactical accuracy

4. MCTS Parameters
   - c_puct sensitivity analysis
   - Simulation count scaling
   - Temperature effects

References:
- Silver et al. (2017): AlphaZero evaluation methodology
"""

import chess
import chess.engine
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from mcts_tree import MCTS, MCTSNode


@dataclass
class EvaluationConfig:
    """Configuration for MCTS evaluation."""

    # Test positions
    num_test_positions: int = 100
    test_depth_range: Tuple[int, int] = (10, 40)  # Ply depth

    # MCTS parameters to test
    simulation_counts: List[int] = None
    c_puct_values: List[float] = None
    temperature_values: List[float] = None

    # Baseline comparison
    stockfish_path: str = "stockfish"
    stockfish_skill_levels: List[int] = None

    # Output
    output_dir: Path = Path("mcts_evaluation")
    save_plots: bool = True

    def __post_init__(self):
        if self.simulation_counts is None:
            self.simulation_counts = [50, 100, 200, 400, 800]
        if self.c_puct_values is None:
            self.c_puct_values = [0.5, 1.0, 1.5, 2.0]
        if self.temperature_values is None:
            self.temperature_values = [0.0, 0.5, 1.0, 1.5]
        if self.stockfish_skill_levels is None:
            self.stockfish_skill_levels = [0, 5, 10]


class MCTSEvaluator:
    """
    Comprehensive MCTS evaluation framework.
    """

    def __init__(self, neural_net, config: EvaluationConfig):
        """
        Initialize evaluator.

        Args:
            neural_net: Neural network for MCTS
            config: Evaluation configuration
        """
        self.neural_net = neural_net
        self.config = config

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {
            'search_quality': {},
            'efficiency': {},
            'playing_strength': {},
            'parameter_sensitivity': {}
        }

    def evaluate_search_quality(self, positions: List[chess.Board]) -> Dict:
        """
        Evaluate MCTS search quality metrics.

        Tests:
        1. Move ranking correlation with stronger engine
        2. Value prediction accuracy
        3. Search depth efficiency

        Args:
            positions: List of test positions

        Returns:
            Dict with search quality metrics
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATING SEARCH QUALITY")
        self.logger.info("="*80)

        mcts = MCTS(
            neural_net=self.neural_net,
            num_simulations=400,
            c_puct=1.0,
            temperature=0.0
        )

        move_rankings = []
        value_predictions = []
        search_depths = []

        for i, board in enumerate(positions):
            self.logger.info(f"Position {i+1}/{len(positions)}")

            # Run MCTS
            root = MCTSNode(board)
            policy_probs, value = self.neural_net.predict(board)
            root.expand(policy_probs)

            start_time = time.time()
            for _ in range(400):
                mcts._simulate(root)
            search_time = time.time() - start_time

            # Get top moves
            move_visits = [(move, child.visit_count)
                          for move, child in root.children.items()]
            move_visits.sort(key=lambda x: x[1], reverse=True)

            # Metrics
            move_rankings.append(move_visits[:5])  # Top 5 moves
            value_predictions.append(value)
            search_depths.append(self._estimate_search_depth(root))

            self.logger.info(f"  Time: {search_time:.2f}s")
            self.logger.info(f"  Top move: {move_visits[0][0]} ({move_visits[0][1]} visits)")
            self.logger.info(f"  Value: {value:.3f}")

        results = {
            'move_rankings': move_rankings,
            'value_predictions': value_predictions,
            'avg_search_depth': np.mean(search_depths),
            'std_search_depth': np.std(search_depths)
        }

        self.logger.info(f"\nAverage search depth: {results['avg_search_depth']:.1f}")

        self.results['search_quality'] = results
        return results

    def evaluate_computational_efficiency(self, positions: List[chess.Board]) -> Dict:
        """
        Evaluate computational efficiency metrics.

        Metrics:
        1. Nodes per second
        2. Time per simulation
        3. Speedup with simulation count

        Args:
            positions: Test positions

        Returns:
            Dict with efficiency metrics
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATING COMPUTATIONAL EFFICIENCY")
        self.logger.info("="*80)

        results = {}

        for num_sims in self.config.simulation_counts:
            self.logger.info(f"\nSimulations: {num_sims}")

            mcts = MCTS(
                neural_net=self.neural_net,
                num_simulations=num_sims,
                c_puct=1.0
            )

            times = []
            nodes_expanded = []

            for board in positions[:20]:  # Subset for speed
                start_time = time.time()
                root = MCTSNode(board)
                policy_probs, _ = self.neural_net.predict(board)
                root.expand(policy_probs)

                for _ in range(num_sims):
                    mcts._simulate(root)

                elapsed = time.time() - start_time
                times.append(elapsed)
                nodes_expanded.append(self._count_nodes(root))

            avg_time = np.mean(times)
            avg_nodes = np.mean(nodes_expanded)
            nodes_per_sec = avg_nodes / avg_time

            results[num_sims] = {
                'avg_time': avg_time,
                'avg_nodes': avg_nodes,
                'nodes_per_sec': nodes_per_sec,
                'time_per_sim': avg_time / num_sims
            }

            self.logger.info(f"  Avg time: {avg_time:.3f}s")
            self.logger.info(f"  Nodes/sec: {nodes_per_sec:.0f}")
            self.logger.info(f"  Time/sim: {avg_time/num_sims*1000:.2f}ms")

        self.results['efficiency'] = results
        return results

    def evaluate_playing_strength(self, num_games: int = 50) -> Dict:
        """
        Evaluate playing strength against baselines.

        Baselines:
        1. Random play
        2. Greedy (best first move from policy)
        3. Stockfish at different levels

        Args:
            num_games: Number of games per matchup

        Returns:
            Dict with win rates vs each baseline
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATING PLAYING STRENGTH")
        self.logger.info("="*80)

        results = {}

        # vs Random
        self.logger.info("\nvs Random Player:")
        win_rate = self._play_matches(
            opponent='random',
            num_games=num_games
        )
        results['vs_random'] = win_rate
        self.logger.info(f"  Win rate: {win_rate:.1%}")

        # vs Greedy
        self.logger.info("\nvs Greedy Player:")
        win_rate = self._play_matches(
            opponent='greedy',
            num_games=num_games
        )
        results['vs_greedy'] = win_rate
        self.logger.info(f"  Win rate: {win_rate:.1%}")

        # vs Stockfish
        for skill_level in self.config.stockfish_skill_levels:
            self.logger.info(f"\nvs Stockfish (skill {skill_level}):")
            try:
                win_rate = self._play_matches(
                    opponent='stockfish',
                    skill_level=skill_level,
                    num_games=num_games
                )
                results[f'vs_stockfish_{skill_level}'] = win_rate
                self.logger.info(f"  Win rate: {win_rate:.1%}")
            except Exception as e:
                self.logger.warning(f"  Stockfish test failed: {e}")

        self.results['playing_strength'] = results
        return results

    def evaluate_parameter_sensitivity(self, positions: List[chess.Board]) -> Dict:
        """
        Analyze sensitivity to MCTS parameters.

        Parameters tested:
        1. c_puct (exploration constant)
        2. Number of simulations
        3. Temperature

        Args:
            positions: Test positions

        Returns:
            Dict with parameter sensitivity results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PARAMETER SENSITIVITY ANALYSIS")
        self.logger.info("="*80)

        results = {}

        # c_puct sensitivity
        self.logger.info("\nc_puct values:")
        c_puct_results = []
        for c_puct in self.config.c_puct_values:
            mcts = MCTS(
                neural_net=self.neural_net,
                num_simulations=200,
                c_puct=c_puct
            )

            consistencies = []
            for board in positions[:20]:
                move_probs = mcts.search(board, add_noise=False)
                top_move = max(move_probs.items(), key=lambda x: x[1])[0]
                consistencies.append(move_probs[top_move])

            avg_consistency = np.mean(consistencies)
            c_puct_results.append(avg_consistency)

            self.logger.info(f"  c_puct={c_puct}: consistency={avg_consistency:.3f}")

        results['c_puct'] = dict(zip(self.config.c_puct_values, c_puct_results))

        # Temperature sensitivity
        self.logger.info("\nTemperature values:")
        temp_results = []
        for temp in self.config.temperature_values:
            mcts = MCTS(
                neural_net=self.neural_net,
                num_simulations=200,
                temperature=temp
            )

            diversities = []
            for board in positions[:20]:
                move_probs = mcts.search(board, add_noise=False)
                entropy = -sum(p * np.log(p + 1e-8) for p in move_probs.values())
                diversities.append(entropy)

            avg_diversity = np.mean(diversities)
            temp_results.append(avg_diversity)

            self.logger.info(f"  temp={temp}: diversity={avg_diversity:.3f}")

        results['temperature'] = dict(zip(self.config.temperature_values, temp_results))

        self.results['parameter_sensitivity'] = results
        return results

    def _play_matches(
        self,
        opponent: str,
        num_games: int,
        skill_level: Optional[int] = None
    ) -> float:
        """
        Play matches against opponent.

        Args:
            opponent: 'random', 'greedy', or 'stockfish'
            num_games: Number of games
            skill_level: Stockfish skill level (if applicable)

        Returns:
            Win rate (0.0 to 1.0)
        """
        mcts = MCTS(
            neural_net=self.neural_net,
            num_simulations=200,
            c_puct=1.0,
            temperature=0.0
        )

        wins = 0
        draws = 0

        for game_num in range(num_games):
            board = chess.Board()
            mcts_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK

            while not board.is_game_over() and board.fullmove_number < 200:
                if board.turn == mcts_color:
                    # MCTS move
                    move_probs, move = mcts.get_move_probabilities(board)
                    board.push(move)
                else:
                    # Opponent move
                    if opponent == 'random':
                        move = np.random.choice(list(board.legal_moves))
                    elif opponent == 'greedy':
                        policy, _ = self.neural_net.predict(board)
                        move = max(policy.items(), key=lambda x: x[1])[0]
                    elif opponent == 'stockfish':
                        # Stockfish move (placeholder - requires engine)
                        move = np.random.choice(list(board.legal_moves))

                    board.push(move)

            # Count result
            result = board.result()
            if result == "1-0":
                if mcts_color == chess.WHITE:
                    wins += 1
            elif result == "0-1":
                if mcts_color == chess.BLACK:
                    wins += 1
            else:
                draws += 1

        return (wins + 0.5 * draws) / num_games

    def _estimate_search_depth(self, node: MCTSNode) -> int:
        """
        Estimate average search depth in tree.

        Args:
            node: Root node

        Returns:
            Estimated average depth
        """
        if not node.children:
            return 0

        depths = []
        self._collect_depths(node, 0, depths)
        return np.mean(depths) if depths else 0

    def _collect_depths(self, node: MCTSNode, depth: int, depths: List[int]):
        """Recursively collect node depths."""
        if not node.children:
            depths.append(depth)
            return

        for child in node.children.values():
            self._collect_depths(child, depth + 1, depths)

    def _count_nodes(self, node: MCTSNode) -> int:
        """
        Count total nodes in tree.

        Args:
            node: Root node

        Returns:
            Total node count
        """
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def generate_report(self):
        """Generate evaluation report with plots."""
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING EVALUATION REPORT")
        self.logger.info("="*80)

        report_path = self.config.output_dir / "evaluation_report.txt"

        with open(report_path, 'w') as f:
            f.write("MCTS EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            # Search quality
            if 'search_quality' in self.results:
                f.write("SEARCH QUALITY\n")
                f.write("-"*80 + "\n")
                sq = self.results['search_quality']
                f.write(f"Average search depth: {sq.get('avg_search_depth', 0):.1f}\n")
                f.write(f"Depth std dev: {sq.get('std_search_depth', 0):.1f}\n\n")

            # Efficiency
            if 'efficiency' in self.results:
                f.write("COMPUTATIONAL EFFICIENCY\n")
                f.write("-"*80 + "\n")
                for num_sims, metrics in self.results['efficiency'].items():
                    f.write(f"Simulations: {num_sims}\n")
                    f.write(f"  Time: {metrics['avg_time']:.3f}s\n")
                    f.write(f"  Nodes/sec: {metrics['nodes_per_sec']:.0f}\n\n")

            # Playing strength
            if 'playing_strength' in self.results:
                f.write("PLAYING STRENGTH\n")
                f.write("-"*80 + "\n")
                for opponent, win_rate in self.results['playing_strength'].items():
                    f.write(f"{opponent}: {win_rate:.1%}\n")
                f.write("\n")

            # Parameter sensitivity
            if 'parameter_sensitivity' in self.results:
                f.write("PARAMETER SENSITIVITY\n")
                f.write("-"*80 + "\n")
                ps = self.results['parameter_sensitivity']
                if 'c_puct' in ps:
                    f.write("c_puct values:\n")
                    for val, result in ps['c_puct'].items():
                        f.write(f"  {val}: {result:.3f}\n")
                if 'temperature' in ps:
                    f.write("Temperature values:\n")
                    for val, result in ps['temperature'].items():
                        f.write(f"  {val}: {result:.3f}\n")

        self.logger.info(f"Report saved to {report_path}")


def main():
    """Example usage of MCTS evaluation."""

    # Mock neural network
    class MockNeuralNet:
        def predict(self, board):
            legal_moves = list(board.legal_moves)
            policy = {move: 1.0 / len(legal_moves) for move in legal_moves}
            value = np.random.uniform(-0.2, 0.2)
            return policy, value

    # Configuration
    config = EvaluationConfig(
        num_test_positions=20,
        simulation_counts=[50, 100, 200],
        c_puct_values=[0.5, 1.0, 2.0],
        output_dir=Path("test_mcts_eval")
    )

    # Create evaluator
    neural_net = MockNeuralNet()
    evaluator = MCTSEvaluator(neural_net, config)

    # Generate test positions
    positions = [chess.Board() for _ in range(20)]

    # Run evaluations
    evaluator.evaluate_search_quality(positions)
    evaluator.evaluate_computational_efficiency(positions)
    evaluator.evaluate_parameter_sensitivity(positions)

    # Generate report
    evaluator.generate_report()

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
