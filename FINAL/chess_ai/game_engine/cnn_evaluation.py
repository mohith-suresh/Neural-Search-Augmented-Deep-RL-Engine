import json
import logging
import chess
import chess.engine
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime
from tqdm import tqdm

from cnn import ChessCNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class StockfishEloEvaluator:
    STOCKFISH_ELO_MAP = {
        1: 800,
        2: 1000,
        3: 1200,
        4: 1400,
        5: 1600,
        6: 1800,
        7: 2000,
        8: 2200,
        9: 2400,
        10: 2600,
    }

    def __init__(self, stockfish_path: str = "stockfish"):
        self.stockfish_path = stockfish_path
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            logger.info("Stockfish engine initialized")
        except Exception as e:
            logger.warning(f"Stockfish not available: {e}. Skipping Elo estimation.")
            self.engine = None

    def get_best_move(self, board: chess.Board, depth: int = 15) -> Optional[chess.Move]:
        if not self.engine:
            return None

        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                info=chess.engine.INFO_ALL
            )
            if 'pv' in info and len(info['pv']) > 0:
                return info['pv'][0]
            return None
        except Exception as e:
            logger.warning(f"Engine analysis failed: {e}")
            return None

    def play_game(self, model_predictor: Callable, max_moves: int = 200) -> Dict:
        if not self.engine:
            return None

        board = chess.Board()
        move_count = 0
        model_moves_count = 0

        while not board.is_game_over() and move_count < max_moves:
            if board.turn:
                model_move = model_predictor(board)
                if not model_move:
                    break
                board.push(model_move)
                model_moves_count += 1
            else:
                stockfish_move = self.get_best_move(board)
                if not stockfish_move:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        stockfish_move = legal_moves[0]
                    else:
                        break
                board.push(stockfish_move)

            move_count += 1

        result = board.result()
        if result == '1-0':
            outcome = 'model_win'
        elif result == '0-1':
            outcome = 'stockfish_win'
        else:
            outcome = 'draw'

        return {
            'outcome': outcome,
            'model_moves': model_moves_count,
            'total_plies': move_count,
            'result': result
        }

    def estimate_elo(self, model_predictor: Callable, num_games_per_level: int = 3) -> Dict:
        results_by_level = {}

        for skill_level in sorted(self.STOCKFISH_ELO_MAP.keys()):
            logger.info(f"Testing vs Stockfish level {skill_level} ({self.STOCKFISH_ELO_MAP[skill_level]} Elo)")

            wins = 0
            draws = 0
            losses = 0

            for game_num in tqdm(range(num_games_per_level), desc=f"Level {skill_level}"):
                try:
                    game_result = self.play_game(model_predictor)
                    if not game_result:
                        losses += 1
                        continue

                    if game_result['outcome'] == 'model_win':
                        wins += 1
                    elif game_result['outcome'] == 'draw':
                        draws += 1
                    else:
                        losses += 1

                except Exception as e:
                    logger.error(f"Game failed: {e}")
                    losses += 1

            total = wins + draws + losses
            score = (wins + 0.5 * draws) / total if total > 0 else 0

            results_by_level[skill_level] = {
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'score': float(score),
                'elo': self.STOCKFISH_ELO_MAP[skill_level]
            }

            logger.info(f"Level {skill_level}: W={wins} D={draws} L={losses} Score={score:.3f}")

        overall_score = np.mean([r['score'] for r in results_by_level.values()])
        estimated_elo = self._score_to_elo(overall_score)
        confidence_lower = estimated_elo - 100
        confidence_upper = estimated_elo + 100

        return {
            'estimated_elo': int(estimated_elo),
            'confidence_lower': int(confidence_lower),
            'confidence_upper': int(confidence_upper),
            'margin_of_error': 100,
            'overall_score': float(overall_score),
            'results_by_level': results_by_level,
            'method': 'stockfish_bracketing',
            'games_played': len(results_by_level) * num_games_per_level
        }

    def _score_to_elo(self, score: float) -> float:
        if score <= 0:
            return 400
        elif score >= 1.0:
            return 2600
        else:
            return 2000 + 400 * np.log10(score / (1 - score)) / np.log(10)

    def close(self):
        if self.engine:
            self.engine.quit()
            logger.info("Stockfish engine closed")


class ModelEvaluator:
    def __init__(self, model_path: str, device: str = 'cuda', stockfish_path: str = 'stockfish'):
        self.device = device
        self.model_path = model_path
        self.stockfish_path = stockfish_path

        self.model = ChessCNN(num_filters=128, num_residual_blocks=10).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.model.eval()

        self.results = {}

    def encode_board(self, board: chess.Board) -> np.ndarray:
        piece_to_channel = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }

        tensor = np.zeros((12, 8, 8), dtype=np.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = piece_to_channel[(piece.piece_type, piece.color)]
                tensor[channel, rank, file] = 1.0

        return tensor

    def predict_move(self, board: chess.Board) -> chess.Move:
        with torch.no_grad():
            piece_channels = self.encode_board(board)
            board_tensor = torch.from_numpy(piece_channels).float()
            board_tensor = board_tensor.unsqueeze(0).to(self.device)

            policy_logits, _ = self.model(board_tensor)
            policy_logits = policy_logits.cpu().numpy()[0]

            legal_moves = list(board.legal_moves)

            best_prob = -1
            best_move = None

            for move in legal_moves:
                from_sq = move.from_square
                to_sq = move.to_square

                if move.promotion:
                    move_idx = from_sq * 64 + to_sq + 4096
                else:
                    move_idx = from_sq * 64 + to_sq

                if move_idx < 8192:
                    prob = policy_logits[move_idx]
                    if prob > best_prob:
                        best_prob = prob
                        best_move = move

            return best_move if best_move else legal_moves[0]

    def estimate_elo_stockfish(self, num_games_per_level: int = 3) -> Optional[Dict]:
        logger.info("Starting Stockfish Elo estimation")
        
        evaluator = StockfishEloEvaluator(self.stockfish_path)
        
        try:
            if not evaluator.engine:
                logger.warning("Stockfish engine not available, skipping Elo estimation")
                return None

            elo_results = evaluator.estimate_elo(self.predict_move, num_games_per_level=num_games_per_level)
            return elo_results
        except Exception as e:
            logger.error(f"Elo estimation failed: {e}")
            return None
        finally:
            evaluator.close()

    def benchmark_opening_positions(self) -> Dict:
        opening_positions = {
            'Italian Game': 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 4',
            'Sicilian Defense': 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
            'French Defense': 'rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
            'Ruy Lopez': 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 4',
        }

        results = {}

        for position_name, fen in opening_positions.items():
            board = chess.Board(fen)
            model_move = self.predict_move(board)

            results[position_name] = {
                'model_move': model_move.uci() if model_move else None,
            }

        return {
            'num_positions': len(opening_positions),
            'results': results
        }

    def analyze_game_quality(self, num_games: int = 5) -> Dict:
        game_analyses = []
        wins = 0
        losses = 0
        draws = 0

        for game_num in tqdm(range(num_games), desc="Game Analysis"):
            board = chess.Board()
            moves_played = 0
            max_moves = 200

            while not board.is_game_over() and moves_played < max_moves:
                move = self.predict_move(board)
                board.push(move)
                moves_played += 1

            result = board.result()
            
            if result == '1-0':
                wins += 1
                outcome = 'white_win'
            elif result == '0-1':
                losses += 1
                outcome = 'black_win'
            elif result == '1/2-1/2':
                draws += 1
                outcome = 'draw'
            else:
                outcome = 'unfinished'

            game_analyses.append({
                'game_num': game_num,
                'moves': moves_played,
                'result': result,
                'outcome': outcome
            })

        total_moves = sum(g['moves'] for g in game_analyses)

        return {
            'num_games': num_games,
            'total_moves': total_moves,
            'avg_moves_per_game': total_moves / num_games if num_games > 0 else 0,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games if num_games > 0 else 0,
            'draw_rate': draws / num_games if num_games > 0 else 0,
            'loss_rate': losses / num_games if num_games > 0 else 0,
        }

    def plot_training_curves(self, output_dir: str) -> Optional[str]:
        if not self.training_history or 'train_loss' not in self.training_history:
            return None

        train_loss = self.training_history.get('train_loss', [])
        val_loss = self.training_history.get('val_loss', [])
        
        if len(train_loss) <= 1:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

        train_acc = self.training_history.get('train_accuracy', [])
        val_acc = self.training_history.get('val_accuracy', [])

        epochs = range(1, len(train_loss) + 1)

        axes[0, 0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Accuracy', fontsize=11)
        axes[0, 1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        train_policy_loss = self.training_history.get('train_policy_loss', [])
        val_policy_loss = self.training_history.get('val_policy_loss', [])
        
        if train_policy_loss and val_policy_loss:
            axes[1, 0].plot(epochs, train_policy_loss, 'g-', label='Training Policy Loss', linewidth=2)
            axes[1, 0].plot(epochs, val_policy_loss, 'm-', label='Validation Policy Loss', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=11)
            axes[1, 0].set_ylabel('Policy Loss', fontsize=11)
            axes[1, 0].set_title('Policy Loss', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        train_value_loss = self.training_history.get('train_value_loss', [])
        val_value_loss = self.training_history.get('val_value_loss', [])
        
        if train_value_loss and val_value_loss:
            axes[1, 1].plot(epochs, train_value_loss, 'c-', label='Training Value Loss', linewidth=2)
            axes[1, 1].plot(epochs, val_value_loss, 'orange', label='Validation Value Loss', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=11)
            axes[1, 1].set_ylabel('Value Loss', fontsize=11)
            axes[1, 1].set_title('Value Loss', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        
        plot_path = Path(output_dir) / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def plot_elo_stockfish(self, elo_data: Dict, output_dir: str) -> Optional[str]:
        if not elo_data:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        elo = elo_data['estimated_elo']
        lower = elo_data['confidence_lower']
        upper = elo_data['confidence_upper']

        ax1.barh(['Model'], [elo], color='steelblue', height=0.5)
        ax1.errorbar([elo], ['Model'], xerr=[[elo - lower], [upper - elo]], 
                    fmt='none', ecolor='red', capsize=10, capthick=2, elinewidth=2)

        ax1.set_xlabel('Elo Rating', fontsize=12, fontweight='bold')
        ax1.set_title(f'Model Elo: {elo} ± {elo_data["margin_of_error"]}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 2800)
        ax1.grid(True, alpha=0.3, axis='x')

        ax1.text(elo, -0.25, f'{elo}', ha='center', fontsize=12, fontweight='bold')

        levels = []
        scores = []
        elo_labels = []
        
        for level, result in sorted(elo_data['results_by_level'].items()):
            levels.append(f"L{level}")
            scores.append(result['score'])
            elo_labels.append(result['elo'])

        colors = ['green' if s > 0.5 else 'orange' if s > 0.3 else 'red' for s in scores]
        ax2.bar(levels, scores, color=colors, alpha=0.7)
        ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Equal')
        ax2.set_ylabel('Score (W + 0.5D)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Stockfish Skill Level', fontsize=12, fontweight='bold')
        ax2.set_title('Performance vs Stockfish Levels', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (level, score, elo_val) in enumerate(zip(levels, scores, elo_labels)):
            ax2.text(i, score + 0.02, f'{elo_val}\n{score:.2f}', ha='center', fontsize=9)

        ax2.legend()

        plt.tight_layout()
        
        plot_path = Path(output_dir) / 'elo_stockfish.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def generate_report(self, output_dir: str, version: str = 'v1') -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("Running comprehensive model evaluation")

        elo_data = self.estimate_elo_stockfish(num_games_per_level=3)
        self.results['opening_benchmark'] = self.benchmark_opening_positions()
        self.results['game_analysis'] = self.analyze_game_quality(num_games=5)

        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'device': self.device,
            'version': version,
            'elo_rating_stockfish': elo_data,
            'results': self.results
        }

        json_path = output_path / f"evaluation_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        text_report = self._generate_text_report(report)

        text_path = output_path / f"evaluation_summary_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write(text_report)

        self.plot_training_curves(output_dir)
        if elo_data:
            self.plot_elo_stockfish(elo_data, output_dir)

        return text_report

    def _generate_text_report(self, report: Dict) -> str:
        lines = [
            "=" * 80,
            "CHESS AI MODEL COMPREHENSIVE EVALUATION REPORT",
            "=" * 80,
            f"Timestamp: {report['evaluation_timestamp']}",
            f"Model: {report['model_path']}",
            f"Device: {report['device']}",
            f"Version: {report['version']}",
            "",
        ]

        if report.get('elo_rating_stockfish'):
            elo = report['elo_rating_stockfish']
            lines.extend([
                "ELO RATING (STOCKFISH BENCHMARK - INDUSTRY STANDARD)",
                "-" * 80,
                f"Estimated Elo: {elo['estimated_elo']}",
                f"Confidence Range: {elo['confidence_lower']} - {elo['confidence_upper']}",
                f"Margin of Error: ±{elo['margin_of_error']}",
                f"Overall Score vs Stockfish: {elo['overall_score']:.3f}",
                f"Total Games: {elo['games_played']}",
                f"Method: {elo['method']}",
                "",
                "Results by Stockfish Level:",
            ])
            
            for level, results in sorted(elo['results_by_level'].items()):
                lines.append(f"  Level {level} ({results['elo']} Elo): W={results['wins']} D={results['draws']} L={results['losses']} Score={results['score']:.3f}")
            
            lines.append("")
        else:
            lines.extend([
                "ELO RATING",
                "-" * 80,
                "Stockfish not available - Elo estimation skipped",
                "",
            ])

        if 'opening_benchmark' in report['results']:
            open_data = report['results']['opening_benchmark']
            lines.extend([
                "OPENING BENCHMARK",
                "-" * 80,
                f"Positions Tested: {open_data['num_positions']}",
                "",
            ])

        if 'game_analysis' in report['results']:
            game_data = report['results']['game_analysis']
            lines.extend([
                "GAME ANALYSIS (SELF-PLAY)",
                "-" * 80,
                f"Games Analyzed: {game_data['num_games']}",
                f"Total Moves: {game_data['total_moves']}",
                f"Average Moves per Game: {game_data['avg_moves_per_game']:.1f}",
                "",
                "Game Outcomes:",
                f"  Wins (White): {game_data['wins']} ({game_data['win_rate']*100:.1f}%)",
                f"  Losses (Black): {game_data['losses']} ({game_data['loss_rate']*100:.1f}%)",
                f"  Draws: {game_data['draws']} ({game_data['draw_rate']*100:.1f}%)",
                "",
            ])

        lines.extend([
            "GENERATED OUTPUTS",
            "-" * 80,
            "training_curves.png - Training/Validation curves (if 2+ epochs)",
            "elo_stockfish.png - Elo estimate and performance vs Stockfish",
            "evaluation_report_*.json - Full metrics in JSON format",
            "",
        ])

        lines.append("=" * 80)

        return "\n".join(lines)


def evaluate_model(model_path: str, output_dir: str = 'evaluation_reports', version: str = 'v1'):
    evaluator = ModelEvaluator(model_path)
    report = evaluator.generate_report(output_dir, version)
    print(report)
    return evaluator.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--output-dir', default='evaluation_reports', help='Output directory for results')
    parser.add_argument('--version', default='v1', help='Model version name')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')

    args = parser.parse_args()

    evaluate_model(args.model_path, args.output_dir, args.version)
