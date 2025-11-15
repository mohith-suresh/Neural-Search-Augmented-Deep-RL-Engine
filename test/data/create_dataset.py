# create_dataset_verbose.py
"""
Dataset creation with verbose progress tracking
"""

import chess.pgn
import numpy as np
from tqdm import tqdm
import subprocess
from pathlib import Path
from datetime import datetime
import time

class VerboseChessDatasetCreator:
    """Enhanced version with better progress tracking"""
    
    def __init__(self, min_elo=1900, time_control_min=600):
        self.min_elo = min_elo
        self.time_control_min = time_control_min
        
        self.piece_to_channel = {
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
        
        self.stats = {
            'games_processed': 0,
            'games_used': 0,
            'positions_extracted': 0,
            'start_time': None
        }
    
    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float16)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = self.piece_to_channel[(piece.piece_type, piece.color)]
                tensor[channel][rank][file] = 1.0
        return tensor
    
    def move_to_index(self, move):
        from_sq = move.from_square
        to_sq = move.to_square
        if move.promotion:
            return from_sq * 64 + to_sq + 4096
        return from_sq * 64 + to_sq
    
    def passes_filters(self, game):
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            if white_elo < self.min_elo or black_elo < self.min_elo:
                return False

            time_control = game.headers.get("TimeControl", "")
            if not time_control or time_control == "-":
                return False
            base_time = int(time_control.split('+')[0])
            if base_time < self.time_control_min:
                return False

            # Calculate move count from actual game moves (PlyCount header is often missing)
            ply_count = game.headers.get("PlyCount", None)
            if ply_count:
                move_count = int(ply_count) // 2
            else:
                # Count moves manually
                move_count = sum(1 for _ in game.mainline_moves()) // 2

            if move_count < 15 or move_count > 200:
                return False

            termination = game.headers.get("Termination", "")
            if termination not in ["Normal", "Time forfeit"]:
                return False

            return True
        except (ValueError, TypeError, KeyError):
            return False
    
    def get_result_value(self, game):
        result = game.headers.get("Result", "*")
        return 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
    
    def stream_from_compressed(self, zst_file):
        print(f"🔄 Streaming from: {zst_file.name}")
        process = subprocess.Popen(
            ['zstdcat', str(zst_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024*1024,
            universal_newlines=True
        )
        
        try:
            while True:
                game = chess.pgn.read_game(process.stdout)
                if game is None:
                    break
                yield game
        finally:
            process.terminate()
            process.wait()
    
    def print_progress(self, positions_count, target):
        """Print detailed progress every 1000 games"""
        elapsed = time.time() - self.stats['start_time']
        
        pass_rate = (self.stats['games_used'] / self.stats['games_processed'] * 100) if self.stats['games_processed'] > 0 else 0
        progress_pct = (positions_count / target * 100)
        
        # Estimate time remaining
        if positions_count > 0:
            rate = positions_count / elapsed  # positions per second
            remaining = (target - positions_count) / rate if rate > 0 else 0
            eta = remaining / 3600  # hours
        else:
            eta = 0
        
        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress Update")
        print(f"{'='*70}")
        print(f"Games processed:     {self.stats['games_processed']:>10,}")
        print(f"Games used:          {self.stats['games_used']:>10,} ({pass_rate:.1f}% pass rate)")
        print(f"Positions extracted: {positions_count:>10,} ({progress_pct:.2f}%)")
        print(f"Time elapsed:        {elapsed/3600:>10.1f} hours")
        print(f"Estimated remaining: {eta:>10.1f} hours")
        print(f"{'='*70}\n")
    
    def create_dataset(self, input_files, output_file, target_positions=3_000_000):
        if not isinstance(input_files, list):
            input_files = [input_files]
        
        print("\n" + "=" * 70)
        print("CREATING CHESS DATASET (VERBOSE MODE)")
        print("=" * 70)
        print(f"Target: {target_positions:,} positions")
        print(f"Filters: ELO≥{self.min_elo}, Time≥{self.time_control_min}s")
        print("=" * 70 + "\n")

        # Pre-allocate numpy arrays for memory efficiency
        positions = np.zeros((target_positions, 12, 8, 8), dtype=np.float16)
        moves = np.zeros(target_positions, dtype=np.int32)
        results = np.zeros(target_positions, dtype=np.float16)
        current_idx = 0
        
        self.stats['start_time'] = time.time()
        last_print = 0
        
        for input_file in input_files:
            input_path = Path(input_file)
            
            if not input_path.exists():
                print(f"⚠️ File not found: {input_path}")
                continue
            
            game_stream = self.stream_from_compressed(input_path)
            
            for game in game_stream:
                self.stats['games_processed'] += 1
                
                # Print progress every 1000 games
                if self.stats['games_processed'] % 1000 == 0:
                    self.print_progress(current_idx, target_positions)
                
                if not self.passes_filters(game):
                    continue
                
                self.stats['games_used'] += 1
                
                board = game.board()
                game_result = self.get_result_value(game)
                move_count = 0
                
                for move in game.mainline_moves():
                    if move_count >= 5:
                        positions[current_idx] = self.board_to_tensor(board)
                        moves[current_idx] = self.move_to_index(move)

                        if board.turn == chess.BLACK:
                            results[current_idx] = -game_result
                        else:
                            results[current_idx] = game_result

                        current_idx += 1

                        if current_idx >= target_positions:
                            break

                    board.push(move)
                    move_count += 1

                if current_idx >= target_positions:
                    break

            if current_idx >= target_positions:
                break
        
        # Final summary
        print("\n\n" + "=" * 70)
        print("DATASET CREATION COMPLETE")
        print("=" * 70)
        
        elapsed = time.time() - self.stats['start_time']
        
        print(f"\nFinal Statistics:")
        print(f"  Games processed: {self.stats['games_processed']:,}")
        print(f"  Games used: {self.stats['games_used']:,}")
        print(f"  Pass rate: {self.stats['games_used']/self.stats['games_processed']*100:.1f}%")
        print(f"  Positions: {current_idx:,}")
        print(f"  Total time: {elapsed/3600:.1f} hours")

        # Arrays are already in numpy format, just calculate size
        size_mb = (positions.nbytes + moves.nbytes + results.nbytes) / (1024**2)
        print(f"\nDataset size: {size_mb:.1f} MB")
        
        print(f"\nSaving to {output_file}...")
        np.savez_compressed(
            output_file,
            positions=positions,
            moves=moves,
            results=results,
            metadata=np.array([{
                'min_elo': self.min_elo,
                'time_control_min': self.time_control_min,
                'total_positions': current_idx,
                'games_processed': self.stats['games_processed'],
                'games_used': self.stats['games_used'],
                'creation_time_hours': elapsed/3600,
                'creation_date': datetime.now().isoformat()
            }])
        )
        
        print(f"✅ Dataset saved successfully!")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    input_files = [
        "lichess_data/lichess_db_standard_rated_2025-09.pgn.zst"
    ]
    
    creator = VerboseChessDatasetCreator(min_elo=1900, time_control_min=600)
    creator.create_dataset(
        input_files=input_files,
        output_file="../outputs/chess_elo1900_500K.npz",
        target_positions=500_000
    )