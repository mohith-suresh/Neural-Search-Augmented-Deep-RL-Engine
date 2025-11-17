"""
Process Lichess games into training dataset for chess AI.

Streams compressed PGN files and creates optimized .npz dataset.
Memory-efficient with real-time progress bar and ETA.
"""

import chess.pgn
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys


class ChessDatasetCreator:
    """Create chess training dataset from PGN files."""
    
    def __init__(self, min_elo=1750, time_control_min=900):
        """
        Initialize dataset creator.
        
        Args:
            min_elo: Minimum player Elo rating
            time_control_min: Minimum time control in seconds (900 = 15 min)
        """
        self.min_elo = min_elo
        self.time_control_min = time_control_min
        
        # Piece to channel mapping for board encoding
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
            'positions': 0,
            'start_time': None,
            'last_update': None,
        }
    
    def board_to_tensor(self, board):
        """Convert board position to 12x8x8 tensor."""
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
        """Convert move to index (0-8191)."""
        from_sq = move.from_square
        to_sq = move.to_square
        if move.promotion:
            return from_sq * 64 + to_sq + 4096
        return from_sq * 64 + to_sq
    
    def passes_filters(self, game):
        """Check if game meets quality filters."""
        try:
            # Check Elo rating
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            if white_elo < self.min_elo or black_elo < self.min_elo:
                return False
            
            # Check time control
            time_control = game.headers.get("TimeControl", "")
            if not time_control or time_control == "-":
                return False
            
            try:
                base_time = int(time_control.split('+')[0])
            except (ValueError, IndexError):
                return False
            
            if base_time < self.time_control_min:
                return False
            
            # Check game length
            ply_count = game.headers.get("PlyCount", None)
            if ply_count:
                try:
                    move_count = int(ply_count) // 2
                except ValueError:
                    return False
            else:
                move_count = sum(1 for _ in game.mainline_moves()) // 2
            
            if move_count < 15 or move_count > 200:
                return False
            
            # Check termination
            termination = game.headers.get("Termination", "")
            if termination not in ["Normal", "Time forfeit"]:
                return False
            
            return True
            
        except (ValueError, TypeError, KeyError):
            return False
    
    def get_result_value(self, game):
        """Convert result to numeric value."""
        result = game.headers.get("Result", "*")
        return 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
    
    def stream_from_compressed(self, zst_file):
        """Stream games from compressed .zst file."""
        try:
            process = subprocess.Popen(
                ['zstdcat', str(zst_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024*1024,
                universal_newlines=True
            )
        except FileNotFoundError:
            print("\nError: zstdcat not found. Install with: sudo apt install zstd")
            return None
        
        try:
            while True:
                try:
                    game = chess.pgn.read_game(process.stdout)
                    if game is None:
                        break
                    yield game
                except Exception:
                    continue
        finally:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                pass
    
    def print_progress_bar(self, current, target):
        """Print real-time progress bar with ETA."""
        if current == 0:
            return
        
        elapsed = time.time() - self.stats['start_time']
        percent = min(100, (current / target) * 100)
        
        # Calculate speed and ETA
        speed = current / elapsed
        remaining_secs = (target - current) / speed if speed > 0 else 0
        eta = datetime.now() + timedelta(seconds=remaining_secs)
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Format output
        status = f"\r[{bar}] {percent:5.1f}% | {current:>8,}/{target:>8,} | "
        status += f"{speed:>6.0f} pos/s | ETA {eta.strftime('%H:%M:%S')}"
        
        print(status, end='', flush=True)
    
    def create_dataset(self, input_files, output_file, target_positions=1_000_000):
        """
        Create dataset from PGN files.
        
        Args:
            input_files: List of .pgn.zst files or single file
            output_file: Output .npz file path
            target_positions: Target number of positions
        """
        if not isinstance(input_files, list):
            input_files = [input_files]
        
        # Print header
        print("\n" + "="*80)
        print("CHESS DATASET CREATION")
        print("="*80)
        print(f"Target positions:  {target_positions:,}")
        print(f"Minimum Elo:       {self.min_elo}")
        print(f"Min time control:  {self.time_control_min}s")
        print(f"Input files:       {len(input_files)}")
        print("="*80)
        
        # Pre-allocate arrays
        positions = np.zeros((target_positions, 12, 8, 8), dtype=np.float16)
        moves = np.zeros(target_positions, dtype=np.int32)
        results = np.zeros(target_positions, dtype=np.float16)
        
        current_idx = 0
        self.stats['start_time'] = time.time()
        self.stats['last_update'] = time.time()
        
        # Process each file
        for file_num, input_file in enumerate(input_files, 1):
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"\nWarning: File not found {input_path}")
                continue
            
            print(f"\n\nFile {file_num}/{len(input_files)}: {input_path.name}")
            print(f"Size: {input_path.stat().st_size / (1024**3):.1f}GB")
            
            game_stream = self.stream_from_compressed(input_path)
            if game_stream is None:
                continue
            
            # Process games from this file
            for game in game_stream:
                self.stats['games_processed'] += 1
                
                # Update progress every 2 seconds
                current_time = time.time()
                if current_time - self.stats['last_update'] >= 2:
                    self.print_progress_bar(current_idx, target_positions)
                    self.stats['last_update'] = current_time
                
                # Check if we're done
                if current_idx >= target_positions:
                    break
                
                # Skip games that don't meet filters
                if not self.passes_filters(game):
                    continue
                
                self.stats['games_used'] += 1
                
                # Extract positions from game
                try:
                    board = game.board()
                    game_result = self.get_result_value(game)
                    move_count = 0
                    
                    for move in game.mainline_moves():
                        # Skip opening moves (first 5 moves = 10 plies)
                        if move_count >= 5:
                            if current_idx < target_positions:
                                positions[current_idx] = self.board_to_tensor(board)
                                moves[current_idx] = self.move_to_index(move)
                                
                                # Result is from perspective of player to move
                                if board.turn == chess.BLACK:
                                    results[current_idx] = -game_result
                                else:
                                    results[current_idx] = game_result
                                
                                current_idx += 1
                            else:
                                break
                        
                        board.push(move)
                        move_count += 1
                    
                except Exception:
                    continue
                
                if current_idx >= target_positions:
                    break
            
            if current_idx >= target_positions:
                break
        
        # Final progress bar
        print()
        self.print_progress_bar(current_idx, target_positions)
        
        # Print summary
        elapsed = time.time() - self.stats['start_time']
        pass_rate = (self.stats['games_used'] / max(1, self.stats['games_processed'])) * 100
        
        print(f"\n\n" + "="*80)
        print("DATASET CREATION COMPLETE")
        print("="*80)
        print(f"Games processed:  {self.stats['games_processed']:>10,}")
        print(f"Games used:       {self.stats['games_used']:>10,} ({pass_rate:.1f}% pass rate)")
        print(f"Positions:        {current_idx:>10,}")
        print(f"Time:             {elapsed/3600:>10.1f} hours")
        print(f"Speed:            {current_idx/max(1, elapsed):>10.0f} pos/sec")
        
        # Save dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Trim arrays to actual size
        positions = positions[:current_idx]
        moves = moves[:current_idx]
        results = results[:current_idx]
        
        print(f"\nSaving dataset...")
        np.savez_compressed(
            output_file,
            positions=positions,
            moves=moves,
            results=results
        )
        
        file_size_mb = Path(output_file).stat().st_size / (1024**2)
        print(f"Output file:      {output_file}")
        print(f"File size:        {file_size_mb:.1f} MB")
        print("="*80 + "\n")
        
        return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create chess dataset from Lichess PGN files')
    parser.add_argument('--input-dir', type=str, default='data/lichess_raw',
                        help='Input directory with .pgn.zst files')
    parser.add_argument('--output-file', type=str, default='data/training_data/classical.npz',
                        help='Output .npz file')
    parser.add_argument('--min-elo', type=int, default=1750,
                        help='Minimum player Elo rating')
    parser.add_argument('--positions', type=int, default=1000000,
                        help='Target number of positions')
    
    args = parser.parse_args()
    
    # Find input files
    input_dir = Path(args.input_dir)
    pgn_files = sorted(input_dir.glob('*.pgn.zst'))
    
    if not pgn_files:
        print(f"Error: No .pgn.zst files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(pgn_files)} files:")
    for f in pgn_files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"  {f.name:50} ({size_gb:>6.1f}GB)")
    
    # Create dataset
    creator = ChessDatasetCreator(min_elo=args.min_elo)
    result = creator.create_dataset(
        input_files=pgn_files,
        output_file=args.output_file,
        target_positions=args.positions
    )
