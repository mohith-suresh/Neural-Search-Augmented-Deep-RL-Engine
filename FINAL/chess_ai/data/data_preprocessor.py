#!/usr/bin/env python3
"""
Chess Data Preprocessor - "13th Plane" & Memmap Optimized
=========================================================

Project: EE542 - AlphaZero Hybrid (Laptop/GCP Optimized)
Target: 20M positions (Stratified)
Output: Uncompressed Binary Memmaps (Zero-Copy Load)

CHANGELOG:
1. Added 13th input channel (Turn Indicator) to fix Value Head collapse.
2. Switched from .npz (compressed) to .bin (memmap) for 50x faster loading.
3. Optimized intermediate chunking to prevent RAM spikes.

"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import psutil

try:
    import chess
    import chess.pgn
except ImportError:
    print("ERROR: chess library required. Install: pip install python-chess")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

class ProductionConfig:
    """Production extraction configuration."""
    
    # Stratified Elo buckets
    ELO_BUCKETS = [
        (800, 1000), (1000, 1200), (1200, 1400), (1400, 1600),
        (1600, 1800), (1800, 2000), (2000, 2200), (2200, 2600),
    ]
    
    # Target: 20M positions
    TARGET_POSITIONS_TOTAL = 500_000
    TARGET_POSITIONS_PER_BUCKET = TARGET_POSITIONS_TOTAL // len(ELO_BUCKETS)
    
    # Filters
    time_control_min = 300
    game_length_min = 20
    game_length_max = 500
    
    # I/O
    chunk_size = 500_000 
    validate_moves = True
    
    # Paths
    OUTPUT_DIR = Path('/home/krish/EE542-Project/FINAL/chess_ai/data/training_data')
    LOG_DIR = Path('/home/krish/EE542-Project/FINAL/chess_ai/data/processing_logs')
    
    def get_bucket_name(self, elo_min: int, elo_max: int) -> str:
        names = {
            (800, 1000): "Beginner", (1000, 1200): "Novice",
            (1200, 1400): "Intermediate", (1400, 1600): "Advanced",
            (1600, 1800): "Strong", (1800, 2000): "Master",
            (2000, 2200): "SuperGM", (2200, 2600): "Engine",
        }
        return names.get((elo_min, elo_max), f"Unknown({elo_min}-{elo_max})")

class SystemMonitor:
    def __init__(self):
        self.start_memory_mb = psutil.virtual_memory().used / (1024 ** 2)
        self.peak_memory_mb = self.start_memory_mb
        self.start_time = None
    
    def current_memory_mb(self) -> float:
        return psutil.virtual_memory().used / (1024 ** 2)
    
    def update_peak(self):
        current = self.current_memory_mb()
        if current > self.peak_memory_mb:
            self.peak_memory_mb = current

# ============================================================================
# Core Logic: 13-Plane Encoding
# ============================================================================

class BoardEncoder:
    """Encode board to 13x8x8 tensor (Includes Turn Indicator)."""
    
    PIECE_TO_CHANNEL = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }
    
    @staticmethod
    def board_to_tensor(board: chess.Board) -> np.ndarray:
        # CRITICAL FIX: 13 channels instead of 12
        # Channel 12 is the "Turn Indicator" (All 0s for White, All 1s for Black)
        tensor = np.zeros((13, 8, 8), dtype=np.float16)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = BoardEncoder.PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
                tensor[channel, rank, file] = 1.0
        
        # 13th Plane: Turn Indicator
        if board.turn == chess.BLACK:
            tensor[12, :, :] = 1.0
            
        return tensor

class MoveEncoder:
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        from_sq = move.from_square
        to_sq = move.to_square
        if move.promotion:
            return from_sq * 64 + to_sq + 4096
        return from_sq * 64 + to_sq

# ============================================================================
# Pipeline Components
# ============================================================================

class StratifiedGameFilter:
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.stats = {
            'total_games_processed': 0,
            'bucket_stats': {}
        }
        for elo_min, elo_max in config.ELO_BUCKETS:
            name = config.get_bucket_name(elo_min, elo_max)
            self.stats['bucket_stats'][name] = {
                'elo_range': (elo_min, elo_max),
                'positions_created': 0,
                'target': config.TARGET_POSITIONS_PER_BUCKET,
            }
    
    def get_elo_bucket(self, white_elo: int, black_elo: int) -> Optional[str]:
        avg = (white_elo + black_elo) // 2
        for elo_min, elo_max in self.config.ELO_BUCKETS:
            if elo_min <= avg < elo_max:
                return self.config.get_bucket_name(elo_min, elo_max)
        return None
    
    def passes_filters(self, game: chess.pgn.GameNode) -> Tuple[bool, Optional[str]]:
        try:
            self.stats['total_games_processed'] += 1
            headers = game.headers
            
            # Elo Check
            try:
                w_elo = int(headers.get("WhiteElo", 0))
                b_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                return False, None
            
            bucket = self.get_elo_bucket(w_elo, b_elo)
            if not bucket: return False, None
            
            # Bucket Full Check
            if self.stats['bucket_stats'][bucket]['positions_created'] >= \
               self.stats['bucket_stats'][bucket]['target']:
                return False, None

            # Time Control Check (Standard/Rapid only)
            tc = headers.get("TimeControl", "")
            if not tc or tc == "-": return False, None
            try:
                base = int(tc.split('+')[0])
                if base < self.config.time_control_min: return False, None
            except: return False, None
            
            return True, bucket
            
        except Exception:
            return False, None

    def get_result_value(self, game: chess.pgn.GameNode) -> float:
        res = game.headers.get("Result", "*")
        if res == "1-0": return 1.0
        if res == "0-1": return -1.0
        return 0.0

class PGNStreamReader:
    def __init__(self, logger):
        self.logger = logger
    
    def stream_from_zst(self, zst_file: Path):
        try:
            process = subprocess.Popen(
                ['zstdcat', str(zst_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024 * 1024, # 1MB buffer
                universal_newlines=True,
                text=True
            )
            while True:
                game = chess.pgn.read_game(process.stdout)
                if game is None: break
                yield game
        except FileNotFoundError:
            self.logger.error("zstdcat not found. Install: sudo apt install zstd")
        except Exception as e:
            self.logger.error(f"Stream error: {e}")

# ============================================================================
# Main Extractor (Memmap Optimized)
# ============================================================================

class ProductionDataExtractor:
    def __init__(self, config: ProductionConfig, logger, monitor: SystemMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.encoder = BoardEncoder()
        self.filter = StratifiedGameFilter(config)
        self.reader = PGNStreamReader(logger)
        
        self.stats = {
            'positions': 0,
            'start_time': None
        }

    def extract_dataset(self, input_files: List[Path]):
        self.stats['start_time'] = time.time()
        self.monitor.start_time = self.stats['start_time']
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Temp chunk storage
        chunk_files = []
        current_chunk_pos = []
        current_chunk_mov = []
        current_chunk_res = []
        chunk_idx = 0
        
        self.logger.info("Starting Extraction...")
        
        for input_file in input_files:
            if self.stats['positions'] >= self.config.TARGET_POSITIONS_TOTAL: break
            
            self.logger.info(f"Reading {input_file.name}...")
            
            for game in self.reader.stream_from_zst(input_file):
                if self.stats['positions'] >= self.config.TARGET_POSITIONS_TOTAL: break
                
                passed, bucket_name = self.filter.passes_filters(game)
                if not passed: continue
                
                # Extract
                board = game.board()
                game_res = self.filter.get_result_value(game)
                
                for move in game.mainline_moves():
                    # Validation
                    if self.config.validate_moves and move not in board.legal_moves: break
                    
                    # Encode
                    pos_tensor = self.encoder.board_to_tensor(board)
                    move_idx = MoveEncoder.move_to_index(move)
                    
                    # Result perspective
                    result = game_res if board.turn == chess.WHITE else -game_res
                    
                    current_chunk_pos.append(pos_tensor)
                    current_chunk_mov.append(move_idx)
                    current_chunk_res.append(result)
                    
                    # Stats
                    self.stats['positions'] += 1
                    self.filter.stats['bucket_stats'][bucket_name]['positions_created'] += 1
                    
                    # Chunk write
                    if len(current_chunk_pos) >= self.config.chunk_size:
                        self._dump_chunk(chunk_idx, current_chunk_pos, current_chunk_mov, current_chunk_res, chunk_files)
                        current_chunk_pos, current_chunk_mov, current_chunk_res = [], [], []
                        chunk_idx += 1
                        self._print_progress()
                    
                    board.push(move)
        
        # Final residual chunk
        if current_chunk_pos:
            self._dump_chunk(chunk_idx, current_chunk_pos, current_chunk_mov, current_chunk_res, chunk_files)
            chunk_files.append(self.config.OUTPUT_DIR / f'temp_{chunk_idx}.npz')

        # CONSOLIDATE TO MEMMAP
        self._consolidate_to_memmap(chunk_files)

    def _dump_chunk(self, idx, pos, mov, res, file_list):
        """Save temp chunk as uncompressed NPZ for speed."""
        path = self.config.OUTPUT_DIR / f'temp_{idx}.npz'
        # Uncompressed save is faster
        np.savez(path, 
                 p=np.array(pos, dtype=np.float16),
                 m=np.array(mov, dtype=np.int32),
                 r=np.array(res, dtype=np.float16))
        file_list.append(path)

    def _consolidate_to_memmap(self, chunk_files):
        """Merge all temp chunks into final binary memmaps."""
        self.logger.info("\n" + "="*60)
        self.logger.info("CONSOLIDATING TO MEMMAP (Zero-Copy Format)")
        self.logger.info("="*60)
        
        total = self.stats['positions']
        
        # 1. Create Metadata
        metadata = {
            'count': total,
            'pos_shape': [total, 13, 8, 8],
            'pos_dtype': 'float16',
            'mov_shape': [total],
            'mov_dtype': 'int32',
            'res_shape': [total],
            'res_dtype': 'float16'
        }
        with open(self.config.OUTPUT_DIR / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # 2. Allocate Memmaps
        self.logger.info(f"Allocating disk space for {total:,} positions...")
        
        fp_pos = np.memmap(self.config.OUTPUT_DIR / 'positions.bin', dtype='float16', mode='w+', shape=(total, 13, 8, 8))
        fp_mov = np.memmap(self.config.OUTPUT_DIR / 'moves.bin', dtype='int32', mode='w+', shape=(total,))
        fp_res = np.memmap(self.config.OUTPUT_DIR / 'results.bin', dtype='float16', mode='w+', shape=(total,))
        
        # 3. Stream Copy
        ptr = 0
        for i, cf in enumerate(chunk_files):
            try:
                data = np.load(cf)
                n = len(data['p'])
                
                fp_pos[ptr:ptr+n] = data['p']
                fp_mov[ptr:ptr+n] = data['m']
                fp_res[ptr:ptr+n] = data['r']
                
                ptr += n
                
                if i % 5 == 0:
                    self.logger.info(f"Merged chunk {i+1}/{len(chunk_files)}...")
                
                # Cleanup temp file immediately
                cf.unlink()
                
            except Exception as e:
                self.logger.error(f"Error merging chunk {cf}: {e}")
        
        # Flush to disk
        fp_pos.flush()
        fp_mov.flush()
        fp_res.flush()
        
        self.logger.info(f"\n✓ SUCCESS. Dataset ready at: {self.config.OUTPUT_DIR}")
        self.logger.info(f"  Total Positions: {total:,}")
        self.logger.info(f"  Format: Raw Binary (Memmap)")

    def _print_progress(self):
        elapsed = time.time() - self.stats['start_time']
        rate = self.stats['positions'] / max(1, elapsed)
        print(f"\rExtracted: {self.stats['positions']:,} | Speed: {rate:.0f} pos/s", end="")

def setup_logging(config):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(config.LOG_DIR / "extract.log"), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/home/krish/EE542-Project/FINAL/chess_ai/data/lichess_raw')
    args = parser.parse_args()
    
    config = ProductionConfig()
    logger = setup_logging(config)
    monitor = SystemMonitor()
    
    input_files = sorted(Path(args.input_dir).glob('*.pgn.zst'))
    if not input_files:
        print("No .pgn.zst files found!")
        sys.exit(1)
        
    extractor = ProductionDataExtractor(config, logger, monitor)
    extractor.extract_dataset(input_files)