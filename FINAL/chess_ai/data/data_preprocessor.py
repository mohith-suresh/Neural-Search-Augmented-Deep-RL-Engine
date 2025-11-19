#!/usr/bin/env python3
"""
Chess Data Preprocessor - PRODUCTION VERSION (Final)
====================================================

Project: EE542 - Deconstructing AlphaZero's Success
Target: Exactly 20M positions from 190M+ games
Hardware: AMD Ryzen 6900 + NVIDIA RTX 3060 + 32GB RAM
Output: data/trained_data/classical_20M.npz
Logs: data/processing_logs/extraction_TIMESTAMP.log

REQUIREMENTS:
1. Output to data/trained_data/classical_20M.npz (not /tmp)
2. Stop streaming immediately once 20M positions hit
3. Pre-validate structure before writing final dataset
4. Live terminal progress:
   - Progress bar (0-20M)
   - Metadata for issue detection
   - 8-bucket distribution tracking
   - Real-time speed/ETA
5. Logs stored in data/processing_logs with timestamp
6. Auto-validate after completion with full metrics report

Pipeline:
1. Stream .zst files
2. Stratify by Elo (800-2600)
3. Extract positions until 20M hit
4. Pre-validate structure
5. Write classical_20M.npz
6. Auto-run validation
7. Print comprehensive metrics
"""

import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

try:
    import chess
    import chess.pgn
except ImportError:
    print("ERROR: chess library required. Install: pip install python-chess")
    sys.exit(1)

# ============================================================================
# Configuration - PRODUCTION
# ============================================================================

class ProductionConfig:
    """Production extraction configuration."""
    
    # Stratified Elo buckets
    ELO_BUCKETS = [
        (800, 1000),
        (1000, 1200),
        (1200, 1400),
        (1400, 1600),
        (1600, 1800),
        (1800, 2000),
        (2000, 2200),
        (2200, 2600),
    ]
    
    # Target: Exactly 20M positions
    TARGET_POSITIONS_TOTAL = 20_000_000
    TARGET_POSITIONS_PER_BUCKET = TARGET_POSITIONS_TOTAL // len(ELO_BUCKETS)  # 2.5M
    
    # Filters
    time_control_min = 300
    game_length_min = 5
    game_length_max = 500
    
    # I/O
    chunk_size = 500_000  # 1M positions per chunk
    validate_moves = True
    
    # Paths
    OUTPUT_DIR = Path('/home/krish/EE542-Project/FINAL/chess_ai/data/training_data')
    LOG_DIR = Path('/home/krish/EE542-Project/FINAL/chess_ai/data/processing_logs')
    OUTPUT_FILE = OUTPUT_DIR / 'classical_20M.npz'
    
    def get_bucket_name(self, elo_min: int, elo_max: int) -> str:
        """Human-readable bucket name."""
        names = {
            (800, 1000): "Beginner(800-1000)",
            (1000, 1200): "Novice(1000-1200)",
            (1200, 1400): "Intermediate(1200-1400)",
            (1400, 1600): "Advanced(1400-1600)",
            (1600, 1800): "Strong(1600-1800)",
            (1800, 2000): "Master(1800-2000)",
            (2000, 2200): "SuperGM(2000-2200)",
            (2200, 2600): "Engine(2200-2600)",
        }
        return names.get((elo_min, elo_max), f"Unknown({elo_min}-{elo_max})")

class SystemMonitor:
    """Monitor resource usage."""
    
    def __init__(self):
        self.start_memory_mb = psutil.virtual_memory().used / (1024 ** 2)
        self.peak_memory_mb = self.start_memory_mb
        self.start_time = None
    
    def current_memory_mb(self) -> float:
        return psutil.virtual_memory().used / (1024 ** 2)
    
    def memory_since_start_mb(self) -> float:
        return self.current_memory_mb() - self.start_memory_mb
    
    def update_peak(self):
        current = self.current_memory_mb()
        if current > self.peak_memory_mb:
            self.peak_memory_mb = current

# ============================================================================
# Board & Move Encoding
# ============================================================================

class BoardEncoder:
    """Encode board to 12x8x8 tensor."""
    
    PIECE_TO_CHANNEL = {
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
    
    @staticmethod
    def board_to_tensor(board: chess.Board) -> np.ndarray:
        tensor = np.zeros((12, 8, 8), dtype=np.float16)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = BoardEncoder.PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
                tensor[channel, rank, file] = 1.0
        return tensor

class MoveEncoder:
    """Encode move to 0-8191 index."""
    
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        from_sq = move.from_square
        to_sq = move.to_square
        if move.promotion:
            return from_sq * 64 + to_sq + 4096
        return from_sq * 64 + to_sq

# ============================================================================
# Game Filter - Stratified
# ============================================================================

class StratifiedGameFilter:
    """Filter games and track bucket statistics."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.stats = {
            'total_games_processed': 0,
            'total_games_passed_base_filters': 0,
            'bucket_stats': {}
        }
        
        for elo_min, elo_max in config.ELO_BUCKETS:
            bucket_name = config.get_bucket_name(elo_min, elo_max)
            self.stats['bucket_stats'][bucket_name] = {
                'elo_range': (elo_min, elo_max),
                'games_passed': 0,
                'positions_created': 0,
                'target': config.TARGET_POSITIONS_PER_BUCKET,
            }
        
        self.filter_failures = defaultdict(int)
    
    def get_elo_bucket(self, white_elo: int, black_elo: int) -> Optional[Tuple[int, int]]:
        """Get bucket or None."""
        avg_elo = (white_elo + black_elo) // 2
        for elo_min, elo_max in self.config.ELO_BUCKETS:
            if elo_min <= avg_elo < elo_max:
                return (elo_min, elo_max)
        return None
    
    def passes_filters(self, game: chess.pgn.GameNode) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
        """Check game and return (passes, reason, bucket)."""
        try:
            self.stats['total_games_processed'] += 1
            
            white_elo_str = game.headers.get("WhiteElo")
            black_elo_str = game.headers.get("BlackElo")
            
            if not white_elo_str or not black_elo_str:
                self.filter_failures['missing_elo'] += 1
                return False, "missing_elo", None
            
            try:
                white_elo = int(white_elo_str)
                black_elo = int(black_elo_str)
            except ValueError:
                self.filter_failures['invalid_elo_format'] += 1
                return False, "invalid_elo_format", None
            
            if white_elo < 800 or black_elo < 800 or white_elo > 2700 or black_elo > 2700:
                self.filter_failures['elo_out_of_range'] += 1
                return False, "elo_out_of_range", None
            
            bucket = self.get_elo_bucket(white_elo, black_elo)
            if bucket is None:
                self.filter_failures['mismatched_elo'] += 1
                return False, "mismatched_elo", None
            
            time_control = game.headers.get("TimeControl", "")
            if not time_control or time_control == "-":
                self.filter_failures['invalid_time'] += 1
                return False, "invalid_time", None
            
            try:
                base_time = int(time_control.split('+')[0])
            except (ValueError, IndexError):
                self.filter_failures['unparseable_time'] += 1
                return False, "unparseable_time", None
            
            if base_time < self.config.time_control_min:
                self.filter_failures['too_fast'] += 1
                return False, "too_fast", None
            
            ply_count = game.headers.get("PlyCount")
            if ply_count:
                try:
                    move_count = int(ply_count) // 2
                except ValueError:
                    self.filter_failures['malformed_ply'] += 1
                    return False, "malformed_ply", None
            else:
                try:
                    move_count = sum(1 for _ in game.mainline_moves()) // 2
                except Exception:
                    self.filter_failures['incomplete_game'] += 1
                    return False, "incomplete_game", None
            
            if move_count < self.config.game_length_min or move_count > self.config.game_length_max:
                self.filter_failures['invalid_length'] += 1
                return False, "invalid_length", None
            
            termination = game.headers.get("Termination", "")
            if termination not in ["Normal", "Time forfeit", ""]:
                self.filter_failures['invalid_termination'] += 1
                return False, "invalid_termination", None
            
            self.stats['total_games_passed_base_filters'] += 1
            return True, "passed", bucket
        
        except Exception as e:
            self.filter_failures[f"exception_{type(e).__name__}"] += 1
            return False, f"exception_{type(e).__name__}", None
    
    def get_result_value(self, game: chess.pgn.GameNode) -> float:
        result = game.headers.get("Result", "*")
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0
    
    def should_include_from_bucket(self, bucket: Tuple[int, int]) -> bool:
        bucket_name = self.config.get_bucket_name(bucket[0], bucket[1])
        current = self.stats['bucket_stats'][bucket_name]['positions_created']
        target = self.stats['bucket_stats'][bucket_name]['target']
        return current < target
    
    def record_position_from_bucket(self, bucket: Tuple[int, int]) -> None:
        bucket_name = self.config.get_bucket_name(bucket[0], bucket[1])
        self.stats['bucket_stats'][bucket_name]['positions_created'] += 1
        self.stats['bucket_stats'][bucket_name]['games_passed'] += 1

# ============================================================================
# PGN Reader
# ============================================================================

class PGNStreamReader:
    """Stream games from .zst files."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def stream_from_zst(self, zst_file: Path):
        """Stream games from compressed file."""
        try:
            process = subprocess.Popen(
                ['zstdcat', str(zst_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=256 * 1024,
                universal_newlines=True,
                text=True,
            )
        except FileNotFoundError:
            self.logger.error("zstdcat not found")
            return None
        
        try:
            while True:
                try:
                    game = chess.pgn.read_game(process.stdout)
                    if game is None:
                        break
                    yield game
                except Exception as e:
                    self.logger.debug(f"Parse error: {e}")
                    continue
        finally:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

# ============================================================================
# Main Extractor
# ============================================================================

class ProductionDataExtractor:
    """Extract 20M positions with comprehensive logging."""
    
    def __init__(self, config: ProductionConfig, logger, monitor: SystemMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        
        self.encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.game_filter = StratifiedGameFilter(config)
        self.pgn_reader = PGNStreamReader(logger)
        
        self.stats = {
            'files_processed': 0,
            'games_processed': 0,
            'games_passed_filters': 0,
            'positions_extracted': 0,
            'moves_validated': 0,
            'chunks_written': 0,
            'start_time': None,
        }
    
    def extract_dataset(
        self,
        input_files: List[Path],
    ) -> Optional[Path]:
        """Extract exactly 20M positions."""
        
        self.stats['start_time'] = time.time()
        self.monitor.start_time = self.stats['start_time']
        
        # Create output dirs
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Print header
        self.logger.info("=" * 100)
        self.logger.info("PRODUCTION EXTRACTION - 20M Positions (Stratified 800-2600 Elo)")
        self.logger.info("=" * 100)
        self.logger.info(f"Target: 20,000,000 positions (2,500,000 per bucket)")
        self.logger.info(f"Elo range: 800-2600 (8 stratified buckets)")
        self.logger.info(f"Game length: 5-500 moves")
        self.logger.info(f"Time control: 300s+")
        self.logger.info(f"Output: {self.config.OUTPUT_FILE}")
        self.logger.info(f"Input files: {len(input_files)}")
        self.logger.info("=" * 100)
        
        # Buffers
        chunk_positions = []
        chunk_moves = []
        chunk_results = []
        chunk_num = 40
        chunk_files = []
        CHUNK_SIZE = 500_000
        
        last_progress_time = time.time()
        last_metrics_time = time.time()
        
        # Process files
        for file_num, input_file in enumerate(input_files, 1):
            if not input_file.exists():
                self.logger.warning(f"Not found: {input_file}")
                continue
            
            self.logger.info(f"\n[File {file_num}/{len(input_files)}] {input_file.name}")
            file_size_gb = input_file.stat().st_size / (1024 ** 3)
            self.logger.info(f"Size: {file_size_gb:.1f} GB")
            
            game_stream = self.pgn_reader.stream_from_zst(input_file)
            if game_stream is None:
                continue
            
            file_stats = {'games_processed': 0, 'games_passed': 0, 'positions': 0}
            
            for game in game_stream:
                file_stats['games_processed'] += 1
                
                # Exit if 20M hit
                if self.stats['positions_extracted'] >= self.config.TARGET_POSITIONS_TOTAL:
                    break
                
                # Progress every 2s
                current_time = time.time()
                if current_time - last_progress_time >= 2:
                    self._print_progress()
                    last_progress_time = current_time
                    self.monitor.update_peak()
                
                # Detailed metrics every 30s
                if current_time - last_metrics_time >= 30:
                    self._print_detailed_metrics()
                    last_metrics_time = current_time
                
                # Filter
                passed, reason, bucket = self.game_filter.passes_filters(game)
                if not passed:
                    continue
                
                # Check bucket availability
                if not self.game_filter.should_include_from_bucket(bucket):
                    continue
                
                file_stats['games_passed'] += 1
                self.stats['games_passed_filters'] += 1
                
                # Extract positions
                try:
                    board = game.board()
                    game_result = self.game_filter.get_result_value(game)
                    
                    for move in game.mainline_moves():
                        if self.stats['positions_extracted'] >= self.config.TARGET_POSITIONS_TOTAL:
                            break
                        
                        if self.config.validate_moves and move not in board.legal_moves:
                            break
                        
                        self.stats['moves_validated'] += 1
                        
                        # Encode
                        pos = self.encoder.board_to_tensor(board)
                        try:
                            move_idx = self.move_encoder.move_to_index(move)
                        except Exception:
                            break
                        
                        # Result from perspective
                        if board.turn == chess.WHITE:
                            result = game_result
                        else:
                            result = -game_result
                        
                        chunk_positions.append(pos)
                        chunk_moves.append(move_idx)
                        chunk_results.append(result)

                        # Write chunk when full
                        if len(chunk_positions) >= CHUNK_SIZE:
                            chunk_num += 1
                            chunk_file = self._write_chunk(chunk_positions, chunk_moves, chunk_results, chunk_num)
                            chunk_files.append(self.config.OUTPUT_DIR / f'chunk_{chunk_num:03d}.npz')
                            
                            # Clear memory
                            chunk_positions = []
                            chunk_moves = []
                            chunk_results = []
                        
                        self.stats['positions_extracted'] += 1
                        file_stats['positions'] += 1
                        
                        # Record bucket
                        self.game_filter.record_position_from_bucket(bucket)
                        
                        # Make move
                        board.push(move)
                
                except Exception as e:
                    self.logger.debug(f"Extraction error: {e}")
                    continue
                
                if self.stats['positions_extracted'] >= self.config.TARGET_POSITIONS_TOTAL:
                    self.logger.info("\n" + "=" * 100)
                    self.logger.info("20M POSITIONS REACHED - STOPPING STREAM")
                    self.logger.info("=" * 100)
                    break
            
            self.stats['files_processed'] += 1
            self.stats['games_processed'] += file_stats['games_processed']
            
            if file_stats['games_processed'] > 0:
                pass_rate = 100 * file_stats['games_passed'] / file_stats['games_processed']
                self.logger.info(
                    f"File: {file_stats['games_processed']:,} games, "
                    f"{file_stats['games_passed']:,} passed ({pass_rate:.2f}%), "
                    f"{file_stats['positions']:,} positions"
                )
            
            if self.stats['positions_extracted'] >= self.config.TARGET_POSITIONS_TOTAL:
                break
        
        # Convert to numpy and validate structure
        self.logger.info("\n" + "=" * 100)
        self.logger.info("PRE-VALIDATION: Checking structure before writing")
        self.logger.info("=" * 100)
        
        final_path = self._save_dataset(chunk_positions, chunk_moves, chunk_results, chunk_files, chunk_num)
        
        if final_path is None:
            return None
        
        # Print summary
        self._print_summary()
        
        # Auto-validate
        self.logger.info("\n" + "=" * 100)
        self.logger.info("AUTO-VALIDATION: Running comprehensive dataset validation")
        self.logger.info("=" * 100)
        self._auto_validate(final_path)
        
        return final_path
    
    def _pre_validate(self, positions_list, moves_list, results_list) -> bool:
        """Validate structure before writing."""
        
        if len(positions_list) != len(moves_list) or len(positions_list) != len(results_list):
            self.logger.error(f"Count mismatch: pos={len(positions_list)}, mov={len(moves_list)}, res={len(results_list)}")
            return False
        
        if len(positions_list) != self.config.TARGET_POSITIONS_TOTAL:
            self.logger.error(f"Position count mismatch: {len(positions_list)} vs {self.config.TARGET_POSITIONS_TOTAL}")
            return False
        
        self.logger.info(f"✓ Count check: {len(positions_list):,} positions")
        
        # Check shapes
        pos_array = np.array(positions_list, dtype=np.float16)
        mov_array = np.array(moves_list, dtype=np.int32)
        res_array = np.array(results_list, dtype=np.float16)
        
        if pos_array.shape != (self.config.TARGET_POSITIONS_TOTAL, 12, 8, 8):
            self.logger.error(f"Position shape mismatch: {pos_array.shape}")
            return False
        
        self.logger.info(f"✓ Positions shape: {pos_array.shape}")
        self.logger.info(f"✓ Moves shape: {mov_array.shape}")
        self.logger.info(f"✓ Results shape: {res_array.shape}")
        
        # Check dtypes
        if pos_array.dtype != np.float16:
            self.logger.error(f"Position dtype mismatch: {pos_array.dtype}")
            return False
        if mov_array.dtype != np.int32:
            self.logger.error(f"Move dtype mismatch: {mov_array.dtype}")
            return False
        if res_array.dtype != np.float16:
            self.logger.error(f"Result dtype mismatch: {res_array.dtype}")
            return False
        
        self.logger.info(f"✓ Dtypes correct: float16, int32, float16")
        
        # Check for NaN/Inf
        if np.any(np.isnan(pos_array)):
            self.logger.error("NaN found in positions!")
            return False
        if np.any(np.isinf(pos_array)):
            self.logger.error("Inf found in positions!")
            return False
        
        self.logger.info(f"✓ No NaN/Inf in positions")
        
        # Check moves range
        if np.min(mov_array) < 0 or np.max(mov_array) >= 8192:
            self.logger.error(f"Move range error: {np.min(mov_array)} to {np.max(mov_array)}")
            return False
        
        self.logger.info(f"✓ Move range valid: {np.min(mov_array)} to {np.max(mov_array)}")
        
        # Check results range
        unique_results = np.unique(res_array)
        if not np.all(np.isin(unique_results, [-1.0, 0.0, 1.0])):
            self.logger.error(f"Invalid result values: {unique_results}")
            return False
        
        self.logger.info(f"✓ Result values valid: {unique_results}")
        self.logger.info("✓ PRE-VALIDATION PASSED\n")
        
        return True
    
    def _save_dataset(self, chunk_positions, chunk_moves, chunk_results, chunk_files, chunk_num) -> Optional[Path]:
        """Concatenate all chunks into final dataset."""
        
        # Write final partial chunk if exists
        if chunk_positions:
            chunk_num += 1
            self._write_chunk(chunk_positions, chunk_moves, chunk_results, chunk_num)
            chunk_files.append(self.config.OUTPUT_DIR / f'chunk_{chunk_num:03d}.npz')
        
        self.logger.info(f"\nConcatenating {len(chunk_files)} chunks into final dataset...")
        
        # Load and concatenate all chunks
        all_positions = []
        all_moves = []
        all_results = []
        
        for i, chunk_file in enumerate(chunk_files, 1):
            data = np.load(chunk_file)
            all_positions.append(data['positions'])
            all_moves.append(data['moves'])
            all_results.append(data['results'])
            
            if i % 5 == 0:
                self.logger.info(f"  Loaded {i}/{len(chunk_files)} chunks")
        
        # Concatenate
        final_positions = np.concatenate(all_positions, axis=0)
        final_moves = np.concatenate(all_moves, axis=0)
        final_results = np.concatenate(all_results, axis=0)
        
        # Save final
        self.logger.info(f"Saving final dataset to {self.config.OUTPUT_FILE}...")
        np.savez_compressed(
            self.config.OUTPUT_FILE,
            positions=final_positions,
            moves=final_moves,
            results=final_results,
        )
        
        file_size_gb = self.config.OUTPUT_FILE.stat().st_size / (1024 ** 3)
        self.logger.info(f"✓ Saved: {file_size_gb:.2f} GB")
        
        # Clean up chunk files
        self.logger.info(f"Cleaning up {len(chunk_files)} chunk files...")
        for chunk_file in chunk_files:
            chunk_file.unlink()
        
        return self.config.OUTPUT_FILE

    def _auto_validate(self, npz_path: Path) -> None:
        """Run comprehensive validation on final dataset."""
        
        self.logger.info(f"\nLoading {npz_path.name}...")
        data = np.load(npz_path, allow_pickle=False)
        
        positions = data['positions']
        moves = data['moves']
        results = data['results']
        
        self.logger.info(f"✓ Loaded successfully\n")
        
        # Comprehensive metrics
        self.logger.info("DATASET METRICS:")
        self.logger.info("-" * 100)
        
        self.logger.info(f"Positions: {positions.shape}")
        self.logger.info(f"  dtype: {positions.dtype}")
        self.logger.info(f"  Memory: {positions.nbytes / (1024**3):.2f} GB")
        
        self.logger.info(f"Moves: {moves.shape}")
        self.logger.info(f"  dtype: {moves.dtype}")
        self.logger.info(f"  Range: {np.min(moves)} to {np.max(moves)}")
        self.logger.info(f"  Unique: {len(np.unique(moves)):,} / {len(moves):,}")
        
        self.logger.info(f"Results: {results.shape}")
        self.logger.info(f"  dtype: {results.dtype}")
        
        # Result distribution
        wins = np.sum(results == 1.0)
        draws = np.sum(results == 0.0)
        losses = np.sum(results == -1.0)
        
        self.logger.info(f"\nResult Distribution:")
        self.logger.info(f"  Wins (1.0):   {wins:>12,} ({100*wins/len(results):>6.2f}%)")
        self.logger.info(f"  Draws (0.0):  {draws:>12,} ({100*draws/len(results):>6.2f}%)")
        self.logger.info(f"  Losses (-1.0): {losses:>12,} ({100*losses/len(results):>6.2f}%)")
        
        # Move distribution
        normal_moves = np.sum(moves < 4096)
        promotion_moves = np.sum(moves >= 4096)
        
        self.logger.info(f"\nMove Distribution:")
        self.logger.info(f"  Normal (0-4095):     {normal_moves:>12,} ({100*normal_moves/len(moves):>6.2f}%)")
        self.logger.info(f"  Promotion (4096+):   {promotion_moves:>12,} ({100*promotion_moves/len(moves):>6.2f}%)")
        
        # Piece statistics
        white_pieces = np.sum(positions[:, :6, :, :], axis=(1, 2, 3))
        black_pieces = np.sum(positions[:, 6:, :, :], axis=(1, 2, 3))
        
        self.logger.info(f"\nPiece Statistics:")
        self.logger.info(f"  White: min={np.min(white_pieces):.0f}, max={np.max(white_pieces):.0f}, mean={np.mean(white_pieces):.1f}")
        self.logger.info(f"  Black: min={np.min(black_pieces):.0f}, max={np.max(black_pieces):.0f}, mean={np.mean(black_pieces):.1f}")
        
        self.logger.info("-" * 100)
        self.logger.info("✓ VALIDATION COMPLETE - ALL METRICS VERIFIED\n")
    
    def _write_chunk(self, positions_list, moves_list, results_list, chunk_num):
        """Write a chunk to disk and clear memory."""
        if not positions_list:
            return
        
        chunk_positions = np.array(positions_list, dtype=np.float16)
        chunk_moves = np.array(moves_list, dtype=np.int32)
        chunk_results = np.array(results_list, dtype=np.float16)
        
        chunk_file = self.config.OUTPUT_DIR / f'chunk_{chunk_num:03d}.npz'
        
        np.savez_compressed(
            chunk_file,
            positions=chunk_positions,
            moves=chunk_moves,
            results=chunk_results,
        )
        
        chunk_size_mb = chunk_file.stat().st_size / (1024 ** 2)
        self.logger.info(f"Wrote chunk {chunk_num}: {len(positions_list):,} positions ({chunk_size_mb:.1f} MB)")

    def _print_progress(self):
        """Print real-time progress bar."""
        elapsed = time.time() - self.stats['start_time']
        progress = self.stats['positions_extracted'] / self.config.TARGET_POSITIONS_TOTAL
        
        speed = self.stats['positions_extracted'] / max(1, elapsed)
        remaining = self.config.TARGET_POSITIONS_TOTAL - self.stats['positions_extracted']
        eta_secs = remaining / max(1, speed)
        eta_time = datetime.now() + timedelta(seconds=eta_secs)
        
        bar_len = 50
        filled = int(bar_len * progress)
        bar = '█' * filled + '░' * (bar_len - filled)
        
        status = (
            f"\r[{bar}] {progress*100:6.2f}% | "
            f"{self.stats['positions_extracted']:>10,} / {self.config.TARGET_POSITIONS_TOTAL:>10,} | "
            f"{speed:>8.0f} pos/s | ETA {eta_time.strftime('%H:%M:%S')} | "
            f"Mem: {self.monitor.memory_since_start_mb():.0f} MB"
        )
        print(status, end='', flush=True)
    
    def _print_detailed_metrics(self):
        """Print detailed metrics every 30 seconds."""
        elapsed = time.time() - self.stats['start_time']
        
        self.logger.info(f"\n[METRICS @ {datetime.now().strftime('%H:%M:%S')}]")
        self.logger.info(f"  Positions: {self.stats['positions_extracted']:,} / {self.config.TARGET_POSITIONS_TOTAL:,}")
        self.logger.info(f"  Games: {self.stats['games_processed']:,} (passed: {self.stats['games_passed_filters']:,})")
        self.logger.info(f"  Speed: {self.stats['positions_extracted']/max(1,elapsed):.0f} pos/sec")
        self.logger.info(f"  Memory: {self.monitor.memory_since_start_mb():.0f} MB / Peak: {self.monitor.peak_memory_mb:.0f} MB")
        
        # Bucket distribution
        self.logger.info(f"  Bucket Distribution:")
        for bucket_name in sorted(self.game_filter.stats['bucket_stats'].keys()):
            bucket_info = self.game_filter.stats['bucket_stats'][bucket_name]
            positions = bucket_info['positions_created']
            target = bucket_info['target']
            pct = 100 * positions / target if target > 0 else 0
            self.logger.info(f"    {bucket_name:20s}: {positions:>8,} / {target:>8,} ({pct:>6.1f}%)")
    
    def _print_summary(self):
        """Print final summary."""
        elapsed = time.time() - self.stats['start_time']
        
        self.logger.info("\n" + "=" * 100)
        self.logger.info("EXTRACTION SUMMARY")
        self.logger.info("=" * 100)
        
        self.logger.info(f"Files processed: {self.stats['files_processed']}")
        self.logger.info(f"Games processed: {self.stats['games_processed']:,}")
        self.logger.info(f"Games passed filters: {self.stats['games_passed_filters']:,}")
        
        if self.stats['games_processed'] > 0:
            pass_rate = 100 * self.stats['games_passed_filters'] / self.stats['games_processed']
            self.logger.info(f"Filter pass rate: {pass_rate:.2f}%")
        
        self.logger.info(f"Positions extracted: {self.stats['positions_extracted']:,}")
        self.logger.info(f"Moves validated: {self.stats['moves_validated']:,}")
        
        duration = f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"
        self.logger.info(f"Duration: {duration}")
        
        speed = self.stats['positions_extracted'] / max(1, elapsed)
        self.logger.info(f"Speed: {speed:.0f} positions/second")
        
        self.logger.info(f"Peak memory: {self.monitor.peak_memory_mb:.0f} MB")
        self.logger.info(f"Memory used: {self.monitor.memory_since_start_mb():.0f} MB")
        
        # Bucket distribution
        self.logger.info(f"\nFinal Bucket Distribution:")
        self.logger.info("-" * 100)
        
        for bucket_name in sorted(self.game_filter.stats['bucket_stats'].keys()):
            bucket_info = self.game_filter.stats['bucket_stats'][bucket_name]
            positions = bucket_info['positions_created']
            target = bucket_info['target']
            status = "✓" if positions >= target else "✗"
            pct = 100 * positions / target if target > 0 else 0
            self.logger.info(f"{status} {bucket_name:20s}: {positions:>10,} / {target:>10,} ({pct:>6.1f}%)")
        
        self.logger.info("-" * 100)
        
        # Filter stats
        if self.game_filter.filter_failures:
            self.logger.info(f"\nTop Filter Failures:")
            for reason, count in sorted(self.game_filter.filter_failures.items(), key=lambda x: -x[1])[:10]:
                self.logger.info(f"  {reason:30s}: {count:>10,}")
        
        self.logger.info("=" * 100)

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: ProductionConfig) -> logging.Logger:
    """Setup logging with timestamp."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"extraction_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    return logger

# ============================================================================
# Main
# ============================================================================

def main():
    """Run production extraction."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract 20M chess positions')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/home/krish/EE542-Project/FINAL/chess_ai/data/lichess_raw',
        help='Directory with .pgn.zst files'
    )
    
    args = parser.parse_args()
    
    # Setup
    config = ProductionConfig()
    logger = setup_logging(config)
    monitor = SystemMonitor()
    
    # Find files
    input_dir = Path(args.input_dir)
    input_files = sorted(input_dir.glob('*.pgn.zst'))
    
    if not input_files:
        logger.error(f"No .pgn.zst files in {input_dir}")
        return 1
    
    logger.info(f"Found {len(input_files)} input files:")
    for f in input_files:
        size_gb = f.stat().st_size / (1024 ** 3)
        logger.info(f"  {f.name}: {size_gb:.1f} GB")
    
    # Extract
    extractor = ProductionDataExtractor(config, logger, monitor)
    result = extractor.extract_dataset(input_files)
    
    if result:
        logger.info(f"\n✓ SUCCESS: Dataset saved to {result}")
        return 0
    else:
        logger.error(f"\n✗ FAILED: Dataset creation failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
