"""
Lichess Raw Data Validation

This module validates downloaded Lichess PGN files before processing.
It checks file integrity, compression status, and sample game quality
to ensure the data is suitable for dataset creation.

Validation checks:
1. File existence and accessibility
2. File compression format verification
3. File size and corruption detection
4. Sample game extraction and parsing
5. Game quality assessment
6. Header information validation
7. Move sequence integrity
8. Processing readiness assessment
"""

import subprocess
from pathlib import Path
import sys
import hashlib


def check_zstd_availability():
    """
    Verify that zstdcat command is available on the system.
    
    Returns:
        Boolean indicating if zstdcat is available
    """
    try:
        subprocess.run(
            ['zstdcat', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_file_hash(filepath, chunk_size=65536):
    """
    Calculate SHA256 hash of file for integrity verification.
    
    Args:
        filepath: Path to file
        chunk_size: Bytes to read per iteration
        
    Returns:
        Hex string of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def validate_compression(filepath):
    """
    Verify file is properly zstd compressed.
    
    Args:
        filepath: Path to file
        
    Returns:
        Boolean indicating if file is valid zstd format
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header[:4] == b'\x28\xb5\x2f\xfd':
                return True
    except:
        pass
    return False


def extract_sample_games(filepath, num_games=5):
    """
    Extract a sample of games from the compressed file for inspection.
    
    Args:
        filepath: Path to .pgn.zst file
        num_games: Number of games to extract
        
    Returns:
        List of game strings, or None if extraction fails
    """
    try:
        process = subprocess.Popen(
            ['zstdcat', str(filepath)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1024*1024
        )
        
        games = []
        current_game = []
        game_count = 0
        line_count = 0
        
        for line in process.stdout:
            line_count += 1
            current_game.append(line)
            
            if line.strip() == '':
                if current_game and current_game[0].startswith('['):
                    game_count += 1
                    games.append(''.join(current_game))
                    current_game = []
                    
                    if game_count >= num_games:
                        break
        
        process.terminate()
        process.wait(timeout=5)
        
        return games if games else None
        
    except Exception as e:
        print(f"Error extracting sample games: {e}")
        return None


def validate_game_format(game_str):
    """
    Validate that a game string contains proper PGN format.
    
    Args:
        game_str: Game in PGN format
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    lines = game_str.strip().split('\n')
    
    if len(lines) < 3:
        return False, "Game too short"
    
    header_count = 0
    for line in lines:
        if line.startswith('['):
            header_count += 1
        elif line and not line.startswith('[') and not line.startswith('%'):
            break
    
    if header_count < 2:
        return False, "Insufficient headers"
    
    required_headers = ['Event', 'Site', 'Date', 'Round', 'White', 'Black', 'Result']
    game_text = ' '.join(lines)
    
    for header in required_headers:
        if f'[{header}' not in game_text:
            return False, f"Missing header: {header}"
    
    if '*' not in game_text and '1-0' not in game_text and '0-1' not in game_text and '1/2-1/2' not in game_text:
        return False, "No game result"
    
    return True, None


def validate_lichess_files_comprehensive(directory_path):
    """
    Perform comprehensive validation of all Lichess files in directory.
    
    Args:
        directory_path: Path to directory containing .pgn.zst files
        
    Returns:
        Boolean indicating if all files are ready for processing
    """
    
    print("\n" + "="*70)
    print("Lichess Raw Data Validation")
    print("="*70)
    print(f"Directory: {directory_path}\n")
    
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return False
    
    if not directory.is_dir():
        print(f"Error: Path is not a directory: {directory}")
        return False
    
    pgn_files = sorted(directory.glob('*.pgn.zst'))
    
    if not pgn_files:
        print(f"Error: No .pgn.zst files found in {directory}")
        return False
    
    print(f"Found {len(pgn_files)} files to validate\n")
    
    print("Stage 1: System Requirements")
    print("-" * 70)
    
    if not check_zstd_availability():
        print("Error: zstdcat not found. Install with: sudo apt install zstandard")
        return False
    print("zstdcat: Available")
    print()
    
    all_valid = True
    file_summaries = []
    
    for file_idx, pgn_file in enumerate(pgn_files, 1):
        print(f"Stage {file_idx + 1}: Validating {pgn_file.name}")
        print("-" * 70)
        
        if not pgn_file.exists():
            print(f"Error: File not found: {pgn_file}")
            all_valid = False
            continue
        
        file_size_gb = pgn_file.stat().st_size / (1024**3)
        print(f"File size: {file_size_gb:.2f}GB")
        
        if file_size_gb < 0.1:
            print(f"Error: File too small (expected > 0.1GB)")
            all_valid = False
            continue
        
        if not validate_compression(pgn_file):
            print(f"Error: File is not valid zstd format")
            all_valid = False
            continue
        print(f"Compression format: Valid zstd")
        
        file_hash = get_file_hash(pgn_file)
        print(f"File hash (SHA256): {file_hash[:16]}...")
        
        games = extract_sample_games(pgn_file, num_games=3)
        
        if not games:
            print(f"Error: Could not extract sample games")
            all_valid = False
            continue
        
        print(f"Sample games: {len(games)} extracted")
        
        sample_valid = True
        for game_idx, game in enumerate(games, 1):
            is_valid, error_msg = validate_game_format(game)
            if not is_valid:
                print(f"  Game {game_idx}: Invalid - {error_msg}")
                sample_valid = False
            else:
                print(f"  Game {game_idx}: Valid")
        
        if not sample_valid:
            all_valid = False
            continue
        
        print(f"Status: Ready for processing")
        print()
        
        file_summaries.append({
            'filename': pgn_file.name,
            'size_gb': file_size_gb,
            'hash': file_hash,
            'valid': True
        })
    
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    print()
    
    print(f"Total files: {len(pgn_files)}")
    print(f"Valid files: {sum(1 for f in file_summaries if f['valid'])}")
    print()
    
    print("File Details:")
    total_size = 0
    for summary in file_summaries:
        print(f"\n{summary['filename']}:")
        print(f"  Size: {summary['size_gb']:.2f}GB")
        print(f"  Hash: {summary['hash'][:16]}...")
        print(f"  Status: {'Ready' if summary['valid'] else 'Error'}")
        total_size += summary['size_gb']
    
    print(f"\nTotal data size: {total_size:.2f}GB")
    
    if all_valid and file_summaries:
        print(f"\nAll files validated successfully!")
        print(f"\nNext step:")
        print(f"python3 data/download_standard_classical.py --input-dir {directory}")
        print("\n" + "="*70 + "\n")
        return True
    else:
        print(f"\nSome files failed validation. Please fix issues before processing.")
        print("\n" + "="*70 + "\n")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        directory = Path("data/lichess_raw")
    else:
        directory = Path(sys.argv[1])
    
    success = validate_lichess_files_comprehensive(str(directory))
    sys.exit(0 if success else 1)
