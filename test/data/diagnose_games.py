"""
Diagnostic script to understand why games are being filtered out
"""

import chess.pgn
import subprocess
from pathlib import Path
from collections import defaultdict

def stream_from_compressed(zst_file):
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

def analyze_games(input_file, num_games=100):
    """Analyze the first N games to see what's in the dataset"""

    stats = {
        'total': 0,
        'elo_issues': defaultdict(int),
        'time_control_issues': 0,
        'move_count_issues': 0,
        'termination_issues': defaultdict(int),
        'passes_all': 0
    }

    sample_games = []

    print(f"Analyzing first {num_games} games from {input_file}...\n")

    game_stream = stream_from_compressed(Path(input_file))

    for game in game_stream:
        if stats['total'] >= num_games:
            break

        stats['total'] += 1

        # Store first 5 games for detailed inspection
        if len(sample_games) < 5:
            sample_games.append(game)

        # Check each filter condition
        passes = True

        # ELO check
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))

            if white_elo < 1500 or black_elo < 1500:
                stats['elo_issues']['below_1500'] += 1
                passes = False
            elif white_elo == 0 or black_elo == 0:
                stats['elo_issues']['missing_elo'] += 1
                passes = False
        except (ValueError, TypeError):
            stats['elo_issues']['invalid_elo'] += 1
            passes = False

        # Time control check
        try:
            time_control = game.headers.get("TimeControl", "")
            if not time_control or time_control == "-":
                stats['time_control_issues'] += 1
                passes = False
            else:
                base_time = int(time_control.split('+')[0])
                if base_time < 300:
                    stats['time_control_issues'] += 1
                    passes = False
        except (ValueError, TypeError, IndexError):
            stats['time_control_issues'] += 1
            passes = False

        # Move count check
        try:
            ply_count = int(game.headers.get("PlyCount", 0))
            move_count = ply_count // 2
            if move_count < 15 or move_count > 200:
                stats['move_count_issues'] += 1
                passes = False
        except (ValueError, TypeError):
            stats['move_count_issues'] += 1
            passes = False

        # Termination check
        termination = game.headers.get("Termination", "")
        if termination not in ["Normal", "Time forfeit"]:
            stats['termination_issues'][termination if termination else 'missing'] += 1
            passes = False

        if passes:
            stats['passes_all'] += 1

    # Print results
    print("=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)
    print(f"\nGames analyzed: {stats['total']}")
    print(f"Games passing all filters: {stats['passes_all']} ({stats['passes_all']/stats['total']*100:.1f}%)")

    print("\n" + "-" * 70)
    print("FILTER BREAKDOWN:")
    print("-" * 70)

    print(f"\nELO issues: {sum(stats['elo_issues'].values())}")
    for reason, count in stats['elo_issues'].items():
        print(f"  - {reason}: {count}")

    print(f"\nTime control issues: {stats['time_control_issues']}")
    print(f"Move count issues: {stats['move_count_issues']}")

    print(f"\nTermination issues: {sum(stats['termination_issues'].values())}")
    for reason, count in stats['termination_issues'].items():
        print(f"  - {reason}: {count}")

    # Show sample game headers
    print("\n" + "=" * 70)
    print("SAMPLE GAME HEADERS (first 5 games):")
    print("=" * 70)

    for i, game in enumerate(sample_games, 1):
        print(f"\n--- Game {i} ---")
        print(f"WhiteElo: {game.headers.get('WhiteElo', 'N/A')}")
        print(f"BlackElo: {game.headers.get('BlackElo', 'N/A')}")
        print(f"TimeControl: {game.headers.get('TimeControl', 'N/A')}")
        print(f"PlyCount: {game.headers.get('PlyCount', 'N/A')}")
        print(f"Termination: {game.headers.get('Termination', 'N/A')}")
        print(f"Result: {game.headers.get('Result', 'N/A')}")

if __name__ == "__main__":
    input_file = "lichess_data/lichess_db_standard_rated_2025-09.pgn.zst"
    analyze_games(input_file, num_games=1000)
