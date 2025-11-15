#!/usr/bin/env python3
"""
Week 1 Validation Script for Chess AI Project
Verifies all milestone requirements for chess environment.
"""

import sys
import os
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine.chess_env import ChessGame
import chess

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_fail(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.YELLOW}ℹ {text}{Colors.END}")

def test_basic_game_functionality():
    """Test 1: Basic game initialization and move generation"""
    print_header("TEST 1: Basic Game Functionality")
    
    try:
        game = ChessGame()
        print_success("ChessGame initialized successfully")
        
        # Test initial state
        state = game.get_state()
        assert state['move_count'] == 0, "Initial move count should be 0"
        print_success(f"Initial state correct: {state['move_count']} moves")
        
        # Test legal moves generation
        legal_moves = game.legal_moves()
        assert len(legal_moves) == 20, "Starting position should have 20 legal moves"
        print_success(f"Legal moves generated: {len(legal_moves)} moves")
        
        # Test making a move
        move = 'e2e4'
        success = game.push(move)
        assert success, "Should be able to make e2e4"
        print_success(f"Move {move} executed successfully")
        
        # Verify move was recorded
        assert game.last_move == move, "Last move should be recorded"
        assert len(game.moves) == 1, "Move history should have 1 move"
        print_success("Move history tracking works correctly")
        
        return True
    except Exception as e:
        print_fail(f"Basic functionality test failed: {e}")
        return False

def test_random_game_generation(num_games=10):
    """Test 2: Generate multiple random games"""
    print_header(f"TEST 2: Generate {num_games} Random Games")
    
    games_data = []
    errors = []
    
    for i in range(num_games):
        try:
            game = ChessGame()
            moves_count = 0
            
            while not game.is_over and moves_count < 500:  # Limit to prevent infinite loops
                move = game.random_move()
                if move is None:
                    break
                moves_count += 1
            
            games_data.append({
                'game_num': i + 1,
                'moves': game.moves,
                'result': game.result,
                'total_moves': len(game.moves)
            })
            
            if (i + 1) % 10 == 0:
                print_info(f"Completed {i + 1}/{num_games} games...")
                
        except Exception as e:
            errors.append(f"Game {i + 1}: {e}")
            print_fail(f"Game {i + 1} crashed: {e}")
    
    if len(errors) == 0:
        print_success(f"All {num_games} games completed without crashes")
        
        # Statistics
        total_moves = sum(g['total_moves'] for g in games_data)
        avg_moves = total_moves / len(games_data)
        print_info(f"Average moves per game: {avg_moves:.1f}")
        
        results = {}
        for g in games_data:
            result = g['result'] or 'Draw'
            results[result] = results.get(result, 0) + 1
        
        print_info(f"Results distribution: {results}")
        return True
    else:
        print_fail(f"{len(errors)} games failed")
        for error in errors[:5]:  # Show first 5 errors
            print_fail(f"  {error}")
        return False

def test_large_scale_generation():
    """Test 3: Generate 100 games (milestone requirement)"""
    print_header("TEST 3: Generate 100 Random Games (Milestone Test)")
    
    start_time = time.time()
    num_games = 100
    games_data = []
    crashes = 0
    
    print_info(f"Starting generation of {num_games} games...")
    
    for i in range(num_games):
        try:
            game = ChessGame()
            move_limit = 0
            
            while not game.is_over and move_limit < 500:
                game.random_move()
                move_limit += 1
            
            games_data.append({
                'game_id': game.game_id,
                'moves': game.moves,
                'result': game.result,
                'move_count': len(game.moves)
            })
            
            if (i + 1) % 20 == 0:
                print_info(f"Progress: {i + 1}/{num_games} games")
                
        except Exception as e:
            crashes += 1
            print_fail(f"Game {i + 1} crashed: {e}")
    
    elapsed_time = time.time() - start_time
    
    if crashes == 0:
        print_success(f"✓ MILESTONE ACHIEVED: {num_games} games completed without crashes")
        print_info(f"Time taken: {elapsed_time:.2f} seconds")
        print_info(f"Average time per game: {elapsed_time/num_games:.3f} seconds")
        
        # Save results
        output_file = f"data/validation_games_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('data', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(games_data, f, indent=2)
        
        print_success(f"Game data saved to: {output_file}")
        
        # Statistics
        total_moves = sum(g['move_count'] for g in games_data)
        avg_moves = total_moves / len(games_data)
        min_moves = min(g['move_count'] for g in games_data)
        max_moves = max(g['move_count'] for g in games_data)
        
        results_count = {}
        for g in games_data:
            result = g['result'] or '1/2-1/2'
            results_count[result] = results_count.get(result, 0) + 1
        
        print_info("\n📊 Final Statistics:")
        print_info(f"  Total moves played: {total_moves}")
        print_info(f"  Average moves/game: {avg_moves:.1f}")
        print_info(f"  Min moves: {min_moves}")
        print_info(f"  Max moves: {max_moves}")
        print_info(f"  Results: {results_count}")
        
        return True
    else:
        print_fail(f"MILESTONE FAILED: {crashes} crashes detected")
        return False

def test_fen_validity():
    """Test 4: FEN string validity throughout gameplay"""
    print_header("TEST 4: FEN String Validity")
    
    try:
        game = ChessGame()
        
        for i in range(20):  # Test 20 random moves
            fen = game.board.fen()
            
            # Validate FEN by creating a new board from it
            try:
                test_board = chess.Board(fen)
                print_success(f"Move {i}: FEN valid") if i < 5 else None
            except Exception as e:
                print_fail(f"Invalid FEN at move {i}: {e}")
                return False
            
            if game.is_over:
                break
            
            game.random_move()
        
        print_success("All FEN strings are valid throughout gameplay")
        return True
        
    except Exception as e:
        print_fail(f"FEN validation failed: {e}")
        return False

def test_move_legality():
    """Test 5: All moves are legal according to chess rules"""
    print_header("TEST 5: Move Legality Verification")
    
    try:
        game = ChessGame()
        illegal_moves = 0
        
        for i in range(50):
            legal_moves_before = set(game.legal_moves())
            move = game.random_move()
            
            if move is None:
                break
            
            # Check if the move was in the legal moves list
            if move not in legal_moves_before:
                illegal_moves += 1
                print_fail(f"Illegal move detected: {move}")
        
        if illegal_moves == 0:
            print_success("All moves are legal according to chess rules")
            return True
        else:
            print_fail(f"{illegal_moves} illegal moves detected")
            return False
            
    except Exception as e:
        print_fail(f"Move legality test failed: {e}")
        return False

def test_game_termination():
    """Test 6: Games terminate properly"""
    print_header("TEST 6: Game Termination")
    
    try:
        terminated_properly = 0
        
        for i in range(10):
            game = ChessGame()
            moves = 0
            
            while moves < 500:  # Safety limit
                if game.is_over:
                    terminated_properly += 1
                    break
                
                game.random_move()
                moves += 1
        
        print_success(f"{terminated_properly}/10 games terminated properly")
        return terminated_properly == 10
        
    except Exception as e:
        print_fail(f"Termination test failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests"""
    print_header("WEEK 1 CHESS AI VALIDATION SUITE")
    print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all tests
    results.append(("Basic Game Functionality", test_basic_game_functionality()))
    results.append(("Random Game Generation (10 games)", test_random_game_generation(10)))
    results.append(("FEN Validity", test_fen_validity()))
    results.append(("Move Legality", test_move_legality()))
    results.append(("Game Termination", test_game_termination()))
    results.append(("Large Scale Generation (100 games)", test_large_scale_generation()))
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_fail(f"{test_name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*70}")
        print(f"{'🎉 WEEK 1 MILESTONE COMPLETE! 🎉':^70}")
        print(f"{'All validations passed successfully':^70}")
        print(f"{'='*70}{Colors.END}\n")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}{'='*70}")
        print(f"{'⚠ VALIDATION FAILED ⚠':^70}")
        print(f"{f'{total - passed} test(s) need attention':^70}")
        print(f"{'='*70}{Colors.END}\n")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.END}")
        sys.exit(1)
