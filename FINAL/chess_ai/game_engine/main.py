import multiprocessing as mp
import os
import sys
import time
import shutil
import torch
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.neural_net import InferenceServer
from game_engine.mcts import MCTSWorker
from game_engine.chess_env import ChessGame
from game_engine.trainer import train_model
from game_engine.evaluation import Arena, StockfishEvaluator, MetricsLogger
from game_engine.cnn import ChessCNN

# ==========================================
#        PRODUCTION TEST CONFIGURATION
# ==========================================

# --- EXECUTION ---
ITERATIONS = 1000             # Just 1 loop to verify the high-quality settings

# Auto-detect CPUs: Leave 2 cores free for OS & GPU Server
# On a 32-core cloud VM, this gives 30 workers.
NUM_WORKERS = 60

# Generation
GAMES_PER_WORKER = 5       # Lower batch size per worker for higher parallelism
                           # Total Games/Iter = ~60 workers * 5 = 300 games

# --- QUALITY ---
SIMULATIONS = 400          # High quality search (Standard "strong" setting)

# Training
TRAIN_EPOCHS = 2           # Prevent overfitting on small iterative datasets

# Evaluation
EVAL_GAMES = 20            # 10 White, 10 Black
STOCKFISH_GAMES = 10       # Accurate Elo tracking

# --- PATHS ---
STOCKFISH_PATH = "/usr/games/stockfish" 
LOG_FILE = "training_log.txt"
MODEL_DIR = "game_engine/model"
DATA_DIR = "data/self_play"
BEST_MODEL = f"{MODEL_DIR}/best_model.pth"
CANDIDATE_MODEL = f"{MODEL_DIR}/candidate.pth"

# ==========================================
class Logger(object):
    """Redirects stdout to both file and console."""
    def __init__(self):
        self.terminal = sys.stdout
        # buffering=1 means line buffered (writes to disk every new line)
        self.log = open(LOG_FILE, "a", buffering=1, encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except Exception:
            pass # Prevent logging errors from crashing training

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except Exception:
            pass

def setup_child_logging():
    """Forces child processes to use the custom Logger"""
    sys.stdout = Logger()
    sys.stderr = sys.stdout

# ==========================================

def run_worker_batch(worker_id, input_queue, output_queue, game_limit):
    """
    Generates high-quality games with Temperature Decay.
    """
    # CRITICAL FIX: Re-attach logger inside the child process
    setup_child_logging()

    print(f"   [Worker {worker_id}] Starting batch of {game_limit} HQ games (400 sims)...")
    worker = MCTSWorker(worker_id, input_queue, output_queue, simulations=SIMULATIONS)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for i in range(game_limit):
        game_start = time.time()
        game = ChessGame()
        game_data = []
        
        while not game.is_over:
            # --- ALPHA ZERO TEMPERATURE SCHEDULE ---
            # Moves 0-30 (Plies): High Temperature (1.0) for exploration
            # Moves 30+: Low Temperature (0.1) for winning precision
            # len(game.moves) counts plies (half-moves).
            current_temp = 1.0 if len(game.moves) < 30 else 0.1
            
            best_move, mcts_policy = worker.search(game, temperature=current_temp)
            
            game_data.append({
                "state": game.to_tensor(),
                "policy": mcts_policy,
                "turn": game.turn_player
            })
            game.push(best_move)
            
            # Optional: Print progress for long games
            if worker_id == 0 and len(game.moves) % 10 == 0:
                print(f"   [Worker 0] Game {i+1} Move {len(game.moves)}...")
        
        # Save Game
        result = game.result
        final_winner = 0.0
        if result == "1-0": final_winner = 1.0
        elif result == "0-1": final_winner = 0.0
        
        # Value Targets
        values = []
        for g in game_data:
            if result == "1/2-1/2": values.append(0.0)
            elif g["turn"] == final_winner: values.append(1.0)
            else: values.append(-1.0)
            
        timestamp = int(time.time())
        filename = f"{DATA_DIR}/iter_{timestamp}_w{worker_id}_{i}.npz"
        np.savez_compressed(filename, 
                            states=np.array([g["state"] for g in game_data]), 
                            policies=np.array([g["policy"] for g in game_data]), 
                            values=np.array(values, dtype=np.float32))
        
        duration = time.time() - game_start
        print(f"   [Worker {worker_id}] Finished Game {i+1}/{game_limit} ({len(game.moves)} moves) in {duration:.1f}s")
        
    print(f"   [Worker {worker_id}] Batch Complete.")

def run_server_wrapper(server):
    """Wrapper to ensure server process also logs to file"""
    setup_child_logging()
    server.loop()

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE (High Quality) ===")
    
    server = InferenceServer(BEST_MODEL)
    worker_queues = [server.register_worker(i) for i in range(NUM_WORKERS)]
    
    # Updated to use wrapper for logging
    server_process = mp.Process(target=run_server_wrapper, args=(server,))
    server_process.start()
    time.sleep(3) 
    
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=run_worker_batch, 
                       args=(i, server.input_queue, worker_queues[i], GAMES_PER_WORKER))
        p.start()
        workers.append(p)
        
    for p in workers:
        p.join()
        
    server.input_queue.put("STOP")
    server_process.join(timeout=5)
    if server_process.is_alive():
        server_process.terminate()
        
    print(f"=== SELF-PLAY COMPLETE: Generated {NUM_WORKERS * GAMES_PER_WORKER} HQ games ===")

def run_training_phase(iteration):
    print(f"\n=== ITERATION {iteration}: TRAINING PHASE ===")
    train_model(data_path=DATA_DIR, 
                input_model_path=BEST_MODEL, 
                output_model_path=CANDIDATE_MODEL,
                epochs=TRAIN_EPOCHS)

def run_evaluation_phase(iteration, logger):
    print(f"\n=== ITERATION {iteration}: EVALUATION PHASE ===")
    
    # 1. Arena (Simulations match generation to be fair)
    arena = Arena(CANDIDATE_MODEL, BEST_MODEL, simulations=SIMULATIONS)
    win_rate = arena.play_match(num_games=EVAL_GAMES)
    
    # 2. Promotion Logic
    if win_rate >= 0.55:
        print(f"🚀 PROMOTION! Candidate ({win_rate:.2f}) defeated Champion.")
        shutil.move(CANDIDATE_MODEL, BEST_MODEL)
    else:
        print(f"❌ REJECTED. Candidate ({win_rate:.2f}) failed to beat Champion.")
        if os.path.exists(CANDIDATE_MODEL):
            os.remove(CANDIDATE_MODEL)
            
    # 3. Stockfish Benchmark
    sf_eval = StockfishEvaluator(STOCKFISH_PATH, simulations=SIMULATIONS)
    elo = sf_eval.evaluate(BEST_MODEL, num_games=STOCKFISH_GAMES, stockfish_elo=1350) 
    
    logger.log(iteration, policy_loss=0.0, value_loss=0.0, arena_win_rate=win_rate, elo=elo)

if __name__ == "__main__":
    
    # --- LOGGING SETUP ---
    # Redirect stdout and stderr to both console and file
    setup_child_logging()

    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    logger = MetricsLogger()
    
    print("=================================================")
    print(f"STARTING PRODUCTION TEST (1 Iteration)")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS}")
    print("=================================================")

    try:
        for it in range(1, ITERATIONS + 1):
            run_self_play_phase(it)
            run_training_phase(it)
            run_evaluation_phase(it, logger)
            
    except KeyboardInterrupt:
        print("\n\n--- LOOP STOPPED BY USER ---")