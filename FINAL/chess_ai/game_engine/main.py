import multiprocessing as mp
import threading
import os
import sys
import time
import shutil
import gc
import torch
import numpy as np
import math

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.neural_net import InferenceServer
from game_engine.mcts import MCTSWorker
from game_engine.chess_env import ChessGame
from game_engine.trainer import train_model
from game_engine.evaluation import Arena, StockfishEvaluator, MetricsLogger
from game_engine.cnn import ChessCNN

# ==========================================
#        AGGRESSIVE PRODUCTION CONFIG
# ==========================================

# --- EXECUTION ---
ITERATIONS = 1000

# NEW SCALED UP CONFIG (48 vCPUs / 208 GB RAM)
# NOTE: NUM_WORKERS is scaled for performance but limited for RAM safety.
NUM_WORKERS = 100
GAMES_PER_WORKER = 2        # Total Games = 100 * 2 = 200 per iteration.
# Reserve last few cores for the Inference Server
RESERVED_SERVER_CORES = 4
TOTAL_VCPUS = 48 # Configuration assumption

# --- QUALITY ---
SIMULATIONS = 1200          # Deep Training
EVAL_SIMULATIONS = 400      # Fast Evaluation

# --- EVALUATION CONFIG (VRAM SAFE) --
EVAL_WORKERS = 5           
GAMES_PER_EVAL_WORKER = 4
STOCKFISH_GAMES = 20
SF_WORKERS = 5              # Reduced Stockfish workers for VRAM safety
SF_GAMES_PER_WORKER = 4     # 5 * 4 = 20 total games (20 total games are essential for good statistics)
STOCKFISH_ELO = 1350        # Base Elo to evaluate against

# --- AGGRESSIVE RULES ---
MAX_MOVES_PER_GAME = 100   
DRAW_PENALTY = -0.2

# Training
TRAIN_EPOCHS = 2           

# --- PATHS ---
STOCKFISH_PATH = "/usr/games/stockfish" 
LOG_FILE = "training_log.txt"
MODEL_DIR = "game_engine/model"
DATA_DIR = "data/self_play"
BEST_MODEL = f"{MODEL_DIR}/best_model.pth"
CANDIDATE_MODEL = f"{MODEL_DIR}/candidate.pth"

# ==========================================
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOG_FILE, "a", buffering=1, encoding='utf-8')
    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except: pass 
    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except: pass

def setup_child_logging():
    sys.stdout = Logger()
    sys.stderr = sys.stderr

# ==========================================
#        HELPER: QUEUE MONITOR
# ==========================================
def queue_monitor_thread(queue):
    while True:
        try:
            size = queue.qsize()
            if size > 0:
                print(f"   [Server Monitor] Pending Requests in Queue: {size}")
            time.sleep(2.0)
        except: break

# ==========================================
#        PHASE 1: SELF-PLAY WORKERS
# ==========================================
def run_worker_batch(worker_id, input_queue, output_queue, game_limit):
    # --- WORKER PINNING ---
    if hasattr(os, 'sched_setaffinity'):
        try:
            total_cores = os.cpu_count()
            worker_core_count = total_cores - RESERVED_SERVER_CORES
            
            if worker_core_count > 0:
                core_id = worker_id % worker_core_count
                os.sched_setaffinity(0, {core_id})
        except Exception as e:
            print(f"   [Worker {worker_id}] Pinning error: {e}") 

    setup_child_logging()
    # Reduced startup delay for massive worker count
    time.sleep(worker_id * 0.01)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for i in range(game_limit):
        print(f"   [Worker {worker_id}] Starting Game {i+1} (Aggressive Mode)...")
        worker = MCTSWorker(worker_id, input_queue, output_queue, simulations=SIMULATIONS)
        
        game_start = time.time()
        game = ChessGame()
        game_data = []
        
        while not game.is_over:
            if len(game.moves) >= MAX_MOVES_PER_GAME:
                # FIXED: Consistent log message
                print(f"   [Worker {worker_id}] Hit limit ({MAX_MOVES_PER_GAME} plies). Adjudicating Draw.")
                break 

            move_start = time.time()
            current_temp = 1.0 if len(game.moves) < 30 else 0.1
            best_move, mcts_policy = worker.search(game, temperature=current_temp)
            
            if worker_id == 0:
                print(f"   [Worker 0] Move {len(game.moves)+1}: {best_move} ({time.time()-move_start:.2f}s)")
            
            game_data.append({
                "state": game.to_tensor(),
                "policy": mcts_policy,
                "turn": game.turn_player
            })
            game.push(best_move)
        
        # Save Game
        if len(game.moves) >= MAX_MOVES_PER_GAME: result = "1/2-1/2"
        else: result = game.result

        # --- REWARD CALCULATION (WITH PENALTY) ---
        final_winner = 0.0
        if result == "1-0": final_winner = 1.0
        elif result == "0-1": final_winner = 0.0
        
        values = []
        for g in game_data:
            if result == "1/2-1/2": 
                values.append(DRAW_PENALTY)
            elif g["turn"] == final_winner: 
                values.append(1.0)
            else: 
                values.append(-1.0)
            
        timestamp = int(time.time())
        filename = f"{DATA_DIR}/iter_{timestamp}_w{worker_id}_{i}.npz"
        np.savez_compressed(filename, 
                            states=np.array([g["state"] for g in game_data]), 
                            policies=np.array([g["policy"] for g in game_data]), 
                            values=np.array(values, dtype=np.float32))
        
        print(f"   [Worker {worker_id}] Finished Game {i+1}/{game_limit} in {time.time()-game_start:.1f}s.")
        del worker, game, game_data
        gc.collect()

def run_server_wrapper(server):
    if hasattr(os, 'sched_setaffinity'):
        try:
            total_cores = os.cpu_count()
            server_start_core = total_cores - RESERVED_SERVER_CORES
            
            if server_start_core < total_cores:
                server_cores = {i for i in range(server_start_core, total_cores)}
                os.sched_setaffinity(0, server_cores)
                print(f"Server pinned to Cores {server_start_core}-{total_cores-1} ({RESERVED_SERVER_CORES} cores reserved)")
        except Exception as e:
            print(f"Server pinning error: {e}")
            
    setup_child_logging()
    monitor = threading.Thread(target=queue_monitor_thread, args=(server.input_queue,))
    monitor.daemon = True 
    monitor.start()
    server.loop()

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE ===")
    
    if TOTAL_VCPUS > os.cpu_count():
        print(f"WARNING: Configured VCPUS ({TOTAL_VCPUS}) > Detected ({os.cpu_count()}). Pinning limited.")

    server = InferenceServer(BEST_MODEL)
    worker_queues = [server.register_worker(i) for i in range(NUM_WORKERS)]
    
    server_process = mp.Process(target=run_server_wrapper, args=(server,))
    server_process.start()
    time.sleep(5) 
    
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=run_worker_batch, 
                       args=(i, server.input_queue, worker_queues[i], GAMES_PER_WORKER))
        p.start()
        workers.append(p)
        
    for p in workers: p.join()
        
    server.input_queue.put("STOP")
    server_process.join(timeout=10)
    if server_process.is_alive(): server_process.terminate()

# ==========================================
#        PHASE 2: TRAINING
# ==========================================
def run_training_phase(iteration):
    print(f"\n=== ITERATION {iteration}: TRAINING PHASE ===")
    train_model(data_path=DATA_DIR, 
                input_model_path=BEST_MODEL, 
                output_model_path=CANDIDATE_MODEL,
                epochs=TRAIN_EPOCHS)

# ==========================================
#        PHASE 3: PARALLEL EVALUATION
# ==========================================

def calculate_elo(base_elo, total_adjusted_win_rate):
    """Calculates Elo difference based on the adjusted win rate."""
    if total_adjusted_win_rate is None or total_adjusted_win_rate <= 0.0 or total_adjusted_win_rate >= 1.0:
        return None 
    
    # Standard Elo difference formula based on winning probability
    elo_diff = -400 * math.log10(1 / total_adjusted_win_rate - 1)
        
    estimated_elo = base_elo + elo_diff
    return int(estimated_elo)

def run_arena_batch(worker_id, result_queue, num_games):
    setup_child_logging()
    try:
        arena = Arena(CANDIDATE_MODEL, BEST_MODEL, simulations=EVAL_SIMULATIONS, max_moves=MAX_MOVES_PER_GAME)
        print(f"   [Eval Worker {worker_id}] Playing {num_games} games (Cand vs Champ)...")
        chunk_win_rate = arena.play_match(num_games=num_games)
        result_queue.put(chunk_win_rate)
        print(f"   [Eval Worker {worker_id}] Batch Complete. Rate: {chunk_win_rate:.2f}")
    except Exception as e:
        print(f"   [Eval Worker {worker_id}] Error: {e}")
        result_queue.put(0.0)

def run_stockfish_batch(worker_id, result_queue, num_games):
    setup_child_logging()
    try:
        sf_eval = StockfishEvaluator(STOCKFISH_PATH, simulations=EVAL_SIMULATIONS)
        
        # StockfishEvaluator returns a dict {'win_rate':..., 'adjusted_win_rate':...}
        results = sf_eval.evaluate(BEST_MODEL, num_games=num_games, stockfish_elo=STOCKFISH_ELO)
        
        result_queue.put(results)
        print(f"   [SF Worker {worker_id}] Batch Complete. Adj Win Rate: {results.get('adjusted_win_rate', 0.5):.3f}")
    except Exception as e:
        print(f"   [SF Worker {worker_id}] Error: {e}")
        result_queue.put({'win_rate': 0.5, 'adjusted_win_rate': 0.5}) 

def run_evaluation_phase(iteration, logger):
    print(f"\n=== ITERATION {iteration}: PARALLEL EVALUATION PHASE ===")
    
    # --- 1. PARALLEL ARENA (Candidate vs Champion) ---
    print(f"Starting Arena: {EVAL_WORKERS} Workers x {GAMES_PER_EVAL_WORKER} Games ({EVAL_SIMULATIONS} sims)")
    
    arena_queue = mp.Queue()
    arena_workers = []
    
    for i in range(EVAL_WORKERS):
        p = mp.Process(target=run_arena_batch, args=(i, arena_queue, GAMES_PER_EVAL_WORKER))
        p.start()
        arena_workers.append(p)
        
    for p in arena_workers:
        p.join()
        
    # Aggregate Arena Results
    total_rates = []
    while not arena_queue.empty():
        total_rates.append(arena_queue.get())
        
    final_win_rate = sum(total_rates) / len(total_rates) if total_rates else 0.0
    
    # Promotion Logic
    if final_win_rate >= 0.55:
        print(f"🚀 PROMOTION! Candidate ({final_win_rate:.2f}) defeated Champion.")
        shutil.move(CANDIDATE_MODEL, BEST_MODEL)
        should_run_stockfish = True
    else:
        print(f"❌ REJECTED. Candidate ({final_win_rate:.2f}) failed to beat Champion.")
        if os.path.exists(CANDIDATE_MODEL):
            os.remove(CANDIDATE_MODEL)
        
        # Skip Stockfish (Your implemented optimization)
        should_run_stockfish = False

    final_elo = None
    
    if should_run_stockfish:
        # --- 2. PARALLEL STOCKFISH (Accurate Centralized Elo) ---
        print(f"\nStarting Stockfish Elo Eval (Model promoted): {SF_WORKERS} Workers x {SF_GAMES_PER_WORKER} Games")
        sf_queue = mp.Queue()
        sf_workers = []
        
        for i in range(SF_WORKERS):
            p = mp.Process(target=run_stockfish_batch, args=(i, sf_queue, SF_GAMES_PER_WORKER))
            p.start()
            sf_workers.append(p)
            
        for p in sf_workers:
            p.join()
            
        # Aggregate ALL raw scores and calculate ONE final Elo
        total_adjusted_score_sum = 0
        total_adjusted_games_sum = 0
        
        while not sf_queue.empty():
            results = sf_queue.get() 
            
            # --- Aggregation logic MUST be done in main.py ---
            # Total adjusted games = games_per_worker + 2 (pseudo-games for smoothing)
            total_adjusted_games_in_batch = SF_GAMES_PER_WORKER + 2.0
            
            adjusted_win_rate = results.get('adjusted_win_rate', 0.5)
            adjusted_score_in_batch = adjusted_win_rate * total_adjusted_games_in_batch
            
            total_adjusted_score_sum += adjusted_score_in_batch
            total_adjusted_games_sum += total_adjusted_games_in_batch
            
        # Calculate TOTAL adjusted win rate
        if total_adjusted_games_sum > 0:
            final_adjusted_win_rate = total_adjusted_score_sum / total_adjusted_games_sum
        else:
            final_adjusted_win_rate = 0.5
            
        final_elo = calculate_elo(STOCKFISH_ELO, final_adjusted_win_rate)

        print(f"--- STOCKFISH AGGREGATE RESULTS ---")
        print(f"Total Games Played vs SF: {SF_WORKERS * SF_GAMES_PER_WORKER}")
        print(f"Total Adjusted Win Rate: {final_adjusted_win_rate:.3f}")
        print(f"Estimated Agent Elo: {final_elo}")

    else:
        print("Stockfish evaluation skipped as Candidate was REJECTED.")

    logger.log(iteration, policy_loss=0.0, value_loss=0.0, arena_win_rate=final_win_rate, elo=final_elo)

if __name__ == "__main__":
    setup_child_logging()
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    logger = MetricsLogger()
    
    print("=================================================")
    print(f"STARTING AGGRESSIVE RUN")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS} | Penalty: {DRAW_PENALTY}")
    print("=================================================")

    try:
        for it in range(1, ITERATIONS + 1):
            run_self_play_phase(it)
            run_training_phase(it)
            run_evaluation_phase(it, logger)
            
    except KeyboardInterrupt:
        print("\n\n--- LOOP STOPPED BY USER ---")