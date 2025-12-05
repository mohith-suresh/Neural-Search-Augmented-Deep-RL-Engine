import multiprocessing as mp
import threading
import os
import sys
import time
import shutil
import gc
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
#        AGGRESSIVE PRODUCTION CONFIG
# ==========================================

# --- EXECUTION ---
ITERATIONS = 1000

# Self-Play Workers (Safe Zone)
NUM_WORKERS = 15
GAMES_PER_WORKER = 5        # Total Games = 15 * 5 = 75.

# --- QUALITY ---
SIMULATIONS = 1200          # Deep Training
EVAL_SIMULATIONS = 400      # Fast Evaluation

# --- EVALUATION CONFIG (VRAM SAFE) ---
EVAL_WORKERS = 5           
GAMES_PER_EVAL_WORKER = 4   
STOCKFISH_GAMES = 20        

# --- AGGRESSIVE RULES ---
# 1. Tighter Cap: 100 Plies (50 Moves). 
# If you can't win by move 50, it's garbage time. Kill it.
MAX_MOVES_PER_GAME = 100   

# 2. Draw Penalty: Skew draws negative to force risk-taking.
# Win = +1.0, Loss = -1.0, Draw = -0.2
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
    sys.stderr = sys.stdout

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
    if hasattr(os, 'sched_setaffinity'):
        try: os.sched_setaffinity(0, {worker_id})
        except: pass

    setup_child_logging()
    time.sleep(worker_id * 0.2) 
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for i in range(game_limit):
        print(f"   [Worker {worker_id}] Starting Game {i+1} (Aggressive Mode)...")
        worker = MCTSWorker(worker_id, input_queue, output_queue, simulations=SIMULATIONS)
        
        game_start = time.time()
        game = ChessGame()
        game_data = []
        
        while not game.is_over:
            if len(game.moves) >= MAX_MOVES_PER_GAME:
                print(f"   [Worker {worker_id}] Hit aggressive limit ({MAX_MOVES_PER_GAME} plies). Adjudicating Draw.")
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
        final_winner = 0.0 # 0.0 is Black, 1.0 is White in logic below
        if result == "1-0": final_winner = 1.0
        elif result == "0-1": final_winner = 0.0
        
        values = []
        for g in game_data:
            if result == "1/2-1/2": 
                # PENALTY APPLIED: Both sides get -0.2 instead of 0.0
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
            # Pin server to last 16 cores
            os.sched_setaffinity(0, {i for i in range(16, os.cpu_count())})
        except: pass
    setup_child_logging()
    monitor = threading.Thread(target=queue_monitor_thread, args=(server.input_queue,))
    monitor.daemon = True 
    monitor.start()
    server.loop()

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE ===")
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

def run_arena_batch(worker_id, result_queue, num_games):
    setup_child_logging()
    try:
        arena = Arena(CANDIDATE_MODEL, BEST_MODEL, simulations=EVAL_SIMULATIONS)
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
        print(f"   [SF Worker {worker_id}] Playing {num_games} games vs Stockfish...")
        sf_eval = StockfishEvaluator(STOCKFISH_PATH, simulations=EVAL_SIMULATIONS)
        elo = sf_eval.evaluate(BEST_MODEL, num_games=num_games, stockfish_elo=1350)
        result_queue.put(elo)
        print(f"   [SF Worker {worker_id}] Batch Complete. Elo: {elo}")
    except Exception as e:
        print(f"   [SF Worker {worker_id}] Error: {e}")
        result_queue.put(1000)

def run_evaluation_phase(iteration, logger):
    print(f"\n=== ITERATION {iteration}: PARALLEL EVALUATION PHASE ===")
    
    # --- 1. PARALLEL ARENA ---
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
    else:
        print(f"❌ REJECTED. Candidate ({final_win_rate:.2f}) failed to beat Champion.")
        if os.path.exists(CANDIDATE_MODEL):
            os.remove(CANDIDATE_MODEL)

    # --- 2. PARALLEL STOCKFISH ---
    SF_WORKERS = 4
    SF_GAMES_PER_WORKER = STOCKFISH_GAMES // SF_WORKERS
    
    print(f"Starting Stockfish Eval: {SF_WORKERS} Workers x {SF_GAMES_PER_WORKER} Games")
    sf_queue = mp.Queue()
    sf_workers = []
    
    for i in range(SF_WORKERS):
        p = mp.Process(target=run_stockfish_batch, args=(i, sf_queue, SF_GAMES_PER_WORKER))
        p.start()
        sf_workers.append(p)
        
    for p in sf_workers:
        p.join()
        
    # Aggregate Elo
    elos = []
    while not sf_queue.empty():
        elos.append(sf_queue.get())
        
    final_elo = sum(elos) / len(elos) if elos else 0
    
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
    print(f"STARTING AGGRESSIVE RUN (1 Iteration)")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS} | Penalty: {DRAW_PENALTY}")
    print("=================================================")

    try:
        for it in range(1, ITERATIONS + 1):
            run_self_play_phase(it)
            run_training_phase(it)
            run_evaluation_phase(it, logger)
            
    except KeyboardInterrupt:
        print("\n\n--- LOOP STOPPED BY USER ---")