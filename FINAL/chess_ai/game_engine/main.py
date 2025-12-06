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

# --- CUDA ---
CUDA_TIMEOUT_INFERENCE = 0.01 # Tighter timeout for responsiveness
CUDA_STREAMS = 4 

# --- EXECUTION ---
ITERATIONS = 1000
NUM_WORKERS = 50            # Reduced slightly as each worker is now 8x more demanding
WORKER_BATCH_SIZE = 8       # <--- NEW: Number of parallel paths per worker
GAMES_PER_WORKER = 2        

# --- QUALITY ---
SIMULATIONS = 800           # 800 Sims / 8 Batch = 100 IPC Calls (Very Fast)
EVAL_SIMULATIONS = 400      

# --- EVALUATION CONFIG ---
EVAL_WORKERS = 4           
GAMES_PER_EVAL_WORKER = 4
STOCKFISH_GAMES = 20
SF_WORKERS = 5              
SF_GAMES_PER_WORKER = 4     
STOCKFISH_ELO = 1350        

# --- RULES ---
MAX_MOVES_PER_GAME = 100   
DRAW_PENALTY = -0.5         # Increased penalty to discourage stalling

# Training
TRAIN_EPOCHS = 1           

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

def queue_monitor_thread(queue):
    while True:
        try:
            size = queue.qsize()
            if size > 100:
                # Only log if really high to avoid spamming
                if size > 200:
                    print(f"   [Server Monitor] High Load: {size} requests pending")
            time.sleep(2.0)
        except: break

def run_worker_batch(worker_id, input_queue, output_queue, game_limit):
    # Robust Affinity Setting
    if hasattr(os, 'sched_setaffinity'):
        try:
            core_count = os.cpu_count() or 1
            core_id = worker_id % core_count
            os.sched_setaffinity(0, {core_id})
        except: pass

    setup_child_logging()
    # Stagger start times slightly to prevent Thundering Herd on the queue
    time.sleep(worker_id * 0.05)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize Worker with Batching
    worker = MCTSWorker(worker_id, input_queue, output_queue, 
                        simulations=SIMULATIONS, 
                        batch_size=WORKER_BATCH_SIZE)
    
    for i in range(game_limit):
        print(f"   [Worker {worker_id}] Starting Game {i+1}...")
        
        game_start = time.time()
        game = ChessGame()
        game_data = []
        
        while not game.is_over:
            if len(game.moves) >= MAX_MOVES_PER_GAME:
                break 

            move_start = time.time()
            
            # Temperature Schedule
            if len(game.moves) < 30:
                current_temp = 1.0 
            else:
                current_temp = 0.1 
            
            best_move, mcts_policy = worker.search(game, temperature=current_temp)
            
            if worker_id == 0:
                dur = time.time() - move_start
                nps = SIMULATIONS / dur if dur > 0 else 0
                print(f"   [Worker 0] Move {len(game.moves)+1}: {best_move} ({dur:.2f}s | {nps:.0f} n/s)")
            
            game_data.append({
                "state": game.to_tensor(),
                "policy": mcts_policy,
                "turn": game.turn_player
            })
            game.push(best_move)
        
        # Save Game Logic
        if len(game.moves) >= MAX_MOVES_PER_GAME: result = "1/2-1/2"
        else: result = game.result

        final_winner = 0.0
        if result == "1-0": final_winner = 1.0
        elif result == "0-1": final_winner = 0.0
        elif result == "1/2-1/2": final_winner = 0.5
        
        values = []
        for g in game_data:
            if final_winner == 0.5: 
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
        
        print(f"   [Worker {worker_id}] Finished Game {i+1} in {time.time()-game_start:.1f}s")
        gc.collect()

def run_server_wrapper(server):
    setup_child_logging()
    monitor = threading.Thread(target=queue_monitor_thread, args=(server.input_queue,))
    monitor.daemon = True 
    monitor.start()
    server.loop()

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE (Batched MCTS) ===")

    server = InferenceServer(BEST_MODEL, batch_size=128, timeout=CUDA_TIMEOUT_INFERENCE, streams=CUDA_STREAMS)
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

def run_training_phase(iteration):
    print(f"\n=== ITERATION {iteration}: TRAINING PHASE ===")
    train_model(data_path=DATA_DIR, 
                input_model_path=BEST_MODEL, 
                output_model_path=CANDIDATE_MODEL,
                epochs=TRAIN_EPOCHS)

def run_evaluation_phase(iteration, logger):
    print(f"\n=== ITERATION {iteration}: EVALUATION SKIPPED FOR SPEED TEST ===")
    pass

if __name__ == "__main__":
    setup_child_logging()
    # Force 'spawn' to avoid CUDA initialization errors in forked processes
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    print("=================================================")
    print(f"STARTING BATCHED RUN")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS} | Batch: {WORKER_BATCH_SIZE}")
    print("=================================================")

    try:
        for it in range(1, ITERATIONS + 1):
            run_self_play_phase(it)
            run_training_phase(it)
            run_evaluation_phase(it, MetricsLogger())
    except KeyboardInterrupt:
        print("\n\n--- LOOP STOPPED BY USER ---")