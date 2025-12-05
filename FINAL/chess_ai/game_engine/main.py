import multiprocessing as mp
import threading
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
#        SAFE PRODUCTION CONFIG (MONITORED)
# ==========================================

# --- EXECUTION ---
ITERATIONS = 1000

# 15 Workers is the "Safe Zone" for your RAM.
NUM_WORKERS = 15

# Generation
GAMES_PER_WORKER = 5       

# --- QUALITY ---
SIMULATIONS = 1200          

# Training
TRAIN_EPOCHS = 2           

# Evaluation
EVAL_GAMES = 20            
STOCKFISH_GAMES = 10       

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
        self.log = open(LOG_FILE, "a", buffering=1, encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except Exception:
            pass 

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except Exception:
            pass

def setup_child_logging():
    sys.stdout = Logger()
    sys.stderr = sys.stdout

# ==========================================

def queue_monitor_thread(queue):
    """
    Runs in the background of the Server process.
    Checks how many requests are waiting for the GPU.
    """
    while True:
        try:
            # qsize() is approximate but good enough for monitoring
            size = queue.qsize()
            if size > 0:
                # Only print if there is actually traffic, to reduce log spam
                print(f"   [Server Monitor] Pending Requests in Queue: {size}")
            time.sleep(2.0)
        except NotImplementedError:
            print("   [Server Monitor] Queue size not supported on this OS.")
            break
        except Exception:
            break

def run_worker_batch(worker_id, input_queue, output_queue, game_limit):
    # --- CPU AFFINITY (PINNING) ---
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, {worker_id})
        except Exception:
            pass

    setup_child_logging()
    time.sleep(worker_id * 0.2) 

    print(f"   [Worker {worker_id}] Starting batch of {game_limit} HQ games ({SIMULATIONS} sims)...")
    worker = MCTSWorker(worker_id, input_queue, output_queue, simulations=SIMULATIONS)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for i in range(game_limit):
        game_start = time.time()
        game = ChessGame()
        game_data = []
        
        while not game.is_over:
            move_start = time.time()
            
            # Temperature Schedule
            current_temp = 1.0 if len(game.moves) < 30 else 0.1
            
            best_move, mcts_policy = worker.search(game, temperature=current_temp)
            move_duration = time.time() - move_start

            if worker_id == 0:
                print(f"   [Worker 0] Move {len(game.moves)+1}: {best_move} ({move_duration:.2f}s)")
            
            game_data.append({
                "state": game.to_tensor(),
                "policy": mcts_policy,
                "turn": game.turn_player
            })
            game.push(best_move)
        
        # Save Game
        result = game.result
        final_winner = 0.0
        if result == "1-0": final_winner = 1.0
        elif result == "0-1": final_winner = 0.0
        
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
        if worker_id < 5:
            print(f"   [Worker {worker_id}] Finished Game {i+1}/{game_limit} ({len(game.moves)} moves) in {duration:.1f}s")
        
    print(f"   [Worker {worker_id}] Batch Complete.")

def run_server_wrapper(server):
    # --- PIN SERVER ---
    if hasattr(os, 'sched_setaffinity'):
        try:
            total_cores = os.cpu_count()
            server_cores = {i for i in range(16, total_cores)}
            if server_cores:
                os.sched_setaffinity(0, server_cores)
        except Exception:
            pass

    setup_child_logging()
    
    # --- START MONITOR THREAD ---
    monitor = threading.Thread(target=queue_monitor_thread, args=(server.input_queue,))
    monitor.daemon = True # Dies when main process dies
    monitor.start()

    server.loop()

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE (Safe & Monitored) ===")
    
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
        
    for p in workers:
        p.join()
        
    server.input_queue.put("STOP")
    server_process.join(timeout=10)
    if server_process.is_alive():
        server_process.terminate()
        
    print(f"=== SELF-PLAY COMPLETE: Generated {NUM_WORKERS * GAMES_PER_WORKER} games ===")

def run_training_phase(iteration):
    print(f"\n=== ITERATION {iteration}: TRAINING PHASE ===")
    train_model(data_path=DATA_DIR, 
                input_model_path=BEST_MODEL, 
                output_model_path=CANDIDATE_MODEL,
                epochs=TRAIN_EPOCHS)

def run_evaluation_phase(iteration, logger):
    print(f"\n=== ITERATION {iteration}: EVALUATION PHASE ===")
    
    arena = Arena(CANDIDATE_MODEL, BEST_MODEL, simulations=SIMULATIONS)
    win_rate = arena.play_match(num_games=EVAL_GAMES)
    
    if win_rate >= 0.55:
        print(f"🚀 PROMOTION! Candidate ({win_rate:.2f}) defeated Champion.")
        shutil.move(CANDIDATE_MODEL, BEST_MODEL)
    else:
        print(f"❌ REJECTED. Candidate ({win_rate:.2f}) failed to beat Champion.")
        if os.path.exists(CANDIDATE_MODEL):
            os.remove(CANDIDATE_MODEL)
            
    sf_eval = StockfishEvaluator(STOCKFISH_PATH, simulations=SIMULATIONS)
    elo = sf_eval.evaluate(BEST_MODEL, num_games=STOCKFISH_GAMES, stockfish_elo=1350) 
    
    logger.log(iteration, policy_loss=0.0, value_loss=0.0, arena_win_rate=win_rate, elo=elo)

if __name__ == "__main__":
    setup_child_logging()
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    logger = MetricsLogger()
    
    print("=================================================")
    print(f"STARTING MONITORED RUN (1 Iteration)")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS}")
    print("=================================================")

    try:
        for it in range(1, ITERATIONS + 1):
            run_self_play_phase(it)
            run_training_phase(it)
            run_evaluation_phase(it, logger)
            
    except KeyboardInterrupt:
        print("\n\n--- LOOP STOPPED BY USER ---")