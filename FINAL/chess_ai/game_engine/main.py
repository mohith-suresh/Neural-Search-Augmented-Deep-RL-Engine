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
import chess 
import signal 

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.neural_net import InferenceServer
from game_engine.mcts import MCTSWorker
from game_engine.chess_env import ChessGame
from game_engine.trainer import train_model
from game_engine.evaluation import Arena, StockfishEvaluator, MetricsLogger
from game_engine.cnn import ChessCNN

# ==========================================
#        BALANCED GCP CONFIG (T4 OPTIMIZED)
# ==========================================

# --- CUDA ---
CUDA_TIMEOUT_INFERENCE = 0.01 
CUDA_STREAMS = 4 

# --- EXECUTION ---
ITERATIONS = 1000
NUM_WORKERS = 42            
WORKER_BATCH_SIZE = 8       
GAMES_PER_WORKER = 5        

# --- QUALITY ---
SIMULATIONS = 800           
EVAL_SIMULATIONS = 800      

# --- EVALUATION CONFIG ---
EVAL_WORKERS = 5           
GAMES_PER_EVAL_WORKER = 4   
STOCKFISH_GAMES = 20
SF_WORKERS = 5              
SF_GAMES_PER_WORKER = 4     
STOCKFISH_ELO = 1350        

# --- RULES ---
MAX_MOVES_PER_GAME = 120   
DRAW_PENALTY = -0.35        

# Training
TRAIN_EPOCHS = 1 
TRAIN_WINDOW = 20           
TRAIN_BATCH_SIZE = 1024      
TRAIN_LR = 0.0001           

# --- PATHS ---
STOCKFISH_PATH = "/usr/games/stockfish" 
LOG_FILE = "training_log.txt"
MODEL_DIR = "game_engine/model"
DATA_DIR = "data/self_play"
BEST_MODEL = f"{MODEL_DIR}/best_model.pth"
CANDIDATE_MODEL = f"{MODEL_DIR}/candidate.pth"

# ==========================================
class GracefulKiller:
    """Catches SIGTERM/SIGINT signals for graceful cloud shutdown"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n\n[Cloud Run] Received termination signal. Finishing current step...")
        self.kill_now = True

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
                print(f"   [Server Monitor] High Load: {size} requests pending")
            time.sleep(2.0)
        except: break

def get_start_iteration(data_dir):
    if not os.path.exists(data_dir):
        return 1
    
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("iter_")]
    if not subdirs:
        return 1
        
    try:
        nums = [int(d.split("_")[1]) for d in subdirs]
        return max(nums) + 1
    except ValueError:
        return 1

def cleanup_memory():
    """Forces garbage collection and clears CUDA cache to prevent OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_worker_batch(worker_id, input_queue, output_queue, game_limit, iteration):
    np.random.seed(int(time.time()) + worker_id)
    if hasattr(os, 'sched_setaffinity'):
        try: os.sched_setaffinity(0, {worker_id})
        except: pass

    setup_child_logging()
    time.sleep(worker_id * 0.05)
    
    iter_dir = os.path.join(DATA_DIR, f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    
    worker = MCTSWorker(worker_id, input_queue, output_queue, 
                        simulations=SIMULATIONS, 
                        batch_size=WORKER_BATCH_SIZE)
    
    for i in range(game_limit):
        print(f"   [Worker {worker_id}] Starting Game {i+1}...")
        game_start = time.time()
        game = ChessGame()
        game_data = []
        
        forced_draw = False
        while not game.is_over:
            if len(game.moves) >= MAX_MOVES_PER_GAME: 
                forced_draw = True
                break 

            move_start = time.time()
            if len(game.moves) < 40: current_temp = 1.0 
            else: current_temp = 0.1 
            
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
        
        # LOG FORCED DRAWS
        if forced_draw:
            print(f"   [Worker {worker_id}] Game {i+1} ended in FORCED DRAW (Max moves {MAX_MOVES_PER_GAME})")
            result = "1/2-1/2"
        else:
            result = game.result

        final_winner = 0.0
        if result == "1-0": final_winner = 1.0
        elif result == "0-1": final_winner = 0.0
        elif result == "1/2-1/2": final_winner = 0.5
        
        values = []
        for g in game_data:
            if final_winner == 0.5: values.append(DRAW_PENALTY)
            elif g["turn"] == final_winner: values.append(1.0)
            else: values.append(-1.0)
            
        timestamp = int(time.time())
        filename = f"{iter_dir}/w{worker_id}_g{i}_{timestamp}.npz"
        np.savez_compressed(filename, 
                            states=np.array([g["state"] for g in game_data]), 
                            policies=np.array([g["policy"] for g in game_data]), 
                            values=np.array(values, dtype=np.float32))
        
        print(f"   [Worker {worker_id}] Finished Game {i+1} in {time.time()-game_start:.1f}s")
        gc.collect()

def run_server_wrapper(server):
    setup_child_logging()
    if hasattr(os, 'sched_setaffinity'):
        try: os.sched_setaffinity(0, {44, 45, 46, 47})
        except: pass
        
    monitor = threading.Thread(target=queue_monitor_thread, args=(server.input_queue,))
    monitor.daemon = True 
    monitor.start()
    server.loop()

# --- DRY WORKER WRAPPERS ---

def run_arena_batch_worker(worker_id, queue, num_games, cand_model, champ_model, sims, max_moves):
    setup_child_logging()
    try:
        arena = Arena(cand_model, champ_model, sims, max_moves)
        w, d, l = arena.play_match(num_games)
        
        # We create a simple result string for logging
        result_str = f"Worker {worker_id}: {w}W - {d}D - {l}L"
        print(f"   [Arena] {result_str}")
        
        queue.put({"wins": w, "draws": d, "losses": l})
    except Exception as e:
        print(f"Arena Worker {worker_id} Failed: {e}")
        queue.put({"wins": 0, "draws": 0, "losses": 0})

def run_stockfish_batch_worker(worker_id, queue, num_games, model_path, sims, sf_elo, sf_path, max_moves):
    setup_child_logging()
    try:
        sf_eval = StockfishEvaluator(sf_path, sims)
        # Pass max_moves here
        score, games = sf_eval.evaluate(model_path, num_games, sf_elo, max_moves)
        queue.put({"score": score, "games": games})
    except Exception as e:
        print(f"SF Worker {worker_id} Failed: {e}")
        queue.put({"score": 0, "games": 0})

# --- PHASES ---

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE (Batched MCTS) ===")
    cleanup_memory() # Clear RAM before starting
    
    server = InferenceServer(BEST_MODEL, batch_size=1024, timeout=CUDA_TIMEOUT_INFERENCE, streams=CUDA_STREAMS)
    worker_queues = [server.register_worker(i) for i in range(NUM_WORKERS)]
    
    server_process = mp.Process(target=run_server_wrapper, args=(server,))
    server_process.start()
    time.sleep(5) 
    
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=run_worker_batch, 
                       args=(i, server.input_queue, worker_queues[i], GAMES_PER_WORKER, iteration))
        p.start()
        workers.append(p)
        
    for p in workers: p.join()
        
    server.input_queue.put("STOP")
    server_process.join(timeout=10)
    if server_process.is_alive(): server_process.terminate()

def run_training_phase(iteration):
    print(f"\n=== ITERATION {iteration}: TRAINING PHASE ===")
    cleanup_memory() # Clear VRAM before training
    
    p_loss, v_loss = train_model(data_path=DATA_DIR, 
                input_model_path=BEST_MODEL, 
                output_model_path=CANDIDATE_MODEL,
                epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
                lr=TRAIN_LR,
                window_size=TRAIN_WINDOW)
    
    return p_loss, v_loss

def run_evaluation_phase(iteration, logger, p_loss, v_loss):
    print(f"\n=== ITERATION {iteration}: EVALUATION PHASE ===")
    cleanup_memory() # Clear VRAM before launching multiple evaluation workers
    
    # 1. ARENA EVALUATION
    print(f" [Arena] Playing {EVAL_WORKERS * GAMES_PER_EVAL_WORKER} games...")
    ctx = mp.get_context('spawn')
    arena_queue = ctx.Queue()
    arena_workers = []
    
    for i in range(EVAL_WORKERS):
        p = ctx.Process(
            target=run_arena_batch_worker,
            args=(i, arena_queue, GAMES_PER_EVAL_WORKER, CANDIDATE_MODEL, BEST_MODEL, EVAL_SIMULATIONS, MAX_MOVES_PER_GAME)
        )
        p.start()
        arena_workers.append(p)
    
    for p in arena_workers: p.join()
    
    total_wins, total_draws, total_losses = 0, 0, 0
    while not arena_queue.empty():
        res = arena_queue.get()
        total_wins += res['wins']
        total_draws += res['draws']
        total_losses += res['losses']
    
    total_games = total_wins + total_draws + total_losses
    win_rate = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0
    
    print(f" [Arena] Final Result: {win_rate*100:.1f}% Win Rate ({total_wins}W - {total_draws}D - {total_losses}L)")
    
    est_elo = None
    
    # 2. PROMOTION LOGIC
    if win_rate >= 0.55:
        print(f" [Arena] \u2b50 Candidate PROMOTED! (WR > 55%) \u2b50")
        shutil.copyfile(CANDIDATE_MODEL, BEST_MODEL)
        
        # 3. STOCKFISH EVALUATION
        print(f" [Stockfish] Playing {SF_WORKERS * SF_GAMES_PER_WORKER} games vs Elo {STOCKFISH_ELO}...")
        cleanup_memory() # Clear again before Stockfish
        sf_queue = ctx.Queue()
        sf_workers = []
        
        for i in range(SF_WORKERS):
            p = ctx.Process(
                target=run_stockfish_batch_worker,
                args=(i, sf_queue, SF_GAMES_PER_WORKER, BEST_MODEL, EVAL_SIMULATIONS, STOCKFISH_ELO, STOCKFISH_PATH, MAX_MOVES_PER_GAME)
            )
            p.start()
            sf_workers.append(p)
        
        for p in sf_workers: p.join()
        
        total_sf_score = 0.0
        total_sf_games = 0
        while not sf_queue.empty():
            res = sf_queue.get()
            total_sf_score += res['score']
            total_sf_games += res['games']
        
        if total_sf_games > 0:
            sf_wr = total_sf_score / total_sf_games
            safe_wr = max(0.01, min(0.99, sf_wr))
            est_elo = STOCKFISH_ELO - 400 * math.log10(1/safe_wr - 1)
            print(f" [Stockfish] Score: {total_sf_score}/{total_sf_games} ({sf_wr*100:.1f}%) | Est. Elo: {est_elo:.0f}")
    
    else:
        print(f" [Arena] Candidate rejected. Skipping Stockfish evaluation.")
    
    logger.log(iteration, p_loss, v_loss, win_rate, est_elo)

if __name__ == "__main__":
    setup_child_logging()
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    # RESUMPTION LOGIC
    start_iter = get_start_iteration(DATA_DIR)
    
    print("=================================================")
    print(f"STARTING RUN")
    print(f"Resuming from Iteration: {start_iter}")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS} | Batch: {WORKER_BATCH_SIZE}")
    print("=================================================")

    killer = GracefulKiller()
    
    try:
        for it in range(start_iter, ITERATIONS + 1):
            if killer.kill_now:
                print("Graceful exit detected. Stopping loop.")
                break
                
            run_self_play_phase(it)
            p_loss, v_loss = run_training_phase(it)
            run_evaluation_phase(it, MetricsLogger(), p_loss, v_loss)
            
    except KeyboardInterrupt:
        print("\n\n--- LOOP STOPPED BY USER ---")