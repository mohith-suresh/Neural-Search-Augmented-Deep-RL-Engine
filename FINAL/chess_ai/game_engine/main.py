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
import sys

class TimeoutHandler:
    """Handle process timeouts to prevent deadlocks"""
    def __init__(self, timeout_seconds=9000):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.timeout_seconds)
    
    def _timeout_handler(self, signum, frame):
        elapsed = int(time.time() - self.start_time)
        print(f"⚠️ TIMEOUT: No progress in {self.timeout_seconds}s ({elapsed}s elapsed)")
        print("⚠️ Likely deadlock detected. Shutting down gracefully...")
        sys.exit(1)
    
    def reset(self):
        """Reset timeout (call when iteration completes)"""
        if self.start_time:
            signal.alarm(self.timeout_seconds)

# Create global timeout handler (8 hours = 28800 seconds)
timeout_handler = TimeoutHandler(timeout_seconds=28800)

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.neural_net import InferenceServer
from game_engine.mcts import MCTSWorker
from game_engine.chess_env import ChessGame
from game_engine.trainer import train_model
from game_engine.evaluation import Arena, StockfishEvaluator, MetricsLogger
from game_engine.cnn import ChessCNN

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
            move_count = len(game.moves)
            if move_count < 15:
                current_temp = 1.2  # Open book exploration
            elif move_count < 40:
                current_temp = 0.6  # Middlegame focus
            elif move_count < 80:
                current_temp = 0.4  # Early endgame, still searching
            else:
                current_temp = 0.25 # Deep endgame, very focused
            
            best_move, mcts_policy = worker.search(game, temperature=current_temp)
            
            if (worker_id % 5) == 0:
                dur = time.time() - move_start
                nps = SIMULATIONS / dur if dur > 0 else 0
                print(f"   [Worker {worker_id}] Move {len(game.moves)+1}: {best_move} ({dur:.2f}s | {nps:.0f} sim/s)")
            
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
        
        print(f"   [Worker {worker_id}] Finished Game {i+1} in {time.time()-game_start:.1f}s |  Total Moves {len(game.moves)} | Result {result}")
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

# ==========================================
#        BALANCED GCP CONFIG (T4 OPTIMIZED)
# ==========================================

# --- PATHS ---
STOCKFISH_PATH = "/usr/games/stockfish" 
LOG_FILE = "training_log.txt"
MODEL_DIR = "game_engine/model"
DATA_DIR = "data/self_play"
BEST_MODEL = f"{MODEL_DIR}/best_model.pth"
CANDIDATE_MODEL = f"{MODEL_DIR}/candidate.pth"

# --- CUDA ---
CUDA_TIMEOUT_INFERENCE = 0.01 
CUDA_STREAMS = 8 

# --- EXECUTION ---
ITERATIONS = 1000
NUM_WORKERS = 42            
WORKER_BATCH_SIZE = 8       
GAMES_PER_WORKER = 5        

# --- QUALITY ---
SIMULATIONS = 1200           
EVAL_SIMULATIONS = 1200      

# --- EVALUATION CONFIG ---
EVAL_WORKERS = 10           
GAMES_PER_EVAL_WORKER = 4   
STOCKFISH_GAMES = 40
SF_WORKERS = 10              
SF_GAMES_PER_WORKER = 4     
STOCKFISH_ELO = 1320        

# --- RULES ---
MAX_MOVES_PER_GAME = 140   
EVAL_MAX_MOVES_PER_GAME = 150 
current_iter = get_start_iteration(DATA_DIR) - 1
if current_iter < 10:
    DRAW_PENALTY = -0.15
elif current_iter < 20:
    DRAW_PENALTY = -0.25
else:
    DRAW_PENALTY = -0.30        

# Training
TRAIN_EPOCHS = 2 
TRAIN_WINDOW = 30           
TRAIN_BATCH_SIZE = 1024      
TRAIN_LR = 0.00005          

# --- DRY WORKER WRAPPERS ---

def run_arena_batch_worker(worker_id, queue, num_games, cand_model, champ_model, sims, max_moves):
    setup_child_logging()
    np.random.seed(worker_id + int(time.time()) % 10000)
    torch.manual_seed(worker_id + int(time.time()) % 10000)

    try:
        arena = Arena(cand_model, champ_model, sims, max_moves)
        w, d, l, fd = arena.play_match(num_games, temperature=0.0, use_dirichlet=False)
        
        # We create a simple result string for logging
        result_str = f"Worker {worker_id}: {w}W - {d}D - {l}L"
        print(f"   [Arena] {result_str}")
        
        queue.put({"wins": w, "draws": d, "losses": l, "forced_draws": fd})
    except Exception as e:
        print(f"Arena Worker {worker_id} Failed: {e}")
        queue.put({"wins": w, "draws": d, "losses": l, "forced_draws": fd})

def run_stockfish_batch_worker(worker_id, queue, num_games, model_path, sims, sf_elo, sf_path, max_moves):
    setup_child_logging()
    np.random.seed(worker_id + int(time.time()) % 10000)
    torch.manual_seed(worker_id + int(time.time()) % 10000)

    try:
        sf_eval = StockfishEvaluator(sf_path, sims)
        # Pass max_moves here
        score, games = sf_eval.evaluate(model_path, num_games, sf_elo, max_moves, use_dirichlet=False)
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
    
    # Start queue monitor to detect stalls
    def monitor_queues(server_ref, input_q, interval=30):
        """Monitor queue sizes and detect stuck situations"""
        last_size = 0
        stall_count = 0
        
        while server_process.is_alive():
            current_size = input_q.qsize()
            
            # If queue size hasn't changed in 3 checks (90 seconds), likely stuck
            if current_size == last_size and current_size > 0:
                stall_count += 1
                if stall_count >= 3:
                    print(f"⚠️ QUEUE STALL DETECTED: {current_size} requests stuck for ~90s")
            else:
                stall_count = 0
            
            last_size = current_size
            time.sleep(interval)
    
    monitor_thread = threading.Thread(
        target=monitor_queues,
        args=(server, server.input_queue),
        daemon=True
    )
    monitor_thread.start()

    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=run_worker_batch, 
                       args=(i, server.input_queue, worker_queues[i], GAMES_PER_WORKER, iteration))
        p.start()
        workers.append(p)
        
    # NEW CODE - ACTIVE MONITORING LOOP
    try:
        while True:
            # 1. Check if Server is alive
            if not server_process.is_alive():
                print("🚨 CRITICAL: Inference Server died unexpectedly! Terminating workers...")
                raise RuntimeError("Inference Server died during self-play.")

            # 2. Check if Workers are done
            alive_workers = [p for p in workers if p.is_alive()]
            
            if not alive_workers:
                print("✅ All workers finished successfully.")
                break
                
            # 3. Sleep to prevent CPU burn
            time.sleep(2)
            
    except Exception as e:
        print(f"❌ Exception in Phase 1 Loop: {e}")
        for p in workers:
            if p.is_alive():
                p.terminate()
        if server_process.is_alive():
            server_process.terminate()
        raise e
        
    finally:
        print("🧹 Cleaning up Phase 1 processes...")
        
        # Kill any straggler workers
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        
        # Stop Server
        if server_process.is_alive():
            server.input_queue.put("STOP")
            server_process.join(timeout=10)
            if server_process.is_alive():
                print("⚠️ Server did not stop gracefully. Force killing...")
                server_process.terminate()
                server_process.join(timeout=5)
                if server_process.is_alive():
                    server_process.kill()
                    server_process.join()

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
            args=(i, arena_queue, GAMES_PER_EVAL_WORKER, CANDIDATE_MODEL, BEST_MODEL, EVAL_SIMULATIONS, EVAL_MAX_MOVES_PER_GAME)
        )
        p.start()
        arena_workers.append(p)

    # --- ACTIVE MONITORING FOR ARENA ---
    try:
        while True:
            alive = [p for p in arena_workers if p.is_alive()]
            if not alive:
                print("✅ All arena workers finished.")
                break
            time.sleep(1)
    except Exception as e:
        print(f"❌ Arena evaluation failed: {e}")
        for p in arena_workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        raise

    
    # Collect Arena Results
    total_wins, total_draws, total_losses, total_forced_draws = 0, 0, 0, 0
    while not arena_queue.empty():
        res = arena_queue.get()
        total_wins += res['wins']
        total_draws += res['draws']
        total_losses += res['losses']
        total_forced_draws += res['forced_draws']
    
    total_score = total_wins + 0.75 * total_forced_draws + 0.5 * total_draws
    total_game_count = total_wins + total_draws + total_forced_draws + total_losses
    win_rate = total_score / total_game_count if total_game_count > 0 else 0
    
    print(f" [Arena] Final Result: {win_rate*100:.1f}% Win Rate ({total_wins}W - {total_draws}D - {total_forced_draws}FD - {total_losses}L)")
    
    est_elo = None

    # 2. PROMOTION LOGIC
    if win_rate >= 0.55:
        print(f" [Arena] ⭐ Candidate PROMOTED! (WR > 55%) ⭐")
        shutil.copyfile(CANDIDATE_MODEL, BEST_MODEL)
        
        # 3. STOCKFISH EVALUATION (only if promoted)
        print(f" [Stockfish] Playing {SF_WORKERS * SF_GAMES_PER_WORKER} games vs Elo {STOCKFISH_ELO}...")
        cleanup_memory() # Clear again before Stockfish
        sf_queue = ctx.Queue()
        sf_workers = []

        for i in range(SF_WORKERS):
            p = ctx.Process(
                target=run_stockfish_batch_worker,
                args=(i, sf_queue, SF_GAMES_PER_WORKER, BEST_MODEL, EVAL_SIMULATIONS, STOCKFISH_ELO, STOCKFISH_PATH, EVAL_MAX_MOVES_PER_GAME)
            )
            p.start()
            sf_workers.append(p)

        # --- ACTIVE MONITORING FOR STOCKFISH ---
        try:
            while True:
                alive = [p for p in sf_workers if p.is_alive()]
                if not alive:
                    print("✅ All Stockfish workers finished.")
                    break
                time.sleep(1)
        except Exception as e:
            print(f"❌ Stockfish evaluation failed: {e}")
            for p in sf_workers:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)
            raise
        
        # Collect Stockfish Results
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
        print(f" [Arena] Candidate rejected (WR <= 55%). Skipping Stockfish evaluation.")
    
    logger.log(iteration, p_loss, v_loss, win_rate, est_elo)

if __name__ == "__main__":
    setup_child_logging()
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    timeout_handler.start()
    print("⏱️ Deadlock timeout: 5 hour per iteration")

    # RESUMPTION LOGIC
    start_iter = get_start_iteration(DATA_DIR)
    
    print("=" * 60)
    print(f"STARTING RUN")
    print(f"Resuming from Iteration: {start_iter}")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS} | Batch: {WORKER_BATCH_SIZE}")
    print("=" * 60)

    killer = GracefulKiller()
    
    try:
        for it in range(start_iter, ITERATIONS + 1):
            
            # CHECK BEFORE ITERATION STARTS
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received BEFORE iteration start")
                print(f"[Main] Gracefully exiting. Next run will resume from Iteration {it}")
                break
            
            iter_start = time.time()
            
            # === PHASE 1: SELF-PLAY ===
            print(f"\n{'='*60}")
            print(f"ITERATION {it} - PHASE 1: SELF-PLAY")
            print(f"{'='*60}")
            
            try:
                run_self_play_phase(it)
                print(f"\n✅ ITERATION {it} - PHASE 1 COMPLETE")
            except Exception as e:
                print(f"\n❌ ITERATION {it} - PHASE 1 FAILED: {e}")
                if killer.kill_now:
                    print("[Main] Kill signal during Phase 1. Exiting...")
                    break
                raise
            
            # CHECK AFTER PHASE 1
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received AFTER Phase 1")
                print("[Main] Saving state and exiting. Training/Eval will run on next startup.")
                break
            
            # === PHASE 2: TRAINING ===
            print(f"\n{'='*60}")
            print(f"ITERATION {it} - PHASE 2: TRAINING")
            print(f"{'='*60}")
            
            try:
                p_loss, v_loss = run_training_phase(it)
                print(f"\n✅ ITERATION {it} - PHASE 2 COMPLETE (Policy Loss: {p_loss:.4f}, Value Loss: {v_loss:.4f})")
            except Exception as e:
                print(f"\n❌ ITERATION {it} - PHASE 2 FAILED: {e}")
                if killer.kill_now:
                    print("[Main] Kill signal during Phase 2. Exiting...")
                    break
                raise
            
            # CHECK AFTER PHASE 2
            if killer.kill_now:
                print("\n[Main] ⚠️  Kill signal received AFTER Phase 2")
                print("[Main] Saving state and exiting. Eval will run on next startup.")
                break
            
            # === PHASE 3: EVALUATION ===
            print(f"\n{'='*60}")
            print(f"ITERATION {it} - PHASE 3: EVALUATION")
            print(f"{'='*60}")
            
            try:
                run_evaluation_phase(it, MetricsLogger(), p_loss, v_loss)
                print(f"\n✅ ITERATION {it} - PHASE 3 COMPLETE")
            except Exception as e:
                print(f"\n❌ ITERATION {it} - PHASE 3 FAILED: {e}")
                if killer.kill_now:
                    print("[Main] Kill signal during Phase 3. Exiting...")
                    break
                raise
            
            # === ITERATION COMPLETE ===
            iter_end = time.time()
            elapsed = iter_end - iter_start
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"\n{'='*60}")
            if hours > 0:
                print(f"✅ ITERATION {it} COMPLETE: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            else:
                print(f"✅ ITERATION {it} COMPLETE: {int(minutes)}m {seconds:.2f}s")
            print(f"{'='*60}\n")
            
            timeout_handler.reset()
            print(f"✅ Iteration {it} completed - timeout reset")
            
            # CHECK BEFORE NEXT ITERATION
            if killer.kill_now:
                print("[Main] ⚠️  Kill signal received. Exiting gracefully...")
                break
    
    except KeyboardInterrupt:
        print("\n\n[Main] ❌ USER INTERRUPTED - EXITING")
    
    except Exception as e:
        print(f"\n\n[Main] ❌ FATAL ERROR: {e}")
        raise
    
    finally:
        print("\n[Main] Cleanup: Closing threads and processes...")
        cleanup_memory()
        print("[Main] ✅ Shutdown complete")
