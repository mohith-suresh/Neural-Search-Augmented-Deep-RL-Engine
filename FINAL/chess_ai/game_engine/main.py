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

# Ensure project root is in path
sys.path.append(os.getcwd())

from game_engine.neural_net import InferenceServer
from game_engine.mcts import MCTSWorker
from game_engine.chess_env import ChessGame
from game_engine.trainer import train_model
from game_engine.evaluation import Arena, StockfishEvaluator, MetricsLogger
from game_engine.cnn import ChessCNN

# ==========================================
#        HIGH-DENSITY GCP CONFIG
# ==========================================

# --- CUDA ---
CUDA_TIMEOUT_INFERENCE = 0.01 
CUDA_STREAMS = 4 

# --- EXECUTION ---
ITERATIONS = 1000
NUM_WORKERS = 75            
WORKER_BATCH_SIZE = 8       # Keep at 8 to allow fast context switching on shared cores
GAMES_PER_WORKER = 2        

# --- QUALITY ---
SIMULATIONS = 800           
EVAL_SIMULATIONS = 400      

# --- EVALUATION CONFIG ---
EVAL_WORKERS = 4           
GAMES_PER_EVAL_WORKER = 4   
STOCKFISH_GAMES = 20
SF_WORKERS = 5              
SF_GAMES_PER_WORKER = 4     
STOCKFISH_ELO = 1350        

# --- RULES ---
MAX_MOVES_PER_GAME = 250   
DRAW_PENALTY = -0.25        

# Training
TRAIN_EPOCHS = 1 
TRAIN_WINDOW = 20           
TRAIN_BATCH_SIZE = 256      
TRAIN_LR = 0.0001           

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
                # With 75 workers, the queue will naturally sit higher
                if size > 500: 
                    print(f"   [Server Monitor] High Load: {size} requests pending")
            time.sleep(2.0)
        except: break

def run_worker_batch(worker_id, input_queue, output_queue, game_limit):
    # --- ROUND ROBIN PINNING (46 CORES) ---
    # We pin workers 0-74 onto cores 0-45.
    # Worker 0 -> Core 0
    # Worker 45 -> Core 45
    # Worker 46 -> Core 0 (Sharing with Worker 0)
    if hasattr(os, 'sched_setaffinity'):
        try:
            core_id = worker_id % 46
            os.sched_setaffinity(0, {core_id})
        except: pass

    setup_child_logging()
    # Increased stagger to prevent 75 processes hitting the queue instantly
    time.sleep(worker_id * 0.05)
    os.makedirs(DATA_DIR, exist_ok=True)
    
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
    # Pin server to the reserved cores (46 and 47) to ensure responsiveness
    if hasattr(os, 'sched_setaffinity'):
        try: os.sched_setaffinity(0, {46, 47})
        except: pass
        
    monitor = threading.Thread(target=queue_monitor_thread, args=(server.input_queue,))
    monitor.daemon = True 
    monitor.start()
    server.loop()

# --- EVALUATION WORKERS ---

def run_arena_worker(worker_id, queue, num_games):
    setup_child_logging()
    from game_engine.evaluation import EvalMCTS
    candidate = EvalMCTS(CANDIDATE_MODEL, simulations=EVAL_SIMULATIONS)
    champion = EvalMCTS(BEST_MODEL, simulations=EVAL_SIMULATIONS)
    wins, draws, losses = 0, 0, 0
    for i in range(num_games):
        game = ChessGame()
        cand_is_white = (i % 2 == 0)
        while not game.is_over and len(game.moves) < MAX_MOVES_PER_GAME:
            if game.board.turn == chess.WHITE:
                move = candidate.search(game) if cand_is_white else champion.search(game)
            else:
                move = champion.search(game) if cand_is_white else candidate.search(game)
            game.push(move)
        res = game.result
        if res == "1-0":
            if cand_is_white: wins += 1
            else: losses += 1
        elif res == "0-1":
            if cand_is_white: losses += 1
            else: wins += 1
        else: draws += 1
    queue.put({"wins": wins, "draws": draws, "losses": losses, "games": num_games})

def run_stockfish_worker(worker_id, queue, num_games):
    setup_child_logging()
    from game_engine.evaluation import EvalMCTS
    import chess.engine
    agent = EvalMCTS(CANDIDATE_MODEL, simulations=EVAL_SIMULATIONS)
    score = 0.0
    if not os.path.exists(STOCKFISH_PATH):
        queue.put({"score": 0, "games": 0})
        return
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})
    except:
        queue.put({"score": 0, "games": 0})
        return
    for i in range(num_games):
        game = ChessGame()
        agent_is_white = (i % 2 == 0)
        while not game.is_over and len(game.moves) < MAX_MOVES_PER_GAME:
            is_agent_turn = (game.board.turn == chess.WHITE and agent_is_white) or \
                            (game.board.turn == chess.BLACK and not agent_is_white)
            if is_agent_turn: move = agent.search(game)
            else:
                try:
                    result = engine.play(game.board, chess.engine.Limit(time=0.05))
                    move = result.move.uci()
                except: break
            game.push(move)
        res = game.result
        if res == "1-0": score += 1.0 if agent_is_white else 0.0
        elif res == "0-1": score += 0.0 if agent_is_white else 1.0
        else: score += 0.5
    engine.quit()
    queue.put({"score": score, "games": num_games})

# --- PHASES ---

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE (Batched MCTS) ===")
    # Server Batch Size 1024 to accommodate 75 workers * 8 items = 600 potential items
    server = InferenceServer(BEST_MODEL, batch_size=1024, timeout=CUDA_TIMEOUT_INFERENCE, streams=CUDA_STREAMS)
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
                epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
                lr=TRAIN_LR,
                window_size=TRAIN_WINDOW)

def run_evaluation_phase(iteration, logger):
    print(f"\n=== ITERATION {iteration}: EVALUATION PHASE ===")
    ctx = mp.get_context('spawn')
    
    print(f"   [Arena] Playing {EVAL_WORKERS * GAMES_PER_EVAL_WORKER} games (Candidate vs Best)...")
    arena_queue = ctx.Queue()
    workers = []
    for i in range(EVAL_WORKERS):
        p = ctx.Process(target=run_arena_worker, args=(i, arena_queue, GAMES_PER_EVAL_WORKER))
        p.start()
        workers.append(p)
    for p in workers: p.join()
        
    total_wins, total_draws, total_losses = 0, 0, 0
    while not arena_queue.empty():
        res = arena_queue.get()
        total_wins += res['wins']
        total_draws += res['draws']
        total_losses += res['losses']
        
    total_games = total_wins + total_draws + total_losses
    win_rate = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0
    
    print(f"   [Arena] Result: {win_rate*100:.1f}% Win Rate ({total_wins}W - {total_draws}D - {total_losses}L)")
    est_elo = None

    if win_rate >= 0.55:
        print(f"   [Arena] \u2b50 Candidate PROMOTED! (WR > 55%) \u2b50")
        shutil.copyfile(CANDIDATE_MODEL, BEST_MODEL)
        
        print(f"   [Stockfish] Playing {SF_WORKERS * SF_GAMES_PER_WORKER} games vs Elo {STOCKFISH_ELO}...")
        sf_queue = ctx.Queue()
        sf_workers = []
        for i in range(SF_WORKERS):
            p = ctx.Process(target=run_stockfish_worker, args=(i, sf_queue, SF_GAMES_PER_WORKER))
            p.start()
            sf_workers.append(p)
        for p in sf_workers: p.join()
            
        sf_score, sf_total = 0, 0
        while not sf_queue.empty():
            res = sf_queue.get()
            sf_score += res['score']
            sf_total += res['games']
            
        sf_wr = sf_score / sf_total if sf_total > 0 else 0
        safe_wr = max(0.01, min(0.99, sf_wr))
        est_elo = STOCKFISH_ELO - 400 * math.log10(1/safe_wr - 1)
        print(f"   [Stockfish] Score: {sf_score}/{sf_total} ({sf_wr*100:.1f}%) | Est. Elo: {est_elo:.0f}")
    else:
        print(f"   [Arena] Candidate rejected. Skipping Stockfish evaluation.")
    
    logger.log(iteration, 0.0, 0.0, win_rate, est_elo)

if __name__ == "__main__":
    setup_child_logging()
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    print("=================================================")
    print(f"STARTING RUN")
    print(f"Workers: {NUM_WORKERS} | Sims: {SIMULATIONS} | Batch: {WORKER_BATCH_SIZE}")
    print("=================================================")

    try:
        for it in range(1, ITERATIONS + 1):
            run_self_play_phase(it)
            run_training_phase(it)
            run_evaluation_phase(it, MetricsLogger())
    except KeyboardInterrupt:
        print("\n\n--- LOOP STOPPED BY USER ---")