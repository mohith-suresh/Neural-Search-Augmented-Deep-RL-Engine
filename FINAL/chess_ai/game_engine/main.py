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
from FINAL.chess_ai.game_engine.cnn_old import ChessCNN

# --- CONFIGURATION ---
ITERATIONS = 2             # Run 2 loops to test model reloading
NUM_WORKERS = 4            # Keep using both CPUs
GAMES_PER_WORKER = 5       # 4 Workers * 5 Games = 20 Games/Iter. Total = 40 Games.
TRAIN_EPOCHS = 3           # Train a bit more to test optimizer stability
EVAL_GAMES = 6             # Balanced evaluation (3 white, 3 black)
STOCKFISH_GAMES = 5        # Quick stockfish check

# --- PATHS ---
STOCKFISH_PATH = "/usr/games/stockfish" # CHECK THIS PATH (run 'which stockfish')

MODEL_DIR = "game_engine/model"
DATA_DIR = "data/self_play"
BEST_MODEL = f"{MODEL_DIR}/best_model.pth"
CANDIDATE_MODEL = f"{MODEL_DIR}/candidate.pth"

def run_worker_batch(worker_id, input_queue, output_queue, game_limit):
    """
    Generates a fixed number of games and then exits.
    """
    print(f"   [Worker {worker_id}] Starting batch of {game_limit} games...")
    worker = MCTSWorker(worker_id, input_queue, output_queue, simulations=50)
    
    # Use existing noise/temperature logic in MCTSWorker
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for i in range(game_limit):
        game = ChessGame()
        game_data = []
        
        while not game.is_over:
            # High Exploration for Self-Play (temperature=1.0 is default in search)
            best_move, mcts_policy = worker.search(game, temperature=1.0)
            
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
        
    print(f"   [Worker {worker_id}] Finished batch.")

def run_self_play_phase(iteration):
    print(f"\n=== ITERATION {iteration}: SELF-PLAY PHASE ===")
    
    # 1. Start Server
    server = InferenceServer(BEST_MODEL)
    worker_queues = [server.register_worker(i) for i in range(NUM_WORKERS)]
    
    server_process = mp.Process(target=server.loop)
    server_process.start()
    time.sleep(3) # Wait for model load
    
    # 2. Start Workers
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=run_worker_batch, 
                       args=(i, server.input_queue, worker_queues[i], GAMES_PER_WORKER))
        p.start()
        workers.append(p)
        
    # 3. Wait for completion
    for p in workers:
        p.join()
        
    # 4. Shutdown Server
    server.input_queue.put("STOP")
    server_process.join(timeout=5)
    if server_process.is_alive():
        server_process.terminate()
        
    print(f"=== SELF-PLAY COMPLETE: Generated {NUM_WORKERS * GAMES_PER_WORKER} games ===")

def run_training_phase(iteration):
    print(f"\n=== ITERATION {iteration}: TRAINING PHASE ===")
    # Train Candidate
    train_model(data_path=DATA_DIR, 
                input_model_path=BEST_MODEL, 
                output_model_path=CANDIDATE_MODEL,
                epochs=TRAIN_EPOCHS)

def run_evaluation_phase(iteration, logger):
    print(f"\n=== ITERATION {iteration}: EVALUATION PHASE ===")
    
    # 1. Arena: Candidate vs Champion
    arena = Arena(CANDIDATE_MODEL, BEST_MODEL, simulations=50)
    win_rate = arena.play_match(num_games=EVAL_GAMES)
    
    # 2. Promotion Logic
    if win_rate >= 0.55:
        print(f"🚀 PROMOTION! Candidate ({win_rate:.2f}) defeated Champion.")
        shutil.move(CANDIDATE_MODEL, BEST_MODEL)
    else:
        print(f"❌ REJECTED. Candidate ({win_rate:.2f}) failed to beat Champion.")
        if os.path.exists(CANDIDATE_MODEL):
            os.remove(CANDIDATE_MODEL)
            
    # 3. Stockfish Benchmark (Track true progress)
    # We test the BEST model (whether it changed or not)
    sf_eval = StockfishEvaluator(STOCKFISH_PATH, simulations=50)
    elo = sf_eval.evaluate(BEST_MODEL, num_games=STOCKFISH_GAMES, stockfish_elo=1350) # Start low
    
    # 4. Log Metrics
    # (Note: We pass dummy loss values here as we didn't extract them from train_model return 
    #  To fix this, train_model should return loss stats, but for now we log 0.0)
    logger.log(iteration, policy_loss=0.0, value_loss=0.0, arena_win_rate=win_rate, elo=elo)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize Model if missing
    if not os.path.exists(BEST_MODEL):
        print("Initializing random model...")
        torch.save(ChessCNN().state_dict(), BEST_MODEL)

    logger = MetricsLogger()
    
    print("=================================================")
    print(f"STARTING MCTS REINFORCEMENT LEARNING LOOP")
    print(f"Workers: {NUM_WORKERS} | Games/Iter: {NUM_WORKERS*GAMES_PER_WORKER}")
    print("=================================================")

    try:
        for it in range(1, ITERATIONS + 1):
            run_self_play_phase(it)
            run_training_phase(it)
            run_evaluation_phase(it, logger)
            
            # Optional: Clean up old data to prevent disk fill-up?
            # For now, we keep it to build a massive dataset.
            
    except KeyboardInterrupt:
        print("\n\n--- LOOP STOPPED BY USER ---")