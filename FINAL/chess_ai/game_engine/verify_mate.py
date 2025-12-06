import torch
import numpy as np
import queue
import threading
import time
import sys
import os

# Ensure imports work
sys.path.append(os.getcwd())

from game_engine.chess_env import ChessGame
from game_engine.mcts import MCTSWorker, Node, move_to_index

# --- CONFIG ---
# FEN: White to move. Rh4 is Checkmate.
# The network might think this is "messy", but MCTS must find the win.
TEST_FEN = "4r3/8/2p2PPk/1p6/pP2p1R1/P1B5/2P2K2/3r4 w - - 1 45" 
MATING_MOVE = "g4h4" # Rh4#

class MockInferenceServer:
    """
    Simulates the Neural Net.
    Always returns uniform policy and 0.0 value (or even -0.5 to simulate a pessimistic net).
    This proves MCTS finds the win *despite* the network.
    """
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        while self.running:
            try:
                # MCTSWorker sends (worker_id, tensor)
                # But wait, MCTSWorker batching logic might send a batch tensor?
                # Let's handle both for safety, though MCTSWorker usually sends one batch item.
                item = self.input_queue.get(timeout=0.1)
                worker_id, tensor = item
                
                # Create Dummy Output
                # Tensor shape is (B, 16, 8, 8) or (16, 8, 8)
                batch_size = tensor.shape[0] if tensor.ndim == 4 else 1
                
                # Uniform policy (logits = 0)
                # Value = 0.0 (Neural net has NO CLUE who is winning)
                policy = np.zeros((batch_size, 8192), dtype=np.float32)
                value = np.zeros((batch_size, 1), dtype=np.float32) 
                
                # Send back
                # MCTSWorker expects: (policies, values) arrays
                self.output_queue.put((policy, value))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Mock Server Error: {e}")
                break

    def stop(self):
        self.running = False
        self.thread.join()

def run_test():
    print(f"♟️  Starting Mate-in-One Verification")
    print(f"📝 FEN: {TEST_FEN}")
    print(f"🎯 Target Move: {MATING_MOVE}")
    
    # 1. Setup Queues & Mock Server
    input_q = queue.Queue()
    output_q = queue.Queue()
    server = MockInferenceServer(input_q, output_q)
    
    # 2. Setup Worker
    # Simulations needs to be high enough to find the move in the tree
    SIMS = 1200

    print(f"SIMS = {SIMS}")

    worker = MCTSWorker(worker_id=0, input_queue=input_q, output_queue=output_q, 
                        simulations=SIMS, batch_size=8)
    
    # 3. Setup Game
    game = ChessGame(fen=TEST_FEN)
    
    # 4. Run Search
    print("\n🚀 Running MCTS...")
    start_time = time.time()
    
    # We use temp=0 to get the greedy best move
    best_move, policy = worker.search(game, temperature=0.0)
    
    duration = time.time() - start_time
    print(f"⏱️  Finished in {duration:.2f}s")
    
    # 5. Analyze Results
    print("\n📊 Analysis:")
    
    # Reconstruct Root Node stats manually to verify visit counts
    # (Since search() returns the action, we need to inspect the root internal state if possible,
    #  or just trust the returned policy vector which is normalized visits)
    
    # The policy vector maps indices to probabilities.
    # Let's find the probability assigned to our target move.
    target_idx = move_to_index(MATING_MOVE)
    target_prob = policy[target_idx]
    
    print(f"   Best Move Selected: {best_move}")
    print(f"   Target Move ({MATING_MOVE}) Probability: {target_prob:.4f}")
    
    # Validation
    if best_move == MATING_MOVE:
        print("\n✅ TEST PASSED: MCTS found the mate!")
        if target_prob > 0.5:
            print("   (Confidence is high, as expected)")
        else:
            print("   (WARNING: Confidence is low. Visits were split?)")
    else:
        print(f"\n❌ TEST FAILED: MCTS played {best_move} instead of {MATING_MOVE}.")
        print("Possible causes:")
        print("1. 'is_game_over()' logic in MCTS is failing to detect checkmate.")
        print("2. 'get_reward_for_turn' is returning wrong values.")
        print("3. Backpropagation logic isn't flipping the sign correctly.")

    server.stop()

if __name__ == "__main__":
    run_test()