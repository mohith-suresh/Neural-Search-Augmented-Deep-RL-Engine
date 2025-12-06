import json
import matplotlib.pyplot as plt
import os
import sys
import time

# Ensure imports work if needed, though this script is mostly standalone
sys.path.append(os.getcwd())

METRICS_FILE = "game_engine/model/metrics.json"
OUTPUT_IMAGE = "game_engine/model/training_metrics.png"

def plot_metrics():
    if not os.path.exists(METRICS_FILE):
        print(f"No metrics file found at {METRICS_FILE}")
        return

    history = []
    # Retry logic to handle potential race conditions if main.py is writing
    for _ in range(3):
        try:
            with open(METRICS_FILE, 'r') as f:
                history = json.load(f)
            break
        except json.JSONDecodeError:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error reading metrics file: {e}")
            return

    if not history:
        print("Metrics file is empty.")
        return

    iterations = [h['iteration'] for h in history]
    p_loss = [h.get('policy_loss', 0.0) for h in history]
    v_loss = [h.get('value_loss', 0.0) for h in history]
    win_rates = [h.get('arena_win_rate', 0.0) for h in history]
    
    # Filter for ELOs (might be None in some entries)
    elo_data = [(h['iteration'], h['elo']) for h in history if h.get('elo') is not None]
    elo_iters = [d[0] for d in elo_data]
    elos = [d[1] for d in elo_data]

    plt.figure(figsize=(12, 8))
    
    # 1. Loss Plot
    plt.subplot(2, 2, 1)
    plt.plot(iterations, p_loss, label='Policy Loss', color='blue')
    plt.plot(iterations, v_loss, label='Value Loss', color='orange')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. Win Rate Plot
    plt.subplot(2, 2, 2)
    plt.plot(iterations, win_rates, label='Win Rate vs Prev Best', color='green', marker='o')
    plt.axhline(y=0.55, color='r', linestyle='--', label='Threshold (0.55)')
    plt.title('Candidate Win Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    
    # 3. Elo Plot
    if elos:
        plt.subplot(2, 2, 3)
        plt.plot(elo_iters, elos, label='Est. Elo', color='purple', marker='x')
        plt.title('Estimated Elo (vs Stockfish)')
        plt.xlabel('Iteration')
        plt.ylabel('Elo')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Metrics plot saved to {OUTPUT_IMAGE}")
    plt.close()

if __name__ == "__main__":
    plot_metrics()