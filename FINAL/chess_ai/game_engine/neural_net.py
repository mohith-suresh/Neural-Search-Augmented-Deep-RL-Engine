import torch
import torch.multiprocessing as mp
import time
import numpy as np
from FINAL.chess_ai.game_engine.cnn_old import ChessCNN

class InferenceServer:
    def __init__(self, model_path, batch_size=256, timeout=0.01):
        self.model_path = model_path
        self.batch_size = batch_size
        self.timeout = timeout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Communication queues
        self.input_queue = mp.Queue()
        self.output_queues = {} # specific queue for each worker ID

    def register_worker(self, worker_id):
        self.output_queues[worker_id] = mp.Queue()
        return self.output_queues[worker_id]

    def loop(self):
        # Load Model ONCE on the GPU
        model = ChessCNN().to(self.device)
        
        print(f"Server: Loading checkpoint from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # --- FIX: Robust Dictionary Loading ---
        if 'model_state_dict' in checkpoint:
            # Format 1: Standard PyTorch tutorial save
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Format 2: Lightning / Custom save (YOUR CASE)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Format 3: Raw state dict (no metadata)
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"Server: Model loaded on {self.device}. Listening...")
        
        # ... rest of the loop remains the same ...

        while True:
            batch = []
            worker_ids = []
            
            # 1. Collect Batch
            start_time = time.time()
            while len(batch) < self.batch_size:
                if not self.input_queue.empty():
                    item = self.input_queue.get()
                    if item == "STOP": return
                    wid, state_tensor = item
                    batch.append(state_tensor)
                    worker_ids.append(wid)
                
                # Dynamic batching: break if waiting too long
                if len(batch) > 0 and (time.time() - start_time > self.timeout):
                    break
                
                if len(batch) == 0:
                    time.sleep(0.001) # Prevent CPU spin if empty
                    continue

            # 2. Inference
            if batch:
                # Stack tensors: (B, 13, 8, 8)
                batch_tensor = torch.stack(batch).to(self.device)
                
                with torch.no_grad():
                    # cnn.py returns (policy, value)
                    policies, values = model(batch_tensor)
                
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()

                # 3. Distribute results
                for i, wid in enumerate(worker_ids):
                    self.output_queues[wid].put((policies[i], values[i]))