import torch
import torch.multiprocessing as mp
import time
import numpy as np
from game_engine.cnn import ChessCNN

class InferenceServer:
    # FINAL OPTIMIZATION: Aggressive timeout (1.0s) to maximize collected batch size 
    # (closer to 100) and dramatically reduce kernel launch overhead/latency.
    def __init__(self, model_path, batch_size=256, timeout=1.0):
        self.model_path = model_path
        # Revert to max batch size (256) to ensure the server processes the full capacity.
        self.batch_size = batch_size 
        # Aggressive timeout: 1.0s to ensure the entire backlog of 90+ queue is collected into one batch.
        self.timeout = timeout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Communication queues
        self.input_queue = mp.Queue()
        self.output_queues = {} # specific queue for each worker ID
        
        # Removed CUDA streams and async logic for simpler, faster single-thread batch processing.

    def register_worker(self, worker_id):
        self.output_queues[worker_id] = mp.Queue()
        return self.output_queues[worker_id]

    def loop(self):
        # Load Model ONCE on the GPU
        model = ChessCNN().to(self.device)
        
        print(f"Server: Loading checkpoint from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"Server: Model loaded on {self.device}. Listening...")
        
        # 

        while True:
            batch = []
            worker_ids = []
            
            # 1. Collect Batch (Maximize collection using long timeout)
            start_time = time.time()
            # We use a long timeout (1.0s) to ensure the queue backlog is collected.
            while (time.time() - start_time < self.timeout):
                if not self.input_queue.empty():
                    item = self.input_queue.get()
                    if item == "STOP": 
                        return
                    wid, state_tensor = item
                    batch.append(state_tensor)
                    worker_ids.append(wid)
                
                # If we hit the max batch size (256) before timeout, process immediately.
                if len(batch) >= self.batch_size:
                    break 
                
                if len(batch) == 0:
                    time.sleep(0.001) 
                    continue

            # 2. Inference (High-efficiency calculation)
            if batch:
                # Stack tensors: (B, 13, 8, 8). This batch size is now maximized (approx 90-100).
                batch_tensor = torch.stack(batch).to(self.device)
                
                with torch.no_grad():
                    policies, values = model(batch_tensor)
                
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()

                # 3. Distribute results
                for i, wid in enumerate(worker_ids):
                    self.output_queues[wid].put((policies[i], values[i]))