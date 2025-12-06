import torch
import torch.multiprocessing as mp
import time
import numpy as np
import logging
from game_engine.cnn import ChessCNN

# Configure logging to see detailed timing info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger("InferenceServer")

class InferenceServer:
    def __init__(self, model_path, batch_size=256, timeout=0.005):
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

        cycle_count = 0

        while True:
            batch = []
            worker_ids = []

            # ====== PHASE 1: COLLECT BATCH ======
            t_collect_start = time.time()
            start_time = t_collect_start

            while len(batch) < self.batch_size:
                if not self.input_queue.empty():
                    item = self.input_queue.get()
                    if item == "STOP":
                        return
                    wid, state_tensor = item
                    batch.append(state_tensor)
                    worker_ids.append(wid)

                # Dynamic batching: break if waiting too long
                if len(batch) > 0 and (time.time() - start_time > self.timeout):
                    break

                if len(batch) == 0:
                    time.sleep(0.001) # Prevent CPU spin if empty

            t_collect_end = time.time()

            if len(batch) == 0:
                continue

            # ====== PHASE 2: INFERENCE ======
            t_infer_start = time.time()

            batch_tensor = torch.stack(batch).to(self.device)

            with torch.no_grad():
                # cnn.py returns (policy, value)
                policies, values = model(batch_tensor)

            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

            t_infer_end = time.time()

            # ====== PHASE 3: DISTRIBUTE RESULTS ======
            t_return_start = time.time()

            for i, wid in enumerate(worker_ids):
                self.output_queues[wid].put((policies[i], values[i]))

            t_return_end = time.time()

            # ====== LOGGING ======
            cycle_count += 1
            collect_time = t_collect_end - t_collect_start
            infer_time = t_infer_end - t_infer_start
            return_time = t_return_end - t_return_start
            total_time = t_return_end - t_collect_start

            logger.info(
                f"Cycle {cycle_count:05d} | "
                f"Batch Size: {len(batch):3d} | "
                f"Collect: {collect_time:6.3f}s | "
                f"Inference: {infer_time:6.3f}s | "
                f"Return: {return_time:6.3f}s | "
                f"Total: {total_time:6.3f}s | "
                f"Queue Size: {self.input_queue.qsize()}"
            )
