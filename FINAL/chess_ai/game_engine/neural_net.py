import torch
import torch.multiprocessing as mp
import time
import numpy as np
import concurrent.futures 
from game_engine.cnn import ChessCNN

class InferenceServer:
    def __init__(self, model_path, batch_size=128, timeout=0.01, streams=4):
        self.model_path = model_path
        self.batch_size = batch_size
        self.timeout = timeout
        # We do NOT create the device or streams here to avoid pickling errors.
        # They will be created inside the process that actually uses them (in loop()).
        self.num_streams = streams
        
        # Communication queues
        self.input_queue = mp.Queue()
        self.output_queues = {} 

    def register_worker(self, worker_id):
        self.output_queues[worker_id] = mp.Queue()
        return self.output_queues[worker_id]

    def process_batch(self, batch, worker_ids, stream, model, device):
        if not batch: return

        with torch.cuda.stream(stream):
            # Move to device (GPU)
            batch_tensor = torch.stack(batch).to(device, non_blocking=True)
            
            with torch.no_grad():
                policies, values = model(batch_tensor)
            
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

        for i, wid in enumerate(worker_ids):
            try:
                self.output_queues[wid].put((policies[i], values[i]))
            except:
                pass

    def loop(self):
        # --- INITIALIZATION INSIDE THE PROCESS ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChessCNN().to(self.device)
        
        # Initialize Streams HERE (Local to this process)
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        self.current_stream_idx = 0
        
        print(f"Server: Loading checkpoint from {self.model_path}...")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Server: Warning - Could not load checkpoint: {e}")
            
        model.eval()
        model.share_memory() 
        print(f"Server: Model loaded on {self.device}. {self.num_streams}-Stream Pipeline Active.")

        # Thread pool matches number of streams
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_streams)

        while True:
            batch = []
            worker_ids = []
            
            start_time = time.time()
            while len(batch) < self.batch_size:
                if not self.input_queue.empty():
                    try:
                        item = self.input_queue.get_nowait()
                        if item == "STOP": 
                            executor.shutdown()
                            return
                        wid, state_tensor = item
                        batch.append(state_tensor)
                        worker_ids.append(wid)
                    except:
                        break
                
                if len(batch) > 0 and (time.time() - start_time > self.timeout):
                    break
                
                if len(batch) == 0:
                    time.sleep(0.0001)

            if batch:
                # Round-robin selection of streams
                stream = self.streams[self.current_stream_idx]
                self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
                
                # We must pass 'self.device' explicitly because 'self' might have a stale device ref from __init__
                executor.submit(self.process_batch, batch, worker_ids, stream, model, self.device)