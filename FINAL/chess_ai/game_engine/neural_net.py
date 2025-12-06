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
        self.num_streams = streams
        
        self.input_queue = mp.Queue()
        self.output_queues = {} 

    def register_worker(self, worker_id):
        self.output_queues[worker_id] = mp.Queue()
        return self.output_queues[worker_id]

    def process_batch(self, batch_data, stream, model, device):
        """
        batch_data: list of (worker_id, tensor)
        tensor can be shape (13, 8, 8) [Single] OR (N, 13, 8, 8) [Batch]
        """
        if not batch_data: return

        worker_ids = [item[0] for item in batch_data]
        raw_tensors = [item[1] for item in batch_data]
        
        # Track original sizes to split results later
        # If tensor is 3D (13,8,8), size is 1. If 4D (N,13,8,8), size is N.
        sizes = [t.shape[0] if t.ndim == 4 else 1 for t in raw_tensors]
        
        with torch.cuda.stream(stream):
            # 1. Normalize all inputs to 4D tensors
            processed_tensors = []
            for t in raw_tensors:
                if t.ndim == 3:
                    processed_tensors.append(t.unsqueeze(0))
                else:
                    processed_tensors.append(t)
            
            # 2. Combine into one Mega-Batch
            mega_batch = torch.cat(processed_tensors, dim=0).to(device, non_blocking=True)
            
            # 3. Inference
            with torch.no_grad():
                policies, values = model(mega_batch)
            
            # 4. Move back to CPU
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

        # 5. Distribute results back to workers
        cursor = 0
        for i, wid in enumerate(worker_ids):
            size = sizes[i]
            
            # Slice the results for this worker
            p_slice = policies[cursor : cursor + size]
            v_slice = values[cursor : cursor + size]
            cursor += size
            
            # If it was a single item request, unwrap it (for compatibility)
            # If it was a batch request, send array
            if size == 1 and raw_tensors[i].ndim == 3:
                self.output_queues[wid].put((p_slice[0], v_slice[0]))
            else:
                self.output_queues[wid].put((p_slice, v_slice))

    def loop(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChessCNN().to(self.device)
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        self.current_stream_idx = 0
        
        print(f"Server: Loading model from {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
            else: model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Server Warning: {e}")
            
        model.eval()
        model.share_memory() # Required for some MP start methods
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_streams)
        print(f"Server Ready: Batch={self.batch_size}, Streams={self.num_streams}")

        while True:
            batch_data = []
            current_batch_count = 0
            
            start_time = time.time()
            
            # Aggressive Batch Collection
            while current_batch_count < self.batch_size:
                if not self.input_queue.empty():
                    try:
                        item = self.input_queue.get_nowait()
                        if item == "STOP": 
                            executor.shutdown()
                            return
                        
                        # Check size of incoming item to avoid overflowing batch
                        tensor = item[1]
                        item_size = tensor.shape[0] if tensor.ndim == 4 else 1
                        
                        batch_data.append(item)
                        current_batch_count += item_size
                    except:
                        break
                
                # Dynamic timeout based on how full we are
                # If we have data, wait less. If empty, wait a tiny bit more.
                if current_batch_count > 0:
                    if (time.time() - start_time > self.timeout): break
                else:
                    time.sleep(0.0001)

            if batch_data:
                stream = self.streams[self.current_stream_idx]
                self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
                executor.submit(self.process_batch, batch_data, stream, model, self.device)