import torch
import torch.multiprocessing as mp
import time
import numpy as np
import concurrent.futures 

from game_engine.cnn import ChessCNN

class InferenceServer:
    def __init__(self, model_path, batch_size=512, timeout=0.01, streams=4):
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
        if not batch_data: return

        worker_ids = [item[0] for item in batch_data]
        raw_tensors = [item[1] for item in batch_data]
        
        # Track sizes (Most will be 8, but we must handle single/variable sizes safely)
        sizes = [t.shape[0] if t.ndim == 4 else 1 for t in raw_tensors]
        
        with torch.cuda.stream(stream):
            # Fast normalization
            processed_tensors = [t if t.ndim == 4 else t.unsqueeze(0) for t in raw_tensors]
            
            # Massive Batch
            mega_batch = torch.cat(processed_tensors, dim=0).to(device, non_blocking=True)
            
            with torch.no_grad():
                policies, values = model(mega_batch)
            
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

        cursor = 0
        for i, wid in enumerate(worker_ids):
            size = sizes[i]
            p_slice = policies[cursor : cursor + size]
            v_slice = values[cursor : cursor + size]
            cursor += size
            
            if size == 1 and raw_tensors[i].ndim == 3:
                self.output_queues[wid].put((p_slice[0], v_slice[0]))
            else:
                self.output_queues[wid].put((p_slice, v_slice))

    def loop(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChessCNN(upgraded=True).to(self.device)
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
        model.share_memory() 
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_streams)
        print(f"Server Ready: Batch={self.batch_size}, Streams={self.num_streams}, Device={self.device}")
        # Deadlock detection: track last successful batch time
        last_successful_batch_time = time.time()
        deadlock_timeout = 60  # 60 seconds without progress = deadlock
                            
        while True:
            batch_data = []
            current_time = time.time()
            
            # Check for deadlock
            if current_time - last_successful_batch_time > deadlock_timeout:
                print(f"🚨 DEADLOCK DETECTED: No batch processed in {deadlock_timeout}s")
                print(f"   Input queue size: {self.input_queue.qsize()}")
                print(f"   Output queues: {len(self.output_queues)} workers")
                return  # Exit server gracefully
            
            current_batch_count = 0
            start_time = time.time()
            
            # Collect data until batch is full or timeout
            while current_batch_count < self.batch_size:
                if not self.input_queue.empty():
                    try:
                        item = self.input_queue.get_nowait()
                        if item == "STOP": 
                            executor.shutdown()
                            return
                        
                        tensor = item[1]
                        item_size = tensor.shape[0] if tensor.ndim == 4 else 1
                        
                        batch_data.append(item)
                        current_batch_count += item_size
                    except: break
                
                # Dynamic timeout
                if current_batch_count > 0:
                    if (time.time() - start_time > self.timeout): break
                else:
                    time.sleep(0.0001)

            if batch_data:
                stream = self.streams[self.current_stream_idx]
                self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
                
                executor.submit(self.process_batch, batch_data, stream, model, self.device)

                effective_size = sum(item[1].shape[0] if item[1].ndim == 4 else 1 for item in batch_data)
                print(f"[Server] Flushed batch: {len(batch_data)} requests, {effective_size} positions, {(time.time()-start_time)*1000:.1f}ms elapsed")
                
