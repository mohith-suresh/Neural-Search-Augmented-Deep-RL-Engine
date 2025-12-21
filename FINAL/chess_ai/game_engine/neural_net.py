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
        # === ADD AT START OF process_batch() ===

        batch_start_time = time.time()
        num_items = len(batch_data)
        total_positions = sum(item.shape if item.ndim == 4 else 1 for item in batch_data)

        if num_items % 10 == 1 or num_items < 10:  # Log small and round batches
            print(f"\n[DEBUG-3.1] process_batch() called:")
            print(f"[DEBUG-3.1]   Items: {num_items}")
            print(f"[DEBUG-3.1]   Total positions: {total_positions}")
            print(f"[DEBUG-3.1]   Avg per item: {total_positions / num_items if num_items > 0 else 0:.1f}")

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

            # === ADD AFTER MEGA_BATCH CREATION ===

            print(f"[DEBUG-3.2] mega_batch shape: {mega_batch.shape}")
            print(f"[DEBUG-3.2]   Device: {mega_batch.device}")
            print(f"[DEBUG-3.2]   Dtype: {mega_batch.dtype}")

            t_infer_start = time.time()

            with torch.no_grad():
                policies, values = model(mega_batch)
            
            # === ADD AFTER GPU INFERENCE (after model(mega_batch)) ===

            torch.cuda.synchronize() 

            inference_time = time.time() - t_infer_start

            print(f"[DEBUG-3.3] Model inference time: {inference_time*1000:.1f}ms for {mega_batch.shape} positions")
            print(f"[DEBUG-3.3]   Time per position: {(inference_time*1000)/mega_batch.shape:.2f}ms")
            print(f"[DEBUG-3.3]   Throughput: {mega_batch.shape/inference_time:.0f} positions/sec")

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
            
            # === ADD AT END OF process_batch() ===

        total_time = time.time() - batch_start_time
        print(f"[DEBUG-3.4] Total process_batch time: {total_time*1000:.1f}ms")
        print(f"[DEBUG-3.4]   Inference: ~{inference_time*1000:.1f}ms")
        print(f"[DEBUG-3.4]   Transfer/post-process: ~{(total_time-inference_time)*1000:.1f}ms")


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

        # === IN loop() AFTER model.eval() ===

        # Check GPU memory before starting
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1e9
            print(f"[DEBUG-5.2] GPU Memory used at start: {initial_mem:.2f}GB")
            print(f"[DEBUG-5.2] GPU Memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

        model.share_memory() 
        
        # executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_streams)
        print(f"Server Ready: Batch={self.batch_size}, Streams={self.num_streams}, Device={self.device}")

        # === ADD AFTER SERVER READY MESSAGE ===

        print(f"\n[DEBUG-2.1] Server timeout set to: {self.timeout}s")
        print(f"[DEBUG-2.1] Batch size limit: {self.batch_size}")
        print(f"[DEBUG-2.1] Queue monitoring starting...")

        # Add tracking variables at start of outer while loop
        batch_iteration = 0
        last_qsize_log = 0.0

        # Deadlock detection: track last successful batch time
        last_successful_batch_time = time.time()
        deadlock_timeout = 600   
                            
        while True:
            batch_data = []
            current_time = time.time()
            
            if current_time - last_successful_batch_time > deadlock_timeout:
                print(f"🚨 DEADLOCK DETECTED: No batch processed in {deadlock_timeout}s")
                return
            
            current_batch_count = 0
            start_time = time.time()
            
            # === ADD THIS INSIDE OUTER LOOP (accumulation phase starts) ===

            batch_iteration += 1
            accumulation_start = time.time()

            if batch_iteration % 10 == 0:  # Log every 10 batches
                print(f"\n[DEBUG-2.2] === Batch #{batch_iteration} Starting ===")
                print(f"[DEBUG-2.2] Queue size at start: {self.input_queue.qsize()}")
                print(f"[DEBUG-2.2] Timeout: {self.timeout}s, Batch size limit: {self.batch_size}")

            # Collect data until batch is full or timeout
            while current_batch_count < self.batch_size:
                try:
                    item = self.input_queue.get(timeout=0.01)
                    if item == "STOP":
                        executor.shutdown()
                        return
                    
                    tensor = item[1]
                    item_size = tensor.shape[0] if tensor.ndim == 4 else 1
                    batch_data.append(item)
                    current_batch_count += item_size
                    
                except: 
                    pass
                
            # === ADD AFTER EXCEPTION HANDLER ===

            # Every iteration of the get() loop, periodically log queue state
            if batch_iteration % 10 == 0:
                current_qsize = self.input_queue.qsize()
                elapsed = time.time() - accumulation_start
                if elapsed > 0.05:  # Only log if 50ms+ has passed
                    if current_qsize != last_qsize_log:
                        print(f"[DEBUG-2.3] t={elapsed*1000:.0f}ms | Queue: {current_qsize} | Accumulated: {current_batch_count}/{self.batch_size}")
                        last_qsize_log = current_qsize

                # Check timeout without sleep
                if current_batch_count > 0 and (time.time() - start_time > self.timeout):
                    break
            
            # === ADD AFTER ACCUMULATION LOOP, BEFORE BATCH PROCESSING ===

            accumulation_time = time.time() - accumulation_start

            if batch_iteration % 10 == 0:
                print(f"[DEBUG-2.4] Batch #{batch_iteration} accumulated in {accumulation_time*1000:.1f}ms")
                print(f"[DEBUG-2.4] Final queue size: {self.input_queue.qsize()}")
                print(f"[DEBUG-2.4] Items in batch: {len(batch_data)}")
                print(f"[DEBUG-2.4] Total positions: {sum(item.shape if item.ndim == 4 else 1 for item in batch_data)}")
                print(f"[DEBUG-2.4] Expected batch time: {self.timeout*1000:.1f}ms, Actual: {accumulation_time*1000:.1f}ms")

            if batch_data:
                stream = self.streams[self.current_stream_idx]
                self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
                
                try:
                    self.process_batch(batch_data, stream, model, self.device)
                    last_successful_batch_time = time.time()
                    
                    effective_size = sum(item.shape if item.ndim == 4 else 1 for item in batch_data)
                    print(f"[Server] Flushed batch: {len(batch_data)} requests, {effective_size} positions")
                except Exception as e:
                    print(f"🚨 FATAL: Batch processing crashed: {e}")
                    import traceback
                    traceback.print_exc()
                    return

