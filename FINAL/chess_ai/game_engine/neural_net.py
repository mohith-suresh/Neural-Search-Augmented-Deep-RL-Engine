import torch
import torch.multiprocessing as mp
import time
import numpy as np
import threading
from collections import deque

from game_engine.cnn import ChessCNN


class InferenceServer:
    """
    High-performance inference server with proper CUDA stream usage.
    
    Architecture:
    ─────────────────────────────────────────────────────────────────────
    
    Main Loop (single thread):
        ┌─────────────────────────────────────────────────────────────┐
        │  1. Check if previous batch (stream N-1) is ready          │
        │     → If ready: transfer D2H, distribute results           │
        │  2. Collect new batch from queue                           │
        │  3. Launch new batch on stream N (non-blocking)            │
        │  4. Rotate to next stream                                  │
        └─────────────────────────────────────────────────────────────┘
    
    This allows overlapping:
        - GPU compute of batch N
        - Result distribution of batch N-1  
        - Collection of batch N+1
    
    ─────────────────────────────────────────────────────────────────────
    """
    
    def __init__(self, model_path, batch_size=512, timeout=0.1, streams=4):
        """
        Initialize inference server.
        
        Args:
            model_path: Path to model checkpoint
            batch_size: Target batch size (in positions)
            timeout: Max seconds to wait for batch to fill (actual seconds, not ms)
            streams: Number of CUDA streams for pipelining
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.timeout = timeout  # Now in seconds directly
        self.num_streams = streams
        
        self.input_queue = mp.Queue()
        self.output_queues = {}
        
    def register_worker(self, worker_id):
        """Register a worker and return its output queue."""
        self.output_queues[worker_id] = mp.Queue()
        return self.output_queues[worker_id]

    def loop(self):
        """Main inference loop with proper CUDA stream pipelining."""
        
        # ═══════════════════════════════════════════════════════════════
        # DEVICE SETUP
        # ═══════════════════════════════════════════════════════════════
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = self.device.type == "cuda"
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL SETUP
        # ═══════════════════════════════════════════════════════════════
        model = ChessCNN(upgraded=True).to(self.device)
        
        print(f"[Server] Loading model from {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"[Server] Model loaded successfully")
        except Exception as e:
            print(f"[Server] Warning: Could not load model: {e}")
            
        model.eval()
        
        # ═══════════════════════════════════════════════════════════════
        # CUDA STREAM SETUP
        # ═══════════════════════════════════════════════════════════════
        if use_cuda:
            # Create streams
            self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
            # Create events for synchronization (more efficient than stream.synchronize())
            self.events = [torch.cuda.Event() for _ in range(self.num_streams)]
        else:
            self.streams = [None] * self.num_streams
            self.events = [None] * self.num_streams
        
        # Pipeline state: track in-flight batches
        # Each slot: (event, worker_ids, sizes, raw_tensors, policies_gpu, values_gpu) or None
        in_flight = [None] * self.num_streams
        current_stream_idx = 0
        
        print(f"[Server] Ready: batch_size={self.batch_size}, timeout={self.timeout}s, "
              f"streams={self.num_streams}, device={self.device}")
        
        # Deadlock detection
        last_activity_time = time.time()
        deadlock_timeout = 600  # 10 minutes
        
        # Stats
        total_batches = 0
        total_positions = 0
        stats_start = time.time()
        
        # ═══════════════════════════════════════════════════════════════
        # MAIN LOOP
        # ═══════════════════════════════════════════════════════════════
        while True:
            current_time = time.time()
            
            # ───────────────────────────────────────────────────────────
            # STEP 1: Check for completed batches and distribute results
            # ───────────────────────────────────────────────────────────
            for slot_idx in range(self.num_streams):
                slot = in_flight[slot_idx]
                if slot is None:
                    continue
                    
                event, worker_ids, sizes, raw_tensors, policies_gpu, values_gpu = slot
                
                # Check if this batch is done (non-blocking query)
                if use_cuda:
                    if not event.query():
                        continue  # Not done yet, check next slot
                
                # Batch is complete! Transfer to CPU and distribute
                policies = policies_gpu.cpu().numpy()
                values = values_gpu.cpu().numpy()
                
                # Distribute results to workers
                cursor = 0
                for i, wid in enumerate(worker_ids):
                    size = sizes[i]
                    p_slice = policies[cursor:cursor + size]
                    v_slice = values[cursor:cursor + size]
                    cursor += size
                    
                    # Handle single position vs batch
                    if size == 1 and raw_tensors[i].ndim == 3:
                        self.output_queues[wid].put((p_slice[0], v_slice[0]))
                    else:
                        self.output_queues[wid].put((p_slice, v_slice))
                
                # Clear slot
                in_flight[slot_idx] = None
                last_activity_time = current_time
            
            # ───────────────────────────────────────────────────────────
            # STEP 2: Collect new batch from input queue
            # ───────────────────────────────────────────────────────────
            batch_data = []
            current_batch_count = 0
            collect_start = time.time()
            
            while current_batch_count < self.batch_size:
                # Calculate remaining time
                elapsed = time.time() - collect_start
                remaining = max(0.001, self.timeout - elapsed)
                
                try:
                    item = self.input_queue.get(timeout=min(0.01, remaining))
                    
                    if item == "STOP":
                        print("[Server] Received STOP signal")
                        # Wait for in-flight batches to complete
                        if use_cuda:
                            torch.cuda.synchronize()
                        return
                    
                    tensor = item[1]
                    item_size = tensor.shape[0] if tensor.ndim == 4 else 1
                    batch_data.append(item)
                    current_batch_count += item_size
                    
                except:
                    pass  # Queue empty or timeout
                
                # Check if we've exceeded collection timeout
                if current_batch_count > 0 and (time.time() - collect_start) >= self.timeout:
                    break
            
            # ───────────────────────────────────────────────────────────
            # STEP 3: Launch new batch if we have data
            # ───────────────────────────────────────────────────────────
            if batch_data:
                # Find a free stream slot (or wait for one)
                attempts = 0
                while in_flight[current_stream_idx] is not None:
                    # Slot is busy, check if it's done
                    slot = in_flight[current_stream_idx]
                    event = slot[0]
                    
                    if use_cuda and not event.query():
                        # Still running, try next stream
                        current_stream_idx = (current_stream_idx + 1) % self.num_streams
                        attempts += 1
                        
                        if attempts >= self.num_streams:
                            # All streams busy, wait for current one
                            if use_cuda:
                                event.synchronize()
                            break
                    else:
                        # This slot is done, process it (will be cleared in next iteration)
                        break
                
                # Prepare batch data
                worker_ids = [item[0] for item in batch_data]
                raw_tensors = [item[1] for item in batch_data]
                sizes = [t.shape[0] if t.ndim == 4 else 1 for t in raw_tensors]
                
                # Concatenate tensors
                processed_tensors = [t if t.ndim == 4 else t.unsqueeze(0) for t in raw_tensors]
                mega_batch_cpu = torch.cat(processed_tensors, dim=0)
                
                # Pin memory for async H2D transfer
                if use_cuda and not mega_batch_cpu.is_pinned():
                    mega_batch_cpu = mega_batch_cpu.pin_memory()
                
                stream = self.streams[current_stream_idx]
                event = self.events[current_stream_idx]
                
                # Launch on stream (non-blocking)
                if use_cuda:
                    with torch.cuda.stream(stream):
                        # Async H2D transfer
                        mega_batch = mega_batch_cpu.to(self.device, non_blocking=True)
                        
                        # Inference
                        with torch.no_grad():
                            policies_gpu, values_gpu = model(mega_batch)
                        
                        # Record event when done
                        event.record(stream)
                else:
                    # CPU path - synchronous
                    mega_batch = mega_batch_cpu.to(self.device)
                    with torch.no_grad():
                        policies_gpu, values_gpu = model(mega_batch)
                
                # Store in-flight batch
                in_flight[current_stream_idx] = (
                    event, worker_ids, sizes, raw_tensors, policies_gpu, values_gpu
                )
                
                # Stats
                total_batches += 1
                total_positions += current_batch_count
                
                # Logging (every 100 batches)
                if total_batches % 100 == 0:
                    elapsed = time.time() - stats_start
                    pos_per_sec = total_positions / elapsed if elapsed > 0 else 0
                    print(f"[Server] Batches: {total_batches}, Positions: {total_positions}, "
                          f"Throughput: {pos_per_sec:.0f} pos/s")
                
                # Move to next stream
                current_stream_idx = (current_stream_idx + 1) % self.num_streams
                last_activity_time = current_time
            
            # ───────────────────────────────────────────────────────────
            # STEP 4: Deadlock detection
            # ───────────────────────────────────────────────────────────
            if current_time - last_activity_time > deadlock_timeout:
                print(f"[Server] DEADLOCK DETECTED: No activity in {deadlock_timeout}s")
                return


class InferenceServerThreaded:
    """
    Alternative implementation using ThreadPoolExecutor for result distribution.
    Use this if the main InferenceServer has issues with your workload.
    """
    
    def __init__(self, model_path, batch_size=512, timeout=0.1, streams=4):
        self.model_path = model_path
        self.batch_size = batch_size
        self.timeout = timeout
        self.num_streams = streams
        
        self.input_queue = mp.Queue()
        self.output_queues = {}

    def register_worker(self, worker_id):
        self.output_queues[worker_id] = mp.Queue()
        return self.output_queues[worker_id]

    def _distribute_results(self, worker_ids, sizes, raw_tensors, policies, values):
        """Distribute results to worker queues (runs in thread)."""
        cursor = 0
        for i, wid in enumerate(worker_ids):
            size = sizes[i]
            p_slice = policies[cursor:cursor + size]
            v_slice = values[cursor:cursor + size]
            cursor += size
            
            if size == 1 and raw_tensors[i].ndim == 3:
                self.output_queues[wid].put((p_slice[0], v_slice[0]))
            else:
                self.output_queues[wid].put((p_slice, v_slice))

    def loop(self):
        """Main loop with threaded result distribution."""
        import concurrent.futures
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = self.device.type == "cuda"
        
        model = ChessCNN(upgraded=True).to(self.device)
        
        print(f"[Server] Loading model from {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"[Server] Warning: {e}")
            
        model.eval()
        
        # Streams and thread pool
        if use_cuda:
            streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        else:
            streams = [None] * self.num_streams
            
        current_stream_idx = 0
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_streams)
        
        print(f"[Server] Ready: batch={self.batch_size}, timeout={self.timeout}s, "
              f"streams={self.num_streams}, device={self.device}")
        
        last_activity = time.time()
        
        while True:
            # Collect batch
            batch_data = []
            batch_count = 0
            start = time.time()
            
            while batch_count < self.batch_size:
                elapsed = time.time() - start
                if batch_count > 0 and elapsed >= self.timeout:
                    break
                    
                try:
                    item = self.input_queue.get(timeout=0.01)
                    if item == "STOP":
                        executor.shutdown(wait=True)
                        return
                    
                    tensor = item[1]
                    size = tensor.shape[0] if tensor.ndim == 4 else 1
                    batch_data.append(item)
                    batch_count += size
                except:
                    pass
            
            if not batch_data:
                if time.time() - last_activity > 600:
                    print("[Server] DEADLOCK: No activity in 600s")
                    return
                continue
            
            last_activity = time.time()
            
            # Process batch
            worker_ids = [item[0] for item in batch_data]
            raw_tensors = [item[1] for item in batch_data]
            sizes = [t.shape[0] if t.ndim == 4 else 1 for t in raw_tensors]
            
            processed = [t if t.ndim == 4 else t.unsqueeze(0) for t in raw_tensors]
            mega_batch_cpu = torch.cat(processed, dim=0)
            
            if use_cuda and not mega_batch_cpu.is_pinned():
                mega_batch_cpu = mega_batch_cpu.pin_memory()
            
            stream = streams[current_stream_idx]
            
            if use_cuda:
                with torch.cuda.stream(stream):
                    mega_batch = mega_batch_cpu.to(self.device, non_blocking=True)
                    with torch.no_grad():
                        policies_gpu, values_gpu = model(mega_batch)
                
                # Sync this stream before CPU transfer
                stream.synchronize()
            else:
                mega_batch = mega_batch_cpu.to(self.device)
                with torch.no_grad():
                    policies_gpu, values_gpu = model(mega_batch)
            
            # Transfer to CPU
            policies = policies_gpu.cpu().numpy()
            values = values_gpu.cpu().numpy()
            
            # Distribute in thread (overlaps with next batch collection)
            executor.submit(
                self._distribute_results,
                worker_ids, sizes, raw_tensors, policies, values
            )
            
            current_stream_idx = (current_stream_idx + 1) % self.num_streams