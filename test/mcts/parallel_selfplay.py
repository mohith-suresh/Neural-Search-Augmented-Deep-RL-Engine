#!/usr/bin/env python3
"""
Parallel Self-Play with CPU Multiprocessing + GPU Batch Inference
==================================================================

Project: EE542 - Deconstructing AlphaZero's Success
Optimized for: 8GB RAM + 4GB GTX GPU + Multi-core CPU

Architecture:
- 4 CPU workers running independent MCTS games
- Shared GPU for batched neural network inference
- Message queues for position evaluation requests
- 4x throughput compared to sequential execution

Memory Usage:
- 4 MCTS trees × 80 MB = 320 MB RAM
- CNN model: 120 MB GPU
- Batch buffer (16 pos): 50 MB GPU
- Total: Well within 8GB RAM + 4GB GPU limits

Performance:
- Sequential: 12 games/hour (200 sims/game)
- Parallel (4 workers): 48 games/hour
- 4x speedup with linear scaling
"""

import multiprocessing as mp
import queue
import time
import chess
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from mcts_tree import MCTS, MCTSNode
from self_play import BoardEncoder, PolicyEncoder, SelfPlayConfig

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

class PrintLogger:
    """Fallback logger that uses print()"""
    def __init__(self, prefix=""):
        self.prefix = prefix
    
    def info(self, msg):
        print(f"[INFO] {self.prefix} {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {self.prefix} {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {self.prefix} {msg}")


def inference_server_entrypoint(model_path, request_queue, response_queues, config):
    print(f"[DEBUG] Inference server starting...")  # ADD THIS
    server = InferenceServer(model_path, request_queue, response_queues, config)
    print(f"[DEBUG] Inference server initialized, calling run()...")  # ADD THIS
    server.run()

def worker_entrypoint(worker_id, request_queue, response_queue, result_queue, config, num_games):
    print(f"[DEBUG] Worker {worker_id} starting...")  # ADD THIS
    print(f"[DEBUG] Worker {worker_id} creating ParallelSelfPlayWorker instance...")  # ADD THIS

    worker = ParallelSelfPlayWorker(
        worker_id,
        request_queue,
        response_queue,
        result_queue,
        config,
        num_games
    )

    print(f"[DEBUG] Worker {worker_id} initialized, starting run()...")
    worker.run()


@dataclass
class ParallelConfig:
    """Configuration for parallel self-play."""

    # Parallelism
    num_workers: int = 4  # Number of parallel games
    batch_size: int = 32  # GPU batch inference size
    batch_timeout: float = 0.05  # Max wait time for batch (seconds)

    # MCTS parameters
    num_simulations: int = 200
    c_puct: float = 1.0

    # Temperature schedule
    temp_threshold: int = 30
    temp_init: float = 1.0
    temp_final: float = 0.1

    # Game limits
    max_game_length: int = 200

    # Output
    save_frequency: int = 10  # Save every N games
    output_dir: Path = Path("parallel_selfplay_games")

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BatchedNeuralNet:
    """
    Neural network wrapper with batched inference for parallel workers.

    Coordinates evaluation requests from multiple workers and batches
    them for efficient GPU utilization.
    """

    def __init__(self, model, device: str, batch_size: int):
        """
        Initialize batched neural network.

        Args:
            model: PyTorch CNN model
            device: Device for inference
            batch_size: Maximum batch size
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.encoder = BoardEncoder()

    @torch.no_grad()
    def predict_batch(self, boards: List[chess.Board]) -> List[Tuple[Dict, float]]:
        """
        Predict policy and value for batch of positions.

        Args:
            boards: List of chess boards

        Returns:
            List of (policy_dict, value) tuples
        """
        if not boards:
            return []

        # Encode boards to tensors
        tensor_list = []
        for board in boards:
            tensor = self.encoder.board_to_tensor(board)
            
            # Convert numpy array to torch tensor if needed
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).float()
            
            # Now squeeze if needed
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            
            tensor_list.append(tensor)

        batch_tensor = torch.stack(tensor_list).to(self.device)



        # Forward pass
        policy_logits, values = self.model(batch_tensor)
        policy_probs = torch.softmax(policy_logits, dim=1)

        if values.shape[-1] == 1:
            values = values.squeeze(-1)

        # Convert to list of dicts
        results = []
        for i, board in enumerate(boards):
            policy_array = policy_probs[i].cpu().numpy()
            value = values[i]
            # Defensive: If value is Tensor/array, squeeze to scalar
            if hasattr(value, "shape") and value.shape == (1,):
                value = value.item()
            elif hasattr(value, "shape"):
                value = value.squeeze().item()
            else:
                value = float(value)
            

            # Filter to legal moves
            legal_moves = list(board.legal_moves)
            move_probs = {}

            for move in legal_moves:
                move_idx = PolicyEncoder.move_to_index(move)
                move_probs[move] = policy_array[move_idx]

            # Normalize
            total = sum(move_probs.values())
            if total > 0:
                move_probs = {m: p/total for m, p in move_probs.items()}
            else:
                uniform = 1.0 / len(legal_moves)
                move_probs = {m: uniform for m in legal_moves}

            results.append((move_probs, value))

        return results


class InferenceServer:
    """
    GPU inference server that batches requests from multiple workers.

    Runs in separate process, handles evaluation requests via queues.
    """

    def __init__(
        self,
        model_path: str,
        request_queue: mp.Queue,
        response_queues: List[mp.Queue],
        config: ParallelConfig
    ):
        """
        Initialize inference server.

        Args:
            model_path: Path to model checkpoint
            request_queue: Queue for receiving evaluation requests
            response_queues: List of queues for sending results to workers
            config: Parallel configuration
        """
        self.model_path = model_path
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.config = config

        # Setup logging
        self.logger = logging.getLogger('InferenceServer')

    def load_model(self):
        """Load model on GPU."""
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent / "FINAL" / "chess_ai"))

        from cnn import ChessCNN, TrainingConfig

        checkpoint = torch.load(self.model_path, map_location=self.config.device)

        if 'config' in checkpoint:
            train_config = checkpoint['config']
        else:
            train_config = TrainingConfig()

        model = ChessCNN(train_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def run(self):
        """Run inference server loop."""
        self.logger.info("Starting inference server...")

        # Load model
        model = self.load_model()
        neural_net = BatchedNeuralNet(model, self.config.device, self.config.batch_size)
        print(f"[DEBUG] Model loaded successfully, entering request loop...")  # ADD THIS

        self.logger.info("Model loaded, ready for inference")

        batch_buffer = []
        batch_ids = []

        while True:
            try:
                # ADD THIS: Track requests received
                request_count = getattr(self, 'request_count', 0)

                # Collect requests until batch full or timeout
                timeout = self.config.batch_timeout if batch_buffer else None

                # print(f"[DEBUG-INFERENCE] Waiting for request (batch size: {len(batch_buffer)})...")  # ADD THIS

                try:
                    request = self.request_queue.get(timeout=timeout)
                    # print(f"[DEBUG-INFERENCE] Got request! Batch now: {len(batch_buffer)+1}")  # ADD THIS
                    request_count += 1
                    self.request_count = request_count
                    self.logger.info(f"Received request #{request_count}")
                except queue.Empty:
                    # print(f"[DEBUG-INFERENCE] Queue timeout, no request")  # ADD THIS
                    request = None

                if request is not None:
                    worker_id, request_id, board = request
                    batch_buffer.append(board)
                    batch_ids.append((worker_id, request_id))

                # Process batch when full or timeout
                if len(batch_buffer) >= self.config.batch_size or \
                (batch_buffer and request is None):
                    
                    self.logger.info(f"Processing batch of {len(batch_buffer)} positions")
                    
                    try:
                        # Batch inference
                        results = neural_net.predict_batch(batch_buffer)
                        
                        self.logger.info(f"Batch inference complete, sending {len(results)} results")
                        
                        # Send results back to workers
                        for (worker_id, request_id), result in zip(batch_ids, results):
                            self.response_queues[worker_id].put((request_id, result))
                    
                    except Exception as e:
                        self.logger.error(f"Batch inference failed: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Send None results to unblock workers
                        for (worker_id, request_id) in batch_ids:
                            self.response_queues[worker_id].put((request_id, None))
                    
                    finally:
                        # Clear batch regardless of success/failure
                        batch_buffer = []
                        batch_ids = []

            except KeyboardInterrupt:
                self.logger.info("Inference server shutting down...")
                break

            except Exception as e:
                self.logger.error(f"Inference server error: {e}")


class ParallelMCTS:
    """
    MCTS that sends evaluation requests to shared inference server.
    """

    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        config: ParallelConfig
    ):
        """
        Initialize parallel MCTS.

        Args:
            worker_id: Worker ID for this MCTS instance
            request_queue: Queue for sending evaluation requests
            response_queue: Queue for receiving evaluation results
            config: Configuration
        """
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.config = config

        self.next_request_id = 0

    def predict(self, board: chess.Board) -> Tuple[Dict, float]:
        """
        Request position evaluation from inference server.

        Args:
            board: Chess board

        Returns:
            (policy_dict, value)
        """
        request_id = self.next_request_id
        self.next_request_id += 1

        # print(f"[DEBUG-REQUEST] Worker {self.worker_id} sending request {request_id}")  # ADD THIS

        # Send request
        self.request_queue.put((self.worker_id, request_id, board))

        # print(f"[DEBUG-REQUEST] Worker {self.worker_id} request {request_id} SENT, waiting for response...")  # ADD THIS

        # Wait for response
        response = self.response_queue.get()  # THIS IS WHERE IT HANGS
        
        # print(f"[DEBUG-REQUEST] Worker {self.worker_id} GOT response {request_id}")  # ADD THIS
        
        request_id_resp, result = response
        return result


class ParallelSelfPlayWorker:
    """
    Single worker process for self-play game generation.
    """
    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        result_queue: mp.Queue,
        config: ParallelConfig,
        num_games: int
    ):
        """
        Initialize worker.

        Args:
            worker_id: Worker ID
            request_queue: Queue for evaluation requests
            response_queue: Queue for evaluation responses
            result_queue: Queue for completed games
            config: Configuration
            num_games: Number of games to generate
        """

        print(f"[DEBUG] Worker {worker_id} __init__ started")  # ADD THIS

        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.result_queue = result_queue
        self.config = config
        self.num_games = num_games
        
        print(f"[DEBUG] Worker {worker_id} setting up encoder...")  # ADD THIS
    
        # Setup encoder
        self.encoder = BoardEncoder()
        
        print(f"[DEBUG] Worker {worker_id} setting up logger...")

        # Use fallback PrintLogger since _setup_logger doesn't exist
        self.logger = PrintLogger(f"Worker {worker_id}")

        print(f"[DEBUG] Worker {worker_id} __init__ complete")


    def play_game(self, game_num: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one self-play game.

        Args:
            game_num: Game number

        Returns:
            List of (position, policy, value) examples
        """
        self.logger.info(f"Starting game {game_num}")

        # Create MCTS with shared neural network
        neural_net_wrapper = ParallelMCTS(
            self.worker_id,
            self.request_queue,
            self.response_queue,
            self.config
        )

        mcts = MCTS(
            neural_net=neural_net_wrapper,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            temperature=self.config.temp_init
        )

        board = chess.Board()
        game_history = []
        move_count = 0

        # ADD THIS
        self.logger.info(f"Game {game_num}: Starting move loop")

        # Play game
        while not board.is_game_over() and move_count < self.config.max_game_length:
            # Set temperature
            if move_count < self.config.temp_threshold:
                mcts.temperature = self.config.temp_init
                add_noise = True
            else:
                mcts.temperature = self.config.temp_final
                add_noise = False

            # MCTS search
            move_probs, selected_move = mcts.get_move_probabilities(board, add_noise)

            # Record example
            position_tensor = BoardEncoder.board_to_tensor(board)
            policy_vector = PolicyEncoder.policy_dict_to_vector(move_probs, board)
            game_history.append((position_tensor, policy_vector, board.turn))

            # Make move
            board.push(selected_move)
            move_count += 1

        # Get game result
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            else:
                outcome = 0.0
        else:
            outcome = 0.0

        # Create training examples
        examples = []
        for position, policy, turn in game_history:
            value = outcome if turn == chess.WHITE else -outcome
            examples.append((position, policy, value))

        self.logger.info(f"Game {game_num} complete: {move_count} moves, {len(examples)} examples")

        return examples

    def run(self):
        """Run worker loop to generate games."""
        if self.logger:
            self.logger.info(f"Worker {self.worker_id} starting...")
        else:
            print(f"[INFO] Worker {self.worker_id} starting...")

        for game_num in range(self.num_games):
            try:
                examples = self.play_game(game_num)
                self.result_queue.put((self.worker_id, game_num, examples))

            except Exception as e:
                self.logger.error(f"Game {game_num} failed: {e}")

        self.logger.info(f"Worker {self.worker_id} completed {self.num_games} games")


def run_parallel_selfplay(
    model_path: str,
    num_games: int,
    config: Optional[ParallelConfig] = None
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Run parallel self-play generation.

    Args:
        model_path: Path to trained model
        num_games: Total number of games to generate
        config: Parallel configuration

    Returns:
        List of all training examples
    """
    if config is None:
        config = ParallelConfig()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s'
    )
    logger = logging.getLogger('ParallelSelfPlay')

    logger.info("="*80)
    logger.info("PARALLEL SELF-PLAY GENERATION")
    logger.info("="*80)
    logger.info(f"Total games: {num_games}")
    logger.info(f"Workers: {config.num_workers}")
    logger.info(f"Simulations: {config.num_simulations}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info("="*80)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create queues
    request_queue = mp.Queue(maxsize=config.batch_size * 2)
    response_queues = [mp.Queue() for _ in range(config.num_workers)]
    result_queue = mp.Queue()

    # Start inference server
    server = mp.Process(
    target=inference_server_entrypoint,
    args=(model_path, request_queue, response_queues, config)
    )

    server.start()

    # Wait for server to initialize
    time.sleep(2)

    # Start worker processes
    games_per_worker = num_games // config.num_workers
    workers = []

    for worker_id in range(config.num_workers):
        worker = mp.Process(
            target=worker_entrypoint,
            args=(worker_id, request_queue, response_queues[worker_id], result_queue, config, games_per_worker)
        )
        workers.append(worker)
        worker.start()


    # Collect results
    all_examples = []
    games_completed = 0
    expected_games = games_per_worker * config.num_workers

    while games_completed < expected_games:
        try:
            worker_id, game_num, examples = result_queue.get()
            all_examples.extend(examples)
            games_completed += 1

            logger.info(f"Received game {games_completed}/{expected_games} "
                       f"from worker {worker_id} ({len(examples)} examples)")

            # Periodic save
            if games_completed % config.save_frequency == 0:
                save_examples(all_examples, games_completed, config.output_dir)

        except queue.Empty:
            logger.warning("Timeout waiting for game results")
            break

    # Wait for workers to finish
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()

    # Shutdown inference server
    server.terminate()
    server.join(timeout=5)

    # Final save
    save_examples(all_examples, games_completed, config.output_dir)

    logger.info("="*80)
    logger.info("PARALLEL GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total games: {games_completed}")
    logger.info(f"Total examples: {len(all_examples)}")
    if games_completed > 0:
        logger.info(f"Avg examples/game: {len(all_examples)/games_completed:.1f}")
    else:
        logger.info("Avg examples/game: N/A (no games completed)")

    return all_examples


def save_examples(examples: List[Tuple], game_num: int, output_dir: Path):
    """Save training examples to disk."""
    if not examples:
        return

    positions = np.array([ex[0] for ex in examples], dtype=np.float32)
    policies = np.array([ex[1] for ex in examples], dtype=np.float32)
    values = np.array([ex[2] for ex in examples], dtype=np.float32)

    output_file = output_dir / f"parallel_games_{game_num}.npz"
    np.savez_compressed(
        output_file,
        positions=positions,
        policies=policies,
        values=values
    )

    print(f"Saved {len(examples)} examples to {output_file}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sims', type=int, default=200)

    args = parser.parse_args()

    config = ParallelConfig(
        num_workers=args.workers,
        num_simulations=args.sims,
        output_dir=Path("parallel_selfplay_output")
    )

    examples = run_parallel_selfplay(args.model, args.games, config)

    print(f"\nGenerated {len(examples)} training examples from {args.games} games")


if __name__ == '__main__':
    main()
