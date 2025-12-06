import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import sys
import collections

# Ensure we can import from the parent directory
sys.path.append(os.getcwd())

from game_engine.cnn import ChessCNN

# Named tuple to store the file boundaries
FileIndex = collections.namedtuple('FileIndex', ['file_path', 'start_idx', 'end_idx'])

class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.file_map = []  # Stores (file_path, start_index, end_index)
        self.total_positions = 0
        
        # Get all .npz files
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not files:
            print(f"Warning: No data found in {data_dir}")
            return

        print(f"Mapping {len(files)} data files...")
        
        for f in files:
            try:
                # Use np.load with mmap_mode to quickly check shape without loading data
                with np.load(f, allow_pickle=True) as data:
                    s_shape = data['states'].shape
                    p_shape = data['policies'].shape
                    v_shape = data['values'].shape
                
                    num_positions = s_shape[0]

                    # Validation: Check shapes
                    if len(s_shape) != 4 or s_shape[1:] != (13, 8, 8):
                        print(f"Skipping corrupt file {f}: State shape {s_shape}")
                        continue
                        
                    if len(p_shape) != 2 or p_shape[1] != 8192:
                        print(f"Skipping corrupt file {f}: Policy shape {p_shape}")
                        continue
                    
                    if num_positions > 0:
                        start_idx = self.total_positions
                        end_idx = self.total_positions + num_positions - 1
                        
                        self.file_map.append(FileIndex(f, start_idx, end_idx))
                        self.total_positions += num_positions
                        
            except Exception as e:
                print(f"Error checking {f}: {e}")
                continue

        # Cache for the currently loaded file
        self._current_file = None
        self._current_data = None
        
        print(f"Dataset Mapped: {self.total_positions} positions across {len(self.file_map)} files.")

    def __len__(self):
        return self.total_positions

    def _load_file_for_index(self, idx):
        # Determine which file the global index 'idx' belongs to
        
        # Binary search or simple iteration (simple iteration is fine for typical file counts)
        target_file = None
        local_idx = 0
        
        for file_info in self.file_map:
            if file_info.start_idx <= idx <= file_info.end_idx:
                target_file = file_info.file_path
                local_idx = idx - file_info.start_idx
                break

        if target_file is None:
            raise IndexError(f"Index {idx} out of range for file map.")
            
        # Check if the file is already cached
        if target_file != self._current_file:
            print(f"Loading new data file: {target_file}")
            data = np.load(target_file, allow_pickle=True)
            self._current_data = {
                'states': torch.from_numpy(data['states']),
                'policies': torch.from_numpy(data['policies']),
                'values': torch.from_numpy(data['values']),
            }
            self._current_file = target_file
            
        return self._current_data, local_idx


    def __getitem__(self, idx):
        # FIX E: Load data lazily based on index
        data, local_idx = self._load_file_for_index(idx)
        
        return {
            'state': data['states'][local_idx].float(),
            'policy': data['policies'][local_idx].float(),
            'value': data['values'][local_idx].float()
        }

def train_model(data_path="data/self_play", 
                input_model_path="game_engine/model/best_model.pth", 
                output_model_path="game_engine/model/candidate.pth", 
                epochs=1, 
                batch_size=128, 
                lr=0.00002):
    """
    Trains the model on data from data_path.
    1. Loads weights from input_model_path (The 'Champion').
    2. Trains for N epochs.
    3. Saves result to output_model_path (The 'Challenger').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # 1. Prepare Data
    dataset = ChessDataset(data_path)
    if len(dataset) == 0:
        print("Skipping training (No Data).")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Load Model
    model = ChessCNN().to(device)
    
    # Load existing weights (The Champion) to continue training
    if os.path.exists(input_model_path):
        print(f"Loading training base from {input_model_path}")
        try:
            checkpoint = torch.load(input_model_path, map_location=device)
            # FIX: Robust Dictionary Loading
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model, starting fresh: {e}")
    else:
        print(f"No existing model at {input_model_path}, starting from random weights.")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 3. Loss Functions
    mse_loss = nn.MSELoss()
    
    # 4. Training Loop
    for epoch in range(epochs):
        total_loss = 0
        p_loss_total = 0
        v_loss_total = 0
        batch_count = 0
        
        for batch in dataloader:
            states = batch['state'].to(device)
            target_policies = batch['policy'].to(device)
            target_values = batch['value'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # cnn returns (policy_logits, value)
            pred_policies, pred_values = model(states)
            
            # --- Value Loss (MSE) ---
            # pred_values shape is (Batch, 1), target is (Batch,)
            # We squeeze prediction to match target
            v_loss = mse_loss(pred_values.squeeze(), target_values)
            
            # --- Policy Loss (Cross Entropy) ---
            # AlphaZero Loss: -Sum(Target * Log(Softmax(Prediction)))
            # This pushes the network logits towards the MCTS probability distribution
            log_probs = torch.log_softmax(pred_policies, dim=1)
            p_loss = -(target_policies * log_probs).sum(dim=1).mean()
            
            # Total Loss
            loss = v_loss + p_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            p_loss_total += p_loss.item()
            v_loss_total += v_loss.item()
            batch_count += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/batch_count:.4f} (Pol: {p_loss_total/batch_count:.4f} Val: {v_loss_total/batch_count:.4f})")

    # 5. Save Model
    # We save to the OUTPUT path (Candidate), never overwriting the Input (Champion) directly
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_model_path)
    print(f"New model saved to {output_model_path}")

if __name__ == "__main__":
    # Test Run
    # This will load 'best_model.pth' and save to 'candidate.pth'
    train_model(epochs=10)