import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import sys

# Ensure we can import from the parent directory
sys.path.append(os.getcwd())

from game_engine.cnn import ChessCNN

class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.states = []
        self.policies = []
        self.values = []
        
        # Get all .npz files
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not files:
            print(f"Warning: No data found in {data_dir}")
            return

        print(f"Loading {len(files)} data files...")
        
        valid_files = 0
        for f in files:
            try:
                data = np.load(f)
                s = data['states']
                p = data['policies']
                v = data['values']
                
                # Validation: Check shapes
                # Expect state: (N, 13, 8, 8)
                if len(s.shape) != 4 or s.shape[1:] != (13, 8, 8):
                    print(f"Skipping corrupt file {f}: State shape {s.shape}")
                    continue
                    
                # Expect policy: (N, 8192)
                if len(p.shape) != 2 or p.shape[1] != 8192:
                    print(f"Skipping corrupt file {f}: Policy shape {p.shape}")
                    continue
                
                self.states.append(s)
                self.policies.append(p)
                self.values.append(v)
                valid_files += 1
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

        if valid_files > 0:
            self.states = np.concatenate(self.states)
            self.policies = np.concatenate(self.policies)
            self.values = np.concatenate(self.values)
            print(f"Dataset Loaded: {len(self.states)} positions.")
        else:
            print("CRITICAL: No valid data loaded.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': torch.tensor(self.states[idx], dtype=torch.float32),
            'policy': torch.tensor(self.policies[idx], dtype=torch.float32),
            'value': torch.tensor(self.values[idx], dtype=torch.float32)
        }

def train_model(data_path="data/self_play", 
                input_model_path="game_engine/model/best_model.pth", 
                output_model_path="game_engine/model/candidate.pth", 
                epochs=1, 
                batch_size=256, 
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