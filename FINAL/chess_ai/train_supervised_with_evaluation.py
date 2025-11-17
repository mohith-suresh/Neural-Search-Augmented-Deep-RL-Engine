"""
Supervised Training with Integrated Evaluation

Trains CNN on Lichess data with:
- Automatic Elo evaluation
- Move quality analysis
- Training visualization
- Checkpoint management
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.settings import get_config
from model.chess_encoder import ChessEncoder
from evaluation.elo_rating import StockfishBenchmark, estimate_model_elo
from evaluation.move_analyzer import MoveQualityAnalyzer


class ChessCNNWrapper:
    """Wraps CNN model for compatibility with evaluation systems"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.encoder = ChessEncoder()
    
    def predict_move(self, board):
        """
        Predict best move from position (for Elo evaluation).
        
        Required interface for evaluate_model_elo()
        """
        board_tensor = self.encoder.board_to_torch(board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits = self.model(board_tensor)[0]  # Get policy head output
        
        # Get legal moves and their probabilities
        legal_moves = list(board.legal_moves)
        legal_indices = [self.encoder.move_to_index(m) for m in legal_moves]
        
        # Find best legal move
        legal_logits = policy_logits[legal_indices]
        best_idx = legal_logits.argmax().item()
        
        return legal_moves[best_idx]


class SupervisedTrainer:
    """Train CNN on Lichess data with evaluation integration"""
    
    def __init__(self, config=None, device=None):
        self.config = config or get_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = ChessEncoder()
        
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'policy_acc': [],
            'value_acc': [],
            'elo': [],
            'avg_cp_loss': []
        }
    
    def load_dataset(self, dataset_path):
        """Load preprocessed dataset"""
        print(f"Loading dataset: {dataset_path}")
        
        data = np.load(dataset_path, allow_pickle=True)
        
        positions = torch.from_numpy(data['positions']).float().to(self.device)
        moves = torch.from_numpy(data['moves']).long().to(self.device)
        results = torch.from_numpy(data['results']).float().to(self.device)
        
        print(f"Dataset loaded:")
        print(f"  Positions: {positions.shape}")
        print(f"  Moves: {moves.shape}")
        print(f"  Results: {results.shape}")
        
        # Create train/val/test split
        n = len(positions)
        train_size = int(n * self.config.TRAINING_DATA['train_split'])
        val_size = int(n * self.config.TRAINING_DATA['val_split'])
        
        train_dataset = TensorDataset(
            positions[:train_size],
            moves[:train_size],
            results[:train_size]
        )
        
        val_dataset = TensorDataset(
            positions[train_size:train_size + val_size],
            moves[train_size:train_size + val_size],
            results[train_size:train_size + val_size]
        )
        
        test_dataset = TensorDataset(
            positions[train_size + val_size:],
            moves[train_size + val_size:],
            results[train_size + val_size:]
        )
        
        batch_size = self.config.TRAINING_DATA['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion_policy, criterion_value):
        """Train one epoch"""
        model.train()
        total_loss = 0
        policy_correct = 0
        policy_total = 0
        
        for positions, moves, results in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            policy_logits, value_pred = model(positions)
            
            # Policy loss (move prediction)
            policy_loss = criterion_policy(policy_logits, moves)
            
            # Value loss (game outcome prediction)
            value_loss = criterion_value(value_pred.squeeze(), results)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy tracking
            _, predicted_moves = policy_logits.max(1)
            policy_correct += (predicted_moves == moves).sum().item()
            policy_total += moves.size(0)
        
        avg_loss = total_loss / len(train_loader)
        policy_acc = 100 * policy_correct / policy_total
        
        return avg_loss, policy_acc
    
    def evaluate(self, model, val_loader, criterion_policy, criterion_value):
        """Evaluate on validation set"""
        model.eval()
        total_loss = 0
        policy_correct = 0
        policy_total = 0
        value_rmse = 0
        
        with torch.no_grad():
            for positions, moves, results in val_loader:
                policy_logits, value_pred = model(positions)
                
                policy_loss = criterion_policy(policy_logits, moves)
                value_loss = criterion_value(value_pred.squeeze(), results)
                
                loss = policy_loss + value_loss
                total_loss += loss.item()
                
                _, predicted_moves = policy_logits.max(1)
                policy_correct += (predicted_moves == moves).sum().item()
                policy_total += moves.size(0)
                
                value_rmse += ((value_pred.squeeze() - results) ** 2).mean().item()
        
        avg_loss = total_loss / len(val_loader)
        policy_acc = 100 * policy_correct / policy_total
        avg_rmse = value_rmse / len(val_loader)
        
        return avg_loss, policy_acc, avg_rmse
    
    def train(self, model, train_loader, val_loader, num_epochs=100):
        """Full training loop"""
        print("\n" + "="*70)
        print("STARTING SUPERVISED TRAINING")
        print("="*70 + "\n")
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.TRAINING['learning_rate'])
        criterion_policy = nn.CrossEntropyLoss()
        criterion_value = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion_policy, criterion_value
            )
            
            # Validate
            val_loss, val_acc, val_rmse = self.evaluate(
                model, val_loader, criterion_policy, criterion_value
            )
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.1f}% | RMSE: {val_rmse:.4f}")
            
            # Track history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['policy_acc'].append(train_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                
                # Save checkpoint
                checkpoint_path = self.config.LOGGING['checkpoint_dir'] / f"best_model.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            else:
                patience += 1
                if patience >= self.config.TRAINING['early_stopping_patience']:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Periodic Elo evaluation
            if (epoch + 1) % 10 == 0:
                print("\nRunning Elo evaluation...")
                model_wrapper = ChessCNNWrapper(model, self.device)
                result = estimate_model_elo(model_wrapper, num_games=20, quick=True)
                
                if result.get("estimated_elo"):
                    self.training_history['elo'].append(result['estimated_elo'])
                    print(f"Estimated Elo: {result['estimated_elo']}")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70 + "\n")
    
    def save_history(self, filepath):
        """Save training history"""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"History saved: {filepath}")
    
    def plot_training(self, output_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.training_history['epoch'], self.training_history['policy_acc'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Policy Accuracy (%)')
        axes[0, 1].set_title('Move Prediction Accuracy')
        axes[0, 1].grid(True)
        
        # Elo
        if self.training_history['elo']:
            elo_epochs = [self.training_history['epoch'][i*10+9] for i in range(len(self.training_history['elo']))]
            axes[1, 0].plot(elo_epochs, self.training_history['elo'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Elo Rating')
            axes[1, 0].set_title('Estimated Elo Rating')
            axes[1, 0].grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"Plot saved: {output_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Configuration loaded from: {config.__class__.__name__}")
    
    # This would be filled in with actual model definition
    print("\nNote: Instantiate your CNN model and pass to trainer")
