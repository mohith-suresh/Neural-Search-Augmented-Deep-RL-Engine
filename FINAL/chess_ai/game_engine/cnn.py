#!/usr/bin/env python3
"""
Chess CNN Model - Enhanced Architecture for Supervised Learning
================================================================

Project: EE542 - Deconstructing AlphaZero's Success
Goal: Supervised learning baseline for chess position evaluation

Architecture Design:
- Input: 12 planes (6 piece types x 2 colors) x 8x8 board
- Backbone: 10 ResNet blocks (7 standard + 3 with SE blocks)
- Heads: Dual-head (policy for moves, value for outcome)
- Regularization: Dropout in heads only (AlphaZero approach)
- Training: Mixed precision (FP16) for 2x speedup

Key Improvements in This Version:
1. SE blocks in last 3 residual blocks (channel attention)
2. AlphaZero-style policy head (Conv→Flatten→FC, no bottleneck)
3. Dropout removed from residual path (better gradient flow)
4. Mixed precision enabled (2x faster on RTX 3060)
5. Value head metrics tracked (MAE, MSE, correlation)
6. Data split optimized (90/5/5 for large dataset)

References:
- Silver et al. (2017): Mastering Chess Without Human Knowledge (AlphaZero)
- He et al. (2016): Identity Mappings in Deep Residual Networks
- Hu et al. (2018): Squeeze-and-Excitation Networks
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast  
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration."""
    
    # Model architecture
    input_channels: int = 12  # 6 piece types x 2 colors
    board_size: int = 8
    filters: int = 128  # ResNet block channels
    num_res_blocks: int = 10
    policy_output_size: int = 8192  # All possible moves (64x64 from-to + promotions)
    value_output_size: int = 1  # Single outcome prediction
    
    # Training hyperparameters
    batch_size: int = 384
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4  # L2 regularization
    dropout_rate: float = 0.3  # Only in heads
    
    # Learning rate schedule
    warmup_epochs: float = 0.1  # UPDATED: 10% of epochs for warmup (was 0.5)
    max_lr: float = 1e-3
    div_factor: float = 25.0  # Initial LR = max_lr / div_factor
    final_div_factor: float = 1e4  # Final LR = max_lr / final_div_factor
    
    # Regularization
    gradient_clip_value: float = 1.0
    label_smoothing: float = 0.0  # Disabled for now (start without)
    
    # Data split - UPDATED: 90/5/5 for large datasets
    train_split: float = 0.90  # 18M positions
    val_split: float = 0.05    # 1M positions
    test_split: float = 0.05   # 1M positions
    
    # Mixed precision - FIX: Re-enabled
    use_mixed_precision: bool = True  # FP16 training for RTX 3060
    
    # Checkpointing
    save_frequency: int = 1  # Save every epoch
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_frequency: int = 100  # Log every N batches
    
    # Paths - FIX: Corrected absolute paths
    data_dir: Path = Path("/home/krish/EE542-Project/FINAL/chess_ai/data/training_data")
    model_dir: Path = Path("/home/krish/EE542-Project/FINAL/chess_ai/game_engine/model")
    log_dir: Path = Path("/home/krish/EE542-Project/FINAL/chess_ai/logs/tensorboard")  # FIX: Added /tensorboard
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# NEW: Squeeze-and-Excitation Block (Channel Attention)
# ============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Allows network to dynamically reweight feature channels based on
    global context (e.g., emphasize knight channels in fork positions).
    
    Reference: Hu et al. (2018) "Squeeze-and-Excitation Networks"
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Args:
            channels: Number of input/output channels
            reduction: Reduction ratio for bottleneck (4 for 128 channels → 32)
        """
        super().__init__()
        
        # Squeeze: Global average pooling (8x8 → 1x1 per channel)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Bottleneck FC layers
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature maps
        
        Returns:
            Reweighted features (B, C, H, W)
        """
        b, c, _, _ = x.size()
        
        # Squeeze: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        y = self.squeeze(x).view(b, c)
        
        # Excitation: (B, C) → (B, C) with learned weights
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale: Apply channel weights
        return x * y.expand_as(x)

# ============================================================================
# Residual Block (Enhanced with Optional SE)
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with optional SE attention.
    
    FIX: Dropout removed from residual path (AlphaZero approach).
    Only used in final policy/value heads.
    """
    
    def __init__(self, channels: int, use_se: bool = False):
        """
        Args:
            channels: Number of input/output channels
            use_se: Whether to include SE block (True for last 3 blocks)
        """
        super().__init__()
        
        # Main path: Conv → BN → ReLU → Conv → BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # NEW: Optional SE block
        self.se = SEBlock(channels, reduction=4) if use_se else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
        
        Returns:
            Residual output (B, C, H, W)
        """
        residual = x
        
        # First conv block
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv block
        out = self.bn2(self.conv2(out))
        
        # NEW: Apply SE attention if enabled
        if self.se is not None:
            out = self.se(out)
        
        # FIX: Residual addition (no dropout on residual path)
        out = out + residual
        
        # Final activation
        out = F.relu(out)
        
        return out

# ============================================================================
# Chess CNN Model
# ============================================================================

class ChessCNN(nn.Module):
    """
    Chess CNN with ResNet backbone and dual policy/value heads.
    
    Architecture:
        Input (12x8x8)
        → Conv 12→128
        → ResBlocks 1-7 (standard)
        → ResBlocks 8-10 (with SE blocks)
        → Policy Head (Conv→Flatten→FC→8192)
        → Value Head (Conv→GlobalPool→FC→1)
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Initial convolution: 12 planes → 128 filters
        self.input_conv = nn.Conv2d(
            config.input_channels,
            config.filters,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(config.filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            # Blocks 1-7: Standard residual blocks
            ResidualBlock(config.filters, use_se=False) for _ in range(7)
        ] + [
            # NEW: Blocks 8-10: With SE attention
            ResidualBlock(config.filters, use_se=True) for _ in range(3)
        ])
        
        # FIX: AlphaZero-style policy head (no bottleneck)
        self.policy_conv = nn.Conv2d(config.filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * config.board_size * config.board_size, config.policy_output_size)
        self.policy_dropout = nn.Dropout(config.dropout_rate)
        
        # Value head (unchanged, already AlphaZero-style)
        self.value_conv = nn.Conv2d(config.filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(config.board_size * config.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 12, 8, 8) board positions
        
        Returns:
            policy_logits: (B, 8192) move probabilities (raw logits)
            values: (B, 1) position evaluation (-1 to +1)
        """
        # Initial convolution
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual blocks (7 standard + 3 with SE)
        for block in self.res_blocks:
            x = block(x)
        
        # FIX: Policy head (AlphaZero-style, preserves spatial structure)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))  # (B, 32, 8, 8)
        policy = policy.view(policy.size(0), -1)  # (B, 2048)
        policy = self.policy_dropout(policy)
        policy_logits = self.policy_fc(policy)  # (B, 8192)
        
        # Value head (global pooling → FC)
        value = F.relu(self.value_bn(self.value_conv(x)))  # (B, 1, 8, 8)
        value = value.view(value.size(0), -1)  # (B, 64)
        value = F.relu(self.value_fc1(value))  # (B, 256)
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value))  # (B, 1), range [-1, 1]
        
        return policy_logits, value

# ============================================================================
# Dataset
# ============================================================================

class ChessDataset(Dataset):
    """
    Lazy-loading chess dataset from multiple chunk files.
    
    Only loads one chunk at a time during iteration (minimal memory footprint).
    
    Expected format per chunk:
        positions: (N, 12, 8, 8) float16
        moves: (N,) int32 (move indices 0-8191)
        results: (N,) float16 (game outcomes -1/0/+1)
    """
    
    def __init__(self, chunk_dir: Path):
        """
        Args:
            chunk_dir: Directory containing chunk_*.npz files
        """
        # Find all chunk files
        self.chunk_files = sorted(Path(chunk_dir).glob('chunk_*.npz'))
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
        
        # Get chunk sizes without loading data
        self.chunk_sizes = []
        for f in self.chunk_files:
            with np.load(f, mmap_mode='r') as data:
                self.chunk_sizes.append(len(data['positions']))
        
        # Calculate cumulative sizes for indexing
        self.cumulative_sizes = np.cumsum([0] + self.chunk_sizes)
        self.total_size = sum(self.chunk_sizes)
        
        # Cache for currently loaded chunk
        self._current_chunk_idx = None
        self._current_chunk_data = None
        
        print(f"Dataset: {len(self.chunk_files)} chunks, {self.total_size:,} total positions")
    
    def __len__(self):
        return self.total_size
    
    def _load_chunk(self, chunk_idx: int):
        """Load a specific chunk into memory (only if not already loaded)."""
        if self._current_chunk_idx != chunk_idx:
            # Unload previous chunk
            if self._current_chunk_data is not None:
                del self._current_chunk_data
            
            # Load new chunk
            data = np.load(self.chunk_files[chunk_idx])
            self._current_chunk_data = {
                'positions': data['positions'],
                'moves': data['moves'],
                'results': data['results']
            }
            self._current_chunk_idx = chunk_idx
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            position: (12, 8, 8) board state
            move: (,) move index
            result: (,) game outcome
        """
        # Find which chunk contains this index
        chunk_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[chunk_idx]
        
        # Load chunk if needed (lazy loading)
        self._load_chunk(chunk_idx)
        
        # Return position from currently loaded chunk
        return (
            torch.from_numpy(self._current_chunk_data['positions'][local_idx]).float(),
            torch.tensor(int(self._current_chunk_data['moves'][local_idx]), dtype=torch.long),
            torch.tensor(float(self._current_chunk_data['results'][local_idx]), dtype=torch.float)
        )


# ============================================================================
# NEW: Value Head Metrics
# ============================================================================

def compute_value_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute value head performance metrics.
    
    Args:
        predictions: (B, 1) predicted values
        targets: (B, 1) actual outcomes
    
    Returns:
        Dictionary with mae, mse, correlation
    """
    predictions = predictions.squeeze()
    targets = targets.squeeze()
    
    # Mean Absolute Error (interpretable)
    mae = F.l1_loss(predictions, targets).item()
    
    # Mean Squared Error (optimization metric)
    mse = F.mse_loss(predictions, targets).item()
    
    # Correlation (ranking quality)
    if len(predictions) > 1:
        pred_centered = predictions - predictions.mean()
        target_centered = targets - targets.mean()
        correlation = (pred_centered * target_centered).sum() / \
                     (torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-8)
        correlation = correlation.item()
    else:
        correlation = 0.0
    
    return {
        'mae': mae,
        'mse': mse,
        'correlation': correlation
    }

# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Chess CNN trainer with comprehensive logging and checkpointing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Model
        self.model = ChessCNN(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        self.value_loss_fn = nn.MSELoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # FIX: Mixed precision scaler (re-enabled)
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Logging
        self.writer = None
        self.global_step = 0
        
        # Best model tracking
        self.best_val_loss = float('inf')

        # Track per-epoch metrics for evaluation
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_policy_loss': [],
            'val_policy_loss': [],
            'train_value_loss': [],
            'val_value_loss': []
        }

    
    def load_dataset(self, config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and split dataset from chunk files.
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Load from chunk directory (lazy loading)
        chunk_dir = config.data_dir
        
        print(f"Loading dataset from: {chunk_dir}")
        dataset = ChessDataset(chunk_dir)  # Stores chunk paths, doesn't load data yet
        
        # Split 90/5/5
        train_size = int(config.train_split * len(dataset))
        val_size = int(config.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

    def setup_training(self, train_loader: DataLoader):
        """Setup LR scheduler and logging."""
        
        # Learning rate scheduler (OneCycle)
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config.num_epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.max_lr,
            total_steps=total_steps,
            div_factor=self.config.div_factor,
            final_div_factor=self.config.final_div_factor,
            pct_start=self.config.warmup_epochs  # FIX: Now correctly 10%
        )
        
        # TensorBoard
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        
        # Checkpoint directory
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        policy_correct = 0
        total_samples = 0
        
        # NEW: Value metrics tracking
        all_value_preds = []
        all_value_targets = []
        
        for batch_idx, (positions, moves, results) in enumerate(train_loader):
            print(f"Moving batch {batch_idx} to {self.device}", flush=True)

            positions = positions.to(self.device)
            moves = moves.to(self.device)
            results = results.to(self.device).unsqueeze(1)  # (B, 1)
            
            self.optimizer.zero_grad()
            
            # FIX: Mixed precision forward pass
            if self.config.use_mixed_precision:
                with autocast():
                    policy_logits, values = self.model(positions)
                    policy_loss = self.policy_loss_fn(policy_logits, moves)
                    value_loss = self.value_loss_fn(values, results)
                    total_loss = policy_loss + value_loss
                
                # Backward with gradient scaling
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # FP32 fallback
                policy_logits, values = self.model(positions)
                policy_loss = self.policy_loss_fn(policy_logits, moves)
                value_loss = self.value_loss_fn(values, results)
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Metrics
            policy_pred = policy_logits.argmax(dim=1)
            policy_correct += (policy_pred == moves).sum().item()
            total_samples += len(positions)
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            
            # NEW: Collect value predictions for metrics
            all_value_preds.append(values.detach())
            all_value_targets.append(results.detach())
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Policy: {policy_loss.item():.4f} | "
                      f"Value: {value_loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")
                
                if self.writer:
                    self.writer.add_scalar('train/policy_loss', policy_loss.item(), self.global_step)
                    self.writer.add_scalar('train/value_loss', value_loss.item(), self.global_step)
                    self.writer.add_scalar('train/total_loss', total_loss.item(), self.global_step)
                    self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)
            
            self.global_step += 1
        
        # Epoch metrics
        num_batches = len(train_loader)
        policy_accuracy = policy_correct / total_samples
        
        # NEW: Compute value metrics
        all_value_preds = torch.cat(all_value_preds, dim=0)
        all_value_targets = torch.cat(all_value_targets, dim=0)
        value_metrics = compute_value_metrics(all_value_preds, all_value_targets)
        
        return {
            'policy_loss': epoch_policy_loss / num_batches,
            'value_loss': epoch_value_loss / num_batches,
            'total_loss': epoch_total_loss / num_batches,
            'policy_accuracy': policy_accuracy,
            'value_mae': value_metrics['mae'],
            'value_mse': value_metrics['mse'],
            'value_correlation': value_metrics['correlation']
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        
        self.model.eval()
        
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_total_loss = 0.0
        policy_correct = 0
        total_samples = 0
        
        # NEW: Value metrics tracking
        all_value_preds = []
        all_value_targets = []
        
        for positions, moves, results in val_loader:
            positions = positions.to(self.device)
            moves = moves.to(self.device)
            results = results.to(self.device).unsqueeze(1)
            
            # FIX: Mixed precision validation
            if self.config.use_mixed_precision:
                with autocast():
                    policy_logits, values = self.model(positions)
                    policy_loss = self.policy_loss_fn(policy_logits, moves)
                    value_loss = self.value_loss_fn(values, results)
                    total_loss = policy_loss + value_loss
            else:
                policy_logits, values = self.model(positions)
                policy_loss = self.policy_loss_fn(policy_logits, moves)
                value_loss = self.value_loss_fn(values, results)
                total_loss = policy_loss + value_loss
            
            policy_pred = policy_logits.argmax(dim=1)
            policy_correct += (policy_pred == moves).sum().item()
            total_samples += len(positions)
            
            val_policy_loss += policy_loss.item()
            val_value_loss += value_loss.item()
            val_total_loss += total_loss.item()
            
            # NEW: Collect value predictions
            all_value_preds.append(values)
            all_value_targets.append(results)
        
        num_batches = len(val_loader)
        policy_accuracy = policy_correct / total_samples
        
        # NEW: Compute value metrics
        all_value_preds = torch.cat(all_value_preds, dim=0)
        all_value_targets = torch.cat(all_value_targets, dim=0)
        value_metrics = compute_value_metrics(all_value_preds, all_value_targets)
        
        return {
            'policy_loss': val_policy_loss / num_batches,
            'value_loss': val_value_loss / num_batches,
            'total_loss': val_total_loss / num_batches,
            'policy_accuracy': policy_accuracy,
            'value_mae': value_metrics['mae'],
            'value_mse': value_metrics['mse'],
            'value_correlation': value_metrics['correlation']
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save latest
        checkpoint_path = self.config.model_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best
        if is_best:
            best_path = self.config.model_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def train(self):
        """Main training loop."""
        
        print("=" * 80)
        print("CHESS CNN TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.config.use_mixed_precision}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print("=" * 80)
        
        # Load dataset
        train_loader, val_loader, test_loader = self.load_dataset(self.config)
        
        # Setup
        self.setup_training(train_loader)
        
        # Training loop
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}/{self.config.num_epochs}")
            print(f"{'='*80}")
            
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)

            # Accumulate per-epoch metrics
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            self.training_history['train_accuracy'].append(train_metrics['policy_accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['policy_accuracy'])
            self.training_history['train_policy_loss'].append(train_metrics['policy_loss'])
            self.training_history['val_policy_loss'].append(val_metrics['policy_loss'])
            self.training_history['train_value_loss'].append(train_metrics['value_loss'])
            self.training_history['val_value_loss'].append(val_metrics['value_loss'])

            
            epoch_time = time.time() - start_time
            
            # Print metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Train - Policy Loss: {train_metrics['policy_loss']:.4f} | "
                  f"Value Loss: {train_metrics['value_loss']:.4f} | "
                  f"Policy Acc: {train_metrics['policy_accuracy']:.4f}")
            print(f"  Train - Value MAE: {train_metrics['value_mae']:.4f} | "
                  f"Value MSE: {train_metrics['value_mse']:.4f} | "
                  f"Value Corr: {train_metrics['value_correlation']:.4f}")
            print(f"  Val   - Policy Loss: {val_metrics['policy_loss']:.4f} | "
                  f"Value Loss: {val_metrics['value_loss']:.4f} | "
                  f"Policy Acc: {val_metrics['policy_accuracy']:.4f}")
            print(f"  Val   - Value MAE: {val_metrics['value_mae']:.4f} | "
                  f"Value MSE: {val_metrics['value_mse']:.4f} | "
                  f"Value Corr: {val_metrics['value_correlation']:.4f}")
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('epoch/train_policy_loss', train_metrics['policy_loss'], epoch)
                self.writer.add_scalar('epoch/train_value_loss', train_metrics['value_loss'], epoch)
                self.writer.add_scalar('epoch/train_policy_accuracy', train_metrics['policy_accuracy'], epoch)
                self.writer.add_scalar('epoch/train_value_mae', train_metrics['value_mae'], epoch)
                self.writer.add_scalar('epoch/train_value_correlation', train_metrics['value_correlation'], epoch)
                
                self.writer.add_scalar('epoch/val_policy_loss', val_metrics['policy_loss'], epoch)
                self.writer.add_scalar('epoch/val_value_loss', val_metrics['value_loss'], epoch)
                self.writer.add_scalar('epoch/val_policy_accuracy', val_metrics['policy_accuracy'], epoch)
                self.writer.add_scalar('epoch/val_value_mae', val_metrics['value_mae'], epoch)
                self.writer.add_scalar('epoch/val_value_correlation', val_metrics['value_correlation'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
            
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(epoch, is_best)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Evaluate on test set
        print("\n" + "=" * 80)
        print("EVALUATING ON FINAL TEST SET")
        print("=" * 80)
        test_metrics = self.validate(test_loader)
        
        print(f"  Test - Policy Loss: {test_metrics['policy_loss']:.4f} | "
              f"Value Loss: {test_metrics['value_loss']:.4f} | "
              f"Policy Acc: {test_metrics['policy_accuracy']:.4f}")
        print(f"  Test - Value MAE: {test_metrics['value_mae']:.4f} | "
              f"Value MSE: {test_metrics['value_mse']:.4f} | "
              f"Value Corr: {test_metrics['value_correlation']:.4f}")
        
        # Save test metrics
        metrics_path = self.config.model_dir / "test_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("Test metrics after final epoch:\n")
            for k, v in test_metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"\nTest metrics saved to {metrics_path}")

        if self.writer:
            self.writer.close()

# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    
    # Configuration
    config = TrainingConfig()
    
    # Train
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
