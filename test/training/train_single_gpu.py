"""
Single GPU Training Script for Chess CNN
Optimized for 4GB GPU RAM with 500K dataset
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import os
from pathlib import Path

print("=" * 70)
print("SYSTEM INFORMATION")
print("=" * 70)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ WARNING: No GPU detected! Training will be very slow on CPU.")

print("=" * 70 + "\n")


# ============================================
# MODEL DEFINITION (OPTIMIZED FOR 4GB GPU)
# ============================================

class SimpleChessCNN(nn.Module):
    """
    Lightweight CNN optimized for 4GB GPU RAM
    Reduced filters to fit in limited memory
    """
    def __init__(self, num_filters=64, num_layers=4):  # Reduced from 128/5
        super(SimpleChessCNN, self).__init__()

        # Input layer: 12 channels (pieces) → num_filters
        self.conv1 = nn.Conv2d(12, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Hidden convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_filters, num_filters, 3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(num_filters))

        # Policy head (move prediction)
        # Dataset uses: from_square * 64 + to_square (0-4095) for normal moves
        #               from_square * 64 + to_square + 4096 (4096-8191) for promotions
        # So we need 8192 output classes
        self.policy_conv = nn.Conv2d(num_filters, 16, kernel_size=1)  # Reduced from 32 to 16
        self.policy_fc = nn.Linear(16 * 8 * 8, 8192)

        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # Hidden layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))

        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================
# DATASET
# ============================================

class ChessDataset(Dataset):
    def __init__(self, npz_file):
        print(f"Loading dataset from {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)

        self.positions = torch.from_numpy(data['positions']).float()
        self.moves = torch.from_numpy(data['moves']).long()
        self.results = torch.from_numpy(data['results']).float()

        print(f"✓ Loaded {len(self.positions):,} positions")
        print(f"  Positions shape: {self.positions.shape}")
        print(f"  Moves shape: {self.moves.shape}")
        print(f"  Results shape: {self.results.shape}")

        # Show metadata if available
        try:
            if 'metadata' in data:
                metadata = data['metadata'].item()
                print(f"\n  Dataset Metadata:")
                print(f"    Min ELO: {metadata.get('min_elo', 'N/A')}")
                print(f"    Time control: {metadata.get('time_control_min', 'N/A')}s+")
                print(f"    Games processed: {metadata.get('games_processed', 'N/A'):,}")
                print(f"    Games used: {metadata.get('games_used', 'N/A'):,}")
        except Exception as e:
            print(f"  (Could not load metadata)")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx], self.results[idx]


# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_epoch(model, loader, optimizer, device, epoch_num):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch_num} [Train]")
    for batch_idx, (positions, moves, results) in enumerate(pbar):
        # Move data to GPU
        positions = positions.to(device, non_blocking=True)
        moves = moves.to(device, non_blocking=True)
        results = results.to(device, non_blocking=True).unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        policy_logits, value_pred = model(positions)

        # Calculate losses
        policy_loss = policy_criterion(policy_logits, moves)
        value_loss = value_criterion(value_pred, results)
        loss = policy_loss + value_loss

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'policy': f'{policy_loss.item():.4f}',
            'value': f'{value_loss.item():.4f}'
        })

    elapsed = time.time() - start_time

    return (
        total_loss / len(loader),
        policy_loss_sum / len(loader),
        value_loss_sum / len(loader),
        elapsed
    )


def validate(model, loader, device, epoch_num):
    """Validate the model"""
    model.eval()

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    correct = 0
    total = 0

    # Top-3 accuracy
    correct_top3 = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch_num} [Val]  ")
        for positions, moves, results in pbar:
            # Move data to GPU
            positions = positions.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            results = results.to(device, non_blocking=True).unsqueeze(1)

            # Forward pass
            policy_logits, value_pred = model(positions)

            # Calculate losses
            policy_loss = policy_criterion(policy_logits, moves)
            value_loss = value_criterion(value_pred, results)
            loss = policy_loss + value_loss

            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()

            # Top-1 accuracy
            _, predicted = policy_logits.max(1)
            correct += predicted.eq(moves).sum().item()

            # Top-3 accuracy
            _, top3_pred = policy_logits.topk(3, dim=1)
            correct_top3 += sum([moves[i] in top3_pred[i] for i in range(len(moves))])

            total += moves.size(0)

            # Update progress bar
            current_acc = correct / total * 100
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    return (
        total_loss / len(loader),
        policy_loss_sum / len(loader),
        value_loss_sum / len(loader),
        correct / total,
        correct_top3 / total
    )


# ============================================
# MAIN TRAINING SCRIPT
# ============================================

def main():
    # ===== CONFIGURATION =====
    DATASET_PATH = '../outputs/chess_elo1900_500K.npz'
    BATCH_SIZE = 64  # Reduced to 64 for 4GB GPU (was 128)
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # Check if dataset exists
    if not Path(DATASET_PATH).exists():
        print(f"❌ ERROR: Dataset not found at {DATASET_PATH}")
        print("Please create the dataset first using create_dataset.py")
        return

    # ===== DEVICE SETUP =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        # Clear GPU cache
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")

    # ===== LOAD DATASET =====
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70 + "\n")

    dataset = ChessDataset(DATASET_PATH)

    # Split into train/val (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\n📊 Dataset Split:")
    print(f"  Training: {len(train_dataset):,} positions")
    print(f"  Validation: {len(val_dataset):,} positions")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"\n📦 DataLoader Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ===== CREATE MODEL =====
    print("\n" + "=" * 70)
    print("MODEL SETUP")
    print("=" * 70 + "\n")

    model = SimpleChessCNN(num_filters=32, num_layers=3).to(device)  # Further reduced for 4GB GPU

    print(f"🔧 Model Architecture:")
    print(f"  Filters: 32 (optimized for 4GB GPU)")
    print(f"  Layers: 3")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Model size: ~{count_parameters(model) * 4 / (1024**2):.1f} MB")

    # ===== OPTIMIZER =====
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\n📐 Training Configuration:")
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Epochs: {NUM_EPOCHS}")

    # ===== GPU MEMORY CHECK =====
    if device.type == 'cuda':
        print(f"\n💾 GPU Memory Status:")
        print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    # ===== TRAINING LOOP =====
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_accuracy_top3': [],
        'epoch_time': []
    }

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*70}")

        # Train
        train_loss, train_policy, train_value, train_time = train_epoch(
            model, train_loader, optimizer, device, epoch + 1
        )

        # Validate
        val_loss, val_policy, val_value, val_acc, val_acc_top3 = validate(
            model, val_loader, device, epoch + 1
        )

        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_acc)
        training_history['val_accuracy_top3'].append(val_acc_top3)
        training_history['epoch_time'].append(train_time)

        # Print results
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"  Training:")
        print(f"    Total Loss:  {train_loss:.4f}")
        print(f"    Policy Loss: {train_policy:.4f}")
        print(f"    Value Loss:  {train_value:.4f}")
        print(f"    Time:        {train_time/60:.2f} min")
        print(f"  Validation:")
        print(f"    Total Loss:  {val_loss:.4f}")
        print(f"    Policy Loss: {val_policy:.4f}")
        print(f"    Value Loss:  {val_value:.4f}")
        print(f"    Top-1 Acc:   {val_acc*100:.2f}%")
        print(f"    Top-3 Acc:   {val_acc_top3*100:.2f}%")

        # GPU memory
        if device.type == 'cuda':
            print(f"  GPU Memory:")
            print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"    Peak:      {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, 'best_chess_model.pth')
            print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")

        # Learning rate decay
        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"  📉 Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")

    # ===== TRAINING COMPLETE =====
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)

    print(f"\n📈 Training Summary:")
    print(f"  Total epochs: {NUM_EPOCHS}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val accuracy: {max(training_history['val_accuracy'])*100:.2f}%")
    print(f"  Final val accuracy: {training_history['val_accuracy'][-1]*100:.2f}%")
    print(f"  Final val top-3 acc: {training_history['val_accuracy_top3'][-1]*100:.2f}%")
    print(f"  Total training time: {sum(training_history['epoch_time'])/60:.1f} min")
    print(f"  Avg time per epoch: {sum(training_history['epoch_time'])/len(training_history['epoch_time'])/60:.1f} min")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history,
        'final_val_accuracy': training_history['val_accuracy'][-1],
    }, 'chess_model_final.pth')

    print("\n💾 Models saved:")
    print("  • best_chess_model.pth (best validation loss)")
    print("  • chess_model_final.pth (final epoch)")

    # Final GPU memory
    if device.type == 'cuda':
        print(f"\n💾 Final GPU Memory Usage:")
        print(f"  Peak allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
