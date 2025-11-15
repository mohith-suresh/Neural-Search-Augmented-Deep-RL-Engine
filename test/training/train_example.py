"""
Example training script using memory-efficient DataLoader
Demonstrates training on 20M dataset with only 4GB RAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add data directory to path
sys.path.append(str(Path(__file__).parent.parent / 'data'))
from memory_efficient_dataloader import create_dataloader

# Simple CNN model (you'll use a bigger one)
class SimpleChessNet(nn.Module):
    """
    Lightweight CNN for demonstration
    Real model would be deeper (ResNet, etc.)
    """

    def __init__(self, num_moves=4672):
        super().__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Policy head (move prediction)
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, num_moves)
        )

        # Value head (position evaluation)
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.conv_layers(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0

    for batch_idx, (positions, moves, results) in enumerate(dataloader):
        # Move to device (GPU if available)
        positions = positions.to(device)
        moves = moves.to(device)
        results = results.to(device).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        policy_logits, value_pred = model(positions)

        # Calculate losses
        policy_loss = policy_criterion(policy_logits, moves)
        value_loss = value_criterion(value_pred, results)
        loss = policy_loss + value_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Policy: {policy_loss.item():.4f} | "
                  f"Value: {value_loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_policy = total_policy_loss / num_batches
    avg_value = total_value_loss / num_batches

    print(f"\nEpoch {epoch} Complete:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg Policy Loss: {avg_policy:.4f}")
    print(f"  Avg Value Loss: {avg_value:.4f}\n")

    return avg_loss


def main():
    """
    Main training loop with memory-efficient data loading
    """
    import psutil
    import os

    def get_ram_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    print("\n" + "=" * 70)
    print("MEMORY-EFFICIENT CHESS MODEL TRAINING")
    print("=" * 70)

    # Configuration
    DATASET_FILE = "../outputs/chess_elo1900_500K.npz"  # Change to 20M file
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    NUM_WORKERS = 2

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Initial RAM usage: {get_ram_usage():.0f} MB\n")

    # Create DataLoader with memory mapping
    print("Creating DataLoader...")
    ram_before = get_ram_usage()

    train_loader = create_dataloader(
        DATASET_FILE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        memory_mapped=True  # ← KEY: Use memory mapping!
    )

    ram_after = get_ram_usage()
    print(f"RAM after DataLoader creation: {ram_after:.0f} MB (+{ram_after - ram_before:.0f} MB)\n")

    # Create model
    print("Creating model...")
    ram_before = get_ram_usage()

    model = SimpleChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ram_after = get_ram_usage()
    print(f"RAM after model creation: {ram_after:.0f} MB (+{ram_after - ram_before:.0f} MB)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.0f} MB\n")

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 70)

        ram_before_epoch = get_ram_usage()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        ram_after_epoch = get_ram_usage()
        ram_peak = ram_after_epoch

        print(f"RAM usage: {ram_peak:.0f} MB (peak during epoch)")
        print(f"RAM increase from start: +{ram_peak - ram_before:.0f} MB")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal RAM usage: {get_ram_usage():.0f} MB")
    print("\n✅ Successfully trained on large dataset without RAM crash!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
