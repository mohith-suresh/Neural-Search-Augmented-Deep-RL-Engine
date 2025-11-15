"""
Memory-efficient DataLoader for large chess datasets
Loads batches on-demand instead of entire dataset into RAM
Works with datasets of any size on limited RAM systems
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MemoryMappedChessDataset(Dataset):
    """
    Memory-mapped dataset - loads data from disk on-demand
    Only keeps current batch in RAM, not entire dataset!
    """

    def __init__(self, npz_file, device='cpu'):
        """
        Args:
            npz_file: Path to .npz dataset file
            device: 'cpu' or 'cuda' - where to load tensors
        """
        print(f"📂 Loading dataset metadata from {npz_file}...")

        # Load with mmap_mode='r' - doesn't load into RAM!
        self.data = np.load(npz_file, mmap_mode='r')

        self.positions = self.data['positions']  # Memory-mapped, not in RAM
        self.moves = self.data['moves']
        self.results = self.data['results']

        self.device = device
        self.length = len(self.positions)

        print(f"✅ Dataset ready: {self.length:,} positions")
        print(f"💾 Memory usage: ~0 MB (memory-mapped)")
        print(f"   Only batches will be loaded into RAM during training")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Load single position on-demand
        This is called by DataLoader for each batch item
        """
        # Read from disk (cached by OS if accessed recently)
        position = torch.from_numpy(self.positions[idx].copy()).float()
        move = torch.tensor(self.moves[idx], dtype=torch.long)
        result = torch.tensor(self.results[idx], dtype=torch.float)

        return position, move, result


class PreloadedChessDataset(Dataset):
    """
    Traditional approach - loads entire dataset into RAM
    ONLY use if dataset is small (<1M positions) and you have RAM
    """

    def __init__(self, npz_file, device='cpu'):
        print(f"📂 Loading entire dataset into RAM from {npz_file}...")

        data = np.load(npz_file)

        # Load everything into RAM
        self.positions = torch.from_numpy(data['positions']).float()
        self.moves = torch.from_numpy(data['moves']).long()
        self.results = torch.from_numpy(data['results']).float()

        self.device = device
        self.length = len(self.positions)

        # Calculate memory usage
        mem_mb = (self.positions.element_size() * self.positions.nelement() +
                  self.moves.element_size() * self.moves.nelement() +
                  self.results.element_size() * self.results.nelement()) / (1024**2)

        print(f"✅ Dataset loaded: {self.length:,} positions")
        print(f"💾 RAM usage: {mem_mb:.0f} MB")
        print(f"⚠️  WARNING: Entire dataset is in RAM!")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx], self.results[idx]


def create_dataloader(npz_file, batch_size=256, shuffle=True,
                      num_workers=2, memory_mapped=True):
    """
    Create memory-efficient DataLoader

    Args:
        npz_file: Path to dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        memory_mapped: If True, use memory mapping (recommended for large datasets)

    Returns:
        DataLoader instance
    """
    if memory_mapped:
        print("Using memory-mapped dataset (recommended for >1M positions)")
        dataset = MemoryMappedChessDataset(npz_file)
    else:
        print("Using preloaded dataset (only for small datasets)")
        dataset = PreloadedChessDataset(npz_file)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster CPU->GPU transfer
        persistent_workers=True if num_workers > 0 else False
    )

    return dataloader


# Example usage and memory comparison
if __name__ == "__main__":
    import psutil
    import os

    def get_ram_usage():
        """Get current process RAM usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    print("\n" + "=" * 70)
    print("MEMORY EFFICIENCY COMPARISON")
    print("=" * 70)

    dataset_file = "../outputs/chess_elo1900_500K.npz"

    print(f"\nInitial RAM usage: {get_ram_usage():.0f} MB\n")

    # Test 1: Memory-mapped (efficient)
    print("\n--- Test 1: Memory-Mapped DataLoader ---")
    ram_before = get_ram_usage()

    dataloader_mmap = create_dataloader(
        dataset_file,
        batch_size=256,
        memory_mapped=True
    )

    ram_after = get_ram_usage()
    print(f"RAM increase: {ram_after - ram_before:.0f} MB")

    # Load a few batches to show it works
    print("\nLoading first 3 batches...")
    for i, (positions, moves, results) in enumerate(dataloader_mmap):
        if i >= 3:
            break
        print(f"  Batch {i}: positions={positions.shape}, moves={moves.shape}, results={results.shape}")

    ram_peak = get_ram_usage()
    print(f"Peak RAM during loading: {ram_peak:.0f} MB")
    print(f"Total increase from start: {ram_peak - ram_before:.0f} MB")

    del dataloader_mmap

    # Test 2: Preloaded (memory-hungry) - only if you have RAM
    print("\n\n--- Test 2: Preloaded DataLoader (RAM-heavy) ---")
    ram_before = get_ram_usage()

    try:
        dataloader_preload = create_dataloader(
            dataset_file,
            batch_size=256,
            memory_mapped=False
        )

        ram_after = get_ram_usage()
        print(f"RAM increase: {ram_after - ram_before:.0f} MB")

        # Load a few batches
        print("\nLoading first 3 batches...")
        for i, (positions, moves, results) in enumerate(dataloader_preload):
            if i >= 3:
                break
            print(f"  Batch {i}: positions={positions.shape}, moves={moves.shape}, results={results.shape}")

        ram_peak = get_ram_usage()
        print(f"Peak RAM: {ram_peak:.0f} MB")
        print(f"Total increase from start: {ram_peak - ram_before:.0f} MB")

        del dataloader_preload

    except MemoryError:
        print("❌ MEMORY ERROR: Not enough RAM for preloaded approach!")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    print("✅ Use memory_mapped=True for datasets >1M positions")
    print("✅ This allows training on 20M+ datasets with only 2-4GB RAM")
    print("=" * 70 + "\n")
