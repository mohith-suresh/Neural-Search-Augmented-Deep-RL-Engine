"""
GPU availability checker
Checks if CUDA/GPU is available for PyTorch training
"""

import torch
import sys

print("\n" + "=" * 70)
print("GPU AVAILABILITY CHECK")
print("=" * 70)

# Check PyTorch installation
print(f"\n✓ PyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA available: {'✅ YES' if cuda_available else '❌ NO'}")

if cuda_available:
    # CUDA details
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    # GPU count
    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of GPUs: {num_gpus}")

    # GPU details
    for i in range(num_gpus):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")

        # Memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"Total memory: {total_memory:.2f} GB")

        # Current usage
        if torch.cuda.memory_allocated(i) > 0:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Cached: {cached:.2f} GB")
        else:
            print(f"Currently free")

        # Compute capability
        capability = torch.cuda.get_device_capability(i)
        print(f"Compute capability: {capability[0]}.{capability[1]}")

    # Test GPU with a simple operation
    print("\n" + "-" * 70)
    print("Testing GPU with simple operation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU test successful! Matrix multiplication works.")

        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ GPU test failed: {e}")

    # Recommended device
    print("\n" + "-" * 70)
    print(f"Recommended device: cuda:0")
    print(f"Usage: device = torch.device('cuda')")

else:
    # No GPU available
    print("\n" + "-" * 70)
    print("❌ No GPU detected!")
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU in system")
    print("  2. CUDA toolkit not installed")
    print("  3. PyTorch CPU-only version installed")
    print("  4. GPU drivers not installed/updated")

    print("\nTo check:")
    print("  • Run: nvidia-smi")
    print("  • If GPU exists but CUDA unavailable, reinstall PyTorch with CUDA")

    print("\n" + "-" * 70)
    print(f"Recommended device: cpu")
    print(f"Usage: device = torch.device('cpu')")
    print("\n⚠️  Training will be MUCH slower on CPU (10-50x slower)")

print("\n" + "=" * 70)
print("DEVICE SETUP FOR TRAINING")
print("=" * 70)

# Show code snippet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nCurrent device: {device}")

print("\nUse this in your training code:")
print("""
    import torch

    # Automatically select GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Move model to device
    model = YourModel().to(device)

    # Move data to device in training loop
    for positions, moves, results in dataloader:
        positions = positions.to(device)
        moves = moves.to(device)
        results = results.to(device)

        # Training code...
""")

print("\n" + "=" * 70 + "\n")
