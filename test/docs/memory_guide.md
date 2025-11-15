# Memory Management Guide for 20M Chess Dataset

## The Problem

Training on a 20M position dataset requires careful memory management on an 8GB RAM system.

---

## Memory Breakdown

### Dataset Size (20M positions)
- **Positions**: 20M × 12 × 8 × 8 × 2 bytes = **3.0 GB**
- **Moves**: 20M × 4 bytes = **76 MB**
- **Results**: 20M × 2 bytes = **38 MB**
- **Total Dataset**: **~3.2 GB**

### Training Memory Requirements

| Component | Memory Usage |
|-----------|--------------|
| Dataset (if loaded fully) | 3.2 GB |
| Small CNN model | 100-200 MB |
| Medium CNN model | 500 MB - 1 GB |
| Large ResNet/Transformer | 1-3 GB |
| Optimizer states (Adam) | 2x model size |
| Gradients | 1x model size |
| Batch tensors | ~200 MB (batch=256) |
| PyTorch overhead | ~500 MB |

### Total RAM Needed (Traditional Approach)

| Model Size | Total RAM Required | 8GB System? |
|------------|-------------------|-------------|
| Small model | ~5-6 GB | ✅ Might work |
| Medium model | ~8-10 GB | ❌ Will crash |
| Large model | ~12-15 GB | ❌ Definitely crashes |

---

## The Solution: Memory-Mapped DataLoader

### How It Works

Instead of loading the entire 3.2 GB dataset into RAM:

1. **Open file in memory-mapped mode** (only metadata loaded)
2. **Load batches on-demand** from disk
3. **OS caches recently used data** automatically
4. **Only current batch in RAM** (~200 MB)

### Memory Usage Comparison

| Approach | RAM for Data | Total RAM (Medium Model) | Works on 8GB? |
|----------|--------------|--------------------------|---------------|
| **Traditional (load all)** | 3.2 GB | ~8-10 GB | ❌ Crashes |
| **Memory-mapped** | ~100 MB | **~3-4 GB** | ✅ **Works!** |

### Code Example

```python
# ❌ BAD: Loads entire 3.2 GB into RAM
data = np.load('chess_20M.npz')
positions = torch.from_numpy(data['positions'])  # CRASH!

# ✅ GOOD: Memory-mapped, only loads batches
data = np.load('chess_20M.npz', mmap_mode='r')
# Dataset opens, but data stays on disk until needed
```

---

## Using Memory-Efficient DataLoader

### Quick Start

```python
from memory_efficient_dataloader import create_dataloader

# Create memory-efficient DataLoader
train_loader = create_dataloader(
    'chess_elo1900_20M.npz',
    batch_size=256,
    shuffle=True,
    num_workers=2,
    memory_mapped=True  # ← KEY: Enables memory mapping
)

# Train normally - no changes needed!
for positions, moves, results in train_loader:
    # Only current batch is in RAM (~200 MB)
    model_output = model(positions)
    loss = criterion(model_output, moves)
    # ...
```

### Memory Usage During Training

```
RAM Usage Timeline:
├─ Start:                    1.5 GB (OS + Python)
├─ Load DataLoader:          1.6 GB (+100 MB, just metadata)
├─ Create Model:             2.1 GB (+500 MB, medium model)
├─ Training (batch 1):       3.0 GB (+900 MB, optimizer + batch)
├─ Training (batch 2):       3.0 GB (steady state)
└─ Peak:                     3.2 GB (well under 8 GB!)
```

---

## Performance Considerations

### Speed Impact

| Aspect | Traditional | Memory-Mapped | Impact |
|--------|-------------|---------------|--------|
| Data loading | Fast (already in RAM) | Disk read | 5-10% slower |
| First epoch | Fast | Slow (cold cache) | 20-30% slower |
| Later epochs | Fast | Fast (cached) | ~Same |
| Overall | - | - | **10-15% slower** |

**Verdict**: Small speed penalty, but enables training that would otherwise crash!

### Optimization Tips

1. **Use SSD** - Much faster than HDD for random reads
2. **Increase batch size** - Fewer disk reads per epoch
3. **Use num_workers=2-4** - Parallel data loading
4. **Enable pin_memory=True** - Faster CPU→GPU transfer
5. **Let first epoch warm cache** - Subsequent epochs faster

---

## Recommended Configuration

### For 8GB RAM System with 20M Dataset

```python
# Dataset
create_dataloader(
    'chess_elo1900_20M.npz',
    batch_size=256,          # Adjust based on GPU/model size
    shuffle=True,
    num_workers=2,           # 2-4 workers
    memory_mapped=True       # CRITICAL: Enable memory mapping
)

# Model Size Guidelines
# Small model:   <50M parameters  → ~2 GB RAM total
# Medium model:  50-200M params   → ~3-4 GB RAM total  ✅ RECOMMENDED
# Large model:   >200M params     → ~5-6 GB RAM total  ⚠️ Risky
```

### For Larger Systems (16GB+ RAM)

```python
# Can use traditional approach for faster training
create_dataloader(
    'chess_elo1900_20M.npz',
    batch_size=512,          # Larger batches
    shuffle=True,
    num_workers=4,
    memory_mapped=False      # Load entire dataset into RAM
)
```

---

## Testing Your Setup

Run the test script to verify memory usage:

```bash
cd test/data
python memory_efficient_dataloader.py
```

Expected output:
```
--- Test 1: Memory-Mapped DataLoader ---
✅ Dataset ready: 500,000 positions
💾 Memory usage: ~0 MB (memory-mapped)
RAM increase: 85 MB

--- Test 2: Preloaded DataLoader ---
✅ Dataset loaded: 500,000 positions
💾 RAM usage: 641 MB
RAM increase: 641 MB
```

For 20M dataset:
- Memory-mapped: **~100 MB increase**
- Preloaded: **~3,200 MB increase** (would crash during training)

---

## Summary

### ✅ DO THIS:
- Use `memory_mapped=True` for datasets >1M positions
- Use medium-sized models (50-200M parameters)
- Set `batch_size=256` or higher
- Use 2-4 data loading workers
- Train on SSD if possible

### ❌ DON'T DO THIS:
- Load entire dataset with `memory_mapped=False` on 8GB system
- Use very large models (>200M params) on limited RAM
- Use batch_size <64 (too many disk reads)
- Train on HDD (very slow)

### Expected Performance (8GB RAM, 20M dataset):
- Peak RAM usage: **3-4 GB** ✅
- Training speed: **~10-15% slower** than full RAM loading
- **Enables training that would otherwise crash!**

---

## Conclusion

With memory-mapped DataLoaders, you can train on **20M+ position datasets** using only **3-4 GB RAM**, leaving plenty of headroom for deeper models in the future.

**The trade-off**: Slightly slower training (10-15%) in exchange for the ability to train on datasets 5-10x larger than available RAM.
