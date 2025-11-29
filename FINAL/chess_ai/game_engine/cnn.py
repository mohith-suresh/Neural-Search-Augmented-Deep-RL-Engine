#!/usr/bin/env python3
"""
Chess CNN - Final Production Build (Stable)
===========================================

Fixes:
- Solved 'AttributeError: scheduler': Initialized in __init__
- Solved 'JSON Tensor Error': Explicit float conversion
- Solved 'Testing Crash': Graceful exit handling
- Thermal: Optimized sleep (0.2s) for speed/safety balance

"""

import json
import sys
import time
import shutil
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    # Architecture
    input_channels: int = 13
    filters: int = 128
    num_res_blocks: int = 10
    
    # Dataset (90/5/5)
    train_split: float = 0.90
    val_split: float = 0.05
    test_split: float = 0.05
    
    # Physics
    total_dataset_size: int = 2_500_000 
    virtual_epoch_size: int = 500_000
    target_batch_size: int = 1024 
    start_batch_size_guess: int = 512
    
    # Thermal & Optimization
    gpu_safety_margin: float = 0.80  
    inter_batch_sleep: float = 0.2   # Lowered to 0.2s (1s was too slow)
    
    # Hyperparams
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    
    # Paths
    data_dir: Path = Path("/home/krish/EE542-Project/FINAL/chess_ai/data/training_data")
    model_dir: Path = Path("/home/krish/EE542-Project/FINAL/chess_ai/game_engine/model")
    log_dir: Path = Path("/home/krish/EE542-Project/FINAL/chess_ai/logs/tensorboard")
    
    resume_checkpoint: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Architecture
# ============================================================================

class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            Mish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = Mish()
        self.se = SEBlock(channels) if use_se else None
    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se: out = self.se(out)
        out += residual
        return self.act(out)

class ChessCNN(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.input_conv = nn.Conv2d(config.input_channels, config.filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(config.filters)
        self.act = Mish()
        self.res_blocks = nn.ModuleList([ResidualBlock(config.filters, use_se=(i>=7)) for i in range(config.num_res_blocks)])
        self.policy_conv = nn.Conv2d(config.filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 8192)
        self.value_conv = nn.Conv2d(config.filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.act(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks: x = block(x)
        p = self.act(self.policy_bn(self.policy_conv(x))).view(x.size(0), -1)
        p = self.policy_fc(p)
        v = self.act(self.value_bn(self.value_conv(x))).view(x.size(0), -1)
        v = torch.tanh(self.value_fc2(self.act(self.value_fc1(v))))
        return p, v

# ============================================================================
# Trainer
# ============================================================================

class MemmapChessDataset(Dataset):
    def __init__(self, data_dir: Path):
        with open(data_dir / 'metadata.json', 'r') as f: self.meta = json.load(f)
        self.positions = np.memmap(data_dir/'positions.bin', dtype=self.meta['pos_dtype'], mode='r', shape=tuple(self.meta['pos_shape']))
        self.moves = np.memmap(data_dir/'moves.bin', dtype=self.meta['mov_dtype'], mode='r', shape=tuple(self.meta['mov_shape']))
        self.results = np.memmap(data_dir/'results.bin', dtype=self.meta['res_dtype'], mode='r', shape=tuple(self.meta['res_shape']))
    def __len__(self): return self.meta['count']
    def __getitem__(self, i):
        return torch.from_numpy(self.positions[i].copy()).float(), torch.tensor(int(self.moves[i]), dtype=torch.long), torch.tensor(float(self.results[i]), dtype=torch.float)

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = ChessCNN(config).to(config.device)
        try: self.model = torch.compile(self.model)
        except: pass
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.policy_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.value_loss = nn.MSELoss()
        
        self.scheduler = None  # FIX: Initialize here to prevent AttributeError
        self.history = {'train_loss':[], 'val_loss':[], 'p_acc':[], 'v_mae':[], 'v_corr':[]}
        self.top_models = []
        if config.resume_checkpoint: self._load_ckpt(config.resume_checkpoint)

    def _load_ckpt(self, path):
        c = torch.load(path)
        self.model.load_state_dict(c['state_dict'])
        self.optimizer.load_state_dict(c['optimizer'])
        self.history = c.get('metrics', self.history)

    def find_safe_batch_size(self):
        print("\n--- 🌡️ Thermally Safe Auto-Scaling ---")
        bs = self.config.start_batch_size_guess
        max_stable = 0
        while True:
            try:
                print(f"Probing Max BS={bs}...", end="", flush=True)
                with autocast():
                    p, v = self.model(torch.randn(bs, 13, 8, 8, device=self.config.device))
                    l = self.policy_loss(p, torch.zeros(bs, dtype=torch.long, device=self.config.device))
                self.scaler.scale(l).backward()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                print(" OK ✅")
                max_stable = bs
                if bs >= 1024: break 
                bs *= 2
            except RuntimeError:
                print(" Fail ❌")
                break
        
        safe_bs = int(max_stable * self.config.gpu_safety_margin)
        safe_bs = max(32, safe_bs - (safe_bs % 32))
        
        acc = max(1, self.config.target_batch_size // safe_bs)
        print(f"-> Safe Target (80%): {safe_bs}")
        print(f"-> Accumulation: {acc}")
        return safe_bs, acc

    def train(self):
        ds = MemmapChessDataset(self.config.data_dir)
        tr_len = int(len(ds) * self.config.train_split)
        val_len = int(len(ds) * self.config.val_split)
        test_len = len(ds) - tr_len - val_len
        train_set, val_set, test_set = random_split(ds, [tr_len, val_len, test_len])
        
        bs, accum = self.find_safe_batch_size()
        loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4)
        
        samples_per_virt = self.config.virtual_epoch_size
        iters_per_virt = samples_per_virt // bs
        
        # Initialize Scheduler HERE, before loop
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.config.learning_rate, 
                                    epochs=self.config.num_epochs, steps_per_epoch=iters_per_virt//accum)
        
        print(f"\n🚀 STARTING TRAINING (Thermal Sleep: {self.config.inter_batch_sleep}s)")
        
        iter_loader = iter(loader)
        
        try:
            for epoch in range(1, self.config.num_epochs + 1):
                self.model.train()
                t_metrics = {'loss':0, 'p_loss':0, 'v_loss':0, 'p_acc':0, 'v_mae':0}
                pbar = tqdm(range(iters_per_virt), desc=f"Ep {epoch} Train", unit="batch")
                
                for _ in pbar:
                    try: batch = next(iter_loader)
                    except StopIteration: iter_loader = iter(loader); batch = next(iter_loader)
                    
                    if self.config.inter_batch_sleep > 0:
                        time.sleep(self.config.inter_batch_sleep)
                    
                    pos, mov, res = batch[0].to(self.config.device), batch[1].to(self.config.device), batch[2].to(self.config.device).unsqueeze(1)
                    
                    with autocast():
                        p, v = self.model(pos)
                        lp = self.policy_loss(p, mov)
                        lv = self.value_loss(v, res)
                        loss = (lp + 4.0 * lv) / accum
                    
                    self.scaler.scale(loss).backward()
                    
                    t_metrics['loss'] += loss.item() * accum
                    t_metrics['p_acc'] += (p.argmax(1) == mov).float().mean().item()
                    pbar.set_postfix({'loss': f"{loss.item()*accum:.2f}", 'acc': f"{t_metrics['p_acc']/(_+1):.1%}"})
                    
                    if (_ + 1) % accum == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        if self.scheduler: self.scheduler.step()

                # Validation
                self.model.eval()
                v_metrics = {'loss':0, 'p_loss':0, 'v_loss':0, 'p_acc':0, 'v_mae':0, 'v_corr':0}
                val_bar = tqdm(val_loader, desc=f"Ep {epoch} Valid", total=min(len(val_loader), 200), unit="batch")
                
                with torch.no_grad():
                    for i, (pos, mov, res) in enumerate(val_bar):
                        if i >= 200: break
                        pos, mov, res = pos.to(self.config.device), mov.to(self.config.device), res.to(self.config.device).unsqueeze(1)
                        with autocast():
                            p, v = self.model(pos)
                            loss = self.policy_loss(p, mov) + 4.0 * self.value_loss(v, res)
                        
                        v_metrics['loss'] += loss.item()
                        v_metrics['p_acc'] += (p.argmax(1) == mov).float().mean().item()
                        v_metrics['v_mae'] += F.l1_loss(v, res).item()
                        
                        vx = v - v.mean(); vy = res - res.mean()
                        corr = (vx*vy).sum() / (torch.sqrt((vx**2).sum() * (vy**2).sum()) + 1e-8)
                        v_metrics['v_corr'] += corr.item()
                        
                        val_bar.set_postfix({'val_loss': f"{loss.item():.2f}", 'val_acc': f"{(p.argmax(1)==mov).float().mean():.1%}"})

                avg_val_loss = v_metrics['loss'] / (i+1)
                print(f"Summary Ep {epoch}: Val Loss {avg_val_loss:.4f} | Pol Acc {v_metrics['p_acc']/(i+1):.1%} | Val Corr {v_metrics['v_corr']/(i+1):.3f}")
                
                self.manage_checkpoints(epoch, avg_val_loss)
                self.plot_metrics(t_metrics, v_metrics, iters_per_virt, i+1)

        except KeyboardInterrupt:
            print("\n🛑 Stop signal received. Finishing up...")
        finally:
            self.final_test()

    def manage_checkpoints(self, epoch, loss):
        fname = self.config.model_dir / f"model_ep{epoch}.pth"
        self.config.model_dir.mkdir(exist_ok=True, parents=True)
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'metrics': self.history}, fname)
        
        self.top_models.append({'path': fname, 'loss': loss})
        self.top_models.sort(key=lambda x: x['loss'])
        if len(self.top_models) > 3:
            rem = self.top_models.pop()
            if rem['path'].exists(): rem['path'].unlink()
            
        names = ["best_model.pth", "checkpoint_2.pth", "checkpoint_3.pth"]
        for idx, item in enumerate(self.top_models[:3]):
            if item['path'].exists(): shutil.copy(item['path'], self.config.model_dir / names[idx])

    def plot_metrics(self, t, v, t_steps, v_steps):
        # Update history
        self.history['train_loss'].append(t['loss']/t_steps)
        self.history['val_loss'].append(v['loss']/v_steps)
        self.history['p_acc'].append(v['p_acc']/v_steps)
        self.history['v_corr'].append(v['v_corr']/v_steps)
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # Helper for smoothing
        def smooth(scalars, weight=0.6):
            if not scalars: return []
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

        ax[0].plot(smooth(self.history['train_loss']), label='Train (Smooth)')
        ax[0].plot(self.history['val_loss'], label='Val')
        ax[0].set_title('Loss')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        ax[1].plot(self.history['p_acc'], color='green', marker='o')
        ax[1].set_title('Validation Accuracy')
        ax[1].grid(True, alpha=0.3)
        
        ax[2].plot(self.history['v_corr'], color='purple', marker='x')
        ax[2].set_title('Validation Value Correlation')
        ax[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.model_dir / "training_curves.png", dpi=150)
        plt.close()

    def final_test(self):
        print("\n" + "="*40)
        print("🧪 FINAL TESTING PHASE (Best Model)")
        print("="*40)
        best = self.config.model_dir / "best_model.pth"
        if best.exists(): 
            try:
                ckpt = torch.load(best, map_location=self.config.device)
                self.model.load_state_dict(ckpt['state_dict'])
                print(f"Loaded: {best}")
            except: print("Could not load best model, using current state.")
        
        self.model.eval()
        pbar = tqdm(self.test_loader, desc="Testing", unit="batch")
        metrics = {'loss':0, 'p_acc':0, 'v_mae':0, 'v_corr':0}
        steps = 0
        
        with torch.no_grad():
            for pos, mov, res in pbar:
                pos, mov, res = pos.to(self.config.device), mov.to(self.config.device), res.to(self.config.device).unsqueeze(1)
                with autocast():
                    p, v = self.model(pos)
                    loss = self.policy_loss(p, mov) + 4*self.value_loss(v, res)
                
                metrics['loss'] += loss.item()
                metrics['p_acc'] += (p.argmax(1)==mov).float().mean().item()
                metrics['v_mae'] += F.l1_loss(v, res).item()
                vx = v - v.mean(); vy = res - res.mean()
                metrics['v_corr'] += (vx*vy).sum() / (torch.sqrt((vx**2).sum() * (vy**2).sum()) + 1e-8)
                steps += 1

        # FIX: Ensure JSON compatibility by casting to float
        final = {k: float(v/steps) for k, v in metrics.items()}
        
        print(f"\n📊 FINAL METRICS:\n{json.dumps(final, indent=2)}")
        with open(self.config.model_dir / "final_test_metrics.json", "w") as f:
            json.dump(final, f, indent=2)
        print(f"✅ Saved to {self.config.model_dir}/final_test_metrics.json")

if __name__ == '__main__':
    Trainer(TrainingConfig()).train()