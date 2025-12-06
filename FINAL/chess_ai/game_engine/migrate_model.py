import torch
import torch.nn as nn
import os
import sys
import shutil

# Ensure imports work
sys.path.append(os.getcwd())
from game_engine.cnn import ChessCNN

MODEL_DIR = "game_engine/model"
OLD_MODEL_PATH = f"{MODEL_DIR}/best_model.pth"
NEW_MODEL_PATH = f"{MODEL_DIR}/best_model_192.pth"
BACKUP_PATH = f"{MODEL_DIR}/best_model_128_backup.pth"

def migrate_filters():
    """
    Migrate model from 128 filters to 192 filters.
    Preserves learned weights, initializes new filters with Kaiming normal.
    """
    
    # Check if old model exists
    if not os.path.exists(OLD_MODEL_PATH):
        print(f"❌ No model found at {OLD_MODEL_PATH}")
        return False
    
    print(f"🔄 Loading old model from {OLD_MODEL_PATH}...")
    
    try:
        checkpoint = torch.load(OLD_MODEL_PATH, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load model file: {e}")
        return False
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            old_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            old_state = checkpoint['state_dict']
        else:
            old_state = checkpoint
    else:
        print("❌ Checkpoint is not a dictionary")
        return False
    
    # Backup original
    shutil.copy(OLD_MODEL_PATH, BACKUP_PATH)
    print(f"📦 Original backed up to {BACKUP_PATH}")
    
    # Initialize new model with 192 filters
    print(f"🔧 Creating new model with 192 filters...")
    new_model = ChessCNN()
    new_state = new_model.state_dict()
    
    # Migrate weights
    migrated_state = {}
    
    for key in new_state.keys():
        if key not in old_state:
            # New key: initialize with new_state value (random init)
            migrated_state[key] = new_state[key]
            continue
        
        old_weight = old_state[key]
        new_weight = new_state[key]
        
        # If shapes match exactly, copy directly
        if old_weight.shape == new_weight.shape:
            migrated_state[key] = old_weight
            print(f"✓ {key}: copied (shapes match)")
        
        # Handle Conv2d layer expansion (most common case)
        elif len(old_weight.shape) == 4 and len(new_weight.shape) == 4:
            old_out, old_in, h, w = old_weight.shape
            new_out, new_in, _, _ = new_weight.shape
            
            if old_in == new_in and h == w:  # Same input channels, same kernel size
                if new_out > old_out:
                    # Expanding filters: copy old, initialize new with Kaiming
                    migrated_weight = torch.zeros_like(new_weight)
                    migrated_weight[:old_out] = old_weight
                    
                    # Initialize new filters with Kaiming normal
                    nn.init.kaiming_normal_(migrated_weight[old_out:], mode='fan_out', nonlinearity='relu')
                    
                    migrated_state[key] = migrated_weight
                    print(f"✓ {key}: expanded from {old_weight.shape} → {new_weight.shape}")
                else:
                    # Truncating filters
                    migrated_state[key] = old_weight[:new_out]
                    print(f"✓ {key}: truncated from {old_weight.shape} → {new_weight.shape}")
            else:
                # Input channels or kernel size mismatch
                migrated_state[key] = new_state[key]
                print(f"⚠ {key}: input mismatch, using new random init")
        
        # Handle FC layer expansion
        elif len(old_weight.shape) == 2 and len(new_weight.shape) == 2:
            old_in, old_out = old_weight.shape
            new_in, new_out = new_weight.shape
            
            if old_in == new_in:  # Input features match
                if new_out > old_out:
                    migrated_weight = torch.zeros_like(new_weight)
                    migrated_weight[:old_in, :old_out] = old_weight
                    nn.init.kaiming_normal_(migrated_weight, mode='fan_out')
                    migrated_state[key] = migrated_weight
                    print(f"✓ {key}: expanded from {old_weight.shape} → {new_weight.shape}")
                else:
                    migrated_state[key] = old_weight[:new_in, :new_out]
                    print(f"✓ {key}: truncated from {old_weight.shape} → {new_weight.shape}")
            else:
                migrated_state[key] = new_state[key]
                print(f"⚠ {key}: input mismatch, using new random init")
        
        else:
            # Unhandled case: use new state
            migrated_state[key] = new_state[key]
            print(f"⚠ {key}: shape mismatch, using new random init")
    
    # Load migrated state into new model
    new_model.load_state_dict(migrated_state, strict=False)
    
    # Save migrated model
    torch.save({
        'model_state_dict': new_model.state_dict(),
        'optimizer_state_dict': None,
        'epoch': 0,
        'metrics': checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}
    }, NEW_MODEL_PATH)
    
    print(f"\n✅ Migration complete!")
    print(f"✅ Old model: 128 filters → {BACKUP_PATH}")
    print(f"✅ New model: 192 filters → {NEW_MODEL_PATH}")
    print(f"\nNext step: cp {NEW_MODEL_PATH} {OLD_MODEL_PATH}")
    
    return True

if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         Model Migration: 128 Filters → 192 Filters            ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    success = migrate_filters()
    
    if success:
        print("\n" + "="*70)
        print("✨ Migration successful! Your 68K training positions are safe.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ Migration failed. Check errors above.")
        print("="*70)
