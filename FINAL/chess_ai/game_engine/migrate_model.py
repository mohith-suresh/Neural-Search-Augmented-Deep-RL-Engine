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
BACKUP_PATH = f"{MODEL_DIR}/best_model_old_backup.pth"

def migrate_weights():
    if not os.path.exists(OLD_MODEL_PATH):
        print(f"❌ No model found at {OLD_MODEL_PATH}")
        return

    print(f"🔄 Loading model from {OLD_MODEL_PATH}...")
    try:
        checkpoint = torch.load(OLD_MODEL_PATH, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load model file: {e}")
        return

    # --- Robust State Dict Extraction ---
    old_state = None
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            old_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            old_state = checkpoint['state_dict']
        else:
            if any('.' in k for k in checkpoint.keys()):
                old_state = checkpoint
            else:
                print(f"❌ Unknown checkpoint structure.")
                return
    else:
        print("❌ Checkpoint is not a dictionary.")
        return

    # Initialize new 16-channel model
    new_model = ChessCNN()
    new_state = new_model.state_dict()

    # --- Find Input Layer ---
    input_layer_name = None
    input_channels_old = 0
    
    print("\n🔍 Inspecting first 5 layers:")
    for i, (k, v) in enumerate(old_state.items()):
        if i >= 5: break
        if isinstance(v, torch.Tensor):
            print(f"   {k}: {v.shape}")

    for key, param in old_state.items():
        if not isinstance(param, torch.Tensor): continue
        if len(param.shape) == 4:
            # Check for 13 or 14 channels
            if param.shape[1] in [13, 14]:
                input_layer_name = key
                input_channels_old = param.shape[1]
                print(f"\n✅ FOUND INPUT LAYER: '{key}' with {input_channels_old} channels")
                break
            elif param.shape[1] == 16:
                print(f"\n✨ Model already has 16 channels! No migration needed.")
                return

    if input_layer_name is None:
        print("\n❌ Could not find compatible input layer (13 or 14 channels).")
        return

    print(f"⚠️ Migrating from {input_channels_old} -> 16 channels...")

    # Backup original
    shutil.copy(OLD_MODEL_PATH, BACKUP_PATH)
    print(f"📦 Original backed up to {BACKUP_PATH}")

    # Copy weights
    for name, param in old_state.items():
        if not isinstance(param, torch.Tensor): continue

        target_name = name
        if name == input_layer_name: target_name = 'input_conv.0.weight' 

        if target_name not in new_state:
            if target_name.startswith('module.'): target_name = target_name[7:]
            if target_name not in new_state: continue

        if target_name == 'input_conv.0.weight':
            old_weight = param
            new_weight = new_state[target_name]
            
            # Copy existing channels
            new_weight[:, :input_channels_old, :, :] = old_weight
            # Zero out new channels (Indices input_channels_old to 15)
            new_weight[:, input_channels_old:, :, :] = 0.0
            
            new_state[target_name] = new_weight
            print(f"   ✨ Patched Input: Copied {input_channels_old} ch, Zeroed rest.")
        else:
            if param.shape == new_state[target_name].shape:
                new_state[target_name] = param

    # Save upgraded model
    torch.save({
        'model_state_dict': new_state,
        'optimizer_state_dict': None, 
        'epoch': 0,
        'metrics': checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}
    }, OLD_MODEL_PATH)

    print("\n✅ Migration to 16 Channels Complete. You can now start main.py")

if __name__ == "__main__":
    migrate_weights()