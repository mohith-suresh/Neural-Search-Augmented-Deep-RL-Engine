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
BACKUP_PATH = f"{MODEL_DIR}/best_model_13ch_backup.pth"

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
            print("   -> Found 'model_state_dict' key.")
        elif 'state_dict' in checkpoint:
            old_state = checkpoint['state_dict']
            print("   -> Found 'state_dict' key.")
        else:
            # Maybe the dict itself is the state dict?
            # Check if keys look like layer names (contain dots)
            if any('.' in k for k in checkpoint.keys()):
                old_state = checkpoint
                print("   -> Assuming dictionary is the state dict directly.")
            else:
                print(f"❌ Unknown checkpoint structure. Top-level keys: {list(checkpoint.keys())}")
                return
    else:
        print("❌ Checkpoint is not a dictionary.")
        return

    # Initialize new 14-channel model
    new_model = ChessCNN()
    new_state = new_model.state_dict()

    # --- Find Input Layer ---
    input_layer_name = None
    
    # Debug: Print shapes of first few keys to help diagnose
    print("\n🔍 Inspecting first 5 layers in checkpoint:")
    for i, (k, v) in enumerate(old_state.items()):
        if i >= 5: break
        if isinstance(v, torch.Tensor):
            print(f"   {k}: {v.shape}")
        else:
            print(f"   {k}: [Non-Tensor]")

    for key, param in old_state.items():
        if not isinstance(param, torch.Tensor): continue
        
        # Check for [Out, In, K, K] format
        if len(param.shape) == 4:
            # Check for 13 channels (Old format)
            if param.shape[1] == 13:
                input_layer_name = key
                print(f"\n✅ FOUND 13-CHANNEL LAYER: '{key}' with shape {param.shape}")
                break
            # Check for 14 channels (Already migrated?)
            elif param.shape[1] == 14:
                print(f"\n✨ Found 14-channel layer '{key}'. Model might already be migrated.")
                return

    if input_layer_name is None:
        print("\n❌ Could not find any layer with 13 input channels.")
        print("   This might mean the model is incompatible or uses a different architecture.")
        return

    print(f"⚠️ Migrating input layer '{input_layer_name}' to 14 channels...")

    # Backup original
    shutil.copy(OLD_MODEL_PATH, BACKUP_PATH)
    print(f"📦 Original backed up to {BACKUP_PATH}")

    # Copy weights
    for name, param in old_state.items():
        if not isinstance(param, torch.Tensor): continue

        target_name = name
        
        # Map old input layer name to new structure
        if name == input_layer_name:
            target_name = 'input_conv.0.weight' 

        if target_name not in new_state:
            # Try to be flexible: maybe 'module.' prefix from DataParallel?
            if target_name.startswith('module.'):
                target_name = target_name[7:]
            
            if target_name not in new_state:
                print(f"   ⚠️ Skipping layer '{name}' (no match in new model)")
                continue

        # Handle the Input Convolution migration
        if target_name == 'input_conv.0.weight':
            old_weight = param
            new_weight = new_state[target_name]
            
            # Double check shapes
            if old_weight.shape[1] != 13:
                print(f"   ❌ Logic error: Selected layer {name} has shape {old_weight.shape}")
                continue

            # Copy 13 channels
            new_weight[:, :13, :, :] = old_weight
            # Zero out 14th channel
            new_weight[:, 13, :, :] = 0.0
            
            new_state[target_name] = new_weight
            print(f"   ✨ Patched {name} -> {target_name}")
        else:
            # Direct copy
            if param.shape != new_state[target_name].shape:
                print(f"   ❌ Shape mismatch {name}: {param.shape} vs {new_state[target_name].shape}")
            else:
                new_state[target_name] = param

    # Save upgraded model
    torch.save({
        'model_state_dict': new_state,
        'optimizer_state_dict': None, 
        'epoch': 0,
        'metrics': checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}
    }, OLD_MODEL_PATH)

    print("\n✅ Migration Complete. You can now start main.py")

if __name__ == "__main__":
    migrate_weights()