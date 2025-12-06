import torch
from cnn import ChessCNN

# Load migrated model
model = ChessCNN()
state = torch.load('game_engine/model/best_model.pth')
if 'model_state_dict' in state:
    state = state['model_state_dict']
model.load_state_dict(state)

# Count parameters
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")
print(f"Expected (~192 filters): 29-30M parameters")

# Test forward pass
test_input = torch.randn(1, 16, 8, 8)
with torch.no_grad():
    policy, value = model(test_input)
    print(f"Policy shape: {policy.shape} (should be [1, 8192])")
    print(f"Value shape: {value.shape} (should be [1, 1])")
    print("✓ Migration successful!")
