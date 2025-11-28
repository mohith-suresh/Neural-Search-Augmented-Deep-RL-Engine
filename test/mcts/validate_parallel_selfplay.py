# Quick sanity check
import numpy as np

data = np.load('parallel_selfplay_output/parallel_games_12.npz')

# Check for NaN/Inf
assert not np.any(np.isnan(data['positions'])), "NaN in positions!"
assert not np.any(np.isnan(data['policies'])), "NaN in policies!"
assert not np.any(np.isnan(data['values'])), "NaN in values!"

# Check policy sums (should be ~1.0)
policy_sums = data['policies'].sum(axis=1)
print(f"Policy sum range: {policy_sums.min():.3f} - {policy_sums.max():.3f}")
assert np.allclose(policy_sums, 1.0, atol=1e-5), "Policies don't sum to 1!"

# Check value range
assert np.all((data['values'] >= -1) & (data['values'] <= 1)), "Invalid values!"

print("✓ All data validated successfully!")
