#!/usr/bin/env python3
"""Fix VAE key naming and test decode via ComfyUI API."""
import sys, time, subprocess, json
import torch

# Step 1: Re-save VAE with correct key format
sys.path.insert(0, '/Users/ryantsudek/ComfyUI')
from safetensors import safe_open
from safetensors.torch import save_file

model_path = '/Users/ryantsudek/ComfyUI/models/vae/LTX-Video-VAE-BF16.safetensors'
print('Loading VAE and re-saving with corrected key format...')
vae_sd = {}
with safe_open(model_path, framework='pt') as f:
    for k in f.keys():
        vae_sd[k] = f.get_tensor(k)

# The ComfyUI node loader adds "model." prefix, so original keys are:
#   decoder.conv_in.conv.bias  → model.decoder.conv_in.conv.bias
# But the WanVideoVAE38 model internally expects:
#   model.decoder.conv1.bias  (conv1, not conv_in.conv)
# Let's check if this is the mismatch by looking at the actual state dict key names

print('Keys in file:')
for k in sorted(vae_sd.keys())[:10]:
    print(' ', k)

# Key renaming: conv_in.conv → conv1
# e.g. model.decoder.conv_in.conv.bias → model.decoder.conv1.bias
fixed_sd = {}
for k, v in vae_sd.items():
    new_k = k.replace('conv_in.conv', 'conv1')
    fixed_sd[new_k] = v

print('\nFixed keys:')
for k in sorted(fixed_sd.keys())[:10]:
    print(' ', k)

# Save as safetensors with model. prefix (like WanVideoVAELoader does)
prefixed_sd = {f'model.{k}': v for k, v in fixed_sd.items()}
save_file(prefixed_sd, '/tmp/ltx_vae_fixed.safetensors')
print('\nSaved to /tmp/ltx_vae_fixed.safetensors')

# Step 2: Create correct latent format
latent = torch.randn(1, 128, 9, 32, 32) * 0.5
latent_dict = {
    'latent_tensor': latent.bfloat16(),
    'latent_format_version': torch.tensor(1)
}
save_file(latent_dict, '/tmp/pma2_latent_v3.safetensors')
print('Saved latent to /tmp/pma2_latent_v3.safetensors')

print('\nNow test with ComfyUI API...')