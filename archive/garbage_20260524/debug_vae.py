#!/usr/bin/env python3
"""Test VAE loading and decode using ComfyUI's internal API."""
import sys, time
sys.path.insert(0, '/Users/ryantsudek/ComfyUI')

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

# Step 1: figure out what keys WanVideoVAE38 expects
# Import the actual model class
import importlib.util

# Load the module despite relative imports
wan_spec = importlib.util.spec_from_file_location(
    'wan_video_vae',
    '/Users/ryantsudek/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/wanvideo/wan_video_vae.py'
)
# Can't run it directly due to relative imports. Instead, check what keys
# the ComfyUI load_torch_file produces vs what the model expects.

# Load current VAE file
vae_path = '/Users/ryantsudek/ComfyUI/models/vae/LTX-Video-VAE-BF16.safetensors'
print('Loading current VAE file...')
vae_sd = load_file(vae_path)
print(f'File has {len(vae_sd)} keys, first={next(iter(vae_sd.keys()))}')
has_model = any(k.startswith('model.') for k in vae_sd.keys())
print(f'Has model. prefix: {has_model}')

# Check conv1.shape
if 'model.decoder.conv1.bias' in vae_sd:
    print('model.decoder.conv1.bias shape:', vae_sd['model.decoder.conv1.bias'].shape)
if 'decoder.conv1.bias' in vae_sd:
    print('decoder.conv1.bias shape:', vae_sd['decoder.conv1.bias'].shape)
if 'decoder.conv_in.conv.bias' in vae_sd:
    print('decoder.conv_in.conv.bias shape:', vae_sd['decoder.conv_in.conv.bias'].shape)

# Check top-level keys (no model. prefix)
top_level = [k for k in vae_sd.keys() if k.count('.') == 1]
print(f'Top-level keys (no model. prefix): {top_level}')

# Now try to actually load it using the WanVideoVAE38 class via the internal API
# The WanVideoVAELoader code adds model. prefix if missing

# What if the file uses a DIFFERENT key scheme entirely (e.g. conv1.conv.weight instead of conv1.bias)?
print('\n--- Checking key patterns ---')
for k in list(vae_sd.keys())[:15]:
    print(f'  {k}')