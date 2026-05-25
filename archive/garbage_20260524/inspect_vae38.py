#!/usr/bin/env python3
"""Try to directly create WanVideoVAE38 and see its expected key structure."""
import sys
sys.path.insert(0, '/Users/ryantsudek/ComfyUI')
sys.path.insert(0, '/Users/ryantsudek/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper')

import os
# Create a minimal test that bypasses the relative import issue
# by copying the necessary code

# Read the wan_video_vae source to understand WanVideoVAE38 init
with open('/Users/ryantsudek/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/wanvideo/wan_video_vae.py') as f:
    src = f.read()

# Find WanVideoVAE38 class definition
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'class WanVideoVAE38' in line:
        print(f'WanVideoVAE38 at line {i+1}')
        # Print 20 lines of init
        for j in range(i, min(i+25, len(lines))):
            print(j+1, lines[j])
        break

# Also find what model it creates
for i, line in enumerate(lines):
    if 'class VideoVAE38_' in line or 'VideoVAE38_(' in line:
        print(f'VideoVAE38_ at line {i+1}: {lines[i].strip()}')