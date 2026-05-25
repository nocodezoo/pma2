#!/usr/bin/env python3
"""
PMA² LTX-Video 2.3 — Direct Safetensors Engine
Reads from ltxv-13b-0.9.8-distilled-fp8.safetensors using memory-mapped I/O.
No GGUF, no Q4_K dequantization — bfloat16 weights streamed layer by layer.

Architecture: 48 blocks, each ~200MB (bf16), stream one block at a time.
Total: 15.7GB model → load one block (~200MB) → compute → load next.
"""

import sys, os, mmap, struct, argparse, time
import numpy as np

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
MODEL_PATH = '/Users/ryantsudek/Projects/pma2-ltx-video/checkpoints/LTX-Video-13B/ltxv-13b-0.9.8-distilled-fp8.safetensors'
TEXT_ENC_PATH = '/Users/ryantsudek/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors'
VAE_PATH = '/Users/ryantsudek/ComfyUI/models/vae/LTX-Video-VAE-BF16.safetensors'

# -----------------------------------------------------------------------
# Model Config (from safetensors inspection)
# -----------------------------------------------------------------------
HIDDEN_DIM = 4096
AUDIO_DIM = 2048
NUM_BLOCKS = 48
NUM_HEADS = 32
HEAD_DIM = 128
FFN_DIM = 16384
LATENT_T = 9
LATENT_H = 60
LATENT_W = 106
LATENT_C = 16

# -----------------------------------------------------------------------
# Safetensors MMap Loader
# -----------------------------------------------------------------------
def load_safetensors_mmap(path):
    """Memory-map a safetensors file for zero-copy reads."""
    with open(path, 'rb') as f:
        # Read header size (8 bytes)
        header_size_bytes = f.read(8)
        header_size = int.from_bytes(header_size_bytes, 'little')
        
        # Read header JSON
        header = f.read(header_size)
        
        # Parse JSON to get tensor metadata
        import json
        metadata = json.loads(header)
        
        # Get tensor info
        tensors = {}
        for name, info in metadata.items():
            data_offset = info['data_offsets'][0]
            data_end = info['data_offsets'][1]
            shape = info['shape']
            dtype = info['dtype']
            tensors[name] = {
                'offset': 8 + header_size + data_offset,
                'size': data_end - data_offset,
                'shape': shape,
                'dtype': dtype
            }
        
    # Create mmap
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    
    return mm, tensors


def read_tensor(mm, tensor_info, np_dtype):
    """Read a tensor from mmap without loading full file."""
    offset = tensor_info['offset']
    size = tensor_info['size']
    shape = tensor_info['shape']
    
    # Read raw bytes
    mm.seek(offset)
    raw = mm.read(size)
    
    # Convert to numpy array
    arr = np.frombuffer(raw, dtype=np_dtype).copy().reshape(shape)
    return arr


# -----------------------------------------------------------------------
# RMSNorm (numpy)
# -----------------------------------------------------------------------
class NumpyRMSNorm:
    def __init__(self, weight):
        self.w = weight.astype(np.float32)
        self.eps = 1e-6
    
    def __call__(self, x):
        axis = list(range(x.ndim - 1))
        norm = np.sqrt(np.mean(x.astype(np.float32)**2, axis=axis, keepdims=True) + self.eps)
        return (x / norm) * self.w


# -----------------------------------------------------------------------
# Load Model Metadata
# -----------------------------------------------------------------------
print(f"Loading safetensors: {MODEL_PATH}")
t0 = time.time()
mm, tensor_info = load_safetensors_mmap(MODEL_PATH)
print(f"  Indexed in {time.time()-t0:.1f}s — {len(tensor_info)} tensors")

# -----------------------------------------------------------------------
# Show Model Structure
# -----------------------------------------------------------------------
print("\nModel structure:")
keys = sorted(tensor_info.keys())
for k in keys[:30]:
    info = tensor_info[k]
    shape_str = str(info['shape']).replace(' ', '')
    print(f"  {k}: {shape_str}, {info['dtype']}")

# Find transformer_blocks structure
block0_keys = [k for k in keys if k.startswith('model.diffusion_model.transformer_blocks.0.')]
print(f"\nTransformer block 0: {len(block0_keys)} tensors")
cats = {}
for k in block0_keys:
    cat = '.'.join(k.split('.')[4:]).split('.')[0]
    cats.setdefault(cat, []).append(k)
for c in sorted(cats.keys()):
    print(f"  {c}: {len(cats[c])} tensors")

# Count total blocks
block_indices = set()
for k in keys:
    if 'transformer_blocks.' in k:
        parts = k.split('.')
        for p in parts[2:4]:
            try:
                idx = int(p)
                block_indices.add(idx)
            except:
                pass
print(f"\nBlock indices found: {sorted(block_indices)[:10]}... total: {len(block_indices)}")