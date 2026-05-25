#!/usr/bin/env python3
"""
serialize_ltx23.py — Stream-serialize LTX-2.3 22B distilled checkpoint into PMA² block format.

Uses indexed access to safetensors — loads one block at a time from disk,
avoiding loading the entire 43GB checkpoint into RAM.

Architecture (from safetensors inspection):
  - 48 transformer blocks
  - Video: hidden=4096, ffn=16384, num_heads=32 (head_dim=128)
  - Audio: hidden=2048, ffn=8192
  - adaln_single: timestep_embedder(256→4096) → (4096→4096) → linear(4096→36864)
  - Each block: attn1/2 (self/cross), FFN, audio_attn1/2, audio_ff, audio_to_video, video_to_audio
"""

import os
import sys
import json
import numpy as np
import torch
import re
from tqdm import tqdm
from safetensors import safe_open

CHECKPOINT_PATH = "/Users/ryantsudek/Projects/pma2-ltx-video/checkpoints/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
MODEL_DIR = "/Users/ryantsudek/Projects/pma2-ltx-video/models/ltx_video_22b_pma"

NUM_BLOCKS = 48
HIDDEN_DIM = 4096
FFN_DIM = 16384
AUDIO_DIM = 2048
AUDIO_FFN_DIM = 8192
NUM_HEADS = 32
HEAD_DIM = 128

def remap_block_key(key):
    """Remap safetensors key to our block format."""
    key = key.replace("model.diffusion_model.transformer_blocks.", "")

    # Handle scale_shift_table variants
    if "scale_shift_table" in key:
        return key.replace(".", "_").replace("-", "_")

    # Handle audio/video prefixed keys
    parts = key.split(".")
    if len(parts) >= 2 and parts[1] in (
        "audio_attn1", "audio_attn2", "audio_ff",
        "audio_to_video_attn", "video_to_audio_attn"
    ):
        return ".".join(parts[1:])

    return ".".join(parts[1:])

def tensor_to_numpy(t):
    """Convert tensor to numpy float32 array."""
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()

def sanitize_key(key):
    """Make key filesystem-safe."""
    return key.replace(".", "_").replace("-", "_").replace("/", "_")

def serialize_embedder(f):
    """Serialize adaln_single embedder to numpy."""
    embedder_dir = f"{MODEL_DIR}/embedder/bf16"
    os.makedirs(embedder_dir, exist_ok=True)

    mappings = {
        "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight": "timestep_embedder.linear_1.weight",
        "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias": "timestep_embedder.linear_1.bias",
        "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.weight": "timestep_embedder.linear_2.weight",
        "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.bias": "timestep_embedder.linear_2.bias",
        "model.diffusion_model.adaln_single.linear.weight": "adaln_linear.weight",
        "model.diffusion_model.adaln_single.linear.bias": "adaln_linear.bias",
    }

    for src_key, dst_key in mappings.items():
        t = f.get_tensor(src_key)
        arr = tensor_to_numpy(t)
        np.save(f"{embedder_dir}/{dst_key}.npy", arr)
        print(f"  embedder/{dst_key}.npy: {arr.shape} {arr.dtype}")

    # Also save audio adaln
    audio_mappings = {
        "model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_1.weight": "audio_timestep_embedder.linear_1.weight",
        "model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_1.bias": "audio_timestep_embedder.linear_1.bias",
        "model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_2.weight": "audio_timestep_embedder.linear_2.weight",
        "model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_2.bias": "audio_timestep_embedder.linear_2.bias",
        "model.diffusion_model.audio_adaln_single.linear.weight": "audio_adaln_linear.weight",
        "model.diffusion_model.audio_adaln_single.linear.bias": "audio_adaln_linear.bias",
    }

    for src_key, dst_key in audio_mappings.items():
        try:
            t = f.get_tensor(src_key)
            arr = tensor_to_numpy(t)
            np.save(f"{embedder_dir}/{dst_key}.npy", arr)
            print(f"  embedder/{dst_key}.npy: {arr.shape} {arr.dtype}")
        except Exception as e:
            print(f"  Skipping {src_key}: {e}")

    # Also audio prompt adaln
    for key in ["linear.weight", "linear.bias", "emb.timestep_embedder.linear_1.weight",
                "emb.timestep_embedder.linear_1.bias", "emb.timestep_embedder.linear_2.weight",
                "emb.timestep_embedder.linear_2.bias"]:
        try:
            src = f"model.diffusion_model.audio_prompt_adaln_single.{key}"
            t = f.get_tensor(src)
            arr = tensor_to_numpy(t)
            dst = f"audio_prompt_adaln_single.{key}".replace(".", "_")
            np.save(f"{embedder_dir}/{dst}.npy", arr)
            print(f"  embedder/{dst}.npy: {arr.shape}")
        except:
            pass

def serialize_patchify(f):
    """Serialize patchify layers."""
    patchify_dir = f"{MODEL_DIR}/patchify/bf16"
    os.makedirs(patchify_dir, exist_ok=True)

    keys = [
        ("model.diffusion_model.video_embedder.proj_in.weight", "video_proj_in.weight"),
        ("model.diffusion_model.video_embedder.proj_in.bias", "video_proj_in.bias"),
        ("model.diffusion_model.video_embedder.proj_out.weight", "video_proj_out.weight"),
        ("model.diffusion_model.video_embedder.proj_out.bias", "video_proj_out.bias"),
        ("model.diffusion_model.audio_embedder.proj_in.weight", "audio_proj_in.weight"),
        ("model.diffusion_model.audio_embedder.proj_in.bias", "audio_proj_in.bias"),
        ("model.diffusion_model.audio_embedder.proj_out.weight", "audio_proj_out.weight"),
        ("model.diffusion_model.audio_embedder.proj_out.bias", "audio_proj_out.bias"),
    ]

    for src_key, dst_key in keys:
        try:
            t = f.get_tensor(src_key)
            arr = tensor_to_numpy(t)
            np.save(f"{patchify_dir}/{dst_key}.npy", arr)
            print(f"  patchify/{dst_key}.npy: {arr.shape}")
        except Exception as e:
            print(f"  WARNING: {src_key} not found: {e}")

def serialize_block(f, block_idx):
    """Serialize one transformer block to numpy."""
    block_dir = f"{MODEL_DIR}/blocks/block_{block_idx:03d}/bf16"
    os.makedirs(block_dir, exist_ok=True)

    all_keys = list(f.keys())
    block_keys = [k for k in all_keys if f"transformer_blocks.{block_idx}." in k]

    if not block_keys:
        print(f"  WARNING: block {block_idx} has no keys!")
        return 0

    meta_keys = []
    for key in block_keys:
        remapped = remap_block_key(key)
        safe_name = sanitize_key(remapped)

        t = f.get_tensor(key)
        arr = tensor_to_numpy(t)
        np.save(f"{block_dir}/{safe_name}.npy", arr)
        meta_keys.append(remapped)

    # Save meta
    meta = {"keys": sorted(meta_keys), "num_tensors": len(meta_keys)}
    with open(f"{MODEL_DIR}/blocks/block_{block_idx:03d}/meta.json", "w") as metaf:
        json.dump(meta, metaf, indent=2)

    return len(block_keys)

def create_config():
    """Create config.json."""
    config = {
        "model": "LTX-2.3 22B distilled v1.1",
        "architecture": "ltx23_video_audio_diT",
        "num_blocks": NUM_BLOCKS,
        "hidden_dim": HIDDEN_DIM,
        "ffn_dim": FFN_DIM,
        "audio_dim": AUDIO_DIM,
        "audio_ffn_dim": AUDIO_FFN_DIM,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "latent_shape": [9, 60, 106, 16],
        "precisions": ["bf16"],
        "checkpoint_path": CHECKPOINT_PATH,
        "has_audio": True,
        "has_video": True,
    }

    with open(f"{MODEL_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config saved: {MODEL_DIR}/config.json")

def main():
    print("=" * 60)
    print("PMA² LTX-2.3 22B Serializer (Streaming)")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    size_gb = os.path.getsize(CHECKPOINT_PATH) / 1e9
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"Size: {size_gb:.1f} GB")

    # Create directories
    os.makedirs(f"{MODEL_DIR}/embedder/bf16", exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/patchify/bf16", exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/blocks", exist_ok=True)

    # Open safetensors (indexed, not loading all into memory)
    print("\nOpening safetensors (indexed access)...")
    f = safe_open(CHECKPOINT_PATH, framework='pt', device='cpu')
    all_keys = list(f.keys())
    print(f"Total tensors: {len(all_keys)}")

    # Serialize embedder
    print("\n--- Serializing Embedder ---")
    serialize_embedder(f)

    # Serialize patchify
    print("\n--- Serializing Patchify ---")
    serialize_patchify(f)

    # Serialize each block
    print("\n--- Serializing 48 Transformer Blocks ---")
    for block_idx in tqdm(range(NUM_BLOCKS), desc="Blocks"):
        count = serialize_block(f, block_idx)
        if block_idx % 10 == 0:
            print(f"  Block {block_idx}: {count} tensors")

    # Create config
    print("\n--- Creating Config ---")
    create_config()

    # Verify block 0
    block0_dir = f"{MODEL_DIR}/blocks/block_000/bf16"
    files = sorted(os.listdir(block0_dir))
    print(f"\nBlock 0: {len(files)} tensor files")
    print(f"First 5: {files[:5]}")

    print("\n" + "=" * 60)
    print("SERIALIZATION COMPLETE")
    print(f"Model: {MODEL_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()