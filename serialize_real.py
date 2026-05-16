"""
serialize_real.py — Serialize real LTX-Video 2B checkpoint into PMA² block format

Reads ltxv-2b-0.9.8-distilled-fp8.safetensors and writes:
  models/ltx_video_2b_pma/
    block_000/  …  block_027/    (28 transformer blocks)
    embedder/                     (timestep + caption embedders)
    patch_embed/                  (patchify + proj_out)
    vae/                          (decoder + encoder)
    config.json

Each block stores:
  - The raw float8_e4m3fn weight tensors for the block's parameters
  - A meta.json with dtype/shape info
  - Index files per precision variant (w4a6, w4a8, w5a8, w6a8)

TAPB: early steps (high frac) → w4a6, late steps → w6a8
LSS:  each block ~70MB at fp8, can be further quantized for streaming
"""

import torch
import os
import json
import shutil
import numpy as np
from safetensors import safe_open
from tqdm import tqdm

CHECKPOINT_PATH = os.path.expanduser(
    "~/Projects/pma2-ltx-video/checkpoints/LTX-Video/ltxv-2b-0.9.8-distilled-fp8.safetensors"
)
OUTPUT_DIR = os.path.expanduser(
    "~/Projects/pma2-ltx-video/models/ltx_video_2b_pma"
)

# ---------------------------------------------------------------------------
# Key structure in ltxv-2b-0.9.8-distilled-fp8.safetensors
# ---------------------------------------------------------------------------
# model.diffusion_model.adaln_single.emb.timestep_embedder.*   → embedder/
# model.diffusion_model.adaln_single.linear.*                  → embedder/
# model.diffusion_model.caption_projection.*                   → embedder/
# model.diffusion_model.patchify_proj.*                         → patch_embed/
# model.diffusion_model.proj_out.*                              → patch_embed/
# model.diffusion_model.scale_shift_table                       → patch_embed/
# model.diffusion_model.transformer_blocks.{N}.*                → block_{N:03d}/
# vae.decoder.*                                               → vae/
# vae.encoder.*                                               → vae/
# vae.per_channel_statistics.*                                → vae/
# ---------------------------------------------------------------------------

PRECISIONS = ["w4a6", "w4a8", "w5a8", "w6a8"]  # index keys only; weights stay fp8

def load_checkpoint():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    f = safe_open(CHECKPOINT_PATH, framework="pt")
    keys = list(f.keys())
    print(f"  Total keys: {len(keys)}")
    return f

def get_block_keys(f, block_num):
    """Get all keys belonging to transformer block N."""
    prefix = f"model.diffusion_model.transformer_blocks.{block_num}."
    return [k for k in f.keys() if k.startswith(prefix)]

def get_embedder_keys(f):
    """Get embedder/caption_projection/patch_embed keys."""
    prefixes = [
        "model.diffusion_model.adaln_single.",
        "model.diffusion_model.caption_projection.",
        "model.diffusion_model.patchify_proj.",
        "model.diffusion_model.proj_out.",
        "model.diffusion_model.scale_shift_table",
    ]
    result = []
    for k in f.keys():
        if any(k.startswith(p) for p in prefixes):
            result.append(k)
    return result

def get_vae_keys(f):
    return [k for k in f.keys() if k.startswith("vae.")]

def save_tensor(f, key, out_dir, precision="fp8"):
    """Save a single tensor. For fp8 weights: store raw bytes + scale.
    For bf16: store as float16 npy.
    Returns metadata dict.
    """
    t = f.get_tensor(key)
    dtype = str(t.dtype).replace("torch.", "")
    shape = list(t.shape)
    size_bytes = t.numel() * t.element_size()

    # Short key for filename (strip model.diffusion_model. etc.)
    short = key.replace("model.diffusion_model.", "").replace(".", "_")

    if dtype == "float8_e4m3fn":
        # FP8 — convert to float32 first (numpy doesn't support float8 directly)
        np_tensor = t.detach().cpu().float().numpy()
        if "to_out" in key or "proj" in key and "weight" in key:
            # Per-channel scale
            if np_tensor.ndim == 2:
                scales = np.abs(np_tensor).max(axis=1)
            else:
                scales = np.abs(np_tensor).max()
            scales[scales == 0] = 1e-6
            scale_path = os.path.join(out_dir, f"{short}_scale.npy")
            np.save(scale_path, scales.astype(np.float32))
        elif "ff.net" in key and "proj" in key:
            if np_tensor.ndim == 2:
                scales = np.abs(np_tensor).max(axis=0)
                scales[scales == 0] = 1e-6
                scale_path = os.path.join(out_dir, f"{short}_scale.npy")
                np.save(scale_path, scales.astype(np.float32))
            else:
                scale_path = None
        else:
            scale = float(np.abs(np_tensor).max())
            if scale == 0: scale = 1e-6
            scale_path = os.path.join(out_dir, f"{short}_scale.npy")
            np.save(scale_path, np.array(scale, dtype=np.float32))

        # Save as float32 npy (bf16→fp32 conversion, compressed)
        data_path = os.path.join(out_dir, f"{short}.npy")
        np.save(data_path, np_tensor.astype(np.float32))
        return {"data_path": data_path, "scale_path": scale_path, "dtype": "float32", "original_dtype": dtype, "shape": shape, "size_bytes": size_bytes}
    else:
        # BF16 / FP16 — convert to float16 for saving
        data_path = os.path.join(out_dir, f"{short}.npy")
        t_fp16 = t.detach().cpu().to(dtype=torch.float16)
        np.save(data_path, t_fp16.numpy())
        return {"data_path": data_path, "dtype": "float16", "shape": shape, "size_bytes": size_bytes}

def serialize_block(f, block_num, out_dir):
    """Serialize one transformer block."""
    block_dir = os.path.join(out_dir, f"block_{block_num:03d}")
    os.makedirs(block_dir, exist_ok=True)

    keys = get_block_keys(f, block_num)
    meta = {}

    for key in keys:
        short = key.replace("model.diffusion_model.transformer_blocks.", "").replace(".", "_")
        m = save_tensor(f, key, block_dir)
        m["original_key"] = key
        meta[short] = m

    # Write meta per precision index
    for prec in PRECISIONS:
        idx = {"block": block_num, "precision": prec, "params": meta}
        idx_path = os.path.join(block_dir, f"index_{prec}.json")
        with open(idx_path, "w") as fp:
            json.dump(idx, fp, indent=2)

    # Write block-level meta
    with open(os.path.join(block_dir, "meta.json"), "w") as fp:
        json.dump({"block_num": block_num, "num_params": len(keys), "keys": list(meta.keys())}, fp)

    return len(keys)

def serialize_embedder(f, out_dir):
    """Serialize embedder (timestep + caption + patch)."""
    ed = os.path.join(out_dir, "embedder")
    os.makedirs(ed, exist_ok=True)
    keys = get_embedder_keys(f)
    meta = {}
    for key in keys:
        short = key.replace("model.diffusion_model.", "").replace(".", "_")
        m = save_tensor(f, key, ed)
        m["original_key"] = key
        meta[short] = m
    with open(os.path.join(ed, "meta.json"), "w") as fp:
        json.dump({"type": "embedder", "num_params": len(keys)}, fp)
    return len(keys)

def serialize_vae(f, out_dir):
    """Serialize VAE (decoder + encoder)."""
    vd = os.path.join(out_dir, "vae")
    os.makedirs(vd, exist_ok=True)
    keys = get_vae_keys(f)
    meta = {}
    for key in keys:
        short = key.replace(".", "_")
        m = save_tensor(f, key, vd)
        m["original_key"] = key
        meta[short] = m
    with open(os.path.join(vd, "meta.json"), "w") as fp:
        json.dump({"type": "vae", "num_params": len(keys)}, fp)
    return len(keys)

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Output dir exists — removing: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    f = load_checkpoint()

    # Figure out block range
    dm_keys = [k for k in f.keys() if "transformer_blocks." in k]
    import re
    nums = sorted(set(int(re.search(r"transformer_blocks\.(\d+)\.", k).group(1)) for k in dm_keys))
    print(f"  Transformer blocks: {nums[0]}–{nums[-1]} ({len(nums)} blocks)")

    # Serialize embedder
    print("Serializing embedder...")
    n_ed = serialize_embedder(f, OUTPUT_DIR)
    print(f"  {n_ed} params → embedder/")

    # Serialize each block
    total_params = 0
    print("Serializing transformer blocks...")
    for bn in tqdm(nums, unit="block"):
        n = serialize_block(f, bn, OUTPUT_DIR)
        total_params += n

    # Serialize VAE
    print("Serializing VAE...")
    n_vae = serialize_vae(f, OUTPUT_DIR)
    print(f"  {n_vae} params → vae/")

    # Write top-level config
    cfg = {
        "model": "ltx-video-2b-0.9.8-distilled-fp8",
        "num_blocks": len(nums),
        "block_range": [nums[0], nums[-1]],
        "hidden_dim": 2048,
        "latent_channels": 16,
        "source_checkpoint": CHECKPOINT_PATH,
        "precisions": PRECISIONS,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as fp:
        json.dump(cfg, fp, indent=2)

    size_total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(OUTPUT_DIR)
        for f in fn
    )
    print(f"\nDone. {total_params} block params + {n_ed} embedder + {n_vae} VAE")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total size: {size_total/1e9:.2f} GB")

if __name__ == "__main__":
    main()