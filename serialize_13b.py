"""
serialize_13b.py — Serialize LTX-Video 13B checkpoint into PMA² block format.

Architecture: 48 transformer blocks, hidden_dim=4096, 40 attention heads.
Each block: attn1 (self-attention) + attn2 (cross-attention) + ff_net (MLP).

Checkpoint: ltxv-13b-0.9.8-distilled-fp8.safetensors (~15GB)
Output:    models/ltx_video_13b_pma/
  block_000/ … block_047/
  embedder/
  patch_embed/
  config.json

Precision variants: bf16, w4a6, w5a8
TAPB: frac > 0.7 → w4a6 | 0.3 < frac ≤ 0.7 → w5a8 | frac ≤ 0.3 → bf16
LSS:  blocks loaded one-at-a-time from disk to fit 16GB RAM constraint
"""

import torch
import os
import json
import numpy as np
from safetensors import safe_open

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
CHECKPOINT_PATH = os.path.expanduser(
    "~/Projects/pma2-ltx-video/checkpoints/LTX-Video-13B/ltxv-13b-0.9.8-distilled-fp8.safetensors"
)
OUTPUT_DIR = os.path.expanduser("~/Projects/pma2-ltx-video/models/ltx_video_13b_pma")

# -----------------------------------------------------------------------
# Architecture constants — confirmed from checkpoint inspection
# -----------------------------------------------------------------------
NUM_BLOCKS = 48
HIDDEN_DIM = 4096
FFN_DIM = 16384        # 4× hidden_dim expansion (confirmed from checkpoint)
NUM_HEADS = 40
PRECISIONS = ["bf16", "w4a6", "w5a8"]


# -----------------------------------------------------------------------
# Quantization helpers
# -----------------------------------------------------------------------

def quantize(arr: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric quantization to `bits`-bit. Returns (quantized_int8, scale)."""
    scale = np.abs(arr).max(axis=0, keepdims=True)
    scale[scale == 0] = 1e-6
    max_val = scale.max()
    if max_val == 0:
        max_val = 1e-6
    n_levels = 2 ** (bits - 1) - 1  # 7 for 8-bit, 15 for 16-bit
    quantized = np.round(arr / max_val * n_levels).astype(np.int8)
    return quantized, np.array(max_val, dtype=np.float32)


def short_key(key: str) -> str:
    return key.replace("model.diffusion_model.", "").replace(".", "_")


# -----------------------------------------------------------------------
# Serialization functions
# -----------------------------------------------------------------------

def get_block_keys(f, block_num: int) -> list:
    prefix = f"model.diffusion_model.transformer_blocks.{block_num}."
    return sorted([k for k in f.keys() if k.startswith(prefix)])


def serialize_tensor(t: torch.Tensor, out_dir: str, sname: str) -> dict:
    """Store tensor as float32 npy. Returns metadata."""
    arr = t.detach().cpu().float().numpy()
    out_path = os.path.join(out_dir, f"{sname}.npy")
    np.save(out_path, arr)
    return {"dtype": "float32", "shape": list(arr.shape), "size_bytes": arr.nbytes}


def serialize_quantized(arr: np.ndarray, out_dir: str, sname: str,
                          bits: int, variant: str) -> dict:
    """Quantize and store. Returns metadata."""
    q_arr, scale = quantize(arr, bits)
    q_path = os.path.join(out_dir, f"{sname}_q.npy")
    np.save(q_path, q_arr)
    scale_path = os.path.join(out_dir, f"{sname}_scale.npy")
    np.save(scale_path, scale)
    return {
        "dtype": "quantized",
        "variant": variant,
        "bits": bits,
        "q_dtype": str(q_arr.dtype),
        "shape": list(arr.shape),
        "size_bytes": q_arr.nbytes + scale.nbytes,
    }


def serialize_block(f, block_num: int, out_dir: str) -> dict:
    """Serialize one transformer block to disk at all precision variants."""
    os.makedirs(out_dir, exist_ok=True)

    # Create precision subdirs
    for prec in PRECISIONS:
        os.makedirs(os.path.join(out_dir, prec), exist_ok=True)

    keys = get_block_keys(f, block_num)
    meta = {"block": block_num, "num_tensors": len(keys), "params": {}}

    for key in keys:
        t = f.get_tensor(key)
        sname = short_key(key)
        arr = t.detach().cpu().float().numpy()

        param_meta = {}

        # bf16: store as float32
        param_meta["bf16"] = serialize_tensor(t, os.path.join(out_dir, "bf16"), sname)

        # w4a6 (4-bit weight, 6-bit activation — use 4-bit quantization)
        param_meta["w4a6"] = serialize_quantized(arr, os.path.join(out_dir, "w4a6"), sname, bits=4, variant="w4a6")

        # w5a8 (5-bit weight, 8-bit activation)
        param_meta["w5a8"] = serialize_quantized(arr, os.path.join(out_dir, "w5a8"), sname, bits=5, variant="w5a8")

        meta["params"][sname] = param_meta

    # Write block meta.json
    with open(os.path.join(out_dir, "meta.json"), "w") as mf:
        json.dump(meta, mf, indent=2)

    total_size = sum(p["bf16"]["size_bytes"] for p in meta["params"].values())
    print(f"  Block {block_num:03d}: {len(keys)} tensors, ~{total_size/1e9:.2f}GB (bf16)")
    return meta


def serialize_embedder_and_patch(f, out_dir: str):
    """Serialize embedder + patch embedding (loaded once, shared across all blocks)."""
    os.makedirs(os.path.join(out_dir, "embedder", "bf16"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "patch_embed", "bf16"), exist_ok=True)

    meta = {"embedder": {}, "patch_embed": {}}

    embedder_prefixes = ["adaln_single.", "caption_projection.", "scale_shift_table"]
    patch_prefixes = ["patchify_proj.", "proj_out."]

    for key in f.keys():
        if "transformer_blocks" in key:
            continue

        is_embedder = any(key.startswith(f"model.diffusion_model.{p}") for p in embedder_prefixes)
        is_patch = any(key.startswith(f"model.diffusion_model.{p}") for p in patch_prefixes)

        if not is_embedder and not is_patch:
            continue

        t = f.get_tensor(key)
        sname = short_key(key)
        arr = t.detach().cpu().float().numpy()

        if is_embedder:
            out_path = os.path.join(out_dir, "embedder", "bf16", f"{sname}.npy")
            np.save(out_path, arr)
            meta["embedder"][sname] = {"path": out_path, "shape": list(arr.shape), "size_bytes": arr.nbytes}

        if is_patch:
            out_path = os.path.join(out_dir, "patch_embed", "bf16", f"{sname}.npy")
            np.save(out_path, arr)
            meta["patch_embed"][sname] = {"path": out_path, "shape": list(arr.shape), "size_bytes": arr.nbytes}

    with open(os.path.join(out_dir, "embedder_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Embedder + patch_embed: {len(meta['embedder'])} + {len(meta['patch_embed'])} tensors")


def main():
    print("=" * 60)
    print("PMA² 13B Serialization")
    print("=" * 60)
    print(f"Source: {CHECKPOINT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Blocks: {NUM_BLOCKS}, Hidden: {HIDDEN_DIM}, FFN: {FFN_DIM}, Heads: {NUM_HEADS}")
    print()

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Fresh start — remove any partial output
    if os.path.exists(os.path.join(OUTPUT_DIR, "block_000")):
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with safe_open(CHECKPOINT_PATH, framework="pt") as f:
        # Write config
        config = {
            "model": "ltx-video-13b-0.9.8-distilled-fp8",
            "num_blocks": NUM_BLOCKS,
            "block_range": [0, NUM_BLOCKS - 1],
            "hidden_dim": HIDDEN_DIM,
            "ffn_dim": FFN_DIM,
            "num_attention_heads": NUM_HEADS,
            "latent_channels": 16,
            "precisions": PRECISIONS,
            "source_checkpoint": CHECKPOINT_PATH,
        }
        with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as cf:
            json.dump(config, cf, indent=2)

        print("Serializing embedder + patch_embed...")
        serialize_embedder_and_patch(f, OUTPUT_DIR)

        print("Serializing transformer blocks...")
        for block_num in range(NUM_BLOCKS):
            block_dir = os.path.join(OUTPUT_DIR, f"block_{block_num:03d}")
            try:
                serialize_block(f, block_num, block_dir)
            except Exception as e:
                print(f"  ERROR at block {block_num}: {e}")
                raise

    print()
    print("=" * 60)
    print("Serialization complete!")
    import subprocess
    result = subprocess.run(["du", "-sh", OUTPUT_DIR], capture_output=True, text=True)
    print(f"Total size: {result.stdout.strip()}")


if __name__ == "__main__":
    main()