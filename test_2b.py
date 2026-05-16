#!/usr/bin/env python3
"""Smoke test 2B model — correct dims, correct dtypes."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

MODEL_DIR = os.path.expanduser("~/Projects/pma2-ltx-video/models/ltx_video_2b_pma")

with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)

NUM_BLOCKS_2B = cfg["num_blocks"]        # 28
HIDDEN_2B     = cfg["hidden_dim"]        # 2048
FFN_2B        = cfg.get("ffn_dim", 8192) # 8192
LATENT_SHAPE  = cfg.get("latent_shape", [9, 60, 106, 16])
LATENT_C      = cfg.get("latent_channels", 16)  # 16

print(f"2B: {NUM_BLOCKS_2B} blocks, hidden={HIDDEN_2B}, ffn={FFN_2B}, latent_ch={LATENT_C}")

# ---------------------------------------------------------------------------
# Load block tensors using index JSON
# ---------------------------------------------------------------------------
def load_block_tensors(block_idx: int, variant: str = "w4a6") -> dict:
    block_dir = f"{MODEL_DIR}/block_{block_idx:03d}"
    with open(f"{block_dir}/index_{variant}.json") as f:
        index = json.load(f)
    tensors = {}
    for key, info in index["params"].items():
        arr = np.load(info["data_path"])
        tensors[key] = arr.astype(np.float32) if arr.dtype == np.float16 else arr
        if "scale_path" in info:
            tensors[key + "_scale"] = np.load(info["scale_path"])
    return tensors

# ---------------------------------------------------------------------------
# Override pma2_inference globals before importing TransformerBlock
# ---------------------------------------------------------------------------
import pma2_inference
pma2_inference.HIDDEN_DIM  = HIDDEN_2B
pma2_inference.FFN_DIM     = FFN_2B
pma2_inference.NUM_BLOCKS  = NUM_BLOCKS_2B
pma2_inference.MODEL_DIR   = MODEL_DIR

from pma2_inference import RMSNorm, FP8Linear, Attention, FFN, TransformerBlock

# ---------------------------------------------------------------------------
# Build 2B engine
# ---------------------------------------------------------------------------
class Engine2B(nn.Module):
    def __init__(self):
        super().__init__()
        self.adaln_timestep_mlp = nn.Sequential(
            nn.Linear(256, HIDDEN_2B), nn.SiLU(), nn.Linear(HIDDEN_2B, HIDDEN_2B))
        self.adaln_final   = nn.Linear(HIDDEN_2B, 6 * HIDDEN_2B)
        self.caption_proj  = nn.Sequential(
            nn.Linear(HIDDEN_2B, HIDDEN_2B), nn.GELU(), nn.Linear(HIDDEN_2B, HIDDEN_2B))
        self.patchify = nn.Linear(LATENT_C, HIDDEN_2B)  # 16→2048
        self.proj_out = nn.Linear(HIDDEN_2B, LATENT_C)  # 2048→16
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_BLOCKS_2B)])

    def load_embedder_weights(self):
        embed_dir = f"{MODEL_DIR}/embedder"
        fnames = [f for f in os.listdir(embed_dir) if f.endswith(".npy")]
        weights = {f[:-4]: np.load(f"{embed_dir}/{f}") for f in fnames}
        print(f"  embedder keys: {list(weights.keys())}")
        with torch.no_grad():
            # timestep MLP
            for src_w, src_b, dst in [
                ("adaln_single_emb_timestep_embedder_linear_1_weight",
                 "adaln_single_emb_timestep_embedder_linear_1_bias", self.adaln_timestep_mlp[0]),
                ("adaln_single_emb_timestep_embedder_linear_2_weight",
                 "adaln_single_emb_timestep_embedder_linear_2_bias", self.adaln_timestep_mlp[2]),
            ]:
                if src_w in weights:
                    w = torch.from_numpy(weights[src_w]).float().to(torch.bfloat16)
                    dst.weight = nn.Parameter(w)
                if src_b in weights:
                    b = torch.from_numpy(weights[src_b]).float().to(torch.bfloat16)
                    dst.bias   = nn.Parameter(b)
            # adaln final
            for src, dst in [("adaln_single_linear_weight", "adaln_final.weight"),
                              ("adaln_single_linear_bias",   "adaln_final.bias")]:
                if src in weights:
                    w = torch.from_numpy(weights[src]).float().to(torch.bfloat16)
                    setattr(self.adaln_final, dst.split('.')[-1], nn.Parameter(w))
            # caption proj
            for src, i in [("caption_projection_linear_1_weight", 0),
                            ("caption_projection_linear_2_weight", 2)]:
                if src in weights:
                    self.caption_proj[i].weight = nn.Parameter(
                        torch.from_numpy(weights[src]).float().to(torch.bfloat16))
            for src, i in [("caption_projection_linear_1_bias", 0),
                            ("caption_projection_linear_2_bias", 2)]:
                if src in weights:
                    self.caption_proj[i].bias = nn.Parameter(
                        torch.from_numpy(weights[src]).float().to(torch.bfloat16))
            # patchify/proj_out — use direct matmul to bypass nn.Linear dtype issues
            # patchify: [B*L, 16] @ [16, 2048].t() → [B*L, 2048]
            # proj_out:  [B*L, 2048] @ [2048, 16].t() → [B*L, 16]
            pw = torch.from_numpy(weights["patchify_proj_weight"]).float().to(torch.bfloat16)
            ow = torch.from_numpy(weights["proj_out_weight"]).float().to(torch.bfloat16)
            self._patchify_weight = pw.t()  # [16, 2048]
            self._proj_out_weight = ow.t()  # [2048, 16]
            self._patchify_bias = torch.from_numpy(weights["patchify_proj_bias"]).float().to(torch.bfloat16) if "patchify_proj_bias" in weights else None
            self._proj_out_bias  = torch.from_numpy(weights["proj_out_bias"]).float().to(torch.bfloat16) if "proj_out_bias" in weights else None
            print(f"  patchify weight: {pw.shape} -> matmul weight {self._patchify_weight.shape}")

    def forward_block(self, h, adaln_c, block_idx, variant):
        tensors = load_block_tensors(block_idx, variant)
        self.blocks[block_idx].set_weights(tensors)
        return self.blocks[block_idx](h, adaln_c)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("\nBuilding engine...")
engine = Engine2B()
print(f"  norm1: {engine.blocks[0].norm1.norm.normalized_shape}")
print(f"  patchify: {engine.patchify.weight.shape}")
print(f"  proj_out: {engine.proj_out.weight.shape}")

print("\nLoading embedder + patch_embed...")
engine.load_embedder_weights()

# Patchify: [B,T,H,W,16] → [B*L, 16] → [B*L, 2048] via manual matmul
# (nn.Linear dtype restrictions prevent direct use with bf16 weights)
latent = torch.randn(1, *LATENT_SHAPE, LATENT_C)  # [1,9,60,106,16]
B, T, H, W, C = latent.shape
latent_flat = latent.view(B, T * H * W, C).bfloat16()  # [B, L, 16]
patchified = torch.matmul(latent_flat, engine._patchify_weight.T)  # [B, L, 16] @ [16, 2048].t() = [B, L, 2048]
if engine._patchify_bias is not None:
    patchified = patchified + engine._patchify_bias
patchified = patchified.view(B, T * H * W, HIDDEN_2B)  # [B*L, 2048]
print(f"Latent: {latent.shape} → patchified: {patchified.shape}")

# Adaln conditioning
adaln_c = torch.randn(1, 6 * HIDDEN_2B).to(torch.bfloat16)

print(f"\nRunning block 0 (w4a6)...")
tensors = load_block_tensors(0, "w4a6")
print(f"  {len(tensors)} tensors")
out = engine.forward_block(patchified, adaln_c, block_idx=0, variant="w4a6")
print(f"Block 0 out: {out.shape}")
print("\nSMOKE TEST PASSED")