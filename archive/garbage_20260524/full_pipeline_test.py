"""
full_pipeline_test.py — Full 48-block inference test for PMA² LTX-Video 13B

Loads all serialized block .npz files and runs the complete 48-block forward pass.
Uses the same architecture as test_single_block.py but scaled to all 48 blocks.
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import LTX_NUM_BLOCKS, LTX_HIDDEN_DIM, LTX_MLP_DIM, LTX_LATENT_CHANNELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_HEADS = 32
HEAD_DIM = LTX_HIDDEN_DIM // NUM_HEADS  # 128
SEQ_LEN = 1024  # Small test — full 720p would be 216000

# ---------------------------------------------------------------------------
# Model components (same as test_single_block.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = RMSNorm(hidden_dim)
        self.k_norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q_norm(q.view(B * L, H * d)).view(B, L, H * d)
        k = self.k_norm(k.view(B * L, H * d)).view(B, L, H * d)

        q = q.view(B, L, H, d).transpose(1, 2).contiguous().view(B * H, L, d)
        k = k.view(B, L, H, d).transpose(1, 2).contiguous().view(B * H, L, d)
        v = v.transpose(1, 2).contiguous().view(B * H, L, d)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.view(B, H, L, d).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class FFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(nn.functional.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(LTX_HIDDEN_DIM)
        self.norm2 = RMSNorm(LTX_HIDDEN_DIM)
        self.norm3 = RMSNorm(LTX_HIDDEN_DIM)
        self.attn1 = Attention(LTX_HIDDEN_DIM, NUM_HEADS, HEAD_DIM)
        self.attn2 = Attention(LTX_HIDDEN_DIM, NUM_HEADS, HEAD_DIM)
        self.ffn = FFN(LTX_HIDDEN_DIM, LTX_MLP_DIM)

    def forward(self, x: torch.Tensor, adaln_c: torch.Tensor):
        sc = adaln_c.float().chunk(6, dim=-1)
        shift1, scale1, shift2, scale2, shift3, scale3 = sc
        x = x + self.attn1(self.norm1(x) * (1 + scale1) + shift1)
        x = x + self.attn2(self.norm2(x) * (1 + scale2) + shift2)
        x = x + self.ffn(self.norm3(x) * (1 + scale3) + shift3)
        return x


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=x.device) / half)
        angles = x[:, None].float() * freqs[None, :]
        emb = torch.zeros(x.shape[0], self.dim, dtype=torch.float32, device=x.device)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_npz(path: Path):
    with np.load(path) as data:
        return {k: torch.from_numpy(v).float() for k, v in data.items()}


def load_top_level(path: Path, sin_emb, mlp, final, patchify, proj_out):
    """Load top_level.npz into the model components."""
    top = load_npz(path)

    with torch.no_grad():
        # AdaLN embedder
        mlp[0].weight.copy_(top["top.adaln.emb.0.weight"])
        mlp[0].bias.copy_(top["top.adaln.emb.0.bias"])
        mlp[2].weight.copy_(top["top.adaln.emb.2.weight"])
        mlp[2].bias.copy_(top["top.adaln.emb.2.bias"])
        final.weight.copy_(top["top.adaln.linear.weight"])
        final.bias.copy_(top["top.adaln.linear.bias"])

        # Patchify / proj_out
        patchify.weight.copy_(top["top.patchify.weight"])
        patchify.bias.copy_(top["top.patchify.bias"])
        proj_out.weight.copy_(top["top.proj_out.weight"])
        proj_out.bias.copy_(top["top.proj_out.bias"])

    del top
    gc.collect()


def load_block(npz_path: Path, block: TransformerBlock, block_idx: int):
    """Load one block .npz into a TransformerBlock."""
    data = load_npz(npz_path)
    prefix = f"tb{block_idx}_"

    with torch.no_grad():
        block.attn1.q_proj.weight.copy_(data[prefix + "attn1_to_q_weight"])
        block.attn1.k_proj.weight.copy_(data[prefix + "attn1_to_k_weight"])
        block.attn1.v_proj.weight.copy_(data[prefix + "attn1_to_v_weight"])
        block.attn1.o_proj.weight.copy_(data[prefix + "attn1_to_out_0_weight"])
        block.attn1.q_norm.norm.weight.copy_(data[prefix + "attn1_q_norm_weight"])
        block.attn1.k_norm.norm.weight.copy_(data[prefix + "attn1_k_norm_weight"])

        block.attn2.q_proj.weight.copy_(data[prefix + "attn2_to_q_weight"])
        block.attn2.k_proj.weight.copy_(data[prefix + "attn2_to_k_weight"])
        block.attn2.v_proj.weight.copy_(data[prefix + "attn2_to_v_weight"])
        block.attn2.o_proj.weight.copy_(data[prefix + "attn2_to_out_0_weight"])
        block.attn2.q_norm.norm.weight.copy_(data[prefix + "attn2_q_norm_weight"])
        block.attn2.k_norm.norm.weight.copy_(data[prefix + "attn2_k_norm_weight"])

        block.ffn.fc1.weight.copy_(data[prefix + "ff_net_0_proj_weight"])
        block.ffn.fc1.bias.copy_(data[prefix + "ff_net_0_proj_bias"])
        block.ffn.fc2.weight.copy_(data[prefix + "ff_net_2_weight"])
        block.ffn.fc2.bias.copy_(data[prefix + "ff_net_2_bias"])

    del data
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PMA² Full 48-Block Pipeline Test")
    print("=" * 60)

    blocks_dir = Path("/Users/ryantsudek/Projects/pma2-ltx-video/blocks")
    top_path = blocks_dir / "top_level.npz"

    if not top_path.exists():
        print(f"ERROR: {top_path} not found — run serialize_blocks.py first")
        return

    print(f"\nConfig: {LTX_NUM_BLOCKS} blocks, hidden={LTX_HIDDEN_DIM}, ffn={LTX_MLP_DIM}, latent_ch={LTX_LATENT_CHANNELS}")

    # Build top-level components
    sin_emb = SinusoidalEmbedding(256)
    mlp = nn.Sequential(nn.Linear(256, LTX_HIDDEN_DIM), nn.SiLU(), nn.Linear(LTX_HIDDEN_DIM, LTX_HIDDEN_DIM))
    final = nn.Linear(LTX_HIDDEN_DIM, 6 * LTX_HIDDEN_DIM)
    patchify = nn.Linear(LTX_LATENT_CHANNELS, LTX_HIDDEN_DIM)
    proj_out = nn.Linear(LTX_HIDDEN_DIM, LTX_LATENT_CHANNELS)

    sin_emb.eval(); mlp.eval(); final.eval(); patchify.eval(); proj_out.eval()

    # Load top-level
    print("\n[1/3] Loading top-level weights...")
    t0 = time.time()
    load_top_level(top_path, sin_emb, mlp, final, patchify, proj_out)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Load all 48 blocks
    print(f"\n[2/3] Loading {LTX_NUM_BLOCKS} transformer blocks...")
    t0 = time.time()
    blocks = []
    for i in range(LTX_NUM_BLOCKS):
        block = TransformerBlock()
        block.eval()
        load_block(blocks_dir / f"block_{i:03d}.npz", block, i)
        blocks.append(block)
        if (i + 1) % 12 == 0:
            print(f"  Block {i:02d}/{LTX_NUM_BLOCKS-1} loaded ({len(blocks)} blocks)")
            gc.collect()
    print(f"  All {LTX_NUM_BLOCKS} blocks loaded in {time.time()-t0:.1f}s")

    # Input
    B = 1
    L = SEQ_LEN
    dummy_latent = torch.randn(B, L, LTX_LATENT_CHANNELS, dtype=torch.float32)
    dummy_timestep = torch.tensor([500])

    print(f"\n[3/3] Running forward pass...")
    print(f"  Input: {dummy_latent.shape}, L={L}, full 720p would be L=216000")

    with torch.no_grad():
        # Patchify
        x = patchify(dummy_latent)
        print(f"  After patchify: {x.shape}")

        # AdaLN conditioning
        t_emb = sin_emb(dummy_timestep).float()
        t_emb = mlp(t_emb)
        adaln_c = final(t_emb)
        print(f"  adaln_c: {adaln_c.shape}")

        # Forward through all 48 blocks
        print(f"  Running {LTX_NUM_BLOCKS} blocks...")
        t0 = time.time()
        for i, block in enumerate(blocks):
            x = block(x, adaln_c)
            if (i + 1) % 16 == 0:
                elapsed = time.time() - t0
                print(f"    Block {i:02d}/{LTX_NUM_BLOCKS-1}: output={x.shape}, elapsed={elapsed:.1f}s")

        block_time = time.time() - t0

        # Project back
        out = proj_out(x)
        print(f"\n  Final output: {out.shape}")
        print(f"  Block forward time: {block_time:.1f}s for {L} tokens × {LTX_NUM_BLOCKS} blocks")

    print(f"\nOutput stats:")
    print(f"  mean: {out.mean():.4f}")
    print(f"  std:  {out.std():.4f}")
    print(f"  min:  {out.min():.4f}")
    print(f"  max:  {out.max():.4f}")

    assert not torch.isnan(out).any(), "NaN detected!"
    assert not torch.isinf(out).any(), "Inf detected!"
    assert out.abs().max() > 1e-5, "Output near zero!"

    print("\n✓ FULL PIPELINE PASSED — all 48 blocks run end-to-end.")

    # Estimate full 720p time
    full_L = 216000
    ratio = full_L / L
    estimated_full = block_time * ratio
    print(f"\nEstimated for full 720p (L={full_L}): {estimated_full:.0f}s ({estimated_full/60:.1f} min)")
    print("(Single-threaded, no NVMe streaming, no tiling — just raw compute)")


if __name__ == "__main__":
    main()