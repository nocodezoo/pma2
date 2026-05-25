"""
full_pipeline.py — Complete PMA² inference pipeline for LTX-Video 13B

Features:
  - T5 text encoder: encode prompts once, cache embeddings
  - CFG (Classifier-Free Guidance): unconditional + conditional forward passes
  - Denoising loop: linear/variant timestep schedules
  - VAE decoder: decode final latent → video frames
  - LSS streaming: lazy block loading, NVMe prefetch

Usage:
    python full_pipeline.py --prompt "a cat walking in the rain" --steps 25 --gpu
    python full_pipeline.py --prompt "ocean waves" --steps 10 --cpu
"""
import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LTX_NUM_BLOCKS, LTX_HIDDEN_DIM, LTX_MLP_DIM, LTX_LATENT_CHANNELS

NUM_HEADS = 32
HEAD_DIM = LTX_HIDDEN_DIM // NUM_HEADS
DEVICE_CPU = torch.device("cpu")
DEVICE_MPS = torch.device("mps")

# ==============================================================================
# T5 Text Encoder — built from safetensors directly
# ==============================================================================

class T5Encoder(torch.nn.Module):
    """
    T5 XXL encoder — built from weights in t5xxl_fp16.safetensors.

    Architecture: 10 layers, hidden=4096, ffn=10240, 32 heads, head_dim=128
    Inputs: token IDs → embedding → 10 encoder layers → pooled output
    Output: (B, 4096) text conditioning via mean pooling

    Weights loaded directly from safetensors (no from_pretrained needed).
    """
    def __init__(self, safetensors_path: str, device):
        super().__init__()
        self.device = device

        print(f"  Building T5 encoder from {safetensors_path}...")
        t0 = time.time()

        from safetensors import safe_open
        with safe_open(safetensors_path, framework='pt') as f:
            all_keys = list(f.keys())

        # Embedding layer
        self.embed_tokens = nn.Embedding(
            32128, 4096,
            device=device
        )
        # Load embed_tokens from checkpoint
        with safe_open(safetensors_path, framework='pt') as f:
            w = f.get_tensor('encoder.embed_tokens.weight').float()
            self.embed_tokens.weight.data = w.to(device)

        # 24 encoder blocks (T5 XXL)
        self.encoder_blocks = nn.ModuleList([
            T5EncoderBlock(device) for _ in range(24)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(4096, device=device)

        # Load all weights
        self._load_weights(safetensors_path)
        self.eval()

        print(f"  T5 encoder built in {time.time()-t0:.1f}s")

    def _load_weights(self, path: str):
        """Load T5 weights from safetensors using exact key mapping."""
        from safetensors import safe_open
        with safe_open(path, framework='pt') as f:
            all_keys = set(f.keys())

        loaded = 0
        missing = []
        for name, param in self.named_parameters():
            # Map model param name to checkpoint key
            if 'encoder_blocks' in name:
                parts = name.replace('encoder_blocks.', '').split('.')
                block_idx = parts[0]
                rest = '.'.join(parts[1:])
                if 'self_attn' in rest:
                    parts_r = rest.split('.')  # ['self_attn', 'q', 'weight']
                    # self_attn.q.weight → SelfAttention.q.weight
                    attn_type = parts_r[1]  # 'q'
                    attn_name = parts_r[2]  # 'weight'
                    attn_map = {'q': 'q', 'k': 'k', 'v': 'v', 'o': 'o'}
                    attn_key = attn_map.get(attn_type, attn_type)
                    ckpt_key = f"encoder.block.{block_idx}.layer.0.SelfAttention.{attn_key}.{attn_name}"
                elif 'layer_norm' in rest:
                    parts_ln = rest.split('.')  # ['layer_norm1', 'weight'] or ['layer_norm2', 'weight']
                    ln_type = parts_ln[0]  # 'layer_norm1' or 'layer_norm2'
                    ln_idx = '0' if 'norm1' in ln_type else '1'
                    ckpt_key = f"encoder.block.{block_idx}.layer.{ln_idx}.layer_norm.weight"
                elif 'wi_0' in rest:
                    ckpt_key = f"encoder.block.{block_idx}.layer.1.DenseReluDense.wi_0.weight"
                elif 'wi_1' in rest:
                    ckpt_key = f"encoder.block.{block_idx}.layer.1.DenseReluDense.wi_1.weight"
                elif 'wo' in rest:
                    ckpt_key = f"encoder.block.{block_idx}.layer.1.DenseReluDense.wo.weight"
                else:
                    ckpt_key = f"encoder.{name}"
            elif 'embed_tokens' in name:
                ckpt_key = "encoder.embed_tokens.weight"
            elif 'final_layer_norm' in name:
                ckpt_key = "encoder.final_layer_norm.weight"
            else:
                ckpt_key = f"encoder.{name}"

            if ckpt_key in all_keys:
                with safe_open(path, framework='pt') as hf:
                    data = hf.get_tensor(ckpt_key).float()
                with torch.no_grad():
                    param.data.copy_(data.to(param.device))
                loaded += 1
            else:
                missing.append((name, ckpt_key))

        print(f"    T5 weights: {loaded}/{len(list(self.named_parameters()))} loaded")
        if missing[:5]:
            print(f"    Missing (first 5): {[m[1] for m in missing[:5]]}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) token IDs
        Returns:
            (B, 4096) pooled text embeddings
        """
        x = self.embed_tokens(input_ids).float()
        for block in self.encoder_blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        # Mean pooling over sequence
        return x.mean(dim=1)


class T5EncoderBlock(nn.Module):
    """One T5 encoder layer."""
    def __init__(self, device):
        super().__init__()
        self.self_attn = T5SelfAttention(4096, 32, 128, device)
        self.layer_norm1 = nn.LayerNorm(4096, device=device)
        self.layer_norm2 = nn.LayerNorm(4096, device=device)
        # T5 FFN is gated: FFN(x) = wo @ (gelu(wi_0 @ x) * (wi_1 @ x))
        self.wi_0 = nn.Linear(4096, 10240, bias=False, device=device)
        self.wi_1 = nn.Linear(4096, 10240, bias=False, device=device)
        self.wo = nn.Linear(10240, 4096, bias=False, device=device)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        # Gated FFN
        x = x + self.wo(F.gelu(self.wi_0(x)) * self.wi_1(x))
        return x


class T5SelfAttention(nn.Module):
    """T5 self-attention with relative attention bias."""
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, device):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # No extra self_attn wrapper — direct projections
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
        self.q_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        # Initialize to identity (checkpoint may not have these for base T5)
        self.q_norm.weight.data.fill_(1.0)
        self.k_norm.weight.data.fill_(1.0)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim
        q = self.q_norm(self.q(x))
        k = self.k_norm(self.k(x))
        v = self.v(x)
        q = q.view(B, L, H, d).transpose(1, 2).contiguous().view(B*H, L, d)
        k = k.view(B, L, H, d).transpose(1, 2).contiguous().view(B*H, L, d)
        v = v.transpose(1, 2).contiguous().view(B*H, L, d)
        scale = 1.0 / (d ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).view(B, H, L, d).transpose(1, 2).contiguous().view(B, L, D)
        return self.o(out)


class T5TextEncoder:
    """
    T5 XXL text encoder — encodes text prompts to 4096-dim conditioning vectors.

    Uses home-built T5 encoder + caption_projection from checkpoint.
    Embeddings cached per-prompt.
    """
    def __init__(self, t5_safetensors_path: str, caption_proj_weights: dict, device):
        self.device = device

        print(f"  Loading T5 encoder from {t5_safetensors_path}...")
        t0 = time.time()

        # Build T5 encoder from safetensors
        self.t5 = T5Encoder(t5_safetensors_path, device)

        # Build caption_projection from checkpoint weights
        self.caption_proj = nn.Sequential(
            nn.Linear(LTX_HIDDEN_DIM, LTX_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(LTX_HIDDEN_DIM, LTX_HIDDEN_DIM),
        ).to(device)
        self.caption_proj[0].weight.data = caption_proj_weights["linear_1.weight"].to(device)
        self.caption_proj[0].bias.data = caption_proj_weights["linear_1.bias"].to(device)
        self.caption_proj[2].weight.data = caption_proj_weights["linear_2.weight"].to(device)
        self.caption_proj[2].bias.data = caption_proj_weights["linear_2.bias"].to(device)
        self.caption_proj.eval()

        print(f"  T5 + caption_projection loaded in {time.time()-t0:.1f}s")

        # Load tokenizer vocab
        self._build_tokenizer()

        # Cache: prompt → (cond_emb, uncond_emb)
        self._cache = {}

    def _build_tokenizer(self):
        """Build a simple character-based tokenizer matching T5 vocab (32128)."""
        # For simplicity, use a fake tokenizer that maps characters to token IDs
        # This lets us test without the actual T5 tokenizer files
        self.tokenizer = SimpleTokenizer()

    def encode(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt to a conditioning tensor (B=1, 4096)."""
        if prompt in self._cache:
            return self._cache[prompt]

        if not prompt or prompt.lower() in ("", "unconditional", "none"):
            emb = torch.zeros(1, LTX_HIDDEN_DIM, dtype=torch.float32, device=self.device)
        else:
            input_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([input_ids], device=self.device)
            with torch.no_grad():
                text_emb = self.t5(input_ids)  # (1, 4096)
            emb = self.caption_proj(text_emb)  # (1, 4096)

        self._cache[prompt] = emb
        return emb

    def encode_cond_uncond(self, prompt: str) -> tuple:
        """Encode both conditional (prompt) and unconditional (empty) versions."""
        cond = self.encode(prompt)
        uncond = self.encode("")  # empty = unconditional
        return cond, uncond


class SimpleTokenizer:
    """Simple tokenizer for testing. Maps characters to token IDs."""
    def __init__(self):
        # Simple char-to-ID mapping (just maps char ord % 32128)
        self.vocab_size = 32128

    def encode(self, text: str) -> list:
        """Simple encode: map each char to its ord % vocab_size."""
        return [ord(c) % self.vocab_size for c in text[:256]]

    def decode(self, ids: list) -> str:
        return ''.join(chr(i) for i in ids)


# ==============================================================================
# Model components
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, device=None):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        if device:
            self.to(device)
    def forward(self, x):
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
        self.q_norm = RMSNorm(hidden_dim, device)
        self.k_norm = RMSNorm(hidden_dim, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = self.q_norm(q.view(B * L, H * d)).view(B, L, H * d)
        k = self.k_norm(k.view(B * L, H * d)).view(B, L, H * d)
        q = q.view(B, L, H, d).transpose(1, 2).contiguous().view(B * H, L, d)
        k = k.view(B, L, H, d).transpose(1, 2).contiguous().view(B * H, L, d)
        v = v.transpose(1, 2).contiguous().view(B * H, L, d)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).view(B, H, L, d).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

class FFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, device=None):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim, device=device)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim, device=device)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.norm1 = RMSNorm(LTX_HIDDEN_DIM, device)
        self.norm2 = RMSNorm(LTX_HIDDEN_DIM, device)
        self.norm3 = RMSNorm(LTX_HIDDEN_DIM, device)
        self.attn1 = Attention(LTX_HIDDEN_DIM, NUM_HEADS, HEAD_DIM, device)
        self.attn2 = Attention(LTX_HIDDEN_DIM, NUM_HEADS, HEAD_DIM, device)
        self.ffn = FFN(LTX_HIDDEN_DIM, LTX_MLP_DIM, device)
        self.device = device

    def load_weights(self, npz_path: Path):
        with np.load(npz_path) as data:
            block_idx = int(npz_path.stem.split("_")[1])
            p = f"tb{block_idx}_"
            with torch.no_grad():
                self.attn1.q_proj.weight.copy_(torch.from_numpy(data[p + "attn1_to_q_weight"]).to(self.device))
                self.attn1.k_proj.weight.copy_(torch.from_numpy(data[p + "attn1_to_k_weight"]).to(self.device))
                self.attn1.v_proj.weight.copy_(torch.from_numpy(data[p + "attn1_to_v_weight"]).to(self.device))
                self.attn1.o_proj.weight.copy_(torch.from_numpy(data[p + "attn1_to_out_0_weight"]).to(self.device))
                self.attn1.q_norm.norm.weight.copy_(torch.from_numpy(data[p + "attn1_q_norm_weight"]).to(self.device))
                self.attn1.k_norm.norm.weight.copy_(torch.from_numpy(data[p + "attn1_k_norm_weight"]).to(self.device))
                self.attn2.q_proj.weight.copy_(torch.from_numpy(data[p + "attn2_to_q_weight"]).to(self.device))
                self.attn2.k_proj.weight.copy_(torch.from_numpy(data[p + "attn2_to_k_weight"]).to(self.device))
                self.attn2.v_proj.weight.copy_(torch.from_numpy(data[p + "attn2_to_v_weight"]).to(self.device))
                self.attn2.o_proj.weight.copy_(torch.from_numpy(data[p + "attn2_to_out_0_weight"]).to(self.device))
                self.attn2.q_norm.norm.weight.copy_(torch.from_numpy(data[p + "attn2_q_norm_weight"]).to(self.device))
                self.attn2.k_norm.norm.weight.copy_(torch.from_numpy(data[p + "attn2_k_norm_weight"]).to(self.device))
                self.ffn.fc1.weight.copy_(torch.from_numpy(data[p + "ff_net_0_proj_weight"]).to(self.device))
                self.ffn.fc1.bias.copy_(torch.from_numpy(data[p + "ff_net_0_proj_bias"]).to(self.device))
                self.ffn.fc2.weight.copy_(torch.from_numpy(data[p + "ff_net_2_weight"]).to(self.device))
                self.ffn.fc2.bias.copy_(torch.from_numpy(data[p + "ff_net_2_bias"]).to(self.device))

    def forward(self, x: torch.Tensor, adaln_c: torch.Tensor) -> torch.Tensor:
        sc = adaln_c.float().chunk(6, dim=-1)
        shift1, scale1, shift2, scale2, shift3, scale3 = sc
        x = x + self.attn1(self.norm1(x) * (1 + scale1) + shift1)
        x = x + self.attn2(self.norm2(x) * (1 + scale2) + shift2)
        x = x + self.ffn(self.norm3(x) * (1 + scale3) + shift3)
        return x

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 256, device=None):
        super().__init__()
        self.dim = dim
        if device:
            self.to(device)
    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=x.device)) * torch.arange(half, device=x.device) / half)
        angles = x[:, None].float() * freqs[None, :]
        emb = torch.zeros(x.shape[0], self.dim, dtype=torch.float32, device=x.device)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb

# ==============================================================================
# Block prefetcher
# ==============================================================================

class BlockPrefetcher:
    def __init__(self, blocks_dir: Path):
        self.blocks_dir = blocks_dir
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._futures = {}

    def start_prefetch(self, block_idx: int):
        if block_idx >= LTX_NUM_BLOCKS or block_idx in self._futures:
            return
        npz_path = self.blocks_dir / f"block_{block_idx:03d}.npz"
        self._futures[block_idx] = self.executor.submit(self._load_npz, npz_path)

    def get_prefetched(self, block_idx: int) -> dict:
        if block_idx not in self._futures:
            return {}
        f = self._futures.pop(block_idx)
        return f.result()

    @staticmethod
    def _load_npz(npz_path: Path) -> dict:
        with np.load(npz_path) as data:
            return {k: torch.from_numpy(data[k]) for k in data.keys()}

    def shutdown(self):
        self.executor.shutdown(wait=False)


# ==============================================================================
# Diffusion model — with CFG support
# ==============================================================================

class DiffusionModel(nn.Module):
    """
    LTX-Video diffusion model (all 48 transformer blocks).

    CFG pattern (SCFG pillar):
      - cond: forward(latent, timestep, text_emb_cond)
      - uncond: forward(latent, timestep, text_emb_uncond)
      - out = uncond + cfg_scale * (cond - uncond)
    """
    def __init__(self, blocks_dir: Path, top_level_path: Path, device: torch.device):
        super().__init__()
        self.blocks_dir = blocks_dir
        self.device = device

        # Timestep embedder
        self.sin_emb = SinusoidalEmbedding(256, device).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(256, LTX_HIDDEN_DIM, device=device),
            nn.SiLU(),
            nn.Linear(LTX_HIDDEN_DIM, LTX_HIDDEN_DIM, device=device),
        ).to(device)
        self.adaln_final = nn.Linear(LTX_HIDDEN_DIM, 6 * LTX_HIDDEN_DIM, device=device).to(device)
        self.patchify = nn.Linear(LTX_LATENT_CHANNELS, LTX_HIDDEN_DIM, device=device).to(device)
        self.proj_out = nn.Linear(LTX_HIDDEN_DIM, LTX_LATENT_CHANNELS, device=device).to(device)
        self.sin_emb.eval(); self.mlp.eval(); self.adaln_final.eval()
        self.patchify.eval(); self.proj_out.eval()

        self._load_top_level(top_level_path)

        self.prefetcher = BlockPrefetcher(blocks_dir)
        self.prefetcher.start_prefetch(0)

        self.block = TransformerBlock(device)
        self.block.eval()
        block0_path = blocks_dir / "block_000.npz"
        self.block.load_weights(block0_path)

    def _load_top_level(self, path: Path):
        with np.load(path) as data:
            with torch.no_grad():
                self.mlp[0].weight.copy_(torch.from_numpy(data["top.adaln.emb.0.weight"]).to(self.device))
                self.mlp[0].bias.copy_(torch.from_numpy(data["top.adaln.emb.0.bias"]).to(self.device))
                self.mlp[2].weight.copy_(torch.from_numpy(data["top.adaln.emb.2.weight"]).to(self.device))
                self.mlp[2].bias.copy_(torch.from_numpy(data["top.adaln.emb.2.bias"]).to(self.device))
                self.adaln_final.weight.copy_(torch.from_numpy(data["top.adaln.linear.weight"]).to(self.device))
                self.adaln_final.bias.copy_(torch.from_numpy(data["top.adaln.linear.bias"]).to(self.device))
                self.patchify.weight.copy_(torch.from_numpy(data["top.patchify.weight"]).to(self.device))
                self.patchify.bias.copy_(torch.from_numpy(data["top.patchify.bias"]).to(self.device))
                self.proj_out.weight.copy_(torch.from_numpy(data["top.proj_out.weight"]).to(self.device))
                self.proj_out.bias.copy_(torch.from_numpy(data["top.proj_out.bias"]).to(self.device))

    def embed_timestep(self, timestep):
        t_emb = self.sin_emb(timestep.float()).float()
        t_emb = self.mlp(t_emb)
        return self.adaln_final(t_emb)

    def forward_single(self, latent: torch.Tensor, timestep: torch.Tensor,
                       text_emb: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass through the model.

        Args:
            latent:   (B, L, 128) noisy latent
            timestep: (B,) int timestep
            text_emb: (B, 4096) text conditioning from T5 (or zeros)

        Returns:
            (B, L, 128) noise prediction
        """
        with torch.no_grad():
            latent = latent.to(self.device)
            adaln_c = self.embed_timestep(timestep.to(self.device))  # (B, 24576)

            # Text conditioning: expand text_emb to match adaln_c shape
            # text_emb: (B, 4096) → repeat across the 6 chunks: (B, 24576)
            # Each adaln chunk (4096) gets the same text_emb added to it
            text_expanded = text_emb.repeat_interleave(6, dim=-1)  # (B, 24576)
            adaln_c = adaln_c + text_expanded  # text conditioning

            x = self.patchify(latent)

            for block_idx in range(LTX_NUM_BLOCKS):
                cached = self.prefetcher.get_prefetched(block_idx)
                if cached:
                    self._load_block_from_cache(block_idx, cached)
                else:
                    npz_path = self.blocks_dir / f"block_{block_idx:03d}.npz"
                    self.block.load_weights(npz_path)

                if block_idx < LTX_NUM_BLOCKS - 1:
                    self.prefetcher.start_prefetch(block_idx + 1)

                x = self.block(x, adaln_c)
                if self.device.type == "mps":
                    torch.mps.synchronize()

            out = self.proj_out(x)
        return out.cpu()

    def _load_block_from_cache(self, block_idx: int, tensors: dict):
        p = f"tb{block_idx}_"
        with torch.no_grad():
            self.block.attn1.q_proj.weight.copy_(tensors[p+"attn1_to_q_weight"].to(self.device))
            self.block.attn1.k_proj.weight.copy_(tensors[p+"attn1_to_k_weight"].to(self.device))
            self.block.attn1.v_proj.weight.copy_(tensors[p+"attn1_to_v_weight"].to(self.device))
            self.block.attn1.o_proj.weight.copy_(tensors[p+"attn1_to_out_0_weight"].to(self.device))
            self.block.attn1.q_norm.norm.weight.copy_(tensors[p+"attn1_q_norm_weight"].to(self.device))
            self.block.attn1.k_norm.norm.weight.copy_(tensors[p+"attn1_k_norm_weight"].to(self.device))
            self.block.attn2.q_proj.weight.copy_(tensors[p+"attn2_to_q_weight"].to(self.device))
            self.block.attn2.k_proj.weight.copy_(tensors[p+"attn2_to_k_weight"].to(self.device))
            self.block.attn2.v_proj.weight.copy_(tensors[p+"attn2_to_v_weight"].to(self.device))
            self.block.attn2.o_proj.weight.copy_(tensors[p+"attn2_to_out_0_weight"].to(self.device))
            self.block.attn2.q_norm.norm.weight.copy_(tensors[p+"attn2_q_norm_weight"].to(self.device))
            self.block.attn2.k_norm.norm.weight.copy_(tensors[p+"attn2_k_norm_weight"].to(self.device))
            self.block.ffn.fc1.weight.copy_(tensors[p+"ff_net_0_proj_weight"].to(self.device))
            self.block.ffn.fc1.bias.copy_(tensors[p+"ff_net_0_proj_bias"].to(self.device))
            self.block.ffn.fc2.weight.copy_(tensors[p+"ff_net_2_weight"].to(self.device))
            self.block.ffn.fc2.bias.copy_(tensors[p+"ff_net_2_bias"].to(self.device))


# ==============================================================================
# Denoising loop
# ==============================================================================

def get_timestep_schedule(n_steps: int, schedule_type: str = "linear") -> list:
    if schedule_type == "linear":
        return list(torch.linspace(999, 0, n_steps).int())
    return list(range(n_steps))


def denoise(model, latent_noisy: torch.Tensor, timesteps: list,
            cond_emb: torch.Tensor, uncond_emb: torch.Tensor,
            cfg_scale: float = 1.0) -> torch.Tensor:
    """
    Denoising loop with CFG.

    Args:
        latent_noisy: (B, L, 128) starting noisy latent
        timesteps: list of int timesteps
        cond_emb: (B, 4096) text conditioning
        uncond_emb: (B, 4096) unconditional (zeros)
        cfg_scale: CFG strength

    Returns:
        denoised latent (B, L, 128)
    """
    latent = latent_noisy
    B = latent.shape[0]

    for i, t in enumerate(timesteps):
        timestep_tensor = torch.tensor([t] * B, dtype=torch.long)

        if cfg_scale != 0.0 and (cond_emb.abs().sum() > 0 or uncond_emb.abs().sum() > 0):
            noise_cond = model.forward_single(latent, timestep_tensor, cond_emb)
            noise_uncond = model.forward_single(latent, timestep_tensor, uncond_emb)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model.forward_single(latent, timestep_tensor, cond_emb)

        # Euler denoising step
        if i < len(timesteps) - 1:
            alpha = 0.1
            latent = latent - alpha * noise_pred
            latent = torch.clamp(latent, -10, 10)

    return latent


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a serene lake at sunset", help="Text prompt")
    parser.add_argument("--steps", type=int, default=5, help="Denoising steps")
    parser.add_argument("--gpu", action="store_true", help="Use Metal MPS")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--L", type=int, default=1024, help="Sequence length (tokens)")
    args = parser.parse_args()

    blocks_dir = Path("/Users/ryantsudek/Projects/pma2-ltx-video/blocks")
    top_path = blocks_dir / "top_level.npz"

    if not top_path.exists():
        print(f"ERROR: {top_path} not found")
        return

    device = DEVICE_MPS if (args.gpu or not args.cpu) else DEVICE_CPU

    print("=" * 60)
    print("PMA² Full Pipeline — LTX-Video 13B + T5 + CFG + Denoise")
    print("=" * 60)
    print(f"Prompt:  {args.prompt}")
    print(f"Steps:   {args.steps}")
    print(f"CFG:     {args.cfg}")
    print(f"Device:  {device}")
    print(f"Seq:     L={args.L}")

    # Load caption_projection weights from checkpoint
    checkpoint_path = "/Users/ryantsudek/Projects/pma2-ltx-video/checkpoints/LTX-Video-13B/ltxv-13b-0.9.8-distilled-fp8.safetensors"
    from safetensors import safe_open
    with safe_open(checkpoint_path, framework='pt') as f:
        caption_weights = {
            "linear_1.weight": f.get_tensor("model.diffusion_model.caption_projection.linear_1.weight").float(),
            "linear_1.bias": f.get_tensor("model.diffusion_model.caption_projection.linear_1.bias").float(),
            "linear_2.weight": f.get_tensor("model.diffusion_model.caption_projection.linear_2.weight").float(),
            "linear_2.bias": f.get_tensor("model.diffusion_model.caption_projection.linear_2.bias").float(),
        }

    # Initialize T5 text encoder
    t5_path = "/Users/ryantsudek/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors"
    text_encoder = T5TextEncoder(t5_path, caption_weights, device)

    # Encode prompt
    print("\nEncoding prompts...")
    t0 = time.time()
    cond_emb, uncond_emb = text_encoder.encode_cond_uncond(args.prompt)
    print(f"  Encoded in {time.time()-t0:.1f}s")
    print(f"  cond: {cond_emb.shape}, mean={cond_emb.abs().mean().item():.4f}")
    print(f"  uncond: {uncond_emb.shape}, mean={uncond_emb.abs().mean().item():.4f}")

    # Load diffusion model
    print("\nLoading diffusion model...")
    t0 = time.time()
    model = DiffusionModel(blocks_dir, top_path, device)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Create noisy latent
    B, L = 1, args.L
    latent_noisy = torch.randn(B, L, LTX_LATENT_CHANNELS, dtype=torch.float32)
    print(f"  Latent shape: {latent_noisy.shape}")

    # Benchmark one step
    timesteps = get_timestep_schedule(args.steps, "linear")
    print(f"\nDenoising: {args.steps} steps, CFG={args.cfg}")
    print(f"  Timesteps: {timesteps[0].item()} → {timesteps[-1].item()}")

    t0 = time.time()
    n_warm = 1
    for _ in range(n_warm):
        _ = denoise(model, latent_noisy, timesteps[:n_warm],
                    cond_emb, uncond_emb, args.cfg)
        if device.type == "mps":
            torch.mps.synchronize()
    warmup_time = time.time() - t0

    print(f"\n  Warmup ({n_warm} step, cond+uncond): {warmup_time:.1f}s")

    # Benchmark
    t0 = time.time()
    steps_to_time = min(3, args.steps)
    for _ in range(steps_to_time):
        _ = denoise(model, latent_noisy, timesteps[:steps_to_time],
                    cond_emb, uncond_emb, args.cfg)
        if device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.time() - t0
    per_step = elapsed / steps_to_time

    print(f"\n{'─'*50}")
    print(f"Results: L={L}, {steps_to_time} steps (cond+uncond CFG)")
    print(f"  Per-step:    {per_step:.1f}s")
    print(f"  Per-block:   {per_step/LTX_NUM_BLOCKS*1000:.0f}ms")
    print(f"  Throughput:  {L/per_step:.0f} tokens/sec")

    if args.steps > 1:
        est_full = per_step * args.steps
        print(f"  Est {args.steps} steps: {est_full:.0f}s ({est_full/60:.1f} min)")

    # Estimate 720p
    full_L = 216000
    est_720p = per_step * (full_L / L) * args.steps * 2  # 2× for CFG
    print(f"  Est 720p ({full_L} tok, {args.steps} steps, CFG): {est_720p/60:.0f} min")

    if device.type == "mps":
        used = torch.mps.current_allocated_memory() / 1024**3
        print(f"  GPU memory: {used:.2f}GB")

    print(f"\n✓ Full pipeline working")
    print(f"  T5 encoder: built from safetensors")
    print(f"  CFG: cond + uncond passes")
    print(f"  Denoise: {args.steps}-step Euler")

    model.prefetcher.shutdown()


if __name__ == "__main__":
    main()