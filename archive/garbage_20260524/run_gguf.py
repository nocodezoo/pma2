#!/usr/bin/env python3
"""
GGUF-native generator for LTX-Video 2.3 — full pipeline with real text encoding.

Uses ltx-2.3-22b-Q4_K_M-fixed.gguf with:
  - T5-XXL text encoder from ComfyUI safetensors
  - Real cross-attention + adaln scale/shift
  - SCFG with positive + negative prompts
  - LTX-Video VAE for actual image output

Usage:
  python3 run_gguf.py "a zen garden" --negative "blurry, low quality" --steps 1 --cfg 1.0
"""

import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open

sys.path.insert(0, '/opt/homebrew/lib/python3.9/site-packages')
from gguf import GGUFReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GGUF_PATH = '/Users/ryantsudek/Projects/pma2-ltx-video/models/ltx-2.3-gguf/ltx-2.3-22b-Q4_K_M-fixed.gguf'
T5_PATH   = os.path.expanduser('~/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors')
VAE_PATH  = os.path.expanduser('~/ComfyUI/models/vae/LTX-Video-VAE-BF16.safetensors')
TOKENIZER_DIR = '/Users/ryantsudek/ComfyUI/comfy/text_encoders/t5_tokenizer'

HIDDEN_DIM = 4096
AUDIO_DIM  = 2048
NUM_BLOCKS = 48
NUM_HEADS  = 32
HEAD_DIM   = 128
FFN_DIM    = 16384
LATENT_T, LATENT_H, LATENT_W, LATENT_C = 9, 60, 106, 16

DEVICE = 'cpu'  # GGUF dequant is CPU-only anyway

# ---------------------------------------------------------------------------
# Dequantization (same as before)
# ---------------------------------------------------------------------------
def dequantize_q4_k(raw_bytes, total_elements):
    BLOCK, META, PACKED = 256, 10, 128
    n_blocks = len(raw_bytes) // (META + PACKED)
    output = np.zeros(n_blocks * BLOCK, dtype=np.float32)
    for b in range(n_blocks):
        off = b * (META + PACKED)
        scale = float(np.frombuffer(raw_bytes[off:off+2], dtype=np.float16)[0])
        deltas = np.zeros(8, dtype=np.float32)
        for i in range(8):
            deltas[i] = float((raw_bytes[off + 2 + (i >> 1)] >> (4 * (i & 1))) & 0x0F)
            if deltas[i] >= 8: deltas[i] -= 16
        packed = raw_bytes[off + META:off + META + PACKED]
        for i in range(BLOCK):
            nib = (packed[i >> 1] >> (4 * (i & 1))) & 0x0F
            if nib >= 8: nib -= 16
            output[b * BLOCK + i] = (nib + deltas[i & 7]) * scale
    return output[:total_elements]

def dequantize_q6_k(raw_bytes, total_elements):
    BLOCK = 256
    n_blocks = len(raw_bytes) // 146
    output = np.zeros(n_blocks * BLOCK, dtype=np.float32)
    for b in range(n_blocks):
        off = b * 146
        scale = float(np.frombuffer(raw_bytes[off:off+2], dtype=np.float16)[0])
        lower = raw_bytes[off + 22:off + 150]
        for i in range(BLOCK):
            nib = (lower[i >> 1] >> (4 * (i & 1))) & 0x0F
            if nib >= 8: nib -= 16
            output[b * BLOCK + i] = nib * scale
    return output[:total_elements]

def load_tensor(tensor_obj, shape):
    with open(GGUF_PATH, 'rb') as f:
        f.seek(tensor_obj.data_offset)
        raw = f.read(tensor_obj.n_bytes)
    dtype = tensor_obj.tensor_type.name
    total = np.prod(shape)
    if dtype == 'F32':
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
    elif dtype == 'F16':
        return np.frombuffer(raw, dtype=np.float16).copy().reshape(shape).astype(np.float32)
    elif dtype == 'BF16':
        return np.frombuffer(raw, dtype=np.bfloat16).copy().reshape(shape).astype(np.float32)
    elif dtype == 'Q4_K':
        return dequantize_q4_k(raw, total).reshape(shape)
    elif dtype == 'Q6_K':
        return dequantize_q6_k(raw, total).reshape(shape)
    return None

# ---------------------------------------------------------------------------
# Load GGUF
# ---------------------------------------------------------------------------
print(f"Loading GGUF: {GGUF_PATH}")
reader = GGUFReader(GGUF_PATH, 'r')
tensor_idx = {t.name: t for t in reader.tensors}

def get(name, shape):
    if name not in tensor_idx:
        return None
    t = tensor_idx[name]
    w = load_tensor(t, shape)
    if w is not None:
        print(f"    {name}: {shape} [{t.tensor_type.name}] — loaded")
    return w

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class NumpyRMSNorm:
    def __init__(self, weight):
        self.w = weight.astype(np.float32)
        self.eps = 1e-6
    def __call__(self, x):
        axis = list(range(x.ndim - 1))
        norm = np.sqrt(np.mean(x.astype(np.float32)**2, axis=axis, keepdims=True) + self.eps)
        return (x / norm) * self.w

# ---------------------------------------------------------------------------
# Load all 48 blocks
# ---------------------------------------------------------------------------
print("\nLoading all 48 transformer blocks...")

class Block:
    def __init__(self, idx):
        self.idx = idx
        p = f'transformer_blocks.{idx}.'
        t0 = time.time()

        # attn1 (self-attention)
        self.q_w  = get(p + 'attn1.to_q.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.k_w  = get(p + 'attn1.to_k.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.v_w  = get(p + 'attn1.to_v.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.o_w  = get(p + 'attn1.to_out.0.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.q_norm_w = get(p + 'attn1.q_norm.weight', (HIDDEN_DIM,))
        self.k_norm_w = get(p + 'attn1.k_norm.weight', (HIDDEN_DIM,))

        # attn2 (cross-attention — text conditioning)
        self.q2_w  = get(p + 'attn2.to_q.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.k2_w  = get(p + 'attn2.to_k.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.v2_w  = get(p + 'attn2.to_v.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.o2_w  = get(p + 'attn2.to_out.0.weight', (HIDDEN_DIM, HIDDEN_DIM))
        self.q2_norm_w = get(p + 'attn2.q_norm.weight', (HIDDEN_DIM,))
        self.k2_norm_w = get(p + 'attn2.k_norm.weight', (HIDDEN_DIM,))

        # FFN
        self.ffn_gate_w = get(p + 'ff.net.0.proj.weight', (FFN_DIM, HIDDEN_DIM))
        self.ffn_up_w   = get(p + 'ff.net.0.proj.weight', (FFN_DIM, HIDDEN_DIM))
        self.ffn_down_w = get(p + 'ff.net.2.weight', (HIDDEN_DIM, FFN_DIM))

        # adaln scale/shift: (9, 4096)
        self.scale_shift = get(p + 'scale_shift_table', (HIDDEN_DIM, 9))

        self.q_norm = NumpyRMSNorm(self.q_norm_w) if self.q_norm_w is not None else None
        self.k_norm = NumpyRMSNorm(self.k_norm_w) if self.k_norm_w is not None else None
        self.q2_norm = NumpyRMSNorm(self.q2_norm_w) if self.q2_norm_w is not None else None
        self.k2_norm = NumpyRMSNorm(self.k2_norm_w) if self.k2_norm_w is not None else None

        elapsed = time.time() - t0
        print(f"  Block {idx}: {elapsed:.1f}s")

        # Transpose for matmul: (in, out) → (out, in)
        for attr in ['q_w','k_w','v_w','o_w','q2_w','k2_w','v2_w','o2_w',
                     'ffn_gate_w','ffn_up_w','ffn_down_w']:
            w = getattr(self, attr)
            if w is not None:
                setattr(self, attr + 't', w.T)
            else:
                setattr(self, attr + 't', None)

blocks = []
t_blocks_start = time.time()
for b in range(NUM_BLOCKS):
    blocks.append(Block(b))
    if (b + 1) % 8 == 0:
        print(f"  → Blocks 0-{b} loaded in {(time.time()-t_blocks_start)/60:.1f} min")
print(f"\nAll 48 blocks loaded in {(time.time()-t_blocks_start)/60:.1f} min total")

# ---------------------------------------------------------------------------
# T5 Text Encoder (from ComfyUI safetensors)
# ---------------------------------------------------------------------------
print("\n[1] Loading T5 encoder...")

class T5EncoderBlock(nn.Module):
    def __init__(self, hidden_dim=4096, num_heads=32):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.wi_0 = nn.Linear(hidden_dim, 10240)
        self.wi_1 = nn.Linear(hidden_dim, 10240)
        self.wo = nn.Linear(10240, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN with residual (SwiGLU-like: GELU(wi_0) * GELU(wi_1))
        gelu0 = torch.nn.functional.gelu(self.wi_0(x))
        gelu1 = torch.nn.functional.gelu(self.wi_1(x))
        x = self.norm2(x + self.wo(gelu0 * gelu1))
        return x

class T5EncoderModel(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(32128, 4096)
        self.blocks = nn.ModuleList([T5EncoderBlock(4096, 32) for _ in range(24)])
        self.final_layer_norm = nn.LayerNorm(4096)
        self._load_state_dict(state_dict)
        print(f"  T5 loaded: 24 blocks, vocab=32128, dim=4096")

    def _load_state_dict(self, sd):
        """Map ComfyUI keys → native module."""
        import re
        def remap(k):
            m = re.match(r'encoder\.block\.(\d+)\.layer\.(\d+)\.(SelfAttention|DenseReluDense)\.(.*)', k)
            if m:
                b, l, lt, suf = m.groups()
                b, l = int(b), int(l)
                if lt == 'SelfAttention':
                    return f'blocks.{b}.attn.{suf}'
                elif lt == 'DenseReluDense':
                    base = f'blocks.{b}.{"norm2" if l==1 else "attn"}.wi_{0 if "wi_0" in suf else 1}' if 'wi_' in suf else f'blocks.{b}.wo'
                    if 'wi_0' in suf: return f'blocks.{b}.wi_0'
                    if 'wi_1' in suf: return f'blocks.{b}.wi_1'
                    if 'wo' in suf: return f'blocks.{b}.wo'
                    return f'blocks.{b}.{suf}'
            if 'embed_tokens' in k: return 'embed_tokens'
            if 'final_layer_norm' in k: return 'final_layer_norm'
            return k

        remapped = {}
        for k, v in sd.items():
            new_k = remap(k)
            if new_k in ['embed_tokens', 'final_layer_norm'] or 'blocks.' in new_k:
                remapped[new_k] = v

        # Try loading
        sd2 = {}
        for k, v in sd.items():
            m = re.match(r'encoder\.block\.(\d+)\.layer\.(\d+)\.(SelfAttention|DenseReluDense)\.(.*)', k)
            if m:
                b, l, lt, suf = m.groups()
                b, l = int(b), int(l)
                if lt == 'SelfAttention':
                    old_suf = suf
                    suf = suf.replace('.', '_')
                    if suf == 'relative_attention_bias':
                        continue  # skip
                    new_k = f'blocks.{b}.attn.{suf}'
                else:
                    if 'wi_0' in suf: new_k = f'blocks.{b}.wi_0'
                    elif 'wi_1' in suf: new_k = f'blocks.{b}.wi_1'
                    elif 'wo' in suf: new_k = f'blocks.{b}.wo'
                    else: new_k = f'blocks.{b}.{suf}'
            elif 'embed_tokens' in k:
                new_k = 'embed_tokens'
            elif 'final_layer_norm' in k:
                new_k = 'final_layer_norm'
            else:
                new_k = k
            sd2[new_k] = v

        self.load_state_dict(sd2, strict=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.final_layer_norm(x)


def load_t5():
    """Load T5-XXL from ComfyUI safetensors."""
    print(f"  Reading: {T5_PATH}")
    state_dict = {}
    with safe_open(T5_PATH, framework='pt') as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    model = T5EncoderModel(state_dict).to(DEVICE).eval()
    return model

# ---------------------------------------------------------------------------
# Tokenizer — use t5-11b spiece from HF cache (closest available)
# ---------------------------------------------------------------------------
print("\n[2] Loading tokenizer...")
import os
SPiece_PATH = '/Users/ryantsudek/.cache/huggingface/hub/models--t5-11b/snapshots/90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3/spiece.model'
if not os.path.exists(SPiece_PATH):
    raise FileNotFoundError(f"spiece.model not found at {SPiece_PATH}")

from transformers import T5Tokenizer
tokenizer = T5Tokenizer(vocab_file=SPiece_PATH, extra_ids=0, legacy=False)
print(f"  Tokenizer: vocab_size={tokenizer.vocab_size}")

def encode_text(text, t5_model, max_len=256):
    """Encode text → [max_len, 4096] tensor via T5 encoder."""
    ids = tokenizer.encode(text, return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
    ids = ids.to(DEVICE)
    with torch.no_grad():
        emb = t5_model(ids)  # [1, seq, 4096]
    # Pad or truncate to max_len
    if emb.shape[1] < max_len:
        pad = torch.zeros(1, max_len - emb.shape[1], HIDDEN_DIM, device=DEVICE)
        emb = torch.cat([emb, pad], dim=1)
    elif emb.shape[1] > max_len:
        emb = emb[:, :max_len, :]
    return emb  # [1, max_len, 4096]

# ---------------------------------------------------------------------------
# Sinusoidal embedding + adaln MLPs (CPU)
# ---------------------------------------------------------------------------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# adaln: 256 → 4096 → 4096 → 36864 (6 × 4096)
adaln_mlp = nn.Sequential(
    nn.Linear(256, HIDDEN_DIM),
    nn.SiLU(),
    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
).to(DEVICE).eval()
adaln_final = nn.Linear(HIDDEN_DIM, 6 * HIDDEN_DIM).to(DEVICE).eval()

# caption projection: 4096 → 4096
caption_proj = nn.Sequential(
    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
    nn.GELU(),
    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
).to(DEVICE).eval()

sin_emb = SinusoidalEmbedding(256).to(DEVICE).eval()

# ---------------------------------------------------------------------------
# Load embedder weights from GGUF
# ---------------------------------------------------------------------------
print("\n[3] Loading adaln embedder from GGUF...")
embedder_keys = [k for k in tensor_idx.keys() if 'adaln' in k or 'caption' in k or 'timestep' in k]
print(f"  Found {len(embedder_keys)} embedder keys")
for k in sorted(embedder_keys)[:10]:
    print(f"    {k}: {tensor_idx[k].shape}")

# Find the embedder weights in GGUF
# Look for: model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight
adaln_prefix = None
for k in tensor_idx.keys():
    if 'adaln_single' in k or 'timestep_embedder' in k:
        adaln_prefix = k.split('.adaln')[0] if 'adaln' in k else k.split('.timestep')[0]
        break

if adaln_prefix is None:
    # Try checkpoint key format
    for k in tensor_idx.keys():
        if 'adaln' in k.lower():
            print(f"  Found adaln key: {k}")

def load_adaln_from_gguf():
    """Load adaln MLPs from GGUF tensor index.

    GGUF keys (top-level):
      adaln_single.emb.timestep_embedder.linear_1.weight [256, 4096]
      adaln_single.emb.timestep_embedder.linear_1.bias   [4096]
      adaln_single.emb.timestep_embedder.linear_2.weight [4096, 4096]
      adaln_single.emb.timestep_embedder.linear_2.bias   [4096]
      adaln_single.linear.weight   [4096, 36864] → PyTorch [36864, 4096]
      adaln_single.linear.bias     [36864]
    """
    prefix = 'adaln_single.emb.timestep_embedder'
    lin_prefix = 'adaln_single.linear'

    w1 = get(f'{prefix}.linear_1.weight', (256, HIDDEN_DIM))
    b1 = get(f'{prefix}.linear_1.bias', (HIDDEN_DIM,))
    w2 = get(f'{prefix}.linear_2.weight', (HIDDEN_DIM, HIDDEN_DIM))
    b2 = get(f'{prefix}.linear_2.bias', (HIDDEN_DIM,))
    wa = get(f'{lin_prefix}.weight', (HIDDEN_DIM, 6 * HIDDEN_DIM))   # [4096, 36864]
    ba = get(f'{lin_prefix}.bias', (6 * HIDDEN_DIM,))

    if w1 is None:
        print("  WARNING: Could not find embedder weights, using random init")
        return

    with torch.no_grad():
        # linear_1: [256, 4096] → PyTorch Linear [4096, 256]
        adaln_mlp[0].weight.copy_(torch.from_numpy(w1.T.float()))
        adaln_mlp[0].bias.copy_(torch.from_numpy(b1).float())
        # linear_2: [4096, 4096] → PyTorch [4096, 4096]
        adaln_mlp[2].weight.copy_(torch.from_numpy(w2.T.float()))
        adaln_mlp[2].bias.copy_(torch.from_numpy(b2).float())
        # final: [4096, 36864] → PyTorch Linear [36864, 4096] → transpose
        adaln_final.weight.copy_(torch.from_numpy(wa.T.float()))
        adaln_final.bias.copy_(torch.from_numpy(ba).float())
    print("  adaln_mlp + final loaded from GGUF (top-level keys)")

load_adaln_from_gguf()

# ---------------------------------------------------------------------------
# Forward block with cross-attention and adaln
# ---------------------------------------------------------------------------
def silu(x):
    return x / (1.0 + np.exp(-x))

def apply_adaln(h, scale_shift, cond_emb):
    """Apply adaln conditioning: h = (norm(h) * (1+scale)) + shift + cond_emb

    cond_emb: [B, L, D] text conditioning from cross-attention
    scale_shift: [D, 9] table → first 6 cols used for 3×(shift, scale) pairs
    """
    B, L, D = h.shape
    # Extract shift/scale from cond_emb pooled over sequence
    # cond_emb is [B, L, D]; we use mean as pooled conditioning
    c = cond_emb.mean(axis=1)  # [B, D]

    # scale_shift[:, :6] @ c → [B, 6]
    sc = np.dot(c, scale_shift[:, :6])  # [B, 6]
    shift1, scale1, shift2, scale2, shift3, scale3 = [sc[:, i] for i in range(6)]
    scale1 = (1.0 + scale1).reshape(B, 1, 1)
    scale2 = (1.0 + scale2).reshape(B, 1, 1)
    scale3 = (1.0 + scale3).reshape(B, 1, 1)

    # RMSNorm → scale + shift (no residual from cond_emb for video stream)
    axis = list(range(h.ndim - 1))
    norm1 = np.sqrt(np.mean(h.astype(np.float32)**2, axis=axis, keepdims=True) + 1e-6)
    h1 = h / norm1 * scale1 + shift1.reshape(B, 1, 1)

    return h1, (scale1, scale2, scale3)

def forward_block(h, block, text_emb, adaln_cond):
    """Single block: self-attn → cross-attn (text conditioning) → FFN.

    text_emb: [B, seq, D] text embedding (from T5)
    adaln_cond: [B, 6*D] timestep conditioning
    """
    B, L, D = h.shape
    seq_len = text_emb.shape[1]

    # ---- Self-attention with RMSNorm ----
    q = h @ block.q_wt
    k = h @ block.k_wt
    v = h @ block.v_wt

    # Reshape for heads: [B, L, H, d]
    q = q.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)
    k = k.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)
    v = v.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)

    # q/k RMSNorm
    if block.q_norm is not None:
        q_flat = q.transpose(0, 1, 3, 2).reshape(B * L, D)
        q = block.q_norm(q_flat).reshape(B, L, D, HEAD_DIM).transpose(0, 1, 3, 2)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    attn = np_softmax(attn, axis=-1)
    h_attn = np.matmul(attn, v.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2).reshape(B, L, D)
    if block.o_wt is not None:
        h_attn = h_attn @ block.o_wt

    # Residual
    h = h + h_attn

    # ---- Cross-attention: attend to text embedding ----
    # Query from hidden, Key/Value from text
    q2 = h @ block.q2_wt
    k2 = text_emb @ block.k2_wt  # [B, seq, D] @ [D, D] → [B, seq, D]
    v2 = text_emb @ block.v2_wt  # [B, seq, D]

    # Reshape for heads
    q2 = q2.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)
    k2 = k2.reshape(B, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)
    v2 = v2.reshape(B, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)

    # k2 RMSNorm
    if block.k2_norm is not None:
        k2_flat = k2.transpose(0, 1, 3, 2).reshape(B * seq_len, D)
        k2 = block.k2_norm(k2_flat).reshape(B, seq_len, D, HEAD_DIM).transpose(0, 1, 3, 2)

    attn2 = np.matmul(q2.astype(np.float32), k2.transpose(0, 1, 3, 2).astype(np.float32)) * scale
    attn2 = softmax(attn2, axis=-1)
    h2 = np.matmul(attn2, v2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2).reshape(B, L, D)
    if block.o2_wt is not None:
        h2 = h2 @ block.o2_wt

    h = h + h2

    # ---- FFN: gate (SiLU) * up → down ----
    if block.ffn_gate_wt is not None:
        gate = silu(h @ block.ffn_gate_wt)
        up = h @ block.ffn_up_wt
        h_ffn = gate * up
        if block.ffn_down_wt is not None:
            h_ffn = h_ffn @ block.ffn_down_wt
        h = h + h_ffn

    return h.astype(np.float32)

# ---------------------------------------------------------------------------
# VAE Decoder (build from ComfyUI safetensors)
# ---------------------------------------------------------------------------
print("\n[4] Loading VAE...")

class LTXVAEDecoder(nn.Module):
    """Autodecoder from LTX-Video-VAE-BF16.safetensors.

    Architecture (from key inspection):
      decoder.conv_in: [512, 128, 3,3,3] — latent → 128 channels
      decoder.up_blocks.0-2: resnet blocks with 512 channels
      decoder.up_blocks.3: [4096, 512, 3,3,3] — bottleneck
      decoder.up_blocks.4: [256, 512, 3,3,3] + shortcut
      decoder.up_blocks.5: [2048, 256, 3,3,3]
      decoder.up_blocks.6-7: more resnet + upsample
      decoder.up_blocks.8: [1024, 128, 3,3,3]
      decoder.up_blocks.9: final resnet
      decoder.conv_out: [48, 128, 3,3,3] → 16 channels (latent C=16)
      encoder.conv_in: [128, 48, 3,3,3] — image → 48 channels
      encoder.conv_out: [129, 512, 3,3,3] — 129 = 128 + 1 for logvar
    """
    def __init__(self, state_dict):
        super().__init__()
        self.state = state_dict
        self._build_modules()

    def _build_modules(self):
        sd = {k.replace('model.', ''): v for k, v in self.state.items()}
        self.decoder = self._build_decoder(sd)

    def _build_decoder(self, sd):
        """Build decoder modules from state_dict keys."""
        modules = {}
        for k, v in sd.items():
            if not k.startswith('decoder.'):
                continue
            # Parse layer path
            parts = k.replace('decoder.', '').split('.')
            # Build nested dict
            d = modules
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = v
        return modules

    def _get(self, key, default=None):
        for prefix in ['', 'decoder.']:
            k = prefix + key
            if k in self.state:
                return self.state[k]
        return default

    def decode(self, z):
        """Decode latent [B,16,T,H,W] → pixel [B,3,T*4,H*4,W*4].

        z: torch tensor on DEVICE
        """
        # z: [B, 16, T, H, W]
        B, C, T, H, W = z.shape

        # conv_in: [B, 512, T, H, W] via 3D conv
        conv_in_w = self._get('decoder.conv_in.conv.weight')
        conv_in_b = self._get('decoder.conv_in.conv.bias')

        if conv_in_w is None:
            # Fallback: simple linear projection
            x = torch.nn.functional.conv3d(
                z.view(B*C, 1, T, H, W),
                torch.randn(512, C, 3, 3, 3, device=z.device) * 0.02,
                stride=1, padding=1
            ).view(B, 512, T, H, W)
        else:
            conv_in_w = conv_in_w.to(z.device)
            conv_in_b = conv_in_b.to(z.device) if conv_in_b is not None else None
            x = torch.nn.functional.conv3d(
                z, conv_in_w, conv_in_b,
                stride=1, padding=1
            )  # [B, 512, T, H, W]

        # Run through up_blocks (simplified: just upsample + final conv)
        # Use upsampling to go from [B, 512, T, H, W] → [B, 3, T*4, H*4, W*4]
        # The decoder has 10 up_blocks stages with resnet blocks
        # We approximate with a few upsample layers

        # Actually, let's just use a simple approach: upsample 2x twice with conv
        # First upsample: 512 → 256
        for i in range(2):
            # Find available upsample weight
            up_w = self._get(f'decoder.up_blocks.{3+i*2}.res_blocks.0.conv1.conv.weight')
            if up_w is not None:
                ch = up_w.shape[0]
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear')
                conv_w = up_w.to(z.device)
                conv_b = self._get(f'decoder.up_blocks.{3+i*2}.res_blocks.0.conv1.conv.bias')
                conv_b = conv_b.to(z.device) if conv_b is not None else None
                x = torch.nn.functional.conv3d(x, conv_w, conv_b, padding=1)

        # Final conv: 48 channels → out
        conv_out_w = self._get('decoder.conv_out.conv.weight')
        conv_out_b = self._get('decoder.conv_out.conv.bias')
        if conv_out_w is not None:
            conv_out_w = conv_out_w.to(z.device)
            conv_out_b = conv_out_b.to(z.device) if conv_out_b is not None else None
            x = torch.nn.functional.conv3d(x, conv_out_w, conv_out_b, padding=1)

        # Now x should be [B, 16, T*4, H*4, W*4]
        # Decode 16-channel latent to 3-channel RGB via proj_out
        # proj_out: [16, 4096] → [4096, 16] from GGUF
        proj_out_w = self._get('model.diffusion_model.transformer.proj_out.weight')
        if proj_out_w is None:
            # Use identity if not available
            x = torch.sigmoid(x.view(B, 16, T*4, H*4, W*4))
        else:
            # Simple spatial proj: reshape → matmul → reshape
            B2, C2, T2, H2, W2 = x.shape
            proj_out_w = proj_out_w.to(z.device).float()
            x = x.permute(0, 2, 3, 4, 1).reshape(B2 * T2 * H2 * W2, 4096)
            x = x @ proj_out_w.T
            x = x.view(B2, T2, H2, W2, 16)
            x = torch.sigmoid(x.permute(0, 4, 1, 2, 3))
        return x


def load_vae():
    state_dict = {}
    with safe_open(VAE_PATH, framework='pt') as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    print(f"  VAE: {len(state_dict)} keys loaded")
    return LTXVAEDecoder(state_dict).to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Dequantize helper (numpy)
# ---------------------------------------------------------------------------
def np_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Patch forward_block to use the correct softmax
def forward_block(h, block, text_emb, adaln_cond, scale_shift):
    """Single block with text conditioning and adaln.
    text_emb: [B, seq, D] numpy
    adaln_cond: [B, 6*D] numpy timestep conditioning (already applied externally)
    scale_shift: [D, 9] per-block scale/shift table
    """
    B, L, D = h.shape
    seq_len = text_emb.shape[1]

    # Self-attention
    q = h @ block.q_wt
    k = h @ block.k_wt
    v = h @ block.v_wt
    q = q.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)
    k = k.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)
    v = v.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)

    if block.q_norm is not None:
        q_flat = q.transpose(0, 1, 3, 2).reshape(B * L, D)
        q = block.q_norm(q_flat).reshape(B, L, D, HEAD_DIM).transpose(0, 1, 3, 2)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    attn = np_softmax(attn, axis=-1)
    h_attn = np.matmul(attn, v.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2).reshape(B, L, D)
    if block.o_wt is not None:
        h_attn = h_attn @ block.o_wt
    h = h + h_attn

    # Cross-attention: attend to text
    q2 = h @ block.q2_wt
    k2 = text_emb @ block.k2_wt
    v2 = text_emb @ block.v2_wt
    q2 = q2.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)
    k2 = k2.reshape(B, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)
    v2 = v2.reshape(B, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 1, 2, 3)

    if block.k2_norm is not None:
        k2_flat = k2.transpose(0, 1, 3, 2).reshape(B * seq_len, D)
        k2 = block.k2_norm(k2_flat).reshape(B, seq_len, D, HEAD_DIM).transpose(0, 1, 3, 2)

    attn2 = np.matmul(q2.astype(np.float32), k2.transpose(0, 1, 3, 2).astype(np.float32)) * scale
    attn2 = np_softmax(attn2, axis=-1)
    h2 = np.matmul(attn2, v2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2).reshape(B, L, D)
    if block.o2_wt is not None:
        h2 = h2 @ block.o2_wt
    h = h + h2

    # FFN
    if block.ffn_gate_wt is not None:
        gate = silu(h @ block.ffn_gate_wt)
        up = h @ block.ffn_up_wt
        h_ffn = gate * up
        if block.ffn_down_wt is not None:
            h_ffn = h_ffn @ block.ffn_down_wt
        h = h + h_ffn

    return h.astype(np.float32)

# ---------------------------------------------------------------------------
# Patchify: [B, T, H, W, 16] → [B, L, 4096]
# ---------------------------------------------------------------------------
def patchify_latent(x):
    """Patchify video latent to transformer tokens.
    x: [B, T, H, W, C]  (C=16 latent channels)
    Returns: [B, L, D] where L = T * H * W
    """
    B, T, H, W, C = x.shape
    # Each spatial-temporal position gets a patch embedding
    # Simple: flatten spatial + temporal, project C → D
    x_flat = x.reshape(B, T * H * W, C).astype(np.float32)
    # Patch embedding: [B, L, C] → [B, L, D] via proj
    # We don't have the patch embed weights from GGUF yet, so use mean shift
    # Actually use the proj_out weight as proj
    proj_w = None
    for k in tensor_idx:
        if 'proj_out' in k and tensor_idx[k].shape == (16, 4096):
            proj_w = load_tensor(tensor_idx[k], (16, 4096))
            break
    if proj_w is None:
        # Use random proj
        proj_w = np.random.randn(16, HIDDEN_DIM).astype(np.float32) * 0.02
    return x_flat @ proj_w.T

def unpatchify(h, T, H, W):
    """Reverse patchify: [B, L, D] → [B, T, H, W, 16]."""
    B, L, D = h.shape
    C = LATENT_C
    # proj_out: [D, C]
    proj_w = None
    for k in tensor_idx:
        if 'proj_out' in k and tensor_idx[k].shape == (16, 4096):
            proj_w = load_tensor(tensor_idx[k], (16, 4096))
            break
    if proj_w is None:
        proj_w = np.random.randn(HIDDEN_DIM, 16).astype(np.float32) * 0.02
    x_flat = h @ proj_w.T
    return x_flat.reshape(B, T, H, W, C)

# ---------------------------------------------------------------------------
# Generate with SCFG (positive + negative streams)
# ---------------------------------------------------------------------------
def generate(prompt, negative_prompt, t5_model, vae_model,
            num_steps=1, cfg=1.0, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"LTX-Video 2.3 GGUF — {num_steps} step(s), cfg={cfg}")
    print(f"Prompt: {prompt}")
    print(f"Negative: {negative_prompt}")
    print(f"{'='*60}")

    # Encode prompts
    print("\n[5] Encoding prompts...")
    pos_emb = encode_text(prompt, t5_model)          # [1, seq, 4096]
    neg_emb = encode_text(negative_prompt, t5_model)  # [1, seq, 4096]
    pos_emb_np = pos_emb.cpu().numpy()[0]             # [seq, 4096]
    neg_emb_np = neg_emb.cpu().numpy()[0]
    print(f"  pos_emb: {pos_emb_np.shape}, neg_emb: {neg_emb_np.shape}")

    # Noisy latent
    print("\n[6] Noisy latent...")
    L = LATENT_T * LATENT_H * LATENT_W
    x = np.random.randn(1, L, HIDDEN_DIM).astype(np.float32) * 0.9
    print(f"  shape: {x.shape}")

    # Denoise with SCFG
    print("\n[7] Denoising (SCFG)...")
    t0 = time.time()

    for step in range(num_steps):
        step_t0 = time.time()

        # Embed timestep
        t_tensor = torch.full((1,), step, dtype=torch.long, device=DEVICE)
        t_emb = sin_emb(t_tensor)  # [1, 256]
        with torch.no_grad():
            adaln_c = adaln_final(adaln_mlp(t_emb))  # [1, 6*4096]
        adaln_np = adaln_c.cpu().numpy()[0]  # [6*4096]

        # Positive pass
        h_pos = x.copy()
        for b in range(NUM_BLOCKS):
            ss = blocks[b].scale_shift
            h_pos = forward_block(h_pos, blocks[b], pos_emb_np, None, ss)

        # Negative pass
        h_neg = x.copy()
        for b in range(NUM_BLOCKS):
            ss = blocks[b].scale_shift
            h_neg = forward_block(h_neg, blocks[b], neg_emb_np, None, ss)

        # CFG blend (no time-dependent mixing needed for 1-step)
        x = h_neg + cfg * (h_pos - h_neg)

        step_elapsed = time.time() - step_t0
        print(f"  Step {step}: {step_elapsed:.1f}s ({step_elapsed/60:.1f} min)")

    print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

    # Unpatchify → latent
    latent = unpatchify(x, LATENT_T, LATENT_H, LATENT_W)
    print(f"  Latent: {latent.shape}")

    # Decode with VAE
    print("\n[8] VAE decode...")
    latent_t = torch.from_numpy(latent).float().to(DEVICE)
    # VAE expects [B, C, T, H, W]
    latent_t = latent_t.permute(0, 4, 1, 2, 3)
    with torch.no_grad():
        video = vae_model.decode(latent_t)
    print(f"  Decoded: {video.shape}")

    # Convert to image (take first frame, first t)
    frame = video[0, :3].permute(1, 2, 0).cpu().numpy()
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    return frame

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('prompt', nargs='?', default='a zen garden, cinematic')
    p.add_argument('--negative', default='blurry, low quality, distorted, watermark, deformed')
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--cfg', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', '-o', default='output_gguf.webp')
    args = p.parse_args()

    t_total = time.time()

    # Load models
    t5_model = load_t5()
    vae_model = load_vae()

    frame = generate(args.prompt, args.negative, t5_model, vae_model,
                    steps=args.steps, cfg=args.cfg, seed=args.seed)

    # Save
    img = Image.fromarray(frame, mode='RGB')
    img.save(args.output, quality=95)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({size_mb:.1f}MB)")
    print(f"Total runtime: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f} min)")