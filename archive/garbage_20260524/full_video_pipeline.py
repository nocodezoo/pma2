#!/usr/bin/env python3
"""Full pipeline: streaming diffusion + ComfyUI VAE decode for a real video."""
import sys, torch, time
from pathlib import Path

# Add ComfyUI paths
sys.path.insert(0, '/Users/ryantsudek/ComfyUI')
sys.path.insert(0, '/Users/ryantsudek/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper')

from streaming_pipeline import StreamingPipeline
from safetensors import safe_open

# Paths
BLOCKS_DIR = Path('/Users/ryantsudek/Projects/pma2-ltx-video/checkpoints/13B_blocks')
TOP_LEVEL   = Path('/Users/ryantsudek/Projects/pma2-ltx-video/checkpoints/13B_blocks/top_level.npz')
CKPT        = Path('/Users/ryantsudek/Projects/pma2-ltx-video/checkpoints/LTX-Video-13B/ltxv-13b-0.9.8-distilled-fp8.safetensors')

device = torch.device('cpu')

print("Loading streaming diffusion model...")
t0 = time.time()
pipe = StreamingPipeline(blocks_dir=BLOCKS_DIR, top_level_path=TOP_LEVEL, device=device)
print(f"Loaded in {time.time()-t0:.1f}s")

# T=9, H=32, W=32 latent shape
# After diffusion: (1, 1024, 128) → proj_out(Linear(4096→128)) → (1, 1024, 128)
# This reshapes to (1, 128, 9, 32, 32) for the VAE

# Build latent tensor for a short "video" at low res
# Simulate a prompt embedding (clip skipped for test - use zeros)
B, L, H = 1, 1024, 128
T_lat, H_lat, W_lat = 9, 32, 32

print(f"\nDiffusion latent: {B}×{T_lat}×{H_lat}×{W_lat} input → {L} tokens")

# Fake prompt embeddings for context
from streaming_pipeline import SinusoidalEmbedding
sin_emb = SinusoidalEmbedding(256, device)
text_emb = torch.zeros(B, L, 4096, device=device)  # T5 text emb (all zeros for test)

timestep = torch.tensor([0.5], device=device)

# Run streaming diffusion
print("Running diffusion (this will take ~30s per block)...")
t0 = time.time()

# Run a few blocks to get a non-trivial output
# (In real use, you'd run all 48 blocks + CFG)
output_latent = None
for block_idx in range(48):
    # Simulate: each block processes the latent
    # First block needs initialization
    if output_latent is None:
        output_latent = torch.randn(B, L, H, device=device) * 0.1

    output_latent = pipe.forward_block(output_latent, text_emb, timestep, block_idx)

    if block_idx % 12 == 0:
        print(f"  Block {block_idx:2d}/47: mean={output_latent.mean():.4f}, std={output_latent.std():.4f}")

print(f"\nDiffusion done in {time.time()-t0:.1f}s")
print(f"Output: {output_latent.shape}")

# Reshape from (B, L, H) = (1, 1024, 128) → (B, 128, T, H, W)
# L = T * H * W = 9 * 32 * 32 = 9216... but we have 1024
# The latent tokens are spatial: each token covers a patch
# At T=9, H=32, W=32 with patch_size=1, we need 9*32*32 = 9216 tokens
# We have 1024 tokens... this means we need different dimensions

# With 1024 tokens and 9*32*32=9216, the compression is 9×
# So actual video dims: T=9, H=32, W=32 but only 1024/288=3.5 patches per frame?
# Actually: 1024 / 9 = 113 tokens per frame
# 113 ≈ 10.6 × 10.6 spatial patches... so H_lat=10, W_lat=10?

# Let's verify: 9*10*10 = 900 tokens... close but not 1024
# Maybe: 9*11*11 = 1089... so maybe H=11, W=11

# Actually: streaming_pipeline uses L=1024 for profiling at 32x32
# Real generation would have different L based on resolution

# For this test, just reshape the 1024 tokens as if they were 32x32 spatial
# (This is what the profiling in streaming_pipeline does)
video_latent = output_latent.view(B, T_lat, H_lat, W_lat, H).permute(0, 4, 1, 2, 3)
# Wait, that's wrong. Let me check: streaming_pipeline uses (B, L, H) where L=tokens
# The tokens are (T, H, W) flattened: L = T * H_patch * W_patch
# For T=9, 32x32 patches: 9*32*32 = 9216, but we have 1024
# 
# Actually, streaming_pipeline.py profiles with:
#   - latent_shape: T=9, H_patch=32, W_patch=32, L=9*32*32=9216
# But then it only runs with L=1024 to test the block mechanism
#
# For real generation: L = T * H_patch * W_patch
# H_patch and W_patch depend on resolution: at 720p, H_patch=90, W_patch=120
# 9*90*120 = 97200 tokens... way more than 1024
#
# Let me just use 1024 = 32*32 (single frame) for testing
video_latent = output_latent.view(B, 1, 32, 32, H).permute(0, 4, 1, 2, 3)
video_latent = video_latent[:, :, :9, :, :]  # Take 9 frames
print(f"\nVideo latent for VAE: {video_latent.shape}")

# Now decode with ComfyUI VAE
print("\nLoading ComfyUI VAE...")
from wanvideo.wan_video_vae import WanVideoVAE38
from comfy.model_management import get_torch_device
offload_device = get_torch_device()

vae_sd = {}
with safe_open('/Users/ryantsudek/ComfyUI/models/vae/LTX-Video-VAE-BF16.safetensors', framework='pt') as f:
    for k in f.keys():
        vae_sd[k] = f.get_tensor(k)

has_model_prefix = any(k.startswith('model.') for k in vae_sd.keys())
if not has_model_prefix:
    vae_sd = {f'model.{k}': v for k, v in vae_sd.items()}

vae = WanVideoVAE38(dtype=torch.bfloat16)
vae.load_state_dict(vae_sd, strict=False)
vae.eval()
vae.to(device=offload_device, dtype=torch.bfloat16)
print(f"VAE loaded: z_dim={vae.z_dim}")

# Decode
print(f"\nDecoding...")
video_latent_bf16 = video_latent.to(torch.bfloat16)
print(f"  Input: {video_latent_bf16.shape}, mean={video_latent_bf16.mean():.3f}, std={video_latent_bf16.std():.3f}")

with torch.no_grad():
    video = vae.decode(video_latent_bf16)

print(f"  Output: {video.shape}")
print(f"  Range: [{video.min().item():.4f}, {video.max().item():.4f}]")
print(f"  NaN: {torch.isnan(video).sum().item()} / {video.numel()}")

if not torch.isnan(video).any():
    torch.save(video.cpu(), '/tmp/pma2_full_video.pt')
    print(f"\nSaved video to /tmp/pma2_full_video.pt")
    print(f"Shape: {video.shape} = (B, 3, T, H, W)")
else:
    print("\nWARNING: NaN in output - VAE not working properly")
    print("Will need to debug the ComfyUI VAE wrapper")