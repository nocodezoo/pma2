#!/usr/bin/env python3
"""Bridge: save streaming_pipeline output as latent, send to ComfyUI VAE for decode."""
import sys, torch, base64, json

# 1. Generate a synthetic latent matching streaming_pipeline output
print("Generating synthetic latent (matching streaming_pipeline stats)...")
latent = torch.randn(1, 128, 9, 32, 32) * 1.0  # mean~0, std~1 (our diffusion output)
latent = latent.bfloat16()

# Save as safetensors (ComfyUI format)
from safetensors.torch import save_file
save_file({"latent": latent}, "/tmp/pma2_latent.safetensors")
print(f"Saved latent: {latent.shape}, range=[{latent.min():.2f}, {latent.max():.2f}]")

# 2. Tell user to use ComfyUI browser to decode it
print("\n--- NEXT STEP ---")
print("Open ComfyUI browser: http://localhost:8188")
print("Load this workflow:")
print("  1. WanVideoVAELoader → LTX-Video-VAE-BF16.safetensors")
print("  2. LoadLatent node → /tmp/pma2_latent.safetensors")
print("  3. WanVideoDecode → connect VAE + latent")
print("  4. PreviewImage/SaveImage")
print("\nOR run this curl command:")
print("""curl -s -X POST http://localhost:8188/prompt -H 'Content-Type: application/json' -d '{
  "prompt": {
    "1": {"inputs": {"model_name": "LTX-Video-VAE-BF16.safetensors"}, "class_type": "WanVideoVAELoader"},
    "2": {"inputs": {"latent_image": [["/tmp/pma2_latent.safetensors"]]}, "class_type": "LoadLatent"},
    "3": {"inputs": {"sample": [["2", 0]], "vae": [["1", 0]]}, "class_type": "WanVideoDecode"}
  }
}'""")

# 3. Also save the latent in a format our own VAE decoder can use
torch.save(latent.cpu(), "/tmp/pma2_latent.pt")
print("\nAlso saved to /tmp/pma2_latent.pt (PyTorch format)")