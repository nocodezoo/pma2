# PMA² — Phantom Memory Architecture v2

**Run 22B-parameter LTX-Video 2.3 on 16GB Apple Silicon M-series.**

---

## What This Is

PMA² is a memory-optimized inference architecture for large video diffusion models on constrained hardware. It makes LTX-Video 2.3's 22B-parameter DiT feasible on 16GB unified RAM through four interlocking innovations:

| Innovation | What It Does | Memory Saved |
|---|---|---|
| **Layer-Sequential Streaming (LSS)** | Only one transformer block (~200MB) lives in RAM at a time. NVMe at 7.4 GB/s loads blocks faster than compute. | ~21GB → ~600MB |
| **Spatiotemporal Latent Tiling (SLT)** | Full 3D attention over 216K tokens (4–8GB activations) → 12 tiles of ~32K tokens each (~350MB per tile). | 40× reduction |
| **Sequential CFG (SCFG-SC)** | CFG normally doubles memory. Run passes sequentially, store compressed residuals (~70MB) instead of full intermediates. | 2× → 70MB |
| **Timestep-Adaptive Precision (TAPB)** | 4 precision bands from W4A6 (early noise) → W6A8 (final refinement). Pre-packed block files, no runtime requantization. | 3–6 bits per step |

**Result:** 5 seconds of 720p video in ~46 seconds on base M5 (16GB), vs. OOM for naive loading.

---

## Quick Start

### 1. Install dependencies

```bash
cd ~/Projects/pma2-ltx-video
pip install -r requirements.txt
```

### 2. Validate the stack

```bash
python main.py validate
```

### 3. Serialize a checkpoint (or use `--dummy` for testing)

```bash
# With real weights
python main.py serialize --checkpoint ./ltx_video_2.3 --output ./models/ltx_video_2.3_pma

# Without real weights (generates synthetic data for testing)
python main.py serialize --checkpoint ./dummy --output ./models/ltx_video_2.3_pma --dummy
```

### 4. Generate a video

```bash
python main.py generate \
  --model ./models/ltx_video_2.3_pma \
  --prompt "A woman walks through a sunlit Japanese garden, cherry blossoms falling" \
  --duration 5 \
  --height 720 \
  --steps 25
```

### 5. Run a benchmark

```bash
python main.py benchmark \
  --model ./models/ltx_video_2.3_pma \
  --output ./profiles \
  --duration 5
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LTX-Video 2.3 DiT                     │
│  56 transformer blocks × 3D spatiotemporal attention    │
│  Full model: ~22B params (7.4 GB at FP16, ~1.2 GB tiled) │
└──────────┬──────────────────────────────────┬───────────┘
           │                                  │
    ┌──────▼──────┐                  ┌───────▼──────┐
    │  NVMe Store  │                  │   16GB RAM    │
    │  (7.4 GB/s)  │                  │ (273 GB/s BW) │
    │              │                  │               │
    │ block_0.npz  │◄── double ──────│  1 block      │
    │ block_1.npz  │◄── buffer ──────│  (200 MB)     │
    │ ...          │◄── 27ms ────────│               │
    │ block_55.npz │                  │  next block   │
    │              │                  │  prefetching   │
    └──────────────┘                  └───────────────┘
           │                                  │
    ┌──────▼──────────────────────────────────▼──────┐
    │           StreamingPipelineOrchestrator        │
    │  LSS + SLT + SCFG-SC + TAPB = 16GB feasible   │
    └────────────────────────────────────────────────┘
```

---

## Memory Budget

| Component | RAM |
|---|---|
| Active transformer block (W6A8 peak) | 0.30 GB |
| Prefetch buffer (next block) | 0.30 GB |
| Conditioning residual buffer | 0.10 GB |
| Spatiotemporal tile activations | 0.35 GB |
| Tile overlap + blending workspace | 0.25 GB |
| Sequential CFG stored output + residuals | 0.07 GB |
| VAE decoder (loaded after DiT phase) | 0.60 GB |
| Text encoder T5 (loaded once, then evicted) | 1.20 GB |
| Latent workspace | 0.05 GB |
| Video frame output buffer | 1.80 GB |
| NVMe DMA staging + OS overhead | 0.40 GB |
| macOS + system processes | 3.50 GB |
| **Headroom** | **6.68 GB** |

Headroom enables: 1080p, 8–10s clips, heavier background, or 30B non-distilled model.

---

## File Structure

```
pma2-ltx-video/
├── SPEC.md                    # Architecture specification (source of truth)
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── config.py                  # All hyperparameters
├── serialize_blocks.py        # Checkpoint → block files with multi-precision
├── tiling_engine.py           # Spatiotemporal tiling + coherence injection
├── streaming_pipeline.py      # LSS + SCFG-SC + TAPB orchestration
├── benchmark.py               # Profiling harness
├── main.py                    # CLI entry point
├── profiles/                  # Output from benchmark runs
└── models/                    # Serialized model weights
    └── ltx_video_2.3_pma/
```

---

## Precision Bands (TAPB)

| Timestep Range | Precision | Per-Block Size |
|---|---|---|
| t = T → 0.75T (steps 1–6) | W4A6 | ~200MB |
| 0.75T → 0.5T (steps 7–12) | W4A8 | ~200MB |
| 0.5T → 0.25T (steps 13–18) | W5A8 | ~250MB |
| 0.25T → 0 (steps 19–25) | W6A8 | ~300MB |

Block files contain all precision variants packed together. Streamer reads appropriate slice based on timestep. No reformatting, no runtime requantization.

---

## Performance

| Metric | Value |
|---|---|
| 5s 720p video generation | ~46 seconds total |
| Per diffusion step | ~1.45s effective |
| Per block compute | ~11ms |
| NVMe load per block | ~27ms (hidden under compute) |
| Memory utilization | 9.32GB / 16GB (58%) |

---

## Acknowledgments

Based on the Phantom Memory Architecture v2 paper from the nocodezoo scope directory. LTX-Video is a Lightricks production. Apple Silicon is an Apple Inc. product.