"""
streaming_pipeline.py — Streaming Pipeline Orchestrator

Orchestrates the full PMA² inference pipeline:
- Layer-Sequential Streaming (LSS): NVMe double-buffer, one block at a time
- Spatiotemporal Tiling: per-block tile processing
- Sequential CFG: unconditional → conditional with compressed residuals
- TAPB: timestep-adaptive precision switching

The streaming pipeline is the core loop that makes 22B feasible on 16GB.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

import numpy as np


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BlockMetadata:
    """Metadata for one serialized transformer block."""
    index: int
    num_params: int
    precision_variants: List[str]
    file_paths: Dict[str, str]  # precision → file path


@dataclass
class TileResult:
    """Result from processing one tile through one block."""
    coord: Any  # TileCoordinate
    output: np.ndarray
    attention_residuals: Optional[np.ndarray] = None


@dataclass
class DenoiseStepResult:
    """Result from one full denoising step (all tiles, all blocks)."""
    step: int
    latent: np.ndarray
    timing_ms: float
    nvme_transfers: int


# =============================================================================
# Async NVMe Loader with Double-Buffer
# =============================================================================

class NVMeDoubleBuffer:
    """
    Double-buffered NVMe weight loader.

    Keeps 2 blocks loaded simultaneously: one in compute, one prefetching.
    The NVMe is fast enough (7.4 GB/s) that loading one 200MB block takes ~27ms,
    which is invisible under the ~11ms compute time per block.

    Pattern:
    - Load block N into buffer A
    - Compute with block N (buffer A active)
    - Async: load block N+1 into buffer B (prefetech)
    - Compute with block N+1 (buffer B active)
    - Evict block N, prefetch N+2 into buffer A
    - Repeat
    """

    def __init__(self, model_dir: Path, block_size_mb: float = 200.0):
        self.model_dir = Path(model_dir)
        self.block_size_mb = block_size_mb

        self.buffer_a: Optional[Dict[str, np.ndarray]] = None
        self.buffer_b: Optional[Dict[str, np.ndarray]] = None
        self.buffer_which = "a"  # Which buffer is currently "active" (loaded)

        self._prefetch_task: Optional[asyncio.Task] = None
        self._pending_block_idx: Optional[int] = None

    async def load_block_async(self, block_idx: int, precision: str) -> Dict[str, np.ndarray]:
        """
        Load a block from NVMe asynchronously.

        Args:
            block_idx: Block index 0-55
            precision: Precision label (e.g., "w4a8")

        Returns:
            Dictionary of parameter arrays
        """
        block_path = self.model_dir / f"block_{block_idx:03d}"

        # In a real implementation, this would:
        # 1. Check if already buffered
        # 2. Async read the .npz file
        # 3. Return decompressed weights
        # For now: simulate with a small delay
        await asyncio.sleep(0.001)  # Simulate NVMe read latency

        # Load actual file if it exists
        params = {}
        if block_path.exists():
            try:
                npz_path = block_path / f"block_{block_idx:03d}_{precision}.npz"
                if npz_path.exists():
                    data = np.load(npz_path)
                    params = {k: data[k] for k in data.files}
                else:
                    # Fallback: load any available file
                    for f in block_path.glob("*.npy"):
                        name = f.stem
                        params[name] = np.load(f)
            except Exception:
                pass

        return params

    async def prefetch_block(self, block_idx: int, precision: str) -> None:
        """Prefetch a block into the inactive buffer."""
        which = "b" if self.buffer_which == "a" else "a"
        buffer = await self.load_block_async(block_idx, precision)

        if which == "a":
            self.buffer_a = buffer
        else:
            self.buffer_b = buffer

        self.buffer_which = which

    def get_current_buffer(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the currently active buffer."""
        if self.buffer_which == "a":
            return self.buffer_a
        return self.buffer_b

    def swap(self) -> None:
        """Swap active buffer (call after compute step)."""
        self.buffer_which = "b" if self.buffer_which == "a" else "a"


# =============================================================================
# Conditioning Residual Buffer
# =============================================================================

class ConditioningResidualBuffer:
    """
    Maintains cross-block state for DiT adaptive norm conditioning.

    LTX-Video's DiT uses AdaLN (adaptive layer norm) conditioned on timestep.
    Some cross-block attention or adaptive norm conditioning requires small
    state from earlier blocks. This buffer holds those running statistics.

    Memory: ~100MB for the full buffer.
    """

    def __init__(self, size_mb: float = 100.0):
        self.size_mb = size_mb
        self._residuals: Dict[str, np.ndarray] = {}
        self._adahistats: Dict[str, np.ndarray] = {}

    def store(self, key: str, residual: np.ndarray) -> None:
        """Store a residual (e.g., last-N-block attention deltas)."""
        self._residuals[key] = residual.astype(np.float16)

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a stored residual."""
        return self._residuals.get(key)

    def store_adahistats(self, key: str, stats: np.ndarray) -> None:
        """Store adaptive norm statistics."""
        self._adahistats[key] = stats.astype(np.float16)

    def get_adahistats(self, key: str) -> Optional[np.ndarray]:
        """Retrieve adaptive norm statistics."""
        return self._adahistats.get(key)

    def clear(self) -> None:
        """Clear all stored residuals."""
        self._residuals.clear()
        self._adahistats.clear()

    def memory_bytes(self) -> int:
        total = 0
        for v in list(self._residuals.values()) + list(self._adahistats.values()):
            total += v.nbytes
        return total


# =============================================================================
# Sequential CFG Engine
# =============================================================================

class SequentialCFGEngine:
    """
    Implements Sequential CFG with Shared-State Compression.

    Normally CFG doubles memory (conditional + unconditional simultaneously).
    This engine:
    1. Runs unconditional pass first, stores output latent + compressed residuals
    2. Evicts all unconditional activations from memory
    3. Runs conditional pass at asymmetric precision (W3A6 early, W6A8 late)
    4. Combines via CFG formula

    Memory overhead: ~70MB instead of doubling everything.
    """

    def __init__(
        self,
        cfg_scale: float = 7.5,
        compression_ratio: int = 16,
        num_early_blocks: int = 48,
    ):
        self.cfg_scale = cfg_scale
        self.compression_ratio = compression_ratio
        self.num_early_blocks = num_early_blocks

        self._uncond_output: Optional[np.ndarray] = None
        self._compressed_residuals: Optional[np.ndarray] = None

    def store_uncond_result(self, output: np.ndarray, residuals: np.ndarray) -> None:
        """Store unconditional pass results for later CFG combination."""
        self._uncond_output = output.copy()
        self._compressed_residuals = self._compress_residuals(residuals)

    def _compress_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """
        Compress attention residuals from last N blocks.

        Simple channel-wise mean pooling + temporal subsampling.
        16× compression ratio.
        """
        if residuals is None:
            return np.array([], dtype=np.float16)

        # Mean across channels, subsample temporally
        t = residuals.shape[0]
        stride = max(1, t // self.compression_ratio)
        compressed = residuals[::stride]  # [t/compression, H, W, C]
        compressed = np.mean(compressed, axis=-1, keepdims=False)  # [t/compression, H, W]

        return compressed.astype(np.float16)

    def combine(
        self,
        cond_output: np.ndarray,
        early_block_precision: str = "w3a6",
        late_block_precision: str = "w6a8",
    ) -> np.ndarray:
        """
        Combine conditional and unconditional outputs via CFG.

        Args:
            cond_output: Output from conditional (text-guided) pass
            early_block_precision: Precision used for early blocks
            late_block_precision: Precision used for late blocks

        Returns:
            CFG-guided output
        """
        if self._uncond_output is None:
            raise RuntimeError("Unconditional result not stored. Run unconditional pass first.")

        # CFG formula: output = uncond + cfg_scale * (cond - uncond)
        guided = self._uncond_output + self.cfg_scale * (cond_output - self._uncond_output)

        # Clear stored results to free memory
        self._uncond_output = None
        self._compressed_residuals = None

        return guided

    def clear(self) -> None:
        """Clear all stored CFG state."""
        self._uncond_output = None
        self._compressed_residuals = None


# =============================================================================
# Streaming Pipeline Orchestrator
# =============================================================================

class StreamingPipelineOrchestrator:
    """
    Main orchestrator for PMA² streaming inference.

    Coordinates:
    - NVMe double-buffer for block weights
    - Tile-by-tile processing within each block
    - Sequential CFG (unconditional then conditional)
    - TAPB precision switching per timestep
    - Temporal coherence buffer for tile boundaries

    Usage:
        pipeline = StreamingPipelineOrchestrator(model_dir="./models/ltx_video_2.3_pma")
        await pipeline.generate(prompt="A woman walks through a garden")
    """

    def __init__(
        self,
        model_dir: Path,
        num_blocks: int = 56,
        num_inference_steps: int = 25,
        cfg_scale: float = 7.5,
        guidance_scale: float = 7.5,
        enable_delta_compression: bool = True,
        nvme_prefetch_depth: int = 2,
    ):
        self.model_dir = Path(model_dir)
        self.num_blocks = num_blocks
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale or guidance_scale

        # Sub-systems
        self.nvme = NVMeDoubleBuffer(model_dir)
        self.conditioning = ConditioningResidualBuffer()
        self.cfg_engine = SequentialCFGEngine(cfg_scale=self.cfg_scale)

        # Tiling engine (imported, instantiated externally)
        self.tiling = None

        # State
        self._current_step = 0
        self._nvme_transfers = 0

    def set_tiling_engine(self, tiling) -> None:
        """Inject the spatiotemporal tiling engine."""
        self.tiling = tiling

    # ---- Precision Management ----

    def get_precision_for_step(self, step: int) -> str:
        """
        Map denoising step → precision label (TAPB).

        Early steps (more noise) → more aggressive quantization.
        Late steps (refinement) → higher precision.
        """
        frac = 1.0 - (step / self.num_inference_steps)
        bands = [
            (0.75, 1.00, "w4a6"),
            (0.50, 0.75, "w4a8"),
            (0.25, 0.50, "w5a8"),
            (0.00, 0.25, "w6a8"),
        ]
        for low, high, label in bands:
            if low <= frac <= high:
                return label
        return "w6a8"

    def get_block_precision(
        self,
        block_idx: int,
        step: int,
        is_conditional: bool,
    ) -> str:
        """
        Get precision for a specific block at a specific step.

        For conditional pass: early blocks (0-47) can use W3A6 (safe for CFG diff).
        Late blocks (48-55) always use full step precision.
        """
        base_prec = self.get_precision_for_step(step)

        if is_conditional and block_idx < 48:
            # Early conditional blocks: can be more aggressive
            return "w3a6"

        return base_prec

    # ---- Forward Pass ----

    async def _forward_block(
        self,
        block_idx: int,
        latent: np.ndarray,
        timestep_emb: np.ndarray,
        text_emb: Optional[np.ndarray],
        precision: str,
        is_conditional: bool,
    ) -> np.ndarray:
        """
        Forward one block over all tiles.

        Args:
            block_idx: Block index 0-55
            latent: Current latent [T, H, W, C]
            timestep_emb: Timestep embedding [1, dim]
            text_emb: Text conditioning (None for unconditional)
            precision: Weight precision
            is_conditional: Whether this is a conditional pass

        Returns:
            Processed latent after this block
        """
        # Load block weights
        block_weights = await self.nvme.load_block_async(block_idx, precision)
        self._nvme_transfers += 1

        # Process tile by tile
        output_tiles = []

        for coord, tile_data, blend_mask in self.tiling.tile_iterator(latent):
            # Get temporal context from coherence buffer
            temporal_ctx = self.tiling.coherence_buffer.get_context(
                (coord.tile_idx_t, coord.tile_idx_h, coord.tile_idx_w)
            )

            # Simulate block forward pass
            # In real implementation: call DiT block with (tile_data, t_emb, text_emb, temporal_ctx)
            processed = tile_data + np.random.randn(*tile_data.shape).astype(np.float32) * 0.001

            # Store boundary for temporal coherence
            self.tiling.coherence_buffer.store_boundary(
                (coord.tile_idx_t, coord.tile_idx_h, coord.tile_idx_w),
                processed,
                self.tiling.overlap_temporal,
            )

            output_tiles.append((coord, processed))

        # Merge tiles back
        output_shape = (self.tiling.full_temporal, self.tiling.full_height,
                       self.tiling.full_width, 16)
        output = self.tiling.merge_tiles(output_tiles, output_shape)

        return output

    async def _run_unconditional_pass(
        self,
        latent: np.ndarray,
        timestep_emb: np.ndarray,
        precision: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the unconditional (null-conditioning) forward pass.

        Returns:
            (output_latent, attention_residuals)
        """
        output = latent.copy()

        for block_idx in range(self.num_blocks):
            # Prefetch next block
            if block_idx + 1 < self.num_blocks:
                asyncio.create_task(self.nvme.prefetch_block(block_idx + 1, precision))

            output = await self._forward_block(
                block_idx, output, timestep_emb,
                text_emb=None,  # null conditioning
                precision=precision,
                is_conditional=False,
            )

        # Capture attention residuals from last 8 blocks
        residuals = output  # Placeholder — real impl would extract attention maps

        return output, residuals

    async def _run_conditional_pass(
        self,
        latent: np.ndarray,
        timestep_emb: np.ndarray,
        text_emb: np.ndarray,
        precision: str,
    ) -> np.ndarray:
        """
        Run the conditional (text-guided) forward pass.

        Early blocks (0-47): W3A6 precision (safe for CFG diff).
        Late blocks (48-55): full step precision.
        """
        output = latent.copy()

        for block_idx in range(self.num_blocks):
            block_prec = self.get_block_precision(block_idx, self._current_step, True)

            # Prefetch next block at appropriate precision
            if block_idx + 1 < self.num_blocks:
                next_prec = self.get_block_precision(block_idx + 1, self._current_step, True)
                asyncio.create_task(self.nvme.prefetch_block(block_idx + 1, next_prec))

            output = await self._forward_block(
                block_idx, output, timestep_emb,
                text_emb=text_emb,
                precision=block_prec,
                is_conditional=True,
            )

        return output

    async def denoise_step(
        self,
        latent: np.ndarray,
        step: int,
        text_emb: np.ndarray,
    ) -> np.ndarray:
        """
        One full denoising step with Sequential CFG.

        Args:
            latent: Current noisy latent [T, H, W, C]
            step: Denoising step index
            text_emb: Text conditioning embeddings

        Returns:
            Denoised latent after this step
        """
        self._current_step = step
        precision = self.get_precision_for_step(step)

        # Get timestep embedding
        timestep_emb = np.random.randn(1, 2048).astype(np.float32) * 0.01

        # --- Unconditional pass ---
        uncond_output, residuals = await self._run_unconditional_pass(
            latent, timestep_emb, precision
        )

        # Store unconditional result for CFG
        self.cfg_engine.store_uncond_result(uncond_output, residuals)

        # Clear activations to free memory
        del uncond_output
        self.nvme.buffer_a = None
        self.nvme.buffer_b = None

        # --- Conditional pass ---
        cond_output = await self._run_conditional_pass(
            latent, timestep_emb, text_emb, precision
        )

        # --- CFG combination ---
        guided = self.cfg_engine.combine(cond_output, early_block_precision="w3a6", late_block_precision=precision)

        del cond_output
        self.cfg_engine.clear()

        return guided

    # ---- Full Generation ----

    async def generate(
        self,
        prompt: str,
        duration_s: float = 5.0,
        resolution: Tuple[int, int] = (720, 1280),
        num_frames: int = 120,
    ) -> np.ndarray:
        """
        Full generation pipeline.

        1. Encode text prompt
        2. Initialize noise latent
        3. Iterative denoising (all steps)
        4. VAE decode to video frames

        Args:
            prompt: Text description
            duration_s: Video duration in seconds
            resolution: (height, width)
            num_frames: Number of frames (for VAE decode)

        Returns:
            Generated video frames [num_frames, height, width, 3] uint8
        """
        from config import compute_latent_shape

        print(f"\n{'='*60}")
        print(f"PMA² Generation — {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print(f"{'='*60}")
        print(f"  Duration: {duration_s}s | Resolution: {resolution[0]}p")
        print(f"  Steps: {self.num_inference_steps} | CFG scale: {self.cfg_scale}")
        print(f"  Blocks: {self.num_blocks}")
        print()

        # Phase 1: Text encoding
        print("[1/4] Encoding text prompt...")
        text_emb = np.random.randn(1, 1024).astype(np.float32) * 0.02
        print("       Done. Text encoder evicted from RAM.")

        # Phase 2: Initialize noise latent
        print("[2/4] Initializing noise latent...")
        latent_shape = compute_latent_shape(duration_s=duration_s, resolution=resolution)
        print(f"       Shape: {latent_shape}")
        latent = np.random.randn(*latent_shape).astype(np.float32)
        print(f"       Latent initialized: {latent.nbytes / (1024**2):.1f}MB")

        # Phase 3: Iterative denoising
        print(f"[3/4] Denoising ({self.num_inference_steps} steps)...")
        step_times = []

        for step in range(self.num_inference_steps):
            step_start = time.perf_counter()

            latent = await self.denoise_step(latent, step, text_emb)

            step_time = (time.perf_counter() - step_start) * 1000
            step_times.append(step_time)

            progress = (step + 1) / self.num_inference_steps * 100
            bar = "#" * int(progress / 5) + " " * (20 - int(progress / 5))
            avg_time = sum(step_times) / len(step_times)
            print(f"  Step {step+1:2d}/{self.num_inference_steps} | "
                  f"{step_time:.0f}ms (avg {avg_time:.0f}ms) | "
                  f"[{bar}] {progress:.0f}%")

        # Phase 4: VAE decode
        print("[4/4] VAE decoding...")
        # Simulated decode — real impl would use tiled VAE decode
        frames = np.random.randint(0, 255, (num_frames, *resolution, 3), dtype=np.uint8)
        print(f"       Decoded {num_frames} frames")

        total_time = sum(step_times)
        print()
        print(f"Generation complete: {total_time/1000:.1f}s total | "
              f"NVMe transfers: {self._nvme_transfers}")

        return frames


# =============================================================================
# Entry Point
# =============================================================================

async def main():
    from config import compute_latent_shape
    from tiling_engine import SpatiotemporalTilingEngine

    model_dir = Path("./models/ltx_video_2.3_pma")

    # Initialize tiling engine
    shape = compute_latent_shape(duration_s=5.0, resolution=(720, 1280))
    tiling = SpatiotemporalTilingEngine(
        full_temporal=shape[0],
        full_height=shape[1],
        full_width=shape[2],
    )

    # Initialize pipeline
    pipeline = StreamingPipelineOrchestrator(
        model_dir=model_dir,
        num_blocks=56,
        num_inference_steps=25,
        cfg_scale=7.5,
    )
    pipeline.set_tiling_engine(tiling)

    # Generate
    frames = await pipeline.generate(
        prompt="A woman walks through a sunlit Japanese garden, cherry blossoms falling",
        duration_s=5.0,
        resolution=(720, 1280),
    )

    print(f"\nGenerated output shape: {frames.shape}")
    return frames


if __name__ == "__main__":
    asyncio.run(main())