"""
config.py — PMA² Hyperparameters & Memory Budgets

All tunable parameters for Phantom Memory Architecture v2.
Single source of truth — imported everywhere.
"""

from dataclasses import dataclass
from typing import Tuple, List

# =============================================================================
# Apple Silicon M-Series Constants
# =============================================================================

M5_MEMORY_BANDWIDTH_GB_S = 273.0
M5_UNIFIED_RAM_GB = 16.0
M5_USABLE_RAM_GB = 12.0          # After macOS + system overhead
M5_NVME_SUSTAINED_GB_S = 7.4     # Sequential read
M5_NVME_BLOCK_LOAD_MS = 27       # Time to load 200MB block


# =============================================================================
# LTX-Video 2.3 DiT Model Constants
# =============================================================================

LTX_NUM_BLOCKS = 56
LTX_HIDDEN_DIM = 2048
LTX_MLP_DIM = LTX_HIDDEN_DIM * 4
LTX_LATENT_CHANNELS = 16
LTX_DEFAULT_NUM_STEPS = 25
LTX_DEFAULT_CFG_SCALE = 7.5


# =============================================================================
# Precision Variants (weight_bits, activation_bits, group_size)
# =============================================================================

@dataclass
class QuantConfig:
    label: str
    weight_bits: int
    activation_bits: int
    group_size: int
    block_size_mb: float  # Estimated on-disk size per block

PRECISIONS: List[QuantConfig] = [
    QuantConfig("w3a6",  weight_bits=3,  activation_bits=6,  group_size=32,  block_size_mb=180.0),
    QuantConfig("w4a6",  weight_bits=4,  activation_bits=6,  group_size=32,  block_size_mb=160.0),
    QuantConfig("w4a8",  weight_bits=4,  activation_bits=8,  group_size=32,  block_size_mb=180.0),
    QuantConfig("w5a8",  weight_bits=5,  activation_bits=8,  group_size=32,  block_size_mb=220.0),
    QuantConfig("w6a8",  weight_bits=6,  activation_bits=8,  group_size=16,  block_size_mb=300.0),
]

# Tapestry: step_fraction → precision label
PRECISION_BANDS = [
    (0.75, 1.00, "w4a6"),   # Early noise — steps 1–6
    (0.50, 0.75, "w4a8"),   # Structure forming — steps 7–12
    (0.25, 0.50, "w5a8"),   # Fine detail — steps 13–18
    (0.00, 0.25, "w6a8"),   # Final refinement — steps 19–25
]


# =============================================================================
# Spatiotemporal Tiling Configuration
# =============================================================================

@dataclass
class TileConfig:
    # Full latent dimensions (720p, 5s at 24fps latent-compressed 8×)
    full_temporal: int = 15       # 120 frames / 8 temporal compression
    full_height: int = 90         # 720 / 8
    full_width: int = 160         # 1280 / 8

    # Tile dimensions
    tile_temporal: int = 7
    tile_height: int = 53         # With overlap: 53 + 8*2 = 69 ≈ 90/2+overlap
    tile_width: int = 88          # 88 + 8*2 = 104 ≈ 160/2+overlap

    # Overlap
    overlap_temporal: int = 2
    overlap_height: int = 8
    overlap_width: int = 8

    # Grid: how many tiles in (T, H, W)
    # Computed: ceil((full - overlap) / (tile - overlap))
    #  T: ceil((15-2)/(7-2)) = ceil(13/5) = 3
    #  H: ceil((90-8)/(53-8)) = ceil(82/45) = 2
    #  W: ceil((160-8)/(88-8)) = ceil(152/80) = 2
    # Total: 3×2×2 = 12 tiles

    # Coherence injection
    coherence_strength: float = 0.15
    coherence_channels: int = 4   # Top-N channels for coherence features

    @property
    def num_tiles_t(self) -> int:
        eff = self.tile_temporal - self.overlap_temporal
        return max(1, (self.full_temporal - self.overlap_temporal + eff - 1) // eff)

    @property
    def num_tiles_h(self) -> int:
        eff = self.tile_height - self.overlap_height
        return max(1, (self.full_height - self.overlap_height + eff - 1) // eff)

    @property
    def num_tiles_w(self) -> int:
        eff = self.tile_width - self.overlap_width
        return max(1, (self.full_width - self.overlap_width + eff - 1) // eff)

    @property
    def total_tiles(self) -> int:
        return self.num_tiles_t * self.num_tiles_h * self.num_tiles_w

    @property
    def effective_tile_t(self) -> int:
        return self.tile_temporal - self.overlap_temporal

    @property
    def effective_tile_h(self) -> int:
        return self.tile_height - self.overlap_height

    @property
    def effective_tile_w(self) -> int:
        return self.tile_width - self.overlap_width


# =============================================================================
# Memory Budget (validated to sum ≤ 16GB)
# =============================================================================

@dataclass
class MemoryBudget:
    # Weights (LSS — one block + prefetch + conditioning)
    active_block_mb: float = 300.0
    prefetch_buffer_mb: float = 300.0
    conditioning_buffer_mb: float = 100.0

    # Activations (SLT — one tile + overlap + blending)
    tile_activations_mb: float = 350.0
    overlap_workspace_mb: float = 250.0

    # CFG (sequential — unconditional output + compressed residuals)
    cfg_overhead_mb: float = 70.0

    # Other components
    vae_decoder_mb: float = 600.0
    text_encoder_mb: float = 1200.0
    latent_workspace_mb: float = 50.0
    frame_output_buffer_mb: float = 1800.0
    nvme_dma_overhead_mb: float = 400.0

    # System
    macos_overhead_gb: float = 3.5

    def total_used_mb(self) -> float:
        return (
            self.active_block_mb +
            self.prefetch_buffer_mb +
            self.conditioning_buffer_mb +
            self.tile_activations_mb +
            self.overlap_workspace_mb +
            self.cfg_overhead_mb +
            self.vae_decoder_mb +
            self.text_encoder_mb +
            self.latent_workspace_mb +
            self.frame_output_buffer_mb +
            self.nvme_dma_overhead_mb
        )

    def total_system_mb(self) -> float:
        return self.macos_overhead_gb * 1024.0

    def headroom_mb(self) -> float:
        total = 16.0 * 1024.0
        return total - self.total_used_mb() - self.total_system_mb()

    def validate(self) -> None:
        used = self.total_used_mb()
        system = self.total_system_mb()
        total = used + system
        headroom = 16.0 * 1024.0 - total

        assert headroom >= 0, (
            f"Memory budget EXCEEDED by {abs(headroom):.0f}MB. "
            f"Used: {used:.0f}MB + System: {system:.0f}MB = {total:.0f}MB / 16384MB. "
            f"Reduce tile size, frame count, or output resolution."
        )

        print(f"  Memory: {used/1024:.2f}GB used | {system/1024:.2f}GB system | "
              f"{headroom/1024:.2f}GB headroom")


# =============================================================================
# Streaming Pipeline Config
# =============================================================================

@dataclass
class PipelineConfig:
    model_path: str
    memory_budget_mb: int = 14_000       # Leave 2GB headroom
    num_inference_steps: int = LTX_DEFAULT_NUM_STEPS
    guidance_scale: float = LTX_DEFAULT_CFG_SCALE
    enable_delta_compression: bool = True
    nvme_prefetch_depth: int = 2          # How many blocks ahead to prefetch
    async_enabled: bool = True
    force_sync_end_of_pass: bool = True   # Free all intermediates before next CFG pass


# =============================================================================
# Latent Shape Computation
# =============================================================================

def compute_latent_shape(
    duration_s: float = 5.0,
    fps: int = 24,
    resolution: Tuple[int, int] = (720, 1280),
    temporal_compression: int = 8,
    spatial_compression: int = 8,
) -> Tuple[int, int, int, int]:
    """
    Compute the 4D latent shape [T, H, W, C] for a video generation.

    Args:
        duration_s: Video duration in seconds
        fps: Frames per second
        resolution: (height, width)
        temporal_compression: VAE temporal compression factor
        spatial_compression: VAE spatial compression factor

    Returns:
        (temporal, height, width, channels) — the [T, H, W, C] shape
    """
    height, width = resolution
    num_frames = int(duration_s * fps)

    # Account for VAE temporal compression
    T = num_frames // temporal_compression
    H = height // spatial_compression
    W = width // spatial_compression
    C = LTX_LATENT_CHANNELS

    return (T, H, W, C)


def compute_tile_count(tile_config: TileConfig) -> Tuple[int, int, int, int]:
    """Compute tile grid dimensions and total count."""
    n_t = tile_config.num_tiles_t
    n_h = tile_config.num_tiles_h
    n_w = tile_config.num_tiles_w
    return (n_t, n_h, n_w, n_t * n_h * n_w)


# =============================================================================
# Precision Lookup
# =============================================================================

def get_precision_for_step(step: int, num_steps: int = LTX_DEFAULT_NUM_STEPS) -> str:
    """
    Map denoising step → precision label.

    Uses TAPB bands: early steps are noisier → more aggressive quantization.
    frac = 1.0 at step 0 (start), 0.0 at step N (end).
    """
    frac = 1.0 - (step / num_steps)  # 1.0 at step 0, 0.0 at step N

    for low, high, label in PRECISION_BANDS:
        if low <= frac <= high:  # inclusive on both ends to catch frac=1.0
            return label

    # Final steps — use highest precision
    return "w6a8"


def get_quant_config(label: str) -> QuantConfig:
    """Get QuantConfig by label."""
    for p in PRECISIONS:
        if p.label == label:
            return p
    raise ValueError(f"Unknown precision label: {label}")


# =============================================================================
# Default configurations
# =============================================================================

DEFAULT_TILE_CONFIG = TileConfig()
DEFAULT_MEMORY_BUDGET = MemoryBudget()


# =============================================================================
# Unit Tests / Validation
# =============================================================================

if __name__ == "__main__":
    print("PMA² Configuration Validation")
    print("=" * 50)

    # Validate latent shape computation
    shape = compute_latent_shape(duration_s=5.0, resolution=(720, 1280))
    print(f"Latent shape (720p, 5s, 24fps): {shape}")
    assert shape == (15, 90, 160, 16), f"Expected (15, 90, 160, 16), got {shape}"

    # Validate tile grid
    tiles = compute_tile_count(DEFAULT_TILE_CONFIG)
    print(f"Tile grid (T×H×W = total): {tiles[0]}×{tiles[1]}×{tiles[2]} = {tiles[3]} tiles")
    assert tiles == (3, 2, 2, 12), f"Expected (3, 2, 2, 12), got {tiles}"

    # Validate precision lookup
    assert get_precision_for_step(0, 25) == "w4a6"
    assert get_precision_for_step(12, 25) == "w4a8"
    assert get_precision_for_step(15, 25) == "w5a8"
    assert get_precision_for_step(22, 25) == "w6a8"
    print("Precision band mapping: OK")

    # Validate memory budget
    print()
    DEFAULT_MEMORY_BUDGET.validate()
    headroom_gb = DEFAULT_MEMORY_BUDGET.headroom_mb() / 1024.0
    print(f"Headroom: {headroom_gb:.2f}GB")

    if headroom_gb < 5.0:
        print("⚠️  Headroom below 5GB — consider reducing output resolution or duration.")

    print()
    print("All validations passed.")