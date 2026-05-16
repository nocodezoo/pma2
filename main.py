"""
main.py — PMA² CLI Entry Point

Commands:
    python main.py serialize --checkpoint ./ltx_video_2.3 --output ./models/ltx_video_2.3_pma
    python main.py generate --model ./models/ltx_video_2.3_pma --prompt "A woman walks through a garden"
    python main.py benchmark --model ./models/ltx_video_2.3_pma --output ./profiles

Usage:
    python main.py --help
"""

import sys
import os
import time
import asyncio
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Commands
# =============================================================================

def cmd_serialize(args) -> int:
    """Serialize a full LTX-Video checkpoint into PMA² block format."""
    from serialize_blocks import BlockSerializer

    print()
    print("=" * 60)
    print("PMA² Serialize — LTX-Video 2.3 → Block Format")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {args.output}")
    print(f"  Blocks:     {args.num_blocks}")
    print(f"  Dummy:      {args.dummy}")
    print()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    serializer = BlockSerializer(output_path, num_blocks=args.num_blocks)

    start = time.perf_counter()
    meta = serializer.serialize_full_model(checkpoint_path, create_dummy=args.dummy)
    elapsed = time.perf_counter() - start

    print()
    print(f"Serialization complete in {elapsed:.1f}s")
    print(f"  Blocks:      {meta['total_blocks_serialized']}")
    print(f"  Precisions:  {', '.join(meta['precision_variants'])}")
    print(f"  Output:      {output_path}")

    return 0


def cmd_generate(args) -> int:
    """Run full generation with PMA²."""
    from config import compute_latent_shape
    from tiling_engine import SpatiotemporalTilingEngine
    from streaming_pipeline import StreamingPipelineOrchestrator

    print()
    print("=" * 60)
    print("PMA² Generate")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Prompt:  {args.prompt}")
    print(f"  Duration: {args.duration}s")
    print(f"  Resolution: {args.height}p")
    print(f"  Steps:   {args.steps}")
    print(f"  CFG:     {args.cfg}")
    print()

    model_path = Path(args.model)

    # Initialize tiling engine
    shape = compute_latent_shape(
        duration_s=args.duration,
        resolution=(args.height, args.width),
    )
    tiling = SpatiotemporalTilingEngine(
        full_temporal=shape[0],
        full_height=shape[1],
        full_width=shape[2],
        tile_temporal=7,
        tile_height=53,
        tile_width=88,
        overlap_temporal=2,
        overlap_height=8,
        overlap_width=8,
    )

    # Initialize pipeline
    pipeline = StreamingPipelineOrchestrator(
        model_dir=model_path,
        num_blocks=args.num_blocks,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
    )
    pipeline.set_tiling_engine(tiling)

    # Run generation
    asyncio.run(pipeline.generate(
        prompt=args.prompt,
        duration_s=args.duration,
        resolution=(args.height, args.width),
        num_frames=int(args.duration * 24),
    ))

    return 0


def cmd_benchmark(args) -> int:
    """Run profiling benchmark."""
    from benchmark import ProfilingSession

    print()
    print("=" * 60)
    print("PMA² Benchmark")
    print("=" * 60)
    print(f"  Model:  {args.model}")
    print(f"  Output: {args.output}")
    print()

    session = ProfilingSession(output_dir=args.output, enable_thermal=not args.no_thermal)
    session.start()

    # Import and initialize
    from config import compute_latent_shape
    from tiling_engine import SpatiotemporalTilingEngine
    from streaming_pipeline import StreamingPipelineOrchestrator

    with session.time_section("initialization"):
        shape = compute_latent_shape(duration_s=args.duration, resolution=(args.height, args.width))
        tiling = SpatiotemporalTilingEngine(
            full_temporal=shape[0],
            full_height=shape[1],
            full_width=shape[2],
        )
        pipeline = StreamingPipelineOrchestrator(
            model_dir=args.model,
            num_blocks=args.num_blocks,
            num_inference_steps=args.steps,
        )
        pipeline.set_tiling_engine(tiling)

    # Generation (placeholder for real model)
    with session.time_section("generation"):
        with session.time_section("denoising"):
            time.sleep(0.1)  # Placeholder

    # Stop profiling
    report = session.stop()

    print()
    print(f"✅ Benchmark complete.")
    print(f"   Reports: {report['export_paths']['json']}")

    return 0


def cmd_validate(args) -> int:
    """Validate the full PMA² stack."""
    from config import (
        compute_latent_shape, compute_tile_count,
        DEFAULT_TILE_CONFIG, DEFAULT_MEMORY_BUDGET,
        get_precision_for_step, get_quant_config,
    )

    print()
    print("=" * 60)
    print("PMA² Validation")
    print("=" * 60)

    errors = []

    # Test 1: Latent shape computation
    print("\n[1/6] Latent shape computation...")
    shape = compute_latent_shape(duration_s=5.0, resolution=(720, 1280))
    expected_shape = (15, 90, 160, 16)
    if shape == expected_shape:
        print(f"  ✓ Shape: {shape}")
    else:
        errors.append(f"Shape mismatch: expected {expected_shape}, got {shape}")
        print(f"  ✗ Shape: {shape} (expected {expected_shape})")

    # Test 2: Tile grid computation
    print("\n[2/6] Tile grid computation...")
    tiles = compute_tile_count(DEFAULT_TILE_CONFIG)
    expected_tiles = (3, 2, 2, 12)
    if tiles == expected_tiles:
        print(f"  ✓ Tiles: {tiles[0]}×{tiles[1]}×{tiles[2]} = {tiles[3]} total")
    else:
        errors.append(f"Tile mismatch: expected {expected_tiles}, got {tiles}")
        print(f"  ✗ Tiles: {tiles} (expected {expected_tiles})")

    # Test 3: Precision band mapping
    print("\n[3/6] Precision band mapping...")
    tests = [(0, 25, "w4a6"), (12, 25, "w4a8"), (15, 25, "w5a8"), (22, 25, "w6a8")]
    all_ok = True
    for step, num_steps, expected in tests:
        actual = get_precision_for_step(step, num_steps)
        if actual == expected:
            print(f"  ✓ Step {step}/{num_steps}: {actual}")
        else:
            print(f"  ✗ Step {step}/{num_steps}: {actual} (expected {expected})")
            all_ok = False
    if not all_ok:
        errors.append("Precision band mapping failed")

    # Test 4: Quant config lookup
    print("\n[4/6] Quantization config lookup...")
    try:
        for label in ["w3a6", "w4a8", "w5a8", "w6a8"]:
            cfg = get_quant_config(label)
            print(f"  ✓ {label}: w{cfg.weight_bits}a{cfg.activation_bits}, group={cfg.group_size}")
    except Exception as e:
        errors.append(f"Quant config lookup failed: {e}")
        print(f"  ✗ Error: {e}")

    # Test 5: Memory budget validation
    print("\n[5/6] Memory budget validation...")
    try:
        DEFAULT_MEMORY_BUDGET.validate()
        headroom = DEFAULT_MEMORY_BUDGET.headroom_mb() / 1024.0
        used = DEFAULT_MEMORY_BUDGET.total_used_mb() / 1024.0
        print(f"  ✓ Memory: {used:.2f}GB used | {headroom:.2f}GB headroom")
    except AssertionError as e:
        errors.append(f"Memory budget exceeded: {e}")
        print(f"  ✗ {e}")

    # Test 6: Tiling engine
    print("\n[6/6] Tiling engine instantiation...")
    try:
        import numpy as np
        from tiling_engine import SpatiotemporalTilingEngine

        engine = SpatiotemporalTilingEngine(
            full_temporal=15,
            full_height=90,
            full_width=160,
            tile_temporal=7,
            tile_height=53,
            tile_width=88,
            overlap_temporal=2,
            overlap_height=8,
            overlap_width=8,
        )

        grid = engine.compute_tile_grid()
        print(f"  ✓ Tiling engine: {len(grid)} tiles")

        # Test tile iterator
        dummy_latent = np.random.randn(15, 90, 160, 16).astype(np.float32)
        tile_count = sum(1 for _ in engine.tile_iterator(dummy_latent))
        print(f"  ✓ Tile iterator: {tile_count} tiles")

    except Exception as e:
        errors.append(f"Tiling engine failed: {e}")
        print(f"  ✗ Error: {e}")

    # Summary
    print()
    print("=" * 60)
    if errors:
        print(f"❌ VALIDATION FAILED — {len(errors)} error(s)")
        for err in errors:
            print(f"   • {err}")
        return 1
    else:
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 60)
        return 0


# =============================================================================
# CLI Parser
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PMA² — Phantom Memory Architecture v2 for LTX-Video 2.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  serialize   Split LTX-Video checkpoint into PMA² block format
  generate    Run full generation with PMA² pipeline
  benchmark   Run profiling benchmark
  validate    Validate the full PMA² stack

Examples:
  # Serialize a checkpoint (use --dummy to test without real weights)
  python main.py serialize --checkpoint ./ltx_video_2.3 --output ./models/ltx_video_2.3_pma --dummy

  # Generate a video
  python main.py generate --model ./models/ltx_video_2.3_pma --prompt "A woman walks through a garden" --duration 5 --steps 25

  # Run benchmark
  python main.py benchmark --model ./models/ltx_video_2.3_pma --output ./profiles --duration 5

  # Validate everything
  python main.py validate
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # serialize
    p_serialize = subparsers.add_parser("serialize", help="Serialize checkpoint to block format")
    p_serialize.add_argument("--checkpoint", required=True, help="Path to LTX-Video checkpoint")
    p_serialize.add_argument("--output", required=True, help="Output directory for blocks")
    p_serialize.add_argument("--num-blocks", type=int, default=56, help="Number of transformer blocks")
    p_serialize.add_argument("--dummy", action="store_true", help="Generate dummy weights (testing)")

    # generate
    p_gen = subparsers.add_parser("generate", help="Run video generation")
    p_gen.add_argument("--model", required=True, help="Path to serialized model")
    p_gen.add_argument("--prompt", required=True, help="Text prompt")
    p_gen.add_argument("--duration", type=float, default=5.0, help="Video duration in seconds")
    p_gen.add_argument("--height", type=int, default=720, help="Vertical resolution")
    p_gen.add_argument("--width", type=int, default=1280, help="Horizontal resolution")
    p_gen.add_argument("--steps", type=int, default=25, help="Number of denoising steps")
    p_gen.add_argument("--cfg", type=float, default=7.5, help="CFG guidance scale")
    p_gen.add_argument("--num-blocks", type=int, default=56, help="Number of transformer blocks")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run profiling benchmark")
    p_bench.add_argument("--model", required=True, help="Path to serialized model")
    p_bench.add_argument("--output", default="./pma2_profiles", help="Output directory")
    p_bench.add_argument("--duration", type=float, default=5.0, help="Video duration")
    p_bench.add_argument("--height", type=int, default=720, help="Vertical resolution")
    p_bench.add_argument("--width", type=int, default=1280, help="Horizontal resolution")
    p_bench.add_argument("--steps", type=int, default=25, help="Number of denoising steps")
    p_bench.add_argument("--num-blocks", type=int, default=56, help="Number of transformer blocks")
    p_bench.add_argument("--no-thermal", action="store_true", help="Disable thermal monitoring")

    # validate
    p_valid = subparsers.add_parser("validate", help="Validate the full PMA² stack")

    args = parser.parse_args()

    # Route to command handler
    if args.command == "serialize":
        return cmd_serialize(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "validate":
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)