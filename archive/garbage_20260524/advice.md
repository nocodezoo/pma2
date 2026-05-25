Adversarial Stress-Test of PMA² Architecture

I've gone through the numbers with a fine-tooth comb. Here's where the math holds, where it's tight, and where I see potential cracks:

✓ What Holds Up Solidly

Block sizing and NVMe streaming: 22B / 56 blocks = ~393M params/block → 196MB at W4. At 7.4 GB/s, that's 27ms load time. Compute per block dominates at 0.5–1s+ depending on chip TFLOPS, so I/O is completely hidden. No hole here.

Total available memory headroom: macOS uses 4-6GB, leaving 10-12GB. Peak inference footprint (calculated below) is 1.4-2GB. Plenty of room. Solid.

⚠️ Potential Holes / Tight Margins

1. The 600MB Attention Memory Claim is Understated

The claim is that active memory stays around 600MB. But let's add it all up properly for a single tile-block pass:

| Component | Size (A6 activations) |
|---|---|
| Current block weights (W4) | 200MB |
| Prefetched next block | 200MB |
| QKV tensors (32K × 5722 × 0.75 × 3) | 216MB |
| FFN intermediate (32K × ~22888 × 0.75) | 549MB |
| Attention output buffer | ~72MB |
| CFG residuals | 60MB |
| Misc buffers | ~100MB |
| Total | ~1.4GB |

The FFN intermediate alone is 549MB at A6. The real active footprint is ~1.4–2GB, not 600MB. The 600MB figure likely only accounts for attention QKV, ignoring FFN activations which are the larger tensor.

Verdict: Not fatal (still well under 10-12GB available), but the stated figure is misleading by ~2.5×. If someone budgets only 600MB headroom, they'll OOM.

2. Flash Attention on Metal — Implementation Gap

32K tokens without FlashAttention means materializing the attention score matrix:

Per head: 32K × 32K × 2 bytes = 2GB per head
24+ heads = catastrophic

FlashAttention is required, not optional. But Metal doesn't have a native FlashAttention kernel like CUDA does. Custom MLX implementations exist but are less optimized, potentially 30-50% slower than theoretical peak. This directly impacts the per-tile timing.

Verdict: Architectural dependency on a kernel that's not first-class on the target platform. Feasible but fragile.

3. The 46s Benchmark vs. 10-Minute Target — Internal Contradiction

These are 13× apart. Let me reconcile:

If total latent ≈ 32K tokens (1 tile): 56 blocks × ~0.5s/block = 28s, with SCFG ×1.5 = 42s ≈ 46s ✓
If total latent ≈ 172K tokens (higher res/longer): 6 tiles × 56 blocks × 0.5s × 1.5 = 252s ≈ 4.2 min

The 46s figure only works if the total latent is ≈32K tokens (i.e., aggressive compression like CogVideoX-style 8×8×4 with patch embedding). If the model's VAE is less aggressive, you immediately jump to multi-minute territory.

Verdict: The benchmark is valid only for a specific compression ratio. The architecture is latent-size-sensitive — a 2× increase in latent tokens means a 2× increase in runtime.

4. Thermal Throttling Under Sustained Load

MacBook thermal envelope sustains peak GPU for 90-120 seconds before throttling 20-40%. For the 46s case, you're fine. For any configuration exceeding 2 minutes of sustained compute, you must account for:

~25% throughput degradation after 2 min
A 4-minute job becomes ~5 minutes real-world
A 10-minute theoretical becomes ~12-13 minutes

Verdict: The 10-minute target for longer/higher-quality configs is at risk from thermals alone.

5. NVMe Sustained Throughput Degradation

7.4 GB/s is peak sequential burst. Under concurrent OS activity and after thermal saturation of the NAND controller:

Realistic sustained: 4.5–6 GB/s
Block load time: 33–44ms instead of 27ms

Not a problem (compute still dominates at 500ms+), but the stated figures should use conservative estimates.

Verdict: No real hole, just optimistic reporting.

6. SCFG-SC Quality Assumption

Compressing unconditional residuals to 60MB (from full-precision 384MB for 32K × 5722 × fp16) implies 6.4× compression. At lossy compression ratios that high, you're introducing quantization noise into the guidance signal. The claim of "negligible quality impact" needs validation across diverse prompts — guidance is particularly sensitive to high-frequency detail loss.

Verdict: Potentially a quality hole rather than a feasibility hole. Edge cases (fine text, complex textures) may degrade noticeably.

Summary Scorecard

| Claim | Status |
|---|---|
| NVMe hides I/O behind compute | ✅ Solid |
| Active memory < 600MB | ⚠️ Real figure is ~1.4-2GB |
| 46s for 5s 720p video | ✅ Valid if latent ≤ 32K tokens |
| Under 10 min for all configs | ⚠️ Thermal throttling threatens this |
| Flash Attention on Metal | ⚠️ Platform dependency risk |
| SCFG residual compression quality | ⚠️ Unvalidated at high compression |
| Overall feasibility on 16GB | ✅ Memory-feasible with real ~1.4-2GB peak |

Bottom Line

The architecture is fundamentally sound — nothing breaks it outright. But the stated memory figure (600MB) is misleadingly low, and the 46s benchmark is only valid under aggressive latent compression assumptions. The biggest real-world risk is the Flash Attention on Metal dependency: without a high-performance kernel, the entire pipeline's timing assumptions fall apart. If you're building this, that kernel is your critical path item.