#!/usr/bin/env python3
"""
run_generation.py — Generate video using the WanVideoWrapper API directly.

Uses ComfyUI's internal API to run Wan2ImageToVideoApi with the 14B I2V model.
"""

import sys
import os

# Add ComfyUI to path
sys.path.insert(0, os.path.expanduser("~/ComfyUI"))
sys.path.insert(0, os.path.expanduser("~/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper"))

# Import inside venv python
venv_python = os.path.expanduser("~/ComfyUI/venv/bin/python3")

def main():
    import argparse
    p = argparse.ArgumentParser(description="PMA² WanVideo I2V Generation")
    p.add_argument("--image", default="oldman_upscaled.png")
    p.add_argument("--prompt", default="An old man walking through a sunlit Japanese garden, cherry blossoms falling")
    p.add_argument("--model", default="wan2.7-i2v")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", default="1080P")
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--negative", default="blurry, low quality, distorted, watermark")
    args = p.parse_args()

    print("=" * 60)
    print("PMA² + WanVideo I2V Generation")
    print("=" * 60)
    print(f"Image:    {args.image}")
    print(f"Prompt:   {args.prompt}")
    print(f"Model:    {args.model}")
    print(f"Seed:     {args.seed}")
    print(f"Res:      {args.resolution}")
    print(f"Duration: {args.duration}s")
    print()

    # Build the prompt dict for ComfyUI API
    prompt = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": args.image,
                "choose_image_to_upload": "image"
            }
        },
        "2": {
            "class_type": "Wan2ImageToVideoApi",
            "inputs": {
                "model": args.model,
                "first_frame": [[1, 0, None]],  # [src_node_id, src_output_slot, link_id]
                "prompt": args.prompt,
                "negative_prompt": args.negative,
                "seed": args.seed,
                "resolution": args.resolution,
                "duration": args.duration,
                "prompt_extend": False,
                "watermark": False,
            }
        }
    }

    print("Prompt structure:")
    print(json.dumps({k: {"class_type": v["class_type"]} for k, v in prompt.items()}, indent=2))
    print()

    # Send to ComfyUI running on port 8188
    import urllib.request
    import json as json_lib

    data = json_lib.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        "http://localhost:8188/api/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json_lib.loads(resp.read())
            print("Result:", json_lib.dumps(result, indent=2)[:500])
    except urllib.error.HTTPError as e:
        body = e.read()
        print(f"HTTP {e.code}: {body[:500]}")

if __name__ == "__main__":
    import json
    main()