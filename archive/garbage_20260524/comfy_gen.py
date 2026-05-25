#!/usr/bin/env python3
"""
ComfyUI Video Generation via REST API

Uses Wan2ImageToVideoApi (local model, 720P/1080P) or LtxvApiTextToVideo (cloud, HD).
 MPS free: ~7GB — 720P should fit.
"""

import requests, json, time, sys, os, uuid, websocket

HOST = "http://localhost:8188"
API = f"{HOST}/prompt"

def submit_prompt(prompt_data):
    """Submit prompt via REST API"""
    resp = requests.post(API, json={"prompt": prompt_data}, timeout=10)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} - {resp.text}")
        resp.raise_for_status()
    return resp.json()

def wait_for_completion(timeout=600):
    """Wait via WebSocket for any execution"""
    ws_url = f"ws://localhost:8188/ws"
    ws = websocket.WebSocket()
    ws.connect(ws_url)
    
    greeting = json.loads(ws.recv())
    sid = greeting["data"]["sid"]
    
    outputs = []
    start = time.time()
    
    while time.time() - start < timeout:
        msg = ws.recv()
        if not msg:
            continue
        
        data = json.loads(msg)
        t = data.get("type", "")
        
        if t == "executing":
            node = data["data"].get("node", "")
            print(f"  {node}")
        elif t == "executed":
            node = data["data"].get("node", "")
            if "images" in data["data"]:
                for img in data["data"]["images"]:
                    outputs.append(img.get("filename") if isinstance(img, dict) else img)
            if "videos" in data["data"]:
                for vid in data["data"]["videos"]:
                    outputs.append(vid.get("filename") if isinstance(vid, dict) else vid)
        elif t == "status":
            q = data["data"]["status"]["exec_info"]
            rem = q.get("queue_remaining", 0)
            if rem == 0 and t == "executing" and data["data"].get("node") is None:
                break
        elif t == "executing" and data["data"].get("node") is None:
            break
    
    ws.close()
    return outputs

def image_to_video(image_path, prompt, duration=5, resolution="1080P", seed=42, model="wan2.7-i2v"):
    """Wan2ImageToVideoApi — local Wan 2.7 I2V model"""
    
    image_name = os.path.basename(image_path)
    
    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
                "choose_image_to_upload": "image"
            }
        },
        "2": {
            "class_type": "Wan2ImageToVideoApi",
            "inputs": {
                "model": [["wan2.7-i2v", "wan2.7-i2v", {}]],
                "first_frame": [["1", 0]],
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, artifacts",
                "seed": seed,
                "resolution": resolution,
                "duration": duration,
                "prompt_extend": False,
                "watermark": False
            }
        }
    }
    
    print(f"Submitting I2V: image={image_name}, prompt={prompt[:60]}, res={resolution}")
    result = submit_prompt(workflow)
    print(f"Queued: {result}")
    return result.get("prompt_id")

def text_to_video(prompt, duration=8, resolution="1920x1080", fps=25, model="LTX-2 (Fast)"):
    """LtxvApiTextToVideo — LTX-2 cloud model"""
    
    workflow = {
        "1": {
            "class_type": "LtxvApiTextToVideo",
            "inputs": {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "fps": fps,
            }
        }
    }
    
    print(f"Submitting T2V: prompt={prompt[:60]}, model={model}, res={resolution}")
    result = submit_prompt(workflow)
    print(f"Queued: {result}")
    return result.get("prompt_id")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="A serene Japanese zen garden with cherry blossoms falling, cinematic drone shot, golden hour lighting")
    parser.add_argument("--image", "-i", default=None)
    parser.add_argument("--duration", "-d", type=int, default=5)
    parser.add_argument("--resolution", "-r", default="1080P", choices=["720P", "1080P"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    
    print("=" * 60)
    print("ComfyUI Video Generation")
    print(f"  Prompt: {args.prompt[:80]}")
    print(f"  Image: {args.image}")
    print(f"  Duration: {args.duration}s, Resolution: {args.resolution}")
    print("=" * 60)
    
    t0 = time.time()
    
    # Check if image exists
    if args.image and not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        # Try in ComfyUI's input directory
        alt = os.path.join(os.path.expanduser("~/ComfyUI"), "input", os.path.basename(args.image))
        if os.path.exists(alt):
            args.image = alt
            print(f"  Using: {args.image}")
    
    if args.image:
        prompt_id = image_to_video(args.image, args.prompt, args.duration, args.resolution, args.seed)
    else:
        prompt_id = text_to_video(args.prompt, args.duration, args.resolution)
    
    print("\nWaiting for generation...")
    outputs = wait_for_completion(args.timeout)
    
    total = time.time() - t0
    print(f"\n=== Done in {total:.0f}s ===")
    print(f"Output files: {outputs}")
    
    if outputs and args.output:
        # Copy outputs to requested path
        for fname in outputs:
            src = os.path.join(os.path.expanduser("~/ComfyUI/output"), fname)
            if os.path.exists(src):
                import shutil
                dst = args.output if len(outputs) == 1 else args.output.rsplit(".", 1)[0] + "_" + fname
                shutil.copy2(src, dst)
                print(f"Saved: {dst}")

if __name__ == "__main__":
    main()