#!/usr/bin/env python3
"""Run Wan2.1 I2V via ComfyUI REST API"""

import requests, json, time, sys, os, websocket

HOST = "http://localhost:8188"

def submit(prompt_data):
    r = requests.post(f"{HOST}/prompt", json={"prompt": prompt_data}, timeout=10)
    if r.status_code != 200:
        print(f"Error: {r.status_code} - {r.text[:300]}")
        return None
    return r.json()

# Wan2ImageToVideoApi with the installed Wan2.1 model
# But we only have wan2.7-i2v in the options list from Wan2ImageToVideoApi
# Let me try Wan2TextToVideoApi instead which also needs wan2.7-i2v

# Actually let me try a simpler approach - use the API nodes correctly
# Wan2ImageToVideoApi is the node that was used in the working history

# First: make sure the image exists
img_path = os.path.expanduser("~/ComfyUI/input/oldman_upscaled.png")
if not os.path.exists(img_path):
    # Check elsewhere
    for p in ["/Users/ryantsudek/Projects/pma2-ltx-video/test_images/oldman_upscaled.png"]:
        if os.path.exists(p):
            img_path = p
            break

print(f"Using image: {img_path}")
print(f"Exists: {os.path.exists(img_path)}")

# Try Wan2ImageToVideoApi - note it only accepts wan2.7-i2v model per options
# But let's try with the model referenced in the node config

workflow = {
    "1": {
        "class_type": "LoadImage",
        "inputs": {
            "image": os.path.basename(img_path),
            "choose_image_to_upload": "image"
        }
    },
    "2": {
        "class_type": "Wan2ImageToVideoApi",
        "inputs": {
            "model": [["wan2.7-i2v", "wan2.7-i2v", {}]],
            "first_frame": [["1", 0]],
            "prompt": "A serene zen garden with cherry blossoms falling, koi pond, morning sunlight, cinematic drone shot, hyper-real motion, 4k",
            "negative_prompt": "blurry, low quality, watermark, deformed, distorted",
            "seed": 42,
            "resolution": "1080P",
            "duration": 5,
            "prompt_extend": False,
            "watermark": False
        }
    }
}

print("Submitting Wan2ImageToVideoApi...")
result = submit(workflow)

if not result:
    sys.exit(1)

print(f"Result: {result}")
prompt_id = result.get("prompt_id") or result.get("number")
print(f"Prompt ID: {prompt_id}")

# Listen
ws = websocket.WebSocket()
ws.connect(f"ws://localhost:8188/ws")
greeting = json.loads(ws.recv())
print(f"Connected, sid={greeting['data']['sid']}")

outputs = []
start = time.time()

while time.time() - start < 1200:
    msg = ws.recv()
    if not msg:
        continue
    data = json.loads(msg)
    t = data.get("type")
    if t == "executing":
        print(f"  {data['data'].get('node', '')}")
    elif t == "executed":
        node = data["data"].get("node", "")
        print(f"  Done: {node}")
        for k in ["images", "videos"]:
            for item in data["data"].get(k, []):
                outputs.append(item.get("filename") if isinstance(item, dict) else item)
    elif t == "executing" and data["data"].get("node") is None:
        print("Done!")
        break

ws.close()
print(f"Output files: {outputs}")