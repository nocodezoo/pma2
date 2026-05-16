#!/usr/bin/env python3
"""
comfy_ws_client.py — Interact with ComfyUI via WebSocket API.

Send a prompt, monitor execution, save output video.
"""

import websocket
import json
import time
import sys
import os
import uuid

HOST = "localhost"
PORT = 8188
WS_URL = f"ws://{HOST}:{PORT}/ws"

def run_workflow(prompt_data, timeout=600):
    """Send prompt over WS, stream progress, return output paths."""
    ws = websocket.WebSocket()
    ws.connect(WS_URL)
    print(f"Connected to ComfyUI WS")

    # Read greeting
    greeting = json.loads(ws.recv())
    sid = greeting["data"]["sid"]
    print(f"Session ID: {sid}")

    prompt_id = str(uuid.uuid4())

    # Submit prompt
    ws.send(json.dumps({
        "type": "prompt",
        "data": {
            "prompt": prompt_data,
            "prompt_id": prompt_id,
            "number": 1,
            "branch": 254,
            "extra_data": {
                "ExtraService": {},
                "queue": 1
            }
        }
    }))
    print(f"Prompt submitted (ID: {prompt_id})")

    output_files = []
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            print("TIMEOUT")
            break

        msg = ws.recv()
        if not msg:
            continue

        msg = json.loads(msg)
        t = msg.get("type", "")

        if t == "executing":
            node = msg["data"].get("node", "")
            print(f"  Executing: {node}")

        elif t == "progress":
            step = msg["data"].get("value", 0)
            max_ = msg["data"].get("max", 1)
            print(f"  Progress: {step}/{max_}")

        elif t == "executed":
            node = msg["data"].get("node", "")
            print(f"  Executed: {node}")
            if "images" in msg["data"]:
                imgs = msg["data"]["images"]
                for img in imgs:
                    if isinstance(img, dict):
                        output_files.append(img.get("filename"))
            if "videos" in msg["data"]:
                vids = msg["data"]["videos"]
                for v in vids:
                    if isinstance(v, dict):
                        output_files.append(v.get("filename"))

        elif t == "status":
            q = msg["data"]["status"]["exec_info"]
            print(f"  Queue remaining: {q.get('queue_remaining', 0)}")

        elif t == "error":
            print(f"ERROR: {msg}")
            break

        elif t == "executing" and msg["data"].get("node") is None:
            print("Done!")
            break

    ws.close()
    return output_files

def main():
    # Minimal I2V workflow via Wan2ImageToVideoApi
    prompt = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "oldman_upscaled.png",
                "choose_image_to_upload": "image"
            }
        },
        "2": {
            "class_type": "Wan2ImageToVideoApi",
            "inputs": {
                "model": "wan2.7-i2v",
                "first_frame": [["1", 0]],
                "prompt": "An old man walking through a sunlit Japanese garden, cherry blossoms falling, cinematic 4K",
                "negative_prompt": "blurry, low quality, distorted",
                "seed": 42,
                "resolution": "1080P",
                "duration": 5,
                "prompt_extend": False,
                "watermark": False
            }
        }
    }

    print("Starting ComfyUI I2V generation...")
    outputs = run_workflow(prompt, timeout=600)
    print(f"\nOutput files: {outputs}")

if __name__ == "__main__":
    main()