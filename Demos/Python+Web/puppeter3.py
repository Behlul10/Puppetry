#broken


#!/usr/bin/env python3
"""
mediapipe_ws_server.py

– Captures webcam frames
– Runs MediaPipe FaceLandmarker (blendshapes + 4×4 head‐pose)
– Serves a WebSocket on ws://0.0.0.0:8080
– Broadcasts JSON every ~33ms: { blendshapes: {...}, head_pose: [...] }
"""

import cv2
import time
import threading
import json
import asyncio

import numpy as np
import websockets

from mediapipe.tasks.python.core import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    VisionRunningMode,
    Image,
    ImageFormat,
)

# WebSocket server settings
WS_HOST = "0.0.0.0"
WS_PORT = 8080

# Path to your downloaded .task file
MODEL_PATH = "models/face_landmarker.task"

# Shared state
latest_result = None
state_lock = threading.Lock()


def compute_vlenshapes(result) -> dict:
    """
    Map the first face's blendshapes to a { name: score } dict.
    """
    out = {}
    bs = result.face_blendshapes
    if not bs:
        return out
    for cat in bs[0].categories:
        out[cat.category_name] = float(cat.score)
    return out


def compute_head_pose(result) -> list:
    """
    Return the first face's 4×4 transform as a flat list (row-major).
    """
    mats = result.facial_transformation_matrixes
    if not mats:
        return []
    # mats[0].data is a sequence of 16 floats
    return list(mats[0].data)


def detection_loop():
    """
    Capture from webcam, run Mediapipe, update latest_result.
    """
    global latest_result
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam device 0")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            # BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_img, ts_ms)

            with state_lock:
                latest_result = result

            # throttle to ~60 Hz capture
            time.sleep(0.016)

    cap.release()


async def ws_handler(ws, path):
    """
    For each connected client, send the latest_result ~30 FPS.
    """
    print(f"[WS] Client connected: {ws.remote_address}")
    try:
        while True:
            await asyncio.sleep(0.033)
            with state_lock:
                res = latest_result
            if res:
                payload = {
                    "blendshapes": compute_vlenshapes(res),
                    "head_pose": compute_head_pose(res),
                }
                await ws.send(json.dumps(payload))
    except websockets.exceptions.ConnectionClosed:
        print(f"[WS] Client disconnected: {ws.remote_address}")


async def main():
    # 1) Start the detection thread
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    # 2) Start WebSocket server
    print(f"[WS] Starting server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
