#Broken


#!/usr/bin/env python3
"""
puppeter3.py
Flask+SocketIO server that captures webcam, runs
MediaPipe FaceLandmarker in live‐stream mode, and
emits pose‐matrix & blendshape data to the browser.
"""

import time
import cv2
import threading
import os
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import mediapipe as mp

# — Flask + SocketIO setup —
app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# — MediaPipe Task API aliases (use mp.tasks.*) —
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode
FaceLandmarkerResult  = mp.tasks.vision.FaceLandmarkerResult

# Path to your .task model (download from MediaPipe Model Zoo)
MODEL_PATH = os.path.join("models", "face_landmarker.task")

# This callback will be invoked from MediaPipe’s background thread
latest_result = None
def _result_callback(
    result: FaceLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    global latest_result
    latest_result = result

# Build the FaceLandmarkerOptions
mp_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    result_callback=_result_callback,
)

def run_detector():
    """Capture from webcam, push frames to MediaPipe, emit via Socket.IO."""
    cap = cv2.VideoCapture(1, cv2.CAP_ANY)
    with FaceLandmarker.create_from_options(mp_options) as detector:
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            # Wrap OpenCV BGR frame into mp.Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame
            )
            ts = int((time.time() - t0) * 1000)
            detector.detect_async(mp_image, ts)

            # Once we have a result from the callback, serialize + emit it
            if latest_result:
                payload = {}
                # 1) 4×4 pose matrix (row‑major flatten)
                mats = latest_result.facial_transformation_matrixes or []
                if mats:
                    payload["matrix"] = mats[0].data  # list of 16 floats
                # 2) blendshape categories
                blends = latest_result.face_blendshapes or []
                if blends:
                    payload["blendshapes"] = [
                        {
                            "categoryName": c.category_name,
                            "score": c.score
                        }
                        for c in blends[0].categories
                    ]
                socketio.emit("mp_data", payload)
            # throttle ~30 FPS
            time.sleep(0.033)
    cap.release()

if __name__ == "__main__":
    # 1) Start the detector in a background task
    socketio.start_background_task(run_detector)
    # 2) Run Flask‐SocketIO
    socketio.run(app, host="0.0.0.0", port=5002)
