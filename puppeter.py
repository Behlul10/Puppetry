import threading
import time
import json
import cv2
import numpy as np
import mediapipe as mp

from flask import Flask, render_template
from flask_socketio import SocketIO

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Flask/SocketIO setup
app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

# Configure the static folder
app.static_folder = 'static'

@app.route("/")
def index():
    return render_template("Index.html")

# Global variables:
# latest_result_raw stores the original MediaPipe result (used for drawing)
# latest_result_dict will be a serializable dictionary sent to clients.
latest_result_raw = None
latest_result_dict = None

# MediaPipe Setup
model_path = 'models/face_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

def draw_landmarks_on_image(rgb_image, detection_result):
    """Visualize landmarks on the image.
       Expects detection_result to be a raw MediaPipe result object.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for face_landmarks in face_landmarks_list:
        # Create a NormalizedLandmarkList proto from the landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                y=landmark.y,
                                                z=landmark.z)
                for landmark in face_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.
                get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.
                get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.
                get_default_face_mesh_iris_connections_style(),
        )
    return annotated_image

# Callback function for asynchronous detection.
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result_raw, latest_result_dict
    # Save the raw result for drawing purposes.
    latest_result_raw = result

    # Process blendshapes.
    blendshape_scores = []
    blendshape_names = []
    if result.face_blendshapes and len(result.face_blendshapes) > 0:
        blendshape_scores = [b.score for b in result.face_blendshapes[0]]
       
