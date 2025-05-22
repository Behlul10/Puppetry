#works
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

# Global variable to hold the latest landmark result.
latest_result = None

# MediaPipe Setup
model_path = 'models/face_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

def draw_landmarks_on_image(rgb_image, detection_result):
    """Visualize landmarks on the image."""
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    return annotated_image

# Callback function for asynchronous detection to update global result.
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Create FaceLandmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    result_callback=print_result,
)

def run_detector():
    """Runs the MediaPipe face landmark detection without gui."""
    cap = cv2.VideoCapture(1)  # use the webcam
    with vision.FaceLandmarker.create_from_options(options) as detector:
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            detector.detect_async(image, frame_timestamp_ms)
            time.sleep(0.005)
            # display_frame = frame
            # if latest_result:
            #     display_frame = draw_landmarks_on_image(frame, latest_result)

            # cv2.imshow("Face Landmarks", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

    cap.release()
    cv2.destroyAllWindows()

def send_landmarks():
    """Background task that emits the latest landmark data to connected clients."""
    global latest_result
    while True:
        if latest_result:
            landmarks_data = {}
            try:
                # Convert the landmarks to a serializable format.
                for idx, face in enumerate(latest_result.face_landmarks):
                    landmarks_data[f"face_{idx}"] = [
                        {"x": lmk.x, "y": lmk.y, "z": lmk.z} for lmk in face
                    ]
            except Exception as e:
                print(f"Error processing landmarks: {e}")

            socketio.emit("landmark_data", landmarks_data)
        socketio.sleep(0.033)  # roughly 30 FPS

if __name__ == "__main__":
    # Start sending landmark data in a background task.
    socketio.start_background_task(send_landmarks)
    # Run the MediaPipe detector in a separate thread.
    detector_thread = threading.Thread(target=run_detector)
    detector_thread.daemon = True
    detector_thread.start()
    # Start the Flask-SocketIO server.
    socketio.run(app, port=5000)
