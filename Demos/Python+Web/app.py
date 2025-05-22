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
app = Flask(__name__, 
    static_folder='static',  # explicitly set static folder
    static_url_path='/static'  # explicitly set static URL path
)
socketio = SocketIO(app, async_mode="threading")

@app.route("/")
def index():
    return render_template("iiindex.html")

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
        try:
            blendshape_names = [b.category_name for b in result.face_blendshapes[0]]
            # Debug print all blendshapes with their scores
            print("\nMediaPipe Blendshapes:")
            for name, score in zip(blendshape_names, blendshape_scores):
                if score > 0.1:  # Only print significant values
                    print(f"{name}: {score:.2f}")
        except AttributeError:
            blendshape_names = []
            print("Warning: Could not get blendshape names")
    else:
        print("No blendshapes detected in this frame")
    
    # Create a serializable dictionary version.
    latest_result_dict = {
        "landmarks": result.face_landmarks,  # We'll convert this below.
        "blendshapes": {
            "scores": blendshape_scores,
            "names": blendshape_names
        }
    }

def result_to_dict(raw_landmarks):
    """
    Convert raw landmarks (a list of lists of NormalizedLandmark) to a dict
    suitable for JSON serialization.
    """
    landmarks_dict = {}
    for idx, face in enumerate(raw_landmarks):
        # Create a list of dicts for each landmark point.
        landmarks_dict[f"face_{idx}"] = [
            {"x": lmk.x, "y": lmk.y, "z": lmk.z} for lmk in face
        ]
    return landmarks_dict

# Define the MediaPipe FaceLandmarker options.
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_face_blendshapes=True,                   # Enable blendshape estimation.
    output_facial_transformation_matrixes=True,     # Enable facial transformation matrices.
    num_faces=1,
    result_callback=print_result,
)

def run_detector():
    """Runs the MediaPipe face landmark detection on webcam frames."""
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
            
            # For visualization, use the raw result.
            display_frame = frame
            if latest_result_raw:
                display_frame = draw_landmarks_on_image(frame, latest_result_raw)

            cv2.imshow("Face Landmarks", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

def send_landmarks():
    """Background task that emits detection data (landmarks and blendshapes)
       to connected clients.
    """
    global latest_result_dict
    while True:
        if latest_result_dict:
            data_to_emit = {}
            try:
                # Convert raw landmarks to a serializable dictionary.
                landmarks_serializable = result_to_dict(latest_result_dict["landmarks"])
                data_to_emit["landmarks"] = landmarks_serializable
                # Add blendshape data as is.
                data_to_emit["blendshapes"] = latest_result_dict.get("blendshapes", {})
            except Exception as e:
                print(f"Error processing data: {e}")

            socketio.emit("landmark_data", data_to_emit)
        socketio.sleep(0.033)  # roughly 30 FPS

if __name__ == "__main__":
    # Start sending detection data as a background task.
    socketio.start_background_task(send_landmarks)
    # Run the MediaPipe detector on a separate thread.
    detector_thread = threading.Thread(target=run_detector)
    detector_thread.daemon = True
    detector_thread.start()
    # Start the Flask-SocketIO server.
    socketio.run(app, port=5000)
