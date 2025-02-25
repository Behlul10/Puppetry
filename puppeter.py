import mediapipe as mp
import cv2  # For webcam access
import numpy as np
import time

# Import MediaPipe drawing utilities
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# STEP 1: Import the necessary modules.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#MediaPipe Trained Model
model_path = 'models/face_landmarker.task'


# STEP 2: Create an FaceLandmarker object.
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1)

# Visualization function (adapted from the notebook)
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

# Result callback function
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    try:
        global latest_result  # Store the latest result
        latest_result = result
    except Exception as e:
        print(f"Error in print_result: {e}")

options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1,
                                       running_mode=VisionRunningMode.LIVE_STREAM,
                                       result_callback=print_result)

# STEP 3: Load the input image.
cap = cv2.VideoCapture(2) # webcam

with vision.FaceLandmarker.create_from_options(options) as detector:
    start_time = time.time()
    frame_count = 0
    latest_result = None # Initialize latest_result

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # STEP 4: Detect face landmarks from the input image.
        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        detector.detect_async(image, frame_timestamp_ms)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = frame #draw_landmarks_on_image(image.numpy_view(), latest_result) #image.numpy_view() #draw_landmarks_on_image(image.numpy_view(), detection_result)
        if latest_result:
            annotated_image = draw_landmarks_on_image(frame, latest_result)

        cv2.imshow("Face Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()