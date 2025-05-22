import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Use camera index 1 (your MacBook's webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Draw face detections
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Photobooth", frame)

    # Capture image on keypress ('s' to save)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("snapshot.png", frame)
        print("Image saved as snapshot.png")
    elif key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()