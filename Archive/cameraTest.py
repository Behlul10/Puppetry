import cv2

cap = cv2.VideoCapture(0)  # Change this index if needed

if not cap.isOpened():
    print("Camera access denied or unavailable.")
else:
    print("Using MacBook's built-in webcam!")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("MacBook Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()