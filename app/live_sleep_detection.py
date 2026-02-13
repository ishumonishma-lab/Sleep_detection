import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import cv2
from src.utils import load_trained_model, predict_state


model_path = os.path.join(ROOT_DIR, "models", "mobilenetv2_sleep_model.h5")
model = load_trained_model(model_path)

print("Model loaded successfully. Starting webcam...")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(face_gray)

        for (ex, ey, ew, eh) in eyes[:2]:  # take only first 2 eyes
            eye_img = face_color[ey:ey + eh, ex:ex + ew]

            state, confidence = predict_state(model, eye_img)

            color = (0, 255, 0) if state == "awake" else (0, 0, 255)

            
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), color, 2)

            
            cv2.putText(
                frame,
                f"{state} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("Sleep Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
