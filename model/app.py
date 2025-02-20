from flask import Flask, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("drowsiness_lenet_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

closed_eye_counter = 0
threshold = 40

@app.route('/status', methods=['GET'])
def check_drowsiness():
    global closed_eye_counter

    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "No frame captured"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    drowsy = False

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            eye_roi = cv2.resize(eye_roi, (32, 32))
            eye_roi = eye_roi / 255.0
            eye_roi = eye_roi.reshape(1, 32, 32, 1)

            prediction = model.predict(eye_roi)
            label = "Closed" if np.argmax(prediction) == 1 else "Open"

            if label == "Closed":
                closed_eye_counter += 1
            else:
                closed_eye_counter = 0

            if closed_eye_counter >= threshold:
                drowsy = True

    return jsonify({"drowsy": drowsy})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)