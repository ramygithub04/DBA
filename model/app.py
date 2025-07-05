from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import time
import os

app = Flask(__name__)
CORS(app)

model = None
face_cascade = None
eye_cascade = None
cap = None
is_monitoring = False
alarm_active = False
closed_eye_counter = 0
threshold = 40
monitoring_thread = None

def load_model_safely():
    global model, face_cascade, eye_cascade
    try:
        if os.path.exists("drowsiness_lenet_model.keras"):
            model = load_model("drowsiness_lenet_model.keras", compile=False)
            print("Model loaded from .keras format")
        else:
            model = load_model("drowsiness_lenet_model.h5", compile=False)
            print("Model loaded from .h5 format")
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def init_camera():
    global cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        return cap.isOpened()
    except Exception as e:
        print(f"Camera error: {e}")
        return False

def monitor_drowsiness():
    global closed_eye_counter, alarm_active, is_monitoring
    
    while is_monitoring:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

            drowsy = False
            eyes_detected = False

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

                for (ex, ey, ew, eh) in eyes:
                    eyes_detected = True
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    eye_roi = cv2.resize(eye_roi, (32, 32))
                    eye_roi = eye_roi / 255.0
                    eye_roi = eye_roi.reshape(1, 32, 32, 1)

                    try:
                        prediction = model.predict(eye_roi, verbose=0)
                        label = "Closed" if np.argmax(prediction) == 1 else "Open"

                        if label == "Closed":
                            closed_eye_counter += 1
                        else:
                            closed_eye_counter = 0

                        if closed_eye_counter >= threshold:
                            drowsy = True
                            if not alarm_active:
                                alarm_active = True
                                print("DROWSINESS ALERT!")
                    except Exception as e:
                        continue

            if not eyes_detected:
                closed_eye_counter = 0

            if not drowsy and alarm_active:
                alarm_active = False

            time.sleep(0.1)

        except Exception as e:
            time.sleep(0.1)

@app.route('/status', methods=['GET'])
def check_drowsiness():
    global closed_eye_counter

    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "No frame captured"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    drowsy = False
    eye_detections = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        print(f"Face detected - Eyes: {len(eyes)}")

        for i, (ex, ey, ew, eh) in enumerate(eyes):
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            eye_roi = cv2.resize(eye_roi, (32, 32))
            eye_roi = eye_roi / 255.0
            eye_roi = eye_roi.reshape(1, 32, 32, 1)

            try:
                prediction = model.predict(eye_roi, verbose=0)
                open_prob = prediction[0][0]
                closed_prob = prediction[0][1]
                label = "Closed" if np.argmax(prediction) == 1 else "Open"
                
                eye_position = "Left" if ex < w//2 else "Right"
                
                print(f"{eye_position} Eye: {label} (Open: {open_prob:.3f}, Closed: {closed_prob:.3f})")
                
                eye_detections.append({
                    "position": eye_position,
                    "open_prob": float(open_prob),
                    "closed_prob": float(closed_prob),
                    "label": label
                })

                if label == "Closed":
                    closed_eye_counter += 1
                else:
                    closed_eye_counter = 0

                if closed_eye_counter >= threshold:
                    drowsy = True
                    
            except Exception as e:
                continue

    print(f"Counter: {closed_eye_counter}/{threshold}, Drowsy: {drowsy}")

    return jsonify({
        "drowsy": drowsy,
        "closed_eye_counter": closed_eye_counter,
        "threshold": threshold,
        "eye_detections": eye_detections
    })

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global is_monitoring, monitoring_thread
    
    if not load_model_safely():
        return jsonify({"error": "Failed to initialize model"}), 500
    
    if not init_camera():
        return jsonify({"error": "Failed to initialize camera"}), 500
    
    if not is_monitoring:
        is_monitoring = True
        monitoring_thread = threading.Thread(target=monitor_drowsiness)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        return jsonify({"message": "Monitoring started"})
    
    return jsonify({"message": "Already monitoring"})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global is_monitoring, alarm_active, closed_eye_counter
    
    is_monitoring = False
    alarm_active = False
    closed_eye_counter = 0
    
    if cap and cap.isOpened():
        cap.release()
    
    return jsonify({"message": "Monitoring stopped"})

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm():
    global alarm_active, closed_eye_counter
    
    alarm_active = False
    closed_eye_counter = 0
    
    return jsonify({"message": "Alarm stopped"})

if __name__ == '__main__':
    print("Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
    