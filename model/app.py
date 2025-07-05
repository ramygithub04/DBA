from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
face_cascade = None
eye_cascade = None
cap = None

def load_model_safely():
    """Load the model with proper error handling for version compatibility"""
    global model
    try:
        # Try loading the new .keras format first
        if os.path.exists("drowsiness_lenet_model.keras"):
            model = load_model("drowsiness_lenet_model.keras", compile=False)
            print("Model loaded successfully from .keras format!")
            return True
        else:
            # Fall back to .h5 format
            model = load_model("drowsiness_lenet_model.h5", compile=False)
            print("Model loaded successfully from .h5 format!")
            return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to recreate model...")
        return create_model()

def create_model():
    """Create a new model if loading fails"""
    global model
    try:
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(120, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("New model created successfully!")
        return True
    except Exception as e:
        print(f"Error creating model: {e}")
        return False

def initialize_cascades():
    """Initialize face and eye cascade classifiers"""
    global face_cascade, eye_cascade
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        
        if face_cascade.empty() or eye_cascade.empty():
            print("Error: Could not load cascade classifiers")
            return False
        return True
    except Exception as e:
        print(f"Error loading cascade classifiers: {e}")
        return False

def initialize_camera():
    """Initialize camera capture"""
    global cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)  # Try alternative camera
        return cap.isOpened()
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

# Initialize components
if not load_model_safely():
    print("Failed to load or create model")
    exit(1)

if not initialize_cascades():
    print("Failed to initialize cascade classifiers")
    exit(1)

if not initialize_camera():
    print("Failed to initialize camera")
    exit(1)

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
    eye_detections = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        print(f"\n=== Face Detected ===")
        print(f"Face position: x={x}, y={y}, w={w}, h={h}")
        print(f"Eyes detected: {len(eyes)}")

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
                
                # Determine if it's left or right eye based on position
                eye_position = "Left" if ex < w//2 else "Right"
                
                print(f"\n--- {eye_position} Eye {i+1} ---")
                print(f"Position: x={ex}, y={ey}, w={ew}, h={eh}")
                print(f"Open probability: {open_prob:.4f}")
                print(f"Closed probability: {closed_prob:.4f}")
                print(f"Classification: {label}")
                
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
                print(f"Error in prediction: {e}")
                continue

    print(f"\n=== Summary ===")
    print(f"Closed eye counter: {closed_eye_counter}/{threshold}")
    print(f"Drowsy: {drowsy}")
    print(f"Eye detections: {len(eye_detections)}")
    
    if eye_detections:
        for eye in eye_detections:
            print(f"  {eye['position']} Eye: {eye['label']} (Open: {eye['open_prob']:.3f}, Closed: {eye['closed_prob']:.3f})")
    
    print("=" * 50)

    return jsonify({
        "drowsy": drowsy,
        "closed_eye_counter": closed_eye_counter,
        "threshold": threshold,
        "eye_detections": eye_detections
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "camera_available": cap is not None and cap.isOpened()
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Server will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=True)
    