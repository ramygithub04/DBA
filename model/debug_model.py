import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def test_model_loading():
    print("Testing Model Loading")
    try:
        if os.path.exists("drowsiness_lenet_model.keras"):
            model = load_model("drowsiness_lenet_model.keras", compile=False)
            print("Model loaded from .keras format")
        else:
            model = load_model("drowsiness_lenet_model.h5", compile=False)
            print("Model loaded from .h5 format")
        
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_camera():
    print("\nTesting Camera")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        
        if cap.isOpened():
            print("Camera accessed successfully")
            ret, frame = cap.read()
            if ret:
                print(f"Frame captured: {frame.shape}")
                cap.release()
                return True
            else:
                print("Failed to capture frame")
                cap.release()
                return False
        else:
            print("Camera not accessible")
            return False
    except Exception as e:
        print(f"Camera error: {e}")
        return False

def test_cascade_classifiers():
    print("\nTesting Cascade Classifiers")
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        
        if face_cascade.empty() or eye_cascade.empty():
            print("Failed to load cascade classifiers")
            return None, None
        
        print("Cascade classifiers loaded successfully")
        return face_cascade, eye_cascade
    except Exception as e:
        print(f"Error loading cascade classifiers: {e}")
        return None, None

def test_prediction(model):
    print("\nTesting Model Prediction")
    try:
        dummy_eye = np.random.rand(1, 32, 32, 1)
        prediction = model.predict(dummy_eye, verbose=0)
        print(f"Prediction successful")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction values: {prediction}")
        print(f"Predicted class: {np.argmax(prediction)}")
        return True
    except Exception as e:
        print(f"Prediction error: {e}")
        return False

def test_live_detection():
    print("\nTesting Live Detection")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        
        face_cascade, eye_cascade = test_cascade_classifiers()
        if face_cascade is None or eye_cascade is None:
            return False
        
        print("Starting live detection test...")
        print("Press 'q' to quit, 's' to save a test image")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                face_roi = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Debug - Face and Eye Detection', frame)
            frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('debug_frame.jpg', frame)
                print("Saved debug_frame.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Live detection test completed")
        return True
        
    except Exception as e:
        print(f"Live detection error: {e}")
        return False

def main():
    print("Starting Model Debug Tests...\n")
    
    model = test_model_loading()
    if model is None:
        print("Cannot proceed without model")
        return
    
    if not test_camera():
        print("Cannot proceed without camera")
        return
    
    face_cascade, eye_cascade = test_cascade_classifiers()
    if face_cascade is None or eye_cascade is None:
        print("Cannot proceed without cascade classifiers")
        return
    
    if not test_prediction(model):
        print("Model prediction failed")
        return
    
    print("\nAll basic tests completed!")
    print("\nNext steps:")
    print("1. Start your Flask server: python app.py")
    print("2. Test the API: curl http://localhost:5000/status")
    print("3. Check if the server responds correctly")

if __name__ == "__main__":
    main() 