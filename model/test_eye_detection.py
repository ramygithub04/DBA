import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def test_eye_detection():
    
    if os.path.exists("drowsiness_lenet_model.keras"):
        model = load_model("drowsiness_lenet_model.keras", compile=False)
        print("Model loaded from .keras format")
    else:
        model = load_model("drowsiness_lenet_model.h5", compile=False)
        print("Model loaded from .h5 format")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    print("\nEye Detection Test")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("=" * 50)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
        
        if frame_count % 30 == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Eye Detection Test")
            print("Press 'q' to quit, 's' to save frame")
            print("=" * 50)
        
        if len(faces) > 0:
            print(f"\nFrame {frame_count}: Face detected!")
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
                
                print(f"  Eyes found: {len(eyes)}")
                
                for i, (ex, ey, ew, eh) in enumerate(eyes):
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    eye_roi = cv2.resize(eye_roi, (32, 32))
                    eye_roi = eye_roi / 255.0
                    eye_roi = eye_roi.reshape(1, 32, 32, 1)
                    
                    try:
                        prediction = model.predict(eye_roi, verbose=0)
                        open_prob = prediction[0][0]
                        closed_prob = prediction[0][1]
                        label = "CLOSED" if np.argmax(prediction) == 1 else "OPEN"
                        
                        eye_position = "LEFT" if ex < w//2 else "RIGHT"
                        
                        print(f"    {eye_position} Eye: {label} (Open: {open_prob:.3f}, Closed: {closed_prob:.3f})")
                        
                        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                        cv2.putText(frame, f"{eye_position}: {label}", (x+ex, y+ey-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                    except Exception as e:
                        print(f"    Error predicting eye {i+1}: {e}")
                        continue
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            print(f"Frame {frame_count}: No face detected")
        
        cv2.imshow('Eye Detection Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'eye_test_frame_{frame_count}.jpg', frame)
            print(f"Saved frame as eye_test_frame_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest completed!")

if __name__ == "__main__":
    test_eye_detection() 