# asl_demo.py
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
# Load model
model = tf.keras.models.load_model('asl_model (1).h5')
labels = ['A', 'B', 'C', 'D', 'E']
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Open webcam
cap = cv2.VideoCapture(0) # Use 0 for default camera
while cap.isOpened():
   ret, frame = cap.read()
   if not ret:
    print("Failed to grab frame")
    break
 # Convert to RGB for MediaPipe
   rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   results = hands.process(rgb_frame)
   if results.multi_hand_landmarks:
     # Get first detected hand
     hand_landmarks = results.multi_hand_landmarks[0]
     # Extract hand coordinates
     x_coords = [lm.x for lm in hand_landmarks.landmark]
     y_coords = [lm.y for lm in hand_landmarks.landmark]
     #Calculate bounding box
     x_min = int(min(x_coords) * frame.shape[1])
     x_max = int(max(x_coords) * frame.shape[1])
     y_min = int(min(y_coords) * frame.shape[0])
     y_max = int(max(y_coords) * frame.shape[0])

     # Crop and resize hand region
     hand_roi = frame[y_min:y_max, x_min:x_max]
     if hand_roi.size != 0: # Check for empty frames
       hand_roi = cv2.resize(hand_roi, (224, 224))
       hand_roi = np.expand_dims(hand_roi, axis=0) / 255.0
    
       # Predict
       prediction = model.predict(hand_roi)
       label = labels[np.argmax(prediction)]
 
       # Display prediction
       cv2.putText(frame, f"Prediction: {label}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
     
    # Show frame
   cv2.imshow('ASL Translator', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()