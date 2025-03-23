import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from mtcnn.mtcnn import MTCNN

# Load the trained model
model = tf.keras.models.load_model('deepfake_detection_model.h5')

# Function to draw bounding boxes and annotate predictions
def draw_bounding_boxes(frame, faces, predictions):
    for i, face in enumerate(faces):
        # Extract bounding box from detected face
        x, y, width, height = face['box']
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Annotate prediction (real or fake)
        prediction = predictions[i]
        prediction_text = "Real" if prediction < 0.4 else "Fake"
        confidence = prediction if prediction < 0.5 else 1 - prediction
        
        # Put text above bounding box
        cv2.putText(frame, f"{prediction_text} ({confidence:.2f})", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Function to visualize video with annotations and debug face detection
def visualize_predictions(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = MTCNN()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect faces in the frame
        faces = detector.detect_faces(frame)
        
        # Check if any faces are detected
        if len(faces) == 0:
            print("No faces detected in this frame.")
        else:
            print(f"{len(faces)} face(s) detected in this frame.")
        
        # Extract face regions for prediction if faces are detected
        face_regions = []
        for face in faces:
            x, y, width, height = face['box']
            face_crop = frame[y:y+height, x:x+width]
            face_crop = cv2.resize(face_crop, (299, 299))  
            face_crop = preprocess_input(np.expand_dims(face_crop, axis=0))
            face_regions.append(face_crop)
        
        if face_regions:
            face_regions = np.vstack(face_regions)
            # Make predictions using the model
            predictions = model.predict(face_regions)
            predictions = predictions.flatten()  # Flatten to match predictions with faces

            # Draw bounding boxes and annotate predictions
            annotated_frame = draw_bounding_boxes(frame, faces, predictions)
            
            # Show the frame with annotations
            cv2.imshow("Deepfake Detection", annotated_frame)
            
            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

# Usage
video_path = r"C:\Users\USER\Desktop\Projects\ML Project\Data set for project\Celeb-fake\id0_id1_0000.mp4"
visualize_predictions(video_path)
