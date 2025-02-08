import cv2
import numpy as np
import tensorflow as tf
import collections
from mtcnn import MTCNN

modelFileName = 'bestmodel.keras'

print("Loading the model...")
model = tf.keras.models.load_model(modelFileName)

class_labels = ['Headtop', 'Helmet', 'Hoodie', 'No headwear']
model.summary()

detector = MTCNN()
print("Opening the webcam...")
cap = cv2.VideoCapture(0)
target_size = (224, 224)

# For smoothing predictions
predictions_buffer = collections.deque(maxlen=10)

print("Press 'q' to exit the video feed.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame. Exiting...")
            break

        faces = detector.detect_faces(frame)
        
        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)
            
            # Expand the region to capture headwear vertically
            top = max(0, y - int(height * 0.7))
            bottom = y + height

            # Expand horizontally by adding a margin, e.g., 20% of the width on each side
            margin_x = int(width * 0.2)
            start_x = max(0, x - margin_x)
            end_x = min(frame.shape[1], x + width + margin_x)

            head_region = frame[top:bottom, start_x:end_x]

            # Draw the expanded head region for debugging
            cv2.rectangle(frame, (start_x, top), (end_x, bottom), (0, 255, 255), 2)

            # Resize and preprocess
            head_region = cv2.resize(head_region, target_size)
            head_region = np.expand_dims(head_region, axis=0) / 255.0

            prediction = model.predict(head_region)
            
            # Smooth predictions using buffer:
            predictions_buffer.append(prediction[0])
            avg_prediction = np.mean(predictions_buffer, axis=0)
            class_index = np.argmax(avg_prediction)
            label = class_labels[class_index]

            # Draw face bounding box and label
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)

        cv2.imshow('Hat Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by the user. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released, video window closed.")