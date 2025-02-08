import cv2
import numpy as np
import tensorflow as tf
import collections
from mtcnn import MTCNN

modelFileName = 'best_model.keras'

# Load the saved model
print ("Loading the model...")
model = tf.keras.models.load_model(modelFileName)

# Class labels for predictions
class_labels = ['Headtop', 'Helmet', 'Hoodie', 'No headwear']

# Print the model summary
model.summary()

# Load the MTCNN face detector
detector = MTCNN()

# Open the webcam
print("Opening the webcam...")
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Define the target size for the images
target_size = (224, 224)

# Initialize a buffer for smoothing predictions
predictions_buffer = collections.deque(maxlen=10)

print("Press 'q' to exit the video feed.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame. Exiting...")
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        
        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)  # Ensure no negative values
            
            # Crop region above face (approximate location for hat)
            head_region = frame[max(0, y - int(height * 0.5)):y + height, x:x + width]

            # Resize to 224x224 for model
            head_region = cv2.resize(head_region, (224, 224))
            head_region = np.expand_dims(head_region, axis=0) / 255.0  # Normalize

            # Predict headwear type
            prediction = model.predict(head_region)
            class_index = np.argmax(prediction)
            label = class_labels[class_index]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Hat Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by the user. Exiting...")

finally:
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released, video window closed.")