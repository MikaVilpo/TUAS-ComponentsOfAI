import cv2
import numpy as np
import tensorflow as tf
import collections
from mtcnn import MTCNN

def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale and new size while preserving aspect ratio
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(image, (new_w, new_h))

    # Create a new image and center the resized image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded

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
            
            # Expanded head bounding box (with horizontal margins)
            margin_x = int(width * 0.2)
            start_x = max(0, x - margin_x)
            end_x = min(frame.shape[1], x + width + margin_x)
            top = max(0, y - int(height * 0.7))
            bottom = y + height

            # Calculate current bbox dimensions
            bbox_width = end_x - start_x
            bbox_height = bottom - top
            max_side = max(bbox_width, bbox_height)

            # Compute the center of the bounding box
            center_x = start_x + bbox_width // 2
            center_y = top + bbox_height // 2

            # Define a new square region centered on the bbox center
            new_start_x = max(0, center_x - max_side // 2)
            new_top = max(0, center_y - max_side // 2)
            new_end_x = new_start_x + max_side
            new_bottom = new_top + max_side

            # Ensure the square region stays within the frame boundaries
            if new_end_x > frame.shape[1]:
                diff = new_end_x - frame.shape[1]
                new_start_x = max(0, new_start_x - diff)
                new_end_x = frame.shape[1]
            if new_bottom > frame.shape[0]:
                diff = new_bottom - frame.shape[0]
                new_top = max(0, new_top - diff)
                new_bottom = frame.shape[0]

            # Crop out the square region
            head_region = frame[new_top:new_bottom, new_start_x:new_end_x]

            # Draw the expanded head region for debugging
            cv2.rectangle(frame, (new_start_x, new_top), (new_end_x, new_bottom), (0, 255, 255), 2)

            # Resize and preprocess
            head_region = cv2.resize(head_region, target_size)
            head_region = np.expand_dims(head_region, axis=0) / 255.0

            prediction = model.predict(head_region)
            
            # Smooth predictions using buffer:
            predictions_buffer.append(prediction[0])
            avg_prediction = np.mean(predictions_buffer, axis=0)
            class_index = np.argmax(avg_prediction)

            # Calculate the confidence (as a percentage)
            confidence = avg_prediction[class_index] * 100

            # Create label with confidence, e.g. "Helmet: 85.34%"
            label = f"{class_labels[class_index]}: {confidence:.2f}%"

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