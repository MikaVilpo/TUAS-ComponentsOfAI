import cv2
import numpy as np
import tensorflow as tf
import collections

# Load the saved model
print ("Loading the model...")
model = tf.keras.models.load_model('EfficientNetB0.h5')

# Class labels for predictions
class_labels = ['Headtop', 'Helmet', 'Hoodie', 'No headwear']

# Print the model summary
model.summary()

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

        # Preprocess the frame
        resized_frame = cv2.resize(frame, target_size)  # Resize to match model input
        img_array = np.expand_dims(resized_frame, axis=0) / 255.0  # Normalize and add batch dimension

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        class_index = np.argmax(predictions[0])
        prediction_label = class_labels[class_index]
        confidence = predictions[0][class_index] * 100

        # Add prediction to the buffer
        predictions_buffer.append(class_index)

        # Smooth predictions using majority voting
        smoothed_prediction = max(set(predictions_buffer), key=predictions_buffer.count)
        smoothed_label = class_labels[smoothed_prediction]

        # Display the prediction on the frame
        cv2.putText(frame, f"{smoothed_label} ({confidence:.2f}%)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (75, 75, 75), 2)

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
#    cv2.destroyAllWindows()
    print("Resources released, video window closed.")