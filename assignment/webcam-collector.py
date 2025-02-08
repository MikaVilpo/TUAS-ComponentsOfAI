import cv2
import os

cap = cv2.VideoCapture(0)
category = input("Enter category (headtop/helmet/hoodie/no_headwear): ")
save_path = f"dataset_in/{category}/"
os.makedirs(save_path, exist_ok=True)

count = 0
while count < 100:  # Capture 100 images
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (224, 224))
    cv2.imshow("Capturing", frame)
    cv2.imwrite(f"{save_path}/{count}.jpg", resized_frame)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
