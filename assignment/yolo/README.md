# YOLOv8n Model Training

This branch includes training the model using **YOLOv8n** for cap detection.

## Dataset
The dataset can be downloaded from [Roboflow](https://universe.roboflow.com/object-detection-cwcjo/caps-gi4un/dataset/2).

After downloading, place the images and labels in the appropriate folders:


## File Overview

- [`data.yaml`](data.yaml): YOLO dataset configuration specifying dataset paths.
- [`train_model.ipynb`](train_model.ipynb): Jupyter notebook for training the YOLOv8n model.
- [`model_webcam_test.ipynb`](model_webcam_test.ipynb): Notebook for testing real-time detection using a webcam.

Ensure that the dataset is correctly placed before training.
