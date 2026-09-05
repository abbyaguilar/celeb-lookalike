# ⭐ Celebrity Lookalike Detector

A computer vision and machine learning application that compares webcam face encodings against a custom celebrity dataset and returns the closest learned class match.

## Overview

The project covers the full workflow from image preparation through model inference:

1. Organize celebrity image samples
2. Extract facial encodings
3. Encode celebrity labels
4. Split training and test data
5. Train a linear SVM classifier
6. Evaluate the classifier on held-out data
7. Serialize the trained model
8. Run live webcam inference
9. Smooth predictions across recent frames for a more stable result

## Tech Stack

- Python
- OpenCV
- face_recognition
- scikit-learn
- SVM classification
- joblib
- Tkinter
- Pillow
- NumPy

## Machine Learning Pipeline

```text
Celebrity images
      |
      v
face_recognition encodings
      |
      v
LabelEncoder
      |
      v
80/20 train-test split
      |
      v
Linear SVM (probability=True)
      |
      v
Accuracy evaluation
      |
      v
joblib model serialization
      |
      v
Webcam inference
```

The training script uses `SVC(kernel="linear", probability=True)` and reports test-set accuracy before saving the classifier and label encoder.

## Real-Time Inference

The desktop application:

- Opens the webcam with OpenCV
- Detects faces in each frame
- Extracts a face encoding
- Calls `predict_proba` on the trained classifier
- Converts the winning encoded label back to a celebrity name
- Keeps a rolling history of predictions
- Uses the most frequent recent prediction to reduce frame-to-frame instability

## Project Structure

```text
celeb-lookalike/
├── app.py
├── process_celebrity_photos.py
├── train_face_recognition_model.py
├── face_recognition_model.pkl
├── label_encoder.pkl
└── requirements.txt
```

The source image dataset is not included in the repository.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare images:

```bash
python process_celebrity_photos.py
```

Train the classifier:

```bash
python train_face_recognition_model.py
```

Run the desktop app:

```bash
python app.py
```

## Project Status

**Educational / experimental computer vision project.**

The output represents classifier-based visual similarity within the supplied dataset and should not be interpreted as identity verification.

## Creator

Built by **Abigail Aguilar**
