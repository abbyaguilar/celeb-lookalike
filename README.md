# ⭐ Celebrity Lookalike Detector

A computer vision application that uses facial recognition and machine learning to compare webcam images with a custom-trained celebrity image dataset and find the closest visual similarity match.

This project was created as an exploration of real-time face detection, image processing, machine learning workflows, and integrating AI models into an interactive web application.

---

# 🌟 Overview

The Celebrity Lookalike Detector uses a webcam feed to:

* Capture live images
* Detect faces in real time
* Process facial features
* Compare results against a trained image dataset
* Return the closest celebrity similarity match

The goal of this project was to explore how machine learning and computer vision can be combined to create an interactive user experience.

---

# ✨ Features

## 📷 Real-Time Webcam Processing

The application provides:

* Live webcam streaming through Flask
* Real-time face detection
* Continuous image processing
* Interactive browser-based experience

---

## 🧠 Machine Learning Pipeline

The project includes a complete workflow:

1. Collect and organize image samples
2. Preprocess images and extract facial information
3. Prepare training data
4. Train a similarity matching model
5. Capture webcam input
6. Compare facial features
7. Display the closest visual match

---

# 🖼️ Dataset Preparation

For experimentation, a custom image dataset was created and organized locally.

The preprocessing pipeline:

* Loads collected image samples
* Detects and processes faces
* Creates structured training data
* Prepares images for model training

The dataset itself is not included in this repository.

Users can provide their own image collection following the expected folder structure.

---

# 🛠️ Technology Stack

## Backend

* Python
* Flask

Used for:

* Web application server
* Routing
* Connecting the machine learning pipeline to the user interface

---

## Machine Learning / Computer Vision

* Face recognition
* Image processing
* Facial feature comparison
* Model training workflow

Used for:

* Face detection
* Feature extraction
* Similarity matching

---

## Frontend

* HTML
* CSS
* JavaScript

Used for:

* User interface
* Webcam interaction
* Displaying results

---

# 📂 Project Structure

```
celeb-lookalike/

├── app.py
├── train_face_recognition_model.py
├── process_celebrity_photos.py
├── requirements.txt
├── app/
├── celebrities_dataset/
├── processed_dataset/
└── face_recognition_models/
```
Note:

The dataset directory is excluded from the public repository.

The following directories contain local data and generated files and are excluded from the public repository:

- `celebrities_dataset/` — custom image collection used for experimentation
- `processed_dataset/` — processed facial data generated during preprocessing
- `face_recognition_models/` — trained model files generated during training

These files can be recreated by running the preprocessing and training scripts locally.
---

# 🚀 Running Locally

## Prerequisites

* Python 3.x
* Git

---

## Installation

Clone the repository:

```bash
git clone [<repository-url>](https://github.com/abbyaguilar/celeb-lookalike.git)
```

Navigate into the project directory:

```bash
cd celeb-lookalike
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🖼️ Preparing Images

Run the preprocessing script:

```bash
python process_celebrity_photos.py
```

This processes the image collection and prepares facial data for training.

---

# 🧠 Training the Model

Train the look-alike model:

```bash
python train_model.py
```

This creates the trained model used for facial similarity comparisons.

---

# ▶️ Running the Application

Start the Flask server:

```bash
python app.py
```

Open the application:

```
http://127.0.0.1:5000/
```

The application will access the webcam, process detected faces, and display the closest similarity result.

---

# 🧠 What I Learned

This project helped me explore:

* Computer vision concepts
* Image preprocessing workflows
* Machine learning model training
* Real-time application development
* Integrating ML models into web applications
* Connecting backend services with AI-powered features

---

# 🔒 Repository Notes

The following files are intentionally excluded:

* Image datasets
* Local training data
* Environment files
* Large generated model files

These files can be recreated locally by following the preprocessing and training steps.

---

# ⚠️ Disclaimer

This project is for educational and experimental purposes.

The results represent algorithmic visual similarity and should not be interpreted as a definitive identification or statement about a person's identity.

---

# 👩‍💻 Creator

**Abigail Aguilar**

Exploring the intersection of software engineering, machine learning, and interactive applications.
