import os
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import LabelEncoder

def load_data(dataset_path):
    X, y = [], []

    for celebrity_name in os.listdir(dataset_path):
        celebrity_dir = os.path.join(dataset_path, celebrity_name)

        for image_name in os.listdir(celebrity_dir):
            image_path = os.path.join(celebrity_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)

            if face_encoding:
                X.append(face_encoding[0])  # Assuming one face per image
                y.append(celebrity_name)

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Support Vector Machine (SVM) classifier
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)

    # Evaluate the model
    accuracy = classifier.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    return classifier

def save_model(classifier, label_encoder, model_path="face_recognition_model.pkl", label_encoder_path="label_encoder.pkl"):
    joblib.dump(classifier, model_path)
    joblib.dump(label_encoder, label_encoder_path)

if __name__ == "__main__":
    dataset_path = "processed_dataset"
    X, y = load_data(dataset_path)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    face_recognition_model = train_model(X, y_encoded)
    save_model(face_recognition_model, label_encoder)
