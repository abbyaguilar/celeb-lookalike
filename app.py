import cv2
import face_recognition
import joblib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from collections import Counter


MODEL_PATH = "face_recognition_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# How many frames to analyze before updating the result
ANALYSIS_FRAMES = 20

# Number of recent results to keep for stability
HISTORY_SIZE = 40


def load_model():
    classifier = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return classifier, label_encoder


def pretty_name(name):
    """Convert Cierra_Ramirez -> Cierra Ramirez."""
    return name.replace("_", " ")


class CelebrityLookalikeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Celebrity Lookalike")
        self.root.geometry("900x760")
        self.root.configure(bg="#111111")

        self.classifier, self.label_encoder = load_model()

        self.prediction_history = []
        self.frame_count = 0
        self.current_result = None
        self.running = True

        # -----------------------------
        # Title
        # -----------------------------
        title = tk.Label(
            root,
            text="✨ Celebrity Lookalike",
            font=("Helvetica", 26, "bold"),
            fg="white",
            bg="#111111"
        )
        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            root,
            text="Look into the camera to discover your closest celebrity match",
            font=("Helvetica", 11),
            fg="#bbbbbb",
            bg="#111111"
        )
        subtitle.pack(pady=(0, 15))

        # -----------------------------
        # Camera
        # -----------------------------
        self.camera_label = tk.Label(
            root,
            bg="#222222",
            bd=0
        )
        self.camera_label.pack()

        # -----------------------------
        # Result section
        # -----------------------------
        self.status_label = tk.Label(
            root,
            text="Analyzing your face...",
            font=("Helvetica", 12),
            fg="#bbbbbb",
            bg="#111111"
        )
        self.status_label.pack(pady=(15, 5))

        self.result_label = tk.Label(
            root,
            text="",
            font=("Helvetica", 24, "bold"),
            fg="white",
            bg="#111111"
        )
        self.result_label.pack()

        self.score_label = tk.Label(
            root,
            text="",
            font=("Helvetica", 13),
            fg="#cccccc",
            bg="#111111"
        )
        self.score_label.pack(pady=(5, 15))

        # -----------------------------
        # Quit button
        # -----------------------------
        quit_button = tk.Button(
            root,
            text="Quit",
            command=self.close,
            font=("Helvetica", 11),
            padx=25,
            pady=8
        )
        quit_button.pack(pady=(0, 20))

        # -----------------------------
        # Camera
        # -----------------------------
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.status_label.config(
                text="Unable to access your camera."
            )
            return

        self.update_frame()

    def analyze_frame(self, frame):
        """Detect faces and return the predicted celebrity."""

        face_locations = face_recognition.face_locations(frame)

        if not face_locations:
            return frame, None

        for (top, right, bottom, left) in face_locations:

            # Draw face box
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (255, 255, 255),
                2
            )

            # Get face encoding
            encodings = face_recognition.face_encodings(
                frame,
                [(top, right, bottom, left)]
            )

            if not encodings:
                continue

            face_encoding = encodings[0]

            try:
                predictions = self.classifier.predict_proba(
                    [face_encoding]
                )

                best_index = int(np.argmax(predictions[0]))

                predicted_label = self.label_encoder.inverse_transform(
                    [best_index]
                )[0]

                return frame, predicted_label

            except Exception:
                return frame, None

        return frame, None

    def calculate_result(self):
        """Determine the most consistent celebrity prediction."""

        if not self.prediction_history:
            return None, 0

        counts = Counter(self.prediction_history)

        winner, winner_count = counts.most_common(1)[0]

        total = len(self.prediction_history)

        consistency = (winner_count / total) * 100

        return winner, consistency

    def update_result(self):
        """Update the UI with a stable result."""

        if len(self.prediction_history) < ANALYSIS_FRAMES:
            progress = int(
                (len(self.prediction_history) / ANALYSIS_FRAMES) * 100
            )

            self.status_label.config(
                text=f"Analyzing your features... {progress}%"
            )

            return

        winner, score = self.calculate_result()

        if winner:

            formatted_name = pretty_name(winner)

            self.current_result = winner

            self.status_label.config(
                text="✨ Your closest celebrity match"
            )

            self.result_label.config(
                text=formatted_name
            )

            self.score_label.config(
                text=f"Match consistency: {score:.0f}%"
            )

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.status_label.config(
                text="Unable to read from camera."
            )
            return

        # Mirror the camera like a normal selfie camera
        frame = cv2.flip(frame, 1)

        frame_result, prediction = self.analyze_frame(frame)

        if prediction:
            self.prediction_history.append(prediction)

            # Keep history from growing forever
            if len(self.prediction_history) > HISTORY_SIZE:
                self.prediction_history.pop(0)

        self.frame_count += 1

        # Update result periodically rather than every frame
        if self.frame_count % 5 == 0:
            self.update_result()

        # Convert OpenCV image to Tkinter image
        frame_rgb = cv2.cvtColor(
            frame_result,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame_rgb)

        # Resize camera display
        image.thumbnail((850, 600))

        imgtk = ImageTk.PhotoImage(image=image)

        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

        # Schedule next frame
        self.root.after(15, self.update_frame)

    def close(self):
        self.running = False

        if self.cap:
            self.cap.release()

        self.root.destroy()


def main():
    root = tk.Tk()

    app = CelebrityLookalikeApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()