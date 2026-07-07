import cv2
import face_recognition
import joblib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

def load_model(model_path="face_recognition_model.pkl", label_encoder_path="label_encoder.pkl"):
    classifier = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    return classifier, label_encoder

def detect_face_and_match(frame, classifier, label_encoder, label_var):
    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    # If faces are found, draw rectangles around them and try to recognize
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Extract face encoding for recognition
        face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])

        if face_encoding:
            # Predict the label and confidence
            predictions = classifier.predict_proba(face_encoding)
            best_match_index = np.argmax(predictions, axis=1)
            confidence = predictions[np.arange(len(best_match_index)), best_match_index][0]  # Get the first element

            # Print recognized labels for debugging
            recognized_labels = label_encoder.inverse_transform(best_match_index)
            print(f"Recognized Labels: {recognized_labels} (Confidence: {confidence:.2f})")

            # Display the recognized label on the frame
            if confidence > 0.10:  # Adjust confidence threshold as needed
                label_text = f"We are confident you look most like {recognized_labels[0]} ({confidence:.2f})"
                cv2.putText(frame, label_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

def main():
    classifier, label_encoder = load_model()

    # Create the main window
    root = tk.Tk()
    root.title("Face Recognition")

    # Create a label variable to update the displayed text
    label_var = tk.StringVar()
    label_var.set("Look into the camera...")
    label = tk.Label(root, textvariable=label_var, font=("Helvetica", 12))
    label.pack(pady=10)

    # Open a connection to the webcam (you may need to adjust the index)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform face detection and recognition
        frame_result = detect_face_and_match(frame, classifier, label_encoder, label_var)

        # Convert the frame to RGB format for display in tkinter
        frame_rgb = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the recognized celebrity
        label.imgtk = imgtk
        label.configure(image=imgtk)

        # Update the tkinter window
        root.update()

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    tk.mainloop()
