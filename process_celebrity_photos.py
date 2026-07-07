import os
from PIL import Image
import face_recognition

def preprocess_celebrity_photos(input_path, output_path):
    for celebrity_name in os.listdir(input_path):
        celebrity_dir = os.path.join(input_path, celebrity_name)
        output_celebrity_dir = os.path.join(output_path, celebrity_name)

        if not os.path.exists(output_celebrity_dir):
            os.makedirs(output_celebrity_dir)

        counter = 1
        for image_name in os.listdir(celebrity_dir):
            image_path = os.path.join(celebrity_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]

                output_image_name = f"{celebrity_name}_face{counter}.jpg"
                output_image_path = os.path.join(output_celebrity_dir, output_image_name)
                pil_image = Image.fromarray(face_image)
                pil_image.save(output_image_path)
                counter += 1

if __name__ == "__main__":
    input_path = "celebrities_dataset" 
    output_path = "processed_dataset"  
    preprocess_celebrity_photos(input_path, output_path)
