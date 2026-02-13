import face_recognition
import os
import numpy as np

known_encodings = []
known_names = []

def load_students():
    path = "students"

    for file in os.listdir(path):
        img_path = f"{path}/{file}"
        img = face_recognition.load_image_file(img_path)

        encodings = face_recognition.face_encodings(img)

        # Check if face found
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(file.split(".")[0])
        else:
            print(f"⚠ No face detected in {file}")

    print("Students Loaded:", known_names)


def recognize_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    present_students = []

    for face in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face)
        face_distances = face_recognition.face_distance(known_encodings, face)

        if len(face_distances) == 0:
            continue

        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]

            # Avoid duplicates
            if name not in present_students:
                present_students.append(name)

    return present_students
