import cv2
import face_recognition
import os

# Step 1: Load the images and their corresponding labels
dataset_path = "Images"
known_people = {}

for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)

    if os.path.isdir(person_path):
        known_faces = []
        known_names = [person_folder]  # Use folder name as the label

        for filename in os.listdir(person_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(person_path, filename)
                known_image = face_recognition.load_image_file(image_path)
                known_face_encoding = face_recognition.face_encodings(known_image)

                if len(known_face_encoding) > 0:
                    known_faces.append(known_face_encoding[0])

        known_people[person_folder] = {'faces': known_faces, 'names': known_names}

# Step 2: Initialize the camera
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

while True:
    # Step 3: Capture frames from the camera
    ret, frame = cap.read()

    # Step 4: Find faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Step 5: Check if the face matches any known faces
        min_distance = float('inf')
        name = "Unknown"

        for person_name, data in known_people.items():
            face_distances = face_recognition.face_distance(data['faces'], face_encoding)
            min_face_distance = min(face_distances)

            if min_face_distance < min_distance:
                min_distance = min_face_distance
                if min_distance < 0.6:  # Adjust this threshold as needed
                    name = data['names'][0]

        # Step 6: Display the results on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"{name} ({(1 - min_distance) * 100:.2f}%)"
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
