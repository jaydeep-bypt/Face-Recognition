import cv2
import face_recognition
import os

# Step 1: Load the images and their corresponding labels
dataset_path = "Image"
known_faces = []
known_names = []

for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(dataset_path, filename)
        known_image = face_recognition.load_image_file(image_path)
        known_face_encoding = face_recognition.face_encodings(known_image)[0]
        known_faces.append(known_face_encoding)
        known_names.append(name)

# Step 2: Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Step 3: Capture frames from the camera
    ret, frame = cap.read()

    # Step 4: Find faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Step 5: Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        percentage_match = 0

        if True in matches:
            first_match_index = matches.index(True)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            percentage_match = (1 - face_distances[first_match_index]) * 100

            if percentage_match > 70:
                name = known_names[first_match_index]
                # Additional information about the match
                match_info = f"Matched with {name} ({percentage_match:.2f}%)"
                print(match_info)

        # Step 6: Display the results on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"{name} ({percentage_match:.2f}%)"
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
