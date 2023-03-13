import cv2
import mediapipe as mp
import glob
import numpy as np
import os
import pickle


# Hyperparameters Setting
input_dir = "../data/3d/head_imgs"
output_pkl = "test.pkl"


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
files = glob.glob(os.path.join(input_dir, "*.jpg"))
print(f"Total images: #{len(files)} ...")


def get_landmarks(face_landmarks):
    lms = []
    while True:
        try:
            lm = face_landmarks.landmark.pop()
            lms.append(np.array([lm.x, lm.y, lm.z]))
        except Exception as exc:
            break
    return np.array(lms)


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
images = []
lms_list = []
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(files):
        if idx % 100 == 0:
            print(f"processing {idx} ...")
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            lms_list.append(get_landmarks(face_landmarks))
            # Draw the annotated images (uncomment the following lines if u wanna see annotated images)
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
            # mp_drawing.draw_landmarks(
            #   image=annotated_image,
            #   landmark_list=face_landmarks,
            #   connections=mp_face_mesh.FACEMESH_IRISES,
            #   landmark_drawing_spec=None,
            #   connection_drawing_spec=mp_drawing_styles
            #   .get_default_face_mesh_iris_connections_style())
        images.append(annotated_image)


with open(output_pkl, "wb") as f:
    pickle.dump(lms_list, f)
print(f"Done! Saved in {output_pkl}.")