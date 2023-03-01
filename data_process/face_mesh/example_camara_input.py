from webcamsource import WebcamSource
import numpy as np
import mediapipe as mp
import cv2

from face_geometry import get_metric_landmarks, PCF, canonical_metric_landmarks, procrustes_landmark_basis

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

points_idx = [33,263,61,291,199]
points_idx = points_idx + [key for (key,val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()
# points_idx = list(range(0,468)); points_idx[0:2] = points_idx[0:2:-1];

frame_height, frame_width, channels = (720, 1280, 3)

# pseudo camera internals
focal_length = frame_width
center = (frame_width/2, frame_height/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

dist_coeff = np.zeros((4, 1))

def main():
    source = WebcamSource()

    pcf = PCF(near=1, far=10000, frame_height=frame_height, frame_width=frame_width, fy=camera_matrix[1, 1])
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        for idx, (frame, frame_rgb) in enumerate(source):
            results = face_mesh.process(frame)
            multi_face_landmarks = results.multi_face_landmarks

            # Draw the face mesh annotations on the image.
            frame.flags.writeable = True
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                # mp_drawing.draw_landmarks(
                #     image=frame,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_iris_connections_style())

            if multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                # print(landmarks.shape)
                landmarks = landmarks.T

                metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)

                model_points = metric_landmarks[0:3, points_idx].T

                image_points = landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None, :]

                success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                            dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)
                # _, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeff)
                # print(f'---rotation_vector:---\n{rotation_vector} {rotation_vector.shape}, \n---translation_vector:---\n{translation_vector}')

                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 25.0)]), rotation_vector,
                                                                 translation_vector, camera_matrix, dist_coeff)
                for ii in points_idx:  # range(landmarks.shape[1]):
                    pos = np.array((frame_width * landmarks[0, ii], frame_height * landmarks[1, ii])).astype(np.int32)
                    frame = cv2.circle(frame, tuple(pos), 1, (0, 255, 0), -1)

                # ---- Project face points ----
                for p in metric_landmarks.T:  # range(landmarks.shape[1]):
                    pos, _ = cv2.projectPoints(p, rotation_vector, translation_vector, camera_matrix, dist_coeff)
                    pos = (int(pos[0][0][0]), int(pos[0][0][1]))
                    frame = cv2.circle(frame, pos, 1, (0, 255, 0), -1)

                # ---- Canonical face points ----
                for p in metric_landmarks.T:  # range(landmarks.shape[1]):
                    pos, _ = cv2.projectPoints(p, np.array([[np.pi, 0, 0]]).T.astype(float),
                                               np.array([[0, 0, 80]]).T.astype(float), camera_matrix, dist_coeff)
                    pos = (int(pos[0][0][0]) - 300, int(pos[0][0][1]) - 300)
                    frame = cv2.circle(frame, pos, 1, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                frame = cv2.line(frame, p1, p2, (255, 0, 0), 2)
            source.show(frame)
            # if idx > 10:
            #     break
if __name__ == '__main__':
    main()