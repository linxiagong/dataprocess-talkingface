import os
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
import tqdm

from .face_geometry import (PCF, canonical_metric_landmarks, get_metric_landmarks, procrustes_landmark_basis)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()
# points_idx = list(range(0,468)); points_idx[0:2] = points_idx[0:2:-1];


def get_face_pose(landmarks: np.array, pcf: PCF, camera_matrix, dist_coeff, debug: bool = False) -> np.array:
    landmarks = landmarks.T

    metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
    model_points = metric_landmarks[0:3, points_idx].T
    image_points = landmarks[0:2, points_idx].T * np.array([pcf.frame_width, pcf.frame_height])[None, :]

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points,
                                                                image_points,
                                                                camera_matrix,
                                                                dist_coeff,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_matrix = np.eye(4, dtype=float)
    rotation_matrix[:3, :3], _ = cv2.Rodrigues(rotation_vector)
    rotation_matrix[:3, 3] = translation_vector.T


    # nose points
    nose_point2D = (int(image_points[0][0]), int(image_points[0][1]))
    (nose_endpoint2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 25.0)]), rotation_vector,
                                                    translation_vector, camera_matrix, dist_coeff)
    nose_endpoint2D = (int(nose_endpoint2D[0][0][0]), int(nose_endpoint2D[0][0][1]))
    return rotation_matrix, nose_point2D, nose_endpoint2D


def transform_faceposes_to_cameraposes(face_poses: dict) -> dict:
    camera_poses = dict()
    for i, p in face_poses.items():
        p = np.array(p)
        rotation_matrix = np.eye(4, dtype=float)
        rot = p[:3, :3]
        rot_inv = np.transpose(rot, (1, 0))
        trans = p[:3, 3]
        trans_inv = np.matmul(rot_inv, trans)
        rotation_matrix[:3, :3] = rot_inv
        rotation_matrix[:3, 3] = trans_inv

        camera_poses[i] = rotation_matrix
    return camera_poses


def extract_face_mesh(img_list: list, debug_freq: int = 0, debug_dir: str = None) -> Tuple[list, list]:
    """Get mediapipe face mesh landmarks and face poses for a list of images."""
    mp_face_mesh = mp.solutions.face_mesh
    debug = (debug_freq > 0)
    if debug:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

    lms = dict()
    face_poses = dict()
    nodes_points = dict()

    images = []

    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=False,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        for idx, bgr_image in enumerate(tqdm.tqdm(img_list, desc='Calculate Mesh')):

            # image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            h, w = bgr_image.shape[:2]
            # pseudo camera internals
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                     dtype="double")
            dist_coeff = np.zeros((4, 1))
            pcf = PCF(near=1, far=10000, frame_height=h, frame_width=w, fy=camera_matrix[1, 1])

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:  # no face in image
                continue

            annotated_image = bgr_image.copy()
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                lms[idx] = landmarks
                rotation_matrix, nose_point2D, nose_endpoint2D = get_face_pose(landmarks=landmarks,
                                                                               pcf=pcf,
                                                                               camera_matrix=camera_matrix,
                                                                               dist_coeff=dist_coeff,
                                                                               debug=debug)
                face_poses[idx] = rotation_matrix
                nodes_points[idx] = (nose_point2D, nose_endpoint2D)

                # Draw the annotated images
                if debug and debug_dir is not None and idx % debug_freq == 0:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                    annotated_image = cv2.line(annotated_image, nose_point2D, nose_endpoint2D, (255, 0, 0), 2)
                    cv2.imwrite(os.path.join(debug_dir, f'{idx}_mesh.jpg'), annotated_image,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return lms, face_poses, nodes_points



class MediapipeFaceMesh:
    def __init__(self) -> None:
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               refine_landmarks=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    def extract_face_mesh(self, img_list: list, debug_freq: int = 0, debug_dir: str = None) -> Tuple[dict, dict]:
        """Get mediapipe face mesh landmarks and face poses for a list of images."""
        debug = (debug_freq > 0)

        lms = dict()
        face_poses = dict()
        nodes_points = dict()
        for idx, bgr_image in enumerate(tqdm.tqdm(img_list, desc='Calculate Mesh')):

            # image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            h, w = bgr_image.shape[:2]
            # pseudo camera internals
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                     dtype="double")
            dist_coeff = np.zeros((4, 1))
            pcf = PCF(near=1, far=10000, frame_height=h, frame_width=w, fy=camera_matrix[1, 1])

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:  # no face in image
                continue

            annotated_image = bgr_image.copy()
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                lms[idx] = landmarks

                rotation_matrix, nose_point2D, nose_endpoint2D = get_face_pose(landmarks=landmarks,
                                                                               pcf=pcf,
                                                                               camera_matrix=camera_matrix,
                                                                               dist_coeff=dist_coeff,
                                                                               debug=debug)
                face_poses[idx] = rotation_matrix
                nodes_points[idx] = (nose_point2D, nose_endpoint2D)

                # Draw the annotated images
                if debug_freq > 0 and debug_dir is not None and idx % debug_freq == 0:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                    annotated_image = cv2.line(annotated_image, nose_point2D, nose_endpoint2D, (255, 0, 0), 2)
                    cv2.imwrite(os.path.join(debug_dir, f'{idx}_mesh.jpg'), annotated_image,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return lms, face_poses, nodes_points
