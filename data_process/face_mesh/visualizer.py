"""
This script is a visualizer for Mediapipe face landmarks, aiming to minimize dependence on Mediapipe objects.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mediapipe as mp
import math
from typing import Union, Tuple

def transform_3d_points(points: np.ndarray, rotation_matrix:np.ndarray) -> np.ndarray:
    """
    Transforms a batch of 3D points using a 4x4 rotation matrix.

    Args:
        points: Batch of 3D points with shape (n, 3), where n is the number of points.
        rotation_matrix: 4x4 rotation matrix.

    Returns:
        Transformed points with shape (n, 3).
    """
    points = np.array(points)
    rotation_matrix = np.array(rotation_matrix)
    # Add homogeneous coordinates to the points
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the transformation
    transformed_points = np.dot(homogeneous_points, rotation_matrix.T)

    # Remove the homogeneous coordinates
    transformed_points = transformed_points[:, :3]

    return transformed_points

def load_pltplot_resize_and_corp(fig_path:str, h, w, corp_ratio:float=0.8):
    import cv2
    img = cv2.resize(cv2.imread(fig_path), (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    new_h = int(h*corp_ratio)
    h_start_idx = int((h-new_h)/2)
    new_w = int(w*corp_ratio)
    w_start_idx = int((w-new_w)/2)

    new_img = img[h_start_idx:h_start_idx+new_h, w_start_idx:w_start_idx+new_w, :]
    new_img = cv2.resize(new_img, (w, h))

    return new_img


def pltplot_landmarks(landmarks: np.ndarray,
                        rotation_matrix: np.ndarray = None,
                        fig_path:str = None,
                        desc:str='',
                        elevation: int = 10,
                        azimuth: int = 10):
    """
    Visualizes 3D landmarks of a human.

    Args:
        landmarks: 3D landmark coordinates with shape (n, 3),
                                   where n is the number of landmarks.
    Returns: rgb_image
    """
    if fig_path is not None:
        plt.ioff()

    # Create a new figure and set up a 3D subplot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)

    # Extract the x, y, and z coordinates from the landmarks
    mp_landmarks = np.array(landmarks)

    uniform_landmarks = np.zeros_like(mp_landmarks)
    uniform_landmarks[:, 0] = -mp_landmarks[:, 2]
    uniform_landmarks[:, 1] = mp_landmarks[:, 0]
    uniform_landmarks[:, 2] = -mp_landmarks[:, 1]
    if rotation_matrix is not None:
        uniform_landmarks = transform_3d_points(uniform_landmarks, rotation_matrix)

    x = uniform_landmarks[:, 0]
    y = uniform_landmarks[:, 1]
    z = uniform_landmarks[:, 2]

    # Plot the landmarks as points in 3D space
    # https://github.com/google/mediapipe/blob/25458138a99132dc8444b5f54270d1b2f5eeb242/mediapipe/python/solutions/drawing_utils.py#L254-L316
    # ax.scatter3D(-z, x, -y, c='r', marker='o')
    ax.scatter3D(x, y, z, c='r', marker='o')

    # Set labels and title
    desc = desc or '3D Landmarks'
    ax.set_title(desc, pad=-20)

    # Draws the connections
    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION
    for connection in connections:
        try:
            start_idx, end_idx = connection[:2]
            ax.plot3D(
                xs=[x[start_idx], x[end_idx]],
                ys=[y[start_idx], y[end_idx]],
                zs=[z[start_idx], z[end_idx]],
                c='r',
                linewidth=0.5,
            )
        except Exception as e:
            print(e)
    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS
    for connection in connections:
        try:
            start_idx, end_idx = connection[:2]
            ax.plot3D(
                xs=[x[start_idx], x[end_idx]],
                ys=[y[start_idx], y[end_idx]],
                zs=[z[start_idx], z[end_idx]],
                c='r',
                linewidth=0.5,
            )
        except Exception as e:
            print(e)

    if fig_path is not None:
        ax.set_axis_off()
        ax.axis('tight')
        fig.savefig(fig_path, format='png', bbox_inches='tight')
    else:
        plt.show()
    plt.ion()


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def draw_landmarks_on_image(image:np.ndarray):

    # https://github.com/google/mediapipe/blob/25458138a99132dc8444b5f54270d1b2f5eeb242/mediapipe/python/solutions/drawing_utils.py#L119
    pass