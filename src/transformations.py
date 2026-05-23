"""Functions for non-rigid transformations of point clouds."""

import numpy as np


def shift_pointcloud(X: np.ndarray, shift_vector: np.ndarray) -> np.ndarray:
    """Shift a point cloud by a given vector."""
    return X + shift_vector


def rotate_pointcloud(X: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Rotate a point cloud using a given rotation matrix."""
    return X @ rotation_matrix.T
