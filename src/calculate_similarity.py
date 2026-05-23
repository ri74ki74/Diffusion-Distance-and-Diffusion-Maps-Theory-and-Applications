"""Calculate similarity between sets using pairwise Euclidean distance."""

import numpy as np


def calculate_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    return np.sum(np.linalg.norm(X - Y, axis=1))
