"""Kernels for calculating the similarity between data points."""

import numpy as np


def gaussian_full_similarity(X: np.ndarray, epsilon: float) -> np.ndarray:
    # Calculating the squared norm for every vector
    sq_n = np.sum(X * X, axis=1)

    # Calcualting the matrix d2 with d2_ij being the Euclidean distance between i-th
    # and j-th datapoint
    d2 = sq_n[:, None] + sq_n[None, :] - 2 * (X @ X.T)

    # Calculating the Gaussian similarity matrix
    W = np.exp(-d2 / epsilon)

    return W
