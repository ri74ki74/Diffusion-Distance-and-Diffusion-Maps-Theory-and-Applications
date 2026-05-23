"""Generates a torus dataset."""

import numpy as np


def generate_torus_pointcloud(
    n=100, R=3.0, r=1.0, noise=0.0, seed=0
) -> (np.ndarray, np.ndarray):  # type: ignore
    rng = np.random.default_rng(seed)

    # Angles
    u = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(0, 2 * np.pi, size=n)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    X = np.column_stack([x, y, z])

    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)

    c = u / (2 * np.pi)
    return X, c
