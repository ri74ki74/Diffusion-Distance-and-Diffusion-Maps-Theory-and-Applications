"""Generates a toroidal helix shaped dataset."""

import numpy as np


def generate_toroidal_helix_pointcloud(
    num_points=3000,
    R=2.5,
    r=1.0,
    k=6,
    turns=1.0,
    noise=0.02,
    seed=None,
) -> (np.ndarray, np.ndarray):  # type: ignore
    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, 2.0 * np.pi * turns, num_points, endpoint=False)

    x = (R + r * np.cos(k * t)) * np.cos(t)
    y = (R + r * np.cos(k * t)) * np.sin(t)
    z = r * np.sin(k * t)

    # Stack the coordinates into a single array
    X = np.column_stack((x, y, z))

    if noise > 0:
        X += rng.normal(scale=noise, size=X.shape)

    c = (t - t.min()) / (t.max() - t.min() + 1e-12)

    return X, c
