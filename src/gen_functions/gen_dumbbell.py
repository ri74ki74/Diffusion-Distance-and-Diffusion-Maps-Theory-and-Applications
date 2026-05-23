"""
Generate a dumbbell-shaped dataset.

Note: the sqrt in _sample_disc is used to enfure uniform distribution of
points on the disc.
"""

import numpy as np


def _sample_disc(n, centre=(0.0, 0.0), radius=1.0, rng=None):
    """
    Sample n points uniformly from a 2D disc.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Angle and radius for polar coordinates
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))

    # Convert polar coordinates to Cartesian coordinates
    x = centre[0] + r * np.cos(theta)
    y = centre[1] + r * np.sin(theta)

    return np.column_stack([x, y])


def _sample_bridge(n, x_start, x_end, half_width=0.12, rng=None):
    """
    Sample n points uniformly from a thin horizontal rectangle,
    which acts as the bottleneck.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = rng.uniform(x_start, x_end, size=n)
    y = rng.uniform(-half_width, half_width, size=n)

    return np.column_stack([x, y])


def generate_dumbbell_pointcloud(
    n_left=400,
    n_right=400,
    n_bridge=120,
    cluster_radius=1.0,
    bridge_half_width=0.5,
    noise=0.02,
    seed=0,
):
    """
    Generate a 2D dumbbell-shaped point cloud with a narrow bottleneck.
    """
    rng = np.random.default_rng(seed)

    # Place the two discs so that the bridge connects them
    left_centre = (-1.5, 0.0)
    right_centre = (1.5, 0.0)

    X_left = _sample_disc(n_left, centre=left_centre, radius=cluster_radius, rng=rng)
    X_right = _sample_disc(n_right, centre=right_centre, radius=cluster_radius, rng=rng)

    x_start = left_centre[0] + cluster_radius
    x_end = right_centre[0] - cluster_radius

    X_bridge = _sample_bridge(
        n_bridge, x_start=x_start, x_end=x_end, half_width=bridge_half_width, rng=rng
    )

    X = np.vstack([X_left, X_bridge, X_right])

    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)

    labels = np.concatenate(
        [
            np.zeros(n_left, dtype=int),
            np.ones(n_bridge, dtype=int),
            2 * np.ones(n_right, dtype=int),
        ]
    )

    return X, labels
