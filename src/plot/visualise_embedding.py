"""Visualise embeddings."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def vis_embedding(
    Psi: np.ndarray,
    labels: np.ndarray,
    dim: int,
    highlight: bool = False,
    points: list[dict, dict, dict] | None = None,  # for highlighting
    cmap: str | None = None,
    color: str | None = None,
) -> None:
    if cmap is None:
        cmap = cm.get_cmap("coolwarm")

    scatter_kwargs = (
        dict(color=color) if color is not None else dict(c=labels, cmap=cmap)
    )

    target_points = []
    if points is not None:
        target_points = [(p.get("index"), p.get("label")) for p in points]

    if dim == 1:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(
            Psi[:, 0], np.zeros_like(Psi[:, 0]), s=12, alpha=0.5, **scatter_kwargs
        )

        if highlight and target_points:
            for idx, label_text in target_points:
                ax.scatter(
                    Psi[idx, 0], 0, s=20, edgecolor="black", facecolor="none", zorder=10
                )
                ax.annotate(
                    label_text,
                    xy=(Psi[idx, 0], 0),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"
                    ),
                )

        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    elif dim == 2:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_facecolor("#f5f5f5")
        ax.scatter(Psi[:, 0], Psi[:, 1], s=4, **scatter_kwargs)

        if highlight and target_points:
            for idx, label_text in target_points:
                ax.scatter(
                    Psi[idx, 0],
                    Psi[idx, 1],
                    s=20,
                    edgecolor="black",
                    facecolor="none",
                    zorder=10,
                )
                ax.annotate(
                    label_text,
                    xy=(Psi[idx, 0], Psi[idx, 1]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"
                    ),
                )

    elif dim == 3:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Psi[:, 0], Psi[:, 1], Psi[:, 2], s=4, **scatter_kwargs)

        if highlight and target_points:
            for idx, label_text in target_points:
                ax.scatter(
                    Psi[idx, 0],
                    Psi[idx, 1],
                    Psi[idx, 2],
                    s=40,
                    edgecolor="black",
                    facecolor="none",
                    zorder=10,
                )
                ax.text(
                    Psi[idx, 0],
                    Psi[idx, 1],
                    Psi[idx, 2] + 0.05,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"
                    ),
                )
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


def vis_embedding_pair(
    Psi_X: np.ndarray,
    Psi_Y: np.ndarray,
    dim: int,
    label_X: str = "Figure 1",
    label_Y: str = "Figure 2",
    color_X: str = "blue",
    color_Y: str = "green",
    cols: tuple[int, int] = (0, 1),
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """Overlay two diffusion map embeddings on the same plot."""
    c0, c1 = cols

    if dim == 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            Psi_X[:, c0], Psi_X[:, c1], s=2, color=color_X, label=label_X, alpha=0.7
        )
        ax.scatter(
            Psi_Y[:, c0], Psi_Y[:, c1], s=2, color=color_Y, label=label_Y, alpha=0.7
        )
        ax.legend(markerscale=5)

        all_data = np.vstack([Psi_X[:, [c0, c1]], Psi_Y[:, [c0, c1]]])
        ranges = np.ptp(all_data, axis=0)
        ax.set_aspect(ranges[1] / ranges[0])

    elif dim == 3:
        c2 = c1 + 1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        for Psi, color, label in [(Psi_X, color_X, label_X), (Psi_Y, color_Y, label_Y)]:
            ax.scatter(
                Psi[:, c0],
                Psi[:, c1],
                Psi[:, c2],
                s=2,
                color=color,
                label=label,
                alpha=0.7,
            )

        all_data = np.vstack([Psi_X[:, [c0, c1, c2]], Psi_Y[:, [c0, c1, c2]]])
        ranges = np.ptp(all_data, axis=0)
        ax.set_box_aspect(ranges / ranges.max())
        ax.legend(markerscale=5)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("lightgray")

    plt.tight_layout()
    plt.show()
