"""Visualise raw dataset."""

import io

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from matplotlib import cm


def _set_axes_equal(ax: plt.Axes, X: np.ndarray, pad: float = 0.05) -> None:
    """Make a 3D plot have equal scale on x, y, z axes."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    centres = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2 * (1 + pad)

    ax.set_xlim(centres[0] - radius, centres[0] + radius)
    ax.set_ylim(centres[1] - radius, centres[1] + radius)
    ax.set_zlim(centres[2] - radius, centres[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def _style_axis_3d(
    ax, hide_panes: bool, hide_lines: bool, clear_ticks: tuple[str, ...]
) -> None:
    """Hide panes, axis lines, and ticks on a 3D axis."""
    if hide_panes:
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("none")
    if hide_lines:
        ax.xaxis.line.set_visible(False)
        ax.yaxis.line.set_visible(False)
        ax.zaxis.line.set_visible(False)
    for axis_name in clear_ticks:
        getattr(ax, f"set_{axis_name}ticks")([])


def vis_pointcloud(
    X: np.ndarray,
    c: np.ndarray | None = None,
    color: str | None = None,
    figsize: tuple[int, int] = (18, 9),
    grid: bool = False,
    axis: bool | str = False,
    elev: int | None = None,
    azim: int | None = None,
    cmap: str | None = None,
    depthshade: bool = True,
    equal_aspect: bool = False,
) -> None:
    fig = plt.figure(figsize=figsize)
    dim = X.shape[1]

    if cmap is None:
        cmap = cm.get_cmap("coolwarm")

    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    scatter_kwargs = (
        dict(c=c, cmap=cmap, s=4) if color is None else dict(color=color, s=4)
    )
    if dim == 3:
        scatter_kwargs["depthshade"] = depthshade

    if dim == 1:
        ax.scatter(X[:, 0], np.zeros_like(X[:, 0]), **scatter_kwargs)
        ax.set_yticks([])
    elif dim == 2:
        ax.scatter(X[:, 0], X[:, 1], **scatter_kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("y", rotation=0, labelpad=10)
        ax.set_aspect("equal")
    elif dim == 3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], **scatter_kwargs)
        if equal_aspect:
            _set_axes_equal(ax, X)
        if elev is not None or azim is not None:
            ax.view_init(elev=elev, azim=azim)

    if dim == 3:
        if axis == "minimal":
            _style_axis_3d(
                ax, hide_panes=True, hide_lines=True, clear_ticks=("x", "y", "z")
            )
            ax.grid(False)
        elif axis is True:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        elif not axis:
            _style_axis_3d(
                ax, hide_panes=True, hide_lines=False, clear_ticks=("x", "y", "z")
            )
    elif not axis:
        ax.axis("off")

    if grid:
        ax.grid(True, alpha=0.25) if dim > 1 else ax.grid(True, axis="x", alpha=0.25)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches=None)
    plt.close(fig)
    buf.seek(0)
    display(Image(buf.read()))
