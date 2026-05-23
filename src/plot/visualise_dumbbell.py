"""Visualise dumbbell pointcloud."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection


def _add_distance_label(ax, p1, p2, colour="black", offset=0.06, fontsize=11):
    p1, p2 = np.asarray(p1), np.asarray(p2)
    mid = 0.5 * (p1 + p2)

    perp = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
    norm = np.linalg.norm(perp)
    perp = perp / norm if norm > 0 else np.zeros(2)

    ax.text(
        *(mid + offset * perp),
        f"{np.linalg.norm(p2 - p1):.2f}",
        color=colour,
        fontsize=fontsize,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
    )


def vis_dumbbell(
    X: np.ndarray,
    labels: np.ndarray,
    points: list | None = None,
    euclidean: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="Dark2", s=14, alpha=0.8)

    if points is not None:
        i, j, k = points

        ax.scatter(*X[[i, j]].T, s=30, color="red", zorder=5)
        ax.scatter(*X[k], s=30, color="orange", zorder=5)

        for idx, label, dy in [(i, "A", 0.05), (j, "B", 0.08), (k, "C", 0.05)]:
            ax.text(X[idx, 0], X[idx, 1] + dy, label, color="black", fontsize=13)

        if euclidean:
            for (a, b), colour in [((i, j), "red"), ((j, k), "orange")]:
                ax.plot(*X[[a, b]].T, color=colour, linestyle="--", linewidth=1.5)
                _add_distance_label(ax, X[a], X[b], colour=colour, offset=0.08)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0, labelpad=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()


def vis_diffusion_graph(G: nx.Graph, nodes_to_label: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    node_list = list(G.nodes())
    pos = np.array([G.nodes[n]["pos"] for n in node_list])  # (x,y)
    labels = np.array(
        [G.nodes[n]["label"] for n in node_list]
    )  # which cluster / bridge

    edges = list(G.edges())
    if edges:
        segments = np.array([[G.nodes[u]["pos"], G.nodes[v]["pos"]] for u, v in edges])
        weights = np.array([G.edges[u, v].get("weight") for u, v in edges])
        ax.add_collection(
            LineCollection(
                segments, colors="black", linewidths=weights, alpha=0.5, zorder=1
            )
        )

        for colour in {
            d.get("color") for _, _, d in G.edges(data=True) if "color" in d
        }:
            coloured = [
                (u, v, d) for u, v, d in G.edges(data=True) if d.get("color") == colour
            ]
            segments_c = np.array(
                [[G.nodes[u]["pos"], G.nodes[v]["pos"]] for u, v, _ in coloured]
            )
            widths = np.array([d.get("path_count", 1) for _, _, d in coloured]) * 0.07
            ax.add_collection(
                LineCollection(
                    segments_c, colors=colour, linewidths=widths, alpha=0.8, zorder=2
                )
            )

    ax.scatter(pos[:, 0], pos[:, 1], c=labels, cmap="Dark2", s=14, zorder=3)

    if nodes_to_label:
        for node_idx, (label_text, label_colour) in nodes_to_label.items():
            node_pos = G.nodes[node_idx]["pos"]
            ax.text(
                node_pos[0],
                node_pos[1] + 0.05,
                label_text,
                color=label_colour,
                fontsize=12,
                ha="center",
                va="center",
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0, labelpad=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()
