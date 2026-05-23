"""
Given an nx graph, simulate and highlight the diffusion paths.

Note: in the thesis I did not use end_node = None option,
the option is left in for generality.
"""

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np


def diffusion_paths(
    G: nx.Graph,
    start_node: int,
    end_node: int | None,
    n_paths: int = 10,
    max_steps: int = 10,
    color: str = "orange",
) -> nx.Graph:
    # Generate color gradient from dark to light:
    cmap = mcolors.LinearSegmentedColormap.from_list("diffusion", [color, "white"])

    for p in range(n_paths):
        current_node = start_node
        # potential path
        temp_path = []

        for step in range(max_steps):
            neighs = list(G.neighbors(current_node))
            if not neighs:
                break

            # Weighted random choice for next node based on edge weights
            weights = np.array([G[current_node][neigh]["weight"] for neigh in neighs])

            # Nummerical inaccuracies, renormalising
            probs = weights / weights.sum()

            next_node = np.random.choice(neighs, p=probs)

            # Recording the path
            temp_path.append((current_node, next_node, step))
            current_node = next_node

            # Checking if we reached the end node, if specified
            is_successful = (end_node is None) or (current_node == end_node)

            # Coloring
            if is_successful:
                # Assigne attributes to the edge with overlappign paths being thicker
                for u, v, step in temp_path:
                    # Color based on the step number (earlier steps are darker)
                    edge_color = mcolors.to_hex(cmap(step / max_steps))

                    G[u][v]["color"] = edge_color
                    if end_node is None:
                        # Sacling for meaningful visualisation
                        G[u][v]["path_count"] = G[u][v].get(
                            "path_count", 0
                        ) + 0.1 * n_paths ** (-1)
                    else:
                        G[u][v]["path_count"] = G[u][v].get("path_count", 0) + 1

            current_node = next_node

            if end_node is not None and current_node == end_node:
                break

    return G
