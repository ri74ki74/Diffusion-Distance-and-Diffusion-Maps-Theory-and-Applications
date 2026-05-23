"""Construct an nx graph based on a point cloud."""

import networkx as nx
import numpy as np


def pointcloud_to_graph(X: np.ndarray, W: np.ndarray, labels: np.ndarray) -> nx.Graph:
    n = X.shape[0]
    G = nx.Graph()

    for i in range(n):
        G.add_node(i, pos=X[i], label=labels[i])

    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                G.add_edge(i, j, weight=W[i, j])

    return G
