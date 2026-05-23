"""Embed a graph into using diffusion maps."""

import numpy as np
from scipy.linalg import eigh


def embed(
    W: np.ndarray,
    t: int = 1,
    Markov: bool = False,
    right_evec: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:

    # Calculating the random walk normalisation matrix dW
    dW = W.sum(axis=1)

    # Calculating the symmetric similarity matrix M_s
    inv_sqrt_DW = 1.0 / np.sqrt(dW)

    M_s = W * inv_sqrt_DW[:, None] * inv_sqrt_DW

    # Eigendecomposition of M_s
    eval_M_s, evec_M_s = eigh(M_s)

    # Sorting the eigenpairs
    idx = np.argsort(eval_M_s)[::-1]
    eval_M_s = eval_M_s[idx]
    evec_M_s = evec_M_s[:, idx]

    # Calculating the right eigenvecotrs of M = D^{-1}W
    right_evec_M = evec_M_s * inv_sqrt_DW[:, None]

    # Calculating the embedding
    Psi_M = right_evec_M[:, 1:] * eval_M_s[1:] ** t

    # Calculating Markov transition matrix M = D^{-1}W for visualisation
    if Markov:
        M = W / dW[:, None]
        return Psi_M, eval_M_s, M
    if right_evec:
        return Psi_M, eval_M_s, right_evec_M
    else:
        return Psi_M, eval_M_s
