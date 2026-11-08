"""Clustering of items by the groups they share: Jaccard similarity, the Markov
Cluster Algorithm (MCL), weighted modularity and the adjusted Rand index.

Written for the disease-trajectory method (Jensen et al. 2014), which clusters
diagnoses by the trajectories they appear in with MCL on Jaccard similarity,
and usable for any similarity graph.
"""

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["jaccard", "mcl", "modularity", "adjusted_rand"]


def jaccard(member: ArrayLike) -> np.ndarray:
    """Jaccard similarity between the rows of a boolean items x groups matrix.

    J[a, b] = groups holding both / groups holding either. An item in no group
    has similarity 0 with everything, itself included.
    """
    m = np.asarray(member, dtype=float)
    both = m @ m.T
    size = m.sum(axis=1)
    either = size[:, None] + size[None, :] - both
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.where(either > 0, both / either, 0.0)
    return j


def _normalise(m: np.ndarray) -> np.ndarray:
    s = m.sum(axis=0)
    s[s == 0] = 1.0
    return m / s


def mcl(
    w: ArrayLike,
    inflation: float = 2.0,
    expansion: int = 2,
    prune: float = 1e-6,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    """Markov Cluster Algorithm (van Dongen 2000) on a symmetric, non-negative
    similarity matrix. Returns one cluster label per item.

    Self-loops are set to each item's largest similarity, the usual choice that
    keeps the flow from oscillating. Each iteration expands (matrix power) and
    inflates (elementwise power, then column normalisation). At convergence
    each item joins the attractor that holds most of its mass. Labels are
    0, 1, ... by decreasing cluster size, ties by the smallest member, so the
    result does not depend on the order of the items beyond those ties.
    Higher inflation gives more, smaller clusters.
    """
    a = np.array(w, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("mcl needs a square similarity matrix")
    if (a < 0).any():
        raise ValueError("mcl needs non-negative similarities")
    n = len(a)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    np.fill_diagonal(a, 0.0)
    loops = a.max(axis=0)
    loops[loops == 0] = 1.0
    a[np.diag_indices(n)] = loops
    m = _normalise(a)
    for _ in range(max_iter):
        prev = m
        m = np.linalg.matrix_power(m, int(expansion))
        m = _normalise(np.power(m, inflation))
        m[m < prune] = 0.0
        m = _normalise(m)
        if np.abs(m - prev).max() < tol:
            break
    attractor = np.argmax(m, axis=0)  # the row holding most of each column's mass
    # Items sharing an attractor, or attracted to each other's attractor, form one cluster.
    root = attractor.copy()
    for _ in range(n):
        nxt = attractor[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    _, raw = np.unique(root, return_inverse=True)
    return _relabel(raw)


def _relabel(raw: np.ndarray) -> np.ndarray:
    """Labels by decreasing size, ties by the smallest member."""
    ids = np.unique(raw)
    size = np.array([(raw == i).sum() for i in ids])
    first = np.array([np.flatnonzero(raw == i)[0] for i in ids])
    order = np.lexsort((first, -size))
    new = np.empty(len(ids), dtype=np.int64)
    new[order] = np.arange(len(ids))
    return new[np.searchsorted(ids, raw)]


def modularity(w: ArrayLike, labels: ArrayLike) -> float:
    """Weighted Newman modularity of a partition of a symmetric graph (self-loops
    ignored): the share of edge weight inside clusters minus its expectation
    when edges are rewired keeping each item's total weight."""
    a = np.array(w, dtype=float)
    np.fill_diagonal(a, 0.0)
    labels = np.asarray(labels)
    two_m = a.sum()
    if two_m == 0:
        return 0.0
    k = a.sum(axis=1)
    same = labels[:, None] == labels[None, :]
    return float(((a - np.outer(k, k) / two_m) * same).sum() / two_m)


def adjusted_rand(a: ArrayLike, b: ArrayLike) -> float:
    """Adjusted Rand index of two labellings of the same items (1 identical,
    about 0 for independent labellings). NaN when there are fewer than two items."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n < 2:
        return float("nan")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)

    def c2(x):
        return (x * (x - 1) / 2).sum()

    index = c2(table)
    ra, rb = c2(table.sum(axis=1)), c2(table.sum(axis=0))
    expected = ra * rb / (n * (n - 1) / 2)
    top = (ra + rb) / 2
    if top == expected:
        return 1.0
    return float((index - expected) / (top - expected))
