import numpy as np
from typing import List, Tuple

def topk_matrix(a, k: int) -> List[Tuple[int, int]]:
    """
    Return coordinates of the top-k elements in matrix `a` as [(row, col), ...].
    Results are sorted by value descending.
    """
    A = np.asarray(a)                     # support list/np.array
    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    if k <= 0:
        return []
    n = A.size
    k = min(k, n)

    flat = A.ravel()
    # find top-k indices (unsorted)
    idx_part = np.argpartition(flat, n - k)[-k:]
    # sort by value descending
    idx_sorted = idx_part[np.argsort(flat[idx_part])[::-1]]
    # convert to 2D coordinates
    rows, cols = np.unravel_index(idx_sorted, A.shape)

    return [(int(r), int(c)) for r, c in zip(rows, cols)]

def maxp_matrix(a, percent: float) -> List[Tuple[int, int]]:
    """
    Return coordinates where A[i,j] > max(A) * percent.
    Results are [(row, col), ...], sorted by value descending.
    NaN values are ignored via nanmax.
    """
    A = np.asarray(a)
    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    if not np.isfinite(percent):
        raise ValueError("percent must be finite")

    maxv = np.nanmax(A)
    thr = maxv * percent

    # filter finite values above threshold
    valid = np.isfinite(A)
    mask = valid & (A > thr)

    if not np.any(mask):
        return []

    rows, cols = np.where(mask)
    order = np.argsort(A[rows, cols])[::-1]  # sort by value descending

    rows, cols = rows[order], cols[order]
    return [(int(r), int(c)) for r, c in zip(rows, cols)]