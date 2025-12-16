import numpy as np
from typing import List, Tuple

def topk_matrix(a, k: int) -> List[Tuple[int, int]]:
    """
    返回矩阵 a 中最大的 k 个元素及其坐标 [(row, col), ...]
    结果按 value 从大到小排序。
    """
    A = np.asarray(a)                     # 支持 list/np.array
    if A.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
    if k <= 0:
        return []
    n = A.size
    k = min(k, n)

    flat = A.ravel()
    # 先找出 top-k 的索引（未排序）
    idx_part = np.argpartition(flat, n - k)[-k:]
    # 再按值从大到小排序
    idx_sorted = idx_part[np.argsort(flat[idx_part])[::-1]]
    # 转二维坐标
    rows, cols = np.unravel_index(idx_sorted, A.shape)

    return [(int(r), int(c)) for r, c in zip(rows, cols)]

def maxp_matrix(a, percent: float) -> List[Tuple[int, int]]:
    """
    返回矩阵 a 中所有满足 A[i,j] > max(A) * percent 的元素及其坐标。
    结果为 [(row, col), ...]，按 value 降序排列。
    说明：使用 nanmax 忽略 NaN。
    """
    A = np.asarray(a)
    if A.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
    if not np.isfinite(percent):
        raise ValueError("percent 必须是有限数")

    maxv = np.nanmax(A)
    thr = maxv * percent

    # 过滤出有效数字且大于阈值
    valid = np.isfinite(A)
    mask = valid & (A > thr)

    if not np.any(mask):
        return []

    rows, cols = np.where(mask)
    order = np.argsort(A[rows, cols])[::-1]  # 按值降序

    rows, cols = rows[order], cols[order]
    return [(int(r), int(c)) for r, c in zip(rows, cols)]