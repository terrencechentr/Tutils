
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import math

def _to_numpy(x):
    if hasattr(x, 'detach'):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def clip_numpy(data, lo_percent=None, hi_percent=None):
    data = _to_numpy(data)
    if lo_percent is not None:
        lo = np.nanpercentile(data, lo_percent)
        data = np.clip(data, lo, data.max())
    if hi_percent is not None:
        hi = np.nanpercentile(data, hi_percent)
        data = np.clip(data, data.min(), hi)
    return data

def _plot(
    fig: Figure,
    fig_name: str,
    out_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    save / show a Figure, and close it safely in the end.

    Args:
        fig: Matplotlib Figure object.
        fig_name: file name (e.g. 'histogram_100.png').
        out_dir: save directory; if None, not saved, only show when show=True.
        show: whether to show on screen.

    Returns:
        save path (Path), None if not saved.
    """
    # normalize save path
    save_path: Optional[Path] = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / fig_name

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return save_path

def summarize_multi_stats(arr2d: np.ndarray):
    rows, cols = arr2d.shape
    stats = {
        "layer": [], "mean": [], "std": [], "min": [], "p1": [], "p25": [], "p50": [], "p75": [], "p99": [], "max": []
    }
    for i in range(rows):
        row = arr2d[i]
        stats["layer"].append(i)
        stats["mean"].append(float(np.nanmean(row)))
        stats["std"].append(float(np.nanstd(row)))
        stats["min"].append(float(np.nanmin(row)))
        stats["p1"].append(float(np.nanpercentile(row, 1)))
        stats["p25"].append(float(np.nanpercentile(row, 25)))
        stats["p50"].append(float(np.nanpercentile(row, 50)))
        stats["p75"].append(float(np.nanpercentile(row, 75)))
        stats["p99"].append(float(np.nanpercentile(row, 99)))
        stats["max"].append(float(np.nanmax(row)))
    return pd.DataFrame(stats)


def plot_box(
    data,
    out_dir=None,
    show=True):
    """
    Args:
        data: (distribution, values)
    visualize boxplot for each distribution
    """
    arr = _to_numpy(data)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    assert arr.ndim == 2

    num_distribution, num_values = arr.shape

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=240)

    ax.boxplot(arr.T, showmeans=True)

    ax.set_title(f"Boxplot ({num_distribution} x {num_values})")
    ax.set_xlabel("Distribution index (0-based)")
    ax.set_ylabel("Values")

    fig.tight_layout()

    path = _plot(fig, "boxplot.png", out_dir=out_dir, show=show)
    return path

def plot_violin(
    data,
    out_dir=None,
    show=True):
    """
    Args:
        data: (distribution, values)
    """
    arr = _to_numpy(data)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    assert arr.ndim == 2

    num_distribution, num_values = arr.shape

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=240)

    ax.violinplot(arr.T, showmeans=False, showmedians=True)

    ax.set_title(f"Violin Plot ({num_distribution} x {num_values})")
    ax.set_xlabel("Distribution index (0-based)")
    ax.set_ylabel("Values")

    fig.tight_layout()

    path = _plot(fig, "violin.png", out_dir=out_dir, show=show)
    return path


def plot_histogram(
    data,
    bins=100,
    density=True,
    out_dir=None,
    show=True):
    """
    Args:
        data: (distribution, values) 形状的数据；一行一个 distribution
        bins: histogram bins
        density: normalize or not
    """
    arr = _to_numpy(data)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    assert arr.ndim == 2
    num_distribution, num_values = arr.shape

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        num_distribution, 1,
        figsize=(14, 3 * num_distribution),
        dpi=240,
        sharex=True
    )

    if num_distribution == 1:
        ax = [ax]

    # pre compute global y max for unified scale
    ys = []
    for i in range(num_distribution):
        c, _ = np.histogram(arr[i], bins=bins, density=density)
        ys.append(c.max())
    ymax = max(ys)

    for i in range(num_distribution):
        ax[i].hist(arr[i], bins=bins, density=density)
        ax[i].set_ylabel("Density" if density else "Count")
        ax[i].set_title(f"Histogram [{i}] ({num_values} values)")
        ax[i].set_ylim(0, ymax)

        ax[-1].set_xlabel("Value")

    fig.tight_layout()

    path = _plot(fig, "histogram.png", out_dir=out_dir, show=show)
    return path

def plot_multi_features(
    data,
    rows_per_feature=16,
    out_dir=None,
    fname="multi_features.png",
    show=True
):
    """
    Args:
        data: (layers, features)
        rows_per_feature: how many row chunks per layer
    """
    arr = _to_numpy(data)
    layers, feats = arr.shape

    # --- split to (layers*rows_per_feature, chunk) with NaN padding ---
    chunk = max(1, math.ceil(feats / rows_per_feature))
    parts = []
    for L in range(layers):
        r = arr[L]
        for i in range(rows_per_feature):
            seg = r[i * chunk:(i + 1) * chunk]
            if seg.size < chunk:
                pad = np.full(chunk - seg.size, np.nan, dtype=r.dtype)
                seg = np.concatenate([seg, pad])
            parts.append(seg)

    M = np.stack(parts, axis=0)               # (rows, cols)
    M = np.ma.masked_invalid(M)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = M.shape

    # --- 自动 figsize：按行列比设置，保证像素方形 ---
    # 让绘图区的宽高比 ≈ cols:rows；colorbar 额外留出 ~0.6 英寸
    base_width = 10.0                          # 你可以改这个基准宽度
    fig_w = base_width
    fig_h = base_width * (rows / cols)
    cbar_extra = 0.6                           # colorbar 预留（英寸）
    fig_w += cbar_extra

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200, constrained_layout=True)

    im = ax.imshow(M, interpolation='nearest')
    ax.set_xlabel("Neuron segment")
    ax.set_ylabel("Layer x segment rows")

    # 保证每个单元是正方形（关键行）
    ax.set_aspect('equal', adjustable='box')

    # layer 边界线
    for L in range(layers):
        y = L * rows_per_feature
        ax.axhline(y - 0.5, linewidth=0.2)

    # NaN 显示为空白
    cmap = im.get_cmap().copy()
    cmap.set_bad(alpha=0.0)
    im.set_cmap(cmap)

    # colorbar（用较小的占比，减少对主图宽度的影响）
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)

    path = _plot(fig, fname, out_dir=out_dir, show=show)
    return path

if __name__ == "__main__":
    import math
    import torch
    data = torch.randn(10, 1000)
    plot_multi_features(clip_numpy(data, lo_percent=0.1), out_dir='.', show=True, rows_per_feature=16)
