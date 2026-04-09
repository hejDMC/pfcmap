"""Helper to compute peak-to-trough metrics (Python port).

Provides:
- peak2trough(temp_wv, channel=0)
- peak2trough_scaled(temp_wv, scalings, channel=0)

Handles template arrays shaped (n_clusters, n_time, n_channels),
(n_clusters, n_time) or 1-D arrays.
"""
from __future__ import annotations
from typing import Union
import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


def peak2trough(temp_wv: ArrayLike, channel: int = 0) -> np.ndarray:
    """Compute peak-to-trough (max - min across time) per cluster.

    Parameters
    - temp_wv: array-like. Expected shapes:
        (n_clusters, n_time, n_channels)
        (n_clusters, n_time)
        (n_time,) or (n_samples,)
    - channel: channel index to use when `temp_wv` has a channel axis.

    Returns
    - 1-D numpy array of length n_clusters with peak-to-trough values.
    """
    arr = np.asarray(temp_wv)
    if arr.ndim == 3:
        if channel < 0 or channel >= arr.shape[2]:
            raise IndexError(f"channel index out of bounds: {channel}")
        data = arr[:, :, channel]
        return data.max(axis=1) - data.min(axis=1)
    elif arr.ndim == 2:
        data = arr
        return data.max(axis=1) - data.min(axis=1)
    elif arr.ndim == 1:
        return np.array([arr.max() - arr.min()])
    else:
        raise ValueError(f"Unsupported input dimensions: {arr.ndim}")


def peak2trough_scaled(temp_wv: ArrayLike, scalings: ArrayLike, channel: int = 0) -> np.ndarray:
    """Compute peak-to-trough and apply per-cluster scaling.

    `scalings` should be a 1-D vector of length equal to number of clusters.
    """
    p2t = peak2trough(temp_wv, channel=channel)
    s = np.asarray(scalings)
    if p2t.shape[0] != s.shape[0]:
        raise ValueError("Length of scalings must match number of units")
    return p2t * s


if __name__ == "__main__":
    # quick self-test
    a = np.array([[0.0, 1.0, -0.5], [2.0, 1.5, 2.5]])
    assert np.allclose(peak2trough(a), np.array([1.5, 1.0]))

    b = np.stack([a, a + 0.1], axis=2)  # shape (2,3,2)
    # use channel 0
    assert np.allclose(peak2trough(b, channel=0), peak2trough(a))

    s = np.array([2.0, 0.5])
    assert np.allclose(peak2trough_scaled(a, s), peak2trough(a) * s)

    print("compute_peak2Trough.py self-test passed")
