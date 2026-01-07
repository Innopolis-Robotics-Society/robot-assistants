from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import open3d as o3d


@dataclass
class DistanceStats:
    n: int
    mean: float
    median: float
    max: float
    p95: float
    p99: float
    over_threshold_ratio: float
    threshold: float


def pointcloud_distances(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
) -> np.ndarray:
    # returns list[float] in Open3D; convert to np
    d = src.compute_point_cloud_distance(tgt)
    return np.asarray(d, dtype=float)


def stats_from_distances(d: np.ndarray, threshold: float) -> DistanceStats:
    if d.size == 0:
        raise ValueError("Empty distance array")

    over = float(np.mean(d > threshold))
    return DistanceStats(
        n=int(d.size),
        mean=float(np.mean(d)),
        median=float(np.median(d)),
        max=float(np.max(d)),
        p95=float(np.percentile(d, 95)),
        p99=float(np.percentile(d, 99)),
        over_threshold_ratio=over,
        threshold=float(threshold),
    )
