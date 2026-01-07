from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import open3d as o3d
import matplotlib.cm as cm


@dataclass
class HeatmapParams:
    vmin: float
    vmax: float
    cmap: str = "turbo"


def colors_from_distances(d: np.ndarray, hp: HeatmapParams) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    d_clip = np.clip(d, hp.vmin, hp.vmax)
    denom = max(hp.vmax - hp.vmin, 1e-12)
    x = (d_clip - hp.vmin) / denom

    cmap = cm.get_cmap(hp.cmap)
    rgba = cmap(x)  # Nx4
    rgb = rgba[:, :3].astype(np.float64)
    return rgb


def colorize_point_cloud_by_distance(
    pcd: o3d.geometry.PointCloud,
    d: np.ndarray,
    hp: HeatmapParams,
) -> o3d.geometry.PointCloud:
    if len(pcd.points) != int(np.asarray(d).shape[0]):
        raise ValueError("pcd points count != distances length")

    colored = o3d.geometry.PointCloud(pcd)  # copy
    colored.colors = o3d.utility.Vector3dVector(colors_from_distances(d, hp))
    return colored
