from __future__ import annotations

import numpy as np
import open3d as o3d


def mesh_distance_to_points(
    mesh: o3d.geometry.TriangleMesh,
    points_xyz: np.ndarray,
) -> np.ndarray:
    """
    Returns unsigned distance from each point to the mesh surface using Open3D RaycastingScene.
    points_xyz: (N,3) float
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3)")

    # Convert legacy mesh to tensor mesh
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    pts = o3d.core.Tensor(points_xyz.astype(np.float32))
    d = scene.compute_distance(pts)  # (N,)
    return d.numpy().astype(float)
