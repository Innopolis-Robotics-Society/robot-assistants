from __future__ import annotations

import open3d as o3d


def mesh_to_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    n_points: int,
    method: str = "poisson",
    init_factor: int = 5,
) -> o3d.geometry.PointCloud:
    if n_points <= 0:
        raise ValueError("n_points must be > 0")

    method = method.lower().strip()
    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=n_points,
            init_factor=init_factor,
        )
    elif method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    else:
        raise ValueError("method must be 'poisson' or 'uniform'")

    return pcd
