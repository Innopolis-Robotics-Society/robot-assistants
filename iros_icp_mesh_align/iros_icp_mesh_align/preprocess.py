from __future__ import annotations

import open3d as o3d


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    remove_outliers: bool = False,
) -> o3d.geometry.PointCloud:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    p = pcd.voxel_down_sample(voxel_size)

    # Normals for point-to-plane ICP and FPFH
    radius_normal = voxel_size * 2.0
    p.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    p.normalize_normals()

    if remove_outliers:
        p, _ = p.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    return p


def compute_fpfh(
    pcd_down: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.pipelines.registration.Feature:
    radius_feature = voxel_size * 5.0
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
