from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import open3d as o3d


@dataclass
class RegistrationResult:
    transformation: np.ndarray  # 4x4
    fitness: float
    inlier_rmse: float
    method: str


def global_ransac_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
    ransac_n: int = 4,
    max_iter: int = 100000,
    max_validation: int = 1000,
) -> RegistrationResult:
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, max_validation),
    )

    T = np.asarray(result.transformation, dtype=float)
    return RegistrationResult(
        transformation=T,
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        method="ransac_fpfh",
    )


def refine_icp_point_to_plane(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    voxel_size: float,
    max_iter: int = 60,
) -> RegistrationResult:
    distance_threshold = voxel_size * 1.5

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance=distance_threshold,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=criteria,
    )

    T = np.asarray(result.transformation, dtype=float)
    return RegistrationResult(
        transformation=T,
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        method="icp_point_to_plane",
    )
