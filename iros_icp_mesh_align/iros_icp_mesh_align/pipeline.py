from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d
import matplotlib

matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt

from .mesh_io import load_mesh, mesh_extent
from .sampling import mesh_to_point_cloud
from .preprocess import preprocess_point_cloud, compute_fpfh
from .registration import global_ransac_registration, refine_icp_point_to_plane
from .distance import pointcloud_distances, stats_from_distances
from .heatmap import HeatmapParams, colorize_point_cloud_by_distance, colors_from_distances
from .raycast_distance import mesh_distance_to_points


LogFn = Callable[[str], None]
WarnFn = Callable[[str], None]


@dataclass
class PipelineOutputs:
    out_dir: Path
    T_scan_to_ref: np.ndarray
    report: dict


def _diag_from_extent(ext: tuple[float, float, float]) -> float:
    v = np.array(ext, dtype=float)
    return float(np.linalg.norm(v))


def _auto_voxel_from_ref_diag(ref_diag: float, voxel_ratio: float, voxel_min: float) -> float:
    v = max(ref_diag * float(voxel_ratio), float(voxel_min))
    return float(v)


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _plot_hist(path: Path, d: np.ndarray, title: str) -> None:
    plt.figure()
    plt.hist(d, bins=80)
    plt.title(title)
    plt.xlabel("Distance (same units as input meshes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_alignment_pipeline(
    *,
    ref_path: str | Path,
    scan_path: str | Path,
    out_dir: str | Path,
    # sizing / mismatch policy (NO scaling)
    size_ratio_fail_gt: float = 3.0,
    size_ratio_fail_lt: float = 0.33,
    size_ratio_warn_gt: float = 1.25,
    size_ratio_warn_lt: float = 0.80,
    # sampling / registration
    n_coarse: int = 30000,
    n_fine: int = 200000,
    sample_method: str = "poisson",
    icp_max_iter: int = 60,
    remove_outliers: bool = False,
    # voxel selection
    auto_voxel: bool = True,
    voxel: float = 1.0,
    voxel_ratio: float = 0.005,  # 0.5% of bbox diagonal
    voxel_min: float = 1e-3,
    # defect metrics / visuals
    dist_thresh: float = 2.0,
    heat_vmin: float = 0.0,
    heat_vmax: float = 5.0,
    write_hist: bool = True,
    write_point_heatmaps: bool = True,
    write_mesh_heatmaps: bool = True,
    # logging callbacks (optional)
    log: Optional[LogFn] = None,
    warn: Optional[WarnFn] = None,
) -> PipelineOutputs:
    def _log(msg: str) -> None:
        if log:
            log(msg)

    def _warn(msg: str) -> None:
        if warn:
            warn(msg)
        else:
            _log(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "status": "started",
        "inputs": {"ref": str(Path(ref_path)), "scan": str(Path(scan_path))},
        "units_note": "All distances are in the same units as the input meshes (often mm for STL).",
        "params": {
            "n_coarse": int(n_coarse),
            "n_fine": int(n_fine),
            "sample_method": str(sample_method),
            "icp_max_iter": int(icp_max_iter),
            "remove_outliers": bool(remove_outliers),
            "auto_voxel": bool(auto_voxel),
            "voxel": float(voxel),
            "voxel_ratio": float(voxel_ratio),
            "voxel_min": float(voxel_min),
            "dist_thresh": float(dist_thresh),
            "heat_vmin": float(heat_vmin),
            "heat_vmax": float(heat_vmax),
            "size_ratio_fail_gt": float(size_ratio_fail_gt),
            "size_ratio_fail_lt": float(size_ratio_fail_lt),
            "size_ratio_warn_gt": float(size_ratio_warn_gt),
            "size_ratio_warn_lt": float(size_ratio_warn_lt),
        },
    }

    t0 = time.perf_counter()
    _log("Loading meshes...")
    ref_mesh = load_mesh(ref_path)
    scan_mesh = load_mesh(scan_path)

    ref_ext = mesh_extent(ref_mesh)
    scan_ext = mesh_extent(scan_mesh)
    ref_diag = _diag_from_extent(ref_ext)
    scan_diag = _diag_from_extent(scan_ext)
    diag_ratio = (scan_diag / ref_diag) if ref_diag > 1e-12 else float("inf")

    report["extents"] = {"ref": ref_ext, "scan": scan_ext}
    report["bbox_diag"] = {"ref": ref_diag, "scan": scan_diag, "scan_over_ref": diag_ratio}

    # strict: no scaling; fail early on huge mismatch
    if diag_ratio > size_ratio_fail_gt or diag_ratio < size_ratio_fail_lt:
        msg = (
            "Size mismatch is too large. This usually means wrong units (e.g., mm vs m) "
            "or incompatible models. No auto-scaling is applied."
        )
        report["status"] = "failed_size_mismatch"
        report["error"] = msg
        _save_json(out_dir / "report.json", report)
        raise RuntimeError(msg)

    size_warning = (diag_ratio > size_ratio_warn_gt) or (diag_ratio < size_ratio_warn_lt)
    report["size_mismatch_warning"] = bool(size_warning)
    if size_warning:
        _warn("Noticeable size ratio difference detected (continuing without scaling). Please verify units/model integrity.")

    # auto voxel
    if auto_voxel:
        voxel_used = _auto_voxel_from_ref_diag(ref_diag, voxel_ratio, voxel_min)
    else:
        voxel_used = float(voxel)
    report["voxel_used"] = voxel_used

    _log(f"Sampling point clouds (coarse={n_coarse}, fine={n_fine})...")
    ref_pcd_coarse = mesh_to_point_cloud(ref_mesh, n_coarse, method=sample_method)
    scan_pcd_coarse = mesh_to_point_cloud(scan_mesh, n_coarse, method=sample_method)
    ref_pcd_fine = mesh_to_point_cloud(ref_mesh, n_fine, method=sample_method)
    scan_pcd_fine = mesh_to_point_cloud(scan_mesh, n_fine, method=sample_method)

    _log(f"Preprocessing (voxel={voxel_used:g})...")
    ref_down = preprocess_point_cloud(ref_pcd_coarse, voxel_used, remove_outliers=remove_outliers)
    scan_down = preprocess_point_cloud(scan_pcd_coarse, voxel_used, remove_outliers=remove_outliers)

    _log("Computing FPFH features...")
    ref_fpfh = compute_fpfh(ref_down, voxel_used)
    scan_fpfh = compute_fpfh(scan_down, voxel_used)

    _log("Global registration (RANSAC+FPFH)...")
    ransac_res = global_ransac_registration(
        source_down=scan_down,
        target_down=ref_down,
        source_fpfh=scan_fpfh,
        target_fpfh=ref_fpfh,
        voxel_size=voxel_used,
    )
    report["ransac"] = {"fitness": ransac_res.fitness, "inlier_rmse": ransac_res.inlier_rmse}

    _log("Refining alignment (ICP point-to-plane)...")
    icp_res = refine_icp_point_to_plane(
        source=scan_down,
        target=ref_down,
        init_T=ransac_res.transformation,
        voxel_size=voxel_used,
        max_iter=icp_max_iter,
    )
    report["icp"] = {"fitness": icp_res.fitness, "inlier_rmse": icp_res.inlier_rmse}

    T = icp_res.transformation

    # aligned fine scan cloud
    scan_fine_aligned = o3d.geometry.PointCloud(scan_pcd_fine)
    scan_fine_aligned.transform(T)

    # aligned scan mesh
    scan_mesh_aligned = o3d.geometry.TriangleMesh(scan_mesh)
    scan_mesh_aligned.transform(T)

    _log("Computing point-cloud distances (scan->ref and ref->scan)...")
    d_scan_to_ref = pointcloud_distances(scan_fine_aligned, ref_pcd_fine)
    d_ref_to_scan = pointcloud_distances(ref_pcd_fine, scan_fine_aligned)
    stats_s2r = stats_from_distances(d_scan_to_ref, threshold=dist_thresh)
    stats_r2s = stats_from_distances(d_ref_to_scan, threshold=dist_thresh)
    report["dist_scan_to_ref_pcd"] = stats_s2r.__dict__
    report["dist_ref_to_scan_pcd"] = stats_r2s.__dict__

    hp = HeatmapParams(vmin=heat_vmin, vmax=heat_vmax, cmap="turbo")

    if write_point_heatmaps:
        _log("Building point-cloud heatmaps...")
        heat_s2r = colorize_point_cloud_by_distance(scan_fine_aligned, d_scan_to_ref, hp)
        heat_r2s = colorize_point_cloud_by_distance(ref_pcd_fine, d_ref_to_scan, hp)
        o3d.io.write_point_cloud(str(out_dir / "heat_scan_to_ref.ply"), heat_s2r)
        o3d.io.write_point_cloud(str(out_dir / "heat_ref_to_scan.ply"), heat_r2s)

    _log("Computing mesh-to-mesh distances via RaycastingScene (vertex->surface)...")
    # true distances: vertices of A to surface of B
    d_scanmesh_to_refmesh = mesh_distance_to_points(ref_mesh, np.asarray(scan_mesh_aligned.vertices))
    d_refmesh_to_scanmesh = mesh_distance_to_points(scan_mesh_aligned, np.asarray(ref_mesh.vertices))

    report["dist_scan_to_ref_mesh_vertices"] = stats_from_distances(d_scanmesh_to_refmesh, threshold=dist_thresh).__dict__
    report["dist_ref_to_scan_mesh_vertices"] = stats_from_distances(d_refmesh_to_scanmesh, threshold=dist_thresh).__dict__

    if write_mesh_heatmaps:
        _log("Colorizing meshes using raycast distances...")
        scan_mesh_heat = o3d.geometry.TriangleMesh(scan_mesh_aligned)
        scan_mesh_heat.vertex_colors = o3d.utility.Vector3dVector(colors_from_distances(d_scanmesh_to_refmesh, hp))

        ref_mesh_heat = o3d.geometry.TriangleMesh(ref_mesh)
        ref_mesh_heat.vertex_colors = o3d.utility.Vector3dVector(colors_from_distances(d_refmesh_to_scanmesh, hp))

        o3d.io.write_triangle_mesh(str(out_dir / "scan_mesh_aligned.ply"), scan_mesh_aligned, write_vertex_colors=False)
        o3d.io.write_triangle_mesh(str(out_dir / "scan_mesh_heat_raycast.ply"), scan_mesh_heat, write_vertex_colors=True)
        o3d.io.write_triangle_mesh(str(out_dir / "ref_mesh_heat_raycast.ply"), ref_mesh_heat, write_vertex_colors=True)

    if write_hist:
        _log("Saving histograms...")
        _plot_hist(out_dir / "hist_scan_to_ref.png", d_scan_to_ref, "Distances: aligned scan -> reference (PCD)")
        _plot_hist(out_dir / "hist_ref_to_scan.png", d_ref_to_scan, "Distances: reference -> aligned scan (PCD)")

    _log("Saving final outputs...")
    (out_dir / "T_scan_to_ref.json").write_text(json.dumps({"T": T.tolist()}, indent=2))
    report["status"] = "ok"
    report["timing_s"] = float(time.perf_counter() - t0)
    _save_json(out_dir / "report.json", report)

    # always save aligned fine scan
    o3d.io.write_point_cloud(str(out_dir / "scan_fine_aligned.ply"), scan_fine_aligned)

    return PipelineOutputs(out_dir=out_dir, T_scan_to_ref=T, report=report)
