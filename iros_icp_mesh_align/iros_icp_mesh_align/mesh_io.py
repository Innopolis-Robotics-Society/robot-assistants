from __future__ import annotations

from pathlib import Path
import open3d as o3d


def load_mesh(path: str | Path) -> o3d.geometry.TriangleMesh:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    mesh = o3d.io.read_triangle_mesh(str(p))
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {p}")

    # Clean-up (safe defaults)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # Normals (useful for point-to-plane ICP later after sampling)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    return mesh


def mesh_extent(mesh: o3d.geometry.TriangleMesh) -> tuple[float, float, float]:
    aabb = mesh.get_axis_aligned_bounding_box()
    ext = aabb.get_extent()
    return float(ext[0]), float(ext[1]), float(ext[2])
