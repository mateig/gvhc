"""Poisson mesh of static points aggregated per-frame within a fixed xy radius of the dynamic mask's median xy position."""

import os

import cv2
import numpy as np
import open3d as o3d


def run(
    alignment: str,
    masks: str,
    output: str,
    dilate_ksize: int,
    radius: float,
    voxel_size: float,
    outlier_nb_neighbors: int,
    outlier_std_ratio: float,
    normal_radius: float,
    normal_max_nn: int,
    poisson_depth: int,
    density_quantile: float,
    smooth_iters: int,
) -> None:
    # (T, H, W, 3) float32 gravity-aligned, NaN where invalid
    pts = np.load(alignment)["points"]
    # sam3 (T, N, H, W) bool collapsed to (T, H, W)
    dynamic = np.load(masks)["masks"].any(axis=1)

    T = dynamic.shape[0]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    valid = np.isfinite(pts).all(axis=-1)
    keep = np.zeros_like(valid)

    for t in range(T):
        s_sel = (cv2.dilate(dynamic[t].astype(np.uint8), k) == 0) & valid[t]
        d_sel = dynamic[t] & valid[t]
        if not d_sel.any() or not s_sel.any():
            continue
        center_xy = np.median(pts[t][d_sel, :2], axis=0)
        d_xy = np.linalg.norm(pts[t][..., :2] - center_xy, axis=-1)
        keep[t] = s_sel & (d_xy < radius)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[keep].astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio
    )

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))
    mesh.remove_unreferenced_vertices()

    cluster_idx, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_idx = np.asarray(cluster_idx)
    cluster_sizes = np.asarray(cluster_sizes)
    if len(cluster_sizes) > 0:
        mesh.remove_triangles_by_mask(cluster_idx != cluster_sizes.argmax())
        mesh.remove_unreferenced_vertices()

    if smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)

    mesh.compute_triangle_normals()

    os.makedirs(os.path.dirname(output), exist_ok=True)
    o3d.io.write_triangle_mesh(output, mesh)
