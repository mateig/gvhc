"""Render per-frame colored point clouds to video via Open3D."""

import os

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


def render(
    points: np.ndarray,
    colors: np.ndarray,
    output: str,
    fps: int,
    point_size: float,
    camera: str,
    axis_length: float = 0.0,
    extrinsic_post: np.ndarray | None = None,
) -> None:
    T, H, W, _ = points.shape
    params = o3d.io.read_pinhole_camera_parameters(camera)
    if extrinsic_post is not None:
        params.extrinsic = np.asarray(params.extrinsic) @ extrinsic_post

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=params.intrinsic.width, height=params.intrinsic.height)
    vis.get_render_option().point_size = point_size

    if axis_length > 0:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length))

    pcd = o3d.geometry.PointCloud()

    os.makedirs(os.path.dirname(output), exist_ok=True)
    writer = None

    for t in tqdm(range(T)):
        m = np.isfinite(points[t]).all(axis=-1)
        pcd.points = o3d.utility.Vector3dVector(points[t][m].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors[t][m].astype(np.float64) / 255.0)
        if len(pcd.points) > 30:
            filt, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
            filt, _ = filt.remove_radius_outlier(nb_points=16, radius=0.05)
            pcd.points = o3d.utility.Vector3dVector(np.asarray(filt.points))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(filt.colors))

        if t == 0:
            vis.add_geometry(pcd)
        else:
            vis.update_geometry(pcd)

        vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

        frame = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    vis.destroy_window()
