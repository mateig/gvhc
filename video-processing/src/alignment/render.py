"""Render gravity-aligned per-frame point cloud as a video with origin axes, applying the alignment transform to the saved camera pose."""

import numpy as np

from src.render import pointcloud


def render(
    video: str,
    points: str,
    output: str,
    fps: int,
    point_size: float,
    axis_length: float,
    camera: str,
    offset: list[float],
) -> None:
    data = np.load(points)
    pts = data["points"]
    E = data["extrinsic"].astype(np.float64)
    s = float(data["scale"])
    off = np.asarray(offset, dtype=np.float64)
    colors = np.load(video)["video"]

    R = E[:3, :3]
    t = E[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ (t + off / s)

    pointcloud.render(
        pts,
        colors,
        output,
        fps,
        point_size,
        camera,
        axis_length=axis_length,
        extrinsic_post=T_inv,
    )
