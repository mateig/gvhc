"""Render MoGe per-frame point cloud as a video from a saved camera pose."""

import numpy as np

from src.render import pointcloud


def render(
    video: str,
    points: str,
    output: str,
    fps: int,
    point_size: float,
    camera: str,
) -> None:
    pts = np.load(points)["points"]
    colors = np.load(video)["video"]
    pointcloud.render(pts, colors, output, fps, point_size, camera)
