"""Gravity-align the sfm world-frame scene using three keypoints from frame 0."""

import os

import numpy as np


def compute(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, known_distance: float
) -> tuple[np.ndarray, float]:
    origin = 0.5 * (p1 + p2)
    up = p3 - origin
    measured = float(np.linalg.norm(up))

    z = up / measured
    y_raw = p1 - origin
    y = y_raw - np.dot(y_raw, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)

    R = np.stack([x, y, z], axis=0)
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3, 3] = -R @ origin
    return E, known_distance / measured


def run(
    keypoints: str,
    sfm: str,
    output: str,
    p1: int,
    p2: int,
    p3: int,
    known_distance: float,
    offset: list[float],
) -> None:
    # (N, 2) int [x, y] pixels
    pixels = np.load(keypoints)["keypoints"].astype(int)
    # (T, H, W, 3) float32 world-frame, NaN where invalid
    world_pts = np.load(sfm)["points"]

    kps3d = world_pts[0][pixels[:, 1], pixels[:, 0]].astype(np.float64)
    E, s = compute(kps3d[p1], kps3d[p2], kps3d[p3], known_distance)

    R32 = E[:3, :3].astype(np.float32)
    t32 = E[:3, 3].astype(np.float32)
    s32 = np.float32(s)
    off32 = np.asarray(offset, dtype=np.float32)

    aligned = s32 * (world_pts @ R32.T + t32) + off32

    os.makedirs(os.path.dirname(output), exist_ok=True)
    # extrinsic: (4, 4) float32 rigid world->gravity; scale: () float32 metric-per-world; points: (T, H, W, 3) float32 gravity-aligned, NaN where invalid
    np.savez(output, extrinsic=E.astype(np.float32), scale=s32, points=aligned)
