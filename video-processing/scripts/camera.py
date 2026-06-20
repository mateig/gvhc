"""Interactive viewer: drag to position camera, press S to save camera.json."""

import os

import numpy as np
import open3d as o3d

from scripts.config import cfg


def recover_focal(points: np.ndarray) -> tuple[float, float]:
    H, W, _ = points.shape
    cx, cy = (W - 1) / 2, (H - 1) / 2
    v_idx, u_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    valid = np.isfinite(points).all(axis=-1)
    x, y, z = points[..., 0][valid], points[..., 1][valid], points[..., 2][valid]
    fx = float(((u_idx[valid] - cx) * z * x).sum() / (x * x).sum())
    fy = float(((v_idx[valid] - cy) * z * y).sum() / (y * y).sum())
    return fx, fy


def position_camera(points: np.ndarray, colors: np.ndarray, camera_out: str) -> None:
    H, W, _ = points.shape
    mask = np.isfinite(points).all(axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[mask].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors[mask].astype(np.float64) / 255.0)

    fx, fy = recover_focal(points)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    init = o3d.camera.PinholeCameraParameters()
    init.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    init.extrinsic = np.eye(4)

    def save(vis):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(camera_out, params)
        print(f"saved {camera_out}", flush=True)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=W, height=H)
    vis.add_geometry(pcd)
    vis.get_view_control().convert_from_pinhole_camera_parameters(init, allow_arbitrary=True)
    vis.register_key_callback(ord("S"), save)
    print("drag to position; press S to save; Q/Esc to close", flush=True)
    vis.run()
    vis.destroy_window()


def main() -> None:
    data = cfg["data"] + "/"
    points = np.load(data + cfg["moge"]["run"]["npz"])["points"][0]
    colors = np.load(data + cfg["video"]["npz"])["video"][0]
    camera_out = data + cfg["moge"]["render"]["camera"]
    os.makedirs(os.path.dirname(camera_out), exist_ok=True)
    position_camera(points, colors, camera_out)


if __name__ == "__main__":
    main()
