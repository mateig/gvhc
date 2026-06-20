"""SfM via dense optical flow + multi-gap pose graph + temporal smoothing.

Per-frame camera-frame point clouds from GeometryCrafter are metrically stable,
so relative poses are solved directly by 3D-3D Kabsch on flow correspondences
(no triangulation, no intrinsics). Dynamic regions and far-depth points are
masked out before matching. Pose graph ties multiple gap lengths, and a final
Savitzky-Golay smoother removes high-frequency jitter from the trajectory.
"""

import os
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def kabsch(
    src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if w is None:
        sc = src.mean(axis=0)
        dc = dst.mean(axis=0)
        H = (src - sc).T @ (dst - dc)
    else:
        ws = w / w.sum()
        sc = (src * ws[:, None]).sum(axis=0)
        dc = (dst * ws[:, None]).sum(axis=0)
        H = ((src - sc) * ws[:, None]).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, dc - R @ sc


def ransac_kabsch(
    src: np.ndarray,
    dst: np.ndarray,
    rng: np.random.Generator,
    threshold: float,
    iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(src)
    best = np.zeros(n, dtype=bool)
    if n < 4:
        return np.eye(3), np.zeros(3), best
    for _ in range(iters):
        idx = rng.choice(n, size=4, replace=False)
        try:
            R, t = kabsch(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.linalg.norm(src @ R.T + t - dst, axis=1) < threshold
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 4:
        return np.eye(3), np.zeros(3), best
    s_in, d_in = src[best], dst[best]
    R, t = kabsch(s_in, d_in)
    for _ in range(3):
        e = np.linalg.norm(s_in @ R.T + t - d_in, axis=1)
        w_irls = 1.0 / np.maximum(e, threshold * 0.3)
        R, t = kabsch(s_in, d_in, w_irls)
    return R, t, best


def static_masks(dynamic: np.ndarray, dilate_ksize: int) -> np.ndarray:
    T, H, W = dynamic.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    out = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        out[t] = cv2.dilate(dynamic[t].astype(np.uint8), k) == 0
    return out


def flow_edge(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    static_a: np.ndarray,
    static_b: np.ndarray,
    flow: cv2.DISOpticalFlow,
    rng: np.random.Generator,
    sample: int,
    depth_percentile: float,
    cycle_threshold: float,
    ransac_threshold: float,
    ransac_iters: int,
    min_inliers: int,
) -> dict | None:
    H, W = gray_a.shape
    f_ab = flow.calc(gray_a, gray_b, None)
    f_ba = flow.calc(gray_b, gray_a, None)

    za = pts_a[..., 2]
    valid_a = np.isfinite(za) & static_a
    if not valid_a.any():
        return None
    z_thr = np.percentile(za[valid_a], depth_percentile)
    cand = valid_a & (za <= z_thr)
    ys, xs = np.where(cand)
    if len(ys) < 2 * min_inliers:
        return None
    if len(ys) > sample:
        idx = rng.choice(len(ys), size=sample, replace=False)
        ys, xs = ys[idx], xs[idx]

    dx = f_ab[ys, xs, 0]
    dy = f_ab[ys, xs, 1]
    xb = xs + dx
    yb = ys + dy
    xc = np.clip(xb, 0, W - 1)
    yc = np.clip(yb, 0, H - 1)
    dx2 = map_coordinates(f_ba[..., 0], [yc, xc], order=1)
    dy2 = map_coordinates(f_ba[..., 1], [yc, xc], order=1)
    cyc = np.sqrt((xb + dx2 - xs) ** 2 + (yb + dy2 - ys) ** 2)
    ok = (cyc < cycle_threshold) & (xb >= 0) & (xb < W) & (yb >= 0) & (yb < H)
    if ok.sum() < 2 * min_inliers:
        return None
    ys, xs, xb, yb = ys[ok], xs[ok], xb[ok], yb[ok]

    xb_i = np.clip(np.round(xb).astype(int), 0, W - 1)
    yb_i = np.clip(np.round(yb).astype(int), 0, H - 1)
    ok_b = static_b[yb_i, xb_i] & np.isfinite(pts_b[yb_i, xb_i, 2])
    if ok_b.sum() < 2 * min_inliers:
        return None
    ys, xs, xb_i, yb_i = ys[ok_b], xs[ok_b], xb_i[ok_b], yb_i[ok_b]

    pa = pts_a[ys, xs].astype(np.float64)
    pb = pts_b[yb_i, xb_i].astype(np.float64)

    R, t, inl = ransac_kabsch(pa, pb, rng, threshold=ransac_threshold, iters=ransac_iters)
    if inl.sum() < min_inliers:
        return None
    resid = np.linalg.norm(pa[inl] @ R.T + t - pb[inl], axis=1)
    return dict(R=R, t=t, n_inl=int(inl.sum()), resid=float(np.median(resid)))


def bfs_init(N: int, edges: list[dict]) -> np.ndarray:
    nbr: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(N)]
    for e in edges:
        T_ab = make_T(e["R"], e["t"])
        nbr[e["a"]].append((e["b"], T_ab))
        nbr[e["b"]].append((e["a"], inv_T(T_ab)))
    poses = np.tile(np.eye(4), (N, 1, 1))
    reached = np.zeros(N, dtype=bool)
    reached[0] = True
    q = deque([0])
    while q:
        i = q.popleft()
        for j, T_ij in nbr[i]:
            if reached[j]:
                continue
            poses[j] = T_ij @ poses[i]
            reached[j] = True
            q.append(j)
    if not reached.all():
        missing = np.where(~reached)[0].tolist()
        raise RuntimeError(f"sfm: frames {missing} have no edges connecting them to frame 0")
    return poses


def pose_graph_optimize(poses0: np.ndarray, edges: list[dict]) -> np.ndarray:
    N = len(poses0)
    if N < 2 or not edges:
        return poses0
    E_inv = [inv_T(make_T(e["R"], e["t"])) for e in edges]
    w = np.array([np.sqrt(max(e["n_inl"], 1)) / max(e["resid"], 1e-3) for e in edges])
    w = w / w.mean()

    p0 = np.zeros((N - 1) * 6)
    for i in range(1, N):
        p0[(i - 1) * 6 : (i - 1) * 6 + 3] = Rotation.from_matrix(poses0[i][:3, :3]).as_rotvec()
        p0[(i - 1) * 6 + 3 : i * 6] = poses0[i][:3, 3]

    def pose_of(i: int, p: np.ndarray) -> np.ndarray:
        if i == 0:
            return np.eye(4)
        seg = p[(i - 1) * 6 : i * 6]
        return make_T(Rotation.from_rotvec(seg[:3]).as_matrix(), seg[3:])

    def residuals(p: np.ndarray) -> np.ndarray:
        res = np.zeros(len(edges) * 6)
        for k, e in enumerate(edges):
            err = pose_of(e["b"], p) @ inv_T(pose_of(e["a"], p)) @ E_inv[k]
            rv = Rotation.from_matrix(err[:3, :3]).as_rotvec()
            res[k * 6 : (k + 1) * 6] = np.concatenate([rv, err[:3, 3]]) * w[k]
        return res

    sol = least_squares(residuals, p0, loss="soft_l1")
    return np.stack([pose_of(i, sol.x) for i in range(N)])


def savgol_poses(poses: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if window < 3 or window > len(poses):
        return poses
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    cam = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)
    cam_s = savgol_filter(cam, window_length=window, polyorder=polyorder, axis=0)
    rv = Rotation.from_matrix(R).as_rotvec()
    rv_s = savgol_filter(rv, window_length=window, polyorder=polyorder, axis=0)
    R_s = Rotation.from_rotvec(rv_s).as_matrix()
    out = np.tile(np.eye(4), (len(poses), 1, 1))
    out[:, :3, :3] = R_s
    out[:, :3, 3] = -np.einsum("nij,nj->ni", R_s, cam_s)
    return out


def transform_to_world(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    T, H, W, _ = points.shape
    out = np.full((T, H, W, 3), np.nan, dtype=np.float32)
    for t in range(T):
        R_cw, t_cw = poses[t, :3, :3], poses[t, :3, 3]
        flat = points[t].reshape(-1, 3).astype(np.float64)
        valid = np.isfinite(flat).all(axis=1)
        world = np.full_like(flat, np.nan)
        world[valid] = (flat[valid] - t_cw) @ R_cw
        out[t] = world.reshape(H, W, 3).astype(np.float32)
    return out


def run(
    video: str,
    points: str,
    masks: str,
    output: str,
    gaps: tuple[int, ...] = (1, 4, 16, 64),
    dilate_ksize: int = 25,
    depth_percentile: float = 50.0,
    sample: int = 3000,
    cycle_threshold: float = 0.5,
    ransac_threshold: float = 0.02,
    ransac_iters: int = 300,
    min_inliers: int = 50,
    smooth_window: int = 11,
    smooth_polyorder: int = 2,
) -> None:
    # (T, H, W, 3) uint8 RGB
    video_np = np.load(video)["video"]
    # (T, H, W, 3) float32 camera-frame, NaN where invalid
    pts = np.load(points)["points"]
    # sam3 (T, N, H, W) bool collapsed to (T, H, W)
    dynamic = np.load(masks)["masks"].any(axis=1)
    T = pts.shape[0]

    grays = [cv2.cvtColor(video_np[t], cv2.COLOR_RGB2GRAY) for t in range(T)]
    statics = static_masks(dynamic, dilate_ksize)

    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rng = np.random.default_rng(0)
    edges: list[dict] = []
    for g in gaps:
        for a in range(T - g):
            b = a + g
            e = flow_edge(
                grays[a],
                grays[b],
                pts[a],
                pts[b],
                statics[a],
                statics[b],
                flow,
                rng,
                sample=sample,
                depth_percentile=depth_percentile,
                cycle_threshold=cycle_threshold,
                ransac_threshold=ransac_threshold,
                ransac_iters=ransac_iters,
                min_inliers=min_inliers,
            )
            if e is None:
                continue
            e["a"] = a
            e["b"] = b
            edges.append(e)

    poses0 = bfs_init(T, edges)
    poses = pose_graph_optimize(poses0, edges)
    poses = savgol_poses(poses, window=smooth_window, polyorder=smooth_polyorder)
    world = transform_to_world(pts, poses)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    # poses: (T, 4, 4) float32 world->camera; points: (T, H, W, 3) float32 world-frame, NaN where invalid
    np.savez(output, poses=poses.astype(np.float32), points=world)
