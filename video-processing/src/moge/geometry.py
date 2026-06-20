"""Focal/shift recovery and point-map reconstruction for MoGe."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares


def normalized_view_plane_uv(width: int, height: int) -> torch.Tensor:
    aspect = width / height
    diag = (1 + aspect**2) ** 0.5
    span_x, span_y = aspect / diag, 1 / diag
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> tuple[float, float]:
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def residual(shift):
        proj = xy / (z + shift)[:, None]
        f = (proj * uv).sum() / np.square(proj).sum()
        return (f * proj - uv).ravel()

    result = least_squares(residual, x0=0, ftol=1e-3, method="lm")
    opt_shift = float(result["x"].squeeze())
    proj = xy / (z + opt_shift)[:, None]
    return opt_shift, float((proj * uv).sum() / np.square(proj).sum())


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float) -> float:
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def residual(shift):
        return (focal * xy / (z + shift)[:, None] - uv).ravel()

    return float(least_squares(residual, x0=0, ftol=1e-3, method="lm")["x"].squeeze())


def recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
    focal: torch.Tensor | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    device = points.device
    shape = points.shape
    h, w = shape[-3], shape[-2]
    points = points.reshape(-1, *shape[-3:])
    if mask is not None:
        mask = mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None

    uv = normalized_view_plane_uv(w, h).to(device)

    pts_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode="nearest").permute(
        0, 2, 3, 1
    )
    uv_lr = (
        F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest")
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_lr = None
    if mask is not None:
        mask_lr = (
            F.interpolate(
                mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest"
            ).squeeze(1)
            > 0
        )

    uv_np = uv_lr.cpu().numpy()
    pts_np = pts_lr.cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_np = mask_lr.cpu().numpy() if mask_lr is not None else None

    out_shift, out_focal = [], []
    for i in range(points.shape[0]):
        if mask_np is None:
            pts_i, uv_i = pts_np[i], uv_np
        else:
            pts_i = pts_np[i][mask_np[i]]
            uv_i = uv_np[mask_np[i]]
        if uv_i.reshape(-1, 2).shape[0] < 2:
            out_focal.append(1.0)
            out_shift.append(0.0)
            continue
        if focal is None:
            shift_i, focal_i = solve_optimal_focal_shift(uv_i, pts_i)
            out_focal.append(focal_i)
        else:
            shift_i = solve_optimal_shift(uv_i, pts_i, focal_np[i])
        out_shift.append(shift_i)

    out_shift = torch.tensor(out_shift, device=device).reshape(shape[:-3])
    if focal is None:
        out_focal = torch.tensor(out_focal, device=device).reshape(shape[:-3])
    else:
        out_focal = focal.reshape(shape[:-3])
    return out_focal, out_shift


def intrinsics_from_focal_center(
    fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor
) -> torch.Tensor:
    intrinsics = torch.zeros(*fx.shape, 3, 3, device=fx.device, dtype=fx.dtype)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = cx
    intrinsics[..., 1, 2] = cy
    intrinsics[..., 2, 2] = 1.0
    return intrinsics


def depth_map_to_point_map(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    device = depth.device
    h, w = depth.shape[-2:]
    u = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=device)
    v = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=device)
    v, u = torch.meshgrid(v, u, indexing="ij")

    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]

    x = (u - cx[..., None, None]) / fx[..., None, None]
    y = (v - cy[..., None, None]) / fy[..., None, None]
    return torch.stack([x * depth, y * depth, depth], dim=-1)


def recover_3d(
    points: torch.Tensor,
    metric_scale: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = points.device
    aspect = points.shape[-2] / points.shape[-3]

    focal, shift = recover_focal_shift(points, mask=mask)

    diag_half = (1 + aspect**2) ** 0.5 / 2
    intrinsics = intrinsics_from_focal_center(
        focal * diag_half / aspect,
        focal * diag_half,
        torch.tensor(0.5, device=device),
        torch.tensor(0.5, device=device),
    )
    points = points.clone()
    points[..., 2] += shift[..., None, None]

    valid = points[..., 2] > 0
    if mask is not None:
        valid = valid & mask
    depth = points[..., 2].clone()

    points = depth_map_to_point_map(depth, intrinsics)
    points = points * metric_scale[:, None, None, None]
    depth = depth * metric_scale[:, None, None]

    nan = torch.tensor(float("nan"), device=device, dtype=points.dtype)
    points = torch.where(valid.unsqueeze(-1), points, nan)
    depth = torch.where(valid, depth, nan)

    return points, intrinsics, depth
