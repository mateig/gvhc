"""Geometric priors from MoGe points for GeoCrafter conditioning."""

import torch


def robust_min_max(tensor: torch.Tensor, quantile: float = 0.99) -> tuple[float, float]:
    min_vals, max_vals = [], []
    for i in range(tensor.shape[0]):
        min_vals.append(torch.quantile(tensor[i], q=1 - quantile, interpolation="nearest").item())
        max_vals.append(torch.quantile(tensor[i], q=quantile, interpolation="nearest").item())
    return min(min_vals), max(max_vals)


def normalized_meshgrid(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1 + 1 / height, 1 - 1 / height, height, device=device)
    x = torch.linspace(-1 + 1 / width, 1 - 1 / width, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def point_map_xy2intrinsic_map(point_map_xy: torch.Tensor) -> torch.Tensor:
    height, width = point_map_xy.shape[-3], point_map_xy.shape[-2]
    mesh_grid = normalized_meshgrid(height, width, point_map_xy.device)
    mesh_grid = mesh_grid.expand_as(point_map_xy)
    nc = point_map_xy.mean(dim=-2).mean(dim=-2)
    nc_map = nc[..., None, None, :].expand_as(point_map_xy)
    nf = ((point_map_xy - nc_map) / mesh_grid).mean(dim=-2).mean(dim=-2)
    nf_map = nf[..., None, None, :].expand_as(point_map_xy)
    return torch.cat([nc_map, nf_map], dim=-1)


def produce_priors(
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    valid = torch.isfinite(points[..., 2])
    points = torch.where(valid.unsqueeze(-1), points, torch.zeros_like(points))

    valid_f = valid.float()
    norm = (points[..., 2] * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    points = points / norm

    z = points[..., 2].clamp_min(1e-3)
    xy_over_z = points[..., :2] / z.unsqueeze(-1)

    disps = (1.0 / z) * valid_f
    lo, hi = robust_min_max(disps)
    disparity = ((disps - lo) / (hi - lo + 1e-4)).clamp(0, 1) * 2 - 1

    log_z = z.log() * valid_f
    point_map = torch.cat([xy_over_z.permute(0, 3, 1, 2), log_z.unsqueeze(1)], dim=1)

    intrinsic_map = point_map_xy2intrinsic_map(xy_over_z).permute(0, 3, 1, 2)

    valid_mask = valid_f * 2 - 1

    return disparity, point_map, intrinsic_map, valid_mask, float(norm)
