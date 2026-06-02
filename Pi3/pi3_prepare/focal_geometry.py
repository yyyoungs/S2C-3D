from typing import Tuple

import torch
import torch.nn.functional as F

from focal_solvers import solve_optimal_focal_shift, solve_optimal_shift


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build normalized image-plane coordinates centered at the optical center."""
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
    focal: torch.Tensor | None = None,
    downsample_size: Tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recover focal length and z-shift from a Pi3 local point map.

    The estimate assumes centered optical axes, no distortion, and equal x/y
    scale in the point map.
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    flat_points = points.reshape(-1, *shape[-3:])
    flat_mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    flat_focal = focal.reshape(-1) if focal is not None else None

    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)
    points_lr = F.interpolate(flat_points.permute(0, 3, 1, 2), downsample_size, mode="nearest").permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest").squeeze(0).permute(1, 2, 0)
    mask_lr = None
    if flat_mask is not None:
        mask_lr = F.interpolate(flat_mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest").squeeze(1) > 0

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = flat_focal.cpu().numpy() if flat_focal is not None else None
    mask_lr_np = None if mask_lr is None else mask_lr.cpu().numpy()

    solved_shifts = []
    solved_focals = []
    for batch_idx in range(flat_points.shape[0]):
        if mask_lr_np is None:
            selected_points = points_lr_np[batch_idx]
            selected_uv = uv_lr_np
        else:
            selected_points = points_lr_np[batch_idx][mask_lr_np[batch_idx]]
            selected_uv = uv_lr_np[mask_lr_np[batch_idx]]

        if selected_uv.shape[0] < 2:
            solved_shifts.append(0.0)
            solved_focals.append(1.0)
            continue

        if flat_focal is None:
            shift_i, focal_i = solve_optimal_focal_shift(selected_uv, selected_points)
            solved_focals.append(float(focal_i))
        else:
            shift_i = solve_optimal_shift(selected_uv, selected_points, focal_np[batch_idx])
        solved_shifts.append(float(shift_i))

    shift = torch.tensor(solved_shifts, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    if flat_focal is None:
        focal = torch.tensor(solved_focals, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        focal = flat_focal.reshape(shape[:-3])

    return focal, shift
