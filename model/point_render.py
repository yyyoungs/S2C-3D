import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from cuda_renderer import accelerated_splatting


def _projection_canvas_size(K: torch.Tensor) -> tuple[int, int]:
    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)
    return width, height


def _output_paths(output_dir: str, name: str) -> tuple[str, str]:
    render_dir = os.path.join(output_dir, "render")
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    return os.path.join(render_dir, name), os.path.join(mask_dir, name)


def project_and_visualize(
    points_xyz: torch.Tensor,
    points_rgb: torch.Tensor,
    # rgb2_gt_tensor: torch.Tensor,
    K2_tensor: torch.Tensor,
    cam_to_world2_tensor: torch.Tensor,
    output_filename: str,
    name_id: str,
    point_size: int = 3,
    error_threshold: float = 0.1,
) -> None:
    """Project a world-space point cloud into a target camera and save image/mask files.

    Args:
        points_xyz (torch.Tensor): Point cloud in world coordinates, [N, 3].
        points_rgb (torch.Tensor): Point colors in [0, 1], [N, 3].
        K2_tensor (torch.Tensor): Target camera intrinsics, [1, 3, 3].
        cam_to_world2_tensor (torch.Tensor): Target camera-to-world matrix, [1, 4, 4].
        output_filename (str): Output directory.
        point_size (int): Square splat size in pixels.
        error_threshold (float): Reserved error threshold.
    """
    K2 = K2_tensor.squeeze(0)
    cam_to_world2 = cam_to_world2_tensor.squeeze(0)
    W, H = _projection_canvas_size(K2)
    device = points_xyz.device

    world_to_cam2 = torch.inverse(cam_to_world2)
    points_xyz_hom = torch.cat(
        [points_xyz.T, torch.ones(1, points_xyz.shape[0], device=device)],
        dim=0,
    )
    cam_coords2_hom = world_to_cam2 @ points_xyz_hom
    cam_coords2 = cam_coords2_hom[:3, :]
    depths2 = cam_coords2[2, :]
    proj_pixels = K2 @ cam_coords2
    u_proj = proj_pixels[0, :] / depths2
    v_proj = proj_pixels[1, :] / depths2

    valid_mask = (
        (u_proj >= 0)
        & (u_proj < W)
        & (v_proj >= 0)
        & (v_proj < H)
        & (depths2 > 0)
    )
    
    u_valid = u_proj[valid_mask]
    v_valid = v_proj[valid_mask]
    depths_valid = depths2[valid_mask]
    colors_valid = points_rgb[valid_mask]

    sorted_indices = torch.argsort(depths_valid, descending=True) 
    u_sorted = torch.round(u_valid[sorted_indices]).long()
    v_sorted = torch.round(v_valid[sorted_indices]).long()
    colors_sorted = colors_valid[sorted_indices]

    rendered_image_np = np.zeros((H, W, 3))
    render_mask_np = np.zeros((H, W), dtype=np.float32) 
    u_cpu = u_sorted.cpu().numpy()
    v_cpu = v_sorted.cpu().numpy()
    colors_cpu = colors_sorted.cpu().numpy()

    half_size = point_size // 2

    for i in range(len(u_cpu)):
        u, v = u_cpu[i], v_cpu[i]
        u_min = max(0, u - half_size)
        u_max = min(W, u + half_size + 1)
        v_min = max(0, v - half_size)
        v_max = min(H, v + half_size + 1)
        rendered_image_np[v_min:v_max, u_min:u_max] = colors_cpu[i]
        render_mask_np[v_min:v_max, u_min:u_max] = 1.0 

    render_filepath, mask_filepath = _output_paths(output_filename, name_id)
    rendered_image_uint8 = (rendered_image_np * 255).astype(np.uint8)
    mask_image_uint8 = (render_mask_np * 255).astype(np.uint8)
    plt.imsave(render_filepath, rendered_image_uint8)
    plt.imsave(mask_filepath, mask_image_uint8, cmap='gray') 

def project_and_visualize_cuda(
    points_xyz: torch.Tensor,
    points_rgb: torch.Tensor,
    K2_tensor: torch.Tensor,
    cam_to_world2_tensor: torch.Tensor,
    output_filename: str,
    name_id: str,
    point_size: int = 3,
    error_threshold: float = 0.1,
) -> None:
    """Project a world-space point cloud into a target camera with the CUDA splat renderer.

    Args:
        points_xyz (torch.Tensor): Point cloud in world coordinates, [N, 3].
        points_rgb (torch.Tensor): Point colors in [0, 1], [N, 3].
        K2_tensor (torch.Tensor): Target camera intrinsics, [1, 3, 3].
        cam_to_world2_tensor (torch.Tensor): Target camera-to-world matrix, [1, 4, 4].
        output_filename (str): Output directory.
        point_size (int): Square splat size in pixels.
        error_threshold (float): Reserved error threshold.
    """
    K2 = K2_tensor.squeeze(0)
    cam_to_world2 = cam_to_world2_tensor.squeeze(0)
    W, H = _projection_canvas_size(K2)

    if not points_xyz.is_cuda:
        points_xyz = points_xyz.cuda()
        points_rgb = points_rgb.cuda()
        K2 = K2.cuda()
        cam_to_world2 = cam_to_world2.cuda()

    device = points_xyz.device
    world_to_cam2 = torch.inverse(cam_to_world2)
    points_xyz_hom = torch.cat(
        [points_xyz.T, torch.ones(1, points_xyz.shape[0], device=device)],
        dim=0,
    )
    cam_coords2_hom = world_to_cam2 @ points_xyz_hom
    cam_coords2 = cam_coords2_hom[:3, :]
    depths2 = cam_coords2[2, :]
    proj_pixels = K2 @ cam_coords2
    inv_depths2 = torch.where(depths2 > 1e-6, 1.0 / depths2, torch.zeros_like(depths2))
    u_proj = proj_pixels[0, :] * inv_depths2
    v_proj = proj_pixels[1, :] * inv_depths2

    valid_mask = (
        (u_proj >= 0)
        & (u_proj < W)
        & (v_proj >= 0)
        & (v_proj < H)
        & (depths2 > 0)
    )
    
    u_valid = u_proj[valid_mask]
    v_valid = v_proj[valid_mask]
    depths_valid = depths2[valid_mask]
    colors_valid = points_rgb[valid_mask]

    sorted_indices = torch.argsort(depths_valid, descending=True) 
    u_sorted = u_valid[sorted_indices].float().contiguous()
    v_sorted = v_valid[sorted_indices].float().contiguous()
    colors_sorted = colors_valid[sorted_indices].float().contiguous()

    render_result = accelerated_splatting(
        u_sorted, 
        v_sorted, 
        colors_sorted, 
        H, 
        W, 
        point_size
    )

    rendered_image_tensor = render_result[..., :3]
    render_mask_tensor = render_result[..., 3]
    rendered_image_np = rendered_image_tensor.cpu().numpy()
    render_mask_np = render_mask_tensor.cpu().numpy()
    render_filepath, mask_filepath = _output_paths(output_filename, name_id)
    rendered_image_uint8 = (rendered_image_np * 255).astype(np.uint8)
    mask_image_uint8 = (render_mask_np * 255).astype(np.uint8)
    plt.imsave(render_filepath, rendered_image_uint8)
    plt.imsave(mask_filepath, mask_image_uint8, cmap='gray')


def unproject_to_pointcloud(
    rgb_tensor: torch.Tensor, 
    depth_tensor: torch.Tensor, 
    K_tensor: torch.Tensor, 
    cam_to_world_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unproject RGB-D tensors into a world-space point cloud."""
    rgb = rgb_tensor.squeeze(0)
    depth = depth_tensor.squeeze(0)
    K = K_tensor.squeeze(0)
    cam_to_world = cam_to_world_tensor.squeeze(0)
    H, W, _ = rgb.shape
    device = rgb.device
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    pixels = torch.stack([u.flatten(), v.flatten(), torch.ones_like(u.flatten())], dim=0)
    K_inv = torch.inverse(K)
    cam_coords_norm = K_inv @ pixels
    cam_coords = cam_coords_norm * depth.flatten().unsqueeze(0)
    cam_coords_hom = torch.cat([cam_coords, torch.ones(1, H * W, device=device)], dim=0)
    world_coords_hom = cam_to_world @ cam_coords_hom
    points_xyz = world_coords_hom[:3, :].T
    points_rgb = (rgb.reshape(-1, 3) * 255).to(torch.uint8)
    return points_xyz, points_rgb.float() / 255.0
