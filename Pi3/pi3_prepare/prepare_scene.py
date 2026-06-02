import argparse
import json
import os
import struct
import shutil
import sys
from pathlib import Path

PREPARE_ROOT = Path(__file__).resolve().parent
DEFAULT_PI3_ROOT = PREPARE_ROOT.parent / "Pi3-main"
if str(PREPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPARE_ROOT))
if DEFAULT_PI3_ROOT.exists() and str(DEFAULT_PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PI3_ROOT))

import numpy as np
import torch
import trimesh

from focal_geometry import recover_focal_shift
from pi3.models.pi3 import Pi3
from utils.geometry import depth_edge, homogenize_points
from utils.image_io import load_images_as_tensor


CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
}


def rotation_matrix_to_colmap_qvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a world-to-camera rotation matrix to COLMAP's qw, qx, qy, qz."""
    trace = np.trace(rotation)
    if trace > 0:
        qw = 0.5 * np.sqrt(1.0 + trace)
        qx = (rotation[2, 1] - rotation[1, 2]) * 0.25 / qw
        qy = (rotation[0, 2] - rotation[2, 0]) * 0.25 / qw
        qz = (rotation[1, 0] - rotation[0, 1]) * 0.25 / qw
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    return qvec / np.linalg.norm(qvec)


def camera_params_from_intrinsics(intrinsics: np.ndarray, fidx: int, camera_type: str) -> np.ndarray:
    if camera_type == "PINHOLE":
        return np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]],
            dtype=np.float64,
        )
    if camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        return np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]], dtype=np.float64)
    raise ValueError(f"Camera type {camera_type} is not supported yet")


def write_colmap_cameras(path: str, intrinsics: np.ndarray, image_size: list[int], camera_type: str) -> None:
    model_id = CAMERA_MODEL_IDS[camera_type]
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(intrinsics)))
        for camera_idx in range(len(intrinsics)):
            params = camera_params_from_intrinsics(intrinsics, camera_idx, camera_type)
            f.write(struct.pack("<IiQQ", camera_idx + 1, model_id, int(image_size[0]), int(image_size[1])))
            f.write(struct.pack("<" + "d" * len(params), *params))


def build_image_observations(points_xyf: np.ndarray, resize_ratio: float, num_frames: int) -> list[list[tuple[float, float, int]]]:
    observations = [[] for _ in range(num_frames)]
    for point_idx, xyf in enumerate(points_xyf):
        frame_idx = int(xyf[2])
        if frame_idx < 0 or frame_idx >= num_frames:
            continue
        x, y = xyf[:2] * resize_ratio
        observations[frame_idx].append((float(x), float(y), point_idx + 1))
    return observations


def write_colmap_images(
    path: str,
    extrinsics: np.ndarray,
    image_names: list[str],
    observations: list[list[tuple[float, float, int]]],
) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(extrinsics)))
        for fidx, w2c in enumerate(extrinsics):
            image_id = fidx + 1
            qvec = rotation_matrix_to_colmap_qvec(w2c[:3, :3])
            tvec = w2c[:3, 3].astype(np.float64)
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", *qvec))
            f.write(struct.pack("<3d", *tvec))
            f.write(struct.pack("<I", image_id))
            f.write(image_names[fidx].encode("utf-8") + b"\x00")

            image_points = observations[fidx]
            f.write(struct.pack("<Q", len(image_points)))
            for x, y, point3d_id in image_points:
                f.write(struct.pack("<ddq", x, y, point3d_id))


def write_colmap_points3d(
    path: str,
    points3d: np.ndarray,
    points_rgb: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
) -> None:
    point_tracks: dict[int, tuple[int, int]] = {}
    for frame_idx, image_points in enumerate(observations):
        for point2d_idx, (_, _, point3d_id) in enumerate(image_points):
            point_tracks[point3d_id] = (frame_idx + 1, point2d_idx)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points3d)))
        for point_idx, xyz in enumerate(points3d):
            point3d_id = point_idx + 1
            rgb = points_rgb[point_idx].astype(np.uint8)
            image_id, point2d_idx = point_tracks.get(point3d_id, (1, 0))
            f.write(struct.pack("<Q", point3d_id))
            f.write(struct.pack("<3d", *xyz.astype(np.float64)))
            f.write(struct.pack("<3B", int(rgb[0]), int(rgb[1]), int(rgb[2])))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<II", image_id, point2d_idx))


def write_colmap_sparse_model(
    output_dir: str,
    points3d: np.ndarray,
    points_xyf: np.ndarray,
    points_rgb: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_size: list[int],
    image_names: list[str],
    resize_ratio: float,
    camera_type: str = "PINHOLE",
) -> None:
    """Write a standard COLMAP binary sparse model without using pycolmap APIs."""
    os.makedirs(output_dir, exist_ok=True)
    observations = build_image_observations(points_xyf, resize_ratio, len(extrinsics))
    write_colmap_cameras(os.path.join(output_dir, "cameras.bin"), intrinsics, image_size, camera_type)
    write_colmap_images(os.path.join(output_dir, "images.bin"), extrinsics, image_names, observations)
    write_colmap_points3d(os.path.join(output_dir, "points3D.bin"), points3d, points_rgb, observations)


def create_pixel_coordinate_grid(num_frames, height, width):
    """Create a grid of pixel coordinates and frame indices."""
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_coords = np.broadcast_to(x_grid[np.newaxis, :, :], (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid[np.newaxis, :, :], (num_frames, height, width))
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)
    return points_xyf


def intrinsics_from_focal_center(fx, fy, cx, cy):
    intrinsic = torch.zeros((*fx.shape, 3, 3), dtype=fx.dtype, device=fx.device)
    intrinsic[..., 0, 0] = fx
    intrinsic[..., 1, 1] = fy
    intrinsic[..., 0, 2] = cx
    intrinsic[..., 1, 2] = cy
    intrinsic[..., 2, 2] = 1
    return intrinsic


def extract_pi3_geometry(result, original_image_sizes, resize_ratio):
    """Extract point maps, intrinsics, masks, and camera poses from Pi3 output."""
    points = result["local_points"]
    masks = torch.sigmoid(result["conf"][..., 0]) > 0.1
    focal, shift = recover_focal_shift(points, masks, downsample_size=(64, 64))

    original_height, original_width = points.shape[-3:-1]
    aspect_ratio = original_width / original_height

    fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio * original_width
    fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 * original_height

    cx = original_width // 2
    cy = original_height // 2
    intrinsic = intrinsics_from_focal_center(fx, fy, cx, cy)

    c2ws = result["camera_poses"]

    masks = torch.sigmoid(result["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(result["local_points"][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    points_3d = result["points"][0]
    num_frames, H_infer, W_infer = points_3d.shape[:-1]
    points_xyf = create_pixel_coordinate_grid(num_frames, H_infer, W_infer)
    shift = shift.squeeze(0).unsqueeze(1).unsqueeze(2)
    points_3d[:, :, :, 2] += shift
    intrinsic[:, :, 0, 0] = intrinsic[:, :, 0, 0] * resize_ratio
    intrinsic[:, :, 1, 1] = intrinsic[:, :, 1, 1] * resize_ratio
    intrinsic[:, :, 0, 2] = original_image_sizes[0] / 2
    intrinsic[:, :, 1, 2] = original_image_sizes[1] / 2
    return points_3d, points_xyf, masks, intrinsic, c2ws, original_width, original_height


def solve_rigid_transform_svd(X, Y):
    """
    Solve a rigid transform (R, t) with SVD and fixed scale.
    Y = R @ X + t
    
    Args:
        X (torch.Tensor): Source points (N, 3)
        Y (torch.Tensor): Target points (N, 3)
        
    Returns:
        tuple: (R, t), Rotation (3, 3), Translation (3)
    """
    center_X = X.mean(dim=0, keepdim=True)
    center_Y = Y.mean(dim=0, keepdim=True)

    Q_X = X - center_X
    Q_Y = Y - center_Y

    covariance = Q_X.T @ Q_Y
    U, _, V_T = torch.linalg.svd(covariance)
    V = V_T.T

    d = torch.det(V @ U.T)
    c = torch.eye(3, dtype=X.dtype, device=X.device)
    c[-1, -1] = d
    R = V @ c @ U.T

    t = center_Y.T - R @ center_X.T

    return R, t.squeeze()


def estimate_rigid_alignment_ransac(X, Y, max_iterations=200, threshold=0.01, min_inliers=5):
    """
    Robustly estimate the rigid transform from X to Y with RANSAC.
    
    Args:
        X (torch.Tensor): Source points (N, 3)
        Y (torch.Tensor): Target points (N, 3)
        max_iterations (int): Maximum number of RANSAC iterations.
        threshold (float): Inlier distance threshold.
        min_inliers (int): Minimum number of samples used to estimate a model.
        
    Returns:
        torch.Tensor: Best 4x4 rigid alignment matrix.
    """
    device = X.device
    N = X.shape[0]
    best_inlier_count = -1
    best_R = None
    best_t = None
    
    if N < min_inliers:
        print(f"Warning: Not enough points for RANSAC. N={N}, required min_inliers={min_inliers}")
        R, t = solve_rigid_transform_svd(X, Y)
        T_align = torch.eye(4, device=device)
        T_align[:3, :3] = R
        T_align[:3, 3] = t
        return T_align

    for _ in range(max_iterations):
        indices = torch.randperm(N)[:min_inliers]
        X_sample = X[indices]
        Y_sample = Y[indices]

        R_model, t_model = solve_rigid_transform_svd(X_sample, Y_sample)

        X_transformed = (R_model @ X.T + t_model.unsqueeze(1)).T

        errors = torch.linalg.norm(X_transformed - Y, dim=1)
        inlier_mask = errors < threshold
        current_inlier_count = torch.sum(inlier_mask).item()

        if current_inlier_count > best_inlier_count:
            best_inlier_count = current_inlier_count

            X_inliers = X[inlier_mask]
            Y_inliers = Y[inlier_mask]

            if X_inliers.shape[0] >= min_inliers:
                best_R, best_t = solve_rigid_transform_svd(X_inliers, Y_inliers)
            else:
                best_R, best_t = R_model, t_model

    print(f"RANSAC finished. Best inliers: {best_inlier_count}/{N} ({best_inlier_count/N:.2f}%)")

    T_align = torch.eye(4, device=device)
    T_align[:3, :3] = best_R
    T_align[:3, 3] = best_t

    return T_align

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    parser.add_argument("--data_path", type=str, default="./data/3/", help="Input scene directory containing images/.")
    parser.add_argument("--interval", type=int, default=-1, help="Image sampling interval.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional local model checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device, e.g. cuda or cpu.")
    parser.add_argument("--view", type=str, default="all", help="Number of selected training views, or all.")
    return parser.parse_args()


def load_pi3_model(args: argparse.Namespace, device: torch.device) -> Pi3:
    """Load either a local Pi3 checkpoint or the default Hugging Face model."""
    print("Loading model...")
    if args.ckpt is None:
        return Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    model = Pi3().to(device).eval()
    if args.ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file

        weight = load_file(args.ckpt)
    else:
        weight = torch.load(args.ckpt, map_location=device, weights_only=False)

    model.load_state_dict(weight)
    return model


def copy_train_val_images(image_dir: str, save_dir: str, image_names: list[str], train_indices: list[int]) -> None:
    train_img_dir = os.path.join(save_dir, "train_img")
    val_img_dir = os.path.join(save_dir, "val_img")
    shutil.rmtree(train_img_dir, ignore_errors=True)
    shutil.rmtree(val_img_dir, ignore_errors=True)
    os.makedirs(train_img_dir, exist_ok=True)

    train_index_set = set(train_indices)
    has_validation_images = len(train_index_set) < len(image_names)
    if has_validation_images:
        os.makedirs(val_img_dir, exist_ok=True)

    for index, name in enumerate(image_names):
        if index in train_index_set:
            target_dir = train_img_dir
        elif has_validation_images:
            target_dir = val_img_dir
        else:
            continue
        shutil.copy(os.path.join(image_dir, name), target_dir)

    with open(os.path.join(save_dir, "cam_idx.json"), "w", encoding="utf-8") as f:
        json.dump(train_indices, f, indent=4, ensure_ascii=False)


def infer_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def prepare_scene(args: argparse.Namespace) -> None:
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith(".mp4") else 1

    print(f"Sampling interval: {args.interval}")
    image_path = os.path.join(args.data_path, "images")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"There is no image data under: {image_path}")

    save_path = os.path.join(args.data_path, f"{args.view}_views")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device(args.device)
    model = load_pi3_model(args, device)

    imgs_tensor, original_image_sizes, base_image_path_list = load_images_as_tensor(image_path, interval=args.interval)
    imgs_tensor = imgs_tensor.to(device)
    _, _, H_infer, W_infer = imgs_tensor.shape

    vggt_fixed_resolution = (W_infer, H_infer)
    print(f"Inference resolution (W, H): {vggt_fixed_resolution}")

    if args.view != "all":
        views = int(args.view)
        select_idx = list(range(views))
    else:
        select_idx = list(range(len(base_image_path_list)))

    copy_train_val_images(image_path, save_path, base_image_path_list, select_idx)

    print("Running model inference...")
    dtype = infer_autocast_dtype(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            result_all = model(imgs_tensor[None])
            if args.view == "all":
                result_part = None
                select_img_tensor = imgs_tensor
            else:
                select_img_tensor = imgs_tensor[select_idx]
                result_part = model(select_img_tensor[None])

    resize_ratio = max(
        [original_image_sizes[0] / vggt_fixed_resolution[0], original_image_sizes[1] / vggt_fixed_resolution[1]]
    )
    _, _, _, intrinsic, c2ws, _, _ = extract_pi3_geometry(
        result_all, original_image_sizes, resize_ratio
    )
    if args.view == "all":
        points_3d_all, points_xyf_all, masks_all, _, _, _, _ = extract_pi3_geometry(
            result_all, original_image_sizes, resize_ratio
        )
        points_rgb = select_img_tensor.permute(0, 2, 3, 1)[masks_all]
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_3d = points_3d_all[masks_all].cpu().numpy()
        points_xyf = points_xyf_all[masks_all.cpu().numpy()]
        print(f"Point range: min={np.min(points_3d):.4f}, max={np.max(points_3d):.4f}")
    else:
        points_3d_p, points_xyf_p, masks_p, _, c2ws_p, _, _ = extract_pi3_geometry(
            result_part, original_image_sizes, resize_ratio
        )

        target_c2ws = c2ws[0, select_idx]
        source_c2ws = c2ws_p[0]

        P_target = target_c2ws[:, :3, 3]
        P_source = source_c2ws[:, :3, 3]

        N_part = P_target.shape[0]
        if N_part < 3:
            raise ValueError(f"Need at least 3 camera poses for alignment, but found only {N_part}.")

        print(f"\n--- Starting RANSAC Rigid Alignment of {N_part} camera centers ---")
        T_align = estimate_rigid_alignment_ransac(
            X=P_source,
            Y=P_target,
            max_iterations=5000,
            threshold=0.2,
            min_inliers=5,
        )

        print("Alignment transformation T_align found (RANSAC Rigid):")
        print(T_align)

        H_infer, W_infer = points_3d_p.shape[-3:-1]
        points_p_flat = points_3d_p.reshape(-1, 3)
        points_p_homo = homogenize_points(points_p_flat)
        aligned_points_homo = T_align @ points_p_homo.T
        aligned_points_p_flat = aligned_points_homo[:3].T
        aligned_points_3d_p = aligned_points_p_flat.reshape(N_part, H_infer, W_infer, 3)

        print("Partial point cloud P' has been aligned to the 'full' coordinate system (RANSAC Rigid Transform applied).")

        points_rgb_p = select_img_tensor.permute(0, 2, 3, 1)[masks_p]
        points_rgb_p_np = (points_rgb_p.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb_p_np

        aligned_points_3d_masked = aligned_points_3d_p[masks_p].cpu().numpy()
        points_xyf = points_xyf_p[masks_p.cpu()]
        points_3d = aligned_points_3d_masked
        print(f"Aligned point range: min={np.min(points_3d):.4f}, max={np.max(points_3d):.4f}")

    print("Converting to COLMAP format")
    camera_type = "PINHOLE"
    w2cs = torch.linalg.inv(c2ws)
    sparse_reconstruction_dir = os.path.join(save_path, "sparse")
    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    write_colmap_sparse_model(
        output_dir=sparse_reconstruction_dir,
        points3d=points_3d,
        points_xyf=points_xyf,
        points_rgb=points_rgb,
        extrinsics=w2cs[0].cpu().numpy(),
        intrinsics=intrinsic[0].cpu().numpy(),
        image_size=[original_image_sizes[0], original_image_sizes[1]],
        image_names=base_image_path_list,
        resize_ratio=resize_ratio,
        camera_type=camera_type,
    )

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(save_path, "sparse/points.ply"))


def main() -> None:
    prepare_scene(parse_args())


if __name__ == "__main__":
    main()
