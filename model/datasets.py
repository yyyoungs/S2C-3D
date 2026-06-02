import json
import os
import random
from typing import Any, Dict, List

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from scipy.spatial.transform import Rotation, Slerp
from typing_extensions import assert_never

from model.cam_planner import CameraPlanner
from model.normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

CameraRecord = Dict[str, Any]


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _numeric_image_sort_key(path: str) -> int:
    return int(path.split(".")[0].split("_")[-1])


def _image_stem(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def _camera_record(
    K: np.ndarray,
    camtoworlds: np.ndarray,
    idx: int,
    image_name: str | None = None,
    image: str | None = None,
    is_train: bool | None = None,
    pre_idx: int | None = None,
    aft_idx: int | None = None,
) -> CameraRecord:
    camera: CameraRecord = {
        "K": K,
        "camtoworlds": camtoworlds,
        "idx": idx,
    }
    if image_name is not None:
        camera["image_name"] = image_name
    if image is not None or is_train is not None:
        camera["image"] = image
    if is_train is not None:
        camera["is_train"] = is_train
    if pre_idx is not None:
        camera["pre_idx"] = pre_idx
    if aft_idx is not None:
        camera["aft_idx"] = aft_idx
    return camera


def _tensor_camera_payload(
    camera: CameraRecord,
    item: int,
    image_path: str | None = None,
) -> Dict[str, Any]:
    data = {
        "K": torch.from_numpy(camera["K"]).float(),
        "camtoworld": torch.from_numpy(camera["camtoworlds"]).float(),
        "image_id": item,
    }
    if image_path is not None:
        image = imageio.imread(image_path)[..., :3]
        data["image"] = torch.from_numpy(image).float()
        data["image_name"] = _image_stem(image_path)
    return data


def _log_training_camera_count(train_count: int) -> None:
    print(f"Loaded {train_count} training cameras.")


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        img_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.img_dir = img_dir
        colmap_dir = os.path.join(data_dir, "sparse")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "colmap", "sparse", "0")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(img_dir, "images")
        image_dir = os.path.join(img_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        if "3dv-dataset-nerfstudio" in data_dir:
            colmap_files = sorted(_get_rel_paths(colmap_image_dir), key=_numeric_image_sort_key)
            image_files = sorted(_get_rel_paths(image_dir), key=_numeric_image_sort_key)
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_names = colmap_files
            image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        elif "DL3DV-Benchmark" in data_dir:
            colmap_files = sorted(_get_rel_paths(colmap_image_dir))
            image_files = sorted(_get_rel_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            if len(colmap_files) != len(image_names):
                print(
                    f"Warning: colmap_files: {len(colmap_files)}, "
                    f"image_names: {len(image_names)}"
                )
                image_names = colmap_files
            image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        else:
            colmap_files = sorted(_get_rel_paths(colmap_image_dir))
            image_files = sorted(_get_rel_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.alpha_mask_paths = None  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (actual_width, actual_height)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class GaussianTrainingDataset:
    """Training images and cameras used to optimize the initial Gaussian scene."""

    def __init__(
        self,
        parser: Parser,
    ):
        self.parser = parser
        self.init_cams = _build_training_cameras(parser)

    def __len__(self):
        return len(self.init_cams)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        cam = self.init_cams[item]
        image_path = cam['image_name']
        return _tensor_camera_payload(cam, item, image_path=image_path)


def _build_training_cameras(parser: Parser) -> list[CameraRecord]:
    init_cams = []
    image_names = sorted(
        _get_rel_paths(os.path.join(parser.img_dir, "images")),
        key=_numeric_image_sort_key,
    )
    for i in range(len(parser.camera_ids)):
        camera_id = parser.camera_ids[i]
        K = parser.Ks_dict[camera_id].copy()
        camtoworlds = parser.camtoworlds[i]
        camera = _camera_record(
            K=K,
            camtoworlds=camtoworlds,
            idx=i,
            image_name=os.path.join(parser.img_dir, "images", image_names[i]),
        )
        init_cams.append(camera)
    return init_cams


def _read_planned_cameras(camera_path: str) -> list[CameraRecord]:
    planned_cameras = []
    cam_npz_files = sorted(f for f in os.listdir(camera_path) if f.endswith(".npz"))
    for cam_npz in cam_npz_files:
        cam_path = os.path.join(camera_path, cam_npz)
        cam = np.load(cam_path, allow_pickle=True)
        K = cam['K']
        camtoworlds = cam['camtoworlds']
        image = cam['image'].item()
        is_train = cam['is_train'].item()
        idx = cam["idx"].item()
        if is_train:
            planned_cameras.append(
                _camera_record(
                    K,
                    camtoworlds,
                    idx,
                    image=image,
                    is_train=is_train,
                )
            )
        else:
            pre_idx = cam['pre_idx'].item()
            aft_idx = cam['aft_idx'].item()
            planned_cameras.append(
                _camera_record(
                    K,
                    camtoworlds,
                    idx,
                    image=image,
                    is_train=is_train,
                    pre_idx=pre_idx,
                    aft_idx=aft_idx,
                )
            )
    return planned_cameras


def _save_planned_camera(camera_path: str, camera: CameraRecord) -> None:
    idx = int(camera["idx"])
    payload = {
        "K": camera["K"],
        "camtoworlds": camera["camtoworlds"],
        "image": camera["image"],
        "is_train": camera["is_train"],
        "idx": camera["idx"],
    }
    if not camera["is_train"]:
        payload["pre_idx"] = camera["pre_idx"]
        payload["aft_idx"] = camera["aft_idx"]
    np.savez(os.path.join(camera_path, f"{idx:04d}.npz"), **payload)


class CameraPlanningDataset:
    """Build and save the planned camera trajectory used by validation/refinement."""

    def __init__(
        self,
        parser: Parser,
        sphere_radius: float = 0.05,
        init_coverage_threshold: float = 0.0,
        new_coverage_threshold: float = 0.1,
        arc_distance_threshold: float = 0.20,
        nearest_camera_count: int = 2,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        xy_sample_range: float = 0.6,
        z_sample_range: float = 0.5,
        max_consecutive_failures: int = 30,
    ):
        self.parser = parser
        self.val_cameras = []
        self.anchor = []
        self.sphere_radius = sphere_radius
        self.init_coverage_threshold = init_coverage_threshold
        self.new_coverage_threshold = new_coverage_threshold
        self.arc_distance_threshold = arc_distance_threshold
        self.nearest_camera_count = nearest_camera_count
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.xy_sample_range = xy_sample_range
        self.z_sample_range = z_sample_range
        self.max_consecutive_failures = max_consecutive_failures
        self.init_cams = _build_training_cameras(parser)
        _log_training_camera_count(len(self.init_cams))
        self.camera_path = os.path.join(self.parser.data_dir, "add_camera")
        os.makedirs(self.camera_path, exist_ok=True)
        self.plan_and_save()

    def plan_and_save(self) -> None:
        plan_args = {
            "point_path": os.path.join(self.parser.data_dir, "sparse", "points.ply"),
            "mesh_path": os.path.join(self.parser.data_dir, "sparse", "mesh.ply"),
            "sphere_radius": self.sphere_radius,
        }
        self.camera_planner = CameraPlanner(plan_args)
        self.cam_planer = self.camera_planner
        self.camera_planner.sample_spheres()
        self.generate_out_cameras()
        for cam in self.val_cameras:
            _save_planned_camera(self.camera_path, cam)
        anchor_path = os.path.join(self.parser.data_dir, "anchor.json")
        with open(anchor_path, "w", encoding="utf-8") as f:
            json.dump(self.anchor, f, indent=4)
        self.valid_cam_idx()

    def valid_cam_idx(self) -> None:
        for i in range(len(self.val_cameras)):
            idx = self.val_cameras[i]["idx"]
            assert idx == i

    def compute_pose_distance(
        self,
        camtoworld1,
        camtoworld2,
        translation_weight=1.0,
        rotation_weight=1.0,
    ) -> float:
        t1, t2 = camtoworld1[:3, 3], camtoworld2[:3, 3]
        translation_dist = np.linalg.norm(t1 - t2)

        R1 = Rotation.from_matrix(camtoworld1[:3, :3])
        R2 = Rotation.from_matrix(camtoworld2[:3, :3])
        q1 = R1.as_quat()
        q2 = R2.as_quat()
        if np.dot(q1, q2) < 0:
            q2 = -q2

        dot_product_squared = np.dot(q1, q2)**2
        cos_theta = 2 * dot_product_squared - 1
        rotation_dist = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return (
            translation_weight * translation_dist
            + rotation_weight * rotation_dist
        )

    def find_nearest_cameras(
        self,
        camtoworld_mid: np.ndarray,
        val_cameras: List[Dict[str, Any]],
        N: int,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        if N <= 0 or not val_cameras:
            return []

        distances_with_cameras = []
        for camera_obj in val_cameras:
            camtoworld_target = camera_obj['camtoworlds']
            distance = self.compute_pose_distance(
                camtoworld_mid,
                camtoworld_target,
                translation_weight,
                rotation_weight,
            )
            distances_with_cameras.append((distance, camera_obj))

        distances_with_cameras.sort(key=lambda x: x[0])
        N = min(N, len(distances_with_cameras))
        return [camera_obj for dist, camera_obj in distances_with_cameras[:N]]

    def _random_endpoint(self, z_axis: int) -> list[float]:
        return [
            random.uniform(-self.xy_sample_range, self.xy_sample_range),
            random.uniform(-self.xy_sample_range, self.xy_sample_range),
            random.uniform(-self.z_sample_range, self.z_sample_range) * z_axis,
        ]

    def _append_training_camera(self, source_camera: CameraRecord, idx: int) -> None:
        self.val_cameras.append(
            _camera_record(
                K=source_camera["K"].copy(),
                camtoworlds=source_camera["camtoworlds"],
                image=None,
                is_train=True,
                idx=idx,
            )
        )

    def generate_arc_trajectory_inner(
        self,
        K: list,
        camtoworlds: list,
        idx: list,
        distance_threshold: float | None = None,
    ) -> tuple[list, int]:
        if distance_threshold is None:
            distance_threshold = self.arc_distance_threshold

        def generate_cams(camtoworlds1, camtoworlds2, K1, K2, idx):
            total_distance = self.compute_pose_distance(
                camtoworlds1, camtoworlds2
            )

            if total_distance < distance_threshold:
                N_new = 0
            else:
                N_steps = int(np.ceil(total_distance / distance_threshold))
                N_new = N_steps - 1

            if N_new <= 0:
                return [], idx

            R1 = camtoworlds1[:3, :3]
            t1 = camtoworlds1[:3, 3]
            R2 = camtoworlds2[:3, :3]
            t2 = camtoworlds2[:3, 3]

            rot1 = Rotation.from_matrix(R1)
            rot2 = Rotation.from_matrix(R2)
            key_times = [0, 1]
            key_rots = Rotation.concatenate([rot1, rot2])
            slerp = Slerp(key_times, key_rots)

            interp_factors = np.linspace(0, 1, N_new + 2)[1:-1]
            generated_cameras = []

            for i, s in enumerate(interp_factors):
                interp_K = (1 - s) * K1 + s * K2
                interp_rot_matrix = slerp(s).as_matrix()
                interp_t = (1 - s) * t1 + s * t2

                interp_camtoworlds = np.eye(4)
                interp_camtoworlds[:3, :3] = interp_rot_matrix
                interp_camtoworlds[:3, 3] = interp_t

                generated_cameras.append(
                    _camera_record(
                        K=interp_K,
                        camtoworlds=interp_camtoworlds,
                        image=None,
                        is_train=False,
                        idx=idx,
                    )
                )
                idx += 1
            return generated_cameras, idx

        K_start, K_mid, K_end = K
        camtoworlds_start, camtoworlds_mid, camtoworlds_end = camtoworlds
        idx_start, cur_idx, idx_end = idx
        int_cam_pre, cur_idx = generate_cams(
            camtoworlds_start,
            camtoworlds_mid,
            K_start,
            K_mid,
            cur_idx,
        )
        for i, camera in enumerate(int_cam_pre):
            if i == 0:
                camera['pre_idx'] = idx_start
                camera['aft_idx'] = camera['idx'] + 1
            else:
                camera['pre_idx'] = camera['idx'] - 1
                camera['aft_idx'] = camera['idx'] + 1
            self.val_cameras.append(camera)
        self.val_cameras.append(
            _camera_record(
                K=K_mid,
                camtoworlds=camtoworlds_mid,
                image=None,
                is_train=False,
                idx=cur_idx,
                pre_idx=cur_idx - 1,
                aft_idx=cur_idx + 1,
            )
        )
        self.anchor.append(cur_idx)
        cur_idx += 1
        int_cam_pre, cur_idx = generate_cams(
            camtoworlds_mid,
            camtoworlds_end,
            K_mid,
            K_end,
            cur_idx,
        )
        for i, camera in enumerate(int_cam_pre):
            if i == (len(int_cam_pre) - 1):
                camera['pre_idx'] = camera['idx'] - 1
                camera['aft_idx'] = idx_end
            else:
                camera['pre_idx'] = camera['idx'] - 1
                camera['aft_idx'] = camera['idx'] + 1
            self.val_cameras.append(camera)
        return cur_idx

    def generate_out_cameras(self) -> None:
        candidate_c2ws = []
        candidate_Ks = []
        cur_idx = 0
        for camera in self.init_cams:
            candidate_c2ws.append(camera["camtoworlds"])
            candidate_Ks.append(camera["K"])
        is_save, _ = self.camera_planner.evaluate_camera_view_whole(
            candidate_c2ws,
            candidate_Ks,
            self.init_coverage_threshold,
        )
        print("Finished initial camera coverage evaluation.")
        for i in range(len(self.init_cams)):
            self._append_training_camera(self.init_cams[i], cur_idx)
            cur_idx += 1

        z_axis = self.camera_planner.determine_obb_z_sign(
            self.init_cams[0]['camtoworlds']
        )
        failed_attempts = 0
        while True:
            point_end_param = self._random_endpoint(z_axis)
            point_start_param = self._random_endpoint(z_axis)
            camtoworlds_mid = self.camera_planner.create_cameras(
                point_start_param,
                point_end_param,
                z_axis,
            )
            K_mid = self.init_cams[0]['K'].copy()
            nearest_n_cameras = self.find_nearest_cameras(
                camtoworlds_mid,
                self.val_cameras,
                N=self.nearest_camera_count,
                translation_weight=self.translation_weight,
                rotation_weight=self.rotation_weight,
            )
            pre_cam = nearest_n_cameras[0]
            aft_cam = nearest_n_cameras[1]
            K_start = pre_cam['K'].copy()
            camtoworlds_start = pre_cam['camtoworlds']
            idx_start = pre_cam['idx']
            K_end = aft_cam['K'].copy()
            camtoworlds_end = aft_cam['camtoworlds']
            idx_end = aft_cam['idx']
            start_idx = cur_idx
            cur_idx = self.generate_arc_trajectory_inner(
                [K_start, K_mid, K_end],
                [camtoworlds_start, camtoworlds_mid, camtoworlds_end],
                [idx_start, cur_idx, idx_end],
            )
            candidate_c2ws = []
            candidate_Ks = []
            for k in range(start_idx, len(self.val_cameras), 1):
                candidate_c2ws.append(self.val_cameras[k]["camtoworlds"])
                candidate_Ks.append(self.val_cameras[k]["K"])
            is_save, _ = self.camera_planner.evaluate_camera_view_whole(
                candidate_c2ws,
                candidate_Ks,
                self.new_coverage_threshold,
            )
            if is_save:
                failed_attempts = 0
            else:
                del self.val_cameras[start_idx:]
                cur_idx = start_idx
                failed_attempts += 1
                del self.anchor[-1]
            print(f"Consecutive rejected camera proposals: {failed_attempts}")
            if failed_attempts >= self.max_consecutive_failures:
                break


class NovelViewDataset:
    """Validation/refinement dataset that reads already planned cameras."""

    def __init__(
        self,
        parser: Parser,
        is_phase1: bool = False,
    ):
        self.parser = parser
        self.val_cameras = []
        self.anchor = []
        self.init_cams = _build_training_cameras(parser)
        _log_training_camera_count(len(self.init_cams))
        camera_path = os.path.join(self.parser.data_dir, "add_camera")
        os.makedirs(camera_path, exist_ok=True)
        if not is_phase1:
            self.val_cameras = _read_planned_cameras(camera_path)
            anchor_path = os.path.join(self.parser.data_dir, "anchor.json")
            if os.path.exists(anchor_path):
                with open(anchor_path, "r", encoding="utf-8") as f:
                    self.anchor = json.load(f)
        self.valid_cameras = self.val_cameras

    def get_valid_cameras(self) -> None:
        self.valid_cameras = [
            camera for camera in self.val_cameras if not camera['is_train']
        ]

    def restore_original_cameras(self) -> None:
        self.valid_cameras = self.val_cameras

    def __len__(self) -> int:
        return len(self.valid_cameras)
    
    def set_imgs(self, img_path: list[str]) -> None:
        for tmp_path in img_path:
            img_name = os.path.basename(tmp_path)
            img_name = img_name.split("_")[1].split(".")[0]
            idx = int(img_name)
            assert (idx == self.val_cameras[idx]['idx'])
            self.val_cameras[idx]['image'] = tmp_path

    def __getitem__(self, item: int) -> Dict[str, Any]:
        cameras = self.valid_cameras[item]
        image_path = cameras['image']
        if image_path is None:
            return _tensor_camera_payload(cameras, item)
        return _tensor_camera_payload(cameras, item, image_path=image_path)


class RenderedNoiseDataset(torch.utils.data.Dataset):
    """Rendered noisy images used to train the diffusion LoRA stage."""

    def __init__(self, dataset_path, split: str, image_processor=None):
        super().__init__()
        self.gt_path = dataset_path[0]
        self.noise_path = dataset_path[1]
        self.input_imgs = []
        self.img_id = []
        self.image_processor = image_processor
        self.split = split
        if split == "train":
            dataset_path = os.path.join(self.noise_path, "Lora", "train")
            for obj in os.listdir(dataset_path):
                dataset_obj_path = os.path.join(dataset_path, obj, "images")
                for idx in os.listdir(dataset_obj_path):
                    dataset_idx_path = os.path.join(dataset_obj_path, idx)
                    self.input_imgs.append(dataset_idx_path)
                    self.img_id.append(obj)
        else:
            dataset_path = os.path.join(self.noise_path, "Lora", "val", "images")
            for idx in os.listdir(dataset_path):
                dataset_idx_path = os.path.join(dataset_path, idx)
                self.input_imgs.append(dataset_idx_path)
                self.img_id.append(idx.replace(".png", ""))

    def __len__(self) -> int:
        return len(self.input_imgs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        input_img_path = self.input_imgs[idx]
        input_img = Image.open(input_img_path).convert("RGB")
        input_img = self.image_processor.preprocess(input_img)
        if self.split == "train":
            gt_path = os.path.join(self.gt_path, "images", f"{self.img_id[idx]}.png")
            gt_img = Image.open(gt_path).convert("RGB")
            gt_img = self.image_processor.preprocess(gt_img)
            out = {
                "input_img": input_img,
                "gt_img": gt_img,
            }
        else:
            out = {
                "input_img": input_img,
            }

        return out


ColmapSceneParser = Parser
CameraPlannerDataset = CameraPlanningDataset
Train_Dataset = GaussianTrainingDataset
Val_Dataset = NovelViewDataset
NoiseDataset = RenderedNoiseDataset
