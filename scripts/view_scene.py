import os
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.pipeline_gs import Gaussian
from configs.config_3dgs import Config
import dearpygui.dearpygui as dpg
import imageio
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
class OrbitCamera:
    """Orbit camera that edits the 4x4 camera-to-world matrix directly."""
    def __init__(self, c2w: np.ndarray, fovy: float):
        self.camtoworlds = c2w.copy().astype(np.float32)
        self.fovy = fovy
        self.focal_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.rotation_speed = 0.002
        self.pan_speed = 0.001
        self.scale_speed = 0.05

    @property
    def position(self) -> np.ndarray:
        return self.camtoworlds[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        return self.camtoworlds[:3, :3]
        
    @property
    def forward_vec(self) -> np.ndarray:
        return -self.rotation[:, 2]

    def _orbit_distance(self) -> float:
        distance = np.linalg.norm(self.position - self.focal_point)
        return max(float(distance), 1e-4)

    def _set_rotation(self, rotation_matrix: np.ndarray):
        u, _, vh = np.linalg.svd(rotation_matrix)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        self.camtoworlds[:3, :3] = rotation.astype(np.float32)

    def orbit(self, dx: float, dy: float):
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right_vec = self.rotation[:, 0]
        rot_yaw = R.from_rotvec(-dx * self.rotation_speed * world_up)
        rot_pitch = R.from_rotvec(-dy * self.rotation_speed * right_vec)
        vec_to_cam = self.position - self.focal_point
        vec_to_cam = rot_yaw.apply(vec_to_cam)
        vec_to_cam = rot_pitch.apply(vec_to_cam)
        self.camtoworlds[:3, 3] = self.focal_point + vec_to_cam
        self.look_at(self.focal_point)

    def scale(self, delta: float):
        
        movement = -delta * self.scale_speed * self.forward_vec
        self.camtoworlds[:3, 3] += movement

    def move_vertical(self, amount: float):
        """Move the camera along the world Y axis."""
        world_up = np.array([0, 1, 0])
        translation = amount * self.pan_speed * world_up
        
        
        self.camtoworlds[:3, 3] += translation
        self.focal_point += translation

    def move_forward(self, amount: float):
        """Move the camera along its viewing direction."""
        translation = amount * self.pan_speed * self.forward_vec
        self.camtoworlds[:3, 3] += translation
        self.focal_point += translation

    def move_sideways(self, amount: float):
        """Move the camera sideways on the horizontal world plane."""
        world_up = np.array([0, 1, 0])
        horizontal_right = np.cross(self.forward_vec, world_up)
        if np.linalg.norm(horizontal_right) < 1e-6:
             horizontal_right = self.rotation[:, 0]
        else:
            horizontal_right = horizontal_right / np.linalg.norm(horizontal_right)
        
        translation = amount * self.pan_speed * horizontal_right
        
        
        self.camtoworlds[:3, 3] += translation
        self.focal_point += translation

    def pan(self, dx: float, dy: float):
        """Pan camera with mouse drag deltas."""
        self.move_sideways(dx)
        self.move_vertical(-dy)

    def yaw(self, amount: float):
        rot_yaw = R.from_rotvec(-amount * self.rotation_speed * np.array([0, 1, 0]))
        vec_to_cam = self.position - self.focal_point
        vec_to_cam = rot_yaw.apply(vec_to_cam)
        self.camtoworlds[:3, 3] = self.focal_point + vec_to_cam
        self.look_at(self.focal_point)

    def rotate_in_place(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0):
        """Rotate the viewing direction without moving the camera center."""
        distance = self._orbit_distance()
        rotation = R.identity()
        if yaw:
            rotation = R.from_rotvec(-yaw * self.rotation_speed * np.array([0, 1, 0])) * rotation
        if pitch:
            rotation = R.from_rotvec(-pitch * self.rotation_speed * self.rotation[:, 0]) * rotation
        if roll:
            rotation = R.from_rotvec(roll * self.rotation_speed * self.forward_vec) * rotation
        self._set_rotation(rotation.as_matrix() @ self.rotation)
        self.focal_point = self.position + self.forward_vec * distance

    def look_at(self, target: np.ndarray):
        
        if np.allclose(self.position, target): return
        forward_vec = target - self.position
        forward_vec /= np.linalg.norm(forward_vec)
        world_up = np.array([0, 1, 0])
        right_vec = np.cross(forward_vec, world_up)
        if np.linalg.norm(right_vec) < 1e-6:
             if forward_vec[1] > 0.99: right_vec = np.cross(np.array([0, 0, -1]), forward_vec)
             else: right_vec = np.cross(np.array([0, 0, 1]), forward_vec)
        right_vec /= np.linalg.norm(right_vec)
        up_vec = np.cross(right_vec, forward_vec)
        new_rot = np.stack([right_vec, -up_vec, -forward_vec], axis=1)
        self._set_rotation(new_rot)
        

def get_fovy_from_K(K, image_height):
    fy = K[1, 1]
    fovy_rad = 2 * np.arctan((image_height / 2) / fy)
    return fovy_rad

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.gs_model = Gaussian(
            cfg=self.cfg,
            device=self.device,
            world_rank=self.world_rank,
            world_size=self.world_size,
        )
        self.gs_model.splats = torch.load(cfg.gs_model_path, map_location=self.device)
        self.mode = "image"
        self.need_update = True
        self.W = 960
        self.H = 640
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.num_cameras = len(self.gs_model.valset.val_cameras)
        self.current_idx = 0 
        initial_camera_dict = self.gs_model.valset.val_cameras[self.current_idx]
        initial_c2w = initial_camera_dict['camtoworlds']
        initial_K = initial_camera_dict['K']
        
        
        width, height = list(self.gs_model.parser.imsize_dict.values())[0]
        initial_fovy = get_fovy_from_K(initial_K, height)

        
        self.cam = OrbitCamera(c2w=initial_c2w, fovy=initial_fovy)
        self.initial_K = torch.from_numpy(initial_K).float().to(self.device)
        
        
        self.selected_camera_idx = 0
        self.selected_anchor_camera_idx = (
            int(self.gs_model.valset.anchor[0]) if self.gs_model.valset.anchor else 0
        )
        self.save_camera_idx = 0
        camera_path = self.gs_model.parser.data_dir + f"/add_camera"
        if os.path.exists(camera_path):
            for cam_npz in os.listdir(camera_path):
                self.save_camera_idx +=1
        
        # self.initial_camtoworlds = self.cam.camtoworlds.copy()
        # self.initial_focal_point = self.cam.focal_point.copy()
        if self.gs_model.cfg.is_gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()
    
    
    def apply_selected_camera(self, camera_idx: int | None = None):
        """Switch to the selected GUI camera."""
        idx = self.selected_camera_idx if camera_idx is None else int(camera_idx)
        idx = max(0, min(idx, len(self.gs_model.valset.val_cameras) - 1))
        self.selected_camera_idx = idx
        print(f"Switching to camera {idx}...")
        
        
        target_camera_dict = self.gs_model.valset.val_cameras[idx]
        new_c2w = target_camera_dict['camtoworlds']
        new_K = target_camera_dict['K']

        
        self.cam.camtoworlds = new_c2w.copy()
        self.initial_K = torch.from_numpy(new_K).float().to(self.device)
        width, height = list(self.gs_model.parser.imsize_dict.values())[0]
        self.cam.fovy = get_fovy_from_K(new_K, height)
        
        self.cam.focal_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.need_update = True

    def apply_selected_anchor_camera(self):
        """Switch to the selected anchor camera."""
        self.apply_selected_camera(self.selected_anchor_camera_idx)

    def _display_buffer_from_render(self, colors, depths):
        if self.mode == "depth" and depths is not None:
            depth_map = depths.squeeze(0).squeeze(0)
            valid_depth = depth_map[torch.isfinite(depth_map) & (depth_map > 0)]
            if valid_depth.numel() > 0:
                near = torch.quantile(valid_depth, 0.02)
                far = torch.quantile(valid_depth, 0.98)
                depth_image = (far - depth_map) / torch.clamp(far - near, min=1e-6)
                depth_image = torch.where(torch.isfinite(depth_image), depth_image, torch.zeros_like(depth_image))
                depth_image = depth_image.clamp(0, 1)
            else:
                depth_image = torch.zeros_like(depth_map)
            image = depth_image[None, None].repeat(1, 3, 1, 1)
        else:
            image = colors.permute(0, 3, 1, 2)

        buffer_image = F.interpolate(
            image, size=(self.H, self.W), mode="bilinear", align_corners=False
        ).squeeze(0)
        return (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def apply_selected_button(self):
        idx = self.selected_camera_idx
        pre_idx = int(idx / (self.cfg.val_camera_nums+1))
        # print(pre_idx)
        aft_idx = pre_idx + 1
        if pre_idx >= len(self.gs_model.parser.camera_ids)-1:
            pre_idx = len(self.gs_model.parser.camera_ids)-1
            aft_idx = 0
        print(pre_idx,aft_idx)
        novel_poses = self.cam.camtoworlds 
        K = self.initial_K 
        pre_arr = np.array(pre_idx, dtype=np.int32)
        aft_arr = np.array(aft_idx, dtype=np.int32)
        camera_path = self.cfg.data_dir + f"/add_camera"
        os.makedirs(camera_path,exist_ok=True)
        np.savez(
        camera_path + f"/{self.save_camera_idx:04d}.npz",
        pre_idx=pre_arr,
        aft_idx=aft_arr,
        novel_poses=novel_poses,
        K=K.cpu()
        )
        self.save_camera_idx = self.save_camera_idx+1
        
        
        

    def __del__(self):
        dpg.destroy_context()

    
    @torch.no_grad()
    def test_step(self):
        if not self.need_update:
            return
        device = self.device
        sh_degree_to_use = self.cfg.sh_degree
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        if self.need_update:
            novel_poses = self.cam.camtoworlds 
            K = self.initial_K 
            camtoworlds = torch.from_numpy(novel_poses).float().to(device).unsqueeze(0)
            width, height = list(self.gs_model.parser.imsize_dict.values())[0]
            Ks = K[None].repeat(camtoworlds.shape[0], 1, 1)
            renders, alphas, info = self.gs_model.rasterize_splats(
                    camtoworlds=camtoworlds, Ks=Ks, width=width, height=height, sh_degree=sh_degree_to_use,
                    near_plane=self.cfg.near_plane, far_plane=self.cfg.far_plane, render_mode="RGB+ED"
                )
            colors, depths = renders[..., 0:3], renders[..., 3:4].permute(0, 3, 1, 2)
            self.buffer_image = self._display_buffer_from_render(colors, depths)
            self.need_update = False
        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
        dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
        dpg.set_value("_texture", self.buffer_image)
        
    @torch.no_grad()
    def test_data(self,img_path):
        device = self.device
        sh_degree_to_use = self.cfg.sh_degree
        total_psnr = 0
        total_ssim = 0
        for eval_idx in range(len(self.gs_model.valset.test_cams)):
            cameras = self.gs_model.valset.test_cams[eval_idx]
            K = cameras['K']
            img_name = cameras['image_name']
            novel_poses = cameras['camtoworlds']
            camtoworlds = torch.from_numpy(novel_poses).float().to(device).unsqueeze(0)
            K = torch.from_numpy(K).float().to(device)
            width, height = list(self.gs_model.parser.imsize_dict.values())[0]
            Ks = K[None].repeat(camtoworlds.shape[0], 1, 1)
            name = f"input_{cameras['idx']:04d}"
            renders, alphas, info = self.gs_model.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                    render_mode="RGB+ED",
                )
            if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            for j in range(renders.shape[0]):
                colors = torch.clamp(renders[j, ..., 0:3], 0.0, 1.0)
                pred_path = f"{img_path}/pred"# [H, W, 3]
                os.makedirs(pred_path,exist_ok=True)
                pred_path = pred_path + f"/{name}.png"
                gt_path = f"{img_path}/gt"# [H, W, 3]
                os.makedirs(gt_path,exist_ok=True)
                gt_path = gt_path + f"/{name}.png"
                com_path = f"{img_path}/com"# [H, W, 3]
                os.makedirs(com_path,exist_ok=True)
                com_path = com_path + f"/{name}.png"
                gt = imageio.imread(img_name)[..., :3]
                gt = torch.from_numpy(gt).float()
                gt = gt.to(device) / 255.0
                com = torch.concat([colors,gt],dim=1)
                preds_nchw = colors.permute(2, 0, 1).unsqueeze(0)
                target_nchw = gt.permute(2, 0, 1).unsqueeze(0)
                psnr_func = peak_signal_noise_ratio(preds_nchw, target_nchw, data_range=1.0)
                ssim_func = structural_similarity_index_measure(preds_nchw, target_nchw, data_range=1.0)
                print(f"{name}: PSNR: {psnr_func.item():.4f},SSIM: {ssim_func.item():.4f}")
                total_psnr += psnr_func.item()
                total_ssim += ssim_func.item()
                colors_canvas = colors.cpu().numpy()
                colors_canvas = (colors_canvas * 255).astype(np.uint8)
                imageio.imwrite(pred_path, colors_canvas)
                gt = gt.cpu().numpy()
                gt = (gt* 255).astype(np.uint8)
                imageio.imwrite(gt_path, gt)
                com = com.cpu().numpy()
                com = (com* 255).astype(np.uint8)
                imageio.imwrite(com_path, com)
        total_ssim = total_ssim / len(self.gs_model.valset.test_cams)
        total_psnr = total_psnr / len(self.gs_model.valset.test_cams)
        print(f"Total : PSNR: {total_psnr:.4f}, SSIM : {total_ssim:.4f}")
            

    def register_dpg(self):
        
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W, self.H, self.buffer_image, format=dpg.mvFormat_Float_rgb, tag="_texture"
            )
        with dpg.window(
            tag="_primary_window", width=self.W, height=self.H, pos=[0, 0],
            no_move=True, no_title_bar=True, no_scrollbar=True
        ):
            dpg.add_image("_texture")

        # control window
        with dpg.window(
            label="Control", tag="_control_window", width=600, height=self.H, pos=[self.W, 0],
            no_move=True, no_title_bar=True
        ):
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            dpg.add_separator()
            with dpg.group(horizontal=True):
                
                def callback_select_camera(sender, app_data):
                    self.selected_camera_idx = int(app_data)
                    self.apply_selected_camera()

                dpg.add_text("In Cams:")
                dpg.add_slider_int(
                    default_value=self.selected_camera_idx,
                    min_value=0,
                    max_value=max(0, len(self.gs_model.valset.val_cameras) - 1),
                    callback=callback_select_camera,
                    width=420
                )
            dpg.add_separator()
            if self.gs_model.valset.anchor:
                with dpg.group(horizontal=True):
                    anchor_min = min(self.gs_model.valset.anchor)
                    anchor_max = max(self.gs_model.valset.anchor)

                    def callback_select_anchor_camera(sender, app_data):
                        selected = int(app_data)
                        anchors = self.gs_model.valset.anchor
                        self.selected_anchor_camera_idx = min(anchors, key=lambda idx: abs(idx - selected))
                        self.apply_selected_anchor_camera()

                    dpg.add_text("Anchor Cams:")
                    dpg.add_slider_int(
                        default_value=self.selected_anchor_camera_idx,
                        min_value=anchor_min,
                        max_value=anchor_max,
                        callback=callback_select_anchor_camera,
                        width=420
                    )
                dpg.add_separator()

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                dpg.add_combo(
                    ("image", "depth"), label="mode", default_value=self.mode, callback=callback_change_mode
                )
        
        
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"): return
            dx, dy = app_data[1], app_data[2]
            self.cam.orbit(dx, dy)
            self.need_update = True
        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"): return
            delta = app_data
            self.cam.scale(delta)
            self.need_update = True
        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"): return
            dx, dy = app_data[1], app_data[2]
            self.cam.pan(dx, dy)
            self.need_update = True
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)
        dpg.create_viewport(
            title="Gaussian3D", width=self.W + 600, height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.setup_dearpygui()
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)
        dpg.show_viewport()
    def _handle_keyboard_input(self):
        """Handle keyboard navigation controls on each GUI frame."""
        pan_step = 60.0
        rotate_step = 25.0
        triggered = False

        if dpg.is_key_down(dpg.mvKey_W):
            self.cam.move_forward(pan_step)
            triggered = True
        
        if dpg.is_key_down(dpg.mvKey_S):
            self.cam.move_forward(-pan_step)
            triggered = True
        
        # --- Sideways Pan (A/D) ---
        
        if dpg.is_key_down(dpg.mvKey_A):
            self.cam.move_sideways(-pan_step)
            triggered = True
        
        if dpg.is_key_down(dpg.mvKey_D):
            self.cam.move_sideways(pan_step)
            triggered = True

        if dpg.is_key_down(dpg.mvKey_Q):
            self.cam.move_vertical(-pan_step)
            triggered = True
        
        if dpg.is_key_down(dpg.mvKey_E):
            self.cam.move_vertical(pan_step)
            triggered = True

        if dpg.is_key_down(dpg.mvKey_J):
            self.cam.rotate_in_place(yaw=-rotate_step)
            triggered = True
        
        if dpg.is_key_down(dpg.mvKey_L):
            self.cam.rotate_in_place(yaw=rotate_step)
            triggered = True

        if dpg.is_key_down(dpg.mvKey_I):
            self.cam.rotate_in_place(pitch=-rotate_step)
            triggered = True
        
        if dpg.is_key_down(dpg.mvKey_K):
            self.cam.rotate_in_place(pitch=rotate_step)
            triggered = True

        if dpg.is_key_down(dpg.mvKey_U):
            self.cam.rotate_in_place(roll=-rotate_step)
            triggered = True
        
        if dpg.is_key_down(dpg.mvKey_O):
            self.cam.rotate_in_place(roll=rotate_step)
            triggered = True
        
        if triggered:
            self.need_update = True


    def render(self):
        if self.gs_model.cfg.is_gui:
            while dpg.is_dearpygui_running():
                self._handle_keyboard_input()
                self.test_step()
                dpg.render_dearpygui_frame()
        else:
            image_path = self.cfg.result_dir + "/test_result"
            os.makedirs(image_path,exist_ok=True)
            self.test_data(image_path)

        
            
        
        
            




def main(local_rank: int, world_rank, world_size: int, cfg: Config):

    runner = Runner(local_rank, world_rank, world_size, cfg)
    runner.render()
    






if __name__ == "__main__":
    # Config objects we can choose between.
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using MCMC densification.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
