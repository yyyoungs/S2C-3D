import os
import random
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision.utils as vutils
import tqdm
import tyro
from PIL import Image
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.pipeline_difix import DifixPipeline
from model.pipeline_gs import Gaussian
from configs.config_3dgs import Config
from model.point_render import project_and_visualize_cuda, unproject_to_pointcloud


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Runner:
    """Runs iterative diffusion-guided Gaussian refinement."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

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
        self.difix = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
        self.difix.set_progress_bar_config(disable=True)
        model_path = cfg.diffx_model_path
        ckpt = load_file(model_path + f"/unet/model.safetensors", device='cpu')
        self.difix.unet.load_state_dict(ckpt, strict=False)
        ckpt = load_file(model_path + f"/vae/model.safetensors", device='cpu')
        self.difix.vae.load_state_dict(ckpt, strict=False)
        print(f'[INFO] Loaded checkpoint from {model_path}')
        self.difix.to("cuda")

    def train(self) -> None:
        cfg = self.cfg
        if not os.path.exists(cfg.gs_model_path):
            self.gs_model.train(init_step=0, max_steps=cfg.end_steps)
        else:
            self.gs_model.splats = torch.load(cfg.gs_model_path, map_location=self.device)
        self.fix(1000)

    def read_correct_imgs(self, point_render_path: str, point_mask_path: str) -> torch.Tensor:
        cor_img = Image.open(point_render_path).convert("RGB")
        cor_img = self.difix.image_processor.preprocess(cor_img)
        cor_mask = Image.open(point_mask_path)
        cor_mask = cor_mask.resize((cor_img.shape[3], cor_img.shape[2]), resample=Image.Resampling.NEAREST)
        cor_mask = [cor_mask]
        cor_mask = [np.array(image).astype(np.float32) / 255.0 for image in cor_mask]
        cor_mask = np.stack(cor_mask, axis=0)
        cor_mask = torch.from_numpy(cor_mask.transpose(0, 3, 1, 2))
        cor_img_final = torch.cat([cor_img, cor_mask], dim=1)
        return cor_img_final
    
    @torch.no_grad()
    def render_imgs(self, img_path: str, is_depth: bool = True) -> tuple[list[str], list[str]]:
        device = self.device
        sh_degree_to_use = self.cfg.sh_degree
        os.makedirs(img_path, exist_ok=True)
        image_r_path = []
        depth_r_path = []
        for eval_idx in range(len(self.gs_model.valset)):
            cameras = self.gs_model.valset.valid_cameras[eval_idx]
            K = cameras['K']
            novel_poses = cameras['camtoworlds']
            camtoworlds = torch.from_numpy(novel_poses).float().to(device).unsqueeze(0)
            K = torch.from_numpy(K).float().to(device)
            width, height = list(self.gs_model.parser.imsize_dict.values())[0]
            Ks = K[None].repeat(camtoworlds.shape[0], 1, 1)
            name = f"input_{cameras['idx']:04d}"
            renders, _, _ = self.gs_model.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                render_mode="RGB+ED",
            )

            for j in range(renders.shape[0]):
                colors = torch.clamp(renders[j, ..., 0:3], 0.0, 1.0)
                colors_path = os.path.join(img_path, f"{name}.png")
                colors_canvas = colors.cpu().numpy()
                colors_canvas = (colors_canvas * 255).astype(np.uint8)
                imageio.imwrite(colors_path, colors_canvas)
                image_r_path.append(colors_path)
                if is_depth:
                    depths = renders[..., 3:4]
                    depths = depths / self.gs_model.depth_max
                    depths = torch.clamp(depths[j], 0.0, 1.0)
                    depths_path = os.path.join(img_path, f"depth_{name}.png")
                    depth_canvas = depths.cpu().numpy()
                    depth_canvas = (depth_canvas * 255).astype(np.uint8)
                    imageio.imwrite(depths_path, depth_canvas[:, :, 0])
                    depth_r_path.append(depths_path)
        return image_r_path, depth_r_path

    def fix_train(self, step: int, refine_step: int, add_gs: bool = False) -> None:
        dataloader = torch.utils.data.DataLoader(
            self.gs_model.valset,
            batch_size=self.gs_model.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.gs_model.novelloaders.clear()
        self.gs_model.novelloaders_iter.clear()
        self.gs_model.novelloaders.append(dataloader)
        self.gs_model.novelloaders_iter.append(iter(dataloader))
        with torch.enable_grad():
            if add_gs:
                self.gs_model.train(init_step=step, max_steps=step + refine_step)
            else:
                self.gs_model.train(init_step=step, max_steps=step + refine_step, cfg_step=step + 1000)

    def point_render(self, image_path: list[str], depth_path: list[str], result_dir: str, is_pre: bool) -> None:
        os.makedirs(result_dir, exist_ok=True)
        device = self.device
        current_ref_idx = -1
        points_xyz = None
        points_rgb = None
        for eval_idx in range(len(self.gs_model.valset)):
            if is_pre:
                ref_idx = self.gs_model.valset.valid_cameras[eval_idx]["pre_idx"]
            else:
                ref_idx = self.gs_model.valset.valid_cameras[eval_idx]["aft_idx"]
            if ref_idx != current_ref_idx:
                current_ref_idx = ref_idx
                cameras = self.gs_model.valset.val_cameras[current_ref_idx]
                K = cameras['K']
                novel_poses = cameras['camtoworlds']
                camtoworlds = torch.from_numpy(novel_poses).float().to(device).unsqueeze(0)
                K = torch.from_numpy(K).float().to(device)
                Ks = K[None].repeat(camtoworlds.shape[0], 1, 1)
                image = torch.from_numpy(imageio.imread(image_path[current_ref_idx])[..., :3]).float()
                depth = torch.from_numpy(imageio.imread(depth_path[current_ref_idx])).float()
                image = image.to(device) / 255.0
                depth = depth.to(device) / 255.0
                depth = depth * self.gs_model.depth_max
                points_xyz, points_rgb = unproject_to_pointcloud(
                    rgb_tensor=image.unsqueeze(0),
                    depth_tensor=depth.unsqueeze(0).unsqueeze(-1),
                    K_tensor=Ks,
                    cam_to_world_tensor=camtoworlds,
                )

            cameras = self.gs_model.valset.valid_cameras[eval_idx]
            K = cameras['K']
            novel_poses = cameras['camtoworlds']
            camtoworlds = torch.from_numpy(novel_poses).float().to(device).unsqueeze(0)
            K = torch.from_numpy(K).float().to(device)
            Ks = K[None].repeat(camtoworlds.shape[0], 1, 1)
            name = f"input_{cameras['idx']:04d}.png"
            project_and_visualize_cuda(
                points_xyz=points_xyz.to(device),
                points_rgb=points_rgb.to(device),
                K2_tensor=Ks,
                cam_to_world2_tensor=camtoworlds,
                output_filename=result_dir,
                name_id=name,
                point_size=1,
                error_threshold=0.15,
            )

    def fix(self, step: int) -> None:
        print("Running fixer...")
        refine_steps = self.cfg.refine_steps
        end_refine_steps = self.cfg.end_refine_steps
        tb_writer = SummaryWriter(log_dir=f"{self.gs_model.results_dir}/tf_log")
        prompt = "remove degradation"
        with torch.no_grad():
            img_path = f"{self.gs_model.results_dir}/final/0/rendered"
            _, _ = self.render_imgs(img_path=img_path, is_depth=False)
            print("Finishing original rendering...")
            pbar_dppm = tqdm.tqdm(range(self.cfg.repeat_times))
            for x_step in pbar_dppm:
                self.gs_model.valset.restore_original_cameras()
                img_path = f"{self.gs_model.results_dir}/rendered/{x_step}"
                image_paths, _ = self.render_imgs(img_path=img_path, is_depth=False)
                image_paths_phase2 = []
                fixed_paths = f"{self.gs_model.results_dir}/fixed/{x_step}"
                os.makedirs(fixed_paths, exist_ok=True)
                fixed_path = []
                pred_noise = []
                for i in range(0, len(image_paths)):
                    image = Image.open(image_paths[i]).convert("RGB")
                    image_size = image.size
                    image_name = os.path.basename(image_paths[i])
                    image = self.difix.image_processor.preprocess(image)
                    pred_img, noise = self.difix(
                        prompt,
                        image=image,
                        save_noise=True,
                        num_inference_steps=1,
                        timesteps=[199],
                        guidance_scale=0.0,
                    )
                    pred_noise.append(noise)
                    if not self.gs_model.valset.val_cameras[i]['is_train']:
                        pred_img = torch.nn.functional.interpolate(
                            pred_img,
                            size=(image_size[1], image_size[0]),
                            mode="bilinear",
                        )
                        fix_img_path = os.path.join(fixed_paths, image_name)
                        fixed_path.append(fix_img_path)
                        vutils.save_image((pred_img + 1.0) / 2.0, fix_img_path)
                        image_paths_phase2.append(image_paths[i])
                self.gs_model.valset.set_imgs(fixed_path)
                self.gs_model.valset.get_valid_cameras()
                if x_step == self.cfg.repeat_times - 1:
                    self.fix_train(step, end_refine_steps, add_gs=True)
                else:
                    self.fix_train(step, refine_steps, add_gs=True)
                if x_step == self.cfg.repeat_times - 1:
                    self.gs_model.valset.restore_original_cameras()
                    img_path = f"{self.gs_model.results_dir}/final/{x_step}/rendered"
                    _, _ = self.render_imgs(img_path=img_path, is_depth=False)
                    break

                img_path = f"{self.gs_model.results_dir}/fixed-rendered/{x_step}"
                self.gs_model.valset.restore_original_cameras()
                point_img_path, point_dep_path = self.render_imgs(img_path=img_path, is_depth=True)
                result_dir_for = f"{self.gs_model.results_dir}/correct/{x_step}/pre"
                result_dir_bef = f"{self.gs_model.results_dir}/correct/{x_step}/aft"
                self.gs_model.valset.get_valid_cameras()
                self.point_render(point_img_path, point_dep_path, result_dir_for, True)
                self.point_render(point_img_path, point_dep_path, result_dir_bef, False)
                print("Finishing depth rendering!")
                loss_total = 0
                count_idx = 0
                re_fixed_paths = []
                fixed_paths = f"{self.gs_model.results_dir}/correct-fixed/{x_step}"
                os.makedirs(fixed_paths, exist_ok=True)
                for i in range(0, len(image_paths_phase2)):
                    pre_ref = self.gs_model.valset.valid_cameras[i]["pre_idx"]
                    aft_ref = self.gs_model.valset.valid_cameras[i]["aft_idx"]
                    current_noise = torch.cat([pred_noise[pre_ref], pred_noise[aft_ref]], dim=0)
                    img_path = image_paths_phase2[i]
                    img_name = os.path.basename(img_path)
                    image = Image.open(img_path).convert("RGB")
                    image_size = image.size
                    image = self.difix.image_processor.preprocess(image)

                    point_render_path = os.path.join(result_dir_for, "render", img_name)
                    point_mask_path = os.path.join(result_dir_for, "mask", img_name)
                    pre_imgs = self.read_correct_imgs(point_render_path, point_mask_path)
                    point_render_path = os.path.join(result_dir_bef, "render", img_name)
                    point_mask_path = os.path.join(result_dir_bef, "mask", img_name)
                    pre_imgs2 = self.read_correct_imgs(point_render_path, point_mask_path)

                    with torch.enable_grad():
                        pred_img, loss = self.difix(
                            prompt,
                            image=image,
                            correct_img=torch.cat([pre_imgs, pre_imgs2], dim=0),
                            correct_noise=current_noise,
                            num_inference_steps=1,
                            timesteps=[199],
                            guidance_scale=0.0,
                        )
                    loss_total = loss_total + loss
                    count_idx = count_idx + 1
                    pred_img = torch.nn.functional.interpolate(
                        pred_img,
                        size=(image_size[1], image_size[0]),
                        mode="bilinear",
                    )
                    re_fixed_path = os.path.join(fixed_paths, img_name)
                    re_fixed_paths.append(re_fixed_path)
                    vutils.save_image((pred_img + 1.0) / 2.0, re_fixed_path)

                self.gs_model.valset.set_imgs(re_fixed_paths)
                self.gs_model.valset.get_valid_cameras()
                self.fix_train(step, refine_steps, add_gs=True)
                desc = f"loss={(loss_total / count_idx):.3f}|"
                tb_writer.add_scalar('Metric/PSNR', loss_total / count_idx, x_step)
                pbar_dppm.set_description(desc)

        gs_save_path = os.path.join(self.gs_model.results_dir, "gs")
        os.makedirs(gs_save_path, exist_ok=True)
        torch.save(self.gs_model.splats, os.path.join(gs_save_path, "gs.pth"))
        width, height = list(self.gs_model.parser.imsize_dict.values())[0]
        self.gs_model.valset.restore_original_cameras()
        render_path = [f"{self.gs_model.results_dir}/rendered/{x_step}" for x_step in range(self.cfg.repeat_times)]
        render_name_sets = [
            {p.name for p in Path(path).glob("*.png")}
            for path in render_path
            if Path(path).is_dir()
        ]
        common_render_names = set.intersection(*render_name_sets) if render_name_sets else set()
        image_names = [
            f"input_{camera['idx']:04d}.png"
            for camera in self.gs_model.valset.valid_cameras
        ]
        image_names = list(dict.fromkeys(name for name in image_names if name in common_render_names))
        separate_path = os.path.join(self.gs_model.results_dir, "vis")
        os.makedirs(separate_path, exist_ok=True)
        for i in range(0, len(image_names), 8):
            tmp_names = image_names[i:i + 8]
            idx_vis = int(i / 8)
            self.gs_model.save_final_result(
                [height, width],
                render_path,
                os.path.join(separate_path, f"final_{idx_vis}.png"),
                image_name_list=tmp_names,
            )




def main(local_rank: int, world_rank, world_size: int, cfg: Config) -> None:
    runner = Runner(local_rank, world_rank, world_size, cfg)
    runner.train()


if __name__ == "__main__":
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
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
