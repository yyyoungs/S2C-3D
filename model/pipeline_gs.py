import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import assert_never

if hasattr(torch, "mps") and not hasattr(torch.mps, "is_available"):
    torch.mps.is_available = lambda: False

from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from gsplat.optimizers import SelectiveAdam
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from model.datasets import Parser, Train_Dataset, Val_Dataset
from model.utils import knn, rgb_to_sh


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 5e-4 ),
        ("scales", torch.nn.Parameter(scales), 1e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 1e-2),
    ]


    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        # params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3 / 50))
        # params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20 / 50))
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 1e-4))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 1e-4))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

class Gaussian:

    def __init__(
        self,
        cfg,
        device,
        world_rank,
        world_size,
        is_phase1=False,
    ):
        self.cfg = cfg
        self.device = device
        self.world_rank = world_rank
        self.world_size = world_size
        self.results_dir = self.cfg.result_dir
        os.makedirs(self.results_dir, exist_ok=True)
        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            img_dir=cfg.img_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Train_Dataset(self.parser)
        self.valset = Val_Dataset(self.parser, is_phase1=is_phase1)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)
        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        self.novelloaders = []
        self.novelloaders_iter = []
        self.depth_max = 10.0
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        self.trainloader_iter = iter(self.trainloader)
            
        
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info
    def create_random_mask_and_apply(self,image_tensor, gt_tensor, cocient):
        """Randomly keep a fraction of tensor entries and apply the same mask to image and GT."""
        device = image_tensor.device
        shape = image_tensor.shape
        total_elements = image_tensor.numel()
        num_non_zero = int(total_elements*cocient)
        
        if num_non_zero < 0:
            num_non_zero = 0
        if num_non_zero > total_elements:
            print(
                f"Warning: num_non_zero ({num_non_zero}) exceeds total elements "
                f"({total_elements}); using {total_elements}."
            )
            num_non_zero = total_elements

        indices_to_keep = torch.randperm(total_elements, device=device)[:num_non_zero]
        mask_flat = torch.zeros(total_elements, device=device)
        mask_flat[indices_to_keep] = 1
        mask = mask_flat.view(shape)
        masked_image = image_tensor * mask
        masked_gt = gt_tensor * mask
        return masked_image, masked_gt
    
    def render_loss(self,data,device,step,sh_degree_to_use,is_mask=False,is_depth=False):
        camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
        # depths_gt = data["depth"].to(device)
        if is_depth:
            depth = data["depth"].to(device)/ 255.0
            depth = depth * self.depth_max
            # points = data["points"].to(device)  # [1, M, 2]
            # depths_gt = data["depths"].to(device)  # [1, M]
        # num_train_rays_per_step = (
        #     pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
        # )
        image_ids = data["image_id"].to(device)
        masks = (data["mask"].to(device))/255.0 if "mask" in data else None  # [1, H, W]
        # masks = masks/ 255.0
        height, width = pixels.shape[1:3]


        # forward
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if is_depth else "RGB",
            # masks=masks,
        )
        if renders.shape[-1] == 4:
            colors, render_depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, render_depths = renders, None
        self.cfg.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        # loss
        if not is_mask:
            l1loss = F.l1_loss(colors, pixels)
        else:
            # colors, pixels = self.create_random_mask_and_apply(colors,pixels,0.5)
            l1loss = F.l1_loss(colors*masks, pixels*masks)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda
        if is_depth:
            render_depths = render_depths.squeeze(3)
            disp = torch.where(render_depths > 0.0, 1.0 / render_depths, torch.zeros_like(render_depths))
            disp_gt = 1.0 / depth  # [1, M]
            depthloss = F.l1_loss(disp, disp_gt)
            # depthloss = F.l1_loss(depths, depths_gt)*0.1
            loss = loss + depthloss
            
        return loss,l1loss,ssimloss,info
        
    @torch.no_grad()
    def save_final_result(
        self, 
        image_size: tuple[int, int],
        folder_paths: list[str],
        output_filename: str,
        image_name_list: list[str] | None = None 
    ):
        """Create an image grid from matching filenames across multiple folders."""
        print("Creating image grid...")

        if not folder_paths:
            print("Error: folder_paths is empty.")
            sys.exit(1)
        path_objects = [Path(p) for p in folder_paths]
        for p in path_objects:
            if not p.is_dir():
                print(f"Error: folder '{p}' does not exist or is not a directory.")
                sys.exit(1)

        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        list_of_file_dicts = []
        for p in path_objects:
            files = {f.name: f for f in p.iterdir() if f.suffix.lower() in supported_extensions}
            list_of_file_dicts.append(files)
            print(f"Found {len(files)} images in folder '{p.name}'.")

        if not list_of_file_dicts:
            print("Error: failed to read files from the given folders.")
            sys.exit(1)
        key_sets = [set(d.keys()) for d in list_of_file_dicts]
        all_common_filenames = sorted(list(set.intersection(*key_sets)))

        filenames_to_process = []
        if image_name_list is not None:
            print("\nFiltering with the provided image_name_list...")
            all_common_set = set(all_common_filenames)
            for name in image_name_list:
                if name in all_common_set:
                    filenames_to_process.append(name)
                else:
                    print(f"  -> Warning: '{name}' was not found in all folders; skipping.")
        else:
            print("\nNo image_name_list provided; using all common filenames.")
            filenames_to_process = all_common_filenames

        if not filenames_to_process:
            print("\nError: no valid common filenames remain after filtering.")
            sys.exit(1)

        print(f"\nFound {len(filenames_to_process)} valid image groups.")

        num_cols = len(folder_paths)
        num_rows = len(filenames_to_process) 
        img_height, img_width = image_size[0], image_size[1]
        canvas_width = img_width * num_cols
        canvas_height = img_height * num_rows

        print(f"Creating {canvas_width}x{canvas_height} canvas ({num_cols} cols x {num_rows} rows).")
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        current_y = 0
        for i, filename in enumerate(filenames_to_process):
            print(f"Processing row {i + 1}/{num_rows}: {filename}")
            
            try:
                for col_idx, file_dict in enumerate(list_of_file_dicts):
                    img_path = file_dict[filename]
                    img = Image.open(img_path).convert('RGB')
                    
                    if img.size != (img_width, img_height):
                        img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)

                    paste_x = col_idx * img_width
                    canvas.paste(img, (paste_x, current_y))
                
                current_y += img_height
            except Exception as e:
                print(f"  -> Failed to process {filename}: {e}")

        output_dir = Path(output_filename).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            canvas.save(output_filename, quality=95)
            print(f"\nSaved image grid to '{output_filename}'")
        except Exception as e:
            print(f"Failed to save '{output_filename}': {e}")
            
    def get_outside_indices_spheres(
        self,
        points: torch.Tensor,
        center_indices: torch.Tensor,
        radii: torch.Tensor,
    ) -> list:
        """Return point indices outside all spheres defined by centers and radii."""
        M = points.size(0)
        N = center_indices.size(0)
        
        if center_indices.dim() != 1 or radii.dim() != 1 or radii.size(0) != N:
            raise ValueError("center_indices and radii must both be 1D tensors with length N.")

        centers = points[center_indices]
        is_inside_any_sphere = torch.zeros(M, dtype=torch.bool, device=points.device)

        for i in range(N):
            center = centers[i]
            R = radii[i]         
            displacements = points - center  
            squared_distance = torch.sum(torch.square(displacements), dim=1)
            is_inside_current = squared_distance <= torch.square(R)
            is_inside_any_sphere = torch.logical_or(is_inside_any_sphere, is_inside_current)

        is_outside_all = torch.logical_not(is_inside_any_sphere)
        outside_indices = torch.nonzero(is_outside_all, as_tuple=False).squeeze(1)
        return outside_indices.tolist()

    def apply_random_black_masks_to_image(self,
                                        image_tensor: torch.Tensor,
                                        num_masks: int = 8,  
                                        min_mask_size_ratio: float = 0.05,  
                                        max_mask_size_ratio: float = 0.10,  
                                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random black rectangular masks to an RGB image tensor."""
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("Input image_tensor must be a torch.Tensor.")
        if image_tensor.dim() != 3 or image_tensor.shape[2] != 3:
            raise ValueError("Input image_tensor must have shape [H, W, 3].")
        if num_masks <= 0:
            H, W, _ = image_tensor.shape
            mask_tensor = torch.ones((H, W, 1), dtype=torch.float32, device=image_tensor.device)
            return image_tensor.clone(), mask_tensor

        H, W, C = image_tensor.shape
        output_tensor = image_tensor.clone()
        mask_tensor = torch.ones((H, W, 1), dtype=torch.float32, device=image_tensor.device) 

        original_dtype = output_tensor.dtype
        scale_factor = 1.0
        
        if output_tensor.max() > 1.0:  
            scale_factor = 255.0
            output_tensor = output_tensor.float() / scale_factor
        else:  
            output_tensor = output_tensor.float()

        for _ in range(num_masks):
            mask_h_ratio = random.uniform(min_mask_size_ratio, max_mask_size_ratio)
            mask_w_ratio = random.uniform(min_mask_size_ratio, max_mask_size_ratio)

            mask_h = int(H * mask_h_ratio)
            mask_w = int(W * mask_w_ratio)
            mask_h = max(1, mask_h)
            mask_w = max(1, mask_w)

            y_start_max = max(0, H - mask_h) 
            if y_start_max == 0:
                y_start = 0
            else:
                y_start = random.randint(0, y_start_max)
            
            
            x_start_max = max(0, W - mask_w)
            if x_start_max == 0:
                x_start = 0
            else:
                x_start = random.randint(0, x_start_max)

            y_end = y_start + mask_h
            x_end = x_start + mask_w

            output_tensor[y_start:y_end, x_start:x_end, :] = 0.0
            mask_tensor[y_start:y_end, x_start:x_end, :] = 0.0

        integer_dtype = original_dtype in (torch.uint8, torch.int8, torch.short, torch.int, torch.long)
        if integer_dtype and scale_factor > 1.0:
            output_tensor = (output_tensor * scale_factor).clamp(0, scale_factor).to(original_dtype)
        else:
            output_tensor = output_tensor.to(original_dtype)

        return output_tensor, mask_tensor


    @torch.no_grad()
    def render_noise(self):

        trainloader = self.trainloader
        trainloader_iter = iter(trainloader)
        max_steps = len(trainloader_iter)
        # Training loop.
        global_tic = time.time()
        device = self.device
        sh_degree_to_use = self.cfg.sh_degree
        add_noise_step = self.cfg.add_noise_steps
        remove_step = self.cfg.remove_steps
        mask_step = self.cfg.add_mask_steps
        # sample_cocient = 0.0
        sample_add = 0
        pbar = tqdm.tqdm(range(0, add_noise_step))
        init_gs_params = {}
        mean_gs_params = {}
        std_gs_params = {}
        gs_params = ["means","scales","quats","opacities","sh0","shN"]
        gs_num = self.splats["means"].shape[0]
        path = self.results_dir
        for para_name in gs_params:
            init_gs_params[para_name] = self.splats[para_name].detach().clone()
            mean_gs_params[para_name] = torch.mean(self.splats[para_name].detach().clone(),dim=0)
            std_gs_params[para_name] = torch.std(self.splats[para_name].detach().clone(),dim=0)
        for idx in pbar:
            shuffled_indices = torch.randperm(gs_num)
            if idx >= remove_step:
                sample_add = sample_add+1
            sample_cocient = min((0.0 + 0.1*sample_add),1.0)
            if sample_cocient!=1.0:
                selected_indices = shuffled_indices[:int(gs_num*sample_cocient)]
            else:
                M = gs_num
                N = random.randint(10, 20)
                center_indices = torch.randperm(M, device=device)[:N]
                radii = torch.rand(N, dtype=torch.float32, device=device) * 0.5+0.1
                selected_indices = self.get_outside_indices_spheres(
                    points=self.splats["means"].detach().clone(), 
                    center_indices=center_indices, 
                    radii=radii
                )
            for para_name in gs_params:
                if idx < remove_step:
                    if para_name!="sh0" and para_name!="shN":
                        noise_dropout = 0.98
                        cur_std = std_gs_params[para_name]
                        cur_mean = mean_gs_params[para_name]
                        noise = torch.randn_like(self.splats[para_name]) * cur_std + cur_mean
                        noise[torch.rand_like(self.splats[para_name]) < noise_dropout] = 0
                        self.splats[para_name] = self.splats[para_name] + noise
                elif idx < mask_step:
                    self.splats[para_name] = self.splats[para_name][selected_indices]
            for step in range(max_steps):
                # if len(self.novelloaders) == 0 or random.random() < 0.7:
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)
                camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
                Ks = data["K"].to(device)  # [1, 3, 3]
                pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
                image_ids = data["image_id"].to(device)
                masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
                img_name = data["image_name"][0]
                img_path = f"{path}/Lora/train/{img_name}/images"
                os.makedirs(img_path,exist_ok=True)
                height, width = pixels.shape[1:3]


                # forward
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if self.cfg.depth_loss else "RGB",
                    masks=masks,
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None
                for j in range(renders.shape[0]):
                    colors = torch.clamp(renders[j, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
                    if idx >= mask_step:
                        num_random = random.randint(5, 20)
                        colors,_ = self.apply_random_black_masks_to_image(colors,num_random,0.1,0.3)
                    colors_path = f"{img_path}/{idx:04d}.png"
                    colors_canvas = colors.cpu().numpy()
                    colors_canvas = (colors_canvas * 255).astype(np.uint8)
                    imageio.imwrite(colors_path, colors_canvas)
            for para_name in gs_params:
                self.splats[para_name] = init_gs_params[para_name].detach().clone()
        print("Finishing create noise train sets for Lora training!")
        for eval_idx in range(len(self.valset)):
            cameras = self.valset.val_cameras[eval_idx]
            K = cameras['K']
            novel_poses = cameras['camtoworlds']
            camtoworlds = torch.from_numpy(novel_poses).float().to(device).unsqueeze(0)
            K = torch.from_numpy(K).float().to(device)
            width, height = list(self.parser.imsize_dict.values())[0]
            Ks = K[None].repeat(camtoworlds.shape[0], 1, 1)
            name = f"input_{eval_idx:04d}"
            img_path = f"{path}/Lora/val/images"
            os.makedirs(img_path,exist_ok=True)
            renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                    render_mode="RGB+ED" if self.cfg.depth_loss else "RGB",
                )
            if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            for j in range(renders.shape[0]):
                colors = torch.clamp(renders[j, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
                colors_path = f"{img_path}/{name}.png"
                colors_canvas = colors.cpu().numpy()
                colors_canvas = (colors_canvas * 255).astype(np.uint8)
                imageio.imwrite(colors_path, colors_canvas)

        print("Finishing create noise val sets for Lora validation")
        
                
    def train(self, init_step=0,max_steps=1000,is_mask = False,is_clone_split = True,cfg_step = None):
        cfg = self.cfg
        device = self.device
        if init_step == 0:
            self.is_novel = False
        else:
            self.is_novel = True


        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader = self.trainloader
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            # if len(self.novelloaders) == 0 or random.random() < 0.7:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            sh_degree_to_use = self.cfg.sh_degree
            loss, l1loss, ssimloss, info = self.render_loss(
                data=data,
                device=device,
                step=step,
                sh_degree_to_use=sh_degree_to_use,
                is_depth=False,
            )
            if self.is_novel:
                try:
                    data = next(self.novelloaders_iter[-1])
                except StopIteration:
                    self.novelloaders_iter[-1] = iter(self.novelloaders[-1])
                    data = next(self.novelloaders_iter[-1])  
                loss_r, l1loss_r, ssimloss_r, info = self.render_loss(
                    data=data,
                    device=device,
                    step=step,
                    sh_degree_to_use=sh_degree_to_use,
                    is_mask=is_mask,
                )
                loss = loss + loss_r
                l1loss = l1loss + l1loss_r
                ssimloss = ssimloss + ssimloss_r
            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            pbar.set_description(desc)


            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            # if step % 100 == 0 and step > 1000 :
            # if step % 100 == 0 and (step + 1000)< max_steps:
            if cfg_step is None:
                cfg_step = 10000
            if step % 100 == 0 and is_clone_split and step< cfg_step:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
                else:
                    assert_never(self.cfg.strategy)

                

GaussianSceneTrainer = Gaussian
