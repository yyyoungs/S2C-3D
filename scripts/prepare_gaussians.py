import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import tyro
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_3dgs import Config
from model.pipeline_gs import Gaussian


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Runner:
    """Runs the noisy Gaussian preparation stage."""

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
            is_phase1=True,
        )
        self.add_noise_steps = cfg.add_noise_steps
        self.remove_steps = cfg.remove_steps
        self.add_mask_steps = cfg.add_mask_steps
        self.end_steps = cfg.end_steps

    def train(self, step: int = 0) -> None:
        cfg = self.cfg
        if not len(cfg.ckpt[0]) > 0:
            self.gs_model.train(init_step=0, max_steps=cfg.end_steps)
            gs_save_path = os.path.join(self.cfg.result_dir, "gs")
            os.makedirs(gs_save_path, exist_ok=True)
            torch.save(self.gs_model.splats, os.path.join(gs_save_path, "original_gs.pth"))
        else:
            self.gs_model.splats = torch.load(cfg.ckpt[0], map_location=self.device)
        self.gs_model.render_noise()


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
    cli(main, cfg, verbose=True)
