import argparse
import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = "scripts"

PLAN_CAMERAS_SCRIPT = f"{SCRIPT_DIR}/plan_cameras.py"
PREPARE_GAUSSIANS_SCRIPT = f"{SCRIPT_DIR}/prepare_gaussians.py"
TRAIN_LORA_SCRIPT = f"{SCRIPT_DIR}/train_lora.py"
REFINE_SCENE_SCRIPT = f"{SCRIPT_DIR}/refine_scene.py"
VIEW_SCENE_SCRIPT = f"{SCRIPT_DIR}/view_scene.py"
PI3_PREPARE_SCRIPT = "Pi3/pi3_prepare/batch_prepare.py"

STEP_PI3 = 0
STEP_PHASE1 = 1
STEP_PHASE2 = 2
STEP_PHASE3 = 3
STEP_GUI = 4

PHASE1 = "phase1"
PHASE2 = "phase2"
PHASE3 = "phase3"


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True, cwd=PROJECT_ROOT)


def _is_readable_torch_checkpoint(path: str) -> bool:
    if not os.path.exists(path):
        return False

    try:
        import torch

        torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Checkpoint exists but cannot be loaded, will regenerate it: {path}")
        print(f"Load error: {exc}")
        return False
    return True


def _missing_files(paths: Sequence[str]) -> list[str]:
    return [path for path in paths if not os.path.isfile(path)]


def _log_incomplete_stage(stage_name: str, missing: Sequence[str]) -> None:
    if not missing:
        return
    print(f"{stage_name} is incomplete; missing:")
    for path in missing:
        print(f"  - {path}")


def _append_named_args(command: list[str], args: Mapping[str, Any]) -> None:
    for name, value in args.items():
        command.extend([f"--{name}", str(value)])


def _append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _camera_plan_output_dir(scene_data_dir: str) -> str:
    return os.path.join(scene_data_dir, "add_camera")


def _pi3_output_complete(scene_data_dir: str) -> bool:
    required_files = [
        os.path.join(scene_data_dir, "cam_idx.json"),
        os.path.join(scene_data_dir, "sparse", "cameras.bin"),
        os.path.join(scene_data_dir, "sparse", "images.bin"),
        os.path.join(scene_data_dir, "sparse", "points3D.bin"),
        os.path.join(scene_data_dir, "sparse", "points.ply"),
    ]
    if not all(os.path.isfile(path) for path in required_files):
        return False

    train_img_dir = os.path.join(scene_data_dir, "train_img")
    return os.path.isdir(train_img_dir) and len(os.listdir(train_img_dir)) > 0


def _camera_planning_complete(scene_data_dir: str) -> bool:
    camera_dir = _camera_plan_output_dir(scene_data_dir)
    anchor_path = os.path.join(scene_data_dir, "anchor.json")
    if not os.path.isdir(camera_dir) or not os.path.isfile(anchor_path):
        return False

    camera_files = [name for name in os.listdir(camera_dir) if name.endswith(".npz")]
    return len(camera_files) >= 5


def _phase1_complete(phase1_dir: str) -> bool:
    checkpoint_path = os.path.join(phase1_dir, "gs", "original_gs.pth")
    if not _is_readable_torch_checkpoint(checkpoint_path):
        return False

    train_dir = os.path.join(phase1_dir, "Lora", "train")
    if not os.path.isdir(train_dir):
        _log_incomplete_stage("phase1", [train_dir])
        return False

    return any(
        os.path.isfile(os.path.join(root, filename))
        for root, _, filenames in os.walk(train_dir)
        for filename in filenames
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def _phase2_complete(phase2_dir: str, save_step: int) -> bool:
    model_dir = os.path.join(phase2_dir, "model", str(save_step))
    required_files = [
        os.path.join(model_dir, "vae", "model.safetensors"),
        os.path.join(model_dir, "unet", "model.safetensors"),
    ]
    missing = _missing_files(required_files)
    _log_incomplete_stage("phase2", missing)
    return len(missing) == 0


def _phase3_complete(phase3_dir: str) -> bool:
    checkpoint_path = os.path.join(phase3_dir, "gs", "gs.pth")
    return _is_readable_torch_checkpoint(checkpoint_path)


def _phase_result_dir(result_dir: str, phase_name: str) -> str:
    return os.path.join(result_dir, phase_name)


def _scene_name_from_case_dir(case_dir: str) -> str:
    return os.path.basename(os.path.normpath(case_dir))


def _append_common_scene_args(command: list[str], cfg, scene_data_dir: str, image_root_dir: str) -> None:
    options = cfg.pipeline_options
    _append_named_args(
        command,
        {
            "data_dir": scene_data_dir,
            "data_factor": options.data_factor,
            "test_every": options.test_every,
            "img_dir": image_root_dir,
        },
    )
    _append_flag(command, "--no-normalize-world-space", not options.normalize_world_space)


def _step_enabled(cfg, index: int) -> bool:
    steps = list(cfg.main_model.steps)
    if len(steps) != 5:
        raise ValueError(
            "main_model.steps must have 5 values: "
            "[Pi3, phase1, phase2, phase3, gui]."
        )
    return bool(steps[index])


def _validate_config(cfg) -> None:
    _step_enabled(cfg, STEP_GUI)


@dataclass
class SceneJob:
    scene_data_dir: str
    image_root_dir: str
    result_dir: str


class PipelineOrchestrator:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def run_scene(self, job: SceneJob) -> None:
        os.makedirs(job.result_dir, exist_ok=True)

        self.run_pi3_preparation(job.scene_data_dir, job.image_root_dir)

        camera_process = multiprocessing.Process(
            target=self.run_camera_planning,
            args=(job.scene_data_dir, job.image_root_dir),
            name="camera_planning",
        )
        lora_process = multiprocessing.Process(
            target=self.run_phase1_and_phase2,
            args=(job.scene_data_dir, job.image_root_dir, job.result_dir),
            name="phase1_phase2",
        )

        camera_process.start()
        lora_process.start()
        self._join_process(camera_process)
        self._join_process(lora_process)

        self.run_phase3(job.scene_data_dir, job.image_root_dir, job.result_dir)
        self.run_gui(job.scene_data_dir, job.image_root_dir, job.result_dir)

    @staticmethod
    def _join_process(process: multiprocessing.Process) -> None:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{process.name} failed with exit code {process.exitcode}.")

    def run_pi3_preparation(self, scene_data_dir: str, image_root_dir: str) -> None:
        if _pi3_output_complete(scene_data_dir):
            print(f"Pi3 output already exists: {scene_data_dir}")
            return

        if not _step_enabled(self.cfg, STEP_PI3):
            raise FileNotFoundError(
                f"Pi3 output is incomplete but main_model.steps[0] is disabled: {scene_data_dir}"
            )

        command = [
            sys.executable,
            PI3_PREPARE_SCRIPT,
            "--scene_path",
            image_root_dir,
            "--view",
            "all",
        ]
        _run_command(command)

    def run_camera_planning(self, scene_data_dir: str, image_root_dir: str) -> None:
        if _camera_planning_complete(scene_data_dir):
            print(f"Camera planning output already exists: {scene_data_dir}")
            return

        options = self.cfg.pipeline_options
        planner = self.cfg.camera_planning_model
        command = [
            sys.executable,
            PLAN_CAMERAS_SCRIPT,
        ]
        _append_named_args(
            command,
            {
                "data_dir": scene_data_dir,
                "data_factor": options.data_factor,
                "test_every": options.test_every,
                "img_dir": image_root_dir,
                "sphere_radius": planner.sphere_radius,
                "init_coverage_threshold": planner.init_coverage_threshold,
                "new_coverage_threshold": planner.new_coverage_threshold,
                "arc_distance_threshold": planner.arc_distance_threshold,
                "nearest_camera_count": planner.nearest_camera_count,
                "translation_weight": planner.translation_weight,
                "rotation_weight": planner.rotation_weight,
                "xy_sample_range": planner.xy_sample_range,
                "z_sample_range": planner.z_sample_range,
                "max_consecutive_failures": planner.max_consecutive_failures,
            },
        )
        if options.normalize_world_space:
            _append_named_args(command, {"normalize_world_space": True})
        _run_command(command)

    def run_phase1(self, scene_data_dir: str, image_root_dir: str, result_dir: str) -> None:
        if not _step_enabled(self.cfg, STEP_PHASE1):
            return

        phase1_dir = _phase_result_dir(result_dir, PHASE1)
        if _phase1_complete(phase1_dir):
            print(f"phase1 output already exists: {phase1_dir}")
            return

        noise = self.cfg.noise_model
        command = [
            sys.executable,
            PREPARE_GAUSSIANS_SCRIPT,
            self.cfg.pipeline_options.gs_config,
        ]
        _append_named_args(
            command,
            {
                "result_dir": phase1_dir,
                "end_steps": noise.end_steps,
                "add_noise_steps": noise.add_noise_steps,
                "remove_steps": noise.remove_steps,
                "add_mask_steps": noise.add_mask_steps,
                "ckpt": noise.ckpt,
            },
        )
        _append_common_scene_args(command, self.cfg, scene_data_dir, image_root_dir)
        _run_command(command)

    def run_phase1_and_phase2(self, scene_data_dir: str, image_root_dir: str, result_dir: str) -> None:
        self.run_phase1(scene_data_dir, image_root_dir, result_dir)
        self.run_phase2(scene_data_dir, image_root_dir, result_dir)

    def run_phase2(self, scene_data_dir: str, image_root_dir: str, result_dir: str) -> None:
        if not _step_enabled(self.cfg, STEP_PHASE2):
            return

        phase1_dir = _phase_result_dir(result_dir, PHASE1)
        if not _phase1_complete(phase1_dir):
            raise FileNotFoundError(f"phase2 requires a complete phase1 output: {phase1_dir}")

        phase2_dir = _phase_result_dir(result_dir, PHASE2)
        if _phase2_complete(phase2_dir, self.cfg.lora_model.save_steps):
            print(f"phase2 output already exists: {phase2_dir}")
            return

        lora = self.cfg.lora_model
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--mixed_precision",
            str(lora.mixed_precision),
            TRAIN_LORA_SCRIPT,
        ]
        _append_named_args(
            command,
            {
                "output_dir": phase2_dir,
                "dataset_path": image_root_dir,
                "max_train_steps": lora.Lora_steps,
                "learning_rate": lora.learning_rate,
                "train_batch_size": lora.train_batch_size,
                "dataloader_num_workers": lora.dataloader_num_workers,
                "eval_freq": lora.eval_freq,
                "viz_freq": lora.viz_freq,
                "save_step": lora.save_steps,
                "lambda_lpips": lora.lambda_lpips,
                "lambda_l2": lora.lambda_l2,
                "lambda_gram": lora.lambda_gram,
                "gram_loss_warmup_steps": lora.gram_loss_warmup_steps,
                "report_to": lora.report_to,
                "timestep": lora.timestep,
                "data_factor": lora.data_factor,
            },
        )
        _append_flag(
            command,
            "--enable_xformers_memory_efficient_attention",
            lora.enable_xformers_memory_efficient_attention,
        )
        _run_command(command)

    def run_phase3(self, scene_data_dir: str, image_root_dir: str, result_dir: str) -> None:
        if not _step_enabled(self.cfg, STEP_PHASE3):
            return

        phase1_dir = _phase_result_dir(result_dir, PHASE1)
        phase2_dir = _phase_result_dir(result_dir, PHASE2)
        if not _phase1_complete(phase1_dir):
            raise FileNotFoundError(f"phase3 requires a complete phase1 output: {phase1_dir}")
        if not _phase2_complete(phase2_dir, self.cfg.lora_model.save_steps):
            raise FileNotFoundError(f"phase3 requires a complete phase2 output: {phase2_dir}")

        phase3_dir = _phase_result_dir(result_dir, PHASE3)
        if _phase3_complete(phase3_dir):
            print(f"phase3 output already exists: {phase3_dir}")
            return

        recon = self.cfg.recon_model
        noise = self.cfg.noise_model
        diffx_model_path = os.path.join(
            _phase_result_dir(result_dir, PHASE2),
            "model",
            str(self.cfg.lora_model.save_steps),
        )
        original_gs_path = os.path.join(_phase_result_dir(result_dir, PHASE1), "gs", "original_gs.pth")
        command = [
            sys.executable,
            REFINE_SCENE_SCRIPT,
            self.cfg.pipeline_options.gs_config,
        ]
        _append_named_args(
            command,
            {
                "result_dir": phase3_dir,
                "diffx_model_path": diffx_model_path,
                "gs_model_path": original_gs_path,
                "refine_steps": recon.refine_steps,
                "end_refine_steps": recon.end_refine_steps,
                "repeat_times": recon.repeat_times,
                "end_steps": noise.end_steps,
            },
        )
        _append_common_scene_args(command, self.cfg, scene_data_dir, image_root_dir)
        _run_command(command)

    def run_gui(self, scene_data_dir: str, image_root_dir: str, result_dir: str) -> None:
        if not _step_enabled(self.cfg, STEP_GUI):
            return

        final_gs_path = os.path.join(_phase_result_dir(result_dir, PHASE3), "gs", "gs.pth")
        if not _phase3_complete(_phase_result_dir(result_dir, PHASE3)):
            print(f"There is no corresponding 3DGS of {result_dir}!")
            return

        command = [
            sys.executable,
            VIEW_SCENE_SCRIPT,
            self.cfg.pipeline_options.gs_config,
        ]
        _append_named_args(command, {"result_dir": result_dir, "gs_model_path": final_gs_path})
        _append_common_scene_args(command, self.cfg, scene_data_dir, image_root_dir)
        _append_flag(command, "--is_gui", True)
        _run_command(command)


def _build_jobs(cfg) -> list[SceneJob]:
    case_dir = cfg.main_model.case_dir or ""
    if len(case_dir) > 0:
        scene_name = _scene_name_from_case_dir(case_dir)
        return [
            SceneJob(
                scene_data_dir=f"{case_dir}/all_views",
                image_root_dir=case_dir,
                result_dir=os.path.join(cfg.main_model.result_dir, scene_name),
            )
        ]

    jobs: list[SceneJob] = []
    category_names = sorted(os.listdir(cfg.main_model.data_dir))
    for category_name in category_names:
        jobs.append(
            SceneJob(
                scene_data_dir=f"{cfg.main_model.data_dir}/{category_name}/all_views",
                image_root_dir=f"{cfg.main_model.data_dir}/{category_name}",
                result_dir=f"{cfg.main_model.result_dir}/{category_name}",
            )
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--cuda_devices",
        default="0",
        help="Comma-separated CUDA device ids to make visible for all pipeline stages, e.g. 0 or 0,1.",
    )
    args, extras = parser.parse_known_args()

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_devices)

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    _validate_config(cfg)

    orchestrator = PipelineOrchestrator(cfg)
    for job in _build_jobs(cfg):
        orchestrator.run_scene(job)
        print(f"Finished processing scene at {job.image_root_dir}")


if __name__ == "__main__":
    main()
