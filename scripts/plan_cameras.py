import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.datasets import CameraPlanningDataset, Parser


def parse_args() -> argparse.Namespace:
    cfg = argparse.ArgumentParser()
    cfg.add_argument("--data_dir", default="", type=str)
    cfg.add_argument("--data_factor", default=4, type=int)
    cfg.add_argument("--normalize_world_space", default=False, type=bool)
    cfg.add_argument("--test_every", default=8, type=int)
    cfg.add_argument("--img_dir", default=None, type=str)
    cfg.add_argument("--sphere_radius", default=0.05, type=float)
    cfg.add_argument("--init_coverage_threshold", default=0.0, type=float)
    cfg.add_argument("--new_coverage_threshold", default=0.1, type=float)
    cfg.add_argument("--arc_distance_threshold", default=0.20, type=float)
    cfg.add_argument("--nearest_camera_count", default=2, type=int)
    cfg.add_argument("--translation_weight", default=1.0, type=float)
    cfg.add_argument("--rotation_weight", default=1.0, type=float)
    cfg.add_argument("--xy_sample_range", default=0.6, type=float)
    cfg.add_argument("--z_sample_range", default=0.5, type=float)
    cfg.add_argument("--max_consecutive_failures", default=30, type=int)
    return cfg.parse_args()


def main() -> None:
    cfg = parse_args()
    parser = Parser(
        data_dir=cfg.data_dir,
        img_dir=cfg.img_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=cfg.test_every,
    )
    CameraPlanningDataset(
        parser,
        sphere_radius=cfg.sphere_radius,
        init_coverage_threshold=cfg.init_coverage_threshold,
        new_coverage_threshold=cfg.new_coverage_threshold,
        arc_distance_threshold=cfg.arc_distance_threshold,
        nearest_camera_count=cfg.nearest_camera_count,
        translation_weight=cfg.translation_weight,
        rotation_weight=cfg.rotation_weight,
        xy_sample_range=cfg.xy_sample_range,
        z_sample_range=cfg.z_sample_range,
        max_consecutive_failures=cfg.max_consecutive_failures,
    )


if __name__ == "__main__":
    main()
