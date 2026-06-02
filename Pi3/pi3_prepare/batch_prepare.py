import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PREPARE_ROOT = Path(__file__).resolve().parent
DEFAULT_PI3_ROOT = PREPARE_ROOT.parent / "Pi3-main"
PREPARE_SCENE_SCRIPT = PREPARE_ROOT / "prepare_scene.py"


def scene_has_input_images(scene_dir: Path) -> bool:
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        return False
    return any(
        path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        for path in image_dir.iterdir()
        if path.is_file()
    )


def iter_scene_jobs(data_root: Path, view_dirs: list[str] | None, scenes: list[str] | None):
    candidates = sorted(path for path in data_root.iterdir() if path.is_dir())
    if view_dirs:
        wanted_views = set(view_dirs)
        candidates = [path for path in candidates if path.name in wanted_views]

    for candidate_dir in candidates:
        if scene_has_input_images(candidate_dir):
            if scenes and candidate_dir.name not in set(scenes):
                continue
            yield candidate_dir
            continue

        scene_dirs = sorted(path for path in candidate_dir.iterdir() if path.is_dir())
        if scenes:
            wanted_scenes = set(scenes)
            scene_dirs = [path for path in scene_dirs if path.name in wanted_scenes]

        for scene_dir in scene_dirs:
            if scene_has_input_images(scene_dir):
                yield scene_dir
            else:
                print(f"Skip {scene_dir}: missing images directory or image files.")


def build_subprocess_env(cuda_devices: str | None, pi3_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    pythonpath_parts = [
        str(PREPARE_ROOT),
        str(pi3_root),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
    return env


def run_scene_preparation(
    scene_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    output_dir = scene_dir / "all_views"
    command = [
        args.python,
        str(PREPARE_SCENE_SCRIPT),
        "--data_path",
        str(scene_dir),
        "--view",
        "all",
        "--device",
        args.device,
    ]
    if args.interval is not None:
        command.extend(["--interval", str(args.interval)])
    if args.ckpt:
        command.extend(["--ckpt", args.ckpt])

    print(f"Run Pi3 preparation: {scene_dir} (all views)")
    if args.dry_run:
        print(" ".join(command))
        return

    if output_dir.exists():
        if args.overwrite:
            print(f"Remove existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"Skip {scene_dir}: output already exists.")
            return

    subprocess.run(command, cwd=PREPARE_ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch prepare COLMAP-style sparse data with a vanilla cloned Pi3 repo. "
            "Pi3 source code is not modified; this script uses pi3_prepare as an adapter layer."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data_root",
        type=Path,
        help="Root containing view folders such as 6_views/<scene_id>/images.",
    )
    input_group.add_argument(
        "--scene_path",
        type=Path,
        help="Single scene directory containing an images/ folder.",
    )
    parser.add_argument(
        "--pi3_root",
        type=Path,
        default=DEFAULT_PI3_ROOT,
        help="Path to the cloned Pi3 repository. Default: ../Pi3-main.",
    )
    parser.add_argument(
        "--view",
        default=None,
        help="Deprecated compatibility option. This version always writes all_views/.",
    )
    parser.add_argument(
        "--view_dirs",
        nargs="+",
        default=None,
        help="Optional subset of view folders under --data_root, e.g. 6_views 8_views.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional subset of scene folder names to process under each view folder.",
    )
    parser.add_argument("--ckpt", default=None, help="Optional local Pi3 checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Inference device passed to prepare_scene.py.")
    parser.add_argument("--cuda_devices", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1.")
    parser.add_argument("--interval", type=int, default=None, help="Frame sampling interval passed to prepare_scene.py.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run prepare_scene.py.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing all_views outputs. Default: true.",
    )
    parser.add_argument("--continue_on_error", action="store_true", help="Continue batch processing after failures.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pi3_root = args.pi3_root.resolve()
    if not pi3_root.exists():
        raise FileNotFoundError(f"Pi3 repo not found: {pi3_root}")
    if not PREPARE_SCENE_SCRIPT.exists():
        raise FileNotFoundError(f"prepare_scene.py not found: {PREPARE_SCENE_SCRIPT}")

    env = build_subprocess_env(args.cuda_devices, pi3_root)
    if args.scene_path is not None:
        scene_dir = args.scene_path.resolve()
        if not scene_has_input_images(scene_dir):
            raise FileNotFoundError(f"Scene must contain image files under images/: {scene_dir}")
        jobs = [scene_dir]
    else:
        jobs = list(iter_scene_jobs(args.data_root.resolve(), args.view_dirs, args.scenes))

    if not jobs:
        print("No scenes found to process.")
        return

    failures: list[tuple[Path, Exception]] = []
    for scene_dir in jobs:
        try:
            run_scene_preparation(scene_dir, args, env)
        except Exception as exc:
            failures.append((scene_dir, exc))
            print(f"Failed processing {scene_dir}: {exc}")
            if not args.continue_on_error:
                raise

    if failures:
        print("\nFailed scenes:")
        for scene_dir, exc in failures:
            print(f"- {scene_dir}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
