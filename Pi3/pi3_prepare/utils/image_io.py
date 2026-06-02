import math
import os
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PI3_PIXEL_LIMIT = 255000
PI3_PATCH_SIZE = 14


def load_images_as_tensor(path: str | Path, interval: int = 1, pixel_limit: int = PI3_PIXEL_LIMIT):
    """
    Load images or video frames, resize them to Pi3-compatible dimensions, and
    return a stacked tensor in [N, 3, H, W] format.
    """
    path = Path(path)
    images, image_names = load_image_sources(path, interval)
    if not images:
        raise ValueError(f"No images found under: {path}")

    original_width, original_height = images[0].size
    target_width, target_height = compute_pi3_resize(original_width, original_height, pixel_limit)
    print(f"Found {len(images)} images/frames. Resize to ({target_width}, {target_height}).")

    to_tensor = transforms.ToTensor()
    tensors = [
        to_tensor(image.resize((target_width, target_height), Image.Resampling.LANCZOS))
        for image in images
    ]
    return torch.stack(tensors, dim=0), [original_width, original_height], image_names


def load_image_sources(path: Path, interval: int) -> tuple[list[Image.Image], list[str]]:
    if path.is_dir():
        return load_images_from_directory(path, interval)
    if path.suffix.lower() == ".mp4":
        return load_frames_from_video(path, interval), []
    raise ValueError(f"Unsupported input path, expected image directory or .mp4 file: {path}")


def load_images_from_directory(path: Path, interval: int) -> tuple[list[Image.Image], list[str]]:
    print(f"Loading images from directory: {path}")
    filenames = sorted(name for name in os.listdir(path) if Path(name).suffix.lower() in IMAGE_EXTENSIONS)
    selected_names = filenames[::interval]
    images = [Image.open(path / name).convert("RGB") for name in selected_names]
    return images, selected_names


def load_frames_from_video(path: Path, interval: int) -> list[Image.Image]:
    print(f"Loading frames from video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {path}")

    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_idx += 1
    cap.release()
    return frames


def compute_pi3_resize(width: int, height: int, pixel_limit: int) -> tuple[int, int]:
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1
    target_width = width * scale
    target_height = height * scale

    patch_width = round(target_width / PI3_PATCH_SIZE)
    patch_height = round(target_height / PI3_PATCH_SIZE)
    while (patch_width * PI3_PATCH_SIZE) * (patch_height * PI3_PATCH_SIZE) > pixel_limit:
        if patch_width / patch_height > target_width / target_height:
            patch_width -= 1
        else:
            patch_height -= 1

    return max(1, patch_width) * PI3_PATCH_SIZE, max(1, patch_height) * PI3_PATCH_SIZE
