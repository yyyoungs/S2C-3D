import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge, depth_normal_edge, normal_edge, points_to_normals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpts/pi3.pt")
    parser.add_argument("--out-dir", default="debug_edge_masks")
    parser.add_argument("--pixel-limit", type=int, default=255000)
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--conf-thre", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--normal-tol-deg", type=float, default=5.0)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--ply-dir", default=None)
    return parser.parse_args()


DEFAULT_SOURCES = [
    ("img_parkour", "examples/parkour", 3),
    ("img_valley", "examples/valley", 5),
    ("img_house", "examples/house", 4),
    ("vid_skating", "examples/skating.mp4", 20),
    ("vid_skiing", "examples/skiing.mp4", 20),
    ("vid_gradio_valley", "examples/gradio_examples/valley.mp4", 20),
]


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return None


FONT = load_font(14)
FONT_SMALL = load_font(12)


def to_u8_rgb(img_chw):
    arr = img_chw.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def mask_rgb(mask, color):
    mask = mask.detach().cpu().numpy().astype(bool)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = color
    return out


def overlay_mask(rgb, mask, color=(50, 230, 80), alpha=0.62):
    mask = mask.detach().cpu().numpy().astype(bool)
    out = rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def depth_vis(depth):
    d = depth.detach().float().cpu().numpy()
    valid = np.isfinite(d) & (d > 0)
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out

    lo, hi = np.percentile(d[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1.0e-6
    x = np.clip((d - lo) / (hi - lo), 0, 1)
    r = np.clip(255 * (1.5 * x), 0, 255)
    g = np.clip(255 * (1.5 * x - 0.35), 0, 255)
    b = np.clip(255 * (1.2 - 1.2 * x), 0, 255)
    out = np.stack([r, g, b], axis=-1).astype(np.uint8)
    out[~valid] = 0
    return out


def tile_with_label(arr, label):
    image = Image.fromarray(arr)
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 26), (255, 255, 255))
    canvas.paste(image, (0, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(0, 0, 0), font=FONT_SMALL)
    return canvas


def make_sheet(name, imgs, res, args, out_dir):
    conf = torch.sigmoid(res["conf"][0, ..., 0])
    local = res["local_points"][0]
    depth = local[..., 2]
    valid = conf > args.conf_thre
    depth_edges = depth_edge(depth, rtol=args.rtol, mask=valid) & valid
    normals, normal_mask = points_to_normals(local, valid)
    normal_edges = normal_edge(normals, tol_deg=args.normal_tol_deg, mask=normal_mask)
    final_edges = depth_normal_edge(
        local,
        rtol=args.rtol,
        normal_tol_deg=args.normal_tol_deg,
        mask=valid,
    )

    rows = []
    stats_lines = []
    for index in range(imgs.shape[0]):
        rgb = to_u8_rgb(imgs[index])
        height, _width = rgb.shape[:2]
        valid_count = int(valid[index].sum().item())
        depth_count = int(depth_edges[index].sum().item())
        normal_count = int(normal_edges[index].sum().item())
        final_count = int(final_edges[index].sum().item())
        stats_lines.append(
            f"f{index}: valid={valid_count}, depth={depth_count}, "
            f"normal={normal_count}, final={final_count}"
        )
        tiles = [
            tile_with_label(rgb, f"{name} frame {index}"),
            tile_with_label(depth_vis(depth[index]), "depth"),
            tile_with_label(mask_rgb(depth_edges[index], (255, 80, 50)), f"depth edge {depth_count}"),
            tile_with_label(mask_rgb(normal_edges[index], (80, 140, 255)), f"normal edge {normal_count}"),
            tile_with_label(mask_rgb(final_edges[index], (50, 230, 80)), f"final edge {final_count}"),
            tile_with_label(overlay_mask(rgb, final_edges[index]), "final overlay"),
        ]
        row = Image.new("RGB", (sum(tile.width for tile in tiles), height + 26), (255, 255, 255))
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        rows.append(row)

    header_h = 34
    sheet = Image.new("RGB", (max(row.width for row in rows), header_h + sum(row.height for row in rows)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        f"{name} | pixel_limit={args.pixel_limit} | conf>{args.conf_thre} | rtol={args.rtol} | normal>{args.normal_tol_deg}deg",
        fill=(0, 0, 0),
        font=FONT,
    )
    y = header_h
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height

    out_path = out_dir / f"{name}_edge_masks.png"
    sheet.save(out_path)
    return out_path, stats_lines


def write_filtered_ply(name, imgs, res, args, ply_dir):
    conf = torch.sigmoid(res["conf"][0, ..., 0])
    local = res["local_points"][0]
    valid = conf > args.conf_thre
    final_edges = depth_normal_edge(
        local,
        rtol=args.rtol,
        normal_tol_deg=args.normal_tol_deg,
        mask=valid,
    )
    masks = valid & ~final_edges
    points = res["points"][0][masks]
    colors = imgs.permute(0, 2, 3, 1)[masks]
    out_path = ply_dir / f"{name}_filtered.ply"
    write_ply(points, colors, str(out_path))
    return out_path, int(masks.sum().item())


def build_sources(args):
    if not args.sources:
        return DEFAULT_SOURCES
    selected = set(args.sources)
    return [source for source in DEFAULT_SOURCES if source[0] in selected]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = Path(args.ply_dir) if args.ply_dir is not None else out_dir / "ply"
    if args.write_ply:
        ply_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    print("loading model")
    start = time.time()
    model = Pi3().eval()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    print(f"model ready in {time.time() - start:.1f}s")

    summary = []
    sources = build_sources(args)
    for name, source, interval in sources:
        try:
            print(f"processing {name}: {source}")
            imgs = load_images_as_tensor(source, interval=interval, PIXEL_LIMIT=args.pixel_limit, verbose=False)
            imgs = imgs[: args.max_frames]
            if imgs.numel() == 0:
                raise RuntimeError("no frames loaded")
            imgs = imgs.to(device)
            infer_start = time.time()
            with torch.no_grad():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        res = model(imgs[None])
                else:
                    res = model(imgs[None])
            imgs_cpu = imgs.detach().cpu()
            res_cpu = {k: v.detach().cpu() for k, v in res.items()}
            out_path, stats = make_sheet(name, imgs_cpu, res_cpu, args, out_dir)
            if args.write_ply:
                ply_path, ply_points = write_filtered_ply(name, imgs_cpu, res_cpu, args, ply_dir)
                stats.append(f"ply={ply_path}, points={ply_points}")
            elapsed = time.time() - infer_start
            summary.append((name, str(out_path), elapsed, stats))
            print(f"wrote {out_path} ({elapsed:.1f}s)")
        except Exception as exc:
            summary.append((name, "FAILED", 0.0, [repr(exc)]))
            print(f"FAILED {name}: {exc!r}")

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        for name, path, elapsed, stats in summary:
            f.write(f"{name}\t{path}\t{elapsed:.2f}s\n")
            for line in stats:
                f.write(f"  {line}\n")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
