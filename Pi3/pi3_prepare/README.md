# Pi3 Preparation Adapter

This adapter converts user images into the COLMAP-style layout expected by
S2C-3D without modifying the vanilla Pi3 clone under `../Pi3-main`.

## Run Through S2C-3D

Normally `main.py` runs this step automatically when `main_model.steps[0]` is
enabled:

```bash
conda activate S2C-3D
python main.py --cuda_devices 0
```

## Run Pi3 Preparation Only

Single scene:

```bash
python Pi3/pi3_prepare/batch_prepare.py \
  --scene_path /path/to/scene \
  --cuda_devices 0
```

Batch scenes:

```bash
python Pi3/pi3_prepare/batch_prepare.py \
  --data_root /path/to/dataset \
  --view_dirs 6_views \
  --cuda_devices 0
```

The scene folder must contain `images/`:

```text
/path/to/dataset/
└── 6_views/
    └── scene_000/
        └── images/
            ├── 000.png
            ├── 001.png
            └── ...
```

Output:

```text
scene_000/
├── images/
└── all_views/
    ├── cam_idx.json
    ├── sparse/
    │   ├── cameras.bin
    │   ├── images.bin
    │   ├── points3D.bin
    │   └── points.ply
    └── train_img/
```

Useful options: `--pi3_root`, `--ckpt`, `--device`, `--cuda_devices`,
`--interval`, `--no-overwrite`, and `--continue_on_error`.
