# Sparse-to-Complete: From Sparse Image Captures to Complete 3D Scenes

**Sparse-to-Complete: From Sparse Image Captures to Complete 3D Scenes**  
Yiyang Shen, Yin Yang, Kun Zhou, Tianjia Shao  
SIGGRAPH 2026

[[Project Page](https://gapszju.github.io/S2C-3D/)]
[[Paper](https://arxiv.org/abs/2605.05664)]

This repository contains the source code for "Sparse-to-Complete: From Sparse Image Captures to Complete 3D Scenes".

## Installation

### Clone

Clone the repository with the Pi3 submodule:

```bash
git clone --recursive https://github.com/yyyoungs/S2C-3D.git
cd S2C-3D
```

For an existing clone, initialize the submodule with:

```bash
git submodule update --init --recursive
```

### Environment

Create the S2C-3D conda environment:

```bash
conda env create -f environment.yml
conda activate S2C-3D
```

The default environment uses PyTorch 2.4.0 with CUDA 12.1 wheels. If your CUDA
driver or platform requires another PyTorch build, install the matching PyTorch
wheel first and then install the remaining packages from `environment.yml`.

### CUDA Renderer

If the CUDA renderer extension is not built in your environment, build it with:

```bash
cd cuda_renderer
python setup.py install
cd ..
```

## Data Preparation

Each scene should contain an `images/` folder. The following tree shows the
bundled example scene:

```text
example/6_views/
└── 52599ae063/
    └── images/
        ├── inputs_0000.png
        ├── inputs_0001.png
        ├── inputs_0002.png
        └── ...
```

Image names are sorted lexicographically. Numbered names such as
`inputs_0000.png`, `inputs_0001.png`, ... are recommended.

For your own data, replace `52599ae063` with your scene name and put all input
images under that scene's `images/` folder.

Pi3 preparation is run automatically by `main.py`. It writes:

```text
scene_id/
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

## Run

### Stage Control

`main_model.steps` controls which stages run:

```yaml
main_model:
  # [Pi3, phase1, phase2, phase3, gui]
  steps: [1, 1, 1, 1, 1]
```

### Full Pipeline

Run all scenes under `main_model.data_dir` in `configs/config.yaml`:

```bash
conda activate S2C-3D
python main.py --cuda_devices 0
```

Run a single scene:

```bash
python main.py --cuda_devices 0 \
  main_model.case_dir=/path/to/scene
```

Outputs are written to:

```text
test/<scene_id>/
├── phase1/
├── phase2/
└── phase3/
```

The final Gaussian checkpoint is:

```text
test/<scene_id>/phase3/gs/gs.pth
```

For example, after a scene is fully reconstructed, open only the GUI:

```bash
python main.py --cuda_devices 0 \
  main_model.case_dir=/path/to/scene \
  'main_model.steps=[0,0,0,0,1]'
```

### Pi3 Preparation Only

Pi3 preparation is normally called by `main.py`. You can also run it separately
before launching the full pipeline:

```bash
python Pi3/pi3_prepare/batch_prepare.py \
  --scene_path /path/to/scene \
  --cuda_devices 0
```

Batch mode:

```bash
python Pi3/pi3_prepare/batch_prepare.py \
  --data_root /path/to/dataset \
  --view_dirs 6_views \
  --cuda_devices 0
```

## GUI Controls

Open the GUI for one reconstructed scene:

```bash
python main.py --cuda_devices 0 \
  main_model.case_dir=/path/to/scene \
  'main_model.steps=[0,0,0,0,1]'
```

For the bundled example:

```bash
python main.py --cuda_devices 0 \
  main_model.case_dir=./example/6_views/52599ae063 \
  'main_model.steps=[0,0,0,0,1]'
```

Basic controls:

- `W/S`: move forward/backward.
- `A/D`: move left/right.
- `Q/E`: move down/up.
- `J/L`: yaw left/right.
- `I/K`: pitch up/down.
- `U/O`: roll left/right.
- Mouse wheel: zoom.
- Left drag: orbit.
- Middle drag: pan.

## Repository Layout

```text
S2C-3D/
├── main.py
├── configs/
├── scripts/
│   ├── plan_cameras.py
│   ├── prepare_gaussians.py
│   ├── train_lora.py
│   ├── refine_scene.py
│   └── view_scene.py
├── model/
├── Pi3/
│   ├── Pi3-main/
│   └── pi3_prepare/
├── cuda_renderer/
└── lpipsPyTorch/
```

## BibTeX

```bibtex
@misc{shen2026sparsetocompletesparseimagecaptures,
  title         = {Sparse-to-Complete: From Sparse Image Captures to Complete 3D Scenes},
  author        = {Yiyang Shen and Yin Yang and Kun Zhou and Tianjia Shao},
  year          = {2026},
  eprint        = {2605.05664},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.05664},
}
```

## Acknowledgements

This project builds on excellent open-source work including [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [gsplat](https://github.com/nerfstudio-project/gsplat), [Pi3](https://github.com/yyfz/Pi3), and [Difix3D+](https://github.com/nv-tlabs/Difix3D). We thank all the authors for their great work.
