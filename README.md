# OVIE: One View Is Enough
### In-the-Wild Monocular Pretraining for Novel View Generation

[![Project Page](https://img.shields.io/badge/Project_Page-green?logo=googlechrome&logoColor=white)](https://kyutai.org/blog/2026-04-14-ovie)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.23488)
[![Models](https://img.shields.io/badge/🤗%20HuggingFace-kyutai%2Fovie-yellow)](https://huggingface.co/collections/kyutai/ovie)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/kyutai-labs/ovie/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/kyutai-labs/ovie/actions/workflows/pre-commit.yml)

This repository contains the official implementation and models for **OVIE** (*One View Is Enough: In-the-Wild Monocular Pretraining for Novel View Generation*), NeurIPS 2026.

OVIE is a framework for monocular novel view synthesis that does not require multi-view image pairs for supervision. Instead, it is trained entirely on unpaired internet images.

![OVIE teaser](assets/teaser.jpeg)

---

## 🗂️ Table of Contents
- [Installation](#-installation)
- [Model Weights](#-model-weights)
- [Model Zoo](#-model-zoo)
- [Inference](#-inference)
- [Data Preprocessing](#-data-preprocessing)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🛠️ Installation

We use [`uv`](https://docs.astral.sh/uv/) by Astral to manage the Python environment and dependencies. It is a drastically faster drop-in replacement for standard Python packaging tools.

**1. Install `uv`:**
For macOS/Linux:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(Alternatively, you can install it via macOS Homebrew: `brew install uv`, or refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/) for Windows instructions.)*

**2. Clone this repository and sync dependencies:**
Once `uv` is installed, clone the project and run `uv sync`. This will automatically resolve the required Python version (3.10.9) and install all dependencies from `uv.lock`.
```sh
git clone https://github.com/kyutai-labs/ovie.git
cd ovie
uv sync
```
*Prefix all commands with `uv run` to ensure they run inside the managed environment.*

---

## 📥 Model Weights

Pretrained weights are hosted on the Hugging Face Hub and are downloaded automatically when using `from_pretrained` (see [Inference](#-inference) below). All four released checkpoints are grouped in the [OVIE collection](https://huggingface.co/collections/kyutai/ovie):

| Variant | Hub repository |
|---|---|
| `OVIE` (monocular, 256×256) | [`kyutai/ovie`](https://huggingface.co/kyutai/ovie) |
| `OVIE-512` (monocular, 512×512) | [`kyutai/ovie-512`](https://huggingface.co/kyutai/ovie-512) |
| `OVIE-ft` (RealEstate10K fine-tune) | [`kyutai/ovie-ft-re10k`](https://huggingface.co/kyutai/ovie-ft-re10k) |
| `OVIE-ft` (DL3DV fine-tune) | [`kyutai/ovie-ft-dl3dv`](https://huggingface.co/kyutai/ovie-ft-dl3dv) |

For **inference and evaluation**, load any variant straight from the Hub with `from_pretrained` — no local checkpoint is needed (see [Inference](#-inference) and [Evaluation](#-evaluation)).

Local `.pt` checkpoints come from the [Releases page](https://github.com/kyutai-labs/ovie/releases) and go inside the `assets/` folder:

* `ovie.pt` — base checkpoint (contains EMA weights).
* `ovie_ft_re10k.pt` — **OVIE-ft**, fine-tuned on RealEstate10K (see [Model Zoo](#-model-zoo)).
* `ovie_512.pt` — **OVIE-512**, trained at 512×512.
* `ovie_ft_dl3dv.pt` — **OVIE-ft**, fine-tuned on DL3DV.
* `dino_vit_small_patch8_224.pth` — used only for training; same checkpoint as in [RAE](https://github.com/bytetriper/RAE).

All of these are also in the [Hub collection](https://huggingface.co/collections/kyutai/ovie) and load directly through `from_pretrained`; the `.pt` files are mirrors for scripts that expect a local checkpoint.

```text
OVIE/
├── assets/
│   ├── ovie.pt                         # base checkpoint (monocular-trained, 256)
│   ├── ovie_512.pt                     # OVIE-512 (512)
│   ├── ovie_ft_re10k.pt                # OVIE-ft (RE10K)
│   ├── ovie_ft_dl3dv.pt                # OVIE-ft (DL3DV)
│   ├── dino_vit_small_patch8_224.pth  # training only
│   └── sample_image.jpg
├── configs/
│   └── config_ovie.yaml
├── models/
└── ...
```

---

## 🧬 Model Zoo

OVIE is trained **without any multi-view supervision**: the base model never sees a posed image pair. It is trained on 30M unpaired in-the-wild images (ImageNet-21K, Open Images, Places, OSV5M) with MoGe-2 generating pseudo-pairs on the fly. We additionally release a resolution-agnostic 512×512 variant, plus two short in-domain fine-tunes that quantify what a 50k-step adaptation on top of purely monocular pretraining adds.

Metrics follow the paper: 750 scenes, 14 target frames, per-scene scale sweep, identical protocol for every row. The tables below list the four released checkpoints only; the full baseline comparison is in the paper.

**RealEstate10K** — OVIE and OVIE-512 are out-of-domain; OVIE-ft-re10k is in-domain:

| Checkpoint | Training data | Res. | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | MEt3R ↓ |
|---|---|---|---|---|---|---|---|
| `ovie.pt` (**OVIE**) | in-the-wild monocular images only | 256 | 18.8 | 0.602 | 0.279 | 6.74 | 0.035 |
| `ovie_512.pt` (**OVIE-512**) | same mix, trained at 512 (`eval@256`) | 512 | 19.1 | 0.611 | 0.283 | 7.62 | 0.034 |
| `ovie_ft_re10k.pt` (**OVIE-ft**, *in-domain*) | + RealEstate10K fine-tune (50k steps) | 256 | 21.9 | 0.695 | 0.195 | 5.59 | 0.029 |
| *real video frames (reference)* | — | 256 | — | — | — | — | *0.042* |

**DL3DV** — every row is out-of-domain there except the DL3DV fine-tune:

| Checkpoint | Training data | Res. | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | MEt3R ↓ |
|---|---|---|---|---|---|---|---|
| `ovie.pt` (**OVIE**) | in-the-wild monocular images only | 256 | 14.8 | 0.369 | 0.464 | 13.6 | 0.078 |
| `ovie_512.pt` (**OVIE-512**) | same mix, trained at 512 (`eval@256`) | 512 | 15.2 | 0.389 | 0.464 | 14.8 | 0.077 |
| `ovie_ft_re10k.pt` (**OVIE-ft**) | + RealEstate10K fine-tune (50k steps) | 256 | 14.1 | 0.351 | 0.531 | 24.66 | 0.040 |
| `ovie_ft_dl3dv.pt` (**OVIE-ft**, *in-domain*) | + DL3DV fine-tune (50k steps) | 256 | 17.07 | 0.433 | 0.368 | 17.24 | 0.059 |

In the paper, base OVIE is the best of *all* geometry-free methods on every DL3DV metric except multi-view consistency, which the RealEstate10K fine-tune leads; OVIE-ft is best in-domain on RealEstate10K. OVIE-512 is scored at 256×256 (`eval@256`) for a like-for-like comparison; its native-resolution numbers are in the paper's appendix. MEt3R measures multi-view consistency between consecutive generated views (lower is more consistent); the real-video reference row shows that independently generated views are *more* consistent with one another than consecutive frames of the ground-truth video are. The in-domain DL3DV fine-tune's DL3DV MEt3R is excluded from the paper's ranking for fairness.

**Which checkpoint to pick:** `ovie.pt` for in-the-wild and zero-shot use, `ovie_ft_re10k.pt` when targeting RealEstate10K-style indoor scenes, `ovie_ft_dl3dv.pt` for DL3DV-style scenes, and `ovie_512.pt` when 512×512 outputs are needed. Fine-tuning selects a domain: the DL3DV fine-tune costs FID on DL3DV (13.6 → 17.24) while raising PSNR (14.8 → 17.07).

---

## 🚀 Inference

We provide two Jupyter notebooks to get started quickly:

| Notebook | Weights source |
|---|---|
| `inference_huggingface.ipynb` | Downloaded automatically from the Hub |
| `inference_local.ipynb` | Loaded from a local `assets/ovie.pt` checkpoint |

```sh
uv run jupyter notebook inference_huggingface.ipynb
```

**Loading from the Hugging Face Hub (recommended):**

```python
import torch
from models.models import OVIEModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Any of: "kyutai/ovie", "kyutai/ovie-512",
#         "kyutai/ovie-ft-re10k", "kyutai/ovie-ft-dl3dv"
model = OVIEModel.from_pretrained("kyutai/ovie", revision="v1.0").to(device)
model.eval()
image_size = model.image_size  # 256, read from the saved config
```

The `image_size` (256 for the base and fine-tuned models, 512 for `kyutai/ovie-512`) is read from the saved config, so the same code works for every variant.

**Loading from a local checkpoint:**

```python
import yaml, torch
from models.models import OVIE_models

with open("./configs/config_ovie.yaml") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]
image_size = config["data"]["image_size"]

model = OVIE_models[model_cfg["model_type"]](
    image_size=image_size,
    vit_use_qknorm=model_cfg.get("use_qknorm", False),
    vit_use_swiglu=model_cfg.get("use_swiglu", True),
    vit_use_rope=model_cfg.get("use_rope", False),
    vit_use_rmsnorm=model_cfg.get("use_rmsnorm", True),
    vit_wo_shift=model_cfg.get("wo_shift", False),
    vit_use_checkpoint=model_cfg.get("use_checkpoint", False),
).to(device)

ckpt = torch.load("./assets/ovie.pt", map_location="cpu")
model.load_state_dict(ckpt["ema"])
model.eval()
```

To use the fine-tuned or 512 checkpoints instead, point the same code at the corresponding file — the loading path is identical, only the weights and `image_size` differ:

```python
# OVIE-ft (RealEstate10K), OVIE-ft (DL3DV) — image_size stays 256
ckpt = torch.load("./assets/ovie_ft_re10k.pt", map_location="cpu")
model.load_state_dict(ckpt["ema"])

# OVIE-512 — build the model with image_size=512
```

**Running inference:**

```python
from torchvision.transforms import ToTensor
from PIL import Image
from utils.pose_enc import extri_intri_to_pose_encoding

img_pil = Image.open("./assets/sample_image.jpg").convert("RGB").resize((image_size, image_size))
img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)

extrinsics = torch.tensor([[[1.0, 0.0, 0.0, -1.25],
                            [0.0, 1.0, 0.0,  0.5],
                            [0.0, 0.0, 1.0, -2.0]]], device=device)
dummy_intrinsics = torch.zeros(1, 1, 3, 3, device=device)

camera = extri_intri_to_pose_encoding(
    extrinsics=extrinsics.unsqueeze(0),
    intrinsics=dummy_intrinsics,
    image_size_hw=(image_size, image_size),
)
cam_token = camera[..., :7].squeeze(0)

with torch.no_grad():
    pred_tensor = model(x=img_tensor, cam_params=cam_token)
```

---

## 🧹 Data Preprocessing

Before training or evaluating on specific datasets, raw images must be preprocessed. We provide scripts for both in-the-wild training data and DL3DV evaluation data.

**For in-the-wild training images:**
```sh
uv run python data_preparation/preprocess_in_the_wild_images.py \
    --data_path /PATH/TO/RAW/DATASET \
    --output_path /PATH/TO/PREPROCESSED/DATASET
```
Point the resulting directories to the `data_path` lists in `configs/config_ovie.yaml`.

**For DL3DV evaluation data:**
```sh
uv run python data_preparation/format_dl3dv.py \
    --root_dir /PATH/TO/DL3DV \
    --output_dir /PATH/TO/PROCESSED/DL3DV
```
DL3DV can be downloaded from the [official dataset repository](https://github.com/DL3DV-10K/Dataset).

---

## 🏋️‍♂️ Training

Once data is preprocessed and paths are set in the config, launch distributed training with `torchrun`:

```sh
uv run torchrun --nproc_per_node <number_of_gpus> train.py --config configs/config_ovie.yaml
```

---

## 📊 Evaluation

Use `evaluate.py` to evaluate on benchmark datasets. Requires a local `assets/ovie.pt` checkpoint (see [Model Weights](#-model-weights)).

**Evaluating on Real Estate 10K (RE10K):**
The pre-processed RE10K dataset is available on Hugging Face: [chenchenshi/re10k-sc](https://huggingface.co/datasets/chenchenshi/re10k-sc).

```sh
uv run python evaluate.py \
    --dataset_path /PATH/TO/EVAL/DATASET \
    --config_path configs/config_ovie.yaml \
    --checkpoint_path assets/ovie.pt
```

To reproduce the **OVIE-ft** row of the [Model Zoo](#-model-zoo), load the fine-tuned weights from the Hub — the numbers in that table use `--stride 3 --num_target_frames 14` on the RE10K test split:

```sh
uv run python evaluate.py \
    --dataset_path /PATH/TO/RE10K/TEST \
    --config_path configs/config_ovie.yaml \
    --from_pretrained kyutai/ovie-ft-re10k \
    --stride 3 --num_target_frames 14
```

For **OVIE-512**, pass `--image_size 512` as well, and score at 256×256 for the like-for-like comparison reported in the Model Zoo. Any released variant can be evaluated the same way, e.g. `--from_pretrained kyutai/ovie`.

**Multi-view consistency (MEt3R).** `evaluate_met3r.py` scores a folder of generated frames with [MEt3R](https://github.com/mohammadasim98/met3r), the multi-view consistency metric reported in the paper. It reads the `gen/` folder written by `evaluate.py`, so it runs after generation and needs no access to the model:

```sh
uv run python evaluate_met3r.py \
    --gen-dir evaluation/<run>/gen \
    --out results/ovie_met3r.json \
    --skip-first-frame
```

MEt3R brings its own dependency set (`met3r` with its MASt3R submodule, FeatUp, a source build of `pytorch3d`, and `timm==0.4.12`) that conflicts with this repository's dependencies, so install it in a separate environment or container — see the docstring in `evaluate_met3r.py`.

---

## 🔧 Contributing

This project uses [pre-commit](https://pre-commit.com/) hooks to enforce code style ([ruff](https://docs.astral.sh/ruff/) format + lint) and keep the lockfile in sync. CI runs the same checks on every push and pull request.

**Install the hooks:**
```sh
uv run pre-commit install
```

After this, `ruff format`, `ruff check`, and `uv lock --check` run automatically on every `git commit`. You can also run them manually across all files:
```sh
uv run pre-commit run --all-files
```

The `uv.lock` file is committed to the repository — do not remove it from version control.

---

## 🤝 Acknowledgments and Citation

This project relies on fantastic open-source tools and models, including:
* [DINOv2](https://github.com/facebookresearch/dinov2)
* [DINOv3](https://github.com/facebookresearch/dinov3)
* [MoGe Depth Estimator](https://github.com/Ruicheng/moge)
* [Visually Grounded Geometry Transformer (VGGT)](https://github.com/facebookresearch/vggt)
* [Representation Autoencoders (RAE)](https://github.com/bytetriper/RAE)

If you find our work useful in your research, please consider citing:

```bibtex
@misc{ovie2026,
      title={One View Is Enough: In-the-Wild Monocular Pretraining for Novel View Generation},
      author={Adrien Ramanana Rahary and Nicolas Dufour and Patrick Perez and David Picard},
      year={2026},
      eprint={2603.23488},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.23488},
}
```

The paper appeared at NeurIPS 2026; models and code are at [kyutai-labs/ovie](https://github.com/kyutai-labs/ovie) and [huggingface.co/collections/kyutai/ovie](https://huggingface.co/collections/kyutai/ovie).
