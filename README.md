# OVIE: One View Is Enough!
### Monocular Training for In-the-Wild Novel View Generation

[![Project Page](https://img.shields.io/badge/Project_Page-green?logo=googlechrome&logoColor=white)](https://kyutai.org/blog/2026-04-14-ovie)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.23488)
[![Model](https://img.shields.io/badge/🤗%20HuggingFace-kyutai%2Fovie-yellow)](https://huggingface.co/kyutai/ovie)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the official implementation and models for **OVIE** (*One View Is Enough! Monocular Training for In-the-Wild Novel View Generation*).

OVIE is a framework for monocular novel view synthesis that does not require multi-view image pairs for supervision. Instead, it is trained entirely on unpaired internet images.

![OVIE teaser](assets/teaser.jpeg)

---

## 🗂️ Table of Contents
- [Installation](#-installation)
- [Model Weights](#-model-weights)
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
git clone https://github.com/AdrienRR/ovie.git
cd ovie
uv sync
```
*Prefix all commands with `uv run` to ensure they run inside the managed environment.*

---

## 📥 Model Weights

Pretrained weights are hosted on the Hugging Face Hub at [kyutai/ovie](https://huggingface.co/kyutai/ovie) and are downloaded automatically when using `from_pretrained` (see [Inference](#-inference) below).

For **evaluation and training**, local checkpoint files are also required. Download them from the [Releases page](https://github.com/AdrienRR/ovie/releases) and place them inside the `assets/` folder:

* `ovie.pt` — main checkpoint used for evaluation (contains EMA weights).
* `dino_vit_small_patch8_224.pth` — used only for training; same checkpoint as in [RAE](https://github.com/bytetriper/RAE).

```text
OVIE/
├── assets/
│   ├── ovie.pt                         # evaluation
│   ├── dino_vit_small_patch8_224.pth  # training only
│   └── sample_image.jpg
├── configs/
│   └── config_ovie.yaml
├── models/
└── ...
```

---

## 🚀 Inference

We provide two Jupyter notebooks to get started quickly:

| Notebook | Weights source |
|---|---|
| `inference_huggingface.ipynb` | Downloaded automatically from [kyutai/ovie](https://huggingface.co/kyutai/ovie) |
| `inference_local.ipynb` | Loaded from a local `assets/ovie.pt` checkpoint |

```sh
uv run jupyter notebook inference_huggingface.ipynb
```

**Loading from the Hugging Face Hub (recommended):**

```python
import torch
from models.models import OVIEModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OVIEModel.from_pretrained("kyutai/ovie", revision="v1.0").to(device)
model.eval()
image_size = model.image_size  # 256, read from the saved config
```

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
      title={One View Is Enough! Monocular Training for In-the-Wild Novel View Generation},
      author={Adrien Ramanana Rahary and Nicolas Dufour and Patrick Perez and David Picard},
      year={2026},
      eprint={2603.23488},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.23488},
}
```
