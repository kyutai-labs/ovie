# OVIE: One View Is Enough! 
### Monocular Training for In-the-Wild Novel View Generation

[![Paper](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.23488)
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
- [Acknowledgments](#-acknowledgments)

---

## 🛠️ Installation

We use [`uv`](https://docs.astral.sh/uv/) by Astral to manage the Python environment and dependencies. It is a drastically faster drop-in replacement for standard Python packaging tools.

**1. Install `uv`:**
For macOS/Linux:
```sh
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```
*(Alternatively, you can install it via macOS Homebrew: `brew install uv`, or refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/) for Windows instructions).*

**2. Clone this repository and sync dependencies:**
Once `uv` is installed, clone the project and simply run `uv sync`. This will automatically resolve the required Python version (3.10.9) and install all dependencies defined in the `pyproject.toml`.
```sh
git clone [https://github.com/YOUR_USERNAME/OVIE.git](https://github.com/YOUR_USERNAME/OVIE.git)
cd OVIE
uv sync
```
*Note: Prefix your execution commands with `uv run` to ensure they run within this managed environment.*

---

## 📥 Model Weights

We provide pretrained model weights as a zipped file attached to the GitHub Releases.

1. Download the weights zip file from the [Releases page](https://github.com/AdrienRR/ovie/releases).
2. Extract the contents and place them inside the `assets/` folder at the root of the repository. 

**Important Note on Weights:**
* `ovie.pt`: The main checkpoint used for **inference and evaluation**.
* `dino_vit_small_patch8_224.pth`: This file is used **only for training**. It is the exact same checkpoint utilized in [Representation Autoencoders (RAE)](https://github.com/bytetriper/RAE).

Your directory structure should look like this:
```text
OVIE/
├── assets/
│   ├── dino_vit_small_patch8_224.pth
│   ├── ovie.pt
│   └── sample_image.jpg
├── configs/
│   └── config_ovie.yaml
├── data_preparation/
├── models/
└── ...
```

---

## 🚀 Inference

We provide an interactive Jupyter Notebook (`inference_minimal_example.ipynb`) to help you get started quickly. You can launch it using `uv`:

```sh
uv run jupyter notebook inference_minimal_example.ipynb
```

Alternatively, you can use the following minimal Python snippet to synthesize a novel view from a single image:

```python
import yaml
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image

from models.models import OVIE_models
from utils.pose_enc import extri_intri_to_pose_encoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load configuration and initialize model
config_path = "./configs/config_ovie.yaml"
ckpt_path = "./assets/ovie.pt"
image_size = 256

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]
model = OVIE_models[model_cfg["model_type"]](
    image_size=image_size,
    vit_use_qknorm=model_cfg.get("use_qknorm", False),
    vit_use_swiglu=model_cfg.get("use_swiglu", True),
    vit_use_rope=model_cfg.get("use_rope", False),
    vit_use_rmsnorm=model_cfg.get("use_rmsnorm", True),
    vit_wo_shift=model_cfg.get("wo_shift", False),
    vit_use_checkpoint=model_cfg.get("use_checkpoint", False),
).to(device)

# Load weights
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["ema"])
model.eval()

# 2. Prepare scene and target camera viewpoints
image_path = "./assets/sample_image.jpg"
img_pil = Image.open(image_path).convert("RGB").resize((image_size, image_size))
img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)

extrinsics = torch.tensor([[[1.0, 0.0, 0.0, -1.25],
                            [0.0, 1.0, 0.0, 0.5],
                            [0.0, 0.0, 1.0, -2.0]]], device=device)
dummy_intrinsics = torch.zeros(1, 1, 3, 3, device=device)

camera = extri_intri_to_pose_encoding(
    extrinsics=extrinsics.unsqueeze(0),
    intrinsics=dummy_intrinsics,
    image_size_hw=(image_size, image_size),
)
cam_token = camera[..., :7].squeeze(0) 

# 3. Generate novel view
with torch.no_grad():
    pred_tensor = model(x=img_tensor, cam_params=cam_token)

# 4. Display
pred_display = pred_tensor[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
plt.imshow(pred_display)
plt.axis("off")
plt.show()
```

---

## 🧹 Data Preprocessing

Before you can train the model or evaluate on specific datasets, you must preprocess your raw images. We provide scripts to handle parsing and formatting for both in-the-wild training datasets and DL3DV evaluation datasets.

**For In-the-Wild Training Images:**
Run the preprocessing script to extract depth and DINO features:
```sh
uv run python data_preparation/preprocess_in_the_wild_images.py \
    --data_path /PATH/TO/RAW/DATASET \
    --output_path /PATH/TO/PREPROCESSED/DATASET
```
Ensure that your preprocessed directories are correctly pointed to in the `data_path` lists inside your `configs/config_ovie.yaml` file.

**For DL3DV Evaluation Data:**
If you plan to evaluate on the DL3DV dataset, it must first be formatted into batched torch files using the provided script:
```sh
uv run python data_preparation/format_dl3dv.py \
    --root_dir /PATH/TO/DL3DV \
    --output_dir /PATH/TO/PROCESSED/DL3DV
```

DL3DV can be downloaded by following the [the official dataset Github repo](https://github.com/DL3DV-10K/Dataset)).

---

## 🏋️‍♂️ Training

Once your data is properly preprocessed and linked in the configuration file, you can launch distributed training directly with `uv` and `torchrun`. 

```sh
uv run torchrun --nproc_per_node <number_of_gpus> train.py --config configs/config_ovie.yaml
```

---

## 📊 Evaluation

To evaluate your trained model or the provided checkpoints on a benchmark dataset, use the `evaluate.py` script. 

**Evaluating on Real Estate 10K (RE10K):**
You can download the pre-processed Real Estate 10K dataset directly from Hugging Face here: [chenchenshi/re10k-sc](https://huggingface.co/datasets/chenchenshi/re10k-sc). 

Once downloaded (or if evaluating on the preprocessed DL3DV dataset), run:
```sh
uv run python evaluate.py \
    --dataset_path /PATH/TO/EVAL/DATASET \
    --config_path configs/config_ovie.yaml \
    --checkpoint_path assets/ovie.pt
```

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
