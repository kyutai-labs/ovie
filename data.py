from PIL import Image
import numpy as np
import os
from glob import glob
from torch.utils.data import Dataset
from safetensors import safe_open
import io
import random
from typing import Optional, Tuple
import torch
from torch.utils.data import IterableDataset
import torchvision.transforms as T
from utils.utils import center_crop_arr


def parse_cameras_from_pt(
    camera_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(camera_tensor, torch.Tensor):
        camera_tensor = torch.tensor(camera_tensor)

    fx = camera_tensor[..., 0]
    fy = camera_tensor[..., 1]
    cx = camera_tensor[..., 2]
    cy = camera_tensor[..., 3]

    zero = torch.zeros_like(fx)
    one = torch.ones_like(fx)

    intrinsics = torch.stack(
        [
            torch.stack([fx, zero, cx], dim=-1),
            torch.stack([zero, fy, cy], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )

    pose_raw = camera_tensor[..., 6:]
    batch_shape = camera_tensor.shape[:-1]
    pose_3x4 = pose_raw.reshape(*batch_shape, 3, 4)

    bottom_row = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=camera_tensor.device, dtype=camera_tensor.dtype
    )
    bottom_row = bottom_row.expand(*batch_shape, 1, 4)

    pose_c2w = torch.cat([pose_3x4, bottom_row], dim=-2)
    extrinsics_w2c = torch.linalg.inv(pose_c2w)

    return intrinsics, extrinsics_w2c, pose_c2w


def extract_pair_of_indices(
    seq_len: int, range_selection: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    if range_selection is None:
        return random.randrange(seq_len), random.randrange(seq_len)

    min_d, max_d = range_selection
    if min_d < 0 or max_d < min_d:
        raise ValueError("range_selection must satisfy 0 <= min_dist <= max_dist")

    max_possible_d = min(max_d, seq_len - 1)
    d_vals = list(range(max_possible_d + 1))
    weights = []

    for d in d_vals:
        if d < min_d or d > max_d:
            weights.append(0)
        elif d == 0:
            weights.append(seq_len)
        else:
            weights.append(2 * (seq_len - d))

    if sum(weights) == 0:
        return None

    d = random.choices(d_vals, weights=weights, k=1)[0]
    sign = 0 if d == 0 else (1 if random.random() < 0.5 else -1)
    num_starts = seq_len - d
    i0 = random.randrange(num_starts)
    return (i0, i0 + d) if sign >= 0 else (i0 + d, i0)


def byte_stream_to_image(byte_array):
    byte_stream = np.asarray(byte_array, dtype=np.int8).view(np.uint8)
    image_bytes = io.BytesIO(byte_stream.tobytes())
    img = Image.open(image_bytes)
    img.load()
    return img


class ImageOnlyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = []

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for f in os.listdir(root):
            if os.path.splitext(f)[1].lower() in exts:
                self.files.append(os.path.join(root, f))

        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


class PreprocessedDataset(Dataset):
    """
    Dataset for loading .safetensors files. All keys present in each file are loaded automatically.
    """

    def __init__(self, data_dirs):
        if isinstance(data_dirs, str):
            self.data_dirs = [data_dirs]
        else:
            self.data_dirs = data_dirs

        self.files = []
        for d in self.data_dirs:
            dir_files = sorted(glob(os.path.join(d, "*.safetensors")))
            self.files.extend(dir_files)

        if not self.files:
            raise RuntimeError(f"No .safetensors files found in {self.data_dirs}")

        self.img_to_file_map = self.get_img_to_safefile_map()

    def get_img_to_safefile_map(self):
        img_to_file = {}
        counter = 0
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt") as f:
                all_keys = list(f.keys())
                # Use the first key to determine the number of samples
                num_samples = f.get_slice(all_keys[0]).get_shape()[0]
                for i in range(num_samples):
                    img_to_file[counter] = {"safe_file": safe_file, "idx_in_file": i}
                    counter += 1
        return img_to_file

    def __len__(self):
        return len(self.img_to_file_map)

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file = img_info["safe_file"]
        img_idx = img_info["idx_in_file"]

        data = {}
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                data[key] = f.get_slice(key)[img_idx]

        return data


class MultiViewSequenceDataset(IterableDataset):
    def __init__(
        self,
        pt_dir,
        stride: int = 1,
        num_target_frames: int = 2,
        image_size=256,
        transform=None,
    ):
        self.pt_dir = pt_dir
        self.pt_files = [
            os.path.join(pt_dir, f)
            for f in os.listdir(pt_dir)
            if f.endswith((".pt", ".torch"))
        ]

        self.stride = stride
        self.num_target_frames = num_target_frames
        self.image_size = image_size
        self.transform = transform

    def __iter__(self):
        pt_files = self.pt_files.copy()

        for pt_path in pt_files:
            scenes = torch.load(pt_path, map_location="cpu")

            for scene in scenes:
                num_frames = scene["cameras"].shape[0]
                needed_span = (self.num_target_frames - 1) * self.stride

                if num_frames <= needed_span:
                    continue

                max_start_idx = num_frames - needed_span
                start_idx = random.randrange(max_start_idx)

                indices = [
                    start_idx + i * self.stride for i in range(self.num_target_frames)
                ]

                # Parse all cameras
                _, extrinsics_w2c, pose_c2w = parse_cameras_from_pt(scene["cameras"])

                # Get Extrinsics (World-to-Camera) for the first frame
                first_frame_c2w = pose_c2w[indices[0]]

                # Get Extrinsics (World-to-Camera) for all selected frames
                selected_extrinsics = extrinsics_w2c[indices]

                # Compute relative poses: [N, 4, 4] (broadcast first_frame_c2w to match batch size)
                relative_poses = selected_extrinsics @ first_frame_c2w.unsqueeze(0)

                # Crop to 3x4 (B, N, 3, 4)
                relative_poses = relative_poses[..., :3, :].float()

                images_list = []
                for idx in indices:
                    img = byte_stream_to_image(scene["images"][idx])
                    img = center_crop_arr(img, self.image_size)
                    img_tensor = T.ToTensor()(img)
                    if self.transform:
                        img_tensor = self.transform(img_tensor)
                    images_list.append(img_tensor)

                images_stack = torch.stack(images_list)

                yield {
                    "images": images_stack,  # Shape: [N, 3, H, W]
                    "poses": relative_poses,  # Shape: [N, 3, 4] (Relative to indices[0])
                }
