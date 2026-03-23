import os
import json
import glob
import argparse
import torch
import numpy as np
from tqdm import tqdm


def find_image_folder(scene_root, sample_filename):
    # handles cases where image subdirectories have unpredictable names
    try:
        candidates = os.listdir(scene_root)
    except OSError:
        return None
    for item in candidates:
        item_path = os.path.join(scene_root, item)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, sample_filename)):
                return item_path
    return None


def parse_dl3dv_scene(json_path):
    scene_root_dir = os.path.dirname(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    w = float(data["w"])
    h = float(data["h"])

    # normalize intrinsics relative to image dimensions
    intrinsics = torch.tensor(
        [data["fl_x"] / w, data["fl_y"] / h, data["cx"] / w, data["cy"] / h],
        dtype=torch.float32,
    )

    distortion = torch.tensor([data["k1"], data["k2"]], dtype=torch.float32)

    at_np = np.array(data["applied_transform"], dtype=np.float32)

    # pad affine transform to 4x4 homogeneous matrix if needed
    if at_np.shape == (3, 4):
        bottom_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
        at_np = np.vstack([at_np, bottom_row])
    applied_transform = torch.from_numpy(at_np)

    # opengl to opencv coordinate flip matrix
    flip_matrix = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32
    )

    list_urls = []
    list_cameras = []
    list_images = []

    frames = data.get("frames", [])
    if not frames:
        return None

    # ensure deterministic ordering across runs
    frames.sort(key=lambda x: x["file_path"])

    first_frame_filename = os.path.basename(frames[0]["file_path"])
    image_dir = find_image_folder(scene_root_dir, first_frame_filename)
    if image_dir is None:
        print(f"Warning: Missing images for {scene_root_dir}")
        return None

    for frame in frames:
        filename = os.path.basename(frame["file_path"])
        abs_image_path = os.path.join(image_dir, filename)

        if not os.path.exists(abs_image_path):
            continue

        list_urls.append(abs_image_path)

        with open(abs_image_path, "rb") as img_f:
            img_bytes_np = np.frombuffer(img_f.read(), dtype=np.uint8).copy()
            list_images.append(torch.from_numpy(img_bytes_np))

        # align with re10k camera parameterization (flattened w2c extrinsics)
        frame_matrix = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        pose_c2w_opencv = applied_transform @ frame_matrix @ flip_matrix
        pose_w2c_opencv = torch.inverse(pose_c2w_opencv)
        extrinsics_flat = pose_w2c_opencv[:3, :].flatten()
        camera_vec = torch.cat([intrinsics, distortion, extrinsics_flat])
        list_cameras.append(camera_vec)

    if not list_images:
        return None

    return {
        "url": list_urls,
        "cameras": torch.stack(list_cameras),
        "images": list_images,
    }


def process_dataset(root_dir, output_dir, scenes_per_batch=16):
    print(f"Searching in {root_dir}...")
    json_files = glob.glob(
        os.path.join(root_dir, "**", "transforms.json"), recursive=True
    )
    print(f"Found {len(json_files)} scenes.")
    os.makedirs(output_dir, exist_ok=True)

    # chunk into batches to prevent out-of-memory errors on large datasets
    for i in tqdm(
        range(0, len(json_files), scenes_per_batch), desc="Processing Batches"
    ):
        batch_jsons = json_files[i : i + scenes_per_batch]
        batch_data = []
        for json_file in batch_jsons:
            try:
                scene_dict = parse_dl3dv_scene(json_file)
                if scene_dict:
                    batch_data.append(scene_dict)
            except Exception as e:
                print(f"Error {json_file}: {e}")

        if batch_data:
            batch_idx = i // scenes_per_batch
            output_path = os.path.join(
                output_dir, f"dl3dv_re10k_style_batch_{batch_idx:04d}.torch"
            )
            torch.save(batch_data, output_path)

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="process dl3dv dataset into batched torch files"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="path to the root directory of the dl3dv dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory where processed torch files will be saved",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="number of scenes per batch"
    )

    args = parser.parse_args()

    process_dataset(args.root_dir, args.output_dir, args.batch_size)
