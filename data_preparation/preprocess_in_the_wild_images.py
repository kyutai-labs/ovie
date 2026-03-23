import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from moge2.moge.model.v2 import MoGeModel
from utils.utils import center_crop_arr
from transformers import AutoModel
from torchvision import transforms
import json

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    """
    Run feature extraction on full dataset and save the features.
    """
    assert torch.cuda.is_available(), (
        "Extract features currently requires at least one GPU."
    )

    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    output_dir = args.output_path
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        config_dict = {
            "data_path": args.data_path,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "depth_model": args.depth_model,
            "dino_model": args.dino_model,
        }
        config_filename = os.path.join(output_dir, "extract_features_config.json")
        with open(config_filename, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Saved config file to {config_filename}")

    dist.barrier()

    depth_estimator = MoGeModel.from_pretrained(args.depth_model).to(device)
    depth_estimator.eval()
    dino = AutoModel.from_pretrained(args.dino_model)
    dino.to(device)
    dino.eval()

    transform = transforms.Compose(
        [lambda img: center_crop_arr(img, args.image_size), transforms.ToTensor()]
    )
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total_data_in_loop = len(loader.dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    run_images = 0
    saved_files = 0

    images_list = []
    points3d_list = []
    intrinsics_list = []
    mask_list = []
    normals_list = []
    dino_features_list = []

    for batch_idx, (x, _) in enumerate(loader):
        run_images += len(x)
        if run_images % 100 == 0 and rank == 0:
            print(
                f"{datetime.now()} processing {run_images} of {total_data_in_loop} images"
            )

        x = x.to(device)  # (N, C, H, W)

        # dino requires standard imagenet stats for its input
        x_imagenet_normalized = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )(x)

        with torch.no_grad():
            depth_estimation = depth_estimator.infer(x)
            points3d, intrinsics, mask, normals = (
                depth_estimation["points"].detach().cpu(),
                depth_estimation["intrinsics"].detach().cpu(),
                depth_estimation["mask"].detach().cpu(),
                depth_estimation["normal"].detach().cpu(),
            )

            dino_last_hidden_state = (
                dino(x_imagenet_normalized).last_hidden_state.detach().cpu()
            )

            # strip out the cls and register tokens to isolate just the spatial patch embeddings
            dino_features = dino_last_hidden_state[
                :, 1 + dino.config.num_register_tokens :
            ]  # B, N, C

            # unflatten the 1d sequence back into a 2d spatial feature map
            dino_features = (
                dino_features.permute(0, 2, 1)
                .contiguous()
                .reshape(
                    dino_features.shape[0],
                    dino_features.shape[2],
                    int(dino_features.shape[1] ** 0.5),
                    int(dino_features.shape[1] ** 0.5),
                )
            )  # B, C, H, W

        if batch_idx == 0 and rank == 0:
            print("points3d shape", points3d.shape, "dtype", points3d.dtype)
            print("intrinsics shape", intrinsics.shape, "dtype", intrinsics.dtype)
            print("mask shape", mask.shape, "dtype", mask.dtype)
            print("normals shape", normals.shape, "dtype", normals.dtype)
            print(
                "dino_features shape", dino_features.shape, "dtype", dino_features.dtype
            )

        images_list.append(x.cpu())
        points3d_list.append(points3d)
        intrinsics_list.append(intrinsics)
        mask_list.append(mask)
        normals_list.append(normals)
        dino_features_list.append(dino_features)

        # flush to disk in chunks of ~10k images
        if len(images_list) == 10000 // args.batch_size:
            images_list_tensor = torch.cat(images_list, dim=0)
            points3d = torch.cat(points3d_list, dim=0)
            intrinsics = torch.cat(intrinsics_list, dim=0)
            mask = torch.cat(mask_list, dim=0)
            normals = torch.cat(normals_list, dim=0)
            dino_features = torch.cat(dino_features_list, dim=0)
            save_dict = {
                "images": images_list_tensor,
                "points3d": points3d,
                "intrinsics": intrinsics,
                "mask": mask,
                "normals": normals,
                "dino_features": dino_features,
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_filename = os.path.join(
                output_dir,
                f"features_rank{rank:02d}_shard{saved_files:03d}.safetensors",
            )
            save_file(
                save_dict,
                save_filename,
                metadata={
                    "total_size": f"{images_list_tensor.shape[0]}",
                    "dtype": f"{images_list_tensor.dtype}",
                    "device": f"{images_list_tensor.device}",
                },
            )
            if rank == 0:
                print(f"Saved {save_filename}")

            images_list = []
            points3d_list = []
            intrinsics_list = []
            mask_list = []
            normals_list = []
            dino_features_list = []
            saved_files += 1

            # clean up
            del (
                x,
                x_imagenet_normalized,
                dino_last_hidden_state,
                depth_estimation,
                images_list_tensor,
            )
            torch.cuda.empty_cache()

    if len(images_list) > 0:
        images_list_tensor = torch.cat(images_list, dim=0)
        points3d = torch.cat(points3d_list, dim=0)
        intrinsics = torch.cat(intrinsics_list, dim=0)
        mask = torch.cat(mask_list, dim=0)
        normals = torch.cat(normals_list, dim=0)
        dino_features = torch.cat(dino_features_list, dim=0)
        save_dict = {
            "images": images_list_tensor,
            "points3d": points3d,
            "intrinsics": intrinsics,
            "mask": mask,
            "normals": normals,
            "dino_features": dino_features,
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(
            output_dir, f"features_rank{rank:02d}_shard{saved_files:03d}.safetensors"
        )
        save_file(
            save_dict,
            save_filename,
            metadata={
                "total_size": f"{images_list_tensor.shape[0]}",
                "dtype": f"{images_list_tensor.dtype}",
                "device": f"{images_list_tensor.device}",
            },
        )
        if rank == 0:
            print(f"Saved {save_filename}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from images")

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save output files"
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Size to resize/crop images to"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for DataLoader"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--depth_model",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="Path or name of the pretrained depth estimator",
    )
    parser.add_argument(
        "--dino_model",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Path or name of the pretrained DINO model",
    )

    args = parser.parse_args()
    main(args)
