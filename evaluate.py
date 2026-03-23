#!/usr/bin/env python3
import argparse
import json
import os
import random
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image

import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data import MultiViewSequenceDataset
from models.models import OVIE_models
from utils.pose_enc import extri_intri_to_pose_encoding


def prepare_scale_for_batch_sweep_batch(
    transforms_0_to_1: torch.Tensor, translation_scale_sweep: torch.Tensor
):
    """
    transforms_0_to_1: (B, 3, 4) where B = B * N
    translation_scale_sweep: (S,)
    returns: (B * S, 3, 4)
    """
    R = transforms_0_to_1[..., :3]  # (B, 3, 3)
    t = transforms_0_to_1[..., 3]  # (B, 3)
    B = transforms_0_to_1.shape[0]
    S = translation_scale_sweep.shape[0]

    t_scaled = t.unsqueeze(1) * translation_scale_sweep.unsqueeze(0).unsqueeze(
        -1
    )  # (B, S, 3)

    R_expanded = R.unsqueeze(1).expand(B, S, 3, 3)  # (B, S, 3, 3)

    extrinsics_b_s = torch.cat(
        [R_expanded, t_scaled.unsqueeze(-1)], dim=-1
    )  # (B, S, 3, 4)

    return extrinsics_b_s.reshape(B * S, 3, 4)  # (B*S, 3, 4)


def batch_psnr(
    batch_images: torch.Tensor, ref_image: torch.Tensor, max_val: float = 1.0
):
    if ref_image.ndim == 3:
        ref_image = ref_image.unsqueeze(0)
    mse = torch.mean((batch_images - ref_image) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10((max_val**2) / mse)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_to_pil(tensor):
    """
    Converts a tensor [C, H, W] in [0, 1] to a PIL Image.
    """
    if tensor.ndim == 4:
        # If batch dim exists, take first
        tensor = tensor[0]

    tensor = tensor.detach().cpu().clamp(0, 1)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    print("Creating dataset")
    dataset = MultiViewSequenceDataset(
        args.dataset_path,
        stride=args.stride,
        num_target_frames=args.num_target_frames + 1,
        image_size=args.image_size,
    )

    print("Creating dataloader")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,
        collate_fn=lambda x: x,
        shuffle=False,
    )

    print("Loading NVS model")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    checkpoint_basename = os.path.basename(args.checkpoint_path).split(".")[0]
    subfolder_name = f"{experiment_name}_{checkpoint_basename}_s={args.stride}_nframes={args.num_target_frames}_removefirstframe=True"
    args.output_folder = os.path.join(args.output_folder, subfolder_name)
    os.makedirs(args.output_folder, exist_ok=True)

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    model_cfg = config["model"]
    model_type = model_cfg["model_type"]

    model = OVIE_models[model_type](
        image_size=args.image_size,
        vit_use_qknorm=model_cfg.get("use_qknorm", False),
        vit_use_swiglu=model_cfg.get("use_swiglu", True),
        vit_use_rope=model_cfg.get("use_rope", False),
        vit_use_rmsnorm=model_cfg.get("use_rmsnorm", True),
        vit_wo_shift=model_cfg.get("wo_shift", False),
        vit_use_checkpoint=model_cfg.get("use_checkpoint", False),
    ).to(device)

    model.load_state_dict(ckpt["ema"])
    model.eval()

    # Metrics
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    if args.global_scale_factor is not None:
        print(f"Using fixed global scale factor: {args.global_scale_factor}")
        translation_scale_sweep = torch.tensor(
            [args.global_scale_factor], device=device, dtype=torch.float32
        )
    else:
        print(
            f"Sweeping scales from {args.scale_min} to {args.scale_max} with {args.num_scales} steps."
        )
        translation_scale_sweep = torch.linspace(
            args.scale_min, args.scale_max, steps=args.num_scales, device=device
        )

    # Prepare directories for saving images
    gt_root = os.path.join(args.output_folder, "gt")
    gen_root = os.path.join(args.output_folder, "gen")
    gifs_root = os.path.join(args.output_folder, "gifs")
    os.makedirs(gt_root, exist_ok=True)
    os.makedirs(gen_root, exist_ok=True)
    os.makedirs(gifs_root, exist_ok=True)

    # Streaming stats
    metric_keys = ["psnr", "ssim", "lpips", "psnr_copy", "ssim_copy", "lpips_copy"]
    stats = {
        "count": 0,
    }
    for k in metric_keys:
        stats[k] = {"mean": 0.0, "M2": 0.0}

    # Container for detailed metrics
    detailed_metrics_results = []

    def update(stat_dict, value):
        c = stats["count"]
        delta = value - stat_dict["mean"]
        stat_dict["mean"] += delta / c
        delta2 = value - stat_dict["mean"]
        stat_dict["M2"] += delta * delta2

    pbar = tqdm(total=args.num_samples, desc="Processing samples")

    # Evaluation loop
    for batch_list in dataloader:
        if stats["count"] >= args.num_samples:
            break

        # Parse dataloader output
        b_images = torch.stack([s["images"] for s in batch_list])  # [B, N, 3, H, W]
        b_poses = torch.stack([s["poses"] for s in batch_list])  # [B, N, 3, 4]

        B, N, C, H, W = b_images.shape

        if N < 1:
            continue

        # Baseline metrics (copy frame 0)
        if N > 1:
            base_src = b_images[:, 0].to(device)  # [B, 3, H, W]
            base_tgt = b_images[:, 1:].to(device)  # [B, N-1, 3, H, W]
            base_pred = base_src.unsqueeze(1).expand_as(base_tgt)  # [B, N-1, 3, H, W]

            base_pred_flat = base_pred.reshape(-1, C, H, W)
            base_tgt_flat = base_tgt.reshape(-1, C, H, W)

            base_psnr = batch_psnr(base_pred_flat, base_tgt_flat, max_val=args.max_val)
            base_psnr = base_psnr.view(B, -1).mean(dim=1)

            base_ssim_flat = torch.zeros(base_pred_flat.shape[0], device=device)
            for i in range(base_pred_flat.shape[0]):
                base_ssim_flat[i] = ssim(
                    base_pred_flat[i : i + 1],
                    base_tgt_flat[i : i + 1],
                    data_range=args.max_val,
                )
            base_ssim = base_ssim_flat.view(B, -1).mean(dim=1)

            with torch.no_grad():
                base_lpips = lpips_fn(base_pred_flat * 2 - 1, base_tgt_flat * 2 - 1)
            base_lpips = base_lpips.view(B, -1).mean(dim=1)
        else:
            base_psnr = torch.zeros(B, device=device)
            base_ssim = torch.zeros(B, device=device)
            base_lpips = torch.zeros(B, device=device)

        # Model evaluation

        src_imgs = b_images[:, 0]
        tgt_imgs = b_images
        tgt_poses = b_poses

        num_targets = N

        src_imgs_expanded = src_imgs.unsqueeze(1).expand(-1, num_targets, -1, -1, -1)

        image_0 = src_imgs_expanded.reshape(-1, C, H, W)
        image_gt = tgt_imgs.reshape(-1, C, H, W)
        transforms = tgt_poses.reshape(-1, 3, 4)

        B_eff = image_0.shape[0]

        transforms = transforms.to(device)
        image_0 = image_0.to(device)
        image_gt = image_gt.to(device)

        extrinsics_bs = prepare_scale_for_batch_sweep_batch(
            transforms, translation_scale_sweep
        )

        dummy_intrinsics = torch.zeros(1, extrinsics_bs.shape[0], 3, 3, device=device)

        camera = extri_intri_to_pose_encoding(
            extrinsics=extrinsics_bs.unsqueeze(0),
            intrinsics=dummy_intrinsics,
            image_size_hw=(args.image_size, args.image_size),
        )
        cam_token = camera[..., :7].squeeze(0)

        S = translation_scale_sweep.shape[0]
        B_times_S = B_eff * S

        image_0_repeat = (
            image_0.unsqueeze(1)
            .expand(B_eff, S, -1, -1, -1)
            .reshape(B_times_S, *image_0.shape[1:])
        )
        image_gt_repeat = (
            image_gt.unsqueeze(1)
            .expand(B_eff, S, -1, -1, -1)
            .reshape(B_times_S, *image_gt.shape[1:])
        )

        if "_4C" in model_type:
            full_valid_mask = torch.ones(
                image_0_repeat.shape[0],
                1,
                image_0_repeat.shape[2],
                image_0_repeat.shape[3],
                device=device,
                dtype=image_0_repeat.dtype,
            )
            image_0_repeat = torch.cat([image_0_repeat, full_valid_mask], dim=1)

        with torch.no_grad():
            preds = model(x=image_0_repeat, cam_params=cam_token)

        # Metrics
        psnr_vals = batch_psnr(preds, image_gt_repeat, max_val=args.max_val)
        psnr_vals = psnr_vals.reshape(B_eff, S)

        ssim_vals = torch.zeros(B_times_S, device=device)
        for i in range(B_times_S):
            ssim_vals[i] = ssim(
                preds[i : i + 1], image_gt_repeat[i : i + 1], data_range=args.max_val
            )
        ssim_vals = ssim_vals.reshape(B_eff, S)

        with torch.no_grad():
            lpips_out = lpips_fn(preds * 2 - 1, image_gt_repeat * 2 - 1)
        lpips_out = lpips_out.view(B_times_S, -1).mean(dim=1)
        lpips_vals = lpips_out.reshape(B_eff, S)

        # Select best scale per scene
        psnr_b_n_s = psnr_vals.view(B, N, S)
        ssim_b_n_s = ssim_vals.view(B, N, S)
        lpips_b_n_s = lpips_vals.view(B, N, S)

        scene_avg_psnr = psnr_b_n_s.mean(dim=1)
        best_idx_scene = torch.argmax(scene_avg_psnr, dim=1)
        gather_indices = best_idx_scene.view(B, 1, 1).expand(B, N, 1)

        final_psnr = torch.gather(psnr_b_n_s, 2, gather_indices).squeeze(2).view(-1)
        final_ssim = torch.gather(ssim_b_n_s, 2, gather_indices).squeeze(2).view(-1)
        final_lpips = torch.gather(lpips_b_n_s, 2, gather_indices).squeeze(2).view(-1)

        final_psnr_bn = final_psnr.view(B, N)
        final_ssim_bn = final_ssim.view(B, N)
        final_lpips_bn = final_lpips.view(B, N)

        # Metric Slicing (removing first frame)
        frame_start_idx = 1
        final_psnr_bn_sliced = final_psnr_bn[:, 1:]
        final_ssim_bn_sliced = final_ssim_bn[:, 1:]
        final_lpips_bn_sliced = final_lpips_bn[:, 1:]

        # Reshape images for saving
        preds_reshaped = preds.view(B, N, S, C, H, W)
        gt_reshaped = image_gt_repeat.view(B, N, S, C, H, W)[:, :, 0]

        # Loop over samples
        for b in range(B):
            if stats["count"] >= args.num_samples:
                break

            current_sample_id = stats["count"]
            stats["count"] += 1

            # Model metrics
            avg_psnr_clip = final_psnr_bn_sliced[b].mean().item()
            avg_ssim_clip = final_ssim_bn_sliced[b].mean().item()
            avg_lpips_clip = final_lpips_bn_sliced[b].mean().item()

            update(stats["psnr"], avg_psnr_clip)
            update(stats["ssim"], avg_ssim_clip)
            update(stats["lpips"], avg_lpips_clip)

            # Baseline metrics
            if N > 1:
                update(stats["psnr_copy"], base_psnr[b].item())
                update(stats["ssim_copy"], base_ssim[b].item())
                update(stats["lpips_copy"], base_lpips[b].item())

            # Save detailed metrics
            raw_indices = list(range(frame_start_idx, N))
            stride_adjusted_indices = [idx * args.stride for idx in raw_indices]

            sample_metrics = {
                "sample_idx": current_sample_id,
                "frame_idx": stride_adjusted_indices,
                "psnr": final_psnr_bn_sliced[b].cpu().tolist(),
                "ssim": final_ssim_bn_sliced[b].cpu().tolist(),
                "lpips": final_lpips_bn_sliced[b].cpu().tolist(),
            }
            detailed_metrics_results.append(sample_metrics)

            # Save images and GIF
            best_s = best_idx_scene[b]
            gif_frames = []

            for n_idx in range(N):
                img_gen = preds_reshaped[b, n_idx, best_s]
                img_gt = gt_reshaped[b, n_idx]

                filename = f"sample_{current_sample_id:05d}_frame_{n_idx:02d}.png"

                save_image(img_gen, os.path.join(gen_root, filename))
                save_image(img_gt, os.path.join(gt_root, filename))

                # GIF Frame
                pil_gen = tensor_to_pil(img_gen)
                pil_gt = tensor_to_pil(img_gt)

                dst = Image.new("RGB", (pil_gen.width + pil_gt.width, pil_gen.height))
                dst.paste(pil_gt, (0, 0))
                dst.paste(pil_gen, (pil_gt.width, 0))

                gif_frames.append(dst)

            if len(gif_frames) > 0:
                gif_path = os.path.join(
                    gifs_root, f"sample_{current_sample_id:05d}.gif"
                )
                gif_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=gif_frames[1:],
                    optimize=False,
                    duration=90,
                    loop=0,
                )

            pbar.update(1)

    pbar.close()

    # Finalize stats
    results = {}
    for k in metric_keys:
        if stats["count"] > 1:
            std = (stats[k]["M2"] / (stats["count"] - 1)) ** 0.5
        else:
            std = 0.0
        results[k] = {
            "mean": stats[k]["mean"],
            "std": std,
        }

    output = {
        "args": vars(args),
        "results": results,
        "num_samples": stats["count"],
        "lpips_net": "vgg",
        "model_type": model_type,
    }

    os.makedirs(os.path.dirname(args.output_folder), exist_ok=True)

    # Save JSON summary
    output_path = os.path.join(
        args.output_folder, f"eval_seq_{config['experiment_name']}.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save detailed metrics torch file
    if len(detailed_metrics_results) > 0:
        detailed_path = os.path.join(
            args.output_folder, f"eval_seq_{config['experiment_name']}_details.pt"
        )
        torch.save(detailed_metrics_results, detailed_path)
        print(f"Saved per-frame metrics to {detailed_path}")

    # Save NPZ files
    print("\nCompiling saved images into .npz files...")

    # Process both GT and Gen folders
    for folder_name in ["gt", "gen"]:
        src_dir = os.path.join(args.output_folder, folder_name)
        npz_path = os.path.join(args.output_folder, f"{folder_name}.npz")
        fnames = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])

        if len(fnames) == 0:
            print(f"No images found in {src_dir}, skipping NPZ creation.")
            continue

        img_list = []
        print(f"Loading {len(fnames)} images from {folder_name}...")

        for fname in tqdm(fnames, desc=f"Loading {folder_name}"):
            p = os.path.join(src_dir, fname)
            try:
                im = Image.open(p).convert("RGB")
                img_list.append(np.array(im))
            except Exception as e:
                print(f"Error loading {p}: {e}")

        if len(img_list) > 0:
            all_imgs = np.stack(img_list, axis=0)
            print(f"Saving {npz_path} with shape {all_imgs.shape}...")
            np.savez_compressed(npz_path, all_imgs)
        else:
            print(f"Failed to load any images for {folder_name}.")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Translation scale sweep evaluation")

    # Paths
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="evaluation")

    # Dataset arguments
    parser.add_argument(
        "--stride", type=int, default=3, help="Stride for frame selection"
    )
    parser.add_argument(
        "--num_target_frames",
        type=int,
        default=14,
        help="Number of frames (1 source + targets)",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)

    # Scale sweep
    parser.add_argument("--scale_min", type=float, default=0.0)
    parser.add_argument("--scale_max", type=float, default=2.0)
    parser.add_argument("--num_scales", type=int, default=16)
    parser.add_argument(
        "--global_scale_factor",
        type=float,
        default=None,
        help="If provided, uses this scale for all scenes instead of searching.",
    )

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_val", type=float, default=1.0)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=750,
        help="Total number of samples (including identity pairs) to process",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
