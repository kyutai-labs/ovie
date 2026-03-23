import torch
import numpy as np
from PIL import Image
import logging
import sys
import math
from disc import (
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
import os


def novel_view_by_reprojection(
    points3d,
    images,
    intrinsics,
    new_extrinsics,
    splat_size=1.5,
    valid_mask=None,
    sigma_scale=0.5,
    normals_world=None,
):
    """
    Smooth, Gaussian splatting rendering of 3D points into a new camera view,
    with optional backface culling using world-space normal maps.

    normals_world: optional tensor of shape (B, 3, H, W). Assumes the same
                   raster order mapping between points3d and (H,W).
    """
    B, N, _ = points3d.shape
    _, C, H, W = images.shape
    device = points3d.device

    if valid_mask is not None:
        mask_flat = valid_mask.reshape(B, -1)
    else:
        mask_flat = torch.ones(B, N, dtype=torch.bool, device=device)

    # Transform points to new camera coords (rotation + translation)
    # new_extrinsics assumed shape (B, 3, 4) with [R | t]
    R = new_extrinsics[:, :, :3]  # (B, 3, 3)
    t = new_extrinsics[:, :, 3:]  # (B, 3, 1)
    points_new = torch.bmm(points3d, R.transpose(1, 2)) + t.transpose(1, 2)
    # invalidate masked points by pushing them far away (large z)
    points_new[~mask_flat] = torch.tensor([0.0, 0.0, 1e6], device=device)

    x, y, z = points_new[..., 0], points_new[..., 1], points_new[..., 2].clamp(min=1e-6)
    fx = intrinsics[:, 0, 0].unsqueeze(1)
    fy = intrinsics[:, 1, 1].unsqueeze(1)
    cx = intrinsics[:, 0, 2].unsqueeze(1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)

    u = (fx * (x / z) + cx) * (W - 1)
    v = (fy * (y / z) + cy) * (H - 1)

    # --- Prepare Gaussian splatting ---
    radius = int(splat_size)
    offsets = torch.arange(-radius, radius + 1, device=device)
    du, dv = torch.meshgrid(offsets, offsets, indexing="ij")
    du = du.flatten().float()
    dv = dv.flatten().float()
    sigma = splat_size * sigma_scale
    gauss_weights = torch.exp(-(du**2 + dv**2) / (2 * sigma**2))
    gauss_weights = gauss_weights / gauss_weights.sum()

    # Expand for batch
    K = du.shape[0]
    u_offsets = (u.unsqueeze(-1) + du.view(1, 1, K)).clamp(0, W - 1)
    v_offsets = (v.unsqueeze(-1) + dv.view(1, 1, K)).clamp(0, H - 1)

    # Flatten for scatter
    B_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N * K).reshape(-1)
    u_flat = u_offsets.reshape(-1)
    v_flat = v_offsets.reshape(-1)
    z_flat = z.unsqueeze(-1).expand(-1, -1, K).reshape(-1)

    # NOTE: assumes points correspond in raster order to image pixels.
    rgb_flat = images.permute(0, 2, 3, 1).reshape(B, -1, C)
    rgb_flat = rgb_flat.unsqueeze(2).expand(-1, -1, K, -1).reshape(-1, C)

    weight_flat = gauss_weights.unsqueeze(0).expand(B, N, K).reshape(-1)

    # compute linear index for each splat contribution
    linear_idx = (B_idx * (H * W) + v_flat.long() * W + u_flat.long()).long()
    total_pixels = B * H * W

    # --- Compute per-pixel minimum depth (true z-buffer) ---
    min_z_per_pixel = torch.full((total_pixels,), 1e9, device=device)
    min_z_per_pixel.scatter_reduce_(
        0, linear_idx, z_flat, reduce="amin", include_self=True
    )

    # For each contribution, get the min depth of its pixel
    min_z_at_point = min_z_per_pixel[linear_idx]

    # Depth difference relative to nearest surface in that pixel
    depth_diff = z_flat - min_z_at_point
    depth_tol = torch.maximum(
        1e-4 * min_z_at_point.abs(), torch.tensor(1e-4, device=device)
    )

    # Occlusion mask: keep contributions that are effectively the nearest (within tolerance)
    occluded_mask = depth_diff <= depth_tol

    # --- Backface culling using world-space normal map (if provided) ---
    if normals_world is not None:
        # sanity checks
        assert (
            normals_world.dim() == 4
            and normals_world.shape[0] == B
            and normals_world.shape[1] == 3
        ), "normals_world must be shape (B,3,H,W)"
        assert normals_world.shape[2] == H and normals_world.shape[3] == W, (
            f"normals_world H,W must match images H,W (got {normals_world.shape[2:]}, expected {(H, W)})"
        )
        # reshape normals to (B, N, 3) in raster order to match points3d
        n_world = normals_world.permute(0, 2, 3, 1).reshape(B, -1, 3)  # (B, N, 3)
        if n_world.shape[1] != N:
            raise ValueError(
                f"Number of points N={N} does not match normals map pixels {n_world.shape[1]}"
            )

        # normalize normals (avoid zero-length)
        n_world = n_world / (n_world.norm(dim=-1, keepdim=True) + 1e-9)

        # transform normals into camera space using same rotation (no translation)
        # same transform as for points: v_cam = v_world @ R^T
        n_cam = torch.bmm(n_world, R.transpose(1, 2))  # (B, N, 3)

        # view direction from point towards camera (camera at origin): view = camera - point = -points_new
        view_dir = -points_new  # (B, N, 3)

        # compute dot product between normal and view direction
        # positive dot => normal points towards camera (facing)
        dot_nv = (n_cam * view_dir).sum(dim=-1)  # (B, N)

        face_mask = dot_nv > 0.0  # (B, N), True -> front-facing
        # also invalidate points that were masked originally
        face_mask = face_mask & mask_flat

        # expand to K contributions per point
        face_mask_k = face_mask.unsqueeze(-1).expand(-1, -1, K).reshape(-1)
    else:
        # if no normal map provided, keep all points (subject to occlusion)
        face_mask_k = torch.ones(B * N * K, dtype=torch.bool, device=device)

    # --- Accumulate depth-weighted contributions, but only from non-occluded + front-facing pts ---
    z_buffer = min_z_per_pixel  # final z-buffer = per-pixel minimum
    rgb_buffer = torch.zeros((total_pixels, C), device=device)
    weight_buffer = torch.zeros((total_pixels,), device=device)

    eps = 1e-6
    # use inverse depth so closer points get larger base weight
    inv_depth_factor = 1.0 / (z_flat + eps)

    # final per-contribution weight: gaussian spatial * inverse depth
    final_weight = weight_flat * inv_depth_factor

    # apply occlusion mask (hard)
    keep_mask = occluded_mask & (final_weight > 0)
    # combine with front-face mask
    keep_mask = keep_mask & face_mask_k

    if keep_mask.any():
        idx_keep = linear_idx[keep_mask]
        w_keep = final_weight[keep_mask]
        rgb_keep = rgb_flat[keep_mask] * w_keep.unsqueeze(-1)

        # accumulate weighted colors and weights
        rgb_buffer.index_add_(0, idx_keep, rgb_keep)
        weight_buffer.index_add_(0, idx_keep, w_keep)

    # Normalize by accumulated weights
    rendered_img = (
        (rgb_buffer / (weight_buffer.unsqueeze(-1) + eps))
        .view(B, H, W, C)
        .permute(0, 3, 1, 2)
    )

    visibility_mask = (weight_buffer > 0).view(B, H, W).unsqueeze(1)

    return rendered_img, visibility_mask


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def sample_extrinsics(points, max_angle, dist_range, points_mask, off_center_sigma=0.0):
    """
    Sample camera positions on a cone around the line from weighted center to origin.

    Args:
        points: (B, N, 3) - 3D point positions
        max_angle: float - maximum cone angle in degrees
        dist_range: tuple (min, max) - distance multipliers
        points_mask: (B, N) - boolean mask for valid points
        off_center_sigma: float - standard deviation multiplier for Gaussian noise added to the center position

    Returns:
        extrinsics: (B, 3, 4) - camera extrinsic matrices [R | t] (camera-to-world)
        extrinsics_inv: (B, 3, 4) - inverse extrinsic matrices [R^T | -R^T t] (world-to-camera)
    """
    B, N, _ = points.shape
    device = points.device

    # --- Weighted center computation ---
    valid_points = torch.where(
        points_mask.unsqueeze(-1).expand_as(points), points, torch.zeros_like(points)
    )

    points_norm = torch.norm(valid_points, dim=-1, keepdim=True)
    valid_with_norm = points_mask.unsqueeze(-1) & (points_norm > 1e-6)
    weights = torch.where(
        valid_with_norm, 1.0 / points_norm, torch.zeros_like(points_norm)
    )
    weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    weights = weights / weights_sum
    weighted_center = (valid_points * weights).sum(dim=1)
    center_norm = torch.norm(weighted_center, dim=-1, keepdim=True).clamp(min=1e-8)

    if off_center_sigma > 0.0:
        noise = torch.randn_like(weighted_center) * (off_center_sigma * center_norm)
        weighted_center = weighted_center + noise

    # --- Sample camera position on cone ---
    dist_mult = (
        torch.rand(B, 1, device=device) * (dist_range[1] - dist_range[0])
        + dist_range[0]
    )
    camera_distance = dist_mult * center_norm
    cone_axis = -weighted_center / center_norm

    max_angle_rad = torch.deg2rad(torch.tensor(max_angle, device=device))
    theta = torch.rand(B, 1, device=device) * max_angle_rad
    phi = torch.rand(B, 1, device=device) * 2 * torch.pi

    up = torch.zeros(B, 3, device=device)
    up[:, 2] = 1.0
    alignment = torch.abs((cone_axis * up).sum(dim=-1))
    use_x = alignment > 0.9
    up[use_x, 2] = 0.0
    up[use_x, 0] = 1.0

    perp = torch.cross(cone_axis, up, dim=-1)
    perp = perp / torch.norm(perp, dim=-1, keepdim=True).clamp(min=1e-8)
    perp2 = torch.cross(cone_axis, perp, dim=-1)
    perp2 = perp2 / torch.norm(perp2, dim=-1, keepdim=True).clamp(min=1e-8)

    direction = (
        cone_axis * torch.cos(theta)
        + perp * (torch.sin(theta) * torch.cos(phi))
        + perp2 * (torch.sin(theta) * torch.sin(phi))
    )
    direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
    camera_pos = weighted_center + direction * camera_distance

    # --- Build camera extrinsics ---
    forward = weighted_center - camera_pos
    forward = forward / torch.norm(forward, dim=-1, keepdim=True).clamp(min=1e-8)

    world_up = torch.zeros(B, 3, device=device)
    world_up[:, 1] = 1.0
    right = torch.cross(world_up, forward, dim=-1)
    right_norm = torch.norm(right, dim=-1, keepdim=True)
    parallel_mask = right_norm.squeeze(-1) < 1e-4
    if parallel_mask.any():
        alt_ref = torch.zeros(B, 3, device=device)
        alt_ref[:, 0] = 1.0
        right[parallel_mask] = torch.cross(
            alt_ref[parallel_mask], forward[parallel_mask], dim=-1
        )
        right_norm = torch.norm(right, dim=-1, keepdim=True).clamp(min=1e-8)
    right = right / right_norm.clamp(min=1e-8)

    up = torch.cross(forward, right, dim=-1)
    up = up / torch.norm(up, dim=-1, keepdim=True).clamp(min=1e-8)

    R = torch.stack([right, up, forward], dim=-1)
    t = camera_pos.unsqueeze(-1)
    extrinsics = torch.cat([R, t], dim=-1)

    # --- Compute inverse extrinsics ---
    R_inv = R.transpose(1, 2)  # R^T
    t_inv = -torch.bmm(R_inv, t)  # -R^T t
    extrinsics_inv = torch.cat([R_inv, t_inv], dim=-1)

    return extrinsics, extrinsics_inv


def setup_logging(log_file="log.txt"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_infinite_loader(dataloader, sampler=None):
    """
    Yields batches indefinitely. Automatically increments the sampler's
    internal epoch seed to ensure proper shuffling across passes.
    """
    internal_epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(internal_epoch)
        for batch in dataloader:
            yield batch
        internal_epoch += 1


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        cosine_decay = 0.5 * (
            1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        )

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), skip_keywords=()):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            (len(param.shape) == 1)
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, config, total_steps):
    if not config or not config.get("enabled", False):
        return None

    warmup_steps = config.get("warmup_steps", 0)
    if "warmup_ratio" in config:
        warmup_steps = int(total_steps * config["warmup_ratio"])

    base_lr = optimizer.param_groups[0].get(
        "initial_lr", optimizer.param_groups[0]["lr"]
    )

    min_lr = config.get("min_lr", 0.0)
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )


def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param.data, alpha=1 - decay)


def charbonnier_loss(input, target, mask=None, epsilon=1e-3):
    input = input.to(torch.float32)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32) if mask is not None else None

    diff = input - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return loss.mean()


def l1_loss(input, target, mask=None):
    input = input.to(torch.float32)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32) if mask is not None else None

    diff = input - target
    loss = torch.abs(diff)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return loss.mean()


def l2_loss(input, target, mask=None):
    input = input.to(torch.float32)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32) if mask is not None else None

    diff = input - target
    loss = diff * diff
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return loss.mean()


def get_recon_loss_fn(config):
    loss_type = config["train"].get("recon_loss", "charbonnier")
    if loss_type == "charbonnier":
        epsilon = config["train"].get("charbonnier_epsilon", 1e-3)
        return lambda i, t, mask=None: charbonnier_loss(
            i, t, mask=mask, epsilon=epsilon
        )
    elif loss_type == "l1":
        return lambda i, t, mask=None: l1_loss(i, t, mask=mask)
    elif loss_type == "l2":
        return lambda i, t, mask=None: l2_loss(i, t, mask=mask)
    else:
        raise ValueError(f"Unknown reconstruction loss type: {loss_type}")


def save_checkpoint(
    model,
    ema_model,
    optimizer,
    scheduler,
    global_step,
    output_dir,
    logger,
    discriminator,
    disc_optimizer,
    disc_scheduler,
):
    ckpt = {
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "discriminator": discriminator.module.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
    }

    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if disc_scheduler is not None:
        ckpt["disc_scheduler"] = disc_scheduler.state_dict()

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_{global_step}.pt")
    torch.save(ckpt, ckpt_path)
    logger.info(f"Saved checkpoint at step {global_step} -> {ckpt_path}")
