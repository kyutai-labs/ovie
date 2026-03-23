import torch


def sample_pure_translation(points, translation_safety_factor=0.3):
    """
    Computes a pure camera translation (no rotation) based on the spread of the object.
    Constraint: The Z translation is clamped such that no point in the object ends up
    behind the camera (Z_cam < 0).
    """
    B, N, _ = points.shape
    device = points.device

    # Compute safe std
    mask = torch.isfinite(points)
    counts = mask.sum(dim=1).clamp(min=1.0)
    clean_points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    mu = clean_points.sum(dim=1) / counts
    diff = points - mu.unsqueeze(1)
    diff[~mask] = 0.0  # Zero out invalid diffs
    var = (diff**2).sum(dim=1) / (counts - 1).clamp(min=1.0)
    points_std = torch.sqrt(var)

    # Compute limits
    limits = points_std * translation_safety_factor
    rand_vec = torch.rand(B, 3, device=device) * 2 - 1
    t_raw = rand_vec * limits
    tx, ty, tz_raw = t_raw[:, 0], t_raw[:, 1], t_raw[:, 2]

    # Compute safe min Z
    z_vals = points[:, :, 2].clone()
    z_vals[~torch.isfinite(z_vals)] = float("inf")

    min_z = z_vals.min(dim=1).values
    min_z = torch.nan_to_num(min_z, posinf=0.0, neginf=0.0)

    # Construct extrinsics
    tz = torch.min(tz_raw, min_z)
    camera_pos = torch.stack([tx, ty, tz], dim=-1)

    R = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    t = camera_pos.unsqueeze(-1)

    extrinsics_cam_to_world = torch.cat([R, t], dim=-1)

    R_inv = R.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, t)
    extrinsics_world_to_cam = torch.cat([R_inv, t_inv], dim=-1)

    return extrinsics_cam_to_world, extrinsics_world_to_cam


def sample_pure_rotation(points, intrinsics, rotation_safety_factor=0.3):
    """
    Computes a pure camera rotation (no translation) with respect to the camera's FOV.
    """
    B = points.shape[0]
    device = points.device

    focal_length = intrinsics[:, 0, 0]
    half_width = intrinsics[:, 0, 2]
    half_fov_rad = torch.atan(half_width / focal_length)
    max_angle_rad = half_fov_rad * rotation_safety_factor

    # Standard Rotation Logic
    base_forward = torch.zeros(B, 3, device=device)
    base_forward[:, 2] = 1.0

    ref_x = torch.zeros(B, 3, device=device)
    ref_x[:, 0] = 1.0

    perp1 = ref_x
    perp2 = torch.cross(base_forward, perp1, dim=-1)

    theta = torch.rand(B, 1, device=device) * max_angle_rad.unsqueeze(-1)
    phi = torch.rand(B, 1, device=device) * 2 * torch.pi

    forward = (
        base_forward * torch.cos(theta)
        + perp1 * (torch.sin(theta) * torch.cos(phi))
        + perp2 * (torch.sin(theta) * torch.sin(phi))
    )
    forward = forward / torch.norm(forward, dim=-1, keepdim=True)

    world_up = torch.zeros(B, 3, device=device)
    world_up[:, 1] = 1.0

    right = torch.cross(world_up, forward, dim=-1)
    right_norm = torch.norm(right, dim=-1, keepdim=True)
    parallel_mask = right_norm.squeeze(-1) < 1e-4
    if parallel_mask.any():
        right[parallel_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        right = right / right_norm

    up = torch.cross(forward, right, dim=-1)

    R = torch.stack([right, up, forward], dim=-1)
    t = torch.zeros(B, 3, 1, device=device)

    extrinsics_world_to_cam = torch.cat([R, t], dim=-1)
    R_inv = R.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, t)
    extrinsics_cam_to_world = torch.cat([R_inv, t_inv], dim=-1)

    return extrinsics_cam_to_world, extrinsics_world_to_cam


def sample_rotation_translation(
    points,
    intrinsics,
    translation_safety_factor=0.3,
    rotation_safety_factor=0.3,
):
    # Sample pure translation
    trans_c2w, _ = sample_pure_translation(
        points,
        translation_safety_factor=translation_safety_factor,
    )

    # Sample pure rotation
    rot_c2w, _ = sample_pure_rotation(
        points,
        intrinsics,
        rotation_safety_factor=rotation_safety_factor,
    )

    # Compose: T_hybrid = T_rot ∘ T_trans
    R_t = trans_c2w[:, :, :3]  # identity
    t_t = trans_c2w[:, :, 3:]  # (B, 3, 1)

    R_r = rot_c2w[:, :, :3]  # rotation
    t_r = rot_c2w[:, :, 3:]  # zero

    R_hybrid = torch.bmm(R_r, R_t)  # = R_r
    t_hybrid = torch.bmm(R_r, t_t) + t_r  # = R_r @ t_t

    extrinsics_cam_to_world = torch.cat([R_hybrid, t_hybrid], dim=-1)

    # Invert for world → camera
    R_inv = R_hybrid.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, t_hybrid)
    extrinsics_world_to_cam = torch.cat([R_inv, t_inv], dim=-1)

    return extrinsics_cam_to_world, extrinsics_world_to_cam


def sample_from_normals_extrinsics(
    points,
    normals,
    dist_range=(0.8, 1.2),
    vertical_threshold=0.9,
):
    """
    Samples camera poses based on source image's normal map using a log-uniform
    distance along the normal, with optional verticality filtering.

    Args:
        points: (B, N, 3) tensor of 3D points
        normals: (B, N, 3) tensor of normals
        dist_range: tuple (min_dist, max_dist) scaling factor along normals
        vertical_threshold: max absolute Y component to filter "too vertical" normals

    Returns:
        extrinsics_cam_to_world, extrinsics_world_to_cam: (B, 3, 4) each
    """
    B, N, _ = points.shape
    device = points.device
    dtype = points.dtype

    # --- base weights (inverse distance) ---
    points_norm = torch.norm(points, dim=-1).clamp(min=1e-6)
    weights = 1.0 / points_norm

    # --- verticality filter ---
    if vertical_threshold is not None and vertical_threshold < 1.0:
        verticality = normals[..., 1].abs()
        valid_mask = verticality < vertical_threshold
        weights = weights * valid_mask.float()

    # --- identify completely invalid batches ---
    weights_sum = weights.sum(dim=1)
    failure_mask = weights_sum < 1e-8  # (B,)

    # --- safe copies ---
    safe_points = points.clone()
    safe_normals = normals.clone()
    safe_weights = weights.clone()

    if failure_mask.any():
        safe_weights[failure_mask] = 1.0
        up_vec = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        safe_points[failure_mask] = 0.0
        safe_normals[failure_mask] = up_vec.expand_as(safe_normals[failure_mask])

    # --- normalize weights to get proper categorical distribution ---
    probs = safe_weights / safe_weights.sum(dim=1, keepdim=True)

    # --- sample anchor points & normals ---
    indices = torch.multinomial(probs, num_samples=1)
    gather_idx = indices.unsqueeze(-1).expand(B, 1, 3)
    p_anchor = torch.gather(safe_points, 1, gather_idx).squeeze(1)
    n_anchor = torch.gather(safe_normals, 1, gather_idx).squeeze(1)

    # --- log-uniform distance along normal ---
    p_anchor_norm = torch.norm(p_anchor, dim=-1, keepdim=True).clamp(min=1e-6)
    log_min = torch.log(torch.tensor(dist_range[0], device=device, dtype=dtype))
    log_max = torch.log(torch.tensor(dist_range[1], device=device, dtype=dtype))
    rand_log = torch.rand(B, 1, device=device, dtype=dtype)
    rand_scale = torch.exp(log_min + rand_log * (log_max - log_min))
    d = p_anchor_norm * rand_scale
    camera_pos = p_anchor + n_anchor * d

    # --- build extrinsics via lookat ---
    R = lookat(camera_pos, p_anchor).to(device)
    t = camera_pos.unsqueeze(-1)
    extrinsics_cam_to_world = torch.cat([R, t], dim=-1)
    R_inv = R.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, t)
    extrinsics_world_to_cam = torch.cat([R_inv, t_inv], dim=-1)

    # --- overwrite totally-invalid batch items with identity ---
    if failure_mask.any():
        id_c2w, id_w2c = sample_identity_view(points)
        extrinsics_cam_to_world[failure_mask] = id_c2w[failure_mask]
        extrinsics_world_to_cam[failure_mask] = id_w2c[failure_mask]

    return extrinsics_cam_to_world, extrinsics_world_to_cam


def lookat(camera_pos, target_pos, up=torch.tensor([0.0, 1.0, 0.0])):
    if up.dim() == 1:
        up = up.expand_as(camera_pos)
    up = up.to(camera_pos.device)

    forward = target_pos - camera_pos
    forward = forward / torch.norm(forward, dim=-1, keepdim=True).clamp(min=1e-8)

    right = torch.cross(up, forward, dim=-1)
    right = right / torch.norm(right, dim=-1, keepdim=True).clamp(min=1e-8)

    up = torch.cross(forward, right, dim=-1)
    R = torch.stack([right, up, forward], dim=-1)
    return R


def sample_frontal_hemisphere_views(
    points,
    anchor_noise_scale=0.1,
    max_angle_deg=30.0,
    dist_range=(0.85, 1.3),
):
    B, N, _ = points.shape
    device = points.device

    valid = torch.isfinite(points).all(dim=-1)
    points_norm = torch.norm(points, dim=-1).clamp(min=1e-6)
    weights = (1.0 / points_norm) * valid.float()

    weights_sum = weights.sum(dim=1)
    failure_mask = weights_sum < 1e-8  # (B,)

    # --- Create safe local copies ---
    safe_points = points.clone()
    safe_weights = weights.clone()

    if failure_mask.any():
        # Make weights uniform but finite
        safe_weights[failure_mask] = 1.0

        # Replace invalid geometry with something harmless
        safe_points[failure_mask] = torch.zeros_like(safe_points[failure_mask])

    probs = safe_weights / safe_weights.sum(dim=1, keepdim=True)

    indices = torch.multinomial(probs, num_samples=1)
    gather_idx = indices.unsqueeze(-1).expand(B, 1, 3)
    target_anchor = torch.gather(safe_points, 1, gather_idx).squeeze(1)

    target_dist = target_anchor.norm(dim=1, keepdim=True).clamp(min=1e-3)
    anchor_noise = torch.randn(B, 3, device=device) * (target_dist * anchor_noise_scale)
    target_anchor = target_anchor + anchor_noise

    ref_dir = -target_anchor / (target_dist + 1e-6)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, 3)
    local_right = torch.cross(ref_dir, world_up, dim=1)
    local_right = local_right / (torch.norm(local_right, dim=1, keepdim=True) + 1e-6)
    local_up = torch.cross(local_right, ref_dir, dim=1)

    limit_rad = max_angle_deg * torch.pi / 180
    azimuth = (torch.rand(B, 1, device=device) * 2 - 1) * limit_rad
    elevation = (torch.rand(B, 1, device=device) * 2 - 1) * limit_rad

    new_dir = (
        ref_dir + local_right * torch.tan(azimuth) + local_up * torch.tan(elevation)
    )
    new_dir = new_dir / (torch.norm(new_dir, dim=1, keepdim=True) + 1e-6)

    zoom = torch.exp(
        torch.rand(B, 1, device=device)
        * (
            torch.log(torch.tensor(dist_range[1], device=device))
            - torch.log(torch.tensor(dist_range[0], device=device))
        )
        + torch.log(torch.tensor(dist_range[0], device=device))
    )

    final_dist = target_dist * zoom
    new_cam_pos = target_anchor + new_dir * final_dist

    R = lookat(new_cam_pos, target_anchor).to(device)
    t = new_cam_pos.unsqueeze(-1)

    extrinsics_cam_to_world = torch.cat([R, t], dim=-1)

    R_inv = R.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, t)
    extrinsics_world_to_cam = torch.cat([R_inv, t_inv], dim=-1)

    # --- Hard overwrite failures ---
    if failure_mask.any():
        id_c2w, id_w2c = sample_identity_view(points)
        extrinsics_cam_to_world[failure_mask] = id_c2w[failure_mask]
        extrinsics_world_to_cam[failure_mask] = id_w2c[failure_mask]

    return extrinsics_cam_to_world, extrinsics_world_to_cam


def sample_identity_view(points):
    """
    Samples the identity view (no transformation).
    """
    B = points.shape[0]
    device = points.device
    extrinsics_4x4 = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
    extrinsics = extrinsics_4x4[:, :3, :]
    return extrinsics, extrinsics


class ExtrinsicsSamplingRouter:
    """
    Routes input data to a specific sampling function based on weighted probabilities, with batched per-element sampling.
    """

    method_to_code = {
        "identity": 0,
        "pure_translation": 1,
        "pure_rotation": 2,
        "normals_derived": 3,
        "frontal_hemisphere": 4,
        "rotation_and_translation": 5,
    }

    def __init__(self, method_weights):
        """
        Args:
            method_weights (dict): Dictionary mapping function names (str) to weights (float/int).
        """
        self.methods = {
            "pure_translation": sample_pure_translation,
            "pure_rotation": sample_pure_rotation,
            "normals_derived": sample_from_normals_extrinsics,
            "frontal_hemisphere": sample_frontal_hemisphere_views,
            "identity": sample_identity_view,
            "rotation_and_translation": sample_rotation_translation,
        }

        self.method_names = list(method_weights.keys())
        self.weights = torch.tensor(
            [method_weights[k] for k in self.method_names], dtype=torch.float32
        )
        self.probs = self.weights / self.weights.sum()

    def __call__(self, points, config, normals=None, intrinsics=None):
        """
        Samples a method per batch element and executes it.

        Returns:
            extrinsics_cam_to_world: (B, 3, 4)
            extrinsics_world_to_cam: (B, 3, 4)
            sampling_method: (B,) integer tensor indicating method used (0=identity, 1=translation, ...)
        """
        B = points.shape[0]
        device = points.device

        # Sample one method index per batch element
        sampled_method_idx = torch.multinomial(
            self.probs, num_samples=B, replacement=True
        ).to(device)
        sampling_method = torch.zeros(B, dtype=torch.long, device=device)

        # Allocate output tensors
        out_cam_to_world = torch.empty(B, 3, 4, device=device)
        out_world_to_cam = torch.empty(B, 3, 4, device=device)

        # Process each method separately for efficiency
        for i, method_name in enumerate(self.method_names):
            mask = sampled_method_idx == i
            idxs = mask.nonzero(as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue

            points_subset = points[idxs]
            kwargs = {}

            # Map config to function args
            if method_name == "pure_translation":
                if "translation_safety_factor" in config:
                    kwargs["translation_safety_factor"] = config[
                        "translation_safety_factor"
                    ]

            elif method_name == "pure_rotation":
                if intrinsics is None:
                    # Fallback to identity if missing input
                    out_cam_to_world[idxs], out_world_to_cam[idxs] = (
                        sample_identity_view(points_subset)
                    )
                    sampling_method[idxs] = self.method_to_code["identity"]
                    continue
                if "rotation_safety_factor" in config:
                    kwargs["rotation_safety_factor"] = config["rotation_safety_factor"]

            elif method_name == "normals_derived":
                if normals is None:
                    # Fallback to identity if missing input
                    out_cam_to_world[idxs], out_world_to_cam[idxs] = (
                        sample_identity_view(points_subset)
                    )
                    sampling_method[idxs] = self.method_to_code["identity"]
                    continue

                normals_subset = normals[idxs]
                if "dist_range" in config:
                    kwargs["dist_range"] = config["dist_range"]
                if "vertical_threshold" in config:
                    kwargs["vertical_threshold"] = config["vertical_threshold"]

            elif method_name == "rotation_and_translation":
                if intrinsics is None:
                    # Fallback to identity if missing input
                    out_cam_to_world[idxs], out_world_to_cam[idxs] = (
                        sample_identity_view(points_subset)
                    )
                    sampling_method[idxs] = self.method_to_code["identity"]
                    continue
                if "rotation_safety_factor" in config:
                    kwargs["rotation_safety_factor"] = config["rotation_safety_factor"]
                if "translation_safety_factor" in config:
                    kwargs["translation_safety_factor"] = config[
                        "translation_safety_factor"
                    ]

            elif method_name == "frontal_hemisphere":
                param_map = {
                    "anchor_noise_scale": "anchor_noise_scale",
                    "max_angle_deg": "max_angle_deg",
                    "dist_range": "dist_range",
                }
                for config_key, arg_name in param_map.items():
                    if config_key in config:
                        kwargs[arg_name] = config[config_key]

            # Call the method
            if (
                method_name == "pure_rotation"
                or method_name == "rotation_and_translation"
            ):
                cam2world_sub, world2cam_sub = self.methods[method_name](
                    points_subset, intrinsics[idxs], **kwargs
                )
            elif method_name == "normals_derived":
                cam2world_sub, world2cam_sub = self.methods[method_name](
                    points_subset, normals_subset, **kwargs
                )
            else:
                cam2world_sub, world2cam_sub = self.methods[method_name](
                    points_subset, **kwargs
                )

            # Scatter back to outputs
            out_cam_to_world[idxs] = cam2world_sub
            out_world_to_cam[idxs] = world2cam_sub
            sampling_method[idxs] = self.method_to_code[method_name]

        return out_cam_to_world, out_world_to_cam, sampling_method
