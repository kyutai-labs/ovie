# Adapted from: https://github.com/facebookresearch/vggt

import torch
from .rotation import mat_to_quat, quat_to_mat


def extri_intri_to_pose_encoding(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple,
    pose_encoding_type="absT_quaR_FoV",
):
    """
    Convert MoGe extrinsics + normalized intrinsics to pose encoding.

    Args:
        extrinsics (torch.Tensor): Bx3x4 extrinsics [R|t]
        intrinsics (torch.Tensor): Bx3x3 normalized intrinsics (fx, fy in [0,1], cx, cy in [0,1])
        image_size_hw (tuple): (H, W) in pixels
        pose_encoding_type (str): currently only 'absT_quaR_FoV'

    Returns:
        pose_encoding (torch.Tensor): Bx9 tensor
            [:3] = translation
            [3:7] = quaternion
            [7:9] = FOV (radians)
    """
    H, W = image_size_hw

    # Convert normalized intrinsics to pixel units
    intrinsics_pix = intrinsics.clone()
    intrinsics_pix[..., 0, 0] *= W  # fx
    intrinsics_pix[..., 1, 1] *= H  # fy
    intrinsics_pix[..., 0, 2] *= W  # cx
    intrinsics_pix[..., 1, 2] *= H  # cy

    # Compute FOVs
    fov_h = 2 * torch.atan((H / 2) / intrinsics_pix[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics_pix[..., 0, 0])

    # Convert rotation to quaternion
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3]
    quat = mat_to_quat(R)

    # Concatenate into pose encoding
    pose_encoding = torch.cat(
        [T, quat, fov_h[..., None], fov_w[..., None]], dim=-1
    ).float()
    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding: torch.Tensor,
    image_size_hw: tuple,
    build_intrinsics=True,
    pose_encoding_type="absT_quaR_FoV",
):
    """
    Convert pose encoding back to MoGe-compatible extrinsics + normalized intrinsics.

    Args:
        pose_encoding (torch.Tensor): Bx9 tensor (absT_quaR_FoV)
        image_size_hw (tuple): (H, W)
        build_intrinsics (bool): whether to return normalized intrinsics
        pose_encoding_type (str): currently only 'absT_quaR_FoV'

    Returns:
        extrinsics (torch.Tensor): Bx3x4
        intrinsics (torch.Tensor or None): Bx3x3 normalized intrinsics
    """
    H, W = image_size_hw
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    # Rotation matrix
    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    intrinsics = None
    if build_intrinsics:
        # Convert FOV to focal lengths in pixels
        fy = (H / 2) / torch.tan(fov_h / 2)
        fx = (W / 2) / torch.tan(fov_w / 2)

        intrinsics_pix = torch.zeros(
            pose_encoding.shape[0], 3, 3, device=pose_encoding.device
        )
        intrinsics_pix[..., 0, 0] = fx
        intrinsics_pix[..., 1, 1] = fy
        intrinsics_pix[..., 0, 2] = W / 2
        intrinsics_pix[..., 1, 2] = H / 2
        intrinsics_pix[..., 2, 2] = 1.0

        # Normalize for MoGe
        intrinsics = intrinsics_pix.clone()
        intrinsics[..., 0, 0] /= W
        intrinsics[..., 1, 1] /= H
        intrinsics[..., 0, 2] /= W
        intrinsics[..., 1, 2] /= H

    return extrinsics, intrinsics
