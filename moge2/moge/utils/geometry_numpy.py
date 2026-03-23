from typing import *
from functools import partial
import math

import numpy as np
from scipy.ndimage import zoom, maximum_filter
import utils3d

from .tools import timeit


def weighted_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int,...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        return (x * w).mean(axis=axis) / np.clip(w.mean(axis=axis), eps, None)


def harmonic_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int,...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return 1 / (1 / np.clip(x, eps, None)).mean(axis=axis)
    else:
        w = w.astype(x.dtype)
        return 1 / (weighted_mean_numpy(1 / (x + eps), w, axis=axis, keepdims=keepdims, eps=eps) + eps)


def normalized_view_plane_uv_numpy(width: int, height: int, aspect_ratio: float = None, dtype: np.dtype = np.float32) -> np.ndarray:
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    uv = np.stack([u, v], axis=-1)
    return uv


def focal_to_fov_numpy(focal: np.ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal_numpy(fov: np.ndarray):
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov_numpy(intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fov_x = focal_to_fov_numpy(intrinsics[..., 0, 0])
    fov_y = focal_to_fov_numpy(intrinsics[..., 1, 1])
    return fov_x, fov_y


def point_map_to_depth_legacy_numpy(points: np.ndarray):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv_numpy(width, height, dtype=points.dtype)
    _, uv = np.broadcast_arrays(points[..., :2], uv)

    b = (uv * points[..., 2:]).reshape(*points.shape[:-3], -1)
    A = np.stack([points[..., :2], -uv], axis=-1).reshape(*points.shape[:-3], -1, 2)

    M = A.swapaxes(-2, -1) @ A 
    solution = (np.linalg.inv(M + 1e-6 * np.eye(2)) @ (A.swapaxes(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution

    depth = points[..., 2] + shift[..., None, None]
    fov_x = np.arctan(width / diagonal / focal) * 2
    fov_y = np.arctan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[: , None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def recover_focal_shift_numpy(points: np.ndarray, mask: np.ndarray = None, focal: float = None, downsample_size: Tuple[int, int] = (64, 64)):
    assert points.shape[-1] == 3, "Points should (H, W, 3)"

    height, width = points.shape[-3], points.shape[-2]

    uv = normalized_view_plane_uv_numpy(width=width, height=height)
    
    if mask is None:
        factors = (downsample_size[0] / height, downsample_size[1] / width, 1)
        points_lr = zoom(points, factors, order=1).reshape(-1, 3)
        uv_lr = zoom(uv, factors[:2] + (1,), order=1).reshape(-1, 2)
    else:
        points_lr, uv_lr, mask_lr = utils3d.np.masked_nearest_resize(points, uv, mask=mask, size=downsample_size)
    
    if points_lr.size < 2:
        return 1., 0.
    
    if focal is None:
        shift,focal = solve_optimal_focal_shift(uv_lr, points_lr)
    else:
        shift = solve_optimal_shift(uv_lr, points_lr, focal)

    return focal, shift


def norm3d(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]) + np.square(x[..., 2]))
    

def depth_occlusion_edge_numpy(depth: np.ndarray, mask: np.ndarray, thickness: int = 1, tol: float = 0.1):
    disp = np.where(mask, 1 / depth, 0)
    disp_pad = np.pad(disp, (thickness, thickness), constant_values=0)
    mask_pad = np.pad(mask, (thickness, thickness), constant_values=False)
    kernel_size = 2 * thickness + 1
    disp_window = utils3d.np.sliding_window(disp_pad, (kernel_size, kernel_size), 1, axis=(-2, -1))
    mask_window = utils3d.np.sliding_window(mask_pad, (kernel_size, kernel_size), 1, axis=(-2, -1))

    disp_mean = weighted_mean_numpy(disp_window, mask_window, axis=(-2, -1))
    fg_edge_mask = mask & (disp > (1 + tol) * disp_mean)
    bg_edge_mask = mask & (disp_mean > (1 + tol) * disp)

    edge_mask = (maximum_filter(fg_edge_mask.astype(np.uint8), size=3, mode='constant') > 0) & \
                (maximum_filter(bg_edge_mask.astype(np.uint8), size=3, mode='constant') > 0)

    return edge_mask


def disk_kernel(radius: int) -> np.ndarray:
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    kernel = ((X**2 + Y**2) <= radius**2).astype(np.float32)
    kernel /= np.sum(kernel)
    return kernel


def disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply disk blur using pure NumPy convolution.
    """
    if radius == 0:
        return image
    kernel = disk_kernel(radius)
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    if image.ndim == 2:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        blurred = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                blurred[i, j] = np.sum(padded[i:i+kH, j:j+kW] * kernel)
        return blurred

    elif image.ndim == 3:
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            padded = np.pad(image[..., c], ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    blurred[i, j, c] = np.sum(padded[i:i+kH, j:j+kW] * kernel)
        return blurred

    else:
        raise ValueError("Image must be 2D or 3D.")


def depth_of_field(img: np.ndarray, disp: np.ndarray, focus_disp: float, max_blur_radius: int = 10) -> np.ndarray:
    max_disp = np.max(disp)
    disp = disp / max_disp
    focus_disp = focus_disp / max_disp

    from scipy.ndimage import maximum_filter, uniform_filter

    # Dilated depth map
    dilated_disp = []
    for radius in range(max_blur_radius + 1):
        size = 2*radius + 1
        dilated_disp.append(maximum_filter(disp, size=size, mode='constant'))

    blur_radii = np.clip(abs(disp - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
    for radius in range(max_blur_radius + 1):
        dilated_blur_radii = np.clip(abs(dilated_disp[radius] - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
        mask = (dilated_blur_radii >= radius) & (dilated_blur_radii >= blur_radii) & (dilated_disp[radius] > disp)
        blur_radii[mask] = dilated_blur_radii[mask]
    blur_radii = np.clip(blur_radii, 0, max_blur_radius)
    blur_radii = uniform_filter(blur_radii.astype(np.float32), size=5)

    unique_radii = np.unique(blur_radii)
    precomputed = {r: disk_blur(img, r) for r in unique_radii}

    output = np.zeros_like(img)
    for r in unique_radii:
        mask = blur_radii == r
        output[mask] = precomputed[r][mask]

    return output
