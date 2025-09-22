#!/usr/bin/env python3
# clarifi3d/filters.py
"""
FILTERS for Clarifi3D segmentation pipeline optimized for HPC environments.
"""

# ------------------------
# Standard Library Imports
# ------------------------
import os
import time
import logging
import math
from typing import Union, Sequence

# ------------------------
# Third-Party Imports
# ------------------------
import numpy as np
import scipy.ndimage as ndi

import cupy as cp
import cupyx.scipy.ndimage as cndi
from cupyx.scipy.ndimage import (
    gaussian_filter as gpu_gaussian,
    gaussian_laplace as gpu_log,
    maximum_filter as gpu_max,
    convolve as gpu_convolve,
    label as gpu_label,
    grey_dilation,
    grey_erosion
)

# ------------------------
# Internal Imports
# ------------------------
from clarifi3d.utils import (
    profile_memory,
    util_profile_gpu_memory,
)

# Configure module logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(message)s')


def _get_module(arr):
    return cp.get_array_module(arr)

def filter_gaussian(volume, sigma):
    """ND Gaussian smoothing, CPU (SciPy) or GPU (CuPy/CUDA)."""
    xp = _get_module(volume)
    if xp is np:
        return ndi.gaussian_filter(volume, sigma=sigma, mode='reflect')
    # Use CuPy/scipy interface if available; else, custom kernel
    try:
        return cndi.gaussian_filter(volume, sigma=sigma, mode='reflect')
    except ImportError:
        return gpu_gaussian(volume, sigma=sigma, mode='reflect')

def filter_laplacian(volume):
    """Pure Laplacian (second derivative) on CPU/GPU."""
    xp = _get_module(volume)
    dims = volume.ndim
    kernel = xp.zeros((3,)*dims, dtype=xp.float32)
    center = (1,)*dims
    kernel[center] = -2*dims
    for ax in range(dims):
        idx = list(center)
        for delta in (-1, 1):
            idx[ax] = 1 + delta
            kernel[tuple(idx)] = 1
            idx[ax] = 1  # reset
    if xp is np:
        return ndi.convolve(volume, kernel, mode='reflect')

    return cndi.convolve(volume, kernel, mode='reflect')


fused_morph_laplace3d_kernel = cp.RawKernel(r'''
extern "C" __global__
void morph_laplace3d_ellipsoid(
    const float* src, float* dst,
    const bool* se,       // structuring element (3D, mask)
    int sz, int sy, int sx,         // volume dims
    int sz_se, int sy_se, int sx_se, // se dims
    int cz, int cy, int cx           // se center offsets (half size)
) {
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= sz || y >= sy || x >= sx) return;

    int idx = z * sy * sx + y * sx + x;
    float val = src[idx];
    float minval = val;
    float maxval = val;

    // Iterate SE
    for (int dz = 0; dz < sz_se; ++dz) {
        int zz = z + (dz - cz);
        if (zz < 0 || zz >= sz) continue;
        for (int dy = 0; dy < sy_se; ++dy) {
            int yy = y + (dy - cy);
            if (yy < 0 || yy >= sy) continue;
            for (int dx = 0; dx < sx_se; ++dx) {
                int xx = x + (dx - cx);
                if (xx < 0 || xx >= sx) continue;
                int se_idx = dz * sy_se * sx_se + dy * sx_se + dx;
                if (!se[se_idx]) continue;
                int n_idx = zz * sy * sx + yy * sx + xx;
                float nval = src[n_idx];
                if (nval > maxval) maxval = nval;
                if (nval < minval) minval = nval;
            }
        }
    }
    dst[idx] = maxval - 2.0f * val + minval;
}
''', 'morph_laplace3d_ellipsoid')

def fused_morphological_laplacian_3d_ellipsoid(
    volume: cp.ndarray,
    se_mask: cp.ndarray
) -> cp.ndarray:
    """
    volume: 3D cp.ndarray (float32).
    se_mask: 3D cp.ndarray, bool (arbitrary ellipsoid or mask).
    Returns Laplacian (float32, shape of volume).
    """
    assert volume.ndim == 3
    assert se_mask.ndim == 3

    sz, sy, sx = volume.shape
    sz_se, sy_se, sx_se = se_mask.shape
    # Find center of SE
    cz, cy, cx = sz_se // 2, sy_se // 2, sx_se // 2

    out = cp.empty_like(volume, dtype=cp.float32)
    threads = (8, 8, 8)
    blocks = ((sz + threads[0] - 1)//threads[0],
              (sy + threads[1] - 1)//threads[1],
              (sx + threads[2] - 1)//threads[2])
    fused_morph_laplace3d_kernel(
        blocks, threads,
        (
            volume.astype(cp.float32).ravel(),
            out.ravel(),
            se_mask.ravel(),
            sz, sy, sx,
            sz_se, sy_se, sx_se,
            cz, cy, cx
        )
    )
    return out


def morph_laplacian_fiji_cp(volume, structure):
    """
    GPU version of the FIJI-style morphological Laplacian.
    Returns 8-bit integer output consistent with ImageJ behavior.
    """
    dilated = cndi.grey_dilation(volume, footprint=structure)
    eroded = cndi.grey_erosion(volume, footprint=structure)
    lap = dilated - 2 * volume + eroded
    lap = lap + 128
    return cp.clip(lap, 0, 255).astype(cp.uint8)


def filter_laplacian_of_gaussian(volume, sigma):
    """LoG (spot enhancement); negative for positive peaks."""
    xp = _get_module(volume)
    if xp is np:
        return -ndi.gaussian_laplace(volume, sigma=sigma, mode='reflect')
    return -cndi.gaussian_laplace(volume, sigma=sigma, mode='reflect')

def filter_log_enhance(volume, sigma):
    """Gaussian smooth, then Laplacian."""
    return filter_laplacian(filter_gaussian(volume, sigma))

def threshold_otsu_cp(volume: cp.ndarray, bins: int = 256) -> float:
    """
    Fully CuPy-native implementation of Otsu's method that mimics FIJI's 8-bit behavior.
    Automatically rescales to [0, 255] range before computing the histogram.
    Returns threshold in original intensity units.
    """
    assert volume.ndim >= 1
    flat = volume.ravel()
    dtype = volume.dtype

    # Rescale to [0, 255]
    vmin = float(cp.min(flat))
    vmax = float(cp.max(flat))
    if vmin == vmax:
        return vmin  # flat image

    scaled = ((flat - vmin) / (vmax - vmin)) * 255.0
    scaled = cp.clip(scaled, 0, 255)
    scaled = scaled.astype(cp.uint8)

    # Histogram
    hist = cp.bincount(scaled, minlength=256).astype(cp.float64)
    total = cp.sum(hist)
    bin_centers = cp.arange(256, dtype=cp.float64)

    # Class weights
    w1 = cp.cumsum(hist)
    w2 = total - w1

    # Class means
    sum1 = cp.cumsum(hist * bin_centers)
    mean1 = sum1 / (w1 + 1e-8)
    mean2 = (sum1[-1] - sum1) / (w2 + 1e-8)

    # Between-class variance
    sigma_b_squared = w1 * w2 * (mean1 - mean2) ** 2

    # Max variance index
    idx = int(cp.argmax(sigma_b_squared))
    threshold_scaled = bin_centers[idx]  # in [0, 255]

    # Convert back to original intensity scale
    threshold_original = vmin + (threshold_scaled / 255.0) * (vmax - vmin)
    return float(threshold_original)

def local_maxima(dist_map, mask, min_distance):
    """
    Local maxima with at least `min_distance` separation.
    """
    xp = _get_module(dist_map)
    size = 2 * min_distance + 1
    footprint = xp.ones((size,)*dist_map.ndim, dtype=bool)
    if xp is np:
        local = (dist_map == ndi.maximum_filter(dist_map, footprint=footprint)) & mask
        seeds, _ = ndi.label(local)
    else:
        local = (dist_map == cndi.maximum_filter(dist_map, footprint=footprint)) & mask
        seeds, _ = cndi.label(local)
    return seeds.astype(xp.int32)


def filter_gpu_stack_default_threshold(volume: cp.ndarray) -> int:
    """
    Mimics FIJI's 'Default' thresholding:
    - Uses histogram over the entire stack
    - Ignores first and last bins
    - Applies iterative intermeans (IsoData variant)
    - Returns integer threshold using floor (FIJI behavior)
    """
    arr = volume.ravel()
    hist = cp.histogram(arr, bins=256, range=(0, 256))[0].get()  # CPU-side
    hist[0] = 0
    hist[-1] = 0
    bin_centers = np.arange(256)
    threshold = filter_isodata_threshold(hist, bin_centers)
    threshold_int = int(np.floor(threshold))
    logging.info(f"[THRESH] FIJI/Default: threshold={threshold_int}")
    return threshold_int

def filter_isodata_threshold(hist, bin_centers):
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0: 
        return bin_centers[len(bin_centers)//2]
    thresh = bin_centers[len(bin_centers)//2]
    while True:
        below = bin_centers < thresh
        above = ~below
        mean1 = (hist[below] * bin_centers[below]).sum() / (hist[below].sum() or 1)
        mean2 = (hist[above] * bin_centers[above]).sum() / (hist[above].sum() or 1)
        new_thresh = 0.5 * (mean1 + mean2)
        if abs(new_thresh - thresh) < 1e-6:
            return new_thresh
        thresh = new_thresh



def triangle_threshold_gpu(lap: cp.ndarray) -> int:
    """
    Compute the Triangle threshold for an 8-bit CuPy array `lap`.
    Returns an integer threshold [0–255].
    """
    # 1. GPU histogram
    hist_gpu = cp.histogram(lap.ravel(), bins=256, range=(0, 256))[0]
    hist = hist_gpu.get()  # small array → CPU

    # 2. First/last non-zero bins
    nz = np.nonzero(hist)[0]
    if nz.size == 0:
        return 0
    first, last = nz[0], nz[-1]

    # 3. Peak bin
    peak = hist.argmax()

    # 4. Flip if right tail is longer
    flipped = False
    if (peak - first) < (last - peak):
        hist = hist[::-1]
        first = len(hist) - 1 - last
        peak = len(hist) - 1 - peak
        flipped = True

    # 5. Line parameters
    x0, y0 = first, hist[first]
    x1, y1 = peak, hist[peak]
    dx, dy = x1 - x0, y1 - y0
    norm = np.hypot(dx, dy)

    # 6. Max perpendicular distance
    xs = np.arange(first, peak + 1)
    ys = hist[first : peak + 1]
    d = np.abs(dy * xs - dx * ys + x1*y0 - y1*x0) / norm
    thresh = xs[d.argmax()]

    # 7. Un-flip
    if flipped:
        thresh = len(hist) - 1 - thresh

    return int(thresh)

def threshold_triangle_gpu(lap: cp.ndarray) -> cp.ndarray:
    """
    Returns a boolean mask on GPU: True where lap < triangle-threshold.
    """
    t = triangle_threshold_gpu(lap)
    return lap < t

# — Usage in your segment_peroxisomes —
# lap = morph_laplacian_fiji_cp(img_gf, structure)
# mask = threshold_triangle_gpu(lap)
# util_report_binary_stats(mask, f"Triangle mask < {triangle_threshold_gpu(lap)}")


# ----------------------------------------------------------------------------
# GPU-accelerated Hessian eigenvalue computation
# ----------------------------------------------------------------------------
def filter_compute_hessian_largest_abs_eigenvalue(
    input_volume,
    voxel_spacing: tuple[float, float, float] = (0.2, 0.05179016, 0.05179016),
    hessian_sigma_um: float = 1.0,
    batch_size: int = 2_000_000,
    logger=logging
) -> cp.ndarray:
    """
    Compute the largest absolute eigenvalue of the local Hessian tensor at every voxel.
    Accepts either np.ndarray or cp.ndarray, always processes internally as CuPy.
    Returns CuPy array.
    """
    logger.info("Begin Hessian eigenvalue computation (GPU-accelerated)")

    # Accept either, always convert to cp.ndarray
    if isinstance(input_volume, np.ndarray):
        assert input_volume.ndim == 3, "input_volume must be 3D"
        vol_cp = cp.asarray(input_volume, dtype=cp.float32)
    elif isinstance(input_volume, cp.ndarray):
        assert input_volume.ndim == 3, "input_volume must be 3D"
        vol_cp = input_volume.astype(cp.float32, copy=False)
    else:
        raise TypeError("input_volume must be a 3D np.ndarray or cp.ndarray")

    sigma_vox = tuple(hessian_sigma_um / v for v in voxel_spacing)
    Z, Y, X = vol_cp.shape
    N = Z * Y * X

    # 2. Allocate flat GPU buffers for Hessian components
    hxx = cp.empty(N, dtype=cp.float32)
    hyy = cp.empty(N, dtype=cp.float32)
    hzz = cp.empty(N, dtype=cp.float32)
    hxy = cp.empty(N, dtype=cp.float32)
    hxz = cp.empty(N, dtype=cp.float32)
    hyz = cp.empty(N, dtype=cp.float32)

    # 3. Compute each Hessian component (sequentially, GPU-resident)
    for order, buf, lbl in [
        ((0,0,2), hzz, "Hzz"),
        ((0,2,0), hyy, "Hyy"),
        ((2,0,0), hxx, "Hxx"),
        ((0,1,1), hxy, "Hxy"),
        ((1,0,1), hxz, "Hxz"),
        ((1,1,0), hyz, "Hyz"),
    ]:
        tmp = gpu_gaussian(vol_cp, sigma=sigma_vox, order=order, mode='reflect')
        buf[:] = tmp.ravel()
        del tmp

    # 4. Allocate output buffer
    largest_flat = cp.empty(N, dtype=cp.float32)
    two_pi_over_3 = 2.0 * math.pi / 3.0

    # 5. Process in batches for OOM safety
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        sz = end - start
        a = hxx[start:end]; d = hyy[start:end]; f = hzz[start:end]
        b = hxy[start:end]; c = hxz[start:end]; e = hyz[start:end]

        # Closed-form 3x3 symmetric eigenvalue computation (per batch)
        q = (a + d + f) / 3.0
        p1 = b*b + c*c + e*e
        p2 = ((a - q)**2 + (d - q)**2 + (f - q)**2 + 2.0*p1) / 6.0
        p = cp.sqrt(p2)
        det_m = (
            (a - q)*(d - q)*(f - q)
            + 2.0 * b * c * e
            - (a - q) * e*e
            - (d - q) * c*c
            - (f - q) * b*b
        )
        r = det_m / (2.0 * p**3 + 1e-24)
        r = cp.clip(r, -1.0, 1.0)
        phi = cp.arccos(r) / 3.0
        e1 = q + 2.0 * p * cp.cos(phi)
        e2 = q + 2.0 * p * cp.cos(phi + two_pi_over_3)
        e3 = q + 2.0 * p * cp.cos(phi + 2.0 * two_pi_over_3)
        # Assign largest absolute eigenvalue per voxel
        largest_flat[start:end] = cp.maximum(cp.maximum(cp.abs(e1), cp.abs(e2)), cp.abs(e3))

    largest = largest_flat.reshape((Z, Y, X))
    util_profile_gpu_memory("Post-Hessian closed-form computation")
    logger.info(f"[hessian] Largest abs eigenvalue (GPU) computed, shape: {largest.shape}")

    # --- Return GPU array; export to CPU only at IO/CLI ---
    return largest
