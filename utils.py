#!/usr/bin/env python3
# clarifi3d/utils.py
"""
UTILS for Clarifi3D segmentation pipeline optimized for HPC environments (NVIDIA A100).
"""

# ------------------------
# Standard Library Imports
# ------------------------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import logging
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

# ------------------------
# Third-Party Imports
# ------------------------
import gc
import numpy as np
import pandas as pd
import cupy as cp
from scipy.stats import mode
from cupyx.scipy.ndimage import label as label_gpu
from cupyx.scipy.ndimage import maximum_filter as gpu_maximum_filter
from cupyx.scipy.ndimage import minimum_filter as gpu_minimum_filter
from cupyx.scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table
import torch
from torch import Tensor


# ------------------------
# Logging Configuration
# ------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(message)s')

# ========================
# 1. Memory Profiling and Allocation Utilities
# ========================

def profile_memory(msg: str) -> None:
    mempool = cp.get_default_memory_pool()
    used_mb = mempool.used_bytes() / 1e6
    logging.info(f"{msg} | GPU memory used: {used_mb:.2f} MB")

def setup_logging(log_path: Union[str, Path], level: int = logging.INFO) -> None:
    """Configure root logger to file and stdout."""
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%S"
    )

    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def util_profile_gpu_memory(message: str) -> None:
    pool = cp.get_default_memory_pool()
    logging.info(f"{message} | GPU Memory used: {pool.used_bytes() / 1e6:.2f} MB")


def util_clear_gpu_memory(*variables):
    """Delete provided variables, clear GPU and Python memory."""
    for var in variables:
        try:
            del var
        except Exception:
            pass
    cp.get_default_memory_pool().free_bytes()
    torch.cuda.empty_cache()
    gc.collect()


def add_file_logger(log_path: Union[str, Path], level: int = logging.DEBUG) -> logging.FileHandler:
    """
    Add a file-specific logger handler to the root logger.

    Args:
        log_path: Path to the log file.
        level: Logging level for the file handler (default: DEBUG).

    Returns:
        The FileHandler object for optional later removal.
    """
    logger = logging.getLogger()

    # Avoid duplicating handlers for the same file
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path):
            return h  # Already exists

    handler = logging.FileHandler(log_path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return handler


def safe_allocate_check(shape: tuple, dtype=cp.float32, usage_fraction: float = 0.85) -> None:
    bytes_req = np.prod(shape) * cp.dtype(dtype).itemsize
    available = cp.get_default_memory_pool().get_limit()
    if bytes_req > usage_fraction * available:
        raise MemoryError(
            f"Requesting {bytes_req/1e6:.2f} MB exceeds {usage_fraction*100:.0f}% of available GPU memory"
        )
    logging.info(f"Safe to allocate {bytes_req/1e6:.2f} MB for shape {shape}")

# ========================
# 2. Statistics and Logging Utilities
# ========================
def util_report_binary_stats(mask, label):
    nz = int(cp.count_nonzero(mask).get())
    nvox = int(mask.size)
    frac = nz / max(1, nvox)
    logging.info(f"[STAT] {label}: {nz} / {nvox} voxels ({frac:.3%})")
    
    
def util_compute_stats_and_log(
    image: np.ndarray,
    tag: str,
    output_dir: str,
    percentiles: list = None,
    write_file: bool = True,
    logger=logging
) -> dict:
    """
    Compute and log descriptive statistics of an image array (NumPy or CuPy).
    """
    if isinstance(image, cp.ndarray):
        image_cpu = cp.asnumpy(image)
    elif isinstance(image, np.ndarray):
        image_cpu = image
    else:
        raise TypeError("Input must be a NumPy or CuPy ndarray.")

    if image_cpu.size == 0:
        logger.warning(f"[{tag}] Empty image; skipping stats computation.")
        stats = {}
    else:
        stats = {
            'min': float(image_cpu.min()),
            'max': float(image_cpu.max()),
            'mean': float(image_cpu.mean()),
            'median': float(np.median(image_cpu)),
            'mode': float(mode(image_cpu.ravel(), keepdims=False).mode)
        }
        if percentiles is None:
            percentiles = list(range(10, 100, 10))
        p_vals = np.percentile(image_cpu, percentiles)
        stats.update({f"p{int(p)}": float(val) for p, val in zip(percentiles, p_vals)})

    logger.info(f"[{tag}] stats: {stats}")

    if write_file and output_dir:
        stats_path = os.path.join(output_dir, f"{tag}_stats.json")
        try:
            with open(stats_path, 'w') as fd:
                json.dump(stats, fd, indent=2)
            logger.info(f"[{tag}] Stats written to {stats_path}")
        except Exception as e:
            logger.warning(f"[{tag}] Failed to write stats: {e}")

    return stats

# ========================
# 3. Centroid & Region Properties
# ========================

def util_compute_centroids_gpu(label_img: cp.ndarray, logger=logging) -> cp.ndarray:
    """
    Compute centroids of labeled regions in a 3D CuPy array.
    Returns (N, 3) array of centroids for labels 1..N.
    """
    assert isinstance(label_img, cp.ndarray) and label_img.ndim == 3, \
        "Input label_img must be a 3D CuPy array"
    flat = label_img.ravel()
    mask = flat > 0
    labels = flat[mask]

    coords = cp.indices(label_img.shape, dtype=cp.float32).reshape(3, -1)[:, mask]
    ids, counts = cp.unique(labels, return_counts=True)
    if ids.size == 0:
        logger.warning("[centroids] No seed labels found in input.")
        return cp.empty((0, 3), dtype=cp.float32)

    max_id = int(ids.max().item()) + 1
    weighted = cp.zeros((3, max_id), dtype=cp.float32)
    totals = cp.zeros((max_id,), dtype=cp.float32)

    for axis in range(3):
        weighted[axis, :] = cp.bincount(labels, weights=coords[axis], minlength=max_id)
    totals = cp.bincount(labels, minlength=max_id).clip(min=1)

    mean_pos = weighted[:, ids] / totals[ids]
    centroids = mean_pos.T  # (N, 3)
    logger.info(f"[centroids] Computed {centroids.shape[0]} centroids.")
    return centroids


def util_compute_region_statistics(
    label_img,
    voxel_spacing: tuple[float, float, float] = None,
    properties: list = None,
    logger=logging
) -> pd.DataFrame:
    """
    Computes region statistics from a labeled image (CuPy or NumPy array).
    """
    if properties is None:
        properties = ["label", "area", "centroid", "bbox"]

    # Ensure array is NumPy (regionprops_table only supports NumPy)
    if isinstance(label_img, cp.ndarray):
        logger.info("[region_stats] Converting CuPy to NumPy for regionprops_table.")
        label_img_np = cp.asnumpy(label_img)
    elif isinstance(label_img, np.ndarray):
        label_img_np = label_img
    else:
        raise TypeError(f"label_img must be np.ndarray or cp.ndarray, got {type(label_img)}")

    if label_img_np.max() == 0:
        logger.warning("[region_stats] Input label image has no regions (max label == 0).")
        return pd.DataFrame()

    tab = regionprops_table(label_img_np, properties=properties)
    df = pd.DataFrame(tab)

    if voxel_spacing and "area" in df.columns:
        z, y, x = voxel_spacing
        df["volume_um3"] = df["area"] * z * y * x
        if all(f"centroid-{i}" in df for i in range(3)):
            df["centroid_z_um"] = df["centroid-0"] * z
            df["centroid_y_um"] = df["centroid-1"] * y
            df["centroid_x_um"] = df["centroid-2"] * x

    logger.info(f"[region_stats] Computed {len(df)} region(s).")
    return df

# ========================
# 4. Label/Seed Manipulation & Morphology
# ========================

def util_get_log_spot_seeds(log_img, mask, min_distance_voxels, threshold):
    size = tuple([max(3, int(s)) for s in min_distance_voxels])
    locmax = (log_img == gpu_maximum_filter(log_img, size=size, mode='reflect'))
    spots = locmax & (log_img > threshold) & mask
    labeled_spots, nseed = label_gpu(spots)
    logging.info(f"[SEED] Local maxima above {threshold:.5g}: {int(nseed)} seeds")
    return labeled_spots


def gpu_remove_small_objects(arr, min_size):
    """Remove objects smaller than min_size in a binary (CuPy) array."""
    labeled, num = label_gpu(arr)
    sizes = cp.bincount(labeled.ravel())
    # 0 is background
    mask_sizes = sizes >= min_size
    mask_sizes[0] = 0
    keep_mask = mask_sizes[labeled]
    return arr & keep_mask


def detect_adaptive_peroxisomal_seeds(
    lap_img: cp.ndarray,
    mask: cp.ndarray,
    voxel_size: tuple,
    logger=logging
) -> cp.ndarray:
    logger.info("[SEED] Begin GPU-optimized adaptive peroxisomal seed detection")

    voxel_volume = float(cp.asnumpy(cp.prod(cp.asarray(voxel_size))))
    median_volume_um3 = 0.4213
    cluster_cutoff_um3 = 0.9671

    median_diameter_um = ((6 * median_volume_um3 / np.pi) ** (1. / 3.)) * 2
    footprint_voxels = tuple(max(3, int(round(median_diameter_um / v)) | 1) for v in voxel_size)
    logger.info(f"[SEED] Using footprint (voxels): {footprint_voxels}")

    labeled, n = label_gpu(mask)
    if n == 0:
        logger.warning("[SEED] No objects found in mask.")
        return cp.zeros_like(mask, dtype=cp.uint16)

    flat = labeled.ravel()
    labels = cp.unique(flat[flat > 0])
    sizes = cp.bincount(flat, minlength=int(cp.max(labels)) + 1)
    label_sizes = sizes[labels]
    size_um3 = label_sizes * voxel_volume

    logger.info(f"[SEED] Total labeled objects: {int(labels.shape[0])}")
    pcts = cp.percentile(size_um3, [25, 50, 75])
    logger.info(f"[SEED] Volume percentiles (µm³): 25%={pcts[0]:.4f}, 50%={pcts[1]:.4f}, 75%={pcts[2]:.4f}")

    small_mask = size_um3 <= cluster_cutoff_um3
    large_mask = size_um3 > cluster_cutoff_um3
    small_labels = labels[small_mask]
    large_labels = labels[large_mask]

    seeds = cp.zeros_like(mask, dtype=cp.uint16)
    global_id_counter = 1
    small_seed_count = 0
    large_seed_count = 0

    # --- Small object centroids ---
    if small_labels.size > 0:
        centroids = compute_centroids_gpu(labeled, small_labels)
        centroids_int = cp.round(centroids).astype(cp.int32)
        Z, Y, X = labeled.shape
        in_bounds = (
            (centroids_int[:, 0] >= 0) & (centroids_int[:, 0] < Z) &
            (centroids_int[:, 1] >= 0) & (centroids_int[:, 1] < Y) &
            (centroids_int[:, 2] >= 0) & (centroids_int[:, 2] < X)
        )
        centroids_int = centroids_int[in_bounds]
        small_seed_count = centroids_int.shape[0]
        ids = cp.arange(global_id_counter, global_id_counter + small_seed_count, dtype=cp.uint16)
        seeds[centroids_int[:, 0], centroids_int[:, 1], centroids_int[:, 2]] = ids
        global_id_counter += small_seed_count

    # --- Large object filtered minima ---
    if large_labels.size > 0:
        large_obj_mask = cp.isin(labeled, large_labels)
        seeds_large, large_seed_count = spatially_filtered_minima_seeds(
            lap_img=lap_img,
            obj_mask=large_obj_mask,
            voxel_size=voxel_size,
            logger=logger,
            start_label=global_id_counter
        )
        seeds = cp.maximum(seeds, seeds_large)
        global_id_counter += large_seed_count

    cp._default_memory_pool.free_all_blocks()
    gc.collect()

    logger.info(f"[SEED] Seed counts → Small: {small_seed_count}, Large: {large_seed_count}, Total: {small_seed_count + large_seed_count}")
    logger.info(f"[SEED] Final seed count: {int(cp.count_nonzero(seeds))}")
    return seeds


def bounding_box(mask: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    coords = cp.argwhere(mask)
    start = cp.min(coords, axis=0)
    end = cp.max(coords, axis=0) + 1
    return start, end

def crop_with_bbox(volume: cp.ndarray, bbox: Tuple[cp.ndarray, cp.ndarray]) -> cp.ndarray:
    slices = tuple(slice(int(s), int(e)) for s, e in zip(*bbox))
    return volume[slices]

def edt_3d_gpu_objectwise(
    volume: cp.ndarray,
    voxel_size: Tuple[float, float, float],
    memory_threshold_gb: float = 1.0,
    logger=logging
) -> cp.ndarray:
    """
    Memory-aware, GPU-only, 3D EDT with bounding box cropping and fallback to slice-wise EDT when needed.

    Parameters:
        volume: 3D CuPy binary mask (e.g., a large peroxisome object)
        voxel_size: (z, y, x) voxel size in µm
        memory_threshold_gb: fallback threshold

    Returns:
        edt_full: 3D EDT (float32) same shape as input
    """
    assert volume.ndim == 3 and isinstance(volume, cp.ndarray)

    coords = cp.argwhere(volume)
    if coords.size == 0:
        return cp.zeros_like(volume, dtype=cp.float32)

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    cropped = volume[z0:z1, y0:y1, x0:x1]

    shape = cropped.shape
    est_mem_bytes = cropped.size * 4 * 2
    est_mem_gb = est_mem_bytes / (1024 ** 3)

    logger.debug(f"[EDT] Cropped object shape: {shape}, Estimated mem: {est_mem_gb:.2f} GB")

    try:
        if est_mem_gb <= memory_threshold_gb:
            edt_crop = distance_transform_edt(cropped.astype(cp.bool_), sampling=voxel_size).astype(cp.float32)
            logger.debug("[EDT] Used full 3D GPU EDT.")
        else:
            raise MemoryError("Estimated memory too high, fallback to 2.5D.")
    except Exception as e:
        logger.warning(f"[EDT] Fallback to 2.5D slice-wise GPU EDT due to: {str(e)}")
        edt_crop = cp.empty_like(cropped, dtype=cp.float32)
        for z in range(shape[0]):
            edt_crop[z] = distance_transform_edt(cropped[z], sampling=voxel_size[1:]).astype(cp.float32)

    edt_full = cp.zeros_like(volume, dtype=cp.float32)
    edt_full[z0:z1, y0:y1, x0:x1] = edt_crop
    cp._default_memory_pool.free_all_blocks()
    return edt_full

def spatially_filtered_minima_seeds(
    lap_img: cp.ndarray,
    obj_mask: cp.ndarray,
    voxel_size: Tuple[float, float, float],
    logger=logging,
    batch_size: int = 64,
    radius_um: float = 0.6617,
    max_seeds_per_object: int = 5,
    start_label: int = 1
) -> Tuple[cp.ndarray, int]:
    seeds = cp.zeros_like(obj_mask, dtype=cp.uint16)
    labeled, num = label_gpu(obj_mask)
    if num == 0:
        logger.warning("[SEED] No objects found for large seed placement.")
        return seeds, 0

    label_ids = cp.arange(1, num + 1, dtype=cp.int32)
    batch_ids = [label_ids[i:i + batch_size] for i in range(0, num, batch_size)]
    radius_vox = tuple(max(1, int(round(radius_um / v))) for v in voxel_size)
    logger.info(f"[SEED] NMS radius (voxels): {radius_vox}")

    global_seed_id = start_label
    total_seeds = 0
    contrast_used = 0
    fallback_used = 0

    for b, ids in enumerate(batch_ids):
        logger.debug(f"[SEED] Processing batch {b+1}/{len(batch_ids)}: {len(ids)} labels")

        for lbl in ids:
            mask_lbl = (labeled == lbl)
            if not cp.any(mask_lbl): continue

            bbox = bounding_box(mask_lbl)
            mask_crop = crop_with_bbox(mask_lbl, bbox)
            lap_crop = crop_with_bbox(lap_img, bbox)

            lap_masked = lap_crop.copy()
            lap_masked[~mask_crop] = lap_crop.max() + 1
            lap_min = gpu_minimum_filter(lap_masked, footprint=cp.ones(radius_vox))

            minima_mask = (lap_masked == lap_min) & mask_crop
            coords = cp.argwhere(minima_mask)

            if coords.shape[0] > 0:
                lap_vals = lap_crop[coords[:, 0], coords[:, 1], coords[:, 2]]
                lap_std = cp.std(lap_crop[mask_crop])
                lap_mean = cp.mean(lap_crop[mask_crop])
                contrast_thresh = lap_mean - 0.25 * lap_std
                contrast_mask = lap_vals < contrast_thresh
                coords = coords[contrast_mask]

                if coords.shape[0] > 0:
                    obj_volume_vox = cp.count_nonzero(mask_crop)
                    max_seeds = min(
                        max_seeds_per_object,
                        max(1, int(obj_volume_vox * cp.prod(cp.asarray(voxel_size)) / 1.0))
                    )
                    if coords.shape[0] > max_seeds:
                        topk_idx = cp.argsort(lap_vals[contrast_mask])[:max_seeds]
                        coords = coords[topk_idx]

                    global_coords = coords + bbox[0]
                    ids = cp.arange(global_seed_id, global_seed_id + global_coords.shape[0], dtype=cp.uint16)
                    seeds[global_coords[:, 0], global_coords[:, 1], global_coords[:, 2]] = ids
                    global_seed_id += global_coords.shape[0]
                    total_seeds += global_coords.shape[0]
                    contrast_used += 1
                    continue

            # Fallback: centroid
            centroid = cp.round(cp.mean(cp.argwhere(mask_lbl), axis=0)).astype(cp.int32)
            z, y, x = map(int, centroid)
            if 0 <= z < seeds.shape[0] and 0 <= y < seeds.shape[1] and 0 <= x < seeds.shape[2]:
                seeds[z, y, x] = global_seed_id
                logger.debug(f"[SEED] Fallback centroid used for label {lbl}")
                global_seed_id += 1
                total_seeds += 1
                fallback_used += 1

        cp._default_memory_pool.free_all_blocks()
        gc.collect()

    logger.info(f"[SEED] Final filtered minima seeds: {total_seeds}")
    logger.info(f"[SEED] Objects processed: {num}, with local minima: {contrast_used}, with fallback centroid: {fallback_used}")
    return seeds, total_seeds


def compute_centroids_gpu(label_volume: cp.ndarray, label_ids: cp.ndarray = None) -> cp.ndarray:
    """
    Computes 3D centroids of labeled regions in a CuPy ndarray without CPU fallback.
    
    Parameters:
        label_volume: cp.ndarray
            A 3D CuPy array where each object has a unique label (uint16 or uint32).
        label_ids: cp.ndarray (optional)
            A 1D array of label IDs to compute centroids for (excluding 0). If None, inferred from volume.

    Returns:
        centroids: cp.ndarray
            An array of shape (N, 3) with (z, y, x) centroids for each label ID.
    """
    assert label_volume.ndim == 3

    # Flatten volume and extract coordinates
    zyx = cp.argwhere(label_volume > 0)
    labels = label_volume[zyx[:, 0], zyx[:, 1], zyx[:, 2]]

    if label_ids is None:
        label_ids = cp.unique(labels)
        label_ids = label_ids[label_ids > 0]

    # Convert to float32 for accumulation
    zyx = zyx.astype(cp.float32)
    labels = labels.astype(cp.int32)

    max_label = int(cp.max(label_ids))
    counts = cp.bincount(labels, minlength=max_label + 1).astype(cp.float32)

    # Accumulate coordinates
    z_sum = cp.bincount(labels, weights=zyx[:, 0], minlength=max_label + 1)
    y_sum = cp.bincount(labels, weights=zyx[:, 1], minlength=max_label + 1)
    x_sum = cp.bincount(labels, weights=zyx[:, 2], minlength=max_label + 1)

    # Avoid divide-by-zero
    counts = cp.maximum(counts, 1e-8)

    z_centroid = z_sum[label_ids] / counts[label_ids]
    y_centroid = y_sum[label_ids] / counts[label_ids]
    x_centroid = x_sum[label_ids] / counts[label_ids]

    centroids = cp.stack([z_centroid, y_centroid, x_centroid], axis=1)
    return centroids


def util_detect_minima_seeds_physical(
    lap_img: cp.ndarray,
    mask: cp.ndarray,
    voxel_size: tuple,
    seed_min_dist_um: float = 0.509, # Default 0.509 μm
    min_size_um: float = 0.2036 # Minimum region size in microns
) -> cp.ndarray:
    """
    Find regional minima in lap_img within mask, with physical units for filter size and region area.
    Returns seed marker image with unique label per minimum region.
    """
    # Calculate filter window size in voxels (odd integers, at least 3)
    min_distance_voxels = tuple(max(3, int(round(seed_min_dist_um / v)) | 1) for v in voxel_size)
    # Compute minimum region size in voxels
    min_vox = int(cp.ceil(min_size_um / cp.prod(cp.array(voxel_size))))
    
    # Step 1: Detect regional minima in Laplacian image (within mask)
    min_img = gpu_minimum_filter(lap_img, size=min_distance_voxels, mode='reflect')
    minima = (lap_img == min_img) & mask

    # Step 2: Label all minima regions
    seeds, nseed = label_gpu(minima)
    
    # Step 3: Remove seeds smaller than min_vox (physically implausible)
    if min_vox > 1 and nseed > 0:
        counts = cp.bincount(seeds.ravel())
        keep = cp.where(counts >= min_vox)[0]
        # Zero is always background
        keep = keep[keep != 0]
        keep_mask = cp.isin(seeds, keep)
        seeds = seeds * keep_mask
        nseed = len(keep)
    
    logging.info(f"[SEED] Regional minima in Laplacian: {int(nseed)} valid seeds (min_dist {seed_min_dist_um} μm, min_size {min_size_um} μm³)")
    util_report_binary_stats(seeds > 0, "Seed locations")
    return seeds


def util_prune_seed_labels_gpu(
    seed_label: cp.ndarray,
    z_margin: int = 4,
    remove_border_seeds: bool = True,
    logger=logging
) -> cp.ndarray:
    """
    Prune seeds touching Z boundaries and relabel (GPU).
    """
    assert isinstance(seed_label, cp.ndarray) and seed_label.ndim == 3, \
        "Input seed_label must be a 3D CuPy array"
    labels = seed_label.astype(cp.int32)
    Z = labels.shape[0]

    if remove_border_seeds:
        margin = cp.zeros_like(labels, dtype=cp.bool_)
        margin[:z_margin, :, :] = True
        margin[-z_margin:, :, :] = True
        border_ids = cp.unique(labels[margin])
        border_ids = border_ids[border_ids > 0]
        if border_ids.size > 0:
            mask = cp.isin(labels, border_ids)
            labels[mask] = 0
            logger.info(f"[seeds] Removed {border_ids.size} seed(s) touching Z boundaries.")
        else:
            logger.info("[seeds] No seeds touched Z boundaries.")
    else:
        logger.info("[seeds] Skipping Z-boundary seed filtering.")

    relabeled, num = label_gpu(labels > 0)
    logger.info(f"[seeds] {num} seed region(s) retained after pruning and relabeling.")
    return relabeled


def util_make_structuring_ellipsoid(
    sigma_um,
    voxel_spacing: tuple[float, float, float],
    logger=None
) -> cp.ndarray:
    """
    Create a 3D triaxial ellipsoidal structuring element on the GPU.
    - sigma_um: float or (z_sigma, y_sigma, x_sigma), in microns.
    - voxel_spacing: (z, y, x) in microns.
    Returns:
        mask: cp.ndarray, boolean, shape determined by sigmas and spacing.
    """

    # Accept numpy/cupy arrays as well as tuple/list
    if isinstance(sigma_um, (float, int)):
        rz = ry = rx = float(sigma_um)
        if logger:
            logger.info(f"[structuring_ellipsoid] Using isotropic radius {rz:.3f} µm")
    elif (
        isinstance(sigma_um, (tuple, list)) and len(sigma_um) == 3
    ) or (
        hasattr(sigma_um, "shape") and len(sigma_um) == 3
    ):
        # Handles np.ndarray, cp.ndarray
        if not isinstance(sigma_um, (tuple, list)):
            sigma_um = sigma_um.tolist()
        rz, ry, rx = [float(v) for v in sigma_um]
        if logger:
            logger.info(f"[structuring_ellipsoid] Using radii: z={rz:.3f}, y={ry:.3f}, x={rx:.3f} µm")
    else:
        raise ValueError(f"sigma_um must be a float or a tuple/list/array of three floats, got {repr(sigma_um)}")

    assert all(v > 0 for v in (rz, ry, rx)), "All radii must be positive"

    # Compute bounding box in voxels
    nz = int(cp.ceil(rz / voxel_spacing[0]))
    ny = int(cp.ceil(ry / voxel_spacing[1]))
    nx = int(cp.ceil(rx / voxel_spacing[2]))
    if logger:
        logger.info(f"[structuring_ellipsoid] Structuring element shape: ({2*nz+1}, {2*ny+1}, {2*nx+1}) voxels")

    zz, yy, xx = cp.ogrid[-nz:nz+1, -ny:ny+1, -nx:nx+1]
    mask = (
        (zz * voxel_spacing[0] / rz) ** 2 +
        (yy * voxel_spacing[1] / ry) ** 2 +
        (xx * voxel_spacing[2] / rx) ** 2
    ) <= 1
    return mask

def util_normalize_cost_map(arr: cp.ndarray, logger=logging, eps: float = 1e-12) -> cp.ndarray:
    """
    Normalize a CuPy array to [0, 1] for cost map use, on GPU.
    """
    assert isinstance(arr, cp.ndarray), "Input must be a CuPy array"
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < eps:
        logger.warning(f"[normalize_cost_map] Near-constant input: min={mn:.4g}, max={mx:.4g}")
        return cp.zeros_like(arr, dtype=cp.float32)
    norm = (arr - mn) / (mx - mn)
    return norm.astype(cp.float32)

# ========================
# 5. Seed and Coverage Evaluation
# ========================

def evaluate_seed_coverage(
    seed_centroids: cp.ndarray,
    nucleus_mask: cp.ndarray,
    cell_mask: cp.ndarray
) -> tuple[int, int, int]:
    """
    Compute the number of seeds in nucleus/cell masks. Fully GPU.
    """
    if seed_centroids.size == 0:
        logging.warning("No centroids to evaluate.")
        return 0, 0, 0

    rounded = cp.round(seed_centroids).astype(cp.int32)

    z_nuc = cp.clip(rounded[:, 0], 0, nucleus_mask.shape[0] - 1)
    y_nuc = cp.clip(rounded[:, 1], 0, nucleus_mask.shape[1] - 1)
    x_nuc = cp.clip(rounded[:, 2], 0, nucleus_mask.shape[2] - 1)

    z_cell = cp.clip(rounded[:, 0], 0, cell_mask.shape[0] - 1)
    y_cell = cp.clip(rounded[:, 1], 0, cell_mask.shape[1] - 1)
    x_cell = cp.clip(rounded[:, 2], 0, cell_mask.shape[2] - 1)

    nuc_idx = cp.ravel_multi_index((z_nuc, y_nuc, x_nuc), nucleus_mask.shape)
    cell_idx = cp.ravel_multi_index((z_cell, y_cell, x_cell), cell_mask.shape)

    in_nucleus = nucleus_mask.ravel()[nuc_idx] > 0
    in_cell    = cell_mask.ravel()[cell_idx] > 0

    n_in_nucleus = int(cp.count_nonzero(in_nucleus).get())
    n_in_cell    = int(cp.count_nonzero(in_cell).get())
    total = int(seed_centroids.shape[0])

    logging.info(
        f"Seed coverage: {n_in_nucleus}/{total} in nucleus, {n_in_cell}/{total} in cell"
    )
    if cp.any((rounded < 0) | (rounded >= cp.array(nucleus_mask.shape))):
        logging.warning("[seed_coverage] Some seed centroids were out of bounds and were clamped.")
    return n_in_nucleus, n_in_cell, total

# ========================
# 6. Intensity Logging
# ========================

def log_seed_intensity_profiles(
    centroid_coords: np.ndarray,
    nucleus_volume: np.ndarray,
    cell_volume: np.ndarray
) -> None:
    """
    Log intensity statistics at seed centroid locations for nucleus/cell volumes.
    """
    if centroid_coords.size == 0:
        logging.warning("No seed centroids available for intensity logging.")
        return

    idxs = np.round(centroid_coords).astype(int)
    zi, yi, xi = idxs[:,0], idxs[:,1], idxs[:,2]
    mask = (
        (zi>=0)&(zi<nucleus_volume.shape[0]) &
        (yi>=0)&(yi<nucleus_volume.shape[1]) &
        (xi>=0)&(xi<nucleus_volume.shape[2])
    )
    zi, yi, xi = zi[mask], yi[mask], xi[mask]

    nuc_vals = nucleus_volume[zi,yi,xi]
    cell_vals = cell_volume[zi,yi,xi]

    def summarize_log(vals, name):
        if vals.size == 0:
            logging.warning(f"No valid {name} intensities.")
            return
        stats = {
            'min': float(vals.min()), 'max': float(vals.max()),
            'mean': float(vals.mean()), 'median': float(np.median(vals)),
            'p10': float(np.percentile(vals,10)), 'p90': float(np.percentile(vals,90))
        }
        logging.info(f"{name} intensities at centroids: {stats}")

    summarize_log(nuc_vals, "Nucleus")
    summarize_log(cell_vals, "Cell")


# ----------------------------------------------------------------------------
# Patch & Blending Utilities
# ----------------------------------------------------------------------------
def compute_patch_grid(
    shape: Tuple[int, int, int],
    patch_shape: Tuple[int, int, int] = (32, 1024, 1024),
    pad: Tuple[int, int, int] = (8, 64, 64)
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Compute coordinates of overlapping 3D patches for full-volume traversal.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Full volume shape as (Z, Y, X).
    patch_shape : Tuple[int, int, int]
        Target shape of each patch (Z, Y, X).
    pad : Tuple[int, int, int]
        Overlap padding added to each side (Z, Y, X).

    Returns
    -------
    List[Tuple[int, int, int, int, int, int]]
        List of coordinates as (z0, z1, y0, y1, x0, x1) for each patch.
    """
    Z, Y, X = shape
    pz, py, px = patch_shape
    dz, dy, dx = pad

    coords: List[Tuple[int, int, int, int, int, int]] = []

    for z in range(0, Z, pz):
        for y in range(0, Y, py):
            for x in range(0, X, px):
                z0, z1 = max(z - dz, 0), min(z + pz + dz, Z)
                y0, y1 = max(y - dy, 0), min(y + py + dy, Y)
                x0, x1 = max(x - dx, 0), min(x + px + dx, X)
                coords.append((z0, z1, y0, y1, x0, x1))

    return coords



def blend_patches(
    accum: Tensor,
    norm: Tensor,
    patch: Tensor,
    coords: Tuple[int, int, int],
    weight_cache: dict,
):
    """
    Blend `patch` into `accum` with normalized overlap using a 3D Hanning window.

    Parameters
    ----------
    accum : Tensor
        Accumulated output tensor (C, Z, Y, X), float32.
    norm : Tensor
        Normalization weights (1, Z, Y, X), float32.
    patch : Tensor
        Patch tensor to add (C, dZ, dY, dX), float32 or float16 (autocast supported).
    coords : Tuple[int, int, int]
        Top-left corner where the patch should be placed in (Z, Y, X).
    weight_cache : dict
        Dictionary that caches blending weights per patch shape to avoid recomputation.
    """
    _, dz, dy, dx = patch.shape
    z0, y0, x0 = coords
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx

    # Cache and retrieve blending weights for current patch size
    key = (dz, dy, dx)
    if key not in weight_cache:
        cp_wt = blending_weights_cp(key)
        torch_wt = torch.from_dlpack(cp_wt.toDlpack()).to(device=patch.device, dtype=patch.dtype)
        weight_cache[key] = torch_wt
    weights = weight_cache[key]  # shape: (dz, dy, dx)

    # Reshape for broadcasting
    weights = weights[None, :, :, :]  # (1, dZ, dY, dX)

    # Accumulate weighted patch
    accum[:, z0:z1, y0:y1, x0:x1] += patch * weights
    norm[:, z0:z1, y0:y1, x0:x1] += weights


def blending_weights_cp(shape: Tuple[int, int, int]) -> cp.ndarray:
    """
    Generate a 3D CuPy-based Hanning window for patch blending on GPU.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of the patch (Z, Y, X).

    Returns
    -------
    cp.ndarray
        3D array of blending weights with float32 precision on GPU.
    """
    z, y, x = shape

    wz = cp.hanning(z) if z > 1 else cp.ones(1, dtype=cp.float32)
    wy = cp.hanning(y) if y > 1 else cp.ones(1, dtype=cp.float32)
    wx = cp.hanning(x) if x > 1 else cp.ones(1, dtype=cp.float32)

    weights = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    return weights.astype(cp.float32)
# ========================
# END OF FILE
# ========================