#!/usr/bin/env python3
# clarifi3d/seg.py
"""
clarifi3d.seg: High-performance 3D segmentation and feature extraction for A100 GPUs.
"""

# --- Standard Library Imports ---
import logging
import time
import gc
from math import ceil
from typing import Tuple, Optional, Union, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# --- Third-Party Imports ---
import numpy as np
import pandas as pd
import cupy as cp

# --- CuPy / SciPy GPU Filtering & Morphology ---
import cupyx.scipy.ndimage as cndi
from cupyx.scipy.ndimage import (
    gaussian_filter as gpu_gaussian,
    median_filter as gpu_median_filter,
    binary_closing as gpu_binary_closing,
    binary_opening as gpu_binary_opening,
    label as label_gpu,
)
from cupyx.scipy.ndimage import binary_fill_holes
from cupyx.scipy.ndimage import maximum_filter
from cupyx.scipy.ndimage import label as cplabel

# --- clarifi3d Internal Imports ---
from clarifi3d.filters import (
    filter_compute_hessian_largest_abs_eigenvalue,
    filter_gaussian,
    morph_laplacian_fiji_cp,
    filter_gpu_stack_default_threshold,
    threshold_otsu_cp,
    triangle_threshold_gpu,
)

from clarifi3d.utils import (
    profile_memory,
    detect_adaptive_peroxisomal_seeds,
    util_prune_seed_labels_gpu,
    util_compute_centroids_gpu,
    util_make_structuring_ellipsoid,
    util_normalize_cost_map,
    util_compute_stats_and_log,
    util_report_binary_stats,
    gpu_remove_small_objects,
)

from clarifi3d.watershed import dijkstra_26n_watershed


def segment_peroxisomes(
    raw: cp.ndarray,
    voxel_size: tuple,
    gauss_sigma_um: float = 0.15537048,
    laplacian_radii_um: tuple = (0.5179016, 0.2589508, 0.2589508),
    closing_radii_um: tuple = (0.5179016, 0.15537048, 0.15537048),
    min_size_um: float = 0.10358032,
    profile: bool = False,
    debug: bool = False
) -> cp.ndarray:
    logger = logging.getLogger()
    logger.info("[PEROX] ---- BEGIN PEROXISOME SEGMENTATION ----")

    assert raw.ndim == 3 and isinstance(raw, cp.ndarray), "raw must be 3D CuPy array"
    assert len(voxel_size) == 3 and all(isinstance(v, (float, int)) for v in voxel_size)
    # Force uint8 early to replicate FIJI
    raw8 = cp.clip(raw, 0, 255).astype(cp.uint8)

    # Step 1: Gaussian smoothing
    sigmas = tuple(gauss_sigma_um / v for v in voxel_size)
    img_gf = filter_gaussian(raw8, sigmas)
    img_gf = cp.clip(img_gf, 0, 255).astype(cp.uint8)
    del raw; cp._default_memory_pool.free_bytes(); gc.collect()

    # Step 2: Morphological Laplacian
    structure = util_make_structuring_ellipsoid(laplacian_radii_um, voxel_size)
    lap = morph_laplacian_fiji_cp(img_gf, structure)
    del img_gf; cp._default_memory_pool.free_bytes(); gc.collect()

    # Step 3: Global threshold (FIJI Default)
    threshold = 122
    mask = lap < threshold
    util_report_binary_stats(mask, "Binary mask < threshold")

    # Step 4: Median filter for noise reduction
    mask_closed = gpu_median_filter(mask.astype(cp.uint8), size=(1, 2, 2)).astype(bool)
    util_report_binary_stats(mask_closed, "Closed mask")
    del mask; cp._default_memory_pool.free_bytes(); gc.collect()

    # Step 5: Remove small objects
    # if you want FIJI’s behavior, force a 4-voxel cutoff:
    min_vox = 17
    mask_closed = gpu_remove_small_objects(mask_closed, min_vox)
    # if min_size_um:
    #     min_vox = int(cp.ceil(min_size_um / cp.prod(cp.asarray(voxel_size))))
    #     mask_closed = gpu_remove_small_objects(mask_closed, min_vox)
    #     util_report_binary_stats(mask_closed, f"Min size {min_vox} vox")
    #     cp._default_memory_pool.free_bytes(); gc.collect()

    # if cp.count_nonzero(mask_closed) == 0:
    #     logger.warning("[PEROX] Mask empty after filtering. Returning empty label image.")
    #     return cp.zeros_like(mask_closed, dtype=cp.uint16)

    # Step 6: Adaptive seed detection
    seeds = detect_adaptive_peroxisomal_seeds(
        lap_img=lap,
        mask=mask_closed,
        voxel_size=voxel_size,
        logger=logger
    ).astype(cp.uint16)

    util_report_binary_stats(seeds > 0, "Seed locations")
    cp._default_memory_pool.free_bytes(); gc.collect()

    # Step 7: Dijkstra watershed
    cost_map = lap.astype(cp.float32)
    logger.info("[PEROX] Calling Dijkstra 26N watershed...")
    perox_labels = dijkstra_26n_watershed(
        cost_map=cost_map,
        seed_markers=seeds,
        region_mask=mask_closed,
        plateau_limit=5,
        logger=logger,
    )
    logger.info("[PEROX] Dijkstra 26N watershed complete.")
    del cost_map, seeds, mask_closed, lap
    cp._default_memory_pool.free_bytes(); gc.collect()

    logger.info(f"[PEROX] ---- SEGMENTATION COMPLETE: {int(perox_labels.max())} objects ----")
    return perox_labels.astype(cp.uint16)

# -------- Seed Detection --------


def detect_and_filter_seed_points_gpu(
    smoothed_nucleus: cp.ndarray,
    nucleus_mask: cp.ndarray,
    voxel_spacing: Tuple[float, float, float],
    struct_sigma_um: Tuple[float, float, float] = (0.9, 3.7, 3.7),
    min_seed_distance_um: float = 6.0,
    z_margin: int = 4,
    remove_border_seeds: bool = True,
    logger=logging,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    logger.info("[SEED] Begin robust seed point detection from nucleus mask & smoothed signal.")

    # 1. Masked signal (GPU)
    signal = cp.where(nucleus_mask, smoothed_nucleus, 0)

    # 2. Structuring element
    struct = util_make_structuring_ellipsoid(
        sigma_um=struct_sigma_um,
        voxel_spacing=voxel_spacing
    )
    struct_gpu = cp.asarray(struct)
    if max(struct.shape) > 15:
        logger.warning(f"[SEED] Large structuring element: {struct.shape}. Consider reducing struct_sigma_um for speed.")

    # 2a. Maximum filter (fully GPU)
    local_max = maximum_filter(signal, footprint=struct_gpu, mode='reflect')
    peak_mask = (signal == local_max) & nucleus_mask

    # 3. Adaptive thresholding (all cp)
    nonzero_vals = signal[nucleus_mask]
    if nonzero_vals.size > 0:
        low = cp.quantile(nonzero_vals, 0.75)
        high = cp.quantile(nonzero_vals, 0.99)
        dyn_thresh = cp.maximum(low + 0.65 * (high - low), 0.0)
    else:
        logger.warning("[SEED] No nonzero values in nucleus mask; aborting.")
        return (cp.zeros_like(signal, dtype=cp.bool_),
                cp.zeros((0, 3), dtype=cp.float32),
                signal,
                cp.zeros_like(signal, dtype=cp.uint16))

    peaks = peak_mask & (signal > dyn_thresh)
    n_peaks = int(cp.count_nonzero(peaks).get())

    if n_peaks == 0:
        top_thresh = cp.quantile(nonzero_vals, 0.999)
        peaks = peak_mask & (signal > top_thresh)
        n_peaks = int(cp.count_nonzero(peaks).get())
        logger.warning(f"[SEED] Fallback: 99.9th percentile = {float(top_thresh):.6f} → {n_peaks} peaks")

    if n_peaks == 0:
        logger.error("[SEED] No seed candidates found.")
        return (cp.zeros_like(signal, dtype=cp.bool_),
                cp.zeros((0, 3), dtype=cp.float32),
                signal,
                cp.zeros_like(signal, dtype=cp.uint16))

    # 4. Label peaks (GPU)
    labeled, _ = cplabel(peaks)
    pruned = util_prune_seed_labels_gpu(labeled, z_margin=z_margin, remove_border_seeds=remove_border_seeds)

    if pruned.max() == 0:
        logger.warning("[SEED] No seed regions remain after pruning.")
        return (cp.zeros_like(signal, dtype=cp.bool_),
                cp.zeros((0, 3), dtype=cp.float32),
                signal,
                pruned)

    # 5. Centroid computation (fully GPU, vectorized)
    centroids = util_compute_centroids_gpu(pruned)
    if centroids.shape[0] > 1:
        coords_um = centroids * cp.asarray(voxel_spacing, dtype=cp.float32)
        diffs = coords_um[:, None, :] - coords_um[None, :, :]
        dists = cp.sqrt(cp.sum(diffs**2, axis=-1))
        mask = cp.triu(cp.ones_like(dists), k=1)
        close_pairs = cp.argwhere((dists < min_seed_distance_um) & (mask > 0))
        if close_pairs.shape[0] > 0:
            scores = signal[cp.round(centroids).astype(cp.int32)[:,0],
                            cp.round(centroids).astype(cp.int32)[:,1],
                            cp.round(centroids).astype(cp.int32)[:,2]]
            keep = cp.ones(centroids.shape[0], dtype=cp.bool_)
            for a, b in close_pairs:
                a, b = int(a), int(b)
                if scores[a] < scores[b]:
                    keep[a] = False
                else:
                    keep[b] = False
            centroids = centroids[keep]

    logger.info(f"[SEED] {centroids.shape[0]} seed centroids retained after spacing filter.")
    return peaks, centroids, signal, pruned

# -------- Nucleus and Cell Mask Inference --------

def infer_nucleus_mask(
    nucleus_volume: cp.ndarray,
    voxel_spacing: Tuple[float, float, float],
    base: str,
    output_directory: str,
    smoothing_sigma_um: Tuple[float, float, float] = (0.5179016, 0.5179016, 0.5179016),
    structure: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Inference mask for nuclei using 3D Morphological Gradient + Gaussian + Triangle threshold,
    mirroring infer_cell_mask logic.
    Returns a binary nucleus mask.
    """
    logger = logging.getLogger()
    logger.info("Starting nucleus mask inference (Morphological Gradient + Triangle)")

    # Step 1: Morphological Gradient
    if structure is None:
        structure = util_make_structuring_ellipsoid(
            sigma_um=(0.6, 0.36253112, 0.36253112),
            voxel_spacing=voxel_spacing,
            logger=logger
        )
    dilated = cndi.grey_dilation(nucleus_volume, footprint=structure)
    eroded = cndi.grey_erosion(nucleus_volume, footprint=structure)
    gradient = dilated - eroded

    # Step 2: Gaussian Smoothing (voxel-aware)
    sigma = tuple(s / v for s, v in zip(smoothing_sigma_um, voxel_spacing))
    smoothed = gpu_gaussian(gradient.astype(cp.float32), sigma=sigma)

    # Step 3: Normalize to [0, 255] for Triangle thresholding
    gmin, gmax = float(cp.min(smoothed)), float(cp.max(smoothed))
    if gmax - gmin < 1e-6:
        logger.warning("[nucleus_mask] Gradient has near-zero dynamic range; skipping mask generation.")
        nucleus_mask_gpu = cp.zeros_like(smoothed, dtype=cp.bool_)
    else:
        normalized = ((smoothed - gmin) / (gmax - gmin)) * 255
        grad_uint8 = cp.clip(normalized, 0, 255).astype(cp.uint8)

        # Step 4: Triangle thresholding
        threshold = triangle_threshold_gpu(grad_uint8)
        logger.info(f"[triangle] nucleus threshold = {threshold} (uint8 domain), raw min={gmin:.4f}, max={gmax:.4f}")
        nucleus_mask_gpu = grad_uint8 > threshold
        nucleus_mask_gpu = nucleus_mask_gpu.astype(cp.bool_)

    # Step 5: Log stats
    util_compute_stats_and_log(nucleus_mask_gpu.astype(cp.uint8), f"{base}_nucleus_mask", output_directory)

    return nucleus_mask_gpu, smoothed

def infer_cell_mask(
    cell_volume: cp.ndarray,
    voxel_spacing: Tuple[float, float, float],
    base: str,
    output_directory: str,
    smoothing_sigma_um: Tuple[float, float, float] = (0.4, 0.7768524, 0.7768524),
    closing_radius_um: Union[float, Tuple[float, float, float]] = 0.4,
    structure: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Inference mask using 3D Morphological Gradient + Gaussian + Triangle threshold.
    Returns a binary cell mask.
    """
    logger = logging.getLogger()
    logger.info("Starting cell mask inference (Morphological Gradient + Triangle)")

    # Step 1: Morphological Gradient
    if structure is None:
        structure = util_make_structuring_ellipsoid(
            sigma_um=(0.4, 0.2589508, 0.2589508),
            voxel_spacing=voxel_spacing,
            logger=logger
        )
    dilated = cndi.grey_dilation(cell_volume, footprint=structure)
    eroded = cndi.grey_erosion(cell_volume, footprint=structure)
    gradient = dilated - eroded

    # Step 2: Gaussian Smoothing
    sigma = tuple(s / v for s, v in zip(smoothing_sigma_um, voxel_spacing))
    smoothed = gpu_gaussian(gradient.astype(cp.float32), sigma=sigma)

    # Step 3: Normalize to [0, 255] for proper Triangle thresholding
    gmin, gmax = float(cp.min(smoothed)), float(cp.max(smoothed))
    if gmax - gmin < 1e-6:
        logger.warning("[cell_mask] Gradient has near-zero dynamic range; skipping mask generation.")
        cell_mask_gpu = cp.zeros_like(smoothed, dtype=cp.bool_)
    else:
        normalized = ((smoothed - gmin) / (gmax - gmin)) * 255
        grad_uint8 = cp.clip(normalized, 0, 255).astype(cp.uint8)

        # Step 4: Triangle thresholding
        threshold = triangle_threshold_gpu(grad_uint8)
        logger.info(f"[triangle] threshold = {threshold} (uint8 domain), raw min={gmin:.4f}, max={gmax:.4f}")
        cell_mask_gpu = grad_uint8 > threshold
        cell_mask_gpu = cell_mask_gpu.astype(cp.bool_)

    # Step 5: Log stats
    util_compute_stats_and_log(cell_mask_gpu.astype(cp.uint8), f"{base}_cell_mask", output_directory)

    return cell_mask_gpu

# -------- Segmentation Execution --------
def seed_coords_to_marker_volume_cp(
    centroids: cp.ndarray,
    mask: cp.ndarray,           # shape: (D, H, W), bool or binary
    dtype=cp.uint16
) -> cp.ndarray:
    """
    Places markers only at centroids inside mask.
    centroids: (N, 3), float or int, ZYX
    mask:      (D, H, W), bool or binary, region of interest
    Returns marker_vol: (D, H, W), uint16
    """
    shape = mask.shape
    marker_vol = cp.zeros(shape, dtype=dtype)
    if centroids.shape[0] == 0:
        return marker_vol

    inds = cp.rint(centroids).astype(cp.int32)
    z = cp.clip(inds[:, 0], 0, shape[0] - 1)
    y = cp.clip(inds[:, 1], 0, shape[1] - 1)
    x = cp.clip(inds[:, 2], 0, shape[2] - 1)
    # Check which seeds are inside the mask
    valid = mask[z, y, x]
    vals = cp.arange(1, len(z) + 1, dtype=dtype)
    marker_vol[z[valid], y[valid], x[valid]] = vals[valid]
    return marker_vol


def execute_segmentation(
    nucleus_mask: cp.ndarray,
    cell_mask: cp.ndarray,
    seed_centroids: cp.ndarray,
    volume_shape: Tuple[int, int, int],
    nucleus_intensity: cp.ndarray,
    cell_inference_map: cp.ndarray,
    border_crop_voxels: int = 4,
    logger=logging,
    voxel_size: Tuple[float, float, float] = (0.2, 0.05179016, 0.05179016),
) -> Tuple[cp.ndarray, np.ndarray]:
    logger.info("[exec] START segmentation")

    N = seed_centroids.shape[0]
    if N == 0:
        logger.warning("[exec] No seed centroids, returning zero label volumes")
        zero = cp.zeros(volume_shape, dtype=cp.uint16)
        return zero, zero

    # --- 1. Seed marker volume creation ---
    nucleus_marker_vol = seed_coords_to_marker_volume_cp(seed_centroids, nucleus_mask)
    cell_marker_vol    = seed_coords_to_marker_volume_cp(seed_centroids, cell_mask)

    logger.info(f"[exec] Nucleus seed markers placed: {int(cp.count_nonzero(nucleus_marker_vol))}")
    logger.info(f"[exec] Cell seed markers placed: {int(cp.count_nonzero(cell_marker_vol))}")

    # --- 2. Cost map normalization ---
    masked_cell_inference = cp.where(cell_mask, cell_inference_map, 0.0)
    cell_cost = util_normalize_cost_map(1.0 - masked_cell_inference)
    nuc_cost = util_normalize_cost_map(1.0 - nucleus_intensity)

    logger.info(f"[cost_map] cell min: {float(cell_cost.min())}, max: {float(cell_cost.max())}")
    logger.info(f"[cost_map] nucleus min: {float(nuc_cost.min())}, max: {float(nuc_cost.max())}")

    # --- 3. Watershed segmentation ---
    # -- nuclei (old-school GPU Dijkstra or plateau-aware variant) --
    nuc_lbl = dijkstra_26n_watershed(
        cost_map=nuc_cost,                 # (D,H,W), cp.ndarray float32
        seed_markers=nucleus_marker_vol,   # (D,H,W), cp.ndarray uint16
        region_mask=nucleus_mask,          # (D,H,W), cp.ndarray bool/uint8
        plateau_limit=5,
        logger=logger,
    )
    # -- cells (hybrid mesh-based, fallback to CPU for now) --

    cell_lbl = dijkstra_26n_watershed(
        cost_map=cell_cost,                # (D,H,W), cp.ndarray float32
        seed_markers=cell_marker_vol,      # (D,H,W), cp.ndarray uint16 (from cell seed positions)
        region_mask=cell_mask,             # (D,H,W), cp.ndarray bool/uint8
        plateau_limit=5,
        logger=logger,
    )

    # --- 4. Border cropping (optional) ---
    if border_crop_voxels > 0:
        nuc_lbl[:border_crop_voxels] = 0
        nuc_lbl[-border_crop_voxels:] = 0
        cell_lbl[:border_crop_voxels] = 0
        cell_lbl[-border_crop_voxels:] = 0

    # Ensure cell_lbl is cp.ndarray if you want to stay on GPU:
    # cell_lbl = cp.asarray(cell_lbl)

    return nuc_lbl, cell_lbl

# -------- Statistics/Features --------

def get_free_gpu_memory_gb() -> float:
    free_mem, _ = cp.cuda.runtime.memGetInfo()
    return free_mem / 1024**3

def estimate_batch_size(num_elements: int, element_size_gb: float, safety_margin: float = 0.85) -> int:
    free_gb = get_free_gpu_memory_gb()
    usable_gb = free_gb * safety_margin
    return max(1, int(usable_gb / (element_size_gb * num_elements)))

def edge_map(seg_ids, seg):
    # Collect unique labels on all 6 volume faces
    edge_labels = cp.concatenate([
        cp.unique(seg[0]),
        cp.unique(seg[-1]),
        cp.unique(seg[:, :, 0]),
        cp.unique(seg[:, :, -1]),
        cp.unique(seg[:, 0, :]),
        cp.unique(seg[:, -1, :]),
    ])
    edge_set = set(cp.asnumpy(cp.unique(edge_labels)))
    
    # Fast set membership check
    return {int(sid): int(sid in edge_set) for sid in cp.asnumpy(seg_ids)}

def make_row_pid(
    pid: int,
    perox_masks_flat,
    flat_int_cpu,
    coords_all,
    perox_to_cell,
    nuc_to_cell,
    perox_vols,
    cell_vols,
    nuc_vols,
    perox_counts,
    cell_edge,
    nuc_edge,
    C
) -> dict:
    idx = perox_masks_flat[pid]
    coords = coords_all[idx]
    centroid = coords.mean(axis=0)
    vals_cpu = flat_int_cpu[:, idx]
    sum_i = vals_cpu.sum(axis=1)
    mean_i = vals_cpu.mean(axis=1)
    med_i = np.median(vals_cpu, axis=1)
    std_i = vals_cpu.std(axis=1)

    cell_id = perox_to_cell.get(pid, 0)
    nuc_id = next((nid for nid, c in nuc_to_cell.items() if c == cell_id), 0)

    row = {
        'perox_id': pid,
        'cell_id': cell_id,
        'nuc_id': nuc_id,
        'perox_vol_um3': perox_vols.get(pid, 0),
        'cell_vol_um3': cell_vols.get(cell_id, 0),
        'nuc_vol_um3': nuc_vols.get(nuc_id, 0),
        'perox_count': perox_counts.get(cell_id, 0),
        'touches_edge_cell': cell_edge.get(cell_id, 0),
        'touches_edge_nucleus': nuc_edge.get(nuc_id, 0),
        'centroid_z_um': float(centroid[0]),
        'centroid_y_um': float(centroid[1]),
        'centroid_x_um': float(centroid[2]),
    }
    for ch in range(C):
        row[f'int_sum_ch{ch}'] = float(sum_i[ch])
        row[f'int_mean_ch{ch}'] = float(mean_i[ch])
        row[f'int_med_ch{ch}'] = float(med_i[ch])
        row[f'int_std_ch{ch}'] = float(std_i[ch])
    cv = row['cell_vol_um3']
    row['nuc_to_cell_ratio'] = row['nuc_vol_um3'] / cv if cv > 0 else 0
    row['perox_density'] = row['perox_count'] / cv if cv > 0 else 0
    row['has_nucleus'] = int(nuc_id > 0)
    return row

def compute_features(
    cell_seg: np.ndarray,
    nuc_seg: np.ndarray,
    perox_seg: np.ndarray,
    intensities: np.ndarray,
    voxel_size: Tuple[float, float, float],
    max_workers: int = 10,
    profile: bool = False,
) -> pd.DataFrame:
    t_start = time.perf_counter() if profile else None

    C, Zi, Yi, Xi = intensities.shape
    flat_cell_cpu = cell_seg.ravel()
    flat_nuc_cpu = nuc_seg.ravel()
    flat_perox_cpu = perox_seg.ravel()
    flat_int_cpu = intensities.reshape(C, -1)
    coords_all = np.indices((Zi, Yi, Xi)).reshape(3, -1).T * np.array(voxel_size)

    cell_ids = np.unique(flat_cell_cpu)[1:].tolist()
    nuc_ids = np.unique(flat_nuc_cpu)[1:].tolist()
    perox_counts_array = np.bincount(flat_perox_cpu)
    perox_ids = [pid for pid in np.unique(flat_perox_cpu)[1:] if perox_counts_array[pid] >= 8]

    perox_masks_flat = {pid: np.where(flat_perox_cpu == pid)[0] for pid in perox_ids}
    cell_counts = np.bincount(flat_cell_cpu)
    nuc_counts = np.bincount(flat_nuc_cpu)
    voxel_vol = np.prod(voxel_size)
    cell_vols = {cid: int(cell_counts[cid] * voxel_vol) for cid in cell_ids}
    nuc_vols = {nid: int(nuc_counts[nid] * voxel_vol) for nid in nuc_ids}
    perox_vols = {pid: int(perox_counts_array[pid] * voxel_vol) for pid in perox_ids}
    cell_edge = edge_map(cell_ids, cell_seg)
    nuc_edge = edge_map(nuc_ids, nuc_seg)

    perox_to_cell = {
        pid: int(np.bincount(flat_cell_cpu[perox_masks_flat[pid]]).argmax())
        for pid in perox_ids
    }
    nuc_masks_flat = {nid: np.where(flat_nuc_cpu == nid)[0] for nid in nuc_ids}
    nuc_to_cell = {
        nid: int(np.bincount(flat_cell_cpu[nuc_masks_flat[nid]]).argmax())
        for nid in nuc_ids if nuc_masks_flat[nid].size > 0
    }
    perox_counts = {
        parent: sum(1 for p, c in perox_to_cell.items() if c == parent)
        for parent in cell_ids
    }

    rows = []
    per_element_gb = 0.00002  # Rough estimate per object
    batch_size = estimate_batch_size(len(perox_ids), per_element_gb)

    if perox_ids:
        for i in range(0, len(perox_ids), batch_size):
            batch = perox_ids[i:i + batch_size]
            logging.info(f"[FEATURES] Processing batch {i // batch_size + 1} ({len(batch)} objects)...")
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = [
                    exe.submit(
                        make_row_pid,
                        pid,
                        perox_masks_flat,
                        flat_int_cpu,
                        coords_all,
                        perox_to_cell,
                        nuc_to_cell,
                        perox_vols,
                        cell_vols,
                        nuc_vols,
                        perox_counts,
                        cell_edge,
                        nuc_edge,
                        C
                    ) for pid in batch
                ]
                for fut in as_completed(futures):
                    rows.append(fut.result())
    else:
        # fallback: cell-level stats only
        pass  # (Same as before if needed)

    df = pd.DataFrame(rows).sort_values(['cell_id', 'perox_id']).reset_index(drop=True)
    if profile:
        elapsed = time.perf_counter() - t_start
        print(f"[PROFILE] compute_features completed in {elapsed:.2f} s with {len(rows)} rows")
    return df