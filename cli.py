#!/usr/bin/env python3
# clarifi3d/cli.py
"""
CLI for Clarifi3D segmentation pipeline optimized for HPC environments.
"""

# ------------------------
# Standard Library Imports
# ------------------------
import argparse
import gc
import logging
from logging import FileHandler, Formatter
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import time
from pathlib import Path
import sys
import traceback
from typing import Tuple

# ------------------------
# Third-Party Imports
# ------------------------
import cupy as cp
import numpy as np
import torch

print("Torch sees", torch.cuda.device_count(), "GPUs")
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
from cupyx.scipy.ndimage import median_filter as gpu_median_filter

# ------------------------
# Clarifi3D Package Imports
# ------------------------
from clarifi3d.io import (
    CZIReader,
    save_segmentation,
    save_inference_map,
    save_feature_table,
)
from clarifi3d.filters import filter_gaussian
from clarifi3d.models import (
    MultiModelEngine,
    MODEL_ALIAS_REGISTRY,
)
from clarifi3d.normalization import normalize_volume
from clarifi3d.seg import (
    segment_peroxisomes,
    compute_features,
    detect_and_filter_seed_points_gpu,
    infer_nucleus_mask,
    infer_cell_mask,
    execute_segmentation,
)
from clarifi3d.utils import (
    profile_memory,
    setup_logging,
    util_profile_gpu_memory,
    evaluate_seed_coverage,
    util_clear_gpu_memory,
    add_file_logger,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clarifi3D segmentation pipeline")
    parser.add_argument("--raw_dir", type=Path, help="Directory with input .czi files")
    parser.add_argument("--file", type=Path, help="Process a single .czi file")
    parser.add_argument("--file_list", type=Path, help="Optional list of .czi files to process")
    parser.add_argument("--processed_dir", type=Path, required=True, help="Directory to save outputs")
    parser.add_argument("--model_dir", type=Path, required=True, help="Directory containing UNet3D .pth model files")

    parser.add_argument("--peroxisome_channel", type=int, default=0, help="Channel index for peroxisome marker")
    parser.add_argument("--nucleus_channel", type=int, default=1, help="Channel index for nuclear marker")
    parser.add_argument("--cytoplasm_channel", type=int, default=2, help="Channel index for cytoplasm marker")

    parser.add_argument("--voxel_size", type=float, nargs=3, default=[0.2, 0.05179016, 0.05179016], help="Voxel size in microns (Z Y X)")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of CPU workers for feature assembly")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    parser.add_argument("--skip_perox", action="store_true", help="Skip peroxisome segmentation")
    parser.add_argument("--skip_cells", action="store_true", help="Skip cell & nucleus segmentation")
    parser.add_argument("--skip_features", action="store_true", help="Skip feature extraction")


    # --- Peroxisome segmentation parameters ---
    parser.add_argument("--perox_gauss_sigma_um", type=float, default=0.15537048,
                        help="Gaussian sigma for peroxisome smoothing (µm)")
    parser.add_argument("--perox_laplacian_radii_um", type=float, nargs=3, default=[0.5179016, 0.2589508, 0.2589508],
                        help="Morphological Laplacian radii (µm) (Z Y X)")
    parser.add_argument("--perox_closing_radii_um", type=float, nargs=3, default=[0.5179016, 0.15537048, 0.15537048],
                        help="Closing radii (µm) for peroxisome mask (Z Y X)")
    parser.add_argument("--perox_min_size_um", type=float, default=0.20716064,
                        help="Minimum valid peroxisome size (µm³)")

    parser.add_argument("--write_probs", action="store_true", help="Save intermediate probability maps")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    return parser.parse_args()


def main():
    args = parse_args()

    def initialize_logging_and_device():
        args.processed_dir.mkdir(parents=True, exist_ok=True)
        level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f"Invalid log level: {args.log_level}")
        setup_logging(args.processed_dir / "pipeline.log", level=level)

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device detected by PyTorch.")

        # Always use the first visible GPU (as per CUDA_VISIBLE_DEVICES)
        device = torch.device("cuda:0")
        # Log the mapping for diagnostics
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        logging.info(f"[INIT] Using CUDA_VISIBLE_DEVICES={visible} | Model dir: {args.model_dir}")
        return device

    def build_engine(device):
        return MultiModelEngine(
            model_dir=args.model_dir,
            device=device,
            batch_size=args.batch_size,
            use_amp=torch.cuda.is_available(),
        )

    def get_file_list() -> list:
        if args.file_list:
            with open(args.file_list) as f:
                return [Path(line.strip()) for line in f if line.strip()]
        return [args.file] if args.file else sorted(args.raw_dir.glob("*.czi"))

    device = initialize_logging_and_device()
    engine = build_engine(device)
    czi_files = get_file_list()

    if not czi_files:
        logging.error("[ERROR] No .czi files found to process.")
        sys.exit(1)

    for path in czi_files:
        try:
            process_file(
                path=path,
                processed_dir=args.processed_dir,
                engine=engine,
                voxel_size=tuple(args.voxel_size),
                gauss_sigma_um=args.perox_gauss_sigma_um,
                laplacian_radii_um=tuple(args.perox_laplacian_radii_um),
                closing_radii_um=tuple(args.perox_closing_radii_um),
                min_size_um=args.perox_min_size_um,
                skip_cells=args.skip_cells,
                skip_perox=args.skip_perox,
                skip_features=args.skip_features,
                write_probs=args.write_probs,
                perox_ch=args.peroxisome_channel,
                nuc_ch=args.nucleus_channel,
                cyto_ch=args.cytoplasm_channel,
                num_workers=args.num_workers,
                overwrite=args.overwrite
            )
        except Exception:
            logging.error(f"[ERROR] Processing {path.name} failed:\n{traceback.format_exc()}")



def process_file(
    path: Path,
    processed_dir: Path,
    engine: MultiModelEngine,
    voxel_size: tuple,
    gauss_sigma_um: float,
    laplacian_radii_um: tuple,
    closing_radii_um: tuple,
    min_size_um: float,
    skip_cells: bool,
    skip_perox: bool,
    skip_features: bool,
    write_probs: bool,
    perox_ch: int,
    nuc_ch: int,
    cyto_ch: int,
    num_workers: int,
    overwrite: bool,
) -> None:
    base = path.stem
    file_handler = add_file_logger(processed_dir / f"{base}_pipeline.log")
    try:
        logging.info(f"[START] File: {path.name}")
        profile_memory("before_load")
        reader = CZIReader(path)
        gpu_arr = reader.read_gpu(normalize=False)
        profile_memory("after_load")

        # Smoothing only the nucleus channel for robust seed/nucleus detection
        # sigma_voxels = tuple(gauss_sigma_um / v for v in voxel_size)
        # gpu_arr[nuc_ch] = filter_gaussian(gpu_arr[nuc_ch], sigma=sigma_voxels)
        # Cell/cytoplasm (cyto_ch) left unsmoothed to retain edge/membrane fidelity

        t0 = time.perf_counter()

        def run_perox_segmentation(raw_img: cp.ndarray) -> cp.ndarray:
            logging.info("[PEROX] Starting peroxisome segmentation")
            profile_memory("before_perox")
            raw_img = raw_img.astype(cp.uint8) if raw_img.dtype != cp.uint8 else raw_img
            seg = segment_peroxisomes(
                raw=raw_img,
                voxel_size=voxel_size,
                gauss_sigma_um=gauss_sigma_um,
                laplacian_radii_um=laplacian_radii_um,
                closing_radii_um=closing_radii_um,
                min_size_um=min_size_um,
                profile=False,
                debug=logging.getLogger().isEnabledFor(logging.DEBUG),
            )
            profile_memory("after_perox")
            return seg

        perox_seg = None
        perox_path = processed_dir / f"{base}_perox.tiff"
        if not skip_perox and (overwrite or not perox_path.exists()):
            perox_seg_gpu = run_perox_segmentation(gpu_arr[perox_ch])
            save_segmentation(perox_seg_gpu, perox_path)
            perox_seg = perox_seg_gpu.get()
            del perox_seg_gpu
        util_clear_gpu_memory()

        cell_seg, nuc_seg = None, None
        cell_path = processed_dir / f"{base}_cell.tiff"
        nuc_path = processed_dir / f"{base}_nuc.tiff"

        if not skip_cells and (overwrite or not (cell_path.exists() and nuc_path.exists())):
            logging.info("[CELLS] Starting cell and nucleus segmentation pipeline")
            profile_memory("before_cells")

            # -- 1. Cell Inference (ML) --
            alias, ch = "predict_cells", cyto_ch
            fname = MODEL_ALIAS_REGISTRY[alias]
            if fname in engine.predictors:
                engine.predictors[fname][1]["input_channel"] = ch

            cpu_vol = cp.asnumpy(gpu_arr).astype(np.float32)
            util_clear_gpu_memory(); gc.collect()
            cell_p = engine.predict_shared(cpu_vol, keys=["predict_cells"])[0]
            cell_prob_map = cp.asarray(cell_p)
            del cpu_vol, cell_p
            util_clear_gpu_memory()

            if write_probs:
                save_inference_map(
                    cell_prob_map[None, ...],
                    processed_dir / f"{base}_cell_inference.tiff",
                    suffixes=("cell",)
                )
            profile_memory("after_inference")

            # -- 2. Classical Nucleus Segmentation & Seed Detection --
            logging.info("[FUSE] Segmenting nuclei and cells")
            t1 = time.time()

            nuc_mask, nuc_smooth = infer_nucleus_mask(
                nucleus_volume=gpu_arr[nuc_ch],  # DAPI or nucleus channel, not ML inference!
                voxel_spacing=voxel_size,
                base=base,
                output_directory=str(processed_dir)
            )
            seed_mask, seed_centroids, _, _ = detect_and_filter_seed_points_gpu(
                smoothed_nucleus=nuc_smooth,
                nucleus_mask=nuc_mask,
                voxel_spacing=voxel_size,
                logger=logging
            )
            t2 = time.time()

            # -- 3. Cell Mask from Raw Channel --
            cell_raw = gpu_arr[cyto_ch]
            cell_mask = infer_cell_mask(
                cell_volume=cell_raw,
                voxel_spacing=voxel_size,
                base=base,
                output_directory=str(processed_dir),
                smoothing_sigma_um=(0.1527, 0.7635, 0.7335),
                closing_radius_um=0.4,
            )
            util_clear_gpu_memory(cell_raw)

            # -- 4. Prepare cell cost map --
            masked_inference = cp.where(cell_mask, cell_prob_map, 0.0)
            cell_cost = masked_inference
            del cell_prob_map; gc.collect()

            logging.info(
                f"Seed detection: {t2 - t1:.2f}s. "
                f"Nucleus: {time.time() - t2:.2f}s. "
                f"Cell mask: {time.time() - t2:.2f}s."
            )
            evaluate_seed_coverage(seed_centroids, nuc_mask, cell_mask)

            # -- 5. Segmentation --
            nuc_seg, cell_seg = execute_segmentation(
                nucleus_mask=nuc_mask.astype(cp.bool_),
                cell_mask=cell_mask.astype(cp.bool_),
                seed_centroids=seed_centroids,
                volume_shape=nuc_mask.shape,
                nucleus_intensity=nuc_smooth,  # classical, not ML
                cell_inference_map=cell_cost,  # masked ML map (not inverted)
                border_crop_voxels=2,
                logger=logging,
                voxel_size=voxel_size
            )
            util_profile_gpu_memory("Post-Segmentation")
            logging.info(f"Segmentation completed in {time.time() - t2:.2f}s.")
            save_segmentation(cell_seg, cell_path)
            save_segmentation(nuc_seg, nuc_path)
            profile_memory("after_cells")
            util_clear_gpu_memory(seed_mask, seed_centroids, nuc_mask, nuc_smooth, cell_mask)
            del seed_mask, seed_centroids, nuc_mask, nuc_smooth, cell_mask; gc.collect()

        cpu_arr = cp.asnumpy(gpu_arr).astype(np.float32)
        del gpu_arr
        util_clear_gpu_memory(); gc.collect()

        # --- FEATURE EXTRACTION ---
        feat_path = processed_dir / f"{base}_features.csv"
        if not skip_features and perox_seg is not None and cell_seg is not None and (overwrite or not feat_path.exists()):
            try:
                logging.info("[FEATURES] Computing morphological features")
                profile_memory("before_features")
                to_numpy = lambda arr: arr.get() if isinstance(arr, cp.ndarray) else arr
                cell_seg = to_numpy(cell_seg)
                nuc_seg = to_numpy(nuc_seg)
                perox_seg = to_numpy(perox_seg)
                if np.max(cell_seg) == 0 or np.max(perox_seg) == 0:
                    logging.warning("[FEATURES] Skipping: empty masks")
                    return
                df = compute_features(cell_seg, nuc_seg, perox_seg, cpu_arr, voxel_size, max_workers=num_workers, profile=True)
                save_feature_table(df, feat_path)
            except KeyboardInterrupt:
                logging.warning("[FEATURES] Interrupted by user")
            except Exception as e:
                logging.exception(f"[FEATURES] Error: {e}")
            finally:
                profile_memory("after_features")
                util_clear_gpu_memory(); gc.collect()

        logging.info(f"[DONE] {base} in {time.perf_counter() - t0:.2f}s")

    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    main()