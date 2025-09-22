#!/usr/bin/env python3
# clarifi3d/models.py
"""
MODELS for Clarifi3D segmentation pipeline optimized for HPC environments.
"""
# Standard Library
import logging
import contextlib
import gc
import time
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party Libraries
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import cupy as cp
from aicsmlsegment.Net3D.unet_xy_enlarge import UNet3D

# Internal Modules
from clarifi3d.utils import blend_patches, profile_memory
from clarifi3d.normalization import normalize_channel


# ---------------------------------------------------------------------
# FULL MODEL REGISTRY
# Defines pretrained model files and their associated metadata
# ---------------------------------------------------------------------
FULL_MODEL_REGISTRY = {
    "DNA_mask_production.pth": {
        "model_type": "unet_xy_zoom",
        "input_channel": 1,
        "output_ch": [1],             # Foreground class (nuclei)
        "apply_softmax": True,
        "combine_outputs": "first",   # Use first (and only) output class
        "norm": "nuclei"
    },
    "DNA_seed_production.pth": {
        "model_type": "unet_xy_zoom",
        "input_channel": 1,
        "output_ch": [1],
        "apply_softmax": True,
        "combine_outputs": "first",
        "norm": "nuclei"
    },
    "CellMask_edge_production.pth": {
        "model_type": "unet_xy_zoom",
        "input_channel": 0,
        "output_ch": [1],             # Foreground membrane edges
        "apply_softmax": True,
        "combine_outputs": "first",
        "norm": "membrane"
    },
    # "CAAX_production.pth": {
    #     "model_type": "unet_xy_zoom",
    #     "input_channel": 0,
    #     "output_ch": [1],
    #     "apply_softmax": True,
    #     "combine_outputs": "first",
    #     "norm": "caax"
    # },
    # "H2B_coarse.pth": {
    #     "model_type": "unet_xy_zoom",
    #     "input_channel": 0,
    #     "output_ch": [1],
    #     "apply_softmax": True,
    #     "combine_outputs": "first",
    #     "norm": "h2b"
    # },
}

# ---------------------------------------------------------------------
# MODEL TYPE DEFINITIONS
# Defines the shared structural characteristics of each model type
# ---------------------------------------------------------------------
MODEL_TYPE_DEFS = {
    "unet_xy_zoom": {
        "size_in": (52, 420, 420),
        "size_out": (20, 152, 152),
        "nclass": (2, 2, 2),       # output has 2 classes per level: [background, foreground]
    },
}

# ---------------------------------------------------------------------
# MODEL ALIAS REGISTRY
# Provides task-oriented alias names for models
# ---------------------------------------------------------------------
MODEL_ALIAS_REGISTRY = {
    "predict_nuclei": "DNA_mask_production.pth",
    "predict_seed": "DNA_seed_production.pth",
    "predict_cells": "CellMask_edge_production.pth",
    # "predict_caax": "CAAX_production.pth",
    # "predict_h2b": "H2B_coarse.pth",
}

_STATE_CACHE: Dict[str, Any] = {}

# ------------------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------------------

def load_model(
    model_file: Union[str, Path],
    in_channel: int = 1,
    n_classes: Union[int, Tuple[int, ...]] = (2,),
    down_ratio: int = 3,
    batchnorm_flag: bool = True,
) -> UNet3D:
    path = Path(model_file)
    key = str(path)

    if isinstance(n_classes, int):
        n_classes = (n_classes,) * 3
    elif len(n_classes) < 3:
        n_classes = (n_classes + (n_classes[-1],) * (3 - len(n_classes)))

    if key not in _STATE_CACHE:
        ckpt = torch.load(path, map_location="cpu")
        print(f"[DEBUG] type(ckpt): {type(ckpt)}")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            model = UNet3D(in_channel, n_classes, down_ratio, batchnorm_flag=batchnorm_flag)
            model.load_state_dict(state_dict)
        elif isinstance(ckpt, dict):
            model = UNet3D(in_channel, n_classes, down_ratio, batchnorm_flag=batchnorm_flag)
            model.load_state_dict(ckpt)
        elif isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            raise TypeError(f"Unknown checkpoint type for {path}: {type(ckpt)}")

        _STATE_CACHE[key] = model

    model = _STATE_CACHE[key]
    model.eval()
    print(f"[DEBUG] Loaded model: {type(model)}")

    return model

# ------------------------------------------------------------------------------
# Predictor with tiled blending
# ------------------------------------------------------------------------------

class PredictorWithStitching:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        size_in: Tuple[int, int, int],
        size_out: Tuple[int, int, int],
        batch_size: int = 8,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.size_in = size_in
        self.size_out = size_out
        self.batch_size = batch_size
        self.weight_cache = {}
    def predict(
        self,
        volume: torch.Tensor,
        patch_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        out_channels: int,
        blend: bool = True,
        log_prefix: Optional[str] = None,
    ) -> torch.Tensor:
        assert volume.dim() == 4, f"Expected input of shape (1,Z,Y,X), got {volume.shape}"
        _, Z, Y, X = volume.shape
        pZ, pY, pX = patch_size
        sZ, sY, sX = stride
        cz, cy, cx = self.size_out

        pad_z = self._calc_symmetric_pad(Z, pZ, cz, sZ)
        pad_y = self._calc_symmetric_pad(Y, pY, cy, sY)
        pad_x = self._calc_symmetric_pad(X, pX, cx, sX)
        volume_padded = F.pad(volume, pad_x + pad_y + pad_z, mode="reflect")
        _, Zp, Yp, Xp = volume_padded.shape

        output_full = torch.zeros((out_channels, Zp, Yp, Xp), dtype=torch.float32, device=self.device)
        weight_full = torch.zeros_like(output_full)

        if self.size_out not in self.weight_cache:
            self.weight_cache[self.size_out] = self._make_blend_weight(self.size_out).to(self.device)
        patch_weight = self.weight_cache[self.size_out]

        logging.info(f"[Predictor] Model input patch size: {patch_size}, output patch size: {self.size_out}")
        logging.info(f"[Predictor] Output cropping per patch: z={(patch_size[0] - self.size_out[0]) // 2}, "
                    f"y={(patch_size[1] - self.size_out[1]) // 2}, x={(patch_size[2] - self.size_out[2]) // 2}")
        logging.info(f"[Predictor] Using stride: {stride}")
        logging.info(f"[Predictor] Padded input shape: {volume_padded.shape}")

        def safe_starts(size, patch, stride):
            if size <= patch:
                return [0]
            starts = list(range(0, size - patch + 1, stride))
            last = size - patch
            if starts[-1] < last:
                starts.append(last)
            return starts

        z_starts = safe_starts(Zp, pZ, sZ)
        y_starts = safe_starts(Yp, pY, sY)
        x_starts = safe_starts(Xp, pX, sX)
        coords = [(z, y, x) for z in z_starts for y in y_starts for x in x_starts]

        logging.info(f"[Predictor] Tiling with patch size {patch_size} and stride {stride}")
        logging.info(f"[Predictor] Total patches: {len(coords)}")

        t0 = time.perf_counter()

        with torch.no_grad():
            for i in range(0, len(coords), self.batch_size):
                batch_coords = coords[i:i + self.batch_size]
                patches = torch.stack([
                    volume_padded[:, z:z+pZ, y:y+pY, x:x+pX]
                    for z, y, x in batch_coords
                ])  # (B, 1, Z, Y, X)

                with autocast(device_type='cuda'):
                    raw_preds = self.model(patches)

                pred_main = raw_preds[-1] if isinstance(raw_preds, list) else raw_preds
                preds = pred_main

                expected_voxels = np.prod(self.size_out)
                B = len(batch_coords)

                if preds.ndim == 2:
                    if preds.shape[0] == B * expected_voxels and preds.shape[1] == out_channels:
                        preds = preds.view(B, expected_voxels, out_channels).permute(0, 2, 1).contiguous().view(B, out_channels, *self.size_out)
                    elif preds.shape[0] == expected_voxels and preds.shape[1] == out_channels:
                        preds = preds.permute(1, 0).contiguous().view(1, out_channels, *self.size_out)
                    else:
                        raise RuntimeError(
                            f"Flat output shape mismatch: got {preds.shape}, "
                            f"expected {B} × {expected_voxels} voxels × {out_channels} channels"
                        )

                for j, (z, y, x) in enumerate(batch_coords):
                    pred = preds[j]
                    oz, oy, ox = z + (pZ - cz) // 2, y + (pY - cy) // 2, x + (pX - cx) // 2
                    output_full[:, oz:oz+cz, oy:oy+cy, ox:ox+cx] += pred * patch_weight
                    weight_full[:, oz:oz+cz, oy:oy+cy, ox:ox+cx] += patch_weight

                # Proactive cleanup
                del patches, preds, raw_preds, pred_main
                torch.cuda.empty_cache()

                if (i + self.batch_size) % 50 == 0:
                    logging.debug(f"[Predictor] {i + self.batch_size}/{len(coords)} patches processed")

        t1 = time.perf_counter()
        logging.info(f"[Predictor] Inference done in {t1 - t0:.2f}s")

        # --------------------------------------------
        # Normalize in-place in memory-aware chunks
        # --------------------------------------------
        valid_crop = (
            slice(pad_z[0], pad_z[0] + Z),
            slice(pad_y[0], pad_y[0] + Y),
            slice(pad_x[0], pad_x[0] + X),
        )

        # Extract cropped views directly
        output_cropped = output_full[:, valid_crop[0], valid_crop[1], valid_crop[2]]
        weight_cropped = weight_full[:, valid_crop[0], valid_crop[1], valid_crop[2]]

        # Normalize in-place to reduce memory overhead
        for c in range(out_channels):
            for z in range(output_cropped.shape[1]):
                output_cropped[c, z] /= torch.clamp_min(weight_cropped[c, z], 1e-6)

        # Final result: contiguous, post-normalization
        result = output_cropped.contiguous()

        # Cleanup aggressively before returning
        del output_full, weight_full, volume_padded, patch_weight, output_cropped, weight_cropped
        gc.collect()
        torch.cuda.empty_cache()

        return result


    @staticmethod
    def _calc_symmetric_pad(image_size: int, size_in: int, size_out: int, stride: int) -> Tuple[int, int]:
        """
        Compute symmetric padding for tiling, ensuring:
        - At least one full patch fits
        - Enough margin to allow proper cropping (i.e., (size_in - size_out) // 2 per side)
        - Tiling with stride does not cause gaps or overflows
        """

        # Minimum margin needed to recover the valid center crop from the model
        required_margin = (size_in - size_out) // 2

        # Minimal padded size that ensures we can apply a full-size patch
        minimal_padded = image_size + 2 * required_margin

        # Round up to make it stride-aligned after the margin
        remainder = (minimal_padded - size_in) % stride
        extra = (stride - remainder) if remainder != 0 else 0

        total_pad = 2 * required_margin + extra

        pad_before = total_pad // 2
        pad_after = total_pad - pad_before

        return (pad_before, pad_after)

    @staticmethod
    def _make_blend_weight(shape: Tuple[int, int, int]) -> torch.Tensor:
        def cosine_weights(length):
            x = np.linspace(-np.pi, np.pi, length)
            return 0.5 * (1 + np.cos(x))
        wz = cosine_weights(shape[0])[:, None, None]
        wy = cosine_weights(shape[1])[None, :, None]
        wx = cosine_weights(shape[2])[None, None, :]
        weights = wz * wy * wx
        weights /= weights.sum()  # Normalize once
        return torch.from_numpy(weights).float()


# ------------------------------------------------------------------------------
# Multi-model engine
# ------------------------------------------------------------------------------

class MultiModelEngine:
    def __init__(
        self,
        model_dir: Union[str, Path],
        device: torch.device,
        batch_size: int,
        use_amp: bool = False,
        strict: bool = False,  # Optional: raise error if any expected model is missing
    ) -> None:
        self.device = device
        self.use_amp = use_amp
        self.batch_size = batch_size or 4
        self.predictors: Dict[str, Tuple[PredictorWithStitching, Dict[str, Any]]] = {}

        model_dir = Path(model_dir)
        available_models = {p.name for p in model_dir.glob("*.pth")}
        expected_models = set(FULL_MODEL_REGISTRY.keys())

        missing = expected_models - available_models
        if missing:
            msg = f"[Engine Init] Missing expected models: {sorted(missing)}"
            if strict:
                raise FileNotFoundError(msg)
            else:
                logging.warning(msg)

        # Main model initialization loop — always runs
        for fname in sorted(expected_models & available_models):
            meta = FULL_MODEL_REGISTRY[fname]

            assert isinstance(meta["output_ch"], list), \
                f"[Engine Init] 'output_ch' must be a list for model '{fname}', got {type(meta['output_ch'])}"

            model_path = model_dir / fname
            model_type = meta["model_type"]
            model_def = MODEL_TYPE_DEFS[model_type]

            in_ch = meta["input_channel"]

            if isinstance(in_ch, int):
                n_in = 1
            elif isinstance(in_ch, (list, tuple)):
                n_in = len(in_ch)
            else:
                raise ValueError(f"Invalid input_channel definition for model {fname}: {in_ch}")

            model = load_model(
                model_path,
                in_channel=n_in,
                n_classes=model_def["nclass"]
            )
            print(f"[DEBUG] fname={fname}, model type: {type(model)}, is nn.Module: {isinstance(model, torch.nn.Module)}")
            predictor = PredictorWithStitching(
                model=model,
                device=device,
                size_in=tuple(model_def["size_in"]),
                size_out=tuple(model_def["size_out"]),
                batch_size=self.batch_size,
            )
            self.predictors[fname] = (predictor, meta)

            if not self.predictors:
                raise RuntimeError("[Engine Init] No valid models were initialized.")
            else:
                logging.info(f"[Engine Init] Initialized {len(self.predictors)} models.")

    def predict_shared(
        self,
        image: np.ndarray,
        keys: Optional[List[str]] = None,
    ) -> Union[Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        results = {}

        for alias, fname in MODEL_ALIAS_REGISTRY.items():
            if keys is not None and alias not in keys:
                continue
            if fname not in self.predictors:
                raise ValueError(f"Model {fname} ({alias}) not initialized in engine")

            predictor, meta = self.predictors[fname]
            ch = meta["input_channel"]
            ch = meta["input_channel"]

            # Validate and extract channel(s)
            if isinstance(ch, int):
                if not (0 <= ch < image.shape[0]):
                    raise IndexError(f"[predict_shared] input_channel {ch} out of bounds for image with {image.shape[0]} channels")
                input_vol = image[ch][None, ...]  # shape: (1, Z, Y, X)
                input_cp = cp.asarray(input_vol[0])
                strategy = meta.get("norm", "default")
                norm_cp = normalize_channel(input_cp, strategy)
                norm_np = cp.asnumpy(norm_cp)[None, ...]
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"[predict_shared] Extracted channel {ch} with norm '{strategy}'")

            elif isinstance(ch, (list, tuple)):
                vols = []
                for c in ch:
                    if not (0 <= c < image.shape[0]):
                        raise IndexError(f"[predict_shared] input_channel {c} out of bounds for image with {image.shape[0]} channels")
                    cp_vol = cp.asarray(image[c])
                    strat = meta.get("norm", "default")
                    vols.append(cp.asnumpy(normalize_channel(cp_vol, strat)))
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"[predict_shared] Extracted channel {c} with norm '{strat}'")
                norm_np = np.stack(vols, axis=0)
            else:
                raise ValueError(f"Invalid input_channel type for {fname}: {ch} (expected int or list/tuple)")
            # ------------------------------
            # Step 0: Validate input image shape
            # ------------------------------
            if image.ndim != 4:
                raise ValueError(
                    f"[predict_shared] Expected input image of shape (C, Z, Y, X), got {image.shape}"
                )

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"[predict_shared] Input image shape: {image.shape}")
            # ------------------------------
            # Step 1: Extract and Normalize
            # ------------------------------
            if isinstance(ch, int):
                input_vol = image[ch][None, ...]  # shape (1, Z, Y, X)
                input_cp = cp.asarray(input_vol[0])  # (Z, Y, X)
                strategy = meta.get("norm", "default")
                norm_cp = normalize_channel(input_cp, strategy)
                norm_np = cp.asnumpy(norm_cp)[None, ...]
            elif isinstance(ch, (list, tuple)):
                vols = []
                for c in ch:
                    cp_vol = cp.asarray(image[c])
                    strat = meta.get("norm", "default")
                    vols.append(cp.asnumpy(normalize_channel(cp_vol, strat)))
                norm_np = np.stack(vols, axis=0)  # (C, Z, Y, X)
            else:
                raise ValueError(f"Invalid input_channel for {fname}: {ch}")

            # ------------------------------
            # Step 2: Run inference
            # ------------------------------
            model_type = meta["model_type"]
            model_def = MODEL_TYPE_DEFS[model_type]

            nclass_main = model_def["nclass"][-1]  # main branch class count
            logits = predictor.predict(
                torch.from_numpy(norm_np).to(predictor.device),
                patch_size=predictor.size_in,
                stride=tuple(s // 2 for s in predictor.size_out),
                out_channels=nclass_main,
            ).cpu().numpy()

            # ------------------------------
            # Step 3: Optional softmax
            # ------------------------------
            apply_softmax = meta.get("apply_softmax", True)
            if apply_softmax:
                logits = np.exp(logits)
                logits /= np.sum(logits, axis=0, keepdims=True)

            # ------------------------------
            # Step 4: Extract desired channels
            # ------------------------------
            output_ch = meta.get("output_ch", [1])
            combine_strategy = meta.get("combine_outputs", "first")

            if combine_strategy == "sum":
                out = np.sum([logits[i] for i in output_ch], axis=0)
            elif combine_strategy == "max":
                out = np.max([logits[i] for i in output_ch], axis=0)
            elif combine_strategy == "first":
                out = logits[output_ch[0]]
            elif combine_strategy == "none":
                out = {f"class_{i}": logits[i] for i in output_ch}
            else:
                raise ValueError(f"Unknown combine_outputs strategy: {combine_strategy}")

            results[alias] = out

        return results if keys is None else tuple(results[k] for k in keys)


    def get_normalization_map(self, max_channels: int) -> Dict[int, str]:
        """
        Construct a per-channel normalization strategy map based on model input definitions.
        Prioritizes first-encountered assignments to avoid overwriting shared channel logic.
        """
        norm_map: Dict[int, str] = {}

        for _, meta in self.predictors.values():
            ch = meta["input_channel"]
            strategy = meta.get("norm", "default")

            if isinstance(ch, int):
                if ch not in norm_map:
                    norm_map[ch] = strategy
            elif isinstance(ch, (list, tuple)):
                for c in ch:
                    if c not in norm_map:
                        norm_map[c] = strategy

        # Ensure all channels up to max_channels are assigned something
        return {c: norm_map.get(c, "default") for c in range(max_channels)}
    
    def get_available_aliases(self) -> List[str]:
        return [
            alias for alias, fname in MODEL_ALIAS_REGISTRY.items()
            if fname in self.predictors
        ]