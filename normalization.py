#!/usr/bin/env python3
# clarifi3d/normalization.py
"""
clarifi3d.normalization: GPU/CPU normalization primitives and pipelines.
"""
import logging
from typing import Callable, Dict, List, Tuple, Union

import cupy as cp
from clarifi3d.filters import filter_gaussian

PipelineFn = Callable[[cp.ndarray], cp.ndarray]

# ----------------------------------------------------------------------------
# Core GPU Normalization Primitives
# ----------------------------------------------------------------------------
def gpu_percentile_clip_and_normalize(
    vol: cp.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
    clip_range: Tuple[float, float] = (None, None)
) -> cp.ndarray:
    if clip_range[0] is not None:
        vol = cp.maximum(vol, clip_range[0])
    if clip_range[1] is not None:
        vol = cp.minimum(vol, clip_range[1])

    p1, p99 = cp.percentile(vol, [p_low, p_high])
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"[Norm] Percentiles ({p_low}%, {p_high}%): {p1:.4f}, {p99:.4f}")

    if abs(p99 - p1) < 1e-6:
        logging.warning("Flat volume detected during percentile normalization.")
        return cp.zeros_like(vol, dtype=cp.float32)

    return cp.clip((vol - p1) / (p99 - p1 + 1e-8), 0.0, 1.0).astype(cp.float32)

def gpu_statistical_normalize(
    vol: cp.ndarray,
    n_std: float = 3.0
) -> cp.ndarray:
    mu = cp.mean(vol)
    sigma = cp.std(vol)
    lo = cp.maximum(mu - n_std * sigma, cp.min(vol))
    hi = cp.minimum(mu + n_std * sigma, cp.max(vol))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"[Norm] Mean ± {n_std}×STD: {lo:.4f}, {hi:.4f}")

    normed = cp.clip((vol - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return normed.astype(cp.float32)

def gpu_background_subtract(
    volume: cp.ndarray,
    radius: Union[int, Tuple[int, int, int]] = 50
) -> cp.ndarray:
    if isinstance(radius, int):
        sigma = (radius,) * 3
    else:
        sigma = radius
    blurred = filter_gaussian(volume.astype(cp.float32), sigma)
    return cp.clip(volume - blurred, 0, None)

# ----------------------------------------------------------------------------
# Named Normalization Pipelines
# ----------------------------------------------------------------------------
NORMALIZATION_PIPELINES: Dict[str, List[PipelineFn]] = {
    "default": [
        lambda x: gpu_percentile_clip_and_normalize(x, 2.0, 98.0)
    ],
    "nuclei": [
        lambda x: gpu_background_subtract(x, 50),
        lambda x: gpu_percentile_clip_and_normalize(x, 1.0, 99.0)
    ],
    "membrane": [
        lambda x: gpu_statistical_normalize(x, 2.5)
    ],
    "laminB1": [
        lambda x: gpu_background_subtract(x, 50),
        lambda x: gpu_statistical_normalize(x, 5.0)
    ],
    "caax": [
        lambda x: gpu_statistical_normalize(x, 2.0)
    ],
    "h2b": [
        lambda x: gpu_background_subtract(x, 50),
        lambda x: gpu_statistical_normalize(x, 1.5)
    ],
}

# ----------------------------------------------------------------------------
# Dispatcher Functions
# ----------------------------------------------------------------------------
def normalize_channel(
    vol: cp.ndarray,
    strategy: Union[str, PipelineFn] = "default"
) -> cp.ndarray:
    if callable(strategy):
        pipeline = [strategy]
    else:
        pipeline = NORMALIZATION_PIPELINES.get(strategy)
        if pipeline is None:
            raise ValueError(f"Unknown normalization strategy: {strategy}")
    out = vol
    for fn in pipeline:
        out = fn(out)
    return out

def normalize_volume(
    vol: cp.ndarray,
    mode_map: Dict[int, Union[str, PipelineFn]]
) -> cp.ndarray:
    if vol.ndim != 4:
        raise ValueError(f"Expected 4D CuPy array (C, Z, Y, X), got {vol.shape}")
    C = vol.shape[0]
    out = cp.empty_like(vol, dtype=cp.float32)
    for c in range(C):
        strat = mode_map.get(c, "default")
        out[c] = normalize_channel(vol[c], strat)
    return out