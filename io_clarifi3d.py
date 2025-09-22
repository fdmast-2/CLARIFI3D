#!/usr/bin/env python3
# clarifi3d/io_clarifi3d.py
"""
IO_CLARIFI3D: Efficient CZI reading and I/O utilities optimized for HPC.
- CZIReader: lazy, shape-consistent Dask reader with NumPy/GPU export
- save_segmentation: writes uint16 TIFFs from CPU/GPU arrays
- save_inference_map: writes float32 TIFFs for model outputs
- save_feature_table: CSV writer for pandas DataFrame
"""
from pathlib import Path
from typing import Union, Dict, Optional, Tuple, Any

import numpy as np
import cupy as cp
import tifffile
import dask.array as da
from aicsimageio import AICSImage

from clarifi3d.normalization import normalize_volume


class CZIReader:
    """
    Loads a multi-channel CZI via Dask, exports full-volume NumPy or GPU arrays.
    """
    def __init__(
        self,
        path: Union[str, Path],
        max_channels: int = 3,
        norm_modes: Optional[Dict[int, str]] = None,
    ):
        self.img = AICSImage(str(path))
        # Dask array with dims (T, C, Z, Y, X) or fewer
        darr = self.img.dask_data
        # collapse time/scene dims if present
        while darr.ndim > 4:
            darr = darr[0]
        # ensure shape (C, Z, Y, X)
        if darr.ndim == 3:
            darr = darr[None, ...]
        self.array: da.Array = darr
        self.n_channels = min(max_channels, darr.shape[0])
        self.norm_modes = norm_modes or {}

    def read_dask(self) -> da.Array:
        """Return Dask array of shape (C, Z, Y, X)."""
        return self.array[: self.n_channels]

    def read_numpy(
        self,
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute and return a NumPy array (float32) of shape (C, Z, Y, X)."""
        arr = self.read_dask().compute()
        arr = arr.astype(np.float32)
        if normalize:
            arr = normalize_volume(arr, self.norm_modes)
        return arr

    def read_gpu(
        self,
        normalize: bool = True,
    ) -> cp.ndarray:
        """Load to GPU and return a CuPy array (float32) of shape (C, Z, Y, X)."""
        np_vol = self.read_numpy(normalize=normalize)
        return cp.asarray(np_vol)


def save_segmentation(
    volume: Union[cp.ndarray, np.ndarray],
    path: Path,
    description: Optional[str] = None,
) -> None:
    """
    Save a 3D segmentation volume as uint16 TIFF.
    Accepts CuPy or NumPy arrays.
    """
    if isinstance(volume, cp.ndarray):
        vol = volume.get()
    else:
        vol = volume
    tifffile.imwrite(
        str(path),
        vol.astype(np.uint16),
        description=description or "",
        bigtiff=True,
    )


def save_inference_map(
    array: Union[cp.ndarray, np.ndarray],
    output_path: Path,
    suffixes: Optional[Tuple[str, ...]] = None,
) -> None:
    """
    Save single- or multi-channel inference maps as float32 TIFFs.
    For 4D input (C, Z, Y, X), writes separate files with suffixes.
    """
    if isinstance(array, cp.ndarray):
        arr = array.get()
    else:
        arr = array
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        tifffile.imwrite(str(output_path), arr, bigtiff=True)
    elif arr.ndim == 4:
        if not suffixes or len(suffixes) != arr.shape[0]:
            raise ValueError(
                f"Need {arr.shape[0]} suffixes, got {len(suffixes) if suffixes else 0}"
            )
        for i, suffix in enumerate(suffixes):
            out = output_path.with_name(f"{output_path.stem}_{suffix}.tiff")
            tifffile.imwrite(str(out), arr[i], bigtiff=True)
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")


def save_feature_table(
    df: Any,  # pandas.DataFrame
    csv_path: Path,
) -> None:
    """
    Save DataFrame to CSV.
    """
    df.to_csv(csv_path, index=False)
