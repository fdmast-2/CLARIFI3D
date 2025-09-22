import logging
import psutil
from pathlib import Path
from typing import Union

import cupy as cp

# Optional: include torch metrics if available
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def setup_logging(log_path: Union[str, Path], level: int = logging.INFO) -> None:
    """
    Configure root logger to write to both file and stdout with timestamped messages.

    Parameters
    ----------
    log_path : Union[str, Path]
        File path for the log file.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logger = logging.getLogger()
    # Remove any default handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(level)

    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def profile_memory(stage: str = "") -> None:
    """
    Log current CPU and GPU memory usage.

    Parameters
    ----------
    stage : str
        Label for this profiling point.
    """
    # CPU memory (RSS)
    rss = psutil.Process().memory_info().rss / 1e9

    # CuPy memory pool stats
    try:
        pool = cp.get_default_memory_pool()
        gpu_used = pool.used_bytes() / 1e9
        gpu_reserved = pool.total_bytes() / 1e9
    except Exception:
        gpu_used = gpu_reserved = 0.0

    msg = f"[MEM] {stage}: CPU RSS={rss:.2f}GB | GPU used={gpu_used:.2f}GB reserved_pool={gpu_reserved:.2f}GB"
    logging.info(msg)

    # Optionally log PyTorch GPU usage
    if _HAS_TORCH:
        try:
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logging.info(f"[MEM-Torch] {stage}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB")
        except Exception:
            pass
