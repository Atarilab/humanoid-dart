"""Utilities for splitting work across multiple CUDA devices.

The pipeline stays single-process; each GPU-bound stage splits its work
into per-device chunks, runs them in parallel via threads, and merges
the results back.

Usage::

    devices = get_available_devices(num_gpus=4)   # ["cuda:0", ..., "cuda:3"]
    chunks  = split_list(items, len(devices))     # list[list[T]]
    ...                                           # run each chunk on its device
    merged  = merge_array_dicts(per_device_results)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def get_available_devices(num_gpus: int = 1) -> list[str]:
    """Return the exact CUDA device list requested by the pipeline.

    Falls back to ``["cpu"]`` only for single-device execution when CUDA is
    unavailable. Requests that exceed the visible CUDA devices fail fast so the
    launcher can be corrected instead of silently running on fewer GPUs.
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")

    if not torch.cuda.is_available():
        if num_gpus > 1:
            raise ValueError(
                f"Requested {num_gpus} GPUs, but CUDA is not available in this process"
            )
        return ["cpu"]

    available = torch.cuda.device_count()
    if num_gpus > available:
        raise ValueError(
            f"Requested {num_gpus} GPUs, but only {available} visible CUDA devices are available"
        )

    return [f"cuda:{i}" for i in range(num_gpus)]


def split_list(items: list, n: int) -> list[list]:
    """Split *items* into *n* roughly-equal contiguous chunks."""
    if n <= 1:
        return [items]
    k, r = divmod(len(items), n)
    chunks: list[list] = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < r else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def split_array(arr: np.ndarray, n: int) -> list[np.ndarray]:
    """Split a numpy array along axis 0 into *n* roughly-equal chunks."""
    if n <= 1:
        return [arr]
    indices = np.array_split(np.arange(len(arr)), n)
    return [arr[idx] for idx in indices]


def merge_array_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate per-device result dicts along the batch (axis-0) dimension.

    Non-array values are taken from the first dict.
    """
    if len(dicts) == 1:
        return dicts[0]
    merged: dict[str, Any] = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts if key in d]
        if isinstance(vals[0], np.ndarray) and vals[0].ndim >= 1:
            merged[key] = np.concatenate(vals, axis=0)
        else:
            merged[key] = vals[0]
    return merged
