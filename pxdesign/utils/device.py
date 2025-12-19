# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Device utilities for MPS/CUDA/CPU support.
Provides unified device management with priority: MPS > CUDA > CPU
"""

import gc
import logging
from contextlib import nullcontext
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device(local_rank: int = 0) -> torch.device:
    """
    Get best available device with priority: MPS > CUDA > CPU

    Args:
        local_rank: Local rank for distributed training (used for CUDA device selection)

    Returns:
        torch.device: The best available device
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def get_device_type(device: Optional[torch.device] = None) -> str:
    """
    Get device type string for autocast.

    Args:
        device: Optional device to check. If None, detects automatically.

    Returns:
        str: Device type string ("cuda", "mps", or "cpu")
    """
    if device is not None:
        return device.type

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def empty_cache(device: Optional[torch.device] = None) -> None:
    """
    Device-aware cache clearing.

    Args:
        device: Optional device to clear cache for. If None, clears based on available backend.
    """
    if device is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    # Always run garbage collection for better memory management
    gc.collect()


def synchronize(device: Optional[torch.device] = None) -> None:
    """
    Synchronize device operations.

    Args:
        device: Optional device to synchronize. If None, syncs based on available backend.
    """
    if device is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available.

    Returns:
        bool: True if CUDA or MPS is available
    """
    return torch.cuda.is_available() or (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    )


def get_autocast_context(dtype: torch.dtype, device: Optional[torch.device] = None):
    """
    Get appropriate autocast context for current device.

    Args:
        dtype: The dtype for mixed precision (e.g., torch.bfloat16)
        device: Optional device. If None, detects automatically.

    Returns:
        Context manager for autocast or nullcontext if not supported
    """
    device_type = get_device_type(device)

    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    elif device_type == "mps":
        # MPS supports autocast with bf16
        return torch.autocast(device_type="mps", dtype=dtype)
    # CPU doesn't benefit much from autocast
    return nullcontext()


def get_amp_autocast(enabled: bool = False, device: Optional[torch.device] = None):
    """
    Get AMP autocast context, compatible replacement for torch.cuda.amp.autocast.

    Args:
        enabled: Whether autocast is enabled
        device: Optional device. If None, detects automatically.

    Returns:
        Context manager for autocast
    """
    if not enabled:
        return nullcontext()

    device_type = get_device_type(device)

    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    elif device_type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.bfloat16, enabled=enabled)
    return nullcontext()


def set_device(device: torch.device) -> None:
    """
    Set the active device (only applicable for CUDA).

    Args:
        device: The device to set as active
    """
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # MPS and CPU don't need explicit device setting


def get_device_count() -> int:
    """
    Get number of available GPU devices.

    Returns:
        int: Number of available devices (1 for MPS, N for CUDA, 0 for CPU-only)
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 1  # MPS is always a single device
    return 0
