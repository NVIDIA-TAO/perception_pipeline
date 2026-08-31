#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch/device helpers shared by the SAM3 inference paths."""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor, upcast low-precision floats if needed, and move it to NumPy."""
    if tensor.dtype in (torch.bfloat16, torch.float16):
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def inference_context(device: str):
    """Choose the autocast context used during SAM3 inference on the target device."""
    if device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.amp.autocast(device_type="cpu", enabled=False)
