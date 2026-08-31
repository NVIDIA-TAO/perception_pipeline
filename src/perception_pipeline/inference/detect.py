#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detection: text-prompted proposals from SAM3.

One function today, and deliberately its own module: this is the seam a second detector would
plug into, and a `detect.py` that exists is a more obvious place to add one than a shared
`stages.py` that would have to be split first.
"""

from __future__ import annotations

from typing import Any


def base_text_state_from_prompt_state(prompt_state: dict[str, Any]) -> dict[str, Any]:
    """Extract the reusable SAM3 text-prompt state needed for box refinements."""
    return {
        "original_height": prompt_state["original_height"],
        "original_width": prompt_state["original_width"],
        "backbone_out": prompt_state["backbone_out"],
    }
