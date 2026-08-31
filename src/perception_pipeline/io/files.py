#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading the small JSON files a dataset ships.

Lives in `io/` rather than `geometry.py`, where these sat by accident of history -- neither has
anything to do with geometry. Kept as its own module rather than folded into `bop.py` because
`dataset.py` needs both and `bop.py` imports `dataset`, so putting them there would make the
two circular. Nothing here imports from the rest of the package, which is what keeps that true.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@cache


def load_dataset_map(dataset_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load and cache the dataset object-name mappings.

    Lives here rather than in `dataset.py` so `detection.py` can call it without importing
    `dataset`, which imports `detection` -- the deferred import that used to paper over that
    cycle is gone.
    """
    name_to_id = {name: int(obj_id) for name, obj_id in load_json(dataset_dir / "dataset_map.json").items()}
    id_to_name = {obj_id: name for name, obj_id in name_to_id.items()}
    return name_to_id, id_to_name


# `read_json` is the historical name used by the depth-rasterizer call sites.
read_json = load_json
