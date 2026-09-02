#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precompute the per-scene ground-truth z-buffer cache.

`evaluation.gt.render_gt_entries` rasterizes every object in a scene into one shared z-buffer to get
occlusion-aware GT masks. Uncached, that work is repeated for every target in the scene at
~12-14 s each. This precomputes it once per scene, so a sweep pays it once.

Run it before any multi-dataset run:

    python script/build_gt_cache.py --dataset <name>
    python script/build_gt_cache.py --config <name> --all

`--all` needs `--config` where `--dataset` does not: it names no dataset, so there is nothing for
the profile to be inferred from.

The cache is written under the active profile's `dataset.gt_cache_root`. It is regenerable
output, not source -- deleting it costs time, never correctness.
"""

from foundationpose_perception_pipeline.evaluation.gt import main

if __name__ == "__main__":
    main()
