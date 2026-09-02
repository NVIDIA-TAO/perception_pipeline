#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pick stereo partners for a dataset whose "cameras" are frames of a STATIC scene.

Applies to any capture where the scene does not move between frames and the extrinsics are
known, which makes any two frames a valid stereo pair. T-LESS is the worked example (measured
world-pose drift 0.00 mm). A dataset that ships a real multi-camera rig does not need this --
its cameras are given, not chosen.

Selection is two filters in order, and the second is the one that is easy to leave out:

1. **Baseline band.** Distance between camera centres, C = -R_w2c^T t_w2c, which is exactly what
   `stereoRectify` reports as the rectified baseline -- so banding needs no rectification pass.
2. **Does it actually rectify horizontally.** `horizontal_partners` runs the pipeline's OWN
   `compute_rectification_poses` and the same parallax bounds the depth stage applies. Selecting
   on distance alone is not enough and this is the bug it caused on T-LESS: about a quarter of
   pairs fail those bounds, so exposing partners chosen by distance left some scenes with none
   that rectify and the depth stage raised `StereoDepthError`. Running the pipeline's own check
   here means the adapter never emits a scene the depth stage cannot handle.

Why the band matters at all, on T-LESS measured over 80 scenes: a longer baseline is more
precise per pixel of disparity error and harder to match, and the second half is what bites.
Every failure sat above 0.27 m and none below 0.24 m. Past roughly 45 degrees of rectification
rotation the overlap shrinks, matching degrades, and depth becomes the dominant error source.
The right band is a property of the rig's baseline relative to its working distance, so it is a
parameter rather than a constant.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from foundationpose_perception_pipeline.inference.stereo.rectify import (
    DEFAULT_MAX_DEPTH_PARALLAX,
    DEFAULT_MAX_VERTICAL_PARALLAX,
    DEFAULT_MIN_BASELINE_X,
    compute_rectification_poses,
    compute_rotation_errors,
)


def centres(cam: dict[str, Any], keys: list[str]) -> np.ndarray:
    """Camera centres in world coordinates, metres. C = -R_w2c^T t_w2c."""
    out = []
    for k in keys:
        R = np.asarray(cam[k]["cam_R_w2c"], float).reshape(3, 3)
        t = np.asarray(cam[k]["cam_t_w2c"], float).reshape(3) / 1000.0
        out.append(-R.T @ t)
    return np.asarray(out)


def horizontal_partners(
    cam: dict[str, Any],
    keys: list[str],
    base_i: int,
    cand: list[int],
    width: int,
    height: int,
) -> list[tuple[int, float]]:
    """Which candidates actually rectify horizontally, and at what rotation cost.

    Returns `(index, rotation_deg)` sorted by rotation ascending, so a caller that truncates
    keeps the pairs that rectify most cheaply. Candidates that raise during rectification are
    dropped rather than propagated: a pair that cannot be rectified is simply not a pair, and one
    bad candidate must not abort a whole scene.
    """
    K = np.stack([np.asarray(cam[k]["cam_K"], float).reshape(3, 3) for k in keys])
    E = np.stack([np.hstack([np.asarray(cam[k]["cam_R_w2c"], float).reshape(3, 3),
                             np.asarray(cam[k]["cam_t_w2c"], float).reshape(3, 1) / 1000.0])
                  for k in keys])
    out = []
    for i in cand:
        try:
            r1, r2, pb, pp = compute_rectification_poses(base_i, i, K, E, width, height)
        except Exception:  # noqa: BLE001 -- the exception type carries no signal here: a pair
            # that cannot be rectified, for whatever reason, is simply not a pair. Narrowing this
            # would let one unanticipated failure abort a whole scene, where dropping the
            # candidate costs at most one partner out of dozens.
            continue
        d = pb[:3, 3] - pp[:3, 3]
        if not (abs(d[0]) > DEFAULT_MIN_BASELINE_X
                and abs(d[1]) < DEFAULT_MAX_VERTICAL_PARALLAX
                and abs(d[2]) < DEFAULT_MAX_DEPTH_PARALLAX):
            continue
        rot = float(max(compute_rotation_errors(r1[:3, :3][None], np.eye(3)[None])[0],
                        compute_rotation_errors(r2[:3, :3][None], np.eye(3)[None])[0]))
        out.append((i, rot))
    out.sort(key=lambda t: t[1])          # lowest rectification rotation first
    return out


def partners_in_band(
    cam: dict[str, Any],
    keys: list[str],
    C: np.ndarray,
    base_i: int,
    *,
    baseline_min: float,
    baseline_max: float,
    max_partners: int,
    width: int,
    height: int,
) -> tuple[list[str], list[float]]:
    """Both filters together: the partners to expose for `keys[base_i]`, and their baselines.

    Returns `([], [])` when nothing in the band rectifies, which the caller should treat as "skip
    this base frame" rather than as an error -- on a narrow band that is a normal outcome for a
    large fraction of frames.
    """
    dist = np.linalg.norm(C - C[base_i], axis=1)
    band = [i for i in range(len(keys))
            if i != base_i and baseline_min <= dist[i] <= baseline_max]
    band.sort(key=lambda i: dist[i])
    ok = horizontal_partners(cam, keys, base_i, band, width, height)
    chosen = [keys[i] for i, _ in ok[:max_partners]]
    return chosen, [round(float(dist[keys.index(p)]), 4) for p in chosen]
