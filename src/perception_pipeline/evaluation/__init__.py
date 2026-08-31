#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Everything that needs ground truth.

The dividing line this package exists to draw: **nothing in here runs at deployment.** Scoring
predictions against `scene_gt.json`, rasterizing GT masks, matching proposals to instances and
computing pose error are all evaluation-only work. The inference path must never import from
this package -- if it does, the separation has been lost.

The split is not new; it was previously expressed as `with scoring_clock.measure():` blocks
inside `run_pipeline.main`, because the report already had to exclude GT cost from its
"production path only" runtime figure. This package makes the same boundary structural.
"""

from perception_pipeline.evaluation.detection import score_detection
from perception_pipeline.evaluation.gt import render_target_gt
from perception_pipeline.evaluation.pose_error import PoseMatchMetric, score_matched_poses

__all__ = [
    "PoseMatchMetric",
    "render_target_gt",
    "score_detection",
    "score_matched_poses",
]
