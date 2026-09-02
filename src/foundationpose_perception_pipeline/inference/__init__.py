#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deployable path: images and intrinsics in, poses out.

**Nothing in this package may import `foundationpose_perception_pipeline.evaluation`, read `scene_gt.json`, or
take a ground-truth argument.** That is the whole point of the split: what runs here is what
would run on a robot, and the only way to keep that true is to make the dependency impossible
rather than merely discouraged.

The one thing inference legitimately needs that today comes from ground truth is the *list of
objects to look for*. That is a task specification, not a measurement -- a deployment supplies
it as configuration. See `ObjectSpec`.
"""

from foundationpose_perception_pipeline.inference.types import (
    ObjectSpec,
    ProposalPrediction,
    SceneInput,
    StagePredictions,
    TargetPrediction,
    proposals_from_stages,
)

__all__ = [
    "ObjectSpec",
    "ProposalPrediction",
    "SceneInput",
    "StagePredictions",
    "TargetPrediction",
    "proposals_from_stages",
]
