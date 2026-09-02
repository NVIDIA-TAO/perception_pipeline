#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pose error for predictions matched to ground-truth instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PoseMatchMetric:
    """Pose-error metrics for one prediction matched to one GT instance."""

    pred_index_raw: int
    pred_index_pose_input: int
    pred_index_kept: int | None
    gt_index: int
    mask_iou: float
    translation_error_mm: float
    rotation_error_deg: float | None
    add_metric: str
    gt_visible_fraction: float
    add_or_adds_mm: float
    add_or_adds_diameter_frac: float
    pose_success_0p1d: bool
    max_vertex_error_mm: float


def score_matched_poses(
    *,
    matches: list[Any],
    kept_indices: list[int],
    pose_input_source_indices: np.ndarray,
    filter_results: list[Any],
    gt_entries: list[dict[str, Any]],
    gt_visible_fractions: Any,
    pose_metric_registry: Any,
    dataset: str,
    obj_id: int,
) -> list[PoseMatchMetric]:
    """Score every kept prediction that matched a visible GT instance.

    `matches` comes from the *filtered* mask metrics at the pose-match threshold, so three
    conditions are already baked in and all three matter when reading the resulting rate:
    the prediction survived reranking, it overlaps a GT instance by at least that IoU, and
    (checked here) it produced a pose at all.

    Matching is by mask IoU, not by pose agreement. A prediction whose mask lands on the right
    instance is scored even if its pose is inverted -- that is deliberate, and it is why a
    180-degree flip shows up as a large max-vertex error rather than quietly vanishing from the
    population. A prediction that matches no GT instance is a detection false positive and has
    no pose error to report.
    """
    metrics: list[PoseMatchMetric] = []
    for kept_pred_index, gt_index, mask_iou in matches:
        pose_input_pred_index = kept_indices[kept_pred_index]
        raw_pred_index = int(pose_input_source_indices[pose_input_pred_index])
        pose_info = filter_results[pose_input_pred_index]
        if pose_info.pose_row_major is None:
            continue
        gt_entry = gt_entries[gt_index]
        metric = pose_metric_registry.compute(
            dataset=dataset,
            obj_id=obj_id,
            pred_pose_row_major=pose_info.pose_row_major,
            gt_rotation=np.asarray(gt_entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
            gt_translation_mm=np.asarray(gt_entry["cam_t_m2c"], dtype=np.float64),
        )
        metrics.append(
            PoseMatchMetric(
                pred_index_raw=raw_pred_index,
                pred_index_pose_input=pose_input_pred_index,
                pred_index_kept=kept_pred_index,
                gt_index=gt_index,
                mask_iou=float(mask_iou),
                translation_error_mm=float(metric["translation_error_mm"]),
                rotation_error_deg=(
                    float(metric["rotation_error_deg"])
                    if metric["rotation_error_deg"] is not None
                    else None
                ),
                add_metric=str(metric["add_metric"]),
                # Carried per match so the report can band by visibility without re-running;
                # filtering here instead would fix the bands at run time.
                gt_visible_fraction=float(gt_visible_fractions[gt_index]),
                add_or_adds_mm=float(metric["add_or_adds_mm"]),
                add_or_adds_diameter_frac=float(metric["add_or_adds_diameter_frac"]),
                pose_success_0p1d=bool(metric["pose_success_0p1d"]),
                max_vertex_error_mm=float(metric["max_vertex_error_mm"]),
            )
        )
    return metrics
