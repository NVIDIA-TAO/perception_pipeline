#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection: score every proposal and decide which survive.

Every input to the score -- `sam_score`, `render_mask_iou`, `render_box_iou`, `fp_score` -- is
produced by inference itself, so this policy is deployable rather than an evaluation artefact.

Unselected proposals are not discarded; they are marked and carried into the artifact with
their scores. That is what makes an offline cutoff sweep possible without re-running the model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from foundationpose_perception_pipeline.config import KEEP_ALL_RERANK_POLICY, SOFT_GLOBAL_RERANK_POLICY
from foundationpose_perception_pipeline.inference.config import SelectionConfig
from foundationpose_perception_pipeline.pose import PoseFilterResult


def rerank_formula_text(config: SelectionConfig) -> str:
    """Human-readable rerank formula. Thin wrapper; the definition lives on the config so the
    weights and the string that describes them cannot drift apart."""
    return config.formula_text()

def proposal_rerank_row(
    *,
    pred_index: int,
    sam_score: float,
    pose_result: PoseFilterResult,
    config: SelectionConfig,
    selected: bool,
) -> dict[str, Any]:
    """Assemble one proposal-selection diagnostic row for the final JSON report."""
    eligible = pose_result.pose_row_major is not None and pose_result.error is None
    fp_score = float(pose_result.fp_score) if pose_result.fp_score is not None else None
    rerank_score = None
    if eligible:
        fp_term = 0.0 if fp_score is None else (-fp_score / float(config.fp_score_divisor))
        rerank_score = float(
            config.weight_sam_score * float(sam_score)
            + config.weight_render_mask_iou * float(pose_result.proposal_render_mask_iou)
            + config.weight_render_box_iou * float(pose_result.proposal_render_box_iou)
            + fp_term
        )
    return {
        "pred_index": pred_index,
        "selected": bool(selected),
        "eligible": bool(eligible),
        "sam_score": float(sam_score),
        "render_mask_iou": float(pose_result.proposal_render_mask_iou),
        "render_box_iou": float(pose_result.proposal_render_box_iou),
        "fp_score": fp_score,
        "rerank_score": rerank_score,
        "formula": rerank_formula_text(config),
    }

def select_proposals(
    *,
    scores: np.ndarray,
    filter_results: list[PoseFilterResult],
    config: SelectionConfig,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    """Select proposal indices according to the configured post-pose policy."""
    if config.policy == KEEP_ALL_RERANK_POLICY:
        kept_indices = list(range(len(scores)))
        rows = [
            proposal_rerank_row(
                pred_index=pred_index,
                sam_score=float(score),
                pose_result=filter_results[pred_index],
                config=config,
                selected=True,
            )
            for pred_index, score in enumerate(scores)
        ]
        return kept_indices, rows, {
            "policy": config.policy,
            "formula": None,
            "cutoff": None,
            "selected_count": len(kept_indices),
            "eligible_count": sum(1 for row in rows if row["eligible"]),
        }

    if config.policy != SOFT_GLOBAL_RERANK_POLICY:
        raise ValueError(f"Unsupported proposal selection policy: {config.policy}")

    rows: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    for pred_index, score in enumerate(scores):
        row = proposal_rerank_row(
            pred_index=pred_index,
            sam_score=float(score),
            pose_result=filter_results[pred_index],
            config=config,
            selected=False,
        )
        if row["eligible"] and row["rerank_score"] is not None and float(row["rerank_score"]) >= config.cutoff:
            row["selected"] = True
            kept_indices.append(pred_index)
        rows.append(row)

    return kept_indices, rows, {
        "policy": config.policy,
        "formula": rerank_formula_text(config),
        "cutoff": float(config.cutoff),
        "selected_count": len(kept_indices),
        "eligible_count": sum(1 for row in rows if row["eligible"]),
    }

def mark_selected_filter_results(
    filter_results: list[PoseFilterResult],
    kept_indices: list[int],
) -> list[PoseFilterResult]:
    """Return pose-filter rows with the `kept` flag updated from selected indices."""
    kept_lookup = set(kept_indices)
    return [replace(result, kept=result.pred_index in kept_lookup) for result in filter_results]
