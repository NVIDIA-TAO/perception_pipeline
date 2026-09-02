#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The data the inference path consumes and produces.

These types are the seam between inference and evaluation. Evaluation reads a
`TargetPrediction`; it never reaches into the pipeline's loop variables. That is what makes
re-scoring possible without re-inferring -- changing an IoU threshold, a visibility band or a
rerank cutoff becomes seconds of arithmetic over a saved artifact instead of hours of GPU.

Two deliberate choices worth not undoing:

**Unselected proposals are kept.** `TargetPrediction` carries every proposal with its scores and
a `selected` flag, not just the survivors. Dropping the rest at inference time is the one
irreversible loss in the whole design: it is what makes an offline rerank-cutoff sweep possible,
and that sweep has already paid for itself once.

**Masks stay in memory as arrays, and serialize as sidecar paths.** Inlining them as RLE would
make a run self-contained but bloat a JSONL that is routinely read by eye. `to_dict` therefore
takes the paths the caller wrote them to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SceneInput:
    """One frame's worth of input: what a camera and a depth source provide.

    `depth_m` is an array rather than a path on purpose -- the estimator must not know whether
    depth came from FoundationStereo, a sensor, or a file. That ignorance is what makes the
    path deployable.
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    camera_matrix: np.ndarray
    scene_key: str


@dataclass(frozen=True)
class ObjectSpec:
    """One object the pipeline is asked to find, and what it needs to find it.

    This is the task specification. In evaluation it is derived from `scene_gt.json`, but
    nothing about it is a measurement: a deployment supplies the same fields from config.

    `symmetry` is unused by inference and carried only so that pose error has it later, rather
    than maintaining a second object description that must be kept in sync.
    """

    object_key: str
    obj_id: int
    prompt: str
    mesh_path: Path | None = None
    symmetry: Any | None = None


@dataclass
class StagePredictions:
    """Proposals at one pipeline stage: raw SAM3, post-refinement, or post-rerank.

    Three stages are carried separately because the difference between them is load-bearing:
    refinement *replaces* masks and can raise TP, while the rerank only *drops* proposals and
    can only lower it. Collapsing them loses the ability to attribute a change to either.
    """

    boxes_xyxy: np.ndarray
    scores: np.ndarray
    masks: np.ndarray
    source_indices: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        """Number of proposals at this stage."""
        return len(self.boxes_xyxy)


@dataclass
class ProposalPrediction:
    """One proposal's pose and the scores that decided its fate.

    Flattens what the pipeline holds as parallel arrays. Every field here is GT-free: the
    render IoUs compare a proposal's mask against a CAD render of *its own predicted pose*, not
    against ground truth, which is why the whole selection policy is deployable.
    """

    raw_index: int
    pose_input_index: int | None
    kept_index: int | None
    box_xyxy: list[float]
    sam_score: float
    pose_row_major: list[float] | None = None
    fp_score: float | None = None
    render_mask_iou: float | None = None
    render_box_iou: float | None = None
    rerank_score: float | None = None
    selected: bool = False
    # Both are produced by the selection stage, so they are inference output, not metrics:
    # `eligible` is whether the proposal cleared the rerank cutoff, `formula` the rule that
    # decided it. Recorded per proposal so a cutoff can be re-applied offline without a rerun.
    eligible: bool | None = None
    formula: str | None = None


@dataclass
class TargetPrediction:
    """Everything inference produced for one (scene, object) target.

    The unit is a target rather than a scene because that is how the pipeline processes work:
    one object class at a time, so a single FoundationPose context serves all its proposals.
    """

    target_key: str
    dataset: str
    scene_id: int
    im_id: int
    obj_id: int
    object_name: str
    prompt: str
    raw: StagePredictions
    pose_input: StagePredictions
    kept: StagePredictions
    proposals: list[ProposalPrediction] = field(default_factory=list)
    kept_indices: list[int] = field(default_factory=list)
    # Everything else inference produced: the per-stage FoundationPose results, the refinement
    # and selection summaries, and the configuration echoes that make a run self-describing.
    # Kept as a bag rather than typed fields because its contents are dictated by what the
    # stages happen to emit -- typing it would mean re-declaring their internals here.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, mask_paths: dict[str, Path] | None = None) -> dict[str, Any]:
        """Serialize for `predictions.jsonl` -- predictions only, no metrics.

        Deliberately excludes anything ground truth touched. If a field cannot be produced
        without `scene_gt.json`, it does not belong in this artifact.
        """
        record: dict[str, Any] = {
            "target_key": self.target_key,
            "dataset": self.dataset,
            "scene_id": self.scene_id,
            "im_id": self.im_id,
            "obj_id": self.obj_id,
            "object_name": self.object_name,
            "prompt": self.prompt,
            "num_predictions": len(self.raw),
            "num_pose_input_predictions": len(self.pose_input),
            "num_predictions_kept": len(self.kept_indices),
            "boxes_xyxy": self.raw.boxes_xyxy.tolist(),
            "scores": self.raw.scores.tolist(),
            "pose_input_boxes_xyxy": self.pose_input.boxes_xyxy.tolist(),
            "pose_input_scores": self.pose_input.scores.tolist(),
            "pose_input_source_indices": [int(i) for i in self.pose_input.source_indices],
            "kept_indices": list(self.kept_indices),
            "kept_boxes_xyxy": self.kept.boxes_xyxy.tolist(),
            "kept_scores": self.kept.scores.tolist(),
            "proposals": [
                {
                    "raw_index": p.raw_index,
                    "pose_input_index": p.pose_input_index,
                    "kept_index": p.kept_index,
                    "box_xyxy": p.box_xyxy,
                    "sam_score": p.sam_score,
                    "pose_row_major": p.pose_row_major,
                    "fp_score": p.fp_score,
                    "render_mask_iou": p.render_mask_iou,
                    "render_box_iou": p.render_box_iou,
                    "rerank_score": p.rerank_score,
                    "selected": p.selected,
                    "eligible": p.eligible,
                    "formula": p.formula,
                }
                for p in self.proposals
            ],
        }
        record.update(self.extra)
        if mask_paths:
            record.update({key: str(value) for key, value in mask_paths.items()})
        return record


def proposals_from_stages(
    *,
    pose_input: StagePredictions,
    kept_indices: list[int],
    selection_results: list[Any],
    rerank_rows: list[dict[str, Any]],
) -> list[ProposalPrediction]:
    """Flatten the pipeline's parallel arrays into one record per pose-input proposal.

    Indexed on the *pose-input* stage rather than the raw stage, because that is the set
    FoundationPose actually ran on and therefore the only set with poses and scores. Refinement
    can replace a raw proposal, so `raw_index` is a back-reference via `source_indices`, not an
    identity.
    """
    rerank_by_index = {int(row["pred_index"]): row for row in rerank_rows}
    kept_position = {int(index): position for position, index in enumerate(kept_indices)}
    proposals: list[ProposalPrediction] = []
    for pose_input_index in range(len(pose_input)):
        raw_index = (
            int(pose_input.source_indices[pose_input_index])
            if pose_input_index < len(pose_input.source_indices)
            else pose_input_index
        )
        result = selection_results[pose_input_index] if pose_input_index < len(selection_results) else None
        rerank = rerank_by_index.get(pose_input_index, {})
        proposals.append(
            ProposalPrediction(
                raw_index=raw_index,
                pose_input_index=pose_input_index,
                kept_index=kept_position.get(pose_input_index),
                box_xyxy=[float(v) for v in pose_input.boxes_xyxy[pose_input_index]],
                sam_score=float(pose_input.scores[pose_input_index]),
                pose_row_major=getattr(result, "pose_row_major", None),
                fp_score=getattr(result, "fp_score", None),
                render_mask_iou=getattr(result, "proposal_render_mask_iou", None),
                render_box_iou=getattr(result, "proposal_render_box_iou", None),
                rerank_score=rerank.get("rerank_score"),
                selected=pose_input_index in kept_position,
                eligible=rerank.get("eligible"),
                formula=rerank.get("formula"),
            )
        )
    return proposals
