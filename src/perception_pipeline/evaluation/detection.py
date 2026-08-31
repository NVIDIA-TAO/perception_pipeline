#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detection scoring: proposals against ground-truth instances, and the summaries built from it."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from perception_pipeline.io.files import load_dataset_map


def score_detection(
    *,
    boxes: np.ndarray,
    masks: np.ndarray,
    gt_boxes: np.ndarray,
    gt_masks: np.ndarray,
    iou_thresholds: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return `(box_metrics, mask_metrics)` for one proposal set against one GT set.

    Called once per pipeline stage -- raw, post-refinement and post-rerank -- because the three
    together are what separate what SAM3 refinement *gained* from what the rerank cutoff *gave
    back*: refinement replaces masks and can raise TP, the rerank only drops proposals and can
    only lower it. A single number cannot show that, and the net can look like an unambiguous
    win when it is not.
    """
    box_ious, mask_ious = iou_matrices(boxes, masks, gt_boxes, gt_masks)
    return (
        metrics_for_thresholds(box_ious, iou_thresholds),
        metrics_for_thresholds(mask_ious, iou_thresholds),
    )


def save_summary(
    output_dir: Path,
    args: argparse.Namespace,
    result_targets: int,
    evaluated_targets: int,
    error_targets: int,
    aggregate: dict[str, dict[str, Counter]],
    by_dataset: dict[str, dict[str, dict[str, Counter]]],
    by_object: dict[str, dict[str, dict[str, Counter]]],
) -> None:
    """Write JSON and CSV summaries for the current results set."""
    summary = {
        "result_targets": result_targets,
        "evaluated_targets": evaluated_targets,
        "error_targets": error_targets,
        "confidence_threshold": args.confidence_threshold,
        "iou_thresholds": args.iou_thresholds,
        "overall": aggregate_to_rates(aggregate),
        "by_dataset": {name: aggregate_to_rates(value) for name, value in sorted(by_dataset.items())},
        "by_object": {name: aggregate_to_rates(value) for name, value in sorted(by_object.items())},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_type", "group", "kind", "iou_threshold", "tp", "fp", "fn", "precision", "recall"])
        rows = [("overall", "all", summary["overall"])]
        rows += [("dataset", name, value) for name, value in summary["by_dataset"].items()]
        rows += [("object", name, value) for name, value in summary["by_object"].items()]
        for group_type, group_name, group_summary in rows:
            for kind, by_threshold in group_summary.items():
                for threshold, metric in by_threshold.items():
                    writer.writerow([
                        group_type,
                        group_name,
                        kind,
                        threshold,
                        metric["tp"],
                        metric["fp"],
                        metric["fn"],
                        metric["precision"],
                        metric["recall"],
                    ])


def iou_matrices(
    pred_boxes: np.ndarray,
    pred_masks: np.ndarray,
    gt_boxes: np.ndarray,
    gt_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise box IoUs and mask IoUs between predictions and GT."""
    from perception_pipeline.geometry import box_iou

    box_ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            box_ious[pred_idx, gt_idx] = box_iou(pred_box, gt_box)

    mask_ious = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    pred_areas = np.asarray([mask.sum() for mask in pred_masks], dtype=np.float64)
    gt_areas = np.asarray([mask.sum() for mask in gt_masks], dtype=np.float64)
    for pred_idx, pred_mask in enumerate(pred_masks):
        for gt_idx, gt_mask in enumerate(gt_masks):
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = pred_areas[pred_idx] + gt_areas[gt_idx] - inter
            mask_ious[pred_idx, gt_idx] = inter / union if union > 0 else 0.0
    return box_ious, mask_ious


def metrics_for_thresholds(iou_matrix: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    """Run greedy matching metrics at each requested IoU threshold."""
    from perception_pipeline.geometry import greedy_match

    return {str(threshold): greedy_match(iou_matrix, threshold) for threshold in thresholds}


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Strip greedy-match results down to JSON-friendly summary fields."""
    return {
        threshold: {
            "tp": value["tp"],
            "fp": value["fp"],
            "fn": value["fn"],
            "precision": value["precision"],
            "recall": value["recall"],
            "matches": value["matches"],
        }
        for threshold, value in metrics.items()
    }


def update_aggregate(
    aggregate: dict[str, dict[str, Counter]],
    kind: str,
    metrics: dict[str, Any],
) -> None:
    """Accumulate TP/FP/FN counts into an aggregate for one metric kind."""
    for threshold, value in metrics.items():
        aggregate[kind][threshold].update({"tp": value["tp"], "fp": value["fp"], "fn": value["fn"]})


def aggregate_to_rates(aggregate: dict[str, dict[str, Counter]]) -> dict[str, dict[str, Any]]:
    """Convert accumulated counts into precision/recall summaries."""
    summary: dict[str, dict[str, Any]] = {}
    for kind, by_threshold in aggregate.items():
        summary[kind] = {}
        for threshold, counts in by_threshold.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            summary[kind][threshold] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
            }
    return summary


def object_name_for_record(dataset_root: Path, record: dict[str, Any]) -> str:
    """Recover an object display name for a saved result record."""
    if record.get("object_name"):
        return str(record["object_name"])
    try:
        return load_dataset_map(dataset_root / str(record["dataset"]))[1][int(record["obj_id"])]
    except Exception:  # noqa: BLE001 -- a display name, never a decision; any failure to resolve
        # one (missing dataset_map.json, unknown obj_id, malformed record) falls back to the BOP
        # id rather than aborting a scored run over a label.
        return f"obj_{int(record.get('obj_id', -1)):06d}"


def add_record_metrics(
    record: dict[str, Any],
    aggregate: dict[str, dict[str, Counter]],
    by_dataset: dict[str, dict[str, dict[str, Counter]]],
    by_object: dict[str, dict[str, dict[str, Counter]]],
    thresholds: list[float],
    dataset_root: Path,
) -> bool:
    """Fold one saved record's metrics into overall, dataset, and object aggregates."""
    if record.get("error") or "box_metrics" not in record or "mask_metrics" not in record:
        return False

    dataset = str(record["dataset"])
    object_name = object_name_for_record(dataset_root, record)
    for kind in ("box", "mask"):
        metrics = record[f"{kind}_metrics"]
        for threshold in thresholds:
            threshold_key = str(threshold)
            value = metrics.get(threshold_key)
            if value is None:
                continue
            counts = {
                "tp": int(value.get("tp", 0)),
                "fp": int(value.get("fp", 0)),
                "fn": int(value.get("fn", 0)),
            }
            aggregate[kind][threshold_key].update(counts)
            by_dataset[dataset][kind][threshold_key].update(counts)
            by_object[object_name][kind][threshold_key].update(counts)
    return True


def empty_aggregate(thresholds: list[float]) -> dict[str, dict[str, Counter]]:
    """Create zeroed TP/FP/FN counters for box and mask metrics."""
    return {
        "box": {str(threshold): Counter() for threshold in thresholds},
        "mask": {str(threshold): Counter() for threshold in thresholds},
    }


def aggregate_latest_results(
    results_path: Path,
    thresholds: list[float],
    dataset_root: Path,
) -> tuple[
    int,
    int,
    int,
    dict[str, dict[str, Counter]],
    dict[str, dict[str, dict[str, Counter]]],
    dict[str, dict[str, dict[str, Counter]]],
]:
    """Aggregate only the latest record for each target key in an evaluation JSONL."""
    aggregate = empty_aggregate(thresholds)
    by_dataset: dict[str, dict[str, dict[str, Counter]]] = defaultdict(lambda: empty_aggregate(thresholds))
    by_object: dict[str, dict[str, dict[str, Counter]]] = defaultdict(lambda: empty_aggregate(thresholds))
    latest: dict[str, dict[str, Any]] = {}

    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target_key = record.get("target_key")
                if target_key:
                    latest[str(target_key)] = record

    evaluated = 0
    errors = 0
    for record in latest.values():
        if record.get("error"):
            errors += 1
            continue
        if add_record_metrics(record, aggregate, by_dataset, by_object, thresholds, dataset_root):
            evaluated += 1

    return len(latest), evaluated, errors, aggregate, by_dataset, by_object


def detection_template(thresholds: list[float]) -> dict[str, dict[str, Counter]]:
    """Create zeroed TP/FP/FN counters for each detection metric threshold."""
    return {
        "box": {str(threshold): Counter() for threshold in thresholds},
        "mask": {str(threshold): Counter() for threshold in thresholds},
    }


def accumulate_detection(
    aggregate: dict[str, dict[str, Counter]],
    metrics: dict[str, Any],
) -> None:
    """Accumulate TP/FP/FN counts from one target into a detection aggregate."""
    for kind in ("box", "mask"):
        metric_key = f"{kind}_metrics"
        for threshold, value in metrics[metric_key].items():
            aggregate[kind][threshold].update(
                {
                    "tp": int(value["tp"]),
                    "fp": int(value["fp"]),
                    "fn": int(value["fn"]),
                }
            )


def finalize_detection(
    aggregate: dict[str, dict[str, Counter]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Convert accumulated TP/FP/FN counts into precision/recall summaries."""
    return aggregate_to_rates(aggregate)


def detection_counts_from_summary(summary: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    """Extract integer TP/FP/FN counts from one detection summary JSON object."""
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for kind, by_threshold in summary["overall"].items():
        counts[kind] = {}
        for threshold, metric in by_threshold.items():
            counts[kind][threshold] = {
                "tp": int(metric["tp"]),
                "fp": int(metric["fp"]),
                "fn": int(metric["fn"]),
            }
    return counts


def finalize_detection_counts(
    counts: dict[str, dict[str, dict[str, int]]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Convert accumulated integer detection counts into precision/recall metrics."""
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for kind, by_threshold in counts.items():
        summary[kind] = {}
        for threshold, metric in by_threshold.items():
            tp = int(metric["tp"])
            fp = int(metric["fp"])
            fn = int(metric["fn"])
            summary[kind][threshold] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
            }
    return summary


def percentile(values: list[float], q: float) -> float:
    """Return a percentile from a list of floats as a Python float."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _is_identity_4x4(sym: Any, tolerance: float = 1e-6) -> bool:
    """Whether a flattened 4x4 symmetry transformation is the identity.

    A tolerance rather than an equality test: these come out of JSON written by several
    different toolchains, and an identity stored as 0.9999999 is still an identity.
    """
    return np.allclose(np.asarray(sym, dtype=np.float64).reshape(4, 4), np.eye(4), atol=tolerance)


class PoseMetricRegistry:
    """Cache geometry and metadata needed to score predicted poses."""

    def __init__(self, dataset_root: Path, models_subdir: str = "models_cad") -> None:
        """Initialize per-dataset caches for CAD vertices and model metadata.

        `models_subdir` must match what the pose stage loaded meshes from, or the vertices scored
        against are not the vertices posed.
        """
        self.dataset_root = dataset_root
        self.models_subdir = models_subdir
        self.vertices_cache: dict[tuple[str, int], np.ndarray] = {}
        self.models_info_cache: dict[str, dict[str, Any]] = {}

    def vertices(self, dataset: str, obj_id: int) -> np.ndarray:
        """Load and cache CAD vertices for one dataset/object pair."""
        from perception_pipeline.geometry import read_binary_little_endian_ply

        key = (dataset, obj_id)
        if key not in self.vertices_cache:
            path = self.dataset_root / dataset / self.models_subdir / f"obj_{obj_id:06d}.ply"
            vertices, _ = read_binary_little_endian_ply(path)
            self.vertices_cache[key] = vertices.astype(np.float64)
        return self.vertices_cache[key]

    def model_info(self, dataset: str, obj_id: int) -> dict[str, Any]:
        """Load and cache the BOP model metadata record for one object."""
        if dataset not in self.models_info_cache:
            path = self.dataset_root / dataset / self.models_subdir / "models_info.json"
            self.models_info_cache[dataset] = json.loads(path.read_text(encoding="utf-8"))
        return self.models_info_cache[dataset][str(obj_id)]

    def is_symmetric(self, dataset: str, obj_id: int) -> bool:
        """Whether the object's `models_info.json` entry declares a real symmetry.

        The declaration is read directly, and it selects the metric: a symmetric object is scored
        with ADD-S and a bidirectional-nearest-neighbour max-vertex error, an asymmetric one with
        plain ADD and a per-vertex max. The second is far harsher -- a 180-degree flip of a
        near-symmetric part scores roughly its own diameter -- so the answer matters well beyond
        bookkeeping.

        Only NON-IDENTITY transformations count, because an identity says nothing about the
        object and whether one is listed is a property of the writer rather than of the shape.
        The BOP model format stores only the non-identity transformations and prepends the
        identity when building them (`bop_toolkit_lib.misc.get_symmetry_transformations`); other
        toolchains store it explicitly as the first entry. Filtering identities reads both
        conventions the same way, where counting entries misreads one in each direction.

        An object that declares nothing is reported asymmetric. That is the declaration taken at
        its word rather than a measurement of the shape, so a wrong or absent declaration does
        not fail loudly here -- it reports a pose as catastrophically wrong when it is visually
        indistinguishable from the truth.
        """
        info = self.model_info(dataset, obj_id)
        discrete = [sym for sym in info.get("symmetries_discrete", []) if not _is_identity_4x4(sym)]
        continuous = info.get("symmetries_continuous", [])
        return len(discrete) > 0 or len(continuous) > 0

    def diameter_mm(self, dataset: str, obj_id: int) -> float:
        """Return the object diameter in millimeters from models_info.json."""
        return float(self.model_info(dataset, obj_id)["diameter"])

    def compute(
        self,
        *,
        dataset: str,
        obj_id: int,
        pred_pose_row_major: list[float],
        gt_rotation: np.ndarray,
        gt_translation_mm: np.ndarray,
    ) -> dict[str, float | str | bool | None]:
        """Compute pose errors and success metrics for one predicted pose."""
        vertices_mm = self.vertices(dataset, obj_id)
        pred_pose = np.asarray(pred_pose_row_major, dtype=np.float64).reshape(4, 4)
        pred_rotation = pred_pose[:3, :3]
        pred_translation_mm = pred_pose[:3, 3] * 1000.0

        pred_points = vertices_mm @ pred_rotation.T + pred_translation_mm[None, :]
        gt_points = vertices_mm @ gt_rotation.T + gt_translation_mm[None, :]

        translation_error_mm = float(np.linalg.norm(pred_translation_mm - gt_translation_mm))
        symmetric = self.is_symmetric(dataset, obj_id)
        if symmetric:
            tree = cKDTree(gt_points)
            nearest_mm, _ = tree.query(pred_points, k=1, workers=-1)
            add_metric = "adds"
            rotation_error_deg = None
            add_or_adds_mm = float(np.mean(nearest_mm))
            pred_tree = cKDTree(pred_points)
            gt_to_pred_mm, _ = pred_tree.query(gt_points, k=1, workers=-1)
            max_vertex_error_mm = float(max(nearest_mm.max(), gt_to_pred_mm.max()))
        else:
            add_metric = "add"
            delta = pred_rotation @ gt_rotation.T
            cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
            rotation_error_deg = float(math.degrees(math.acos(cosine)))
            per_vertex_mm = np.linalg.norm(pred_points - gt_points, axis=1)
            add_or_adds_mm = float(per_vertex_mm.mean())
            max_vertex_error_mm = float(per_vertex_mm.max())

        diameter_mm = self.diameter_mm(dataset, obj_id)
        diameter_frac = add_or_adds_mm / diameter_mm if diameter_mm > 0 else float("inf")
        return {
            "translation_error_mm": translation_error_mm,
            "rotation_error_deg": rotation_error_deg,
            "add_metric": add_metric,
            "add_or_adds_mm": add_or_adds_mm,
            "add_or_adds_diameter_frac": diameter_frac,
            "pose_success_0p1d": diameter_frac <= 0.1,
            "max_vertex_error_mm": max_vertex_error_mm,
        }


def summarize_pose_rows(
    rows: list[dict[str, Any]],
    *,
    pose_error_threshold_mm: float,
    pose_error_required_rate: float,
) -> dict[str, Any]:
    """Aggregate matched-pose metrics across many prediction rows."""
    if not rows:
        return {
            "matched_predictions": 0,
            "translation_error_mm_mean": None,
            "translation_error_mm_median": None,
            "rotation_error_deg_mean": None,
            "rotation_error_deg_median": None,
            "add_or_adds_mm_mean": None,
            "add_or_adds_mm_median": None,
            "add_or_adds_diameter_frac_mean": None,
            "add_or_adds_diameter_frac_median": None,
            "pose_success_0p1d_rate": None,
            "translation_error_le_3mm_rate": None,
            "add_or_adds_le_3mm_rate": None,
            "max_vertex_error_mm_mean": None,
            "max_vertex_error_mm_median": None,
            "max_vertex_error_mm_p90": None,
            "max_vertex_error_mm_p99": None,
            "max_vertex_error_within_threshold_rate": None,
            "max_vertex_error_meets_required_rate": None,
        }

    translation_errors = [float(row["translation_error_mm"]) for row in rows]
    add_errors = [float(row["add_or_adds_mm"]) for row in rows]
    add_fracs = [float(row["add_or_adds_diameter_frac"]) for row in rows]
    max_vertex_errors = [float(row["max_vertex_error_mm"]) for row in rows]
    rotation_errors = [float(row["rotation_error_deg"]) for row in rows if row["rotation_error_deg"] is not None]
    max_vertex_le_threshold_rate = float(
        sum(error <= pose_error_threshold_mm for error in max_vertex_errors) / len(rows)
    )
    return {
        "matched_predictions": len(rows),
        "translation_error_mm_mean": float(statistics.fmean(translation_errors)),
        "translation_error_mm_median": float(statistics.median(translation_errors)),
        "rotation_error_deg_mean": float(statistics.fmean(rotation_errors)) if rotation_errors else None,
        "rotation_error_deg_median": float(statistics.median(rotation_errors)) if rotation_errors else None,
        "add_or_adds_mm_mean": float(statistics.fmean(add_errors)),
        "add_or_adds_mm_median": float(statistics.median(add_errors)),
        "add_or_adds_diameter_frac_mean": float(statistics.fmean(add_fracs)),
        "add_or_adds_diameter_frac_median": float(statistics.median(add_fracs)),
        "pose_success_0p1d_rate": float(sum(bool(row["pose_success_0p1d"]) for row in rows) / len(rows)),
        "translation_error_le_3mm_rate": float(sum(error <= 3.0 for error in translation_errors) / len(rows)),
        "add_or_adds_le_3mm_rate": float(sum(error <= 3.0 for error in add_errors) / len(rows)),
        "max_vertex_error_mm_mean": float(statistics.fmean(max_vertex_errors)),
        "max_vertex_error_mm_median": float(statistics.median(max_vertex_errors)),
        # p90 alongside p99: the configured rate is a P99-like criterion, but P99 over a few dozen
        # matches is one or two instances and swings run to run. p90 shows whether a change moved
        # the bulk of the distribution or only its tail.
        "max_vertex_error_mm_p90": percentile(max_vertex_errors, 90.0),
        "max_vertex_error_mm_p99": percentile(max_vertex_errors, 99.0),
        "max_vertex_error_within_threshold_rate": max_vertex_le_threshold_rate,
        "max_vertex_error_meets_required_rate": bool(
            max_vertex_le_threshold_rate >= pose_error_required_rate
        ),
    }


def summarize_runtime_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-scene runtime into mean/median/min/max.

    `runtime_sec` is **production work only** -- depth generation, SAM3, FoundationPose,
    refinement and reranking. Two costs are measured separately and subtracted, and both totals
    are reported so what was removed stays visible:

    - `scoring_sec` -- scoring against ground truth, which is unavailable at deployment. Measured
      over one 34-scene dataset: 19.4 s per target against a cold `gt_cache`, 1.1 s once warm.
      Cold, that exceeds the production cost of the same scene; warm it is negligible. Scoring
      READS the cache and never writes it, so the cold cost is paid on every evaluation until
      `build_gt_cache.py` is run -- it is not amortized by the first slow pass. Both figures scale
      with mesh complexity and instance count; read them as an order of magnitude.
    - `overlay_sec` -- diagnostic overlay PNGs, per target and per scene.
    """
    if not rows:
        return {
            "scene_count": 0,
            "runtime_sec_mean": None,
            "runtime_sec_median": None,
            "runtime_sec_min": None,
            "runtime_sec_max": None,
            "runtime_sec_total": 0.0,
            "scoring_sec_total": 0.0,
            "overlay_sec_total": 0.0,
        }
    runtimes = [float(row["runtime_sec"]) for row in rows]
    return {
        "scene_count": len(rows),
        "runtime_sec_mean": float(statistics.fmean(runtimes)),
        "runtime_sec_median": float(statistics.median(runtimes)),
        "runtime_sec_min": float(min(runtimes)),
        "runtime_sec_max": float(max(runtimes)),
        "runtime_sec_total": float(sum(runtimes)),
        "scoring_sec_total": float(sum(float(row.get("scoring_sec", 0.0)) for row in rows)),
        "overlay_sec_total": float(sum(float(row.get("overlay_sec", 0.0)) for row in rows)),
    }


def depth_overall_summary(depth_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-scene depth rows into the report-level summary."""
    valid_count = sum(int(row["valid_count"]) for row in depth_rows)
    comparable_rows = [row for row in depth_rows if row["mae_mm"] is not None]
    if valid_count == 0:
        return {
            "scene_count": len(depth_rows),
            "valid_count": 0,
            "mae_mm": None,
            "rmse_mm": None,
            "mean_scene_median_abs_mm": None,
            "object_valid_count": 0,
            "object_mae_mm": None,
            "object_rmse_mm": None,
            "mean_scene_object_median_abs_mm": None,
        }
    if not comparable_rows:
        return {
            "scene_count": len(depth_rows),
            "valid_count": valid_count,
            "mae_mm": None,
            "rmse_mm": None,
            "mean_scene_median_abs_mm": None,
            "object_valid_count": 0,
            "object_mae_mm": None,
            "object_rmse_mm": None,
            "mean_scene_object_median_abs_mm": None,
        }
    sum_abs = sum(float(row["sum_abs_mm"]) for row in comparable_rows)
    sum_sq = sum(float(row["sum_sq_mm"]) for row in comparable_rows)
    scene_medians = [
        float(row["median_abs_mm"])
        for row in comparable_rows
        if row["median_abs_mm"] is not None and not math.isnan(float(row["median_abs_mm"]))
    ]
    # Object-pixel aggregate, pooled the same way: this is the figure that tracks pose quality,
    # since whole-image error is dominated by the mat, bin and floor.
    object_rows = [row for row in depth_rows if row.get("object_mae_mm") is not None]
    object_count = sum(int(row["object_valid_count"]) for row in object_rows)
    object_medians = [
        float(row["object_median_abs_mm"])
        for row in object_rows
        if row.get("object_median_abs_mm") is not None
        and not math.isnan(float(row["object_median_abs_mm"]))
    ]
    object_summary = {
        "object_valid_count": object_count,
        "object_mae_mm": (
            sum(float(row["object_mae_mm"]) * int(row["object_valid_count"]) for row in object_rows)
            / object_count
            if object_count
            else None
        ),
        "object_rmse_mm": (
            math.sqrt(
                sum(
                    float(row["object_rmse_mm"]) ** 2 * int(row["object_valid_count"])
                    for row in object_rows
                )
                / object_count
            )
            if object_count
            else None
        ),
        "mean_scene_object_median_abs_mm": (
            statistics.fmean(object_medians) if object_medians else None
        ),
    }
    return {
        "scene_count": len(depth_rows),
        "valid_count": valid_count,
        **object_summary,
        "mae_mm": sum_abs / valid_count,
        "rmse_mm": math.sqrt(sum_sq / valid_count),
        "mean_scene_median_abs_mm": float(statistics.fmean(scene_medians)) if scene_medians else None,
    }
