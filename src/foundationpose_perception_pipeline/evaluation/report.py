#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for building and writing pipeline summary artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundationpose_perception_pipeline.config import DEFAULT_POSE_MIN_VISIBLE_FRACTIONS
from foundationpose_perception_pipeline.evaluation.detection import (
    depth_overall_summary,
    detection_counts_from_summary,
    finalize_detection,
    finalize_detection_counts,
    summarize_pose_rows,
    summarize_runtime_rows,
)


@dataclass(frozen=True)
class PipelineReportConfig:
    """Configuration values needed to render the pipeline summaries and report."""

    dataset: str
    confidence_threshold: float
    depth_source: str
    min_visible_fraction: float
    pose_match_threshold: float
    sam3_refinement_policy: str
    proposal_selection_policy: str
    rerank_cutoff: float
    rerank_formula: str | None
    prompt_source: str
    soft_global_rerank_policy: str
    replace_mid_nms06_refinement_policy: str
    default_sam3_refinement_policy: str
    refinement_low_miou: float
    refinement_high_miou: float
    refinement_nms_threshold: float
    max_vertex_error_threshold_mm: float
    max_vertex_error_required_rate: float
    # The base camera every pose is expressed in, and the collected-depth file that was actually
    # read. Quoted rather than hardcoded for the reason `_depth_source_note` gives about
    # `--depth-source`: the report must not name camera 0 for a run scored against camera 1's map.
    # `collected_depth_filename` is the resolved `--collected-depth-filename` rather than the
    # profile's derived value, so an override appears here instead of being papered over.
    base_camera_id: int
    collected_depth_filename: str
    # False under `--no-depth-metrics`. Defaulted so this stays additive for any caller that
    # predates the flag; the report needs it to say *why* the depth table is empty, since
    # "not measured" and "measured as zero error" look identical once both render as `n/a`.
    depth_metrics_enabled: bool = True
    # Whether `rerank_formula` describes the scoring that RAN or only what the profile configures.
    # `--rerank-weight-*` can override the parsed formula for a single run, so the two can differ;
    # when inference recorded its own formula the report quotes that, and when it could not the
    # report has to say so rather than present a profile value as if it were provenance.
    rerank_formula_from_run: bool = True


@dataclass(frozen=True)
class MultiDatasetReportConfig:
    """Configuration values needed to build the all-datasets aggregate report."""

    dataset_root: Path
    output_root: Path
    datasets: list[str]
    generated_on: str
    depth_source: str
    confidence_threshold: float
    resolution: int
    sam3_refinement_policy: str
    proposal_selection_policy: str
    refinement_replace_miou_low: float
    refinement_replace_miou_high: float
    refinement_nms_threshold: float
    rerank_formula: str
    rerank_cutoff: float
    fp_prepare_batch: int
    fp_n_hypotheses: int
    fp_n_refine: int
    pose_error_threshold_mm: float
    pose_error_required_rate: float
    pose_min_visible_fractions: tuple[float, ...] = DEFAULT_POSE_MIN_VISIBLE_FRACTIONS
    # Where the text prompts came from. Passed in rather than hardcoded: prompts moved out of
    # Python and into the dataset profile, so the report must name the profile actually used.
    prompt_source: str = "dataset profile prompts, with object-name fallback"


def write_detection_csv(path: Path, summary: dict[str, Any]) -> None:
    """Write overall and per-object detection metrics to a flat CSV file."""
    lines = ["group_type,group_name,kind,iou_threshold,tp,fp,fn,precision,recall"]
    rows = [("overall", "all", summary["overall"])]
    rows.extend(("object", name, value) for name, value in summary["by_object"].items())
    for group_type, group_name, group_summary in rows:
        for kind, by_threshold in group_summary.items():
            for threshold, metric in by_threshold.items():
                lines.append(
                    ",".join(
                        [
                            group_type,
                            group_name,
                            kind,
                            threshold,
                            str(metric["tp"]),
                            str(metric["fp"]),
                            str(metric["fn"]),
                            f"{metric['precision']:.6f}",
                            f"{metric['recall']:.6f}",
                        ]
                    )
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pose_csv(path: Path, summary: dict[str, Any]) -> None:
    """Write overall and per-object pose metrics to a flat CSV file."""
    lines = [
        "group_type,group_name,matched_predictions,translation_error_mm_mean,translation_error_mm_median,"
        "rotation_error_deg_mean,rotation_error_deg_median,add_or_adds_mm_mean,add_or_adds_mm_median,"
        "add_or_adds_diameter_frac_mean,add_or_adds_diameter_frac_median,pose_success_0p1d_rate,"
        "max_vertex_error_mm_mean,max_vertex_error_mm_median,max_vertex_error_mm_p90,max_vertex_error_mm_p99,"
        "max_vertex_error_within_threshold_rate,max_vertex_error_meets_required_rate"
    ]
    rows = [("overall", "all", summary["overall"])]
    rows.extend(("object", name, value) for name, value in summary["by_object"].items())
    for group_type, group_name, metric in rows:
        lines.append(
            ",".join(
                [
                    group_type,
                    group_name,
                    str(metric["matched_predictions"]),
                    str(metric["translation_error_mm_mean"]),
                    str(metric["translation_error_mm_median"]),
                    str(metric["rotation_error_deg_mean"]),
                    str(metric["rotation_error_deg_median"]),
                    str(metric["add_or_adds_mm_mean"]),
                    str(metric["add_or_adds_mm_median"]),
                    str(metric["add_or_adds_diameter_frac_mean"]),
                    str(metric["add_or_adds_diameter_frac_median"]),
                    str(metric["pose_success_0p1d_rate"]),
                    str(metric["max_vertex_error_mm_mean"]),
                    str(metric["max_vertex_error_mm_median"]),
                    str(metric["max_vertex_error_mm_p90"]),
                    str(metric["max_vertex_error_mm_p99"]),
                    str(metric["max_vertex_error_within_threshold_rate"]),
                    str(metric["max_vertex_error_meets_required_rate"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_depth_csv(path: Path, overall: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write per-scene depth metrics plus the overall summary to CSV."""

    def render_csv(value: Any) -> str:
        """Render nullable values as CSV-safe strings."""
        return "" if value is None else str(value)

    header = (
        "scene_id,depth_source,comparison_mode,left_camera,right_camera,selected_pair,baseline_m,stereo_axis,"
        "valid_count,valid_fraction,estimated_valid_fraction,collected_valid_fraction,"
        "mae_mm,rmse_mm,median_abs_mm,"
        "object_valid_count,object_coverage,object_mae_mm,object_rmse_mm,object_median_abs_mm"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    f"{int(row['scene_id']):06d}",
                    render_csv(row["depth_source"]),
                    render_csv(row["comparison_mode"]),
                    render_csv(row["left_camera"]),
                    render_csv(row["right_camera"]),
                    f"\"{row['selected_pair']}\"" if row["selected_pair"] is not None else "",
                    f"{float(row['baseline_m']):.6f}" if row["baseline_m"] is not None else "",
                    render_csv(row["stereo_axis"]),
                    str(int(row["valid_count"])),
                    f"{float(row['valid_fraction']):.6f}",
                    f"{float(row['estimated_valid_fraction']):.6f}",
                    f"{float(row['collected_valid_fraction']):.6f}",
                    render_csv(row["mae_mm"]),
                    render_csv(row["rmse_mm"]),
                    render_csv(row["median_abs_mm"]),
                    render_csv(row.get("object_valid_count")),
                    render_csv(row.get("object_coverage")),
                    render_csv(row.get("object_mae_mm")),
                    render_csv(row.get("object_rmse_mm")),
                    render_csv(row.get("object_median_abs_mm")),
                ]
            )
        )
    lines.append("")
    lines.append(
        ",".join(
            [
                "overall",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                str(overall["valid_count"]),
                "",
                "",
                "",
                render_csv(overall["mae_mm"]),
                render_csv(overall["rmse_mm"]),
                render_csv(overall["mean_scene_median_abs_mm"]),
                render_csv(overall.get("object_valid_count")),
                "",
                render_csv(overall.get("object_mae_mm")),
                render_csv(overall.get("object_rmse_mm")),
                render_csv(overall.get("mean_scene_object_median_abs_mm")),
            ]
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_pose_visibility_table(by_visibility: dict[str, Any]) -> str:
    """Render pose metrics at each GT-visibility band as one row per band."""
    if not by_visibility:
        return "No visibility bands computed."

    def fmt(value: Any) -> str:
        """Render nullable pose metrics."""
        return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))

    lines = [
        "| Min GT Visibility | Scored / Matched | Median Max-Vertex (mm) | P90 (mm) | P99 (mm) | "
        "Max-Vertex <= 5 mm | Mean ADD/ADD-S (mm) | 10% Diameter Success |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for band, entry in sorted(by_visibility.items(), key=lambda kv: float(kv[0])):
        metric = entry["overall"]
        lines.append(
            f"| {band} | {entry['matched_predictions_scored']} / {entry['matched_predictions_total']} | "
            f"{fmt(metric['max_vertex_error_mm_median'])} | {fmt(metric['max_vertex_error_mm_p90'])} | "
            f"{fmt(metric['max_vertex_error_mm_p99'])} | {fmt(metric['max_vertex_error_within_threshold_rate'])} | "
            f"{fmt(metric['add_or_adds_mm_mean'])} | {fmt(metric['pose_success_0p1d_rate'])} |"
        )
    return "\n".join(lines)


def markdown_runtime_table(summary: dict[str, Any]) -> str:
    """Render per-scene runtime as a Markdown table."""
    if not summary or not summary.get("scene_count"):
        return "No runtime measured."

    def fmt(value: Any) -> str:
        """Render seconds to 2 dp, or n/a."""
        return "n/a" if value is None else f"{float(value):.2f}"

    return "\n".join(
        [
            "| Scenes | Mean (s) | Median (s) | Min (s) | Max (s) | Total (s) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {summary['scene_count']} | {fmt(summary['runtime_sec_mean'])} | "
            f"{fmt(summary['runtime_sec_median'])} | {fmt(summary['runtime_sec_min'])} | "
            f"{fmt(summary['runtime_sec_max'])} | {fmt(summary['runtime_sec_total'])} |",
        ]
    )


def write_runtime_csv(path: Path, overall: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write per-scene runtime plus the aggregate to CSV."""
    lines = ["scene_id,runtime_sec,depth_sec,scene_loop_sec,scoring_sec,overlay_sec"]
    for row in rows:
        lines.append(
            ",".join(
                [
                    f"{int(row['scene_id']):06d}",
                    f"{float(row['runtime_sec']):.4f}",
                    # `.get` with a default, not `row[...]`: rows from a run made before infer.py
                    # recorded the breakdown carry `runtime_sec` and nothing else, and a missing
                    # sub-timing should cost one column, not abort the evaluation with a KeyError.
                    f"{float(row.get('depth_sec', 0.0)):.4f}",
                    f"{float(row.get('scene_loop_sec', 0.0)):.4f}",
                    f"{float(row.get('scoring_sec', 0.0)):.4f}",
                    f"{float(row.get('overlay_sec', 0.0)):.4f}",
                ]
            )
        )
    lines.append("")
    lines.append(f"overall_mean,{overall['runtime_sec_mean']},,,")
    lines.append(f"overall_median,{overall['runtime_sec_median']},,,")
    lines.append(f"overall_min,{overall['runtime_sec_min']},,,")
    lines.append(f"overall_max,{overall['runtime_sec_max']},,,")
    lines.append(
        f"overall_total,{overall['runtime_sec_total']},,,{overall['scoring_sec_total']},"
        f"{overall.get('overlay_sec_total', 0.0)}"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_detection_table(summary: dict[str, Any]) -> str:
    """Render the overall detection summary as a Markdown table."""
    lines = ["| Kind | IoU | TP | FP | FN | Precision | Recall |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind, by_threshold in summary["overall"].items():
        for threshold, metric in by_threshold.items():
            lines.append(
                f"| {kind} | {threshold} | {metric['tp']} | {metric['fp']} | {metric['fn']} | "
                f"{metric['precision']:.3f} | {metric['recall']:.3f} |"
            )
    return "\n".join(lines)


def meets_label(required_rate: float, threshold_mm: float) -> str:
    """Column label for the pass-rate boolean, built from the values it is computed against.

    Derived rather than written out: the label used to hardcode `99%@5mm` while the numbers came
    from `pose.max_vertex_error_required_rate` and `pose.max_vertex_error_threshold_mm`, so editing
    the config silently left the report describing a bar it was no longer testing.
    """
    return f"Meets {required_rate:.0%}@{threshold_mm:g}mm"


def markdown_pose_table(summary: dict[str, Any], required_rate: float, threshold_mm: float) -> str:
    """Render the overall pose summary as a Markdown table."""
    metric = summary["overall"]
    lines = [
        "| Matched | Mean Trans (mm) | Median Trans (mm) | Mean ADD/ADD-S (mm) | Median ADD/ADD-S (mm) | "
        "Mean Frac Diameter | 10% Diameter Success | Mean Max-Vertex (mm) | P90 Max-Vertex (mm) | "
        f"P99 Max-Vertex (mm) | Max-Vertex <= {threshold_mm:g} mm | {meets_label(required_rate, threshold_mm)} |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {metric['matched_predictions']} | {metric['translation_error_mm_mean']} | "
        f"{metric['translation_error_mm_median']} | {metric['add_or_adds_mm_mean']} | "
        f"{metric['add_or_adds_mm_median']} | {metric['add_or_adds_diameter_frac_mean']} | "
        f"{metric['pose_success_0p1d_rate']} | {metric['max_vertex_error_mm_mean']} | "
        f"{metric['max_vertex_error_mm_p90']} | {metric['max_vertex_error_mm_p99']} | "
        f"{metric['max_vertex_error_within_threshold_rate']} | "
        f"{'yes' if metric['max_vertex_error_meets_required_rate'] else 'no'} |",
    ]
    return "\n".join(lines)


def markdown_depth_table(overall: dict[str, Any]) -> str:
    """Render the overall depth summary as a Markdown table."""

    def render_metric(value: Any) -> str:
        """Render nullable depth metrics for Markdown output."""
        return "n/a" if value is None else str(value)

    lines = [
        "| Scope | Valid Pixel Count | Depth MAE (mm) | Depth RMSE (mm) | Mean Scene Median Abs (mm) |",
        "| :--- | ---: | ---: | ---: | ---: |",
        f"| whole image | {overall['valid_count']} | {render_metric(overall['mae_mm'])} | "
        f"{render_metric(overall['rmse_mm'])} | {render_metric(overall['mean_scene_median_abs_mm'])} |",
        f"| **object pixels** | {render_metric(overall.get('object_valid_count'))} | "
        f"{render_metric(overall.get('object_mae_mm'))} | {render_metric(overall.get('object_rmse_mm'))} | "
        f"{render_metric(overall.get('mean_scene_object_median_abs_mm'))} |",
    ]
    return "\n".join(lines)


def write_report(
    *,
    output_dir: Path,
    config: PipelineReportConfig,
    scene_ids: list[int],
    generated_on: str,
    prompts_used: dict[str, str],
    raw_detection_summary: dict[str, Any],
    pose_input_detection_summary: dict[str, Any],
    filtered_detection_summary: dict[str, Any],
    pose_summary: dict[str, Any],
    depth_summary: dict[str, Any],
    runtime_summary: dict[str, Any],
) -> None:
    """Write the human-readable Markdown report for a full pipeline run."""
    prompts = "\n".join(
        f"- `{name}` -> `{prompt}`"
        for name, prompt in sorted(prompts_used.items())
    )
    if config.proposal_selection_policy == config.soft_global_rerank_policy:
        selection_notes = "\n".join(
            [
                f"- Proposal selection policy: `{config.proposal_selection_policy}`",
                f"- Keep rule: `R >= {config.rerank_cutoff}`",
                f"- Rerank formula: `{config.rerank_formula}`"
                + (
                    ""
                    if config.rerank_formula_from_run
                    else "  **(from the config profile, not from this run: inference recorded no "
                    "`rerank_formula`, so any `--rerank-weight-*` override is not reflected here.)**"
                ),
                "- The same rule is applied globally across all proposals; it is object-agnostic but its cutoff was tuned on one dataset's result set.",
            ]
        )
    else:
        selection_notes = "\n".join(
            [
                f"- Proposal selection policy: `{config.proposal_selection_policy}`",
                "- All SAM3 proposals are retained after FoundationPose diagnostics.",
            ]
        )
    if config.sam3_refinement_policy == config.replace_mid_nms06_refinement_policy:
        refinement_notes = "\n".join(
            [
                f"- SAM3 refinement policy: `{config.sam3_refinement_policy}`",
                f"- Replace proposals whose initial render-mask IoU falls in "
                f"`[{config.refinement_low_miou}, {config.refinement_high_miou}]`.",
                "- Replacement masks come from a second SAM3 pass prompted by the FoundationPose-rendered CAD box.",
                f"- The refined proposal set is then deduplicated with mask NMS at `{config.refinement_nms_threshold}` before the final FoundationPose/rerank pass.",
            ]
        )
    else:
        refinement_notes = "\n".join(
            [
                f"- SAM3 refinement policy: `{config.sam3_refinement_policy}`",
                "- Final pose/rerank operates directly on the raw SAM3 proposals.",
            ]
        )

    report = f"""# {config.dataset} Camera-{config.base_camera_id} Pipeline Report

Generated on {generated_on}.

Dataset scope:
- Dataset: `{config.dataset}`
- Scenes evaluated: `{len(scene_ids)}`
- Frame per scene: `rgb/000000.png` only
- SAM3 confidence threshold: `{config.confidence_threshold}`
- Depth source: `{config.depth_source}`
- Prompt source: `{config.prompt_source}`
- SAM3 refinement policy: `{config.sam3_refinement_policy}`
- Proposal selection policy: `{config.proposal_selection_policy}`

Prompts used:
{prompts}

## Raw SAM3 Detection
{markdown_detection_table(raw_detection_summary)}

## Post-Refinement Detection (pre-rerank)
{markdown_detection_table(pose_input_detection_summary)}

## Proposal-Selection Stage Detection
{markdown_detection_table(filtered_detection_summary)}

Ground truth for every table above is **occlusion-aware**: all scene objects are rasterized into
one shared z-buffer, so each GT mask holds only the pixels where that instance is the front-most
surface. GT instances visible below `{config.min_visible_fraction:.0%}` of their full silhouette are
excluded from both detection and pose scoring. Amodal (full-silhouette) GT would score a correctly
segmented but half-occluded object at ~0.45 IoU and record it as a miss.

Stage notes:
- `Raw SAM3` -> `Post-Refinement`: the SAM3 refinement stage *replaces* masks, so TP (and
  therefore recall) can rise here. With `--sam3-refinement-policy none` this table is identical
  to the raw one.
- `Post-Refinement` -> `Proposal-Selection`: the rerank only *removes* proposals, so TP and
  recall can only fall here; precision rises. Compare against the post-refinement row, not the
  raw row, to see what the rerank cutoff actually costs.

## Pose Error on Matched Pose Predictions
{markdown_pose_table(pose_summary, config.max_vertex_error_required_rate, config.max_vertex_error_threshold_mm)}

How to read this:
- `Max-Vertex (mm)` is the largest distance between any object vertex under the GT pose and the same vertex under the predicted pose (nearest-vertex, bidirectional, for symmetric objects).
- `Max-Vertex <= {config.max_vertex_error_threshold_mm:g} mm` is the fraction of matched predictions within that distance, and `{meets_label(config.max_vertex_error_required_rate, config.max_vertex_error_threshold_mm)}` is `yes` when that fraction reaches `{config.max_vertex_error_required_rate:.0%}`.
- Both come from `pose.max_vertex_error_threshold_mm` and `pose.max_vertex_error_required_rate` in the config profile. They are whatever tolerance you set, and the column labels above are rendered from the values this run actually used.

Notes:
- SAM3 refinement notes:
{refinement_notes}
- Proposal selection notes:
{selection_notes}
- FoundationPose render-overlap values are still logged per prediction for analysis.
- Pose matches are defined by selected-mask IoU >= `{config.pose_match_threshold}` for predictions that produced a pose estimate.
- ADD is used for non-symmetric objects; ADD-S is used for symmetric objects.

## Pose Error by GT Visibility
{markdown_pose_visibility_table(pose_summary.get('by_visibility', {}))}

Each band scores only matched instances at least that visible; `0` is every matched instance and
matches the pose table above. **Detection metrics and the GT population are identical across
bands** -- matching happens once, against all ground truth, and the band is applied afterwards
when choosing which matches get a pose measurement. Raising `--min-visible-fraction` instead
would delete occluded instances from ground truth, which turns every detection of one into a
false positive -- measured on an occluded capture, raising it to 0.9 more than halved precision
with the prediction count unchanged.

Comparing bands isolates how much pose error comes from partly-occluded instances: if the tail
shrinks sharply at 0.9, occlusion is driving it; if it does not, the failures are elsewhere.

## Depth Input Summary
{markdown_depth_table(depth_summary)}

## Per-Scene Runtime
{markdown_runtime_table(runtime_summary)}

Runtime is the **production path only** -- depth generation, SAM3, FoundationPose, refinement and
reranking. Two costs are excluded and reported separately, so what was removed stays visible:
scoring against ground truth (GT rasterization or cache load, IoU matrices, pose-error metrics),
which does not exist at deployment and whose GT rasterization alone is ~10 s per target; and
diagnostic overlay rendering. Per-scene values, including both excluded totals, are in
`runtime_summary.csv`.

Note the mean is inflated on short runs: FoundationPose builds its TensorRT engines during the
first scene, so scene 0 costs several times a steady-state scene. Prefer the median.

Depth notes:
{_depth_metrics_note(config)}
- `gt`: collected `{config.collected_depth_filename}` is used directly as FoundationPose input, so depth error metrics are `n/a`.
- `foundationstereo`: depth is predicted from the stereo pair and compared against that same collected map.
- **Read the object-pixel row, not the whole-image one.** Whole-image error is dominated by the mat, bin and floor -- large flat textured surfaces any stereo model matches easily -- and understates error on the parts by several fold. FoundationPose only ever consumes depth inside the SAM3 mask, so the object row is what predicts pose quality. Object pixels are those where a `scene_gt.json` pose is the front-most surface, using the same occlusion-aware masks the detection metrics score against.
- Object-pixel depth error reuses the per-scene z-buffer cache built by `script/build_gt_cache.py`; with `--no-gt-cache` the masks are rasterized instead, which roughly doubles GT cost.

## Artifacts
- Per-scene raw/kept overlays: [`pipeline/output/overlays`]({output_dir / 'overlays'})
- Per-target JSON and masks: [`pipeline/output/predictions`]({output_dir / 'predictions'})
- Per-scene depth metadata / outputs: [`pipeline/output/depth`]({output_dir / 'depth'})
- Machine-readable summaries: `raw_detection_summary.json`, `pose_input_detection_summary.json`, `filtered_detection_summary.json`, `pose_summary.json`, `depth_summary.json`, `runtime_summary.json`
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def write_pipeline_outputs(
    *,
    output_dir: Path,
    config: PipelineReportConfig,
    scene_ids: list[int],
    generated_on: str,
    prompts_used: dict[str, str],
    raw_detection_counts: dict[str, dict[str, Counter]],
    pose_input_detection_counts: dict[str, dict[str, Counter]],
    filtered_detection_counts: dict[str, dict[str, Counter]],
    raw_by_object: dict[str, dict[str, dict[str, Counter]]],
    pose_input_by_object: dict[str, dict[str, dict[str, Counter]]],
    filtered_by_object: dict[str, dict[str, dict[str, Counter]]],
    pose_matches_all: list[dict[str, Any]],
    pose_matches_by_object: dict[str, list[dict[str, Any]]],
    depth_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]] | None = None,
    pose_visibility_bands: list[float] | None = None,
) -> dict[str, Any]:
    """Build pipeline summaries, write all report artifacts, and return the summaries."""
    raw_detection_summary = {
        "overall": finalize_detection(raw_detection_counts),
        "by_object": {name: finalize_detection(value) for name, value in sorted(raw_by_object.items())},
    }
    pose_input_detection_summary = {
        "overall": finalize_detection(pose_input_detection_counts),
        "by_object": {name: finalize_detection(value) for name, value in sorted(pose_input_by_object.items())},
    }
    filtered_detection_summary = {
        "overall": finalize_detection(filtered_detection_counts),
        "by_object": {name: finalize_detection(value) for name, value in sorted(filtered_by_object.items())},
    }
    pose_summary = {
        "overall": summarize_pose_rows(
            pose_matches_all,
            pose_error_threshold_mm=config.max_vertex_error_threshold_mm,
            pose_error_required_rate=config.max_vertex_error_required_rate,
        ),
        "by_object": {
            name: summarize_pose_rows(
                rows,
                pose_error_threshold_mm=config.max_vertex_error_threshold_mm,
                pose_error_required_rate=config.max_vertex_error_required_rate,
            )
            for name, rows in sorted(pose_matches_by_object.items())
        },
    }
    # Pose metrics banded by GT visibility. `overall` stays the all-matches figure so existing
    # consumers (the multi-dataset aggregate) are unaffected.
    bands = sorted(set(pose_visibility_bands or [0.0]))
    pose_summary["by_visibility"] = {}
    for band in bands:
        kept = [row for row in pose_matches_all if float(row.get("gt_visible_fraction", 1.0)) >= band]
        kept_by_object = {
            name: [row for row in rows if float(row.get("gt_visible_fraction", 1.0)) >= band]
            for name, rows in sorted(pose_matches_by_object.items())
        }
        pose_summary["by_visibility"][f"{band:g}"] = {
            "min_visible_fraction": band,
            "matched_predictions_total": len(pose_matches_all),
            "matched_predictions_scored": len(kept),
            "overall": summarize_pose_rows(
                kept,
                pose_error_threshold_mm=config.max_vertex_error_threshold_mm,
                pose_error_required_rate=config.max_vertex_error_required_rate,
            ),
            "by_object": {
                name: summarize_pose_rows(
                    rows,
                    pose_error_threshold_mm=config.max_vertex_error_threshold_mm,
                    pose_error_required_rate=config.max_vertex_error_required_rate,
                )
                for name, rows in kept_by_object.items()
            },
        }

    depth_summary = {
        "overall": depth_overall_summary(depth_rows),
        "by_scene": depth_rows,
    }
    runtime_summary = summarize_runtime_rows(runtime_rows or [])

    (output_dir / "raw_detection_summary.json").write_text(
        json.dumps(raw_detection_summary, indent=2),
        encoding="utf-8",
    )
    write_detection_csv(output_dir / "raw_detection_summary.csv", raw_detection_summary)
    (output_dir / "pose_input_detection_summary.json").write_text(
        json.dumps(pose_input_detection_summary, indent=2),
        encoding="utf-8",
    )
    write_detection_csv(output_dir / "pose_input_detection_summary.csv", pose_input_detection_summary)
    (output_dir / "filtered_detection_summary.json").write_text(
        json.dumps(filtered_detection_summary, indent=2),
        encoding="utf-8",
    )
    write_detection_csv(output_dir / "filtered_detection_summary.csv", filtered_detection_summary)
    (output_dir / "selected_detection_summary.json").write_text(
        json.dumps(filtered_detection_summary, indent=2),
        encoding="utf-8",
    )
    write_detection_csv(output_dir / "selected_detection_summary.csv", filtered_detection_summary)
    (output_dir / "pose_summary.json").write_text(json.dumps(pose_summary, indent=2), encoding="utf-8")
    write_pose_csv(output_dir / "pose_summary.csv", pose_summary)
    (output_dir / "depth_summary.json").write_text(json.dumps(depth_summary, indent=2), encoding="utf-8")
    write_depth_csv(output_dir / "depth_summary.csv", depth_summary["overall"], depth_rows)
    (output_dir / "runtime_summary.json").write_text(
        json.dumps({"overall": runtime_summary, "by_scene": runtime_rows or []}, indent=2), encoding="utf-8"
    )
    if runtime_rows:
        write_runtime_csv(output_dir / "runtime_summary.csv", runtime_summary, runtime_rows)
    write_report(
        output_dir=output_dir,
        config=config,
        scene_ids=scene_ids,
        generated_on=generated_on,
        prompts_used=prompts_used,
        runtime_summary=runtime_summary,
        raw_detection_summary=raw_detection_summary,
        pose_input_detection_summary=pose_input_detection_summary,
        filtered_detection_summary=filtered_detection_summary,
        pose_summary=pose_summary,
        depth_summary=depth_summary["overall"],
    )
    return {
        "raw_detection_summary": raw_detection_summary,
        "pose_input_detection_summary": pose_input_detection_summary,
        "filtered_detection_summary": filtered_detection_summary,
        "selected_detection_summary": filtered_detection_summary,
        "pose_summary": pose_summary,
        "depth_summary": depth_summary,
        "runtime_summary": runtime_summary,
    }

def _depth_metrics_note(config: PipelineReportConfig) -> str:
    """State whether the depth table was measured at all, as the first depth note.

    Without this an `n/a` table is ambiguous between "no reference exists to compare against"
    and "compared, and there was nothing to report" -- and the second reading flatters the run.
    """
    if config.depth_metrics_enabled:
        return (
            "- Depth error **was measured**: predicted depth is scored against the collected "
            "cam0 ground-truth map, per scene."
        )
    return (
        "- Depth error was **not measured** (`--no-depth-metrics`): no collected ground-truth "
        "depth was read, so every error figure below is `n/a` because nothing was compared, "
        "not because the error was zero. Detection and pose metrics are unaffected -- those "
        "come from `scene_gt.json` and the GT cache, which this flag does not touch."
    )


def _depth_source_note(depth_source: str) -> str:
    """Describe where depth actually came from, for the report header.

    The note is derived from `--depth-source` rather than hardcoded, so the header cannot claim
    ground-truth depth for a run that predicted its own.
    """
    if depth_source == "gt":
        return (
            "read from collected ground-truth depth at "
            "`<collected_depth_root>/<dataset>/<split>/<scene>/scene_cam<n>_depth.png`"
        )
    if depth_source == "foundationstereo":
        return (
            "predicted per scene by FoundationStereo from the rectified stereo pair; the "
            "partner camera, model and input normalisation used are recorded in each scene's "
            "`depth/<scene>/metadata.json`"
        )
    return "see each scene's `depth/<scene>/metadata.json` for provenance"


class MultiDatasetReportGenerator:
    """Build and write the aggregate report across many per-dataset pipeline runs."""

    def __init__(self, config: MultiDatasetReportConfig) -> None:
        """Store report configuration.

        No pose-metric registry is needed: `max_vertex_error_mm` is carried forward from each
        target's `matched_pose_metrics` rather than recomputed here. Recomputing would require
        re-deriving the GT entry from `gt_index`, which only addresses the occlusion-aware GT
        list built by `evaluation.gt.render_gt_entries` -- see the comment in `aggregate_outputs`.
        """
        self.config = config

    @staticmethod
    def _load_results_rows(path: Path) -> list[dict[str, Any]]:
        """Load newline-delimited JSON target records from one pipeline run."""
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def aggregate_outputs(self, run_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge per-dataset pipeline artifacts into one aggregate summary object."""
        run_rows_by_dataset = {row["dataset"]: row for row in run_rows}
        aggregate_detection_counts: dict[str, dict[str, dict[str, int]]] | None = None
        all_pose_rows: list[dict[str, Any]] = []
        prompt_map: dict[str, str] = {}
        per_dataset_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        pending: list[str] = []
        total_scenes = 0
        total_targets = 0
        total_runtime_seconds = 0.0
        for dataset in self.config.datasets:
            dataset_output_dir = self.config.output_root / dataset
            report_path = dataset_output_dir / "report.md"
            filtered_summary_path = dataset_output_dir / "filtered_detection_summary.json"
            # `evaluations.jsonl` is what proves the dataset was evaluated -- the report alone
            # does not, since a failed run can leave a partial one behind.
            results_path = dataset_output_dir / "evaluations.jsonl"
            if not (report_path.exists() and filtered_summary_path.exists() and results_path.exists()):
                run_row = run_rows_by_dataset.get(dataset)
                if run_row is None:
                    pending.append(dataset)
                else:
                    failure_row = {
                        "dataset": dataset,
                        "status": "missing_outputs",
                    }
                    failure_row |= run_row
                    failures.append(failure_row)
                continue

            filtered_summary = json.loads(filtered_summary_path.read_text(encoding="utf-8"))
            counts = detection_counts_from_summary(filtered_summary)
            if aggregate_detection_counts is None:
                aggregate_detection_counts = counts
            else:
                for kind, by_threshold in counts.items():
                    for threshold, metric in by_threshold.items():
                        aggregate_detection_counts[kind][threshold]["tp"] += int(metric["tp"])
                        aggregate_detection_counts[kind][threshold]["fp"] += int(metric["fp"])
                        aggregate_detection_counts[kind][threshold]["fn"] += int(metric["fn"])

            dataset_results = self._load_results_rows(results_path)
            dataset_pose_rows: list[dict[str, Any]] = []
            depth_summary: dict[str, Any] | None = None
            for record in dataset_results:
                total_targets += 1
                prompt_map[str(record["object_name"])] = str(record["prompt"])
                # `max_vertex_error_mm` is taken straight from the per-target record. It must NOT
                # be recomputed here by re-deriving the GT entry from `gt_index`: that index
                # addresses the occlusion-aware GT list built by `render_gt_entries`, which drops
                # instances below `min_visible_fraction`. Any independent GT lookup that filters
                # differently -- e.g. on `scene_gt_info.json`'s `visib_fract`, which is a
                # placeholder reading 1.0 even for heavily occluded parts -- keeps those dropped
                # instances, shifting every subsequent index and pairing predictions with the
                # wrong object. That produced aggregate P99 values of 447 mm on one dataset (true 7.87)
                # while datasets with nothing dropped stayed correct.
                for pose_row in record.get("matched_pose_metrics", []):
                    merged = dict(pose_row)
                    merged["dataset"] = dataset
                    merged["object_name"] = str(record["object_name"])
                    dataset_pose_rows.append(merged)
                    all_pose_rows.append(merged)

            depth_summary_path = dataset_output_dir / "depth_summary.json"
            if depth_summary_path.exists():
                depth_summary = json.loads(depth_summary_path.read_text(encoding="utf-8"))
                total_scenes += int(depth_summary["overall"]["scene_count"])

            # Per-scene runtime comes from the dataset's own measurement rather than dividing
            # the subprocess wall time by scene count: that figure already excludes GT scoring
            # and overlay rendering, and the median avoids the first scene's one-off
            # FoundationPose engine build, which on a 34-scene run is ~7x a steady-state scene.
            runtime_median_per_scene = None
            runtime_summary_path = dataset_output_dir / "runtime_summary.json"
            if runtime_summary_path.exists():
                try:
                    runtime_overall = json.loads(runtime_summary_path.read_text(encoding="utf-8"))["overall"]
                    runtime_median_per_scene = runtime_overall.get("runtime_sec_median")
                except (OSError, json.JSONDecodeError, KeyError):
                    runtime_median_per_scene = None

            pose_summary = summarize_pose_rows(
                dataset_pose_rows,
                pose_error_threshold_mm=self.config.pose_error_threshold_mm,
                pose_error_required_rate=self.config.pose_error_required_rate,
            )
            summary_05_box = filtered_summary["overall"]["box"]["0.5"]
            summary_05_mask = filtered_summary["overall"]["mask"]["0.5"]
            run_row = run_rows_by_dataset.get(dataset, {})
            if run_row.get("runtime_seconds") is not None:
                total_runtime_seconds += float(run_row["runtime_seconds"])
            per_dataset_rows.append(
                {
                    "dataset": dataset,
                    "scene_count": int(depth_summary["overall"]["scene_count"]) if depth_summary is not None else None,
                    "target_count": len(dataset_results),
                    "mask_precision_05": float(summary_05_mask["precision"]),
                    "mask_recall_05": float(summary_05_mask["recall"]),
                    "box_precision_05": float(summary_05_box["precision"]),
                    "box_recall_05": float(summary_05_box["recall"]),
                    "matched_predictions": int(pose_summary["matched_predictions"]),
                    "translation_error_mm_mean": pose_summary["translation_error_mm_mean"],
                    "add_or_adds_mm_mean": pose_summary["add_or_adds_mm_mean"],
                    "translation_error_le_3mm_rate": pose_summary["translation_error_le_3mm_rate"],
                    "add_or_adds_le_3mm_rate": pose_summary["add_or_adds_le_3mm_rate"],
                    "pose_success_0p1d_rate": pose_summary["pose_success_0p1d_rate"],
                    "max_vertex_error_mm_mean": pose_summary["max_vertex_error_mm_mean"],
                    "max_vertex_error_mm_median": pose_summary["max_vertex_error_mm_median"],
                    "max_vertex_error_mm_p90": pose_summary["max_vertex_error_mm_p90"],
                    "max_vertex_error_mm_p99": pose_summary["max_vertex_error_mm_p99"],
                    "max_vertex_error_within_threshold_rate": pose_summary["max_vertex_error_within_threshold_rate"],
                    "max_vertex_error_meets_required_rate": pose_summary["max_vertex_error_meets_required_rate"],
                    "runtime_seconds": run_row.get("runtime_seconds"),
                    "runtime_seconds_median_per_scene": runtime_median_per_scene,
                    "status": "ok",
                }
            )

        detection_summary = finalize_detection_counts(aggregate_detection_counts or {})
        pose_summary = summarize_pose_rows(
            all_pose_rows,
            pose_error_threshold_mm=self.config.pose_error_threshold_mm,
            pose_error_required_rate=self.config.pose_error_required_rate,
        )
        # Bands are recomputed from the pooled raw matches, not averaged from the per-dataset
        # summaries. Percentiles do not average: a mean of 32 P99s is not the P99 of the pooled
        # population, and with the failures concentrated in a few datasets the two differ a lot.
        pose_by_visibility = {}
        for band in sorted(set(self.config.pose_min_visible_fractions or [0.0])):
            kept = [row for row in all_pose_rows if float(row.get("gt_visible_fraction", 1.0)) >= band]
            pose_by_visibility[f"{band:g}"] = {
                "min_visible_fraction": band,
                "matched_predictions_total": len(all_pose_rows),
                "matched_predictions_scored": len(kept),
                "overall": summarize_pose_rows(
                    kept,
                    pose_error_threshold_mm=self.config.pose_error_threshold_mm,
                    pose_error_required_rate=self.config.pose_error_required_rate,
                ),
            }
        return {
            "pose_by_visibility": pose_by_visibility,
            "generated_on": self.config.generated_on,
            "dataset_count_requested": len(self.config.datasets),
            "dataset_count_completed": len(per_dataset_rows),
            "dataset_count_failed": len(failures),
            "dataset_count_pending": len(pending),
            "scene_count_total": total_scenes,
            "target_count_total": total_targets,
            "runtime_seconds_total_known": total_runtime_seconds,
            "pipeline": {
                # Depth first, and SAM3 second, because SAM3's scene state is built at the
                # depth map's resolution -- the order is forced by the data, not a convention.
                # This list is emitted into every batch report, so a stale one contradicts the
                # stage flow the rest of the documentation describes.
                "stages": [
                    "depth (predicted from the stereo pair)",
                    "SAM3",
                    "FoundationPose initial pass",
                    "CAD box-prompt SAM3 refinement replace_mid_nms06",
                    "FoundationPose final pass",
                    "soft_global_v1 proposal reranker",
                ],
                "depth_source": self.config.depth_source,
                "prompt_source": self.config.prompt_source,
                "confidence_threshold": self.config.confidence_threshold,
                "resolution": self.config.resolution,
                "sam3_refinement_policy": self.config.sam3_refinement_policy,
                "proposal_selection_policy": self.config.proposal_selection_policy,
                "refinement_replace_miou_low": self.config.refinement_replace_miou_low,
                "refinement_replace_miou_high": self.config.refinement_replace_miou_high,
                "refinement_nms_threshold": self.config.refinement_nms_threshold,
                "rerank_formula": self.config.rerank_formula,
                "rerank_cutoff": self.config.rerank_cutoff,
                "fp_prepare_batch": self.config.fp_prepare_batch,
                "fp_n_hypotheses": self.config.fp_n_hypotheses,
                "fp_n_refine": self.config.fp_n_refine,
                "pose_error_metric": "symmetry_aware_max_vertex_distance_mm",
                "pose_error_threshold_mm": self.config.pose_error_threshold_mm,
                "pose_error_required_rate": self.config.pose_error_required_rate,
            },
            "prompts": dict(sorted(prompt_map.items())),
            "selected_detection_summary": detection_summary,
            "pose_summary": pose_summary,
            "per_dataset": per_dataset_rows,
            "failures": failures,
            "pending": pending,
            "runs": run_rows,
        }

    def write_report(self, aggregate: dict[str, Any]) -> None:
        """Write the aggregate Markdown report across all evaluated datasets."""

        def fmt(value: Any, digits: int = 3) -> str:
            """Format nullable numeric values for Markdown tables."""
            if value is None:
                return "n/a"
            if isinstance(value, float):
                if math.isnan(value):
                    return "n/a"
                return f"{value:.{digits}f}"
            return str(value)

        prompts_lines = [
            f"- `{name}` -> `{prompt}`"
            for name, prompt in aggregate["prompts"].items()
        ]
        detection = aggregate["selected_detection_summary"]
        pose = aggregate["pose_summary"]

        lines = [
            "# Multi-Dataset Base-Camera Evaluation",
            "",
            f"Generated on {aggregate['generated_on']}.",
            "",
            "## Scope",
            f"- Requested datasets: `{aggregate['dataset_count_requested']}`",
            f"- Completed datasets: `{aggregate['dataset_count_completed']}`",
            f"- Failed datasets: `{aggregate['dataset_count_failed']}`",
            f"- Pending datasets: `{aggregate['dataset_count_pending']}`",
            f"- Total scenes evaluated: `{aggregate['scene_count_total']}`",
            f"- Total targets evaluated: `{aggregate['target_count_total']}`",
            "",
            "## Pipeline",
            f"- Stage sequence: `{' -> '.join(aggregate['pipeline']['stages'])}`",
            f"- Depth source: `{aggregate['pipeline']['depth_source']}` -- {_depth_source_note(aggregate['pipeline']['depth_source'])}",
            f"- Prompt source: `{aggregate['pipeline']['prompt_source']}`",
            f"- SAM3 confidence threshold: `{aggregate['pipeline']['confidence_threshold']}`",
            f"- SAM3 resolution: `{aggregate['pipeline']['resolution']}`",
            f"- SAM3 refinement policy: `{aggregate['pipeline']['sam3_refinement_policy']}`",
            f"- Refinement replace render-mask IoU range: `[{aggregate['pipeline']['refinement_replace_miou_low']}, {aggregate['pipeline']['refinement_replace_miou_high']}]`",
            f"- Refinement mask NMS threshold: `{aggregate['pipeline']['refinement_nms_threshold']}`",
            f"- Proposal selection policy: `{aggregate['pipeline']['proposal_selection_policy']}`",
            f"- Rerank formula: `{aggregate['pipeline']['rerank_formula']}`",
            f"- Rerank cutoff: `{aggregate['pipeline']['rerank_cutoff']}`",
            f"- FoundationPose prepare batch: `{aggregate['pipeline']['fp_prepare_batch']}`",
            f"- FoundationPose hypotheses: `{aggregate['pipeline']['fp_n_hypotheses']}`",
            f"- FoundationPose refine iterations: `{aggregate['pipeline']['fp_n_refine']}`",
            f"- Pose error criterion: `>= {aggregate['pipeline']['pose_error_required_rate']:.0%}` of matched predictions must have max-vertex error `<= {aggregate['pipeline']['pose_error_threshold_mm']} mm`",
            "- For non-symmetric objects, max-vertex error is the maximum Euclidean distance over corresponding CAD vertices transformed by the GT and predicted poses.",
            "- For symmetric objects, max-vertex error is the bidirectional nearest-neighbor maximum distance between the GT-transformed and predicted-transformed CAD vertex sets.",
            "",
            "Prompts used:",
            *(prompts_lines or ["- none"]),
            "",
            "## Overall Detection",
            "| Kind | IoU | TP | FP | FN | Precision | Recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for kind, by_threshold in detection.items():
            for threshold, metric in by_threshold.items():
                lines.append(
                    f"| {kind} | {threshold} | {metric['tp']} | {metric['fp']} | {metric['fn']} | "
                    f"{fmt(metric['precision'])} | {fmt(metric['recall'])} |"
                )

        lines.extend(
            [
                "",
                "## Overall Pose",
                f"| Matched | Mean Trans (mm) | Median Trans (mm) | Mean ADD/ADD-S (mm) | Median ADD/ADD-S (mm) | Mean Max-Vertex (mm) | Median Max-Vertex (mm) | P90 Max-Vertex (mm) | P99 Max-Vertex (mm) | Max-Vertex <= {self.config.pose_error_threshold_mm:g} mm | {meets_label(self.config.pose_error_required_rate, self.config.pose_error_threshold_mm)} |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {pose['matched_predictions']} | {fmt(pose['translation_error_mm_mean'])} | {fmt(pose['translation_error_mm_median'])} | "
                f"{fmt(pose['add_or_adds_mm_mean'])} | {fmt(pose['add_or_adds_mm_median'])} | {fmt(pose['max_vertex_error_mm_mean'])} | "
                f"{fmt(pose['max_vertex_error_mm_median'])} | {fmt(pose['max_vertex_error_mm_p90'])} | "
                f"{fmt(pose['max_vertex_error_mm_p99'])} | {fmt(pose['max_vertex_error_within_threshold_rate'])} | "
                f"{'yes' if pose['max_vertex_error_meets_required_rate'] else 'no'} |",
                "",
                "P90 is reported beside P99 because the two answer different questions at this "
                "sample size. P99 over a pooled population is decided by the worst ~1% of "
                "matches, so a handful of gross failures moves it by tens of millimetres while "
                "the bulk of the distribution is unchanged; P90 shows whether the bulk moved.",
                "",
                "## Overall Pose by GT Visibility",
                "| Min GT Visibility | Scored / Matched | Median Max-Vertex (mm) | P90 (mm) | P99 (mm) | Max-Vertex <= 5 mm | Mean ADD/ADD-S (mm) | 10% Diameter Success |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for band, entry in sorted(
            aggregate.get("pose_by_visibility", {}).items(), key=lambda kv: float(kv[0])
        ):
            metric = entry["overall"]
            lines.append(
                f"| {band} | {entry['matched_predictions_scored']} / {entry['matched_predictions_total']} | "
                f"{fmt(metric['max_vertex_error_mm_median'])} | {fmt(metric['max_vertex_error_mm_p90'])} | "
                f"{fmt(metric['max_vertex_error_mm_p99'])} | {fmt(metric['max_vertex_error_within_threshold_rate'])} | "
                f"{fmt(metric['add_or_adds_mm_mean'])} | {fmt(metric['pose_success_0p1d_rate'])} |"
            )
        lines.extend(
            [
                "",
                "Each band scores only matched instances at least that visible, pooled across all "
                "datasets. Detection metrics and the GT population are identical across bands: "
                "matching runs once against all ground truth and the band is applied afterwards, "
                "when choosing which matches get a pose measurement. Percentiles here are "
                "recomputed from the pooled matches, not averaged across per-dataset summaries.",
                "",
                "## Per-Dataset Summary",
                f"| Dataset | Scenes | Targets | Mask P@0.5 | Mask R@0.5 | Box P@0.5 | Box R@0.5 | Mean Trans (mm) | Mean ADD/ADD-S (mm) | P90 Max-Vertex (mm) | P99 Max-Vertex (mm) | Max-Vertex <= {self.config.pose_error_threshold_mm:g} mm | {meets_label(self.config.pose_error_required_rate, self.config.pose_error_threshold_mm)} | Median Runtime/Scene (s) | Runtime (s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in aggregate["per_dataset"]:
            lines.append(
                f"| {row['dataset']} | {row['scene_count']} | {row['target_count']} | {fmt(row['mask_precision_05'])} | {fmt(row['mask_recall_05'])} | "
                f"{fmt(row['box_precision_05'])} | {fmt(row['box_recall_05'])} | {fmt(row['translation_error_mm_mean'])} | "
                f"{fmt(row['add_or_adds_mm_mean'])} | {fmt(row.get('max_vertex_error_mm_p90'))} | "
                f"{fmt(row['max_vertex_error_mm_p99'])} | {fmt(row['max_vertex_error_within_threshold_rate'])} | "
                f"{'yes' if row['max_vertex_error_meets_required_rate'] else 'no'} | "
                f"{fmt(row.get('runtime_seconds_median_per_scene'), 2)} | {fmt(row['runtime_seconds'], 1)} |"
            )

        if aggregate["failures"]:
            lines.extend(
                [
                    "",
                    "## Failures",
                    "| Dataset | Status | Return Code | Output Dir | Error Preview |",
                    "| --- | --- | ---: | --- | --- |",
                ]
            )
            for row in aggregate["failures"]:
                lines.append(
                    f"| {row.get('dataset')} | {row.get('status')} | {row.get('returncode', '')} | "
                    f"{row.get('output_dir', '')} | {str(row.get('error_preview', '')).replace('|', '/')} |"
                )

        if aggregate["pending"]:
            lines.extend(
                [
                    "",
                    "## Pending",
                    ", ".join(f"`{name}`" for name in aggregate["pending"]),
                ]
            )

        lines.extend(
            [
                "",
                "## Artifacts",
                f"- Aggregate summary JSON: [`summary.json`]({self.config.output_root / 'summary.json'})",
                f"- Per-dataset CSV: [`per_dataset.csv`]({self.config.output_root / 'per_dataset.csv'})",
                f"- Run log JSONL: [`run_status.jsonl`]({self.config.output_root / 'run_status.jsonl'})",
                f"- Per-dataset outputs: [`{self.config.output_root.name}`]({self.config.output_root})",
            ]
        )
        (self.config.output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_per_dataset_csv(self, rows: list[dict[str, Any]]) -> None:
        """Write the per-dataset aggregate metrics table to CSV."""
        fieldnames = [
            "dataset",
            "scene_count",
            "target_count",
            "mask_precision_05",
            "mask_recall_05",
            "box_precision_05",
            "box_recall_05",
            "matched_predictions",
            "translation_error_mm_mean",
            "add_or_adds_mm_mean",
            "translation_error_le_3mm_rate",
            "add_or_adds_le_3mm_rate",
            "pose_success_0p1d_rate",
            "max_vertex_error_mm_mean",
            "max_vertex_error_mm_median",
            "max_vertex_error_mm_p90",
            "max_vertex_error_mm_p99",
            "max_vertex_error_within_threshold_rate",
            "max_vertex_error_meets_required_rate",
            "runtime_seconds_median_per_scene",
            "runtime_seconds",
            "status",
        ]
        with (self.config.output_root / "per_dataset.csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def write_outputs(self, aggregate: dict[str, Any], run_rows: list[dict[str, Any]]) -> None:
        """Write all aggregate JSON, CSV, JSONL, and Markdown output artifacts."""
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        (self.config.output_root / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        with (self.config.output_root / "run_status.jsonl").open("w", encoding="utf-8") as f:
            for row in run_rows:
                f.write(json.dumps(row) + "\n")
        self.write_per_dataset_csv(aggregate["per_dataset"])
        self.write_report(aggregate)
