#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline sweep of the soft-global rerank cutoff over saved pipeline results.

The rerank is a pure post-hoc threshold on `rerank_score`, which the pipeline already stores
per proposal in `predictions.jsonl`. That means the whole precision/recall curve can be
recovered from a single completed run -- no GPU, no re-running SAM3 or FoundationPose.

Detection is scored at the post-refinement (pre-rerank) proposal set, so the numbers here
isolate exactly what the cutoff costs, independent of what SAM3 refinement gained.

Usage:
    python tools/sweep_rerank_cutoff.py --config <name> --results-root output/<batch run>
    python tools/sweep_rerank_cutoff.py --config <name> --results-root output --datasets <name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perception_pipeline.config import (
    DEFAULT_CUTOFFS,
    DEFAULT_RERANK_CUTOFF,
    add_config_argument,
    settings_from_argv,
)
from perception_pipeline.geometry import (
    bbox_from_mask,
    box_iou,
    greedy_match,
    read_binary_little_endian_ply,
    render_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CUTOFF = DEFAULT_RERANK_CUTOFF


def parse_args() -> argparse.Namespace:
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--dataset-root", type=Path, default=settings.dataset.root)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=settings.dataset.batch_output_root,
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models-subdir", default="models_cad")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--cutoffs", type=float, nargs="*", default=DEFAULT_CUTOFFS)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=settings.dataset.output_root / "rerank_cutoff_sweep.json",
    )
    return parser.parse_args()


def gt_boxes_for_scene(
    dataset_dir: Path, scene_id: int, obj_id: int, meshes: dict[int, tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    scene_dir = dataset_dir / "test" / f"{scene_id:06d}"
    scene_gt = json.loads((scene_dir / "scene_gt.json").read_text(encoding="utf-8"))["0"]
    scene_camera = json.loads((scene_dir / "scene_camera.json").read_text(encoding="utf-8"))["0"]
    camera_matrix = np.asarray(scene_camera["cam_K"], dtype=np.float64).reshape(3, 3)
    image = cv2.imread(str(scene_dir / "rgb" / "000000.png"))
    width, height = image.shape[1], image.shape[0]
    vertices, faces = meshes[obj_id]
    boxes = []
    for entry in scene_gt:
        if int(entry["obj_id"]) != obj_id:
            continue
        rotation = np.asarray(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(entry["cam_t_m2c"], dtype=np.float64)
        mask = render_mask(vertices, faces, rotation, translation, camera_matrix, (width, height))
        if mask.any():
            boxes.append(bbox_from_mask(mask))
    return np.asarray(boxes, dtype=np.float64)


def sweep_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any] | None:
    # Reads the INFERENCE artifact: proposal boxes and rerank scores are produced by the
    # selection stage and owe nothing to ground truth, which is exactly why this sweep can
    # recover the whole precision/recall curve without a GPU.
    results_path = args.results_root / dataset / "predictions.jsonl"
    if not results_path.exists():
        return None
    dataset_dir = args.dataset_root / dataset
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scenes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    gt_total = 0
    for line in results_path.open(encoding="utf-8"):
        record = json.loads(line)
        obj_id = record["obj_id"]
        if obj_id not in meshes:
            vertices, faces = read_binary_little_endian_ply(
                dataset_dir / args.models_subdir / f"obj_{obj_id:06d}.ply"
            )
            meshes[obj_id] = (np.asarray(vertices, dtype=np.float64), np.asarray(faces))
        gt = gt_boxes_for_scene(dataset_dir, record["scene_id"], obj_id, meshes)
        gt_total += len(gt)
        boxes = np.asarray(record["pose_input_boxes_xyxy"], dtype=np.float64)
        scores = np.full(len(boxes), -np.inf)
        # `predictions.jsonl` carries these as `proposals`, keyed on `pose_input_index`.
        for entry in record.get("proposals", []):
            index = entry.get("pose_input_index")
            if index is not None and index < len(scores) and entry.get("rerank_score") is not None:
                scores[index] = entry["rerank_score"]
        # Precompute the IoU matrix once; the cutoff only changes which rows survive.
        iou = np.zeros((len(boxes), len(gt)))
        for i, box in enumerate(boxes):
            for j, gt_box in enumerate(gt):
                iou[i, j] = box_iou(box, gt_box)
        scenes.append((iou, scores, gt))

    rows = []
    for cutoff in args.cutoffs:
        tp = 0
        pred_count = 0
        for iou, scores, gt in scenes:
            keep = np.nonzero(scores >= cutoff)[0]
            pred_count += len(keep)
            if len(keep) == 0 or len(gt) == 0:
                continue
            tp += greedy_match(iou[keep], args.iou_threshold)["tp"]
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gt_total if gt_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # F2 weights recall 2x -- for bin picking a missed part usually costs more than a spurious one.
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "cutoff": cutoff,
                "predictions": pred_count,
                "tp": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
            }
        )
    return {"dataset": dataset, "gt_instances": gt_total, "rows": rows}


def main() -> None:
    args = parse_args()
    datasets = args.datasets or sorted(
        path.name
        for path in args.results_root.glob(settings_from_argv().dataset.glob)
        if (path / "predictions.jsonl").exists()
    )
    if not datasets:
        raise SystemExit(f"No datasets with predictions.jsonl under {args.results_root}")

    all_results = []
    print(f"{'ds':>7} {'GT':>5} {'best F1 @':>10} {'F1':>6} {'best F2 @':>10} {'F2':>6}"
          f" {'cur R':>6} {'cur F1':>7} {'dR@2.5':>7}")
    for dataset in datasets:
        result = sweep_dataset(dataset, args)
        if result is None:
            print(f"{dataset:>7}  (no predictions.jsonl)")
            continue
        all_results.append(result)
        rows = result["rows"]
        best_f1 = max(rows, key=lambda r: r["f1"])
        best_f2 = max(rows, key=lambda r: r["f2"])
        cur = next(r for r in rows if abs(r["cutoff"] - CURRENT_CUTOFF) < 1e-9)
        at25 = next((r for r in rows if abs(r["cutoff"] - 2.5) < 1e-9), None)
        drecall = (at25["recall"] - cur["recall"]) if at25 else float("nan")
        print(
            f"{dataset:>7} {result['gt_instances']:>5} {best_f1['cutoff']:>10.2f} {best_f1['f1']:>6.3f}"
            f" {best_f2['cutoff']:>10.2f} {best_f2['f2']:>6.3f} {cur['recall']:>6.3f} {cur['f1']:>7.3f}"
            f" {drecall:>+7.3f}",
            flush=True,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output_json}")

    # Aggregate over every dataset: which single global cutoff maximises pooled F1/F2?
    print("\n=== pooled over all datasets ===")
    print(f"{'cutoff':>10} {'preds':>7} {'TP':>6} {'precision':>10} {'recall':>8} {'F1':>7} {'F2':>7}")
    gt_all = sum(r["gt_instances"] for r in all_results)
    pooled = []
    for index, cutoff in enumerate(args.cutoffs):
        tp = sum(r["rows"][index]["tp"] for r in all_results)
        preds = sum(r["rows"][index]["predictions"] for r in all_results)
        precision = tp / preds if preds else 0.0
        recall = tp / gt_all if gt_all else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        pooled.append((cutoff, f1, f2))
        tag = "  <- current" if abs(cutoff - CURRENT_CUTOFF) < 1e-9 else ("  <- no rerank" if cutoff < -1e8 else "")
        print(f"{cutoff:>10.2f} {preds:>7} {tp:>6} {precision:>10.3f} {recall:>8.3f} {f1:>7.3f} {f2:>7.3f}{tag}")
    print(f"\npooled best F1 @ {max(pooled, key=lambda p: p[1])[0]:.2f}"
          f"   pooled best F2 @ {max(pooled, key=lambda p: p[2])[0]:.2f}")


if __name__ == "__main__":
    main()
