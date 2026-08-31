#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlay rendering for qualitative inspection of pipeline outputs.

Colour conventions in `draw_pose_filter_overlay`: green = proposal kept by the rerank,
red = dropped, cyan = the CAD silhouette box rendered at FoundationPose's estimated pose.
Cyan therefore answers "where does FP think the object is", not "what was detected".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from perception_pipeline.pose import PoseFilterResult


def color_for_index(index: int) -> tuple[int, int, int]:
    """Return a stable visualization color for a detection or mask index."""
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
    ]
    return palette[index % len(palette)]


def draw_overlay(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray | None,
    prompt: str,
    gt_box: list[float] | None,
    output_path: Path,
) -> None:
    """Render predicted masks, boxes, prompt text, and optional GT box to an image."""
    overlay = image.convert("RGBA")
    mask_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    if masks is not None:
        for i, mask in enumerate(masks):
            color = color_for_index(i)
            rgba = Image.new("RGBA", overlay.size, (*color, 72))
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
            mask_layer = Image.composite(rgba, mask_layer, mask_img)
    overlay = Image.alpha_composite(overlay, mask_layer)
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = None

    # strict=True: these are parallel arrays from a single detection pass, so a length mismatch
    # is a bug upstream. zip's default would silently draw the shorter of the two.
    for i, (box, score) in enumerate(zip(boxes, scores, strict=True)):
        color = color_for_index(i)
        draw.rectangle(list(map(float, box)), outline=(*color, 255), width=4)
        draw.text((float(box[0]), max(0.0, float(box[1]) - 22)), f"{score:.2f}", fill=(*color, 255), font=font)
    if gt_box is not None:
        draw.rectangle(gt_box, outline=(255, 255, 255, 255), width=5)
        draw.rectangle(gt_box, outline=(0, 0, 0, 255), width=2)
    draw.text((10, 10), prompt, fill=(255, 255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(output_path)


def draw_pose_filter_overlay(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    prompt: str,
    filter_results: list[PoseFilterResult],
    path: Path,
) -> None:
    """Render proposal masks, keep/drop labels, and rendered CAD boxes to an overlay."""
    import cv2

    overlay = np.asarray(image).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for result in filter_results:
        box = boxes[result.pred_index]
        mask = masks[result.pred_index]
        color = np.array([60, 220, 90] if result.kept else [220, 70, 70], dtype=np.uint8)
        overlay[mask] = (
            0.65 * overlay[mask].astype(np.float32) + 0.35 * color.astype(np.float32)
        ).astype(np.uint8)
        x0, y0, x1, y1 = [round(value) for value in box]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color.tolist(), 3)
        status = "keep" if result.kept else "drop"
        fp_score_text = f" fp={result.fp_score:.2f}" if result.fp_score is not None else ""
        label = (
            f"{status} s={scores[result.pred_index]:.2f}"
            f" miou={result.proposal_render_mask_iou:.2f}"
            f" biou={result.proposal_render_box_iou:.2f}"
            f"{fp_score_text}"
        )
        cv2.putText(
            overlay,
            label,
            (x0, max(24, y0 - 8)),
            font,
            0.85,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )
        if result.render_box_xyxy is not None:
            rx0, ry0, rx1, ry1 = [round(value) for value in result.render_box_xyxy]
            cv2.rectangle(overlay, (rx0, ry0), (rx1, ry1), [0, 255, 255], 2)

    cv2.putText(
        overlay,
        prompt,
        (32, 48),
        font,
        1.1,
        [255, 255, 255],
        3,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def xyxy_to_norm_cxcywh(box: list[float], image_size: tuple[int, int]) -> list[float]:
    """Convert a pixel-space XYXY box into normalized center-width-height form."""
    x0, y0, x1, y1 = box
    width, height = image_size
    return [
        ((x0 + x1) / 2) / width,
        ((y0 + y1) / 2) / height,
        max(1.0, x1 - x0) / width,
        max(1.0, y1 - y0) / height,
    ]


def draw_scene_overlay(
    *,
    image: Image.Image,
    predictions: list[dict[str, Any]],
    output_path: Path,
    kept_only: bool,
) -> None:
    """Render a scene-level overlay showing all raw or kept predictions together."""
    overlay = image.convert("RGBA")
    mask_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    global_index = 0
    for pred in predictions:
        if kept_only:
            masks = pred.get("pose_input_masks", pred["masks"])
            boxes = pred.get("pose_input_boxes", pred["boxes"])
            scores = pred.get("pose_input_scores", pred["scores"])
            indices = pred["kept_indices"]
        else:
            masks = pred["masks"]
            boxes = pred["boxes"]
            scores = pred["scores"]
            indices = list(range(len(boxes)))
        for local_idx in indices:
            color = color_for_index(global_index)
            rgba = Image.new("RGBA", overlay.size, (*color, 72))
            mask_img = Image.fromarray((masks[local_idx] > 0).astype(np.uint8) * 255, mode="L")
            mask_layer = Image.composite(rgba, mask_layer, mask_img)
            global_index += 1
    overlay = Image.alpha_composite(overlay, mask_layer)
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = None

    global_index = 0
    legend_lines = []
    for pred in predictions:
        if kept_only:
            boxes = pred.get("pose_input_boxes", pred["boxes"])
            scores = pred.get("pose_input_scores", pred["scores"])
            indices = pred["kept_indices"]
        else:
            boxes = pred["boxes"]
            scores = pred["scores"]
            indices = list(range(len(boxes)))
        legend_lines.append(f"{pred['object_name']}: {pred['prompt']}")
        for local_idx in indices:
            color = color_for_index(global_index)
            box = boxes[local_idx]
            draw.rectangle(list(map(float, box)), outline=(*color, 255), width=4)
            label = f"{pred['object_name']} {scores[local_idx]:.2f}"
            draw.text(
                (float(box[0]), max(0.0, float(box[1]) - 22)),
                label,
                fill=(*color, 255),
                font=font,
            )
            global_index += 1
    title = "Kept predictions" if kept_only else "Raw SAM3 predictions"
    draw.text((12, 10), title, fill=(255, 255, 255, 255), font=font)
    y = 34
    for line in legend_lines:
        draw.text((12, y), line, fill=(255, 255, 255, 255), font=font)
        y += 22
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(output_path)


