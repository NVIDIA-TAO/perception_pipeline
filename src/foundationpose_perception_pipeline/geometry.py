#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-level geometry, mesh I/O, and IoU primitives for the pipeline.

This is the bottom layer: it depends on nothing else in the package, and everything else
depends on it. Rasterization, PLY I/O, mask rendering, IoU and greedy matching were once split
across two modules that had each grown their own `best_ious` and JSON helpers; merging them is
what made a single definition of "which prediction matched which instance" possible.

Two rasterizers live here and the difference matters:

- `render_mask` draws ONE object with no depth buffer -> an *amodal* silhouette that includes
  pixels hidden behind other objects.
- `rasterize_mesh` writes into a caller-supplied shared z-buffer, so rasterizing every scene
  object into the same buffers yields *modal* (visible-region) masks.

Ground-truth masks must use the second one; see `evaluation.gt.render_gt_entries`.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TYPE_MAP = {
    "char": ("i1", 1, "b"),
    "uchar": ("u1", 1, "B"),
    "int8": ("i1", 1, "b"),
    "uint8": ("u1", 1, "B"),
    "short": ("<i2", 2, "h"),
    "ushort": ("<u2", 2, "H"),
    "int16": ("<i2", 2, "h"),
    "uint16": ("<u2", 2, "H"),
    "int": ("<i4", 4, "i"),
    "uint": ("<u4", 4, "I"),
    "int32": ("<i4", 4, "i"),
    "uint32": ("<u4", 4, "I"),
    "float": ("<f4", 4, "f"),
    "float32": ("<f4", 4, "f"),
    "double": ("<f8", 8, "d"),
    "float64": ("<f8", 8, "d"),
}

PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def read_binary_little_endian_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break

        vertex_count = 0
        face_count = 0
        in_vertex = False
        vertex_props: list[tuple[str, str]] = []
        face_list_types: tuple[str, str] | None = None
        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
                in_vertex = False
            elif parts[0] == "property" and in_vertex:
                vertex_props.append((parts[2], parts[1]))
            elif parts[0] == "property" and len(parts) > 2 and parts[1] == "list":
                face_list_types = (parts[2], parts[3])

        dtype = np.dtype([(name, TYPE_MAP[prop_type][0]) for name, prop_type in vertex_props])
        vertices_raw = np.frombuffer(f.read(dtype.itemsize * vertex_count), dtype=dtype, count=vertex_count)
        vertices = np.stack([vertices_raw["x"], vertices_raw["y"], vertices_raw["z"]], axis=1).astype(np.float64)

        if face_list_types is None:
            raise ValueError(f"No face list property in {path}")
        count_type, index_type = face_list_types
        count_size = TYPE_MAP[count_type][1]
        index_size = TYPE_MAP[index_type][1]
        count_fmt = "<" + TYPE_MAP[count_type][2]
        index_fmt = TYPE_MAP[index_type][2]
        faces = []
        for _ in range(face_count):
            count = struct.unpack(count_fmt, f.read(count_size))[0]
            indices = struct.unpack("<" + index_fmt * count, f.read(index_size * count))
            for idx in range(1, count - 1):
                faces.append((indices[0], indices[idx], indices[idx + 1]))
    return vertices, np.asarray(faces, dtype=np.int32)


def render_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    points_cam = (rotation @ vertices.T + translation.reshape(3, 1)).T
    valid_z = points_cam[:, 2] > 1e-6
    uvw = (camera @ points_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    mask = np.zeros((height, width), dtype=np.uint8)
    for tri in faces:
        if not np.all(valid_z[tri]):
            continue
        points = uv[tri]
        if np.any(~np.isfinite(points)):
            continue
        if points[:, 0].max() < 0 or points[:, 0].min() >= width:
            continue
        if points[:, 1].max() < 0 or points[:, 1].min() >= height:
            continue
        cv2.fillConvexPoly(mask, np.round(points).astype(np.int32), 1)
    return mask.astype(bool)


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    return float(intersection / union)


def greedy_match(iou_matrix: np.ndarray, threshold: float) -> dict[str, Any]:
    matches = []
    used_pred = set()
    used_gt = set()
    pairs = [(p, g) for p in range(iou_matrix.shape[0]) for g in range(iou_matrix.shape[1])]
    for pred_idx, gt_idx in sorted(pairs, key=lambda pair: iou_matrix[pair], reverse=True):
        if iou_matrix[pred_idx, gt_idx] < threshold:
            break
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, float(iou_matrix[pred_idx, gt_idx])))
    tp = len(matches)
    pred_count, gt_count = iou_matrix.shape
    return {
        "tp": tp,
        "fp": pred_count - tp,
        "fn": gt_count - tp,
        "precision": tp / pred_count if pred_count else 0.0,
        "recall": tp / gt_count if gt_count else 0.0,
        "matches": matches,
    }


def best_ious(iou_matrix: np.ndarray) -> list[float]:
    if iou_matrix.shape[0] == 0:
        return []
    if iou_matrix.shape[1] == 0:
        return [0.0] * iou_matrix.shape[0]
    return [float(value) for value in iou_matrix.max(axis=1)]


def _projected_bbox(
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    min_x = max(0, math.floor(float(np.min(u))))
    max_x = min(width - 1, math.ceil(float(np.max(u))))
    min_y = max(0, math.floor(float(np.min(v))))
    max_y = min(height - 1, math.ceil(float(np.max(v))))
    if min_x > max_x or min_y > max_y:
        return None
    return min_x, max_x, min_y, max_y


def rasterize_mesh(
    vertices_model_mm: np.ndarray,
    faces: np.ndarray,
    r_m2c: np.ndarray,
    t_m2c: np.ndarray,
    k: np.ndarray,
    zbuf_mm: np.ndarray,
    obj_map: np.ndarray,
    inst_map: np.ndarray,
    obj_id: int,
    inst_id: int,
    near_mm: float,
) -> tuple[int, int]:
    height, width = zbuf_mm.shape
    vertices_cam = vertices_model_mm @ r_m2c.T + t_m2c[None, :]
    z = vertices_cam[:, 2]
    u = k[0, 0] * vertices_cam[:, 0] / z + k[0, 2]
    v = k[1, 1] * vertices_cam[:, 1] / z + k[1, 2]

    rendered_triangles = 0
    touched_pixels = 0
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    for tri in faces:
        tri_z = z[tri]
        if np.any(tri_z <= near_mm):
            continue

        tri_u = u[tri]
        tri_v = v[tri]
        bbox = _projected_bbox(tri_u, tri_v, width, height)
        if bbox is None:
            continue
        min_x, max_x, min_y, max_y = bbox

        p0, p1, p2 = vertices_cam[tri].astype(np.float64)
        normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(normal) < 1e-9:
            continue
        plane_d = -float(np.dot(normal, p0))

        x0, x1, x2 = [float(x) for x in tri_u]
        y0, y1, y2 = [float(y) for y in tri_v]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(xs, ys)

        b0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
        b1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-5) & (b1 >= -1e-5) & (b2 >= -1e-5)
        if not np.any(inside):
            continue

        ray_x = (xx - cx) / fx
        ray_y = (yy - cy) / fy
        ray_den = normal[0] * ray_x + normal[1] * ray_y + normal[2]
        depth = -plane_d / ray_den
        valid = inside & np.isfinite(depth) & (depth > near_mm)
        if not np.any(valid):
            continue

        sub_z = zbuf_mm[min_y : max_y + 1, min_x : max_x + 1]
        update = valid & (depth < sub_z)
        if not np.any(update):
            continue

        sub_z[update] = depth[update].astype(np.float32)
        obj_map[min_y : max_y + 1, min_x : max_x + 1][update] = obj_id
        inst_map[min_y : max_y + 1, min_x : max_x + 1][update] = inst_id
        rendered_triangles += 1
        touched_pixels += int(update.sum())

    return rendered_triangles, touched_pixels


