#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the adapted tree. Every adapter goes through here, so the layout has one definition.

The tree the pipeline reads is::

    <out_root>/<name>/test/<scene:06d>/rgb/<im_id:06d>.png
                                      /scene_camera.json     im_ids are the CAMERAS of a rig
                                      /scene_gt.json
                                      /scene_gt_info.json
    <out_root>/<name>/models/          binary PLY + models_info.json
    <out_root>/<name>/models_eval/     BOP's decimated copy, for the pose metrics
    <out_root>/<name>/dataset_map.json {object_name: obj_id}
    <out_root>/<name>/scene_index.json provenance back to the source dataset
    <depth_root>/<name>/test/<scene:06d>/scene_cam0_depth.png   uint16 millimetres

Four invariants are load-bearing, and each one has a consumer that breaks silently without it:

1. **im_id 0 is the base camera.** The pipeline estimates pose in `rgb/000000.png` and scores
   ground truth for im_id 0; a dataset whose own numbering starts elsewhere gets re-keyed.
2. **im_ids are contiguous from 0.** `select_partner_camera` indexes the camera array.
3. **Meshes are binary little-endian PLY**, because neither
   `geometry.read_binary_little_endian_ply` nor FoundationPose reads ASCII.
4. **Collected depth is uint16 millimetres with 0 as invalid**, which is what
   `io.bop.load_collected_depth_m` assumes.

The split is always `test`. That is not a claim about the source -- a dataset's public-GT split
is often named something else, `test_primesense_bop19` or `val` -- it is the name the adapted
copy is written under, and profiles set `dataset.split: test` to match.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bop_adapt.ply import convert

# `indent=1` for the per-scene files and `indent=2` for the dataset-level ones, matching what BOP
# itself writes. Kept explicit because these files are diffed against the source when an
# adaptation is checked, and re-indenting turns that diff into noise.
SCENE_JSON_INDENT = 1
DATASET_JSON_INDENT = 2


def link(src: Path, dst: Path) -> None:
    """Symlink `dst` -> `src`, relative, replacing whatever is there.

    Relative rather than absolute so the adapted tree survives being moved, and a symlink rather
    than a copy because the RGB frames are the bulk of a BOP dataset and the adapter may expose
    the same source frame from several adapted scenes.
    """
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent))


def write_json(path: Path, payload: Any, indent: int = SCENE_JSON_INDENT) -> None:
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def object_names_path(pipeline_root: Path, dataset: str) -> Path:
    """Where a dataset's obj_id -> name table lives: `config/<dataset>/object_names.json`.

    Beside the profile rather than beside the adapter, because it is dataset METADATA and the
    profile's `prompts:` block is keyed by the names in it. BOP ships numeric obj_ids and no
    names at all, so this table is an invention of this repo and has to be checked in.
    """
    return pipeline_root / "config" / dataset / "object_names.json"


def load_object_names(pipeline_root: Path, dataset: str, override: Path | None = None) -> dict[str, str]:
    path = override or object_names_path(pipeline_root, dataset)
    if not path.is_file():
        raise SystemExit(
            f"No object-name table for {dataset!r} at {path}. It maps each obj_id in "
            f"models_info.json to the name the profile's `prompts:` block is keyed by; BOP ships "
            f"no names, so it has to be written by hand."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_models(
    *,
    dataset_dir: Path,
    models_src: Path,
    models_eval_src: Path,
    models_info_src: Path | None = None,
) -> int:
    """Convert `models_src`'s meshes to binary PLY under `models/`, and link `models_eval/`.

    `models/` is written rather than linked because `GroundTruthRenderer.mesh` hardcodes that
    directory and the meshes may need converting; `models_eval/` is linked through untouched
    because the pose metrics only read it.
    """
    (dataset_dir / "models").mkdir(parents=True, exist_ok=True)
    plys = sorted(models_src.glob("*.ply"))
    for ply in plys:
        convert(ply, dataset_dir / "models" / ply.name)
    link(models_info_src or (models_src / "models_info.json"), dataset_dir / "models" / "models_info.json")

    (dataset_dir / "models_eval").mkdir(parents=True, exist_ok=True)
    for entry in sorted(models_eval_src.iterdir()):
        link(entry, dataset_dir / "models_eval" / entry.name)
    return len(plys)


def write_dataset_map(dataset_dir: Path, names: dict[str, str], models_info: dict[str, Any]) -> None:
    """Write `{object_name: obj_id}` for every object the models declare.

    Driven by `models_info.json` rather than by the name table, so an object present in the
    dataset but missing from the table fails here instead of silently losing its prompt.
    """
    missing = [o for o in models_info if str(o) not in names]
    if missing:
        raise SystemExit(
            f"object_names.json is missing {len(missing)} obj_id(s) present in models_info.json: "
            f"{', '.join(map(str, missing[:10]))}"
        )
    write_json(
        dataset_dir / "dataset_map.json",
        {names[str(o)]: int(o) for o in sorted(models_info, key=int)},
        DATASET_JSON_INDENT,
    )


def write_scene(
    *,
    out_dir: Path,
    chosen: Sequence[str],
    src_rgb: Path,
    cam: dict[str, Any],
    gt: dict[str, Any],
    gt_info: dict[str, Any],
    rgb_name: str = "{:06d}.png",
) -> None:
    """Write one adapted scene: `chosen[0]` becomes im_id 0, the rest follow in order.

    `chosen` holds keys into the SOURCE `scene_camera.json`. Re-keying is the whole job -- the
    source may number frames non-contiguously (T-LESS: 1, 17, 30, ...) or name cameras rather
    than number them, and the pipeline needs 0..K with 0 as the base.
    """
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    scene_camera, scene_gt, scene_gt_info = {}, {}, {}
    for im_id, key in enumerate(chosen):
        link(src_rgb / rgb_name.format(int(key)), out_dir / "rgb" / f"{im_id:06d}.png")
        scene_camera[str(im_id)] = {
            "cam_K": cam[key]["cam_K"],
            "cam_R_w2c": cam[key]["cam_R_w2c"],
            "cam_t_w2c": cam[key]["cam_t_w2c"],
            "depth_scale": cam[key]["depth_scale"],
        }
        scene_gt[str(im_id)] = gt[key]
        scene_gt_info[str(im_id)] = gt_info[key]
    write_json(out_dir / "scene_camera.json", scene_camera)
    write_json(out_dir / "scene_gt.json", scene_gt)
    write_json(out_dir / "scene_gt_info.json", scene_gt_info)


def write_collected_depth(*, depth_split: Path, out_id: int, raw: np.ndarray, depth_scale: float) -> None:
    """Write the base camera's sensor depth as uint16 millimetres, 0 meaning invalid.

    Scaled by the source's own `depth_scale`. Zeros are re-zeroed after rounding rather than
    trusted to survive it: a scale below 1 rounds small non-zero readings to 0 too, and those are
    measurements, not holes -- but a reading that was ALREADY 0 must stay invalid.
    """
    mm = np.rint(raw.astype(np.float64) * float(depth_scale)).astype(np.uint16)
    mm[raw == 0] = 0
    out_dir = depth_split / f"{out_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "scene_cam0_depth.png"), mm)


def read_image_size(path: Path) -> tuple[int, int]:
    """(height, width) of an image, for a check that must match what the depth stage will see."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image.shape[0], image.shape[1]


def read_depth(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    return raw


def write_scene_index(dataset_dir: Path, index: Iterable[dict[str, Any]]) -> None:
    """Record how each adapted scene maps back to the source.

    Not a debugging aid: scoring an adapted run against the ORIGINAL dataset -- a BOP submission,
    say -- needs the source scene and frame for every adapted scene, and nothing else records it.
    """
    write_json(dataset_dir / "scene_index.json", list(index), DATASET_JSON_INDENT)
