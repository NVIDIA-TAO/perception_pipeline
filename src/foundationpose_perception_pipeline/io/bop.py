#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn a BOP dataset directory into the inputs the inference path expects.

The point of this module is a single question: **where does the list of objects to look for
come from?** Today it is derived from `scene_gt.json`, because in an evaluation harness that is
the convenient place to find it. But it is a *task specification*, not a measurement -- a
deployment supplies the same information from configuration -- so the two sources are separate
functions here, and inference depends only on their shared output type.

`object_specs_from_names` is the deployable one. It needs no ground truth at all, which is the
property this whole refactor exists to establish; `object_specs_for_scene` is the evaluation
convenience that reads `scene_gt.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from foundationpose_perception_pipeline.dataset import prompt_for_object
from foundationpose_perception_pipeline.inference.types import ObjectSpec
from foundationpose_perception_pipeline.io.files import load_dataset_map


def dataset_scene_ids(dataset_dir: Path, split: str = "test") -> list[int]:
    """Every scene in the dataset, by numeric id.

    Deliberately does not intersect against a collected-depth tree. The pipeline used to do
    that here, which meant a machine without that tree silently selected *zero* scenes and
    reported success over nothing. Whether depth ground truth is needed is an evaluation
    question, and it is asked on the evaluation side.
    """
    scenes = dataset_dir / split
    if not scenes.exists():
        raise FileNotFoundError(f"Missing split directory: {scenes}")
    return sorted(int(path.name) for path in scenes.glob("*") if path.is_dir() and path.name.isdigit())


def scene_camera_matrix(dataset_dir: Path, scene_id: int, im_id: int = 0, split: str = "test") -> np.ndarray:
    """Intrinsics for one frame, from `scene_camera.json`.

    Intrinsics are an input, not a measurement, even though the ground-truth renderer also
    happens to expose them -- taking them from there made inference depend on the evaluation
    path for no reason.
    """
    path = dataset_dir / split / f"{scene_id:06d}" / "scene_camera.json"
    scene_camera = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(scene_camera[str(im_id)]["cam_K"], dtype=np.float32).reshape(3, 3)


def _mesh_path(dataset_dir: Path, obj_id: int, models_subdir: str) -> Path | None:
    """Locate an object's CAD mesh, or None if the dataset does not ship one."""
    candidate = dataset_dir / models_subdir / f"obj_{obj_id:06d}.ply"
    return candidate if candidate.exists() else None


def object_specs_from_names(
    *,
    dataset_dir: Path,
    object_names: list[str],
    prompt_overrides: dict[str, str] | None = None,
    prompts: dict[str, str] | None = None,
    models_subdir: str = "models_cad",
) -> list[ObjectSpec]:
    """Build specs from an explicit list of object names -- **no ground truth read.**

    This is the deployment-shaped entry point: you are told which objects exist, and you go
    looking for them. Nothing here opens `scene_gt.json`, so it works on a capture that has no
    annotations at all.
    """
    name_to_id, _ = load_dataset_map(dataset_dir)
    unknown = [name for name in object_names if name not in name_to_id]
    if unknown:
        raise KeyError(
            f"Unknown object name(s) {unknown} for {dataset_dir.name}. "
            f"Known names: {sorted(name_to_id)}"
        )
    return [
        ObjectSpec(
            object_key=name,
            obj_id=name_to_id[name],
            prompt=prompt_for_object(dataset_dir, name_to_id[name], prompt_overrides or {}, prompts),
            mesh_path=_mesh_path(dataset_dir, name_to_id[name], models_subdir),
        )
        for name in object_names
    ]


def object_specs_for_scene(
    *,
    dataset_dir: Path,
    scene_id: int,
    im_id: int = 0,
    prompt_overrides: dict[str, str] | None = None,
    prompts: dict[str, str] | None = None,
    models_subdir: str = "models_cad",
    split: str = "test",
) -> list[ObjectSpec]:
    """Build specs for the object classes a scene's ground truth says are present.

    The evaluation convenience. It reads `scene_gt.json` purely to answer "which classes are in
    this scene" -- the per-instance poses and counts there are not consulted, so a spec carries
    no information that could leak an answer into inference.
    """
    gt_path = dataset_dir / split / f"{scene_id:06d}" / "scene_gt.json"
    scene_gt = json.loads(gt_path.read_text(encoding="utf-8"))
    _, id_to_name = load_dataset_map(dataset_dir)
    obj_ids = sorted({int(entry["obj_id"]) for entry in scene_gt[str(im_id)]})
    return [
        ObjectSpec(
            object_key=id_to_name[obj_id],
            obj_id=obj_id,
            prompt=prompt_for_object(dataset_dir, obj_id, prompt_overrides or {}, prompts),
            mesh_path=_mesh_path(dataset_dir, obj_id, models_subdir),
        )
        for obj_id in obj_ids
    ]


def load_collected_depth_m(collected_scene_dir: Path, depth_filename: str = "scene_cam0_depth.png") -> np.ndarray:
    """Load collected base-camera depth in meters, converting missing pixels to NaN.

    `depth_filename` comes from `DatasetProfile.depth_filename`, which derives it from the
    profile's `base_camera_id`. PASS IT. The default here is camera 0 and is a fallback for
    callers with no profile to hand, not a statement about the data: `base_camera_id` is honoured
    when selecting the stereo pair, so a profile set to camera 1 that lands on this default
    predicts depth in camera 1's frame and scores it against camera 0's collected map -- silently,
    since both files exist and both load.
    """
    depth_mm = cv2.imread(str(collected_scene_dir / depth_filename), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(collected_scene_dir / depth_filename)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[depth_mm == 0] = np.nan
    return depth_m
