#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BOP dataset access: targets and prompts.

What the pipeline is asked to find, and where on disk the files for it are. Ground truth is
*not* here: `GroundTruthRenderer` and `render_gt_entries` live in `evaluation/gt.py`, which owns
the definition of ground truth. This module stays free of it so that inference, which has no
ground truth to read, can use the same target loading.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from foundationpose_perception_pipeline.config import active_settings
from foundationpose_perception_pipeline.io.files import load_dataset_map, load_json


@dataclass(frozen=True)
class Target:
    """One dataset/object/frame evaluation target derived from BOP annotations."""

    dataset: str
    scene_id: int
    im_id: int
    obj_id: int
    inst_count: int

    @property
    def key(self) -> str:
        """Return the stable string key used in incremental results files."""
        return f"{self.dataset}:{self.scene_id:06d}:{self.im_id:06d}:{self.obj_id:06d}"


def dataset_dirs(
    dataset_root: Path,
    requested: list[str] | None,
    glob: str | None = None,
) -> list[Path]:
    """Resolve the dataset directories to evaluate and validate they exist.

    `glob` selects which subfolders count as datasets when `requested` is empty; it defaults to
    the active profile's `dataset.glob`, so nothing here hardcodes a naming convention.
    """
    if requested:
        dirs = [dataset_root / name for name in requested]
    else:
        pattern = glob if glob is not None else active_settings().dataset.glob
        dirs = sorted(path for path in dataset_root.glob(pattern) if path.is_dir())
    missing = [str(path) for path in dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset folders: {missing}")
    return dirs


def scene_path(dataset_dir: Path, scene_id: int, split: str = "test") -> Path:
    """Return the canonical BOP scene directory for one scene id.

    `split` is the profile's `dataset.split`. It was parsed and then ignored here, so a profile
    with `split: val` silently read `test/` instead -- the io/bop.py helpers already accepted it,
    but no call site passed it.
    """
    return dataset_dir / split / f"{scene_id:06d}"


def image_path(dataset_dir: Path, scene_id: int, im_id: int, split: str = "test") -> Path:
    """Return the RGB image path for one dataset scene/frame.

    `split` is forwarded to `scene_path`; it used to be dropped here, so a profile with
    `split: val` resolved its images under `test/`.
    """
    return scene_path(dataset_dir, scene_id, split) / "rgb" / f"{im_id:06d}.png"


def load_prompt_overrides(path: Path | None) -> dict[str, str]:
    """Load optional prompt overrides from JSON, or return an empty mapping."""
    if path is None:
        return {}
    data = load_json(path)
    return {str(key): str(value) for key, value in data.items()}


def prompt_for_object(
    dataset_dir: Path,
    obj_id: int,
    overrides: dict[str, str],
    prompts: dict[str, str] | None = None,
) -> str:
    """Choose the text prompt for one object using overrides, then the profile, then the name.

    `prompts` is the active profile's prompt map. Callers that already hold a `Settings` should
    pass `settings.dataset.prompts` so the function stays free of global state; when omitted it
    falls back to the profile the running command resolved (`active_settings()`).
    """
    _, id_to_name = load_dataset_map(dataset_dir)
    name = id_to_name[obj_id]
    if prompts is None:
        prompts = active_settings().dataset.prompts
    return (
        overrides.get(str(obj_id))
        or overrides.get(f"{obj_id:06d}")
        or overrides.get(name)
        or prompts.get(name, name.replace("_", " "))
    )


def targets_from_test_targets(dataset_dir: Path) -> list[Target]:
    """Load evaluation targets from BOP `test_targets_bop19.json` when present."""
    target_path = dataset_dir / "test_targets_bop19.json"
    if not target_path.exists():
        return []
    return [
        Target(
            dataset=dataset_dir.name,
            scene_id=int(row["scene_id"]),
            im_id=int(row["im_id"]),
            obj_id=int(row["obj_id"]),
            inst_count=int(row["inst_count"]),
        )
        for row in load_json(target_path)
    ]


def targets_from_gt(dataset_dir: Path, split: str = "test") -> list[Target]:
    """Enumerate evaluation targets directly from per-frame `scene_gt.json` entries."""
    targets: list[Target] = []
    for scene_dir in sorted((dataset_dir / split).glob("*")):
        if not scene_dir.is_dir():
            continue
        gt_path = scene_dir / "scene_gt.json"
        if not gt_path.exists():
            continue
        scene_id = int(scene_dir.name)
        scene_gt = load_json(gt_path)
        for im_key, entries in sorted(scene_gt.items(), key=lambda item: int(item[0])):
            counts = Counter(int(entry["obj_id"]) for entry in entries)
            for obj_id, inst_count in sorted(counts.items()):
                targets.append(
                    Target(
                        dataset=dataset_dir.name,
                        scene_id=scene_id,
                        im_id=int(im_key),
                        obj_id=obj_id,
                        inst_count=inst_count,
                    )
                )
    return targets


def targets_for_scene_frame0(dataset_dir: Path, scene_id: int, split: str = "test") -> list[Target]:
    """Build one frame-0 target per object class present in a scene."""
    gt_path = dataset_dir / split / f"{scene_id:06d}" / "scene_gt.json"
    scene_gt = json.loads(gt_path.read_text(encoding="utf-8"))
    counts = Counter(int(entry["obj_id"]) for entry in scene_gt["0"])
    return [
        Target(
            dataset=dataset_dir.name,
            scene_id=scene_id,
            im_id=0,
            obj_id=obj_id,
            inst_count=inst_count,
        )
        for obj_id, inst_count in sorted(counts.items())
    ]


def load_targets(dataset_dir: Path, source: str) -> list[Target]:
    """Load targets from the requested source, with `auto` fallback behavior."""
    if source == "test_targets":
        targets = targets_from_test_targets(dataset_dir)
        if not targets:
            raise FileNotFoundError(f"No test_targets_bop19.json in {dataset_dir}")
        return targets
    if source == "auto":
        targets = targets_from_test_targets(dataset_dir)
        return targets or targets_from_gt(dataset_dir)
    return targets_from_gt(dataset_dir)


def completed_keys(results_path: Path) -> set[str]:
    """Read successful target keys already present in the incremental results file."""
    if not results_path.exists():
        return set()
    keys = set()
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("error"):
                keys.add(record["target_key"])
    return keys
