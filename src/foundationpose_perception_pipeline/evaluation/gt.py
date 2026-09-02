#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth rasterization and its on-disk cache.

Evaluation-only: nothing here runs at deployment. It lives behind the evaluation boundary
rather than in the shared namespace so the inference path cannot reach it by accident.

Precompute and cache the expensive part of `render_gt_entries`, defined below.

`render_gt_entries` rebuilds a full-scene occlusion z-buffer (`geometry.rasterize_mesh`, a
per-triangle Python loop) every time it's called -- and it's called once per *target* (once per
distinct object class present in a scene), so a scene with T target classes redoes the same
full-scene rasterization T times. Measured cost on one 13-instance scene (, one mesh with
27k faces, 2 target classes): ~12-14s *per target*, ~26s for the scene.

The z-buffer output depends only on static ground truth (`scene_gt.json`, CAD meshes, camera
intrinsics) -- never on SAM3/FoundationPose predictions -- so it's safe to precompute once per
scene and reuse across every pipeline run, every target in that scene, and every
`--min-visible-fraction` value (the threshold only decides which already-rasterized instances
get kept, so it's applied at load time, not baked into the cache).

What's cached per scene (one small `.npz` file):
    - `inst_map` (H, W) int32:          the occlusion-aware instance-id map from the shared
                                         z-buffer pass (0 = background, i = the i-th scene_gt
                                         entry, 1-indexed). This is the one array that actually
                                         costs the ~12s to build.
    - `amodal_pixel_count` (N,) int64:  unoccluded pixel count for *every* scene_gt entry (not
                                         just one target's), so `visible_fraction` can be
                                         recomputed for any target without ever re-rasterizing.
    - `camera_matrix` (3, 3) float64 and `image_size` (2,) int32: stored for a cheap sanity
      check at load time, not because they're expensive to reload.

Per-target masks/boxes are *not* stored -- `inst_map == inst_id` is a single vectorized
comparison over the full image (sub-millisecond), so there's no reason to pay disk I/O for
masks that are this cheap to derive. `kept_entries` (the raw `scene_gt.json` dicts) aren't
stored either, for the same reason: they're re-filtered from the already-cached
`GroundTruthRenderer.scene_gt()` JSON at load time.

Usage:
    # precompute every scene in one dataset
    python script/build_gt_cache.py --dataset <name>

    # precompute every dataset under --dataset-root
    python script/build_gt_cache.py --all

    # re-render scenes that already have a cache file
    python script/build_gt_cache.py --dataset <name> --overwrite

See `render_gt_entries_cached` below for the drop-in replacement used by
`script/run_pipeline.py`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from foundationpose_perception_pipeline.config import (
    DEFAULT_MIN_VISIBLE_FRACTION,
    GT_RASTERIZER_NEAR_MM,
    active_settings,
    add_config_argument,
    settings_from_argv,
)
from foundationpose_perception_pipeline.dataset import Target, dataset_dirs, scene_path
from foundationpose_perception_pipeline.geometry import bbox_from_mask, rasterize_mesh, render_mask
from foundationpose_perception_pipeline.io.files import load_json


def default_cache_root() -> Path:
    """Return the GT cache root from the active dataset profile."""
    return active_settings().dataset.gt_cache_root


def cache_path(cache_root: Path, dataset: str, scene_id: int) -> Path:
    """Return the on-disk path for one scene's cached GT arrays."""
    return cache_root / dataset / f"{scene_id:06d}.npz"


def _scene_ids(dataset_dir: Path, split: str = "test") -> list[int]:
    """Return every scene id present under `<dataset_dir>/<split>/`."""
    return sorted(int(path.name) for path in (dataset_dir / split).glob("*") if path.is_dir())


def render_scene_gt_cache(
    renderer: GroundTruthRenderer,
    dataset: str,
    scene_id: int,
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Build the cacheable per-scene GT arrays -- the expensive part of `render_gt_entries`.

    Rasterizes every instance in the scene (all object classes together) into one shared
    occlusion z-buffer, exactly like `render_gt_entries` does internally, plus an amodal
    (unoccluded) pixel count per instance so `visible_fraction` never needs re-rasterizing.
    """
    width, height = image_size
    # `scene_gt`/`scene_camera` only key off (dataset, scene_id); obj_id/im_id/inst_count are
    # irrelevant here, since we want every instance in the scene, not one target's.
    scene_key_target = Target(dataset=dataset, scene_id=scene_id, im_id=0, obj_id=-1, inst_count=0)
    camera_matrix = np.asarray(
        renderer.scene_camera(scene_key_target)["0"]["cam_K"], dtype=np.float64
    ).reshape(3, 3)
    scene_entries = renderer.scene_gt(scene_key_target)["0"]

    zbuf_mm = np.full((height, width), np.inf, dtype=np.float64)
    obj_map = np.zeros((height, width), dtype=np.int32)
    inst_map = np.zeros((height, width), dtype=np.int32)
    amodal_pixel_count = np.zeros(len(scene_entries), dtype=np.int64)

    for index, entry in enumerate(scene_entries):
        obj_id = int(entry["obj_id"])
        vertices, faces = renderer.mesh(dataset, obj_id)
        rotation = np.asarray(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(entry["cam_t_m2c"], dtype=np.float64)
        rasterize_mesh(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces),
            rotation,
            translation,
            camera_matrix,
            zbuf_mm,
            obj_map,
            inst_map,
            obj_id,
            index + 1,
            GT_RASTERIZER_NEAR_MM,
        )
        amodal_mask = render_mask(vertices, faces, rotation, translation, camera_matrix, image_size)
        amodal_pixel_count[index] = int(amodal_mask.sum())

    return {
        "inst_map": inst_map,
        "amodal_pixel_count": amodal_pixel_count,
        "camera_matrix": camera_matrix,
        "image_size": np.array([width, height], dtype=np.int32),
    }


def precompute_dataset(
    dataset_root: Path,
    dataset: str,
    cache_root: Path | None = None,
    overwrite: bool = False,
    split: str = "test",
    models_subdir: str = "models",
) -> None:
    """Precompute and save the GT cache for every scene in one dataset.

    `cache_root` defaults to the active profile's `dataset.gt_cache_root`.

    `models_subdir` has to match what scoring will use, and this is the sharper of the two places
    it is threaded: the cache is read back instead of re-rendering, so a cache built from one mesh
    tree and scored against another disagrees silently and for the whole life of the cache, which
    outlives the run that wrote it.
    """
    cache_root = cache_root if cache_root is not None else default_cache_root()
    dataset_dir = dataset_root / dataset
    renderer = GroundTruthRenderer(dataset_root, split, models_subdir)
    scene_ids = _scene_ids(dataset_dir, split)
    out_dir = cache_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene_id in scene_ids:
        out_path = cache_path(cache_root, dataset, scene_id)
        if out_path.exists() and not overwrite:
            print(f"[skip] {out_path} already exists")
            continue
        image_file = scene_path(dataset_dir, scene_id, split) / "rgb" / "000000.png"
        image_size = Image.open(image_file).size
        start = time.perf_counter()
        arrays = render_scene_gt_cache(renderer, dataset, scene_id, image_size)
        elapsed = time.perf_counter() - start
        np.savez_compressed(out_path, **arrays)
        size_mb = out_path.stat().st_size / 1e6
        print(f"[{dataset}/{scene_id:06d}] {elapsed:.2f}s -> {out_path} ({size_mb:.2f} MB)")


def load_scene_gt_cache(cache_root: Path, dataset: str, scene_id: int) -> dict[str, np.ndarray] | None:
    """Load one scene's cached GT arrays, or return None if it hasn't been precomputed."""
    path = cache_path(cache_root, dataset, scene_id)
    if not path.exists():
        return None
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


# Datasets already reported as uncached, so the notice below is printed once per dataset rather
# than once per target. Process-local by design: it is a convenience message, not state anything
# depends on.
_REPORTED_CACHE_MISSES: set[str] = set()


def _report_cache_miss(dataset: str, cache_root: Path) -> None:
    """Say once, on stderr, that this dataset is being rendered without its cache.

    Not a failure: rendering uncached is correct and produces identical numbers. But it is
    ~12-14 s per target against a fraction of a second, so a run that silently takes forty
    minutes instead of two looks like a hang with no cause on screen.
    """
    if dataset in _REPORTED_CACHE_MISSES:
        return
    _REPORTED_CACHE_MISSES.add(dataset)
    print(
        f"note: no GT cache for {dataset} under {cache_root} -- rendering ground truth from "
        f"scratch at ~12-14 s per target. Results are identical either way; precompute it once "
        f"with `python script/build_gt_cache.py --dataset {dataset}` to avoid the wait.",
        file=sys.stderr,
    )


def render_gt_entries_cached(
    renderer: GroundTruthRenderer,
    target: Target,
    image_size: tuple[int, int],
    cache_root: Path | None = None,
    min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION,
) -> tuple[list[dict[str, Any]], list[np.ndarray], np.ndarray, np.ndarray, list[float]]:
    """Drop-in, cache-aware replacement for `render_gt_entries` below.

    Same signature and return contract. Falls back to the uncached (slow) `render_gt_entries`
    for any scene that hasn't been precomputed yet -- run `script/build_gt_cache.py --dataset <name>` ahead
    of time to avoid paying that cost inside a pipeline run.

    That fallback is the path a warm machine never takes, so keep it callable: it was a deferred
    `from foundationpose_perception_pipeline.dataset import render_gt_entries` left behind when the function
    moved into this module, and it raised ImportError on the first cache miss -- which on a
    machine with a fully populated `gt_cache/` is never, and on a fresh checkout is immediately.
    """
    cache_root = cache_root if cache_root is not None else default_cache_root()
    cached = load_scene_gt_cache(cache_root, target.dataset, target.scene_id)
    if cached is None:
        _report_cache_miss(target.dataset, cache_root)
        return render_gt_entries(renderer, target, image_size, min_visible_fraction)

    width, height = image_size
    cached_width, cached_height = (int(value) for value in cached["image_size"])
    if (cached_width, cached_height) != (width, height):
        raise ValueError(
            f"GT cache for {target.dataset}/{target.scene_id:06d} was built at "
            f"{cached_width}x{cached_height}, but this call uses {width}x{height}. "
            "Re-run script/build_gt_cache.py --overwrite for this dataset."
        )

    inst_map = cached["inst_map"]
    amodal_pixel_count = cached["amodal_pixel_count"]
    camera_matrix = cached["camera_matrix"]
    frame_key = str(target.im_id)
    scene_entries = renderer.scene_gt(target)[frame_key]

    kept_entries: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    visible_fractions: list[float] = []
    for index, entry in enumerate(scene_entries):
        if int(entry["obj_id"]) != target.obj_id:
            continue
        visible_mask = inst_map == (index + 1)
        amodal_count = int(amodal_pixel_count[index])
        visible_fraction = float(visible_mask.sum()) / amodal_count if amodal_count else 0.0
        if not visible_mask.any() or visible_fraction < min_visible_fraction:
            continue
        kept_entries.append(entry)
        masks.append(visible_mask)
        boxes.append(bbox_from_mask(visible_mask))
        visible_fractions.append(visible_fraction)
    return kept_entries, masks, np.asarray(boxes, dtype=np.float64), camera_matrix, visible_fractions


def main() -> None:
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--dataset-root", type=Path, default=settings.dataset.root)
    parser.add_argument("--cache-root", type=Path, default=settings.dataset.gt_cache_root)
    parser.add_argument("--dataset", help="Dataset folder to precompute, e.g. the first one listed by --all.")
    parser.add_argument(
        "--all", action="store_true", help="Precompute every dataset under --dataset-root."
    )
    parser.add_argument("--models-subdir", default=settings.dataset.models_subdir,
                        help="Mesh directory the GT is rasterized from. Must match what scoring "
                             "uses, or the cache and the metrics describe different meshes.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render scenes that already have a cache file.")
    parser.add_argument("--split", default=settings_from_argv().dataset.split,
                        help="BOP split directory to enumerate scenes from.")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("pass --dataset <name> or --all")

    datasets = [d.name for d in dataset_dirs(args.dataset_root, None)] if args.all else [args.dataset]
    for dataset in datasets:
        precompute_dataset(args.dataset_root, dataset, args.cache_root, args.overwrite, args.split,
                           args.models_subdir)


if __name__ == "__main__":
    main()


def render_target_gt(
    *,
    gt_renderer: Any,
    target: Any,
    image_size_wh: tuple[int, int],
    gt_cache_root: Path | None,
    use_gt_cache: bool,
    min_visible_fraction: float,
) -> tuple[list[dict[str, Any]], Any, Any, Any, Any]:
    """Return `(entries, masks, boxes, camera_matrix, visible_fractions)` for one target.

    The masks are occlusion-aware: every instance in the scene is rasterized into one shared
    z-buffer, so a mask holds only the pixels where that instance is the front-most surface.

    Cached by default. Rendering from scratch costs ~12-14 s per target and produces
    byte-identical output, which is why the cache exists; `use_gt_cache=False` is the
    escape hatch for when the cache itself is what you doubt.
    """
    if not use_gt_cache:
        return render_gt_entries(gt_renderer, target, image_size_wh, min_visible_fraction)
    return render_gt_entries_cached(gt_renderer, target, image_size_wh, gt_cache_root, min_visible_fraction)


class GroundTruthRenderer:
    """Cache CAD meshes and render GT modal masks and boxes for a target."""

    def __init__(self, dataset_root: Path, split: str = "test", models_subdir: str = "models") -> None:
        """Initialize mesh and annotation caches rooted at the dataset tree.

        `split` is the profile's `dataset.split`. It is held here rather than passed per call
        because every lookup below resolves the same scene directory, and the renderer used to
        hardcode `test/` through `scene_path`'s default -- so a profile with `split: val` read
        annotations from the wrong tree without saying so.

        `models_subdir` is the profile's `dataset.models_subdir`, and it matters for the same
        reason: this class rasterizes the ground truth that DETECTION metrics are scored against,
        while the pose stage loads its meshes from whatever the profile names. Hardcoding `models`
        here meant a profile pointing pose at `models_cad/` was scored against masks rendered from
        a different mesh tree, with no error and no mention in the report. Harmless only while the
        two trees are byte-identical, which is not a property any profile declares.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.models_subdir = models_subdir
        self.mesh_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self.scene_gt_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self.scene_camera_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def mesh(self, dataset: str, obj_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load and cache the BOP model mesh for one dataset/object pair."""
        from foundationpose_perception_pipeline.geometry import read_binary_little_endian_ply

        key = (dataset, obj_id)
        if key not in self.mesh_cache:
            path = self.dataset_root / dataset / self.models_subdir / f"obj_{obj_id:06d}.ply"
            self.mesh_cache[key] = read_binary_little_endian_ply(path)
        return self.mesh_cache[key]

    def scene_gt(self, target: Target) -> dict[str, Any]:
        """Load and cache `scene_gt.json` for the target's scene."""
        key = (target.dataset, target.scene_id)
        if key not in self.scene_gt_cache:
            self.scene_gt_cache[key] = load_json(
                scene_path(self.dataset_root / target.dataset, target.scene_id, self.split) / "scene_gt.json"
            )
        return self.scene_gt_cache[key]

    def scene_camera(self, target: Target) -> dict[str, Any]:
        """Load and cache `scene_camera.json` for the target's scene."""
        key = (target.dataset, target.scene_id)
        if key not in self.scene_camera_cache:
            self.scene_camera_cache[key] = load_json(
                scene_path(self.dataset_root / target.dataset, target.scene_id, self.split) / "scene_camera.json"
            )
        return self.scene_camera_cache[key]

    def render_target(self, target: Target, image_size: tuple[int, int]) -> tuple[list[np.ndarray], np.ndarray]:
        """Render all visible GT masks and boxes for one target object in one frame."""
        from foundationpose_perception_pipeline.geometry import bbox_from_mask, render_mask

        vertices, faces = self.mesh(target.dataset, target.obj_id)
        frame_key = str(target.im_id)
        camera = np.array(self.scene_camera(target)[frame_key]["cam_K"], dtype=np.float64).reshape(3, 3)
        entries = [
            entry
            for entry in self.scene_gt(target)[frame_key]
            if int(entry["obj_id"]) == target.obj_id
        ]
        masks = []
        boxes = []
        for entry in entries:
            rotation = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            translation = np.array(entry["cam_t_m2c"], dtype=np.float64)
            mask = render_mask(vertices, faces, rotation, translation, camera, image_size)
            if not mask.any():
                continue
            masks.append(mask)
            boxes.append(bbox_from_mask(mask))
        return masks, np.asarray(boxes, dtype=np.float64)


def render_gt_entries(
    renderer: GroundTruthRenderer,
    target: Target,
    image_size: tuple[int, int],
    min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION,
) -> tuple[list[dict[str, Any]], list[np.ndarray], np.ndarray, np.ndarray, list[float]]:
    """Render occlusion-aware GT masks, boxes, and visibility for one target.

    This function defines what "ground truth" means for every downstream metric: it fixes
    which instances are scored at all, and which pixels each one owns. Both choices follow
    the BOP Challenge evaluation protocol.

    **Masks are modal (visible-region), not amodal.**
    Every object in the scene -- all classes, not just this target's -- is rasterized into a
    single shared z-buffer, so an instance's mask contains only the pixels where it is the
    front-most surface. This is what a segmenter can actually produce.

    Rendering each instance independently instead yields *amodal* masks that include pixels
    hidden behind other objects, which silently punishes correct predictions: in a cluttered pile
    a half-buried part segmented perfectly scores roughly 0.45 box IoU against its full silhouette
    and is recorded as a miss. Switching to modal masks raised measured raw SAM3 mask@0.5 recall
    substantially, and the more occluded the capture the larger the difference.

    **Instances below `min_visible_fraction` are excluded from GT entirely.**
    BOP considers only instances with at least 10% of their projected surface visible
    (consistent across BOP Challenge 2018-2024), hence the 0.1 default in `config.py`. The
    reasoning is that a ~1000-pixel observation cannot support a 5 mm pose claim, and an
    instance with zero visible pixels would otherwise be an unavoidable false negative that
    measures occlusion rather than detector quality.

    Excluded instances are in neither the numerator nor the denominator of recall: they
    cannot become false negatives, but a prediction that finds one counts as a false
    positive, since there is no GT left to match. That is a real trade -- pass
    `--min-visible-fraction 0.0` to credit those detections instead. Measured, the 0.1 cut
    excludes on the order of 1% of instances and the kept population's 1st-percentile visibility
    sits well above the threshold, so it cuts a sparse tail rather than slicing through a dense
    region. Worth re-checking on a capture with heavier occlusion, where that may not hold.

    **Visibility is measured here, not read from the dataset.**
    BOP normally thresholds `scene_gt_info.json`'s `visib_fract`, but some captures do not
    populate it -- it reads 1.0 for every instance, including ones measured at 3% visible. So
    the fraction is computed directly as `visible_px / amodal_px` from the z-buffer above.
    Same quantity BOP defines, derived rather than trusted.

    Note the pose metric is unaffected by any of this: symmetry-aware max-vertex error (BOP's
    MSSD) compares full CAD vertex sets under two poses in 3D and never consults a mask. This
    function only decides *which* predictions get scored, never *how*.

    Returns `(kept_entries, masks, boxes, camera_matrix, visible_fractions)`. Indices into
    `kept_entries` are what `matched_pose_metrics.gt_index` refers to -- any other GT lookup
    that filters differently will misalign them (see `generate_report.aggregate_outputs`).
    """
    from foundationpose_perception_pipeline.geometry import bbox_from_mask, rasterize_mesh, render_mask

    frame_key = str(target.im_id)
    camera_matrix = np.asarray(renderer.scene_camera(target)[frame_key]["cam_K"], dtype=np.float64).reshape(3, 3)
    scene_entries = renderer.scene_gt(target)[frame_key]
    width, height = image_size

    zbuf_mm = np.full((height, width), np.inf, dtype=np.float64)
    obj_map = np.zeros((height, width), dtype=np.int32)
    inst_map = np.zeros((height, width), dtype=np.int32)
    for index, entry in enumerate(scene_entries):
        obj_id = int(entry["obj_id"])
        vertices, faces = renderer.mesh(target.dataset, obj_id)
        rotation = np.asarray(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(entry["cam_t_m2c"], dtype=np.float64)
        rasterize_mesh(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces),
            rotation,
            translation,
            camera_matrix,
            zbuf_mm,
            obj_map,
            inst_map,
            obj_id,
            index + 1,
            GT_RASTERIZER_NEAR_MM,
        )

    kept_entries: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    visible_fractions: list[float] = []
    target_vertices, target_faces = renderer.mesh(target.dataset, target.obj_id)
    for index, entry in enumerate(scene_entries):
        if int(entry["obj_id"]) != target.obj_id:
            continue
        visible_mask = inst_map == (index + 1)
        rotation = np.asarray(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(entry["cam_t_m2c"], dtype=np.float64)
        amodal_mask = render_mask(target_vertices, target_faces, rotation, translation, camera_matrix, image_size)
        amodal_count = int(amodal_mask.sum())
        visible_fraction = float(visible_mask.sum()) / amodal_count if amodal_count else 0.0
        if not visible_mask.any() or visible_fraction < min_visible_fraction:
            continue
        kept_entries.append(entry)
        masks.append(visible_mask)
        boxes.append(bbox_from_mask(visible_mask))
        visible_fractions.append(visible_fraction)
    return kept_entries, masks, np.asarray(boxes, dtype=np.float64), camera_matrix, visible_fractions
