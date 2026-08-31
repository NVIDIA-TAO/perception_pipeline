#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T-LESS (primesense test split, BOP19 subset) -> the plain layout this pipeline runs on.

T-LESS arrives closer to ready than most BOP datasets: it is already in the plain layout, with
`cam_R_w2c`/`cam_t_w2c` in `scene_camera.json` and a scene that does not move between frames
(measured world-pose drift 0.00 mm). Two things still need doing, and both live here rather than
in `emit` because they are properties of THIS dataset:

1. **RE-KEY SO THE CHOSEN FRAME IS im_id 0.** T-LESS numbers its frames non-contiguously
   (1, 17, 30, 38, ...), so one adapted scene = one (source scene, base frame) pair, with the
   base re-keyed to 0 and its partners to 1..K. That also multiplies the evaluation set: the
   pipeline scores one frame per scene, so 20 source scenes would otherwise give 20 scored
   frames.
2. **CHOOSE WHICH PARTNERS TO EXPOSE.** A rig with one usable partner leaves nothing to decide;
   T-LESS offers ~37 of 49, so partner choice becomes a real tunable and it is the one that sets
   depth precision. See `bop_adapt.partners` for the band and why it is a cliff rather than a
   slope.

`models_cad` ships as ASCII PLY, which `emit.write_models` converts unconditionally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bop_adapt import emit
from bop_adapt.partners import centres, partners_in_band

NAME = "tless"


def add_arguments(parser: Any) -> None:
    parser.add_argument("--src-split", default="test_primesense",
                        help="Split directory under --src. T-LESS has NO `val`: it is a "
                             "BOP-2019-era dataset whose TEST GT is public, so the primesense "
                             "test split is the one with public annotations.")
    parser.add_argument("--baseline-min", type=float, default=0.10)
    parser.add_argument("--baseline-max", type=float, default=0.20)
    parser.add_argument("--max-partners", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=5,
                        help="Base frames sampled per source scene. A narrower baseline band "
                             "leaves more base frames with no partner in it, so a smaller stride "
                             "is what keeps the adapted scene count up.")
    parser.add_argument("--max-scenes", type=int, default=None)


def adapt(args: Any, *, dataset_dir: Path, out_split: Path, depth_split: Path,
          pipeline_root: Path) -> None:
    src_split = args.src / args.src_split
    if not src_split.is_dir():
        raise SystemExit(
            f"{src_split} does not exist. --src should be the directory holding models_cad/ and "
            f"{args.src_split}/, i.e. T-LESS as downloaded."
        )

    n_meshes = emit.write_models(
        dataset_dir=dataset_dir,
        models_src=args.src / "models_cad",
        models_eval_src=args.src / "models_eval",
    )
    print(f"converted {n_meshes} CAD meshes to binary PLY")

    names = emit.load_object_names(pipeline_root, args.name, args.object_names)
    models_info = json.loads((args.src / "models_eval" / "models_info.json").read_text())
    emit.write_dataset_map(dataset_dir, names, models_info)

    scenes = sorted(p for p in src_split.iterdir() if p.is_dir())
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    index: list[dict[str, Any]] = []
    out_id = skipped = 0
    for scene_dir in scenes:
        cam = json.loads((scene_dir / "scene_camera.json").read_text())
        gt = json.loads((scene_dir / "scene_gt.json").read_text())
        gti = json.loads((scene_dir / "scene_gt_info.json").read_text())
        keys = sorted(cam, key=int)
        C = centres(cam, keys)
        # MEASURED from the frames, not assumed. The feasibility check below is the depth stage's
        # own, and it is only equivalent to what the depth stage will do if it is handed the size
        # the depth stage will see. Hardcoding one is how that silently stops being true -- the
        # constant here previously said 1280x1024, which is not this dataset's size at all.
        height, width = emit.read_image_size(scene_dir / "rgb" / f"{int(keys[0]):06d}.png")

        for bi in range(0, len(keys), args.frame_stride):
            base = keys[bi]
            partners, baselines = partners_in_band(
                cam, keys, C, bi,
                baseline_min=args.baseline_min, baseline_max=args.baseline_max,
                max_partners=args.max_partners, width=width, height=height,
            )
            if not partners:
                skipped += 1
                continue

            emit.write_scene(
                out_dir=out_split / f"{out_id:06d}",
                chosen=[base, *partners],
                src_rgb=scene_dir / "rgb",
                cam=cam, gt=gt, gt_info=gti,
            )
            emit.write_collected_depth(
                depth_split=depth_split, out_id=out_id,
                raw=emit.read_depth(scene_dir / "depth" / f"{int(base):06d}.png"),
                depth_scale=cam[base]["depth_scale"],
            )
            index.append({
                "out_scene": out_id,
                "src_scene": int(scene_dir.name),
                "base_frame": int(base),
                "partner_frames": [int(p) for p in partners],
                "baselines_m": baselines,
            })
            out_id += 1

    emit.write_scene_index(dataset_dir, index)
    print(f"wrote {out_id} adapted scenes to {out_split}"
          f"  ({skipped} base frames had no HORIZONTALLY-RECTIFYING partner in "
          f"[{args.baseline_min}, {args.baseline_max}] m)")
    if index:
        allb = [b for r in index for b in r["baselines_m"]]
        mean_partners = sum(len(r["partner_frames"]) for r in index) / len(index)
        print(f"partners per scene: {mean_partners:.1f}  "
              f"baseline {min(allb):.3f}-{max(allb):.3f} m")
