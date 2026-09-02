#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt a BOP dataset into the layout this pipeline runs on.

    ./.venv/bin/python tools/bop_adapt/adapt.py --config tless --src <path to tless>

The dataset module is chosen by the profile's `dataset.name`, so `--config <name>` normally
selects everything: which adapter runs, where the adapted tree is written, and where the
collected depth goes. That is deliberate -- the adapter and the pipeline reading its output must
not be able to disagree about where the data lives, and they cannot if both read one profile.

Each module contributes its own flags, so `--help` shows the selected dataset's options. See
`bop_adapt/__init__.py` for how to add one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))
sys.path.insert(0, str(PIPELINE_ROOT / "tools"))

from bop_adapt.datasets import tless  # noqa: E402
from foundationpose_perception_pipeline.config import add_config_argument, settings_from_argv  # noqa: E402

REGISTRY = {module.NAME: module for module in (tless,)}


def main() -> None:
    settings = settings_from_argv()
    dataset = settings.dataset.name
    module = REGISTRY.get(dataset)

    parser = argparse.ArgumentParser(
        description=f"{__doc__}\nSelected dataset: {dataset}\n\n"
                    f"{module.__doc__ if module else 'No adapter is registered for this dataset.'}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)
    parser.add_argument("--src", type=Path, required=True,
                        help="The dataset as downloaded. No default -- nothing can guess where "
                             "your copy is.")
    # Defaults come from the ACTIVE PROFILE, so the adapter and the pipeline cannot disagree
    # about where the adapted data lives.
    parser.add_argument("--out-root", type=Path, default=settings.dataset.root)
    parser.add_argument("--depth-root", type=Path, default=settings.dataset.collected_depth_root)
    parser.add_argument("--name", default=settings.dataset.name)
    parser.add_argument("--object-names", type=Path, default=None,
                        help="Override the obj_id -> name table. Defaults to "
                             "config/<name>/object_names.json.")
    # Only when there is one -- a profile with no adapter still gets `--help`, listing the flags
    # every dataset shares. Resolving the adapter before building the parser meant `--help` died
    # on the missing adapter instead of printing anything, which is the wrong answer to a request
    # for documentation.
    if module is not None:
        module.add_arguments(parser)
    args = parser.parse_args()
    if module is None:
        raise SystemExit(
            f"No adapter for dataset {dataset!r}. Registered: {', '.join(sorted(REGISTRY)) or 'none'}. "
            f"The adapter is selected by the profile's `dataset.name`; add a module under "
            f"tools/bop_adapt/datasets/ and register it in adapt.py."
        )

    if args.depth_root is None:
        raise SystemExit(
            "No collected-depth root to write to. The adapter emits the base frame's sensor depth "
            "as well as the scene tree, so it needs a destination: set "
            "`dataset.collected_depth_root` in the config profile, or pass --depth-root."
        )
    dataset_dir = args.out_root / args.name
    out_split = dataset_dir / "test"
    depth_split = args.depth_root / args.name / "test"
    out_split.mkdir(parents=True, exist_ok=True)
    depth_split.mkdir(parents=True, exist_ok=True)

    module.adapt(args, dataset_dir=dataset_dir, out_split=out_split,
                 depth_split=depth_split, pipeline_root=PIPELINE_ROOT)


if __name__ == "__main__":
    main()
