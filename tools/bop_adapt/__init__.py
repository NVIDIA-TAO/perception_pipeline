#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn a BOP dataset as downloaded into the plain layout this pipeline runs on.

Every dataset needs a different adaptation and they all have to produce the SAME tree, which is
what this package splits apart:

- `emit` writes that tree, and is the part no adapter should reimplement. The contract it
  enforces -- im_id 0 is the base camera, ids are contiguous, meshes are binary PLY, collected
  depth is uint16 millimetres -- is what the rest of the pipeline assumes everywhere.
- `partners` selects stereo partners for a dataset whose "cameras" are frames of a static scene.
- `ply` converts meshes.
- `datasets/<name>.py` holds what is genuinely specific to one dataset, and that is real code
  rather than a flag. Real examples from datasets this has been pointed at: ground truth
  annotated in only one sensor's frame, so every pose needs transforming into the base
  camera's; 16-bit TIFF images needing one shared linear map to 8-bit, because a per-image
  stretch breaks the photometric consistency the stereo matcher needs; depth delivered at a
  different resolution than the base camera, needing reprojection. None of that is
  expressible as a parameter.

Add a dataset by writing `datasets/<name>.py` with `add_arguments(parser)` and `adapt(...)`, then
naming it in `REGISTRY` in `adapt.py`. The module is chosen by the profile's `dataset.name`, so
`--config <name>` is normally all a caller passes.
"""
