# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""depth -> SAM3 -> FoundationPose evaluation pipeline for BOP-format datasets.

Layered bottom-up, with no import cycles:

- ``config``     -- YAML config loading (``config/defaults.yaml`` plus a dataset profile)
- ``io``         -- BOP and file I/O
- ``geometry``   -- PLY I/O, both rasterizers, IoU, greedy matching
- ``dataset``    -- evaluation targets and prompts; no ground truth
- ``pose``       -- FoundationPose estimator lifecycle and CAD rendering
- ``inference``  -- the stages that produce predictions, and ``stereo`` beneath them for depth
- ``evaluation`` -- ground truth (``gt``), detection and pose error, depth error, reports

``inference`` never imports ``evaluation``; that isolation is what lets ``infer.py`` run on a
capture with no ``scene_gt.json``. A static import-graph check enforces it.

Entry points live in ``script/`` and are not part of this package.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
