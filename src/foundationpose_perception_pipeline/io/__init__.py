#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset adapters: the only code that knows what a BOP directory looks like.

Everything else -- inference and evaluation both -- works on `SceneInput` and `ObjectSpec`.
Confining format knowledge here is what lets a different capture format be supported by adding
a module beside `bop.py` rather than by threading paths through the pipeline.

Deliberately re-exports nothing. `bop` imports `dataset`, and `dataset` imports `files`, so an
eager re-export here would make importing *any* of them circular. Import the module you want:

    from foundationpose_perception_pipeline.io.bop import object_specs_for_scene
    from foundationpose_perception_pipeline.io.files import load_json
"""
