#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone smoke test that FoundationPose is built and its shared library actually loads.

Does not build a TensorRT engine and does not need a CAD mesh or any dataset: it resolves
`libfoundation_pose_nvidia.so` the same way the real pipeline does (`pose.ensure_foundationpose_paths`),
loads it via ctypes, prints its build info, and calls `fp_synchronize_device` to confirm the CUDA
context actually works. Intended as the "Verify the install" step in README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from perception_pipeline.config import FOUNDATIONPOSE_ROOT_DEFAULT
from perception_pipeline.pose import ensure_foundationpose_paths

# FoundationPose is an external checkout, not a pip distribution, so its Python bindings have to
# be put on the path explicitly -- `script/run_pipeline.py` does the same thing at module scope.
# Without this the import below fails even though the library itself is present and fine, which
# reads as "FoundationPose is broken" when it is only "the bindings are not importable yet".
if FOUNDATIONPOSE_ROOT_DEFAULT is not None:
    _bindings = FOUNDATIONPOSE_ROOT_DEFAULT / "python" / "src"
    if _bindings.is_dir() and str(_bindings) not in sys.path:
        sys.path.insert(0, str(_bindings))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-root", type=Path, default=FOUNDATIONPOSE_ROOT_DEFAULT)
    parser.add_argument("--fp-library", type=Path, default=None)
    parser.add_argument("--fp-refine-model-path", type=Path, default=None)
    parser.add_argument("--fp-score-model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validates the checkout layout and sets FP_LIBRARY, same as every real entry-point script.
    # Raises with the FOUNDATIONPOSE_ROOT hint when the root is unset or missing.
    library, refine, score = ensure_foundationpose_paths(args)
    print(f"foundationpose root: {args.foundationpose_root.expanduser().resolve()}")
    print(f"library:             {library}")
    print(f"refine model:        {refine}")
    print(f"score model:         {score}")

    try:
        from foundation_pose_nvidia import load_library
    except ImportError as exc:
        raise SystemExit(
            "Could not import foundation_pose_nvidia. Put its Python bindings on PYTHONPATH, "
            "e.g. PYTHONPATH=.:<foundationpose-root>/python/src:../sam3"
        ) from exc

    try:
        lib = load_library()  # prints build_info; loads via FP_LIBRARY under the hood
    except OSError as exc:
        raise SystemExit(
            f"ctypes failed to load {library}:\n{exc}\n\n"
            "This is almost always a missing transitive shared library (libcudart, "
            "libnvinfer, libnvonnxparser), not a problem with this script. Run "
            f"`ldd {library}` and grep for 'not found', then make sure LD_LIBRARY_PATH "
            "includes wherever those .so files live in .venv (tensorrt_libs/, nvidia/cu13/lib/)."
        ) from exc

    lib.synchronize()
    print("FoundationPose OK: library loaded, build info printed above, CUDA device synchronized.")


if __name__ == "__main__":
    main()
