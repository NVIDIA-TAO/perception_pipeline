#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone smoke test that SAM3 is installed and can run end to end.

Does not depend on a BOP dataset being downloaded: it builds the SAM3 model (loading
the gated checkpoint from Hugging Face Hub), runs a single text-prompted
inference on a synthetic image, and checks that a mask comes back. Intended as
the "Verify the install" step in README.md.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from perception_pipeline.runtime import inference_context, tensor_to_numpy


def make_synthetic_image(size: int = 1024) -> Image.Image:
    """Draw a simple shape on a plain background.

    This is only meant to give the forward pass something to run on. SAM3 is trained on
    natural images, so a flat synthetic doodle is out-of-distribution and will legitimately
    score low/no confident detections -- that alone does not indicate a broken install, so
    this script does not gate success on detection confidence, only on the pipeline running
    end to end and returning well-formed tensors.
    """
    image = Image.new("RGB", (size, size), color=(60, 60, 60))
    draw = ImageDraw.Draw(image)
    margin = size // 4
    draw.rectangle([margin, margin, size - margin, size - margin], fill=(230, 230, 230))
    return image


def installed_sam3_commit() -> str | None:
    """Return the git commit of the editable sam3 checkout, or None if it cannot be resolved.

    sam3 is installed editable from a sibling checkout, so `sam3.__file__` points into a working
    tree and `git -C` answers directly. Returns None rather than raising: an unresolvable commit is
    a reporting gap, not a reason to fail an install check.
    """
    try:
        import sam3

        source = Path(sam3.__file__).resolve().parent
    except Exception:  # noqa: BLE001 -- any import/attribute failure means "cannot resolve"
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            # Not check=True: a non-git directory is a reporting gap, not an install failure.
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--no-hf", action="store_true", help="Load from --checkpoint-path instead of HF Hub.")
    parser.add_argument("--prompt", default="square", help="Text prompt to run against the synthetic image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    commit = installed_sam3_commit()
    print(f"sam3 source commit: {commit[:12] if commit else 'could not resolve'}")

    try:
        model = build_sam3_image_model(
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            load_from_HF=not args.no_hf,
        )
    except Exception as exc:
        message = str(exc)
        if "Cannot access gated repo" in message or "GatedRepoError" in type(exc).__name__:
            raise SystemExit(
                "SAM3 checkpoint access failed. Request access to "
                "https://huggingface.co/facebook/sam3, then run `hf auth login`, "
                "or pass --checkpoint-path /path/to/sam3.pt --no-hf."
            ) from exc
        raise

    # confidence_threshold=0.0 so the raw proposal set comes back regardless of score; a
    # synthetic doodle isn't expected to score highly (see make_synthetic_image docstring),
    # this is checking that inference runs and returns well-formed output, not accuracy.
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.0)

    image = make_synthetic_image()
    with inference_context(args.device):
        state: dict[str, Any] = processor.set_image(image)
        state = processor.set_text_prompt(prompt=args.prompt, state=state)

    boxes = tensor_to_numpy(state["boxes"])
    scores = tensor_to_numpy(state["scores"])
    masks = tensor_to_numpy(state["masks"][:, 0]) if "masks" in state else np.empty((0,))

    max_score = scores.max() if len(scores) else 0.0
    print(f"prompt={args.prompt!r} -> {len(boxes)} proposal(s), max score={max_score:.3f}")

    if len(boxes) == 0 or masks.shape[0] != len(boxes) or masks.shape[-2:] != image.size[::-1]:
        raise SystemExit(
            "SAM3 ran but the output tensors look malformed (missing proposals, or "
            "boxes/masks count or mask resolution mismatch) -- install looks broken, not just "
            "a low-confidence prompt. Check the traceback-free run above for a real error."
        )

    print("SAM3 OK: model loaded from HF, forward pass ran, output tensors well-formed.")
    print(
        "(Low max score is expected here -- this is a synthetic doodle, not a natural image. "
        "Run the pipeline on a couple of real scenes to sanity-check detection quality: "
        "`script/run_pipeline.py --dataset <name> --max-scenes 2`.)"
    )


if __name__ == "__main__":
    main()
