#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extension point: how a site adds a depth backend without editing this package.

The pipeline ships one depth backend -- a TAO `deployable_*` export as a TensorRT engine -- and
one depth source, the stereo pair. A site that has another model, another interpreter, or another
way of obtaining depth should not have to patch `inference/depth.py` to use it, because a patched
checkout is a checkout that cannot be updated.

So both are **registries**, and this module is the loader that lets something outside the package
fill them in:

    # perception_pipeline_extras/__init__.py, anywhere on PYTHONPATH
    from perception_pipeline.inference.depth import register_backend

    register_backend("mine", claims=lambda model: str(model).endswith(".mine"), generate=...)

`load_extensions()` imports every module named in `$PERCEPTION_PIPELINE_EXTENSIONS`
(comma-separated), then `perception_pipeline_extras` if it is importable. Neither has to exist --
a plain checkout has neither and loads nothing.

**Import, not entry points.** A package that registers on import needs no install step and no
metadata, which matters because the thing being registered is usually a local experiment rather
than a published distribution. The cost is that the module must be on `PYTHONPATH`; the benefit
is that putting it there is the whole installation.

Registration happens once, at the first call. Every entry point calls this before parsing
arguments, because the registries decide what the choices for `--depth-backend` and
`--depth-source` even are.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

ENVIRONMENT_VARIABLE = "PERCEPTION_PIPELINE_EXTENSIONS"
# Set to a non-empty value to load nothing, whatever is importable. Two uses: reproducing the
# behaviour of a plain checkout on a machine that has extensions installed, and checking a
# release export on the machine that produced it -- where the extensions are, by construction,
# right there on `sys.path`.
DISABLE_VARIABLE = "PERCEPTION_PIPELINE_NO_EXTENSIONS"
# Imported if present, so a site can register by putting a package of this name on PYTHONPATH
# rather than by setting an environment variable on every invocation.
CONVENTIONAL_MODULE = "perception_pipeline_extras"

_loaded = False


def extension_modules() -> list[str]:
    """Return the module names `load_extensions` will try, in order."""
    configured = os.environ.get(ENVIRONMENT_VARIABLE, "")
    names = [name.strip() for name in configured.split(",") if name.strip()]
    if CONVENTIONAL_MODULE not in names:
        names.append(CONVENTIONAL_MODULE)
    return names


def load_extensions(force: bool = False) -> list[str]:
    """Import the extension modules, and return the names that were actually loaded.

    Idempotent: importing twice would register twice, and a duplicate registration is an error
    rather than a no-op, so the first call wins and later ones are free.

    A module that is simply absent is not an error -- that is the normal case for a plain
    checkout. A module that exists and *fails* to import is an error, because a site that asked
    for an extension and silently did not get it would be scoring a different configuration than
    it thinks.
    """
    global _loaded  # noqa: PLW0603 -- one-shot module load guard; a process-wide latch
    if _loaded and not force:
        return []
    _loaded = True

    if os.environ.get(DISABLE_VARIABLE):
        logger.debug("extensions disabled by %s", DISABLE_VARIABLE)
        return []

    loaded: list[str] = []
    for name in extension_modules():
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            # Only when the extension itself is missing. A missing dependency *inside* it names
            # a different module, and that has to surface.
            if error.name != name:
                raise
            continue
        loaded.append(name)
        logger.debug("loaded extension %s", name)
    return loaded
