#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert ASCII PLY meshes to binary little-endian.

Several BOP datasets ship their CAD meshes as ASCII PLY -- T-LESS's `models_cad/` is one, often
alongside a `models_eval/` that is already binary. Two consumers cannot read ASCII: `foundationpose_perception_pipeline.geometry.read_binary_little_endian_ply`, which parses the
header and then `np.frombuffer`s the body, and FoundationPose, which is handed the `.ply` path
directly. Converting once while adapting the dataset is the fix; nothing downstream then has to
care, and a dataset that already ships binary passes through unchanged.

Nothing here is dataset-specific: `read_ply` accepts either format and `convert` always writes
binary, so an adapter can call it unconditionally rather than testing first.

Vertex normals are preserved when present -- they cost little and FoundationPose's renderer can
use them.

    ./.venv/bin/python tools/bop_adapt/ply.py <src dir> <dst dir>
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np


def read_ply(path: Path):
    """Read an ASCII or binary-little-endian PLY. Returns (vertex array, props, faces)."""
    with path.open("rb") as f:
        header, fmt = [], None
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            if line == "end_header":
                break
        n_vert = n_face = 0
        in_vertex = False
        props: list[str] = []
        for line in header:
            p = line.split()
            if p[:2] == ["element", "vertex"]:
                n_vert, in_vertex = int(p[2]), True
            elif p[:2] == ["element", "face"]:
                n_face, in_vertex = int(p[2]), False
            elif p and p[0] == "property" and in_vertex:
                props.append(p[2])

        if fmt == "ascii":
            verts = np.empty((n_vert, len(props)), dtype=np.float32)
            for i in range(n_vert):
                verts[i] = [float(v) for v in f.readline().split()[: len(props)]]
            faces = []
            for _ in range(n_face):
                vals = f.readline().split()
                idx = [int(v) for v in vals[1 : 1 + int(vals[0])]]
                faces.append(idx)
        elif fmt == "binary_little_endian":
            dt = np.dtype([(p, "<f4") for p in props])
            raw = np.frombuffer(f.read(dt.itemsize * n_vert), dtype=dt, count=n_vert)
            verts = np.stack([raw[p] for p in props], axis=1).astype(np.float32)
            faces = []
            for _ in range(n_face):
                count = struct.unpack("<B", f.read(1))[0]
                faces.append(list(struct.unpack("<" + "i" * count, f.read(4 * count))))
        else:
            raise ValueError(f"unsupported PLY format {fmt!r} in {path}")
    return verts, props, faces


def write_binary_ply(path: Path, verts: np.ndarray, props: list[str], faces: list[list[int]]) -> None:
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(verts)}"]
    header += [f"property float {p}" for p in props]
    header += [f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(verts, dtype="<f4").tobytes())
        for face in faces:
            f.write(struct.pack("<B", len(face)))
            f.write(struct.pack("<" + "i" * len(face), *face))


def convert(src: Path, dst: Path) -> str:
    verts, props, faces = read_ply(src)
    write_binary_ply(dst, verts, props, faces)
    return f"{src.name}: {len(verts)} verts ({','.join(props)}), {len(faces)} faces"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <src dir> <dst dir>")
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    for ply in sorted(src_dir.glob("*.ply")):
        print(convert(ply, dst_dir / ply.name))
