#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stereo pair selection and rectification for the commercial depth path.

Every constant and branch below was arrived at by measurement, and each one produced a
plausible-looking depth map while being centimetres or metres wrong:

- ``alpha=-1`` in ``stereoRectify``, never 0 (0.63 px vs 159 px median epipolar residual on this
  rig, because alpha=0 crops each view to a *different* valid region);
- the caller passes **copies** of the intrinsics/extrinsics into :func:`rectify_scene`, because it
  rewrites them in place, and keeps the true base-camera K -- substituting the rectified principal
  point for the real one translates the whole depth map (0.80 mm -> 11.64 mm object median);
- the base camera must be the *left* camera of the rectified pair, or disparity comes back
  near-zero instead of negative (6-8 m of depth error). :func:`select_partner_camera` finds a
  horizontally-rectifying partner; ``stereo/depth.py`` handles the handedness by mirroring.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from scipy.spatial import transform

logger = logging.getLogger(__name__)

# Bounds on what counts as a horizontally-rectifying pair. A pair that fails these is not merely
# suboptimal -- FoundationStereo predicts horizontal disparity only, so a vertical pair yields a
# depth map that looks fine and is wrong by metres.
DEFAULT_MIN_BASELINE_X: float = 0.01
DEFAULT_MAX_VERTICAL_PARALLAX: float = 1e-4
DEFAULT_MAX_DEPTH_PARALLAX: float = 1e-4


def compute_rotation_errors(poses_l: np.ndarray, poses_r: np.ndarray) -> np.ndarray:
    """Return the rotation error, in degrees, between two batches of rotation matrices."""
    delta_rotation = np.matmul(poses_l[:, :3, :3], np.transpose(poses_r[:, :3, :3], axes=(0, 2, 1)))
    rotation = transform.Rotation.from_matrix(delta_rotation)
    return np.linalg.norm(transform.Rotation.as_rotvec(rotation, degrees=True), axis=1)


def compute_stereo_rectification(
    im_size: tuple[int, int],
    r_mat: np.ndarray,
    t_vec: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    dist_coeffs1: np.ndarray | None = None,
    dist_coeffs2: np.ndarray | None = None,
    alpha: float = -1.0,
    align_optical_axis: bool = True,
) -> dict[str, Any]:
    """Compute rectification transforms, projection matrices and remap tables for a pair.

    `alpha=-1` keeps the full field of view and is not a default to be tuned: at `alpha=0`
    OpenCV crops each view to its own valid region, the two crops disagree, and the pair stops
    being a stereo pair at all (median epipolar residual 159 px against 0.63 px at -1).
    """
    # OpenCV 5.0's stereoRectify requires T as a (3, 1) column; a flat (3,) array fails deep
    # inside gemm with "a_size.width == len". Reshape defensively so callers can pass either.
    t_vec = np.asarray(t_vec, dtype=np.float64).reshape(3, 1)
    r1, r2, p1, p2, dsp2dm, _, _ = cv2.stereoRectify(
        cameraMatrix1=k1,
        distCoeffs1=dist_coeffs1,
        cameraMatrix2=k2,
        distCoeffs2=dist_coeffs2,
        imageSize=im_size,
        R=r_mat,
        T=t_vec,
        R1=None,
        R2=None,
        P1=None,
        P2=None,
        Q=None,
        flags=cv2.CALIB_ZERO_DISPARITY if align_optical_axis else 0,
        alpha=alpha,
    )

    left_map_x, left_map_y = cv2.initUndistortRectifyMap(k1, dist_coeffs1, r1, p1, im_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(k2, dist_coeffs2, r2, p2, im_size, cv2.CV_32FC1)

    # `T` here is SYNTHESISED as [-1/Q[3,2], 0, 0] -- horizontal by construction, for every pair
    # including physically vertical ones. Do not use it to decide whether a pair is horizontal;
    # `select_partner_camera` compares the rectified camera *poses* instead.
    r_rect = np.eye(3)
    t_rect = np.zeros((3, 1))
    t_rect[0][0] = -1.0 / dsp2dm[3, 2]
    k_rect = np.array(
        [
            [dsp2dm[2, 3], 0.0, -dsp2dm[0, 3]],
            [0.0, dsp2dm[2, 3], -dsp2dm[1, 3]],
            [0.0, 0.0, 1.0],
        ]
    )

    return {
        "K": k_rect,
        "R": r_rect,
        "T": t_rect,
        "dsp2dm": dsp2dm,
        "R1": r1,
        "R2": r2,
        "rect_maps": [[left_map_x, left_map_y], [right_map_x, right_map_y]],
        "P1": p1,
        "P2": p2,
    }


def compute_rectification_poses(
    idx_l: int,
    idx_r: int,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(R1, R2, rectified left pose, rectified right pose)` for two cameras."""
    num_cameras = intrinsics.shape[0]
    if not (idx_l < num_cameras and idx_r < num_cameras):
        raise ValueError(f"Camera index out of range: {idx_l}, {idx_r} for {num_cameras} cameras")

    cam_l_pose = np.concatenate((extrinsics[idx_l], np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)
    cam_r_pose = np.concatenate((extrinsics[idx_r], np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)

    rt = cam_r_pose @ np.linalg.inv(cam_l_pose)
    if np.linalg.norm(rt[:3, 3]) < 1e-6:
        raise ValueError(f"Cameras {idx_l} and {idx_r} have zero/negligible baseline distance.")

    rect = compute_stereo_rectification(
        im_size=(image_width, image_height),
        r_mat=rt[:3, :3],
        t_vec=rt[:3, 3],
        k1=intrinsics[idx_l],
        k2=intrinsics[idx_r],
    )

    r1 = np.eye(4)
    r1[:3, :3] = rect["R1"]
    r2 = np.eye(4)
    r2[:3, :3] = rect["R2"]
    return r1, r2, r1 @ cam_l_pose, r2 @ cam_r_pose


def predicted_max_disparity_px(
    focal_px: float, baseline_m: float, image_width: int, model_width: int, min_working_distance_m: float
) -> float:
    """Disparity, in the MODEL's own pixels, that the nearest expected surface would produce.

    The model only ever sees `model_width` columns, so its search range is spent at that scale
    rather than at the rectified image's. Comparing against the range has to happen here too.
    """
    return focal_px * (model_width / float(image_width)) * baseline_m / max(min_working_distance_m, 1e-6)


def select_partner_camera(
    base_camera: int,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    min_working_distance_m: float | None = None,
    model_width: int | None = None,
    max_disparity_px: float = 416.0,
) -> tuple[int, np.ndarray, float] | None:
    """Pick the camera forming the best horizontal stereo pair with `base_camera`.

    Returns `(partner_id, rectified_pose_difference, rotation_error_deg)`, or None if no partner
    rectifies horizontally.

    Searching partners for a *fixed* base, rather than searching all pairs, is required: depth is
    only warpable back to the left camera of the rectified pair, so a globally-best pair that
    excludes the base camera is useless here.

    Horizontality is judged from the rectified camera poses, not from the rectification's `T` --
    see the comment in `compute_stereo_rectification`. This matters concretely: the capture rig
    re-assigns camera ids per scene, so cameras 0 and 3 form a horizontal pair in one scene and a
    vertical one in the next.
    """
    best: tuple[bool, float, int, np.ndarray, float] | None = None
    for partner in range(intrinsics.shape[0]):
        if partner == base_camera:
            continue
        try:
            r1, r2, base_pose, partner_pose = compute_rectification_poses(
                idx_l=base_camera,
                idx_r=partner,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_width=image_width,
                image_height=image_height,
            )
        except (ValueError, cv2.error):
            continue
        difference = base_pose[:3, 3] - partner_pose[:3, 3]
        horizontal = (
            abs(difference[0]) > DEFAULT_MIN_BASELINE_X
            and abs(difference[1]) < DEFAULT_MAX_VERTICAL_PARALLAX
            and abs(difference[2]) < DEFAULT_MAX_DEPTH_PARALLAX
        )
        if not horizontal:
            continue
        rotation_error = float(
            max(
                compute_rotation_errors(r1[:3, :3][np.newaxis, ...], np.eye(3)[np.newaxis, ...])[0],
                compute_rotation_errors(r2[:3, :3][np.newaxis, ...], np.eye(3)[np.newaxis, ...])[0],
            )
        )
        # Feasibility first, rotation error second. A partner whose baseline would push the
        # nearest expected surface past the model's disparity range returns a CLAMPED disparity --
        # a lower bound, which `Z = f*B/d` turns into depth biased too far, in a map that still
        # looks plausible. Ranking rather than filtering is deliberate: on a rig where many
        # scenes have exactly one horizontally-rectifying partner, dropping the infeasible ones
        # would leave those scenes with no pair at all and no depth.
        feasible = True
        if min_working_distance_m and model_width:
            predicted = predicted_max_disparity_px(
                focal_px=float(intrinsics[base_camera][0, 0]),
                baseline_m=abs(float(difference[0])),
                image_width=image_width,
                model_width=model_width,
                min_working_distance_m=min_working_distance_m,
            )
            feasible = predicted <= max_disparity_px
        # Sort key: feasible before infeasible, then shorter baseline among infeasible ones (it
        # saturates less), then lowest rotation error.
        key = (not feasible, abs(float(difference[0])) if not feasible else 0.0, rotation_error)
        if best is None or key < (not best[0], best[1], best[4]):
            best = (feasible, key[1], partner, difference, rotation_error)
    if best is None:
        return None
    if not best[0]:
        logger.warning(
            "camera %d is the best available partner but its %.3f m baseline exceeds the model's "
            "disparity range at %.2f m; the pre-shift will have to cover it",
            best[2], abs(float(best[3][0])), min_working_distance_m,
        )
    return (best[2], best[3], best[4])


def convert_to_homogeneous_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a (..., 3, 4) transform to (..., 4, 4); pass (..., 4, 4) through unchanged."""
    if matrix.shape[-2:] == (4, 4):
        return matrix
    if matrix.shape[-2:] != (3, 4):
        raise ValueError(f"Expected transform shape (..., 3, 4) or (..., 4, 4), got {matrix.shape}")
    batch_shape = matrix.shape[:-2]
    zeros = np.zeros([*list(batch_shape), 1, 3], dtype=matrix.dtype)
    ones = np.ones([*list(batch_shape), 1, 1], dtype=matrix.dtype)
    return np.concatenate([matrix, np.concatenate([zeros, ones], axis=-1)], axis=-2)


def rectify_images(
    rect_maps: Sequence[Sequence[np.ndarray]],
    img_l: np.ndarray,
    img_r: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply precomputed rectification maps to a pair of images."""
    return (
        cv2.remap(img_l, rect_maps[0][0], rect_maps[0][1], interpolation),
        cv2.remap(img_r, rect_maps[1][0], rect_maps[1][1], interpolation),
    )


class RectifiedPair:
    """The rectified stereo pair and the geometry needed to get depth back to the base camera.

    A small class rather than the 15-tuple the original returns: this path has one caller, and
    every field below is used by it. `k_rect` and `r1` are what `depth_rectified_to_base` needs;
    `baseline_x_m` is what turns disparity into metres.
    """

    __slots__ = ("baseline_x_signed_m", "k_rect", "left", "r1", "right")

    def __init__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        k_rect: np.ndarray,
        r1: np.ndarray,
        baseline_x_signed_m: float,
    ) -> None:
        self.left = left
        self.right = right
        self.k_rect = k_rect
        self.r1 = r1
        self.baseline_x_signed_m = baseline_x_signed_m


def rectify_pair(
    img_l: np.ndarray,
    img_r: np.ndarray,
    intrinsics_l: np.ndarray,
    intrinsics_r: np.ndarray,
    extrinsics_l: np.ndarray,
    extrinsics_r: np.ndarray,
    alpha: float = -1.0,
) -> RectifiedPair:
    """Rectify one stereo pair.

    Note that this does **not** rewrite its arguments
    in place. That in-place behaviour is the origin of the worst bug this path has had: passing
    the live intrinsics array let the rectified principal point replace the real one downstream,
    translating the depth map by their difference. Returning new arrays makes the mistake
    unavailable rather than merely documented.
    """
    height, width = img_l.shape[:2]

    cam_l_pose = convert_to_homogeneous_matrix(np.asarray(extrinsics_l, dtype=np.float64))
    cam_r_pose = convert_to_homogeneous_matrix(np.asarray(extrinsics_r, dtype=np.float64))
    rt = cam_r_pose @ np.linalg.inv(cam_l_pose)

    rect = compute_stereo_rectification(
        im_size=(width, height),
        r_mat=rt[:3, :3],
        t_vec=rt[:3, 3],
        k1=np.asarray(intrinsics_l, dtype=np.float64),
        k2=np.asarray(intrinsics_r, dtype=np.float64),
        alpha=alpha,
    )

    rect_left, rect_right = rectify_images(rect["rect_maps"], img_l, img_r)
    return RectifiedPair(
        left=rect_left,
        right=rect_right,
        k_rect=np.asarray(rect["K"], dtype=np.float64),
        r1=np.asarray(rect["R1"], dtype=np.float64),
        # Signed: its sign is what says whether the base camera is the left or the right camera
        # of the rectified pair, which decides whether the images must be mirrored.
        baseline_x_signed_m=float(np.asarray(rect["T"]).reshape(-1)[0]),
    )
