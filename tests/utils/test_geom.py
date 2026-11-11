"""Tests for the quaternion / frame helpers in ``eden.utils.geom``.

Ported helpers are pinned against the ``_ref_*`` oracles, vendored verbatim from the removed
``eden.utils.isaac_math`` (IsaacLab, BSD-3-Clause) as an independent implementation.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import genesis.utils.geom as gg
import numpy as np
import pytest
import torch

from eden.utils import geom

ATOL = 1e-5


# ---------------------------------------------------------------------------
# Vendored isaac_math reference implementations (the equivalence oracle).
# Copied verbatim from the removed ``eden.utils.isaac_math`` (IsaacLab,
# BSD-3-Clause). These are the ground truth the migration must reproduce.
# ---------------------------------------------------------------------------


def _ref_normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def _ref_quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def _ref_quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return _ref_quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def _ref_matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _ref_quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def _ref_quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def _ref_yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    qw, qx, qy, qz = quat_yaw[:, 0], quat_yaw[:, 1], quat_yaw[:, 2], quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = _ref_normalize(quat_yaw)
    return quat_yaw.view(shape)


def _ref_axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


def _ref_quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = _ref_normalize(axis) * theta.sin()
    w = theta.cos()
    return _ref_normalize(torch.cat([w, xyz], dim=-1))


def _ref_quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qw, qx, qy, qz], dim=-1)


def _ref_copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    mag_torch = abs(mag) * torch.ones_like(other)
    return torch.copysign(mag_torch, other)


def _ref_euler_xyz_from_quat(quat: torch.Tensor, wrap_to_2pi: bool = False):
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, _ref_copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw


def _ref_quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    quat_diff = _ref_quat_mul(q1, _ref_quat_conjugate(q2))
    return _ref_axis_angle_from_quat(quat_diff)


def _ref_quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return torch.norm(_ref_quat_box_minus(q1, q2), dim=-1)


def _ref_wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


def _ref_quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _ref_quat_apply(_ref_yaw_quat(quat), vec)


def _ref_subtract_frame_transforms(t01, q01, t02=None, q02=None):
    q10 = _ref_quat_inv(q01)
    q12 = _ref_quat_mul(q10, q02) if q02 is not None else q10
    t12 = _ref_quat_apply(q10, t02 - t01) if t02 is not None else _ref_quat_apply(q10, -t01)
    return t12, q12


def _ref_combine_frame_transforms(t01, q01, t12=None, q12=None):
    q02 = _ref_quat_mul(q01, q12) if q12 is not None else q01
    t02 = t01 + _ref_quat_apply(q01, t12) if t12 is not None else t01
    return t02, q02


def _ref_transform_points(points, pos=None, quat=None):
    points_batch = points.clone()
    is_batched = points_batch.dim() == 3
    if points_batch.dim() == 2:
        points_batch = points_batch[None]
    if quat is not None:
        rot_mat = _ref_matrix_from_quat(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]
        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
        points_batch = points_batch.transpose_(1, 2)
    if pos is not None:
        pos = pos[None, None, :] if pos.dim() == 1 else pos[:, None, :]
        points_batch = points_batch + pos
    if not is_batched:
        points_batch = points_batch.squeeze(0)
    return points_batch


def _rand_quats(*shape: int) -> torch.Tensor:
    """Unit wxyz quaternions of the given (leading...) shape with a trailing 4."""
    q = torch.randn(*shape, 4)
    return q / q.norm(dim=-1, keepdim=True)


# ``eden.utils.geom`` re-exports every genesis quaternion / pose helper that takes a batch of torch
# tensors, so a natural ``from eden.utils.geom import inv_transform_by_quat`` resolves instead of
# raising. Every genesis helper in that family is classified below: re-exported, shadowed by an Eden
# implementation, or excluded for a stated reason.
_GENESIS_REEXPORTS = (
    "R_to_T",
    "R_to_quat",
    "R_to_rot6d",
    "T_to_quat",
    "T_to_trans",
    "T_to_trans_quat",
    "axis_angle_to_quat",
    "inv_quat",
    "inv_transform_by_T",
    "inv_transform_by_quat",
    "inv_transform_by_trans_quat",
    "inv_transform_pos_quat_by_trans_quat",
    "normalize",
    "quat_to_R",
    "rot6d_to_R",
    "slerp",
    "trans_R_to_T",
    "trans_quat_to_T",
    "trans_to_T",
    "transform_by_quat",
    "transform_by_trans_R",
    "transform_by_trans_quat",
    "transform_pos_quat_by_trans_quat",
    "transform_quat_by_quat",
)
# Names both modules define, with Eden's own implementation (numerically genesis', pinned below).
_EDEN_SHADOWS_GENESIS = {"transform_by_R", "transform_by_T"}
# Genesis helpers in the family that are deliberately NOT re-exported, by reason.
_GENESIS_NOT_REEXPORTED = {
    # Euler angles: this module's ``quat_from_euler_xyz`` / ``euler_xyz_from_quat`` are the Eden
    # convention; a genesis converter beside them would be a second convention behind one import.
    "euler convention": {"R_to_xyz", "euler_to_R", "euler_to_quat", "quat_to_xyz", "transform_by_xyz", "xyz_to_quat"},
    # numba- or numpy-only: a torch tensor raises a numba TypingError or a numpy dtype error.
    "no torch path": {
        "R_to_rotvec",
        "axis_angle_to_R",
        "quat_to_rotvec",
        "rotvec_to_R",
        "rotvec_to_quat",
        "scale_to_T",
        "z_to_R",
        "z_to_quat",
    },
    # Take torch, but raise on a batch (``inv_T`` mis-broadcasts ``-R.T @ trans``).
    "no batched torch path": {"inv_T", "z_up_to_R"},
    # Camera and inertia helpers rather than frame math.
    "not frame math": {"T_to_pos_lookat_up", "pos_lookat_up_to_T", "transform_inertia_by_T"},
    "constant or sampler": {"identity_quat", "random_quaternion", "tc_identity_quat"},
}
# A genesis helper is in the family when a ``_``-separated token of its name is one of these.
_FAMILY_TOKENS = {"R", "T", "axis", "euler", "quat", "quaternion", "rot6d", "rotvec", "slerp", "trans", "xyz"}


def _public_functions(module) -> set[str]:
    return {
        name
        for name, obj in vars(module).items()
        if callable(obj) and not name.startswith("_") and getattr(obj, "__module__", None) == gg.__name__
    }


def _genesis_family() -> set[str]:
    return {n for n in _public_functions(gg) if not n.startswith("qd_") and _FAMILY_TOKENS & set(n.split("_"))}


def _pose_inputs():
    torch.manual_seed(7)
    q, t, p = _rand_quats(8), torch.randn(8, 3), torch.randn(8, 3)
    return q, t, p, gg.quat_to_R(q), gg.trans_quat_to_T(t, q)


def _apply_T(T, p):
    return (T[..., :3, :3] @ p[..., None])[..., 0] + T[..., :3, 3]


# name -> (actual, expected) on a batch of 8, each anchored on a plain matmul or a round trip.
_REEXPORT_CHECKS = {
    "R_to_T": lambda q, t, p, R, T: (geom.R_to_T(R)[..., :3, :3], R),
    "R_to_quat": lambda q, t, p, R, T: (geom.quat_to_R(geom.R_to_quat(R)), R),
    "R_to_rot6d": lambda q, t, p, R, T: (geom.rot6d_to_R(geom.R_to_rot6d(R)), R),
    "T_to_quat": lambda q, t, p, R, T: (geom.quat_to_R(geom.T_to_quat(T)), R),
    "T_to_trans": lambda q, t, p, R, T: (geom.T_to_trans(T), t),
    "T_to_trans_quat": lambda q, t, p, R, T: (
        torch.cat([geom.T_to_trans_quat(T)[0], geom.quat_to_R(geom.T_to_trans_quat(T)[1]).flatten(-2)], -1),
        torch.cat([t, R.flatten(-2)], -1),
    ),
    # A rotation about ``axis`` leaves ``axis`` fixed.
    "axis_angle_to_quat": lambda q, t, p, R, T: (geom.transform_by_quat(p, geom.axis_angle_to_quat(t[:, 0], p)), p),
    "inv_quat": lambda q, t, p, R, T: (geom.quat_to_R(geom.inv_quat(q)), R.transpose(-1, -2)),
    "inv_transform_by_T": lambda q, t, p, R, T: (geom.inv_transform_by_T(_apply_T(T, p), T), p),
    "inv_transform_by_quat": lambda q, t, p, R, T: (geom.inv_transform_by_quat((R @ p[..., None])[..., 0], q), p),
    "inv_transform_by_trans_quat": lambda q, t, p, R, T: (geom.inv_transform_by_trans_quat(_apply_T(T, p), t, q), p),
    "inv_transform_pos_quat_by_trans_quat": lambda q, t, p, R, T: (
        geom.inv_transform_pos_quat_by_trans_quat(*geom.transform_pos_quat_by_trans_quat(p, q, t, q), t, q)[0],
        p,
    ),
    "normalize": lambda q, t, p, R, T: (geom.normalize(p).norm(dim=-1), torch.ones(8)),
    "quat_to_R": lambda q, t, p, R, T: (geom.quat_to_R(q), _ref_matrix_from_quat(q)),
    "rot6d_to_R": lambda q, t, p, R, T: (geom.rot6d_to_R(geom.R_to_rot6d(R)), R),
    "slerp": lambda q, t, p, R, T: (
        geom.quat_to_R(geom.slerp(q, geom.inv_quat(q), torch.ones(8))),
        R.transpose(-1, -2),
    ),
    "trans_R_to_T": lambda q, t, p, R, T: (geom.trans_R_to_T(t, R), T),
    "trans_quat_to_T": lambda q, t, p, R, T: (_apply_T(geom.trans_quat_to_T(t, q), p), (R @ p[..., None])[..., 0] + t),
    "trans_to_T": lambda q, t, p, R, T: (geom.trans_to_T(t)[..., :3, 3], t),
    "transform_by_quat": lambda q, t, p, R, T: (geom.transform_by_quat(p, q), (R @ p[..., None])[..., 0]),
    "transform_by_trans_R": lambda q, t, p, R, T: (geom.transform_by_trans_R(p, t, R), _apply_T(T, p)),
    "transform_by_trans_quat": lambda q, t, p, R, T: (geom.transform_by_trans_quat(p, t, q), _apply_T(T, p)),
    "transform_pos_quat_by_trans_quat": lambda q, t, p, R, T: (
        geom.transform_pos_quat_by_trans_quat(p, q, t, q)[0],
        _apply_T(T, p),
    ),
    # ``transform_quat_by_quat(v, u)`` applies ``u`` on the left.
    "transform_quat_by_quat": lambda q, t, p, R, T: (
        geom.quat_to_R(geom.transform_quat_by_quat(q, q.flip(0))),
        R.flip(0) @ R,
    ),
}


@pytest.mark.parametrize("name", _GENESIS_REEXPORTS)
def test_genesis_pose_reexport_is_correct_on_a_torch_batch(name):
    # Being in the family means taking a batch of torch tensors; pin it on values, not just types.
    actual, expected = _REEXPORT_CHECKS[name](*_pose_inputs())
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=ATOL)


@pytest.mark.parametrize("name", sorted(_EDEN_SHADOWS_GENESIS))
def test_eden_shadow_matches_its_genesis_namesake_on_a_torch_batch(name):
    # ``inv_transform_by_T`` is genesis' and ``transform_by_T`` is Eden's; they must stay inverses.
    _, _, p, R, T = _pose_inputs()
    arg = {"transform_by_R": R, "transform_by_T": T}[name]
    assert torch.allclose(getattr(geom, name)(p, arg), getattr(gg, name)(p, arg), atol=ATOL)


def test_every_genesis_pose_helper_is_classified():
    # A genesis bump that adds a pose helper lands here rather than as an ImportError in someone's
    # ``from eden.utils.geom import ...``.
    family = _genesis_family()
    excluded = set().union(*_GENESIS_NOT_REEXPORTED.values())
    unclassified = family - set(_GENESIS_REEXPORTS) - _EDEN_SHADOWS_GENESIS - excluded
    assert not unclassified, (
        f"genesis.utils.geom has pose helpers eden.utils.geom neither re-exports, shadows nor excludes: "
        f"{sorted(unclassified)}. If one takes a batch of torch tensors, add it to the genesis import "
        f"block at the top of eden/utils/geom.py, to _GENESIS_REEXPORTS and to _REEXPORT_CHECKS here; "
        f"otherwise add it to _GENESIS_NOT_REEXPORTED under the reason that holds."
    )
    # Re-exports are pinned by identity above (``normalize`` is one outside the family).
    stale = (_EDEN_SHADOWS_GENESIS | excluded) - family
    assert not stale, f"_GENESIS_NOT_REEXPORTED / _EDEN_SHADOWS_GENESIS name non-family helpers: {sorted(stale)}"
    assert set(_REEXPORT_CHECKS) == set(_GENESIS_REEXPORTS), "every re-export needs a torch-batch check"


def test_no_eden_geom_name_silently_differs_from_its_genesis_namesake():
    # Every name the two modules share is either the genesis object itself or a documented Eden
    # implementation -- never an accidental second meaning behind a genesis name.
    shared = {n for n in vars(geom) if not n.startswith("_") and n in _public_functions(gg)}
    differing = {n for n in shared if getattr(geom, n) is not getattr(gg, n)}
    unexpected = differing - _EDEN_SHADOWS_GENESIS
    assert not unexpected, (
        f"eden.utils.geom defines {sorted(unexpected)}, which genesis.utils.geom also defines as a "
        f"different function. Rename the Eden one if the semantics differ (the ``quat_mul`` / "
        f"``transform_quat_by_quat`` operand order is the trap this guards); only if they are "
        f"numerically identical add it to _EDEN_SHADOWS_GENESIS and to "
        f"test_eden_shadow_matches_its_genesis_namesake_on_a_torch_batch."
    )
    assert differing == _EDEN_SHADOWS_GENESIS, f"no longer shadowed: {sorted(_EDEN_SHADOWS_GENESIS - differing)}"
    assert shared - differing == set(_GENESIS_REEXPORTS), (
        f"the genesis import block in eden/utils/geom.py and _GENESIS_REEXPORTS disagree: "
        f"{sorted((shared - differing) ^ set(_GENESIS_REEXPORTS))}"
    )


def test_the_helpers_can_be_called_from_several_threads_at_once():
    """The motion datasets estimate velocities in a thread pool.

    Under https://github.com/Genesis-Embodied-AI/Genesis/pull/3479 the genesis torch helpers ran through
    ``torch.compile``, and a thread calling one while another traced it raised
    ``TypeError: size(): argument 'dim' ... must be int, not tuple``.
    https://github.com/Genesis-Embodied-AI/Genesis/pull/3527 runs them eagerly again.
    """
    torch._dynamo.reset()  # a helper compiled by an earlier test would otherwise not trace here

    def call(i):
        q = geom.normalize(torch.randn(50 + i, 4))
        return geom.quat_mul(q, q), geom.transform_by_quat(q[:, 1:], q), geom.quat_to_R(q)

    with ThreadPoolExecutor(8) as pool:
        try:
            list(pool.map(call, range(16)))
        except TypeError as e:
            pytest.fail(
                f"A genesis geometry helper raised when called from 8 threads: TypeError: {e}. If the "
                "genesis pin compiles these helpers again "
                "(https://github.com/Genesis-Embodied-AI/Genesis/pull/3479), it has to make them thread-safe first. "
                "Serializing the calls on Eden's side would only hide the problem from other callers."
            )


# ---------------------------------------------------------------------------
# eden.utils.geom functions (ported / composed) vs the isaac_math oracle.
# ---------------------------------------------------------------------------


def test_quat_mul_matches_oracle_and_genesis_swap():
    torch.manual_seed(6)
    q1, q2 = _rand_quats(16), _rand_quats(16)
    assert torch.allclose(geom.quat_mul(q1, q2), _ref_quat_mul(q1, q2), atol=ATOL)
    # geom.quat_mul wraps genesis transform_quat_by_quat with operands swapped.
    assert torch.allclose(geom.quat_mul(q1, q2), gg.transform_quat_by_quat(q2, q1), atol=ATOL)


def _ref_quat_nlerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Per-row shortest-path NLERP oracle (the SONIC pose_streamer helper it replaced)."""
    out = torch.empty_like(q0)
    for i in range(q0.shape[0]):
        a, b = q0[i], q1[i]
        if float(torch.dot(a, b)) < 0.0:
            b = -b
        q = (1.0 - t) * a + t * b
        out[i] = q / q.norm()
    return out


def test_quat_nlerp_matches_reference_and_is_unit():
    torch.manual_seed(21)
    q0, q1 = _rand_quats(32), _rand_quats(32)
    for t in (0.0, 0.25, 0.5, 1.0):
        got = geom.quat_nlerp(q0, q1, t)
        assert torch.allclose(got, _ref_quat_nlerp(q0, q1, t), atol=ATOL)
        assert torch.allclose(got.norm(dim=-1), torch.ones(32), atol=ATOL)


def test_quat_nlerp_endpoints_return_inputs():
    torch.manual_seed(22)
    q0, q1 = _rand_quats(8), _rand_quats(8)
    # t=0 returns q0 exactly; t=1 returns q1 up to the hemisphere flip (q and -q
    # are the same rotation), so compare rotations there.
    assert torch.allclose(geom.quat_nlerp(q0, q1, 0.0), q0, atol=ATOL)
    assert torch.allclose(gg.quat_to_R(geom.quat_nlerp(q0, q1, 1.0)), gg.quat_to_R(q1), atol=ATOL)


def test_quat_nlerp_shortest_path_is_double_cover_invariant():
    """Flipping q1's sign (same rotation) must not change the interpolated rotation."""
    torch.manual_seed(23)
    q0, q1 = _rand_quats(16), _rand_quats(16)
    a = geom.quat_nlerp(q0, q1, 0.3)
    b = geom.quat_nlerp(q0, -q1, 0.3)
    assert torch.allclose(gg.quat_to_R(a), gg.quat_to_R(b), atol=ATOL)


@pytest.mark.parametrize("as_torch", [True, False])
def test_quat_nlerp_takes_one_weight_per_quaternion(as_torch):
    """A ``(B,)`` weight raised against ``(B, 4)`` quats, and blended per component when B == 4."""
    torch.manual_seed(24)
    q0, q1 = _rand_quats(4), _rand_quats(4)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    rows = torch.stack([geom.quat_nlerp(q0[i : i + 1], q1[i : i + 1], float(t[i]))[0] for i in range(4)])
    got = (
        geom.quat_nlerp(q0, q1, t) if as_torch else torch.from_numpy(geom.quat_nlerp(q0.numpy(), q1.numpy(), t.numpy()))
    )
    assert torch.allclose(got, rows, atol=ATOL)


# Run in a fresh interpreter: this session's conftest has already called ``gs.init()``.
_BEFORE_GS_INIT = """
import json

import eden.utils.geom as geom  # eden first, as everywhere
import genesis as gs
import numpy as np
import torch

h = np.sqrt(0.5)
q0 = np.array([1.0, 0.0, 0.0, 0.0])
q1 = -np.array([h, 0.0, 0.0, h])  # +90 deg about z, in the far hemisphere so the flip runs
print(json.dumps({
    "eps_is_none": gs.EPS is None,
    "nlerp_numpy": geom.quat_nlerp(q0, q1, 0.5).tolist(),
    "nlerp_torch": geom.quat_nlerp(torch.from_numpy(q0), torch.from_numpy(q1), 0.5).tolist(),
    "yaw_numpy": float(geom.yaw_from_quat(q1)),
}))
"""


def test_quat_nlerp_and_yaw_from_quat_work_before_genesis_init():
    """``gs.EPS`` is None until ``gs.init()``; the SONIC operator and offline tools call these first.

    Pins why ``yaw_from_quat`` is hand-written rather than ``quat_to_xyz(q, rpy=True)[2]``: genesis'
    numpy ``quat_to_xyz`` is a numba kernel reading ``gs.EPS``, so it raises until ``gs.init()`` runs.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _BEFORE_GS_INIT], capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["eps_is_none"], "genesis was initialised in the subprocess, so this test pins nothing"
    half_45 = [np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]  # NLERP's midpoint is SLERP's: 45 deg about z
    np.testing.assert_allclose(got["nlerp_numpy"], half_45, atol=1e-12)
    np.testing.assert_allclose(got["nlerp_torch"], half_45, atol=1e-12)
    assert got["yaw_numpy"] == pytest.approx(np.pi / 2)


def test_axis_angle_helpers_numpy_match_torch_and_scipy_down_to_zero():
    # The numpy paths are OTRetarget's exp/log maps: small rotation vectors (solver steps) must survive exactly, and
    # genesis rotvec_to_R gives NaN for a zero vector before gs.init and ~1e-10 error at 1e-7 rad.
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(size=(6, 3)), 1e-7 * rng.normal(size=(3, 3)), [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]])
    q = geom.quat_from_axis_angle(v)
    assert isinstance(q, np.ndarray)
    np.testing.assert_allclose(q, geom.quat_from_axis_angle(torch.as_tensor(v)).numpy(), atol=1e-15)
    np.testing.assert_allclose(geom.quat_to_R(q), Rotation.from_rotvec(v).as_matrix(), atol=1e-15)
    back = geom.axis_angle_from_quat(geom.R_to_quat(Rotation.from_rotvec(v).as_matrix()))
    np.testing.assert_allclose(back, Rotation.from_rotvec(v).as_rotvec(), atol=1e-15)
    np.testing.assert_allclose(back, geom.axis_angle_from_quat(torch.as_tensor(-q)).numpy(), atol=1e-14)
    np.testing.assert_array_equal(geom.quat_unique(-q), geom.quat_unique(torch.as_tensor(-q)).numpy())


def test_segment_closest_points_match_dense_sampling():
    rng = np.random.default_rng(0)
    p0, p1, q0, q1 = (rng.normal(size=(50, 3)) for _ in range(4))
    pa, pb = geom.segment_closest_points(p0, p1, q0, q1)
    s = np.linspace(0, 1, 401)
    for k in range(50):
        a = p0[k] + s[:, None] * (p1[k] - p0[k])
        b = q0[k] + s[:, None] * (q1[k] - q0[k])
        brute = np.linalg.norm(a[:, None] - b[None], axis=-1).min()
        assert np.linalg.norm(pa[k] - pb[k]) <= brute + 1e-9
        assert np.linalg.norm(pa[k] - pb[k]) >= brute - 5e-3


def test_segment_closest_points_handle_a_point_segment_in_either_slot():
    # OTRetarget's fit_capsule collapses a link shorter than its diameter to a point, and a pair is
    # ordered by link index, so the point capsule can land in either slot.
    seg0, seg1 = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    point = np.array([[0.5, 0.1, 0.0]])
    on_seg, on_point = geom.segment_closest_points(seg0, seg1, point, point)
    np.testing.assert_allclose(on_seg, [[0.5, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(on_point, point, atol=1e-12)
    on_point, on_seg = geom.segment_closest_points(point, point, seg0, seg1)
    np.testing.assert_allclose(on_seg, [[0.5, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(on_point, point, atol=1e-12)
    pa, pb = geom.segment_closest_points(point, point, point + 0.3, point + 0.3)
    np.testing.assert_allclose(np.linalg.norm(pa - pb, axis=1), [0.3 * np.sqrt(3)])


def test_planar_rigid_recovers_a_turn_and_reads_coincident_points_as_translation_only():
    a = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.05]])
    turn = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    rot, t = geom.planar_rigid_transform(a, a @ turn.T + [0.02, -0.01])
    np.testing.assert_allclose(rot, turn, atol=1e-12)
    np.testing.assert_allclose(t, [0.02, -0.01], atol=1e-12)
    # Three points at one place (OTRetarget: robot probes sharing one human source): atan2(0, 0) used to give a 180 deg turn of a planted foot.
    same = np.tile([[0.4, 0.2]], (3, 1))
    rot, t = geom.planar_rigid_transform(same, same + [0.001, 0.0])
    np.testing.assert_array_equal(rot, np.eye(2))
    np.testing.assert_allclose(t, [0.001, 0.0], atol=1e-15)


@pytest.mark.parametrize("as_torch", [False, True])
def test_skew_is_the_cross_product_matrix_on_any_batch(as_torch):
    a, b = np.random.default_rng(0).normal(size=(2, 4, 5, 3))
    va, vb = (torch.as_tensor(a), torch.as_tensor(b)) if as_torch else (a, b)
    S = geom.skew(va)
    assert type(S) is type(va) and S.shape == (4, 5, 3, 3)
    np.testing.assert_allclose(np.asarray((S @ vb[..., None])[..., 0]), np.cross(a, b), atol=1e-12)
    np.testing.assert_allclose(np.asarray(S + S.swapaxes(-1, -2) if not as_torch else S + S.transpose(-1, -2)), 0.0)


def test_quat_nlerp_numpy_matches_torch():
    """The numpy and torch paths agree (pose_streamer feeds it numpy)."""
    torch.manual_seed(24)
    q0, q1 = _rand_quats(16), _rand_quats(16)
    got_t = geom.quat_nlerp(q0, q1, 0.4)
    got_np = geom.quat_nlerp(q0.numpy(), q1.numpy(), 0.4)
    assert torch.allclose(got_t, torch.from_numpy(got_np).to(got_t.dtype), atol=ATOL)


def test_quat_from_euler_xyz_matches_oracle():
    torch.manual_seed(7)
    r, p, y = torch.randn(16), torch.randn(16), torch.randn(16)
    assert torch.allclose(geom.quat_from_euler_xyz(r, p, y), _ref_quat_from_euler_xyz(r, p, y), atol=ATOL)


def test_euler_xyz_from_quat_matches_oracle():
    torch.manual_seed(8)
    q = _rand_quats(16)
    for wrap in (False, True):
        g = geom.euler_xyz_from_quat(q, wrap_to_2pi=wrap)
        ref = _ref_euler_xyz_from_quat(q, wrap_to_2pi=wrap)
        for a, b in zip(g, ref):
            assert torch.allclose(a, b, atol=ATOL)


def test_euler_quat_round_trip():
    """Quat -> euler -> quat reproduces the rotation (sign-agnostic)."""
    torch.manual_seed(80)
    q = _rand_quats(32)
    r, p, y = geom.euler_xyz_from_quat(q)
    q2 = geom.quat_from_euler_xyz(r, p, y)
    # quaternions are double-cover (q and -q are the same rotation), so compare
    # the rotation matrices rather than the quaternions directly.
    assert torch.allclose(gg.quat_to_R(q), gg.quat_to_R(q2), atol=ATOL)


def test_intrinsic_xyz_from_quat_matches_scipy_and_rebuilds_hinge_chain():
    """(a, b, c) are the angles of hinges about x, then y, then z: Rx(a) @ Ry(b) @ Rz(c) == R(q)."""
    from scipy.spatial.transform import Rotation

    torch.manual_seed(81)
    q = _rand_quats(64)
    abc = geom.intrinsic_xyz_from_quat(q)
    ref = Rotation.from_quat(q[:, [1, 2, 3, 0]].numpy()).as_euler("XYZ")
    assert torch.allclose(abc, torch.as_tensor(ref, dtype=abc.dtype), atol=1e-5)
    x, y, z = torch.eye(3).unbind(0)
    chain = [gg.axis_angle_to_quat(abc[:, i], axis.expand(64, 3)) for i, axis in enumerate((x, y, z))]
    rebuilt = geom.quat_mul(geom.quat_mul(chain[0], chain[1]), chain[2])
    assert torch.allclose(gg.quat_to_R(rebuilt), gg.quat_to_R(q), atol=ATOL)


def _hinge_chain_R(abc: torch.Tensor) -> torch.Tensor:
    """``Rx(a) @ Ry(b) @ Rz(c)`` of ``(..., 3)`` intrinsic X-Y-Z angles."""
    x, y, z = torch.eye(3).unbind(0)
    chain = [gg.axis_angle_to_quat(abc[..., i], axis.expand(*abc.shape[:-1], 3)) for i, axis in enumerate((x, y, z))]
    return gg.quat_to_R(geom.quat_mul(geom.quat_mul(chain[0], chain[1]), chain[2]))


def test_intrinsic_xyz_nearest_bends_past_90_deg_about_y_without_flipping_x_and_z():
    """A y-rotation of 2 rad decomposes as (pi, pi - 2, pi); from a joint at (0, 1.5, 0) it is (0, 2, 0)."""
    raw = geom.intrinsic_xyz_from_quat(gg.axis_angle_to_quat(torch.tensor([2.0]), torch.tensor([[0.0, 1.0, 0.0]])))
    assert torch.allclose(raw.abs(), torch.tensor([[math.pi, math.pi - 2.0, math.pi]]), atol=1e-5)
    near = geom.intrinsic_xyz_nearest(raw, torch.tensor([[0.0, 1.5, 0.0]]))
    assert torch.allclose(near, torch.tensor([[0.0, 2.0, 0.0]]), atol=1e-5)


def test_intrinsic_xyz_nearest_names_the_same_rotation_no_farther_than_any_whole_turn_shift():
    torch.manual_seed(83)
    q = _rand_quats(256)
    raw = geom.intrinsic_xyz_from_quat(q)
    current = (torch.rand(256, 3) * 2 - 1) * 2 * math.pi
    near = geom.intrinsic_xyz_nearest(raw, current)
    assert torch.allclose(_hinge_chain_R(near), gg.quat_to_R(q), atol=1e-5)
    # Never farther from `current` than either decomposition moved by its own nearest whole turns.
    dual = torch.stack([raw[:, 0] + math.pi, math.pi - raw[:, 1], raw[:, 2] + math.pi], dim=-1)
    for other in (raw, dual):
        shifted = other + 2 * math.pi * torch.round((current - other) / (2 * math.pi))
        assert (((near - current) ** 2).sum(-1) <= ((shifted - current) ** 2).sum(-1) + 1e-6).all()


def _hinge_chain_q(abc: torch.Tensor) -> torch.Tensor:
    """Wxyz quaternion of ``Rx(a) @ Ry(b) @ Rz(c)``, in ``abc``'s dtype."""
    x, y, z = torch.eye(3, dtype=abc.dtype).unbind(0)
    chain = [geom.quat_from_axis_angle(abc[..., i : i + 1] * axis) for i, axis in enumerate((x, y, z))]
    return geom.quat_mul(geom.quat_mul(chain[0], chain[1]), chain[2])


def _angles_in_lock_band(n: int, lock_band: float) -> torch.Tensor:
    """``(n, 3)`` float64 angles with ``|cos b| < lock_band``, on either side of the lock, any ``a`` and ``c``."""
    b = torch.where(torch.rand(n) < 0.5, -1.0, 1.0) * torch.acos(torch.rand(n) * lock_band)
    return torch.stack([(torch.rand(n) * 2 - 1) * math.pi, b, (torch.rand(n) * 2 - 1) * math.pi], dim=-1).double()


def test_intrinsic_xyz_nearest_with_a_lock_band_is_exact_outside_it_and_moves_little_inside_it():
    """Outside ``|cos b| < lock_band`` nothing changes; inside, the rotation moves by under ``lock_band / 2``."""
    torch.manual_seed(84)
    q = _rand_quats(4096).double()
    current = ((torch.rand(4096, 3) * 2 - 1) * 2 * math.pi).double()
    exact = geom.intrinsic_xyz_nearest(geom.intrinsic_xyz_from_quat(q), current)
    damped = geom.intrinsic_xyz_nearest(geom.intrinsic_xyz_from_quat(q), current, lock_band=0.3)
    outside = torch.cos(exact[:, 1]).abs() >= 0.3
    assert outside.sum() > 3000
    torch.testing.assert_close(damped[outside], exact[outside], atol=1e-12, rtol=0)

    target = _angles_in_lock_band(20000, 0.3)
    current = ((torch.rand(20000, 3) * 2 - 1) * 2 * math.pi).double()
    damped = geom.intrinsic_xyz_nearest(geom.intrinsic_xyz_from_quat(_hinge_chain_q(target)), current, lock_band=0.3)
    assert geom.quat_error_magnitude(_hinge_chain_q(damped), _hinge_chain_q(target)).max() < 0.15


def test_intrinsic_xyz_nearest_at_the_lock_keeps_the_rotation_and_takes_the_free_combination_from_near():
    """At ``b = pi/2`` only ``a + c`` is defined: the damped angles keep it and take ``a - c`` from ``near``."""
    torch.manual_seed(85)
    target = torch.stack([torch.rand(64) * 2 - 1, torch.full((64,), math.pi / 2), torch.rand(64) * 2 - 1], -1).double()
    near = torch.stack([torch.rand(64) - 0.5, torch.full((64,), 1.5), torch.rand(64) - 0.5], -1).double()
    damped = geom.intrinsic_xyz_nearest(geom.intrinsic_xyz_from_quat(_hinge_chain_q(target)), near, lock_band=0.3)
    assert geom.quat_error_magnitude(_hinge_chain_q(damped), _hinge_chain_q(target)).max() < 1e-6
    torch.testing.assert_close(damped[:, 0] - damped[:, 2], near[:, 0] - near[:, 2], atol=1e-6, rtol=0)


def test_intrinsic_xyz_nearest_with_a_lock_band_bends_a_knee_through_90_deg_without_swinging_x_and_z():
    """A knee bending 1.3 -> 1.9 rad about y over 19 frames, with a 0.02 rad twist about z."""
    y = torch.linspace(1.3, 1.9, 19, dtype=torch.float64)
    q = geom.quat_from_axis_angle(torch.stack([torch.zeros_like(y), y, torch.full_like(y, 0.02)], dim=-1))
    raw = geom.intrinsic_xyz_from_quat(q)
    exact = geom.intrinsic_xyz_nearest(raw, torch.zeros_like(raw))
    damped = geom.intrinsic_xyz_nearest(raw, torch.zeros_like(raw), lock_band=0.3)
    assert exact[:, [0, 2]].abs().max() > 1.0 and (exact[1:] - exact[:-1])[:, [0, 2]].abs().max() > 1.0
    assert damped[:, [0, 2]].abs().max() < 0.1 and (damped[1:] - damped[:-1])[:, [0, 2]].abs().max() < 0.05
    assert geom.quat_error_magnitude(_hinge_chain_q(damped), q).max() < 0.02


def _body_ang_vel(abc: torch.Tensor, rates: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
    """Body-frame angular velocity of the hinge chain moving at ``rates``, by central differences."""
    q0, q1 = _hinge_chain_q(abc - h * rates), _hinge_chain_q(abc + h * rates)
    return geom.axis_angle_from_quat(geom.quat_mul(gg.inv_quat(q0), q1)) / (2 * h)


def test_intrinsic_xyz_rates_turn_the_hinge_chain_at_the_given_angular_velocity():
    torch.manual_seed(86)
    a, b, c = (torch.rand(3, 512, dtype=torch.float64) * 2 - 1) * torch.tensor([3.0, 1.3, 3.0]).double()[:, None]
    abc = torch.stack([a, b, c], dim=-1)
    w = torch.randn(512, 3, dtype=torch.float64)
    rates = geom.intrinsic_xyz_rates(abc, w)
    torch.testing.assert_close(_body_ang_vel(abc, rates), w, atol=1e-6, rtol=0)
    # Outside the band the damped rates are the exact ones.
    outside = torch.cos(b).abs() >= 0.3
    damped = geom.intrinsic_xyz_rates(abc, w, lock_band=0.3)
    torch.testing.assert_close(damped[outside], rates[outside], atol=1e-12, rtol=0)


def test_intrinsic_xyz_rates_near_the_lock_stay_bounded_and_give_up_only_the_unreachable_part():
    """Exact rates blow up as ``1 / cos b``; damped ones stay under ``1.09 / lock_band`` per unit ``|w|``."""
    torch.manual_seed(87)
    abc = _angles_in_lock_band(4096, 0.3)
    w = torch.randn(4096, 3, dtype=torch.float64)
    w = w / w.norm(dim=-1, keepdim=True)
    assert geom.intrinsic_xyz_rates(abc, w)[:, 0].abs().max() > 100
    damped = geom.intrinsic_xyz_rates(abc, w, lock_band=0.3)
    assert damped[:, 0].abs().max() <= 1.09 / 0.3
    # What the chain turns at differs from w only along (cos c, -sin c, 0): the direction the hinges barely reach.
    miss = w - _body_ang_vel(abc, damped)
    c = abc[:, 2]
    unreachable = torch.stack([torch.cos(c), -torch.sin(c), torch.zeros_like(c)], dim=-1)
    off_direction = miss - (miss * unreachable).sum(-1, keepdim=True) * unreachable
    assert off_direction.abs().max() < 1e-5

    at_lock = torch.tensor([[0.4, math.pi / 2, -0.2]], dtype=torch.float64)
    assert torch.isfinite(geom.intrinsic_xyz_rates(at_lock, w[:1], lock_band=0.3)).all()


@pytest.mark.parametrize("lock_band", [-0.1, 1.0])
def test_a_lock_band_outside_0_to_1_is_refused(lock_band):
    with pytest.raises(ValueError, match=rf"lock_band={lock_band} is a bound on \|cos b\|"):
        geom.intrinsic_xyz_nearest(torch.zeros(3), torch.zeros(3), lock_band=lock_band)
    with pytest.raises(ValueError, match=rf"lock_band={lock_band} is a bound on \|cos b\|"):
        geom.intrinsic_xyz_rates(torch.zeros(3), torch.zeros(3), lock_band=lock_band)


def test_quat_slerp_follows_the_short_arc_at_constant_speed():
    """Endpoints, the halfway rotation, and the same answer for either sign of ``q1``."""
    torch.manual_seed(82)
    q0, q1 = _rand_quats(32), _rand_quats(32)
    zeros, ones, half = torch.zeros(32, 1), torch.ones(32, 1), torch.full((32, 1), 0.5)
    torch.testing.assert_close(geom.quat_slerp(q0, q1, zeros), q0, atol=ATOL, rtol=0)
    end = geom.quat_slerp(q0, q1, ones)
    torch.testing.assert_close(gg.quat_to_R(end), gg.quat_to_R(q1), atol=ATOL, rtol=0)
    mid = geom.quat_slerp(q0, q1, half)
    # Halfway along the short arc: the same relative rotation from q0 to mid as from mid to q1.
    first = geom.quat_mul(mid, gg.inv_quat(q0))
    second = geom.quat_mul(end, gg.inv_quat(mid))
    torch.testing.assert_close(gg.quat_to_R(first), gg.quat_to_R(second), atol=1e-5, rtol=0)
    torch.testing.assert_close(geom.quat_slerp(q0, -q1, half), mid, atol=ATOL, rtol=0)


def test_quat_slerp_averages_nearly_equal_rotations_whatever_t():
    """PHC's fallback, kept verbatim: within ~0.1 deg the result is the mean of the two, not a blend at ``t``."""
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    angle = 1.0e-3  # rad; sin(angle / 2) = 5e-4 < 1e-3
    q1 = torch.tensor([[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]])
    torch.testing.assert_close(geom.quat_slerp(q0, q1, torch.tensor([[0.1]])), 0.5 * (q0 + q1))


def test_axis_angle_from_quat_matches_oracle():
    torch.manual_seed(9)
    q = _rand_quats(16)
    assert torch.allclose(geom.axis_angle_from_quat(q), _ref_axis_angle_from_quat(q), atol=ATOL)


def test_axis_angle_from_quat_near_identity_uses_taylor_branch():
    """Tiny rotations hit the Taylor branch (genesis quat_to_rotvec lacks it)."""
    q = torch.tensor([[1.0, 1e-9, -2e-9, 3e-9]])
    q = q / q.norm(dim=-1, keepdim=True)
    out = geom.axis_angle_from_quat(q)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, _ref_axis_angle_from_quat(q), atol=ATOL)


def test_quat_from_axis_angle_matches_genesis_angle_axis_form():
    """The packed-vector form must agree with genesis's (angle, unit-axis) converter."""
    torch.manual_seed(11)
    axis = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    angle = torch.rand(16) * 3.0 + 0.1  # away from 0 and pi, where the sign is ambiguous
    q = geom.quat_from_axis_angle(axis * angle.unsqueeze(-1))
    assert torch.allclose(gg.quat_to_R(q), gg.quat_to_R(_ref_quat_from_angle_axis(angle, axis)), atol=ATOL)


def test_quat_from_axis_angle_inverts_axis_angle_from_quat():
    torch.manual_seed(12)
    q = _rand_quats(16)
    round_trip = geom.quat_from_axis_angle(geom.axis_angle_from_quat(q))
    assert torch.allclose(gg.quat_to_R(round_trip), gg.quat_to_R(q), atol=ATOL)


def test_quat_from_axis_angle_zero_vector_is_identity():
    """A zero axis-angle has no axis; the Taylor branch must return identity, not NaN."""
    out = geom.quat_from_axis_angle(torch.zeros(3, 3))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(3, 4), atol=ATOL)


def test_quat_from_axis_angle_beyond_pi_preserves_the_rotation_not_the_vector():
    """Past pi the round trip is not the identity on vectors, and callers must not assume it is.

    ``axis_angle_from_quat`` canonicalises to ``w >= 0``, so a 1.5*pi turn comes back as the
    equal 0.5*pi turn about the flipped axis: same rotation, different vector.
    """
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    v = axis * (1.5 * torch.pi)
    back = geom.axis_angle_from_quat(geom.quat_from_axis_angle(v))

    same_rotation = gg.quat_to_R(geom.quat_from_axis_angle(back)), gg.quat_to_R(geom.quat_from_axis_angle(v))
    assert torch.allclose(*same_rotation, atol=ATOL)
    assert not torch.allclose(back, v, atol=1e-3)
    assert torch.allclose(back, -axis * (0.5 * torch.pi), atol=ATOL)


def test_axis_angle_from_quat_at_identity_has_finite_gradients():
    """``torch.where`` evaluates both branches; an unclamped 0/0 in the discarded one kills grads."""
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    geom.axis_angle_from_quat(q).sum().backward()
    assert torch.isfinite(q.grad).all()


def test_yaw_quat_matches_oracle_including_roll_pitch():
    torch.manual_seed(10)
    q = _rand_quats(16)
    assert torch.allclose(geom.yaw_quat(q), _ref_yaw_quat(q), atol=ATOL)


def test_quat_apply_yaw_matches_oracle():
    torch.manual_seed(12)
    q, v = _rand_quats(16), torch.randn(16, 3)
    assert torch.allclose(geom.quat_apply_yaw(q, v), _ref_quat_apply_yaw(q, v), atol=ATOL)


def test_yaw_from_quat_matches_yaw_quat_and_euler():
    """The scalar heading agrees with both quat-valued and Euler spellings."""
    torch.manual_seed(14)
    q = _rand_quats(256)
    yaw = geom.yaw_from_quat(q)
    assert torch.allclose(yaw, _ref_euler_xyz_from_quat(q)[2], atol=ATOL)
    assert torch.allclose(yaw, geom.euler_xyz_from_quat(q)[2], atol=ATOL)
    # Rebuilding a pure-z quat from it reproduces yaw_quat (compare as rotations).
    rebuilt = torch.zeros_like(q)
    rebuilt[..., 0], rebuilt[..., 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    assert (1.0 - (rebuilt * geom.yaw_quat(q)).sum(-1).abs()).max() < ATOL


def test_yaw_from_quat_numpy_matches_torch():
    torch.manual_seed(15)
    q = _rand_quats(32)
    assert np.allclose(geom.yaw_from_quat(q.numpy()), geom.yaw_from_quat(q).numpy(), atol=ATOL)


def test_heading_from_quat_defaults_to_x_forward_the_angle_yaw_from_quat_computes():
    torch.manual_seed(16)
    q = _rand_quats(256)
    heading = geom.heading_from_quat(q)
    assert torch.equal(heading, geom.heading_from_quat(q, torch.tensor([1.0, 0.0, 0.0])))
    # Compared modulo 2*pi: round-off can land an angle at -pi on one side and +pi on the other.
    assert geom.wrap_to_pi(heading - geom.yaw_from_quat(q)).abs().max() < ATOL


def test_heading_from_quat_is_exact_for_a_non_unit_quaternion_where_yaw_from_quat_is_not():
    # An eighth of a turn about z, both ways, scaled: the rotation is unchanged, so the headings are still +-pi/4.
    c, s = np.cos(np.pi / 8), np.sin(np.pi / 8)
    q = 1.1 * torch.tensor([[c, 0.0, 0.0, s], [c, 0.0, 0.0, -s]], dtype=torch.float32)
    expected = torch.tensor([np.pi / 4, -np.pi / 4], dtype=torch.float32)
    assert torch.allclose(geom.heading_from_quat(q), expected, atol=ATOL)
    assert not torch.allclose(geom.yaw_from_quat(q), expected, atol=1e-3)


def test_heading_from_quat_numpy_matches_torch():
    torch.manual_seed(17)
    q = _rand_quats(32)
    heading = geom.heading_from_quat(q)
    assert np.allclose(geom.heading_from_quat(q.numpy()), heading.numpy(), atol=ATOL)
    # One (4,) quaternion gives a scalar heading, on either side.
    assert np.ndim(geom.heading_from_quat(q[0].numpy())) == 0
    assert np.isclose(geom.heading_from_quat(q[0].numpy()), heading[0].item(), atol=ATOL)
    assert geom.heading_from_quat(q[0]).shape == ()
    assert torch.isclose(geom.heading_from_quat(q[0]), heading[0], atol=ATOL)


def test_heading_from_quat_follows_a_non_x_forward_axis_over_any_batch_shape():
    # A body whose front is its +y axis, turned theta about world z, heads to theta + pi/2.
    theta = torch.linspace(-3.0, 3.0, 12).reshape(3, 4)
    q = torch.stack([torch.cos(theta / 2), torch.zeros_like(theta), torch.zeros_like(theta), torch.sin(theta / 2)], -1)
    heading = geom.heading_from_quat(q, torch.tensor([[0.0, 1.0, 0.0]]))
    assert heading.shape == (3, 4)
    assert geom.wrap_to_pi(heading - (theta + torch.pi / 2)).abs().max() < ATOL


def test_quat_to_rot6d_cols_matches_isaaclab_layout():
    """First two columns of R, flattened row-major: [R00, R01, R10, R11, R20, R21]."""
    torch.manual_seed(16)
    q = _rand_quats(64)
    R = gg.quat_to_R(q)
    expected = torch.stack([R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 0], R[..., 2, 1]], -1)
    assert torch.allclose(geom.quat_to_rot6d_cols(q), expected, atol=ATOL)
    assert torch.allclose(geom.R_to_rot6d_cols(R), expected, atol=ATOL)
    # Identity -> [1,0, 0,1, 0,0], NOT the column-stacked [1,0,0, 0,1,0].
    eye = geom.quat_to_rot6d_cols(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(eye, torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]), atol=ATOL)


def _ref_humenv_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """HumEnv ``calc_heading_quat_inv``: project the local x-axis onto the ground."""
    ref_dir = torch.zeros_like(q[..., :3])
    ref_dir[..., 0] = 1.0
    rot_dir = gg.transform_by_quat(ref_dir, q)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    axis = torch.zeros_like(ref_dir)
    axis[..., 2] = 1.0
    return gg.axis_angle_to_quat(-heading, axis)


def test_inv_yaw_quat_matches_humenv_heading_oracle():
    """Pins the substitution in ``metamotivo`` ``_calc_heading_quat_inv``.

    Every MetaMotivo proprio channel (local_body_rot/vel/ang) rides on this
    quaternion, so a change to ``yaw_quat`` must not silently move it.
    """
    torch.manual_seed(17)
    q = _rand_quats(256)
    ours = gg.inv_quat(geom.yaw_quat(q))
    assert (1.0 - (ours * _ref_humenv_heading_quat_inv(q)).sum(-1).abs()).max() < ATOL


# ---------------------------------------------------------------------------
# 6D rotation representations
# ---------------------------------------------------------------------------


def _ref_quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """ProtoMotions ``utils.rotations.quat_to_tan_norm``: ``[R·x̂, R·ẑ]``."""
    ref_tan = torch.zeros_like(q[..., :3])
    ref_tan[..., 0] = 1.0
    tan = gg.transform_by_quat(ref_tan, q)
    ref_norm = torch.zeros_like(q[..., :3])
    ref_norm[..., 2] = 1.0
    norm = gg.transform_by_quat(ref_norm, q)
    return torch.cat([tan, norm], dim=-1)


def test_quat_to_tan_norm_matches_protomotions_oracle():
    torch.manual_seed(20)
    q = _rand_quats(64)
    assert torch.allclose(geom.quat_to_tan_norm(q), _ref_quat_to_tan_norm(q), atol=ATOL)


def test_R_to_tan_norm_is_the_matrix_entry_point():
    torch.manual_seed(21)
    q = _rand_quats(4, 5)
    assert torch.allclose(geom.R_to_tan_norm(gg.quat_to_R(q)), geom.quat_to_tan_norm(q), atol=ATOL)


def test_quat_to_tan_norm_at_identity_is_x_then_z():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(geom.quat_to_tan_norm(q), expected, atol=ATOL)


def test_the_three_6d_layouts_are_distinct_scalars_not_reorderings():
    """Pins the footgun documented on :func:`quat_to_rot6d`.

    ``quat_to_rot6d`` (genesis: first two *rows* of R), the IsaacLab/SONIC form
    (first two *columns*, row-major), and ``quat_to_tan_norm`` (``[R·x̂, R·ẑ]``)
    are three different 6-vectors — feeding one to a head trained on another is
    silently wrong, so they must never be treated as interchangeable.
    """
    torch.manual_seed(22)
    q = _rand_quats(32)
    rows = geom.quat_to_rot6d(q)
    cols = geom.quat_to_rot6d_cols(q)
    tan_norm = geom.quat_to_tan_norm(q)

    for a, b in ((rows, cols), (rows, tan_norm), (cols, tan_norm)):
        assert (a - b).abs().max() > 0.1
        # Not a permutation of each other either: the multisets differ.
        assert not torch.allclose(a.sort(dim=-1).values, b.sort(dim=-1).values, atol=ATOL)


def test_transform_by_quat_yaw_approx_is_an_approximation_not_quat_apply_yaw():
    """Pins the semantics the ``_approx`` rename encodes: zeroing qx/qy
    (``transform_by_quat_yaw_approx``) equals the exact atan2 yaw rotation
    (``quat_apply_yaw``) only when roll/pitch are ~0, and diverges otherwise.
    """
    torch.manual_seed(12)
    # Pure-yaw quats: the cheap approximation and the exact path must agree.
    yaw = torch.empty(16).uniform_(-3.14, 3.14)
    flat = geom.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    v = torch.randn(16, 3)
    assert torch.allclose(geom.transform_by_quat_yaw_approx(v, flat), geom.quat_apply_yaw(flat, v), atol=ATOL)

    # With roll/pitch, the approximation must measurably diverge from the exact yaw.
    tilted = _rand_quats(64)
    w = torch.randn(64, 3)
    assert (geom.transform_by_quat_yaw_approx(w, tilted) - geom.quat_apply_yaw(tilted, w)).abs().max() > 0.1


def test_quat_box_minus_matches_oracle():
    torch.manual_seed(13)
    q1, q2 = _rand_quats(16), _rand_quats(16)
    assert torch.allclose(geom.quat_box_minus(q1, q2), _ref_quat_box_minus(q1, q2), atol=ATOL)


def test_quat_error_magnitude_matches_oracle():
    torch.manual_seed(14)
    q1, q2 = _rand_quats(64), _rand_quats(64)
    assert torch.allclose(geom.quat_error_magnitude(q1, q2), _ref_quat_error_magnitude(q1, q2), atol=ATOL)


def test_quat_error_magnitude_zero_at_identity():
    """The atan2 form stays finite/zero where the oracle's axis-angle path
    takes its Taylor branch.
    """
    torch.manual_seed(15)
    q = _rand_quats(8)
    assert torch.allclose(geom.quat_error_magnitude(q, q), torch.zeros(8), atol=1e-6)


def test_wrap_to_pi_matches_oracle():
    torch.manual_seed(16)
    a = torch.randn(50) * 12
    assert torch.allclose(geom.wrap_to_pi(a), _ref_wrap_to_pi(a), atol=ATOL)
    assert (geom.wrap_to_pi(a).abs() <= torch.pi + 1e-6).all()


def test_wrap_to_pi_numpy_and_float_match_torch():
    torch.manual_seed(16)
    a = (torch.randn(50) * 12).double()
    assert np.allclose(geom.wrap_to_pi(a.numpy()), geom.wrap_to_pi(a).numpy())
    # A float stays a float, and +pi (an odd multiple of pi from above) wraps to +pi, as the torch path does.
    assert isinstance(geom.wrap_to_pi(3 * math.pi), float)
    assert geom.wrap_to_pi(3 * math.pi) == geom.wrap_to_pi(torch.tensor(3 * math.pi, dtype=torch.float64)).item()
    assert geom.wrap_to_pi(-0.25 - 2 * math.pi) == pytest.approx(-0.25)


def test_subtract_frame_transforms_matches_oracle():
    torch.manual_seed(17)
    t01, q01 = torch.randn(16, 3), _rand_quats(16)
    t02, q02 = torch.randn(16, 3), _rand_quats(16)
    t_g, q_g = geom.subtract_frame_transforms(t01, q01, t02, q02)
    t_r, q_r = _ref_subtract_frame_transforms(t01, q01, t02, q02)
    assert torch.allclose(t_g, t_r, atol=ATOL)
    assert torch.allclose(q_g, q_r, atol=ATOL)


def test_subtract_frame_transforms_default_args_match_oracle():
    """``t02`` / ``q02`` default to zero translation / identity rotation."""
    torch.manual_seed(18)
    t01, q01 = torch.randn(16, 3), _rand_quats(16)
    t_g, q_g = geom.subtract_frame_transforms(t01, q01)
    t_r, q_r = _ref_subtract_frame_transforms(t01, q01)
    assert torch.allclose(t_g, t_r, atol=ATOL)
    assert torch.allclose(q_g, q_r, atol=ATOL)


def test_transform_points_matches_oracle_batched_and_unbatched():
    torch.manual_seed(19)
    pts, pos, q = torch.randn(16, 5, 3), torch.randn(16, 3), _rand_quats(16)
    assert torch.allclose(geom.transform_points(pts, pos, q), _ref_transform_points(pts, pos, q), atol=ATOL)
    assert torch.allclose(geom.transform_points(pts, None, q), _ref_transform_points(pts, None, q), atol=ATOL)
    assert torch.allclose(geom.transform_points(pts, pos, None), _ref_transform_points(pts, pos, None), atol=ATOL)
    pts2, p3, q4 = torch.randn(5, 3), torch.randn(3), _rand_quats(1)[0]
    assert torch.allclose(geom.transform_points(pts2, p3, q4), _ref_transform_points(pts2, p3, q4), atol=ATOL)


# ---------------------------------------------------------------------------
# Batched (..., 4) leading-dim handling — the migrated terms call these with a
# (N, K, ...) leading shape.
# ---------------------------------------------------------------------------


def test_geom_ops_handle_extra_leading_dims():
    torch.manual_seed(20)
    q, q2, v = _rand_quats(4, 5), _rand_quats(4, 5), torch.randn(4, 5, 3)
    assert torch.allclose(geom.quat_mul(q, q2), _ref_quat_mul(q, q2), atol=ATOL)
    assert torch.allclose(geom.yaw_quat(q), _ref_yaw_quat(q), atol=ATOL)
    assert torch.allclose(geom.quat_error_magnitude(q, q2), _ref_quat_error_magnitude(q, q2), atol=ATOL)
    assert torch.allclose(geom.axis_angle_from_quat(q), _ref_axis_angle_from_quat(q), atol=ATOL)
    for got, ref in zip(geom.euler_xyz_from_quat(q), _ref_euler_xyz_from_quat(q)):
        assert got.shape == q.shape[:-1]
        assert torch.allclose(got, ref, atol=ATOL)
    # yaw-approx must preserve leading dims (regression: the old .view(-1, 4) flattened them)
    assert geom.transform_by_quat_yaw_approx(v, q).shape == v.shape


# ---------------------------------------------------------------------------
# Non-unit-quaternion behaviour. The codebase keeps unit quats everywhere, but
# these document the contract for the genesis-direct and geom paths.
# ---------------------------------------------------------------------------


def _nonunit_quats(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, 4) * 1.7


def test_quat_mul_renormalises_product():
    """Genesis renormalises the Hamilton product, so the output is unit
    regardless of input norms (drift-resistant; differs from the oracle's scaled
    product on non-unit inputs — documented, not a regression for rotations).
    """
    torch.manual_seed(23)
    q1, q2 = _nonunit_quats(16), _nonunit_quats(16)
    out = geom.quat_mul(q1, q2)
    assert torch.allclose(out.norm(dim=-1), torch.ones(16), atol=ATOL)


# ---------------------------------------------------------------------------
# align_up_to_z
# ---------------------------------------------------------------------------


def _assert_proper_rotation_aligning(R, up, front=None) -> None:
    R = torch.as_tensor(R, dtype=torch.float64)
    for v, target in ((up, (0.0, 0.0, 1.0)), (front, (1.0, 0.0, 0.0))):
        if v is None:
            continue
        v = torch.tensor(v, dtype=torch.float64)
        assert torch.allclose(R @ (v / torch.linalg.norm(v)), torch.tensor(target, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(torch.det(R), torch.tensor(1.0, dtype=torch.float64), atol=ATOL), f"det={torch.det(R)}"


@pytest.mark.parametrize("up", [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.3, -0.4, 0.86), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)])
def test_align_up_to_z_is_a_proper_rotation_and_its_branches_agree(up):
    """Regression: for up=-Z the numpy branch returned diag(1, 1, -1), a reflection with no quaternion."""
    R_np = torch.as_tensor(geom.align_up_to_z(up, as_torch=False), dtype=torch.float32)
    R_tc = torch.as_tensor(geom.align_up_to_z(up, as_torch=True), dtype=torch.float32)
    _assert_proper_rotation_aligning(R_np, up)
    _assert_proper_rotation_aligning(R_tc, up)
    assert torch.allclose(R_np, R_tc, atol=1e-5)


@pytest.mark.parametrize("as_torch", [False, True])
def test_align_up_and_front_negative_z_is_a_proper_rotation(as_torch):
    """``entities/base.py`` bakes ``R_to_quat`` of this into the morph offset, so a reflection breaks it."""
    up, front = (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)
    _assert_proper_rotation_aligning(geom.align_up_and_front(up, front, as_torch=as_torch), up, front)


@pytest.mark.parametrize("as_torch", [False, True])
def test_a_zero_vector_is_refused(as_torch):
    with pytest.raises(ValueError):
        geom.align_up_to_z((0.0, 0.0, 0.0), as_torch=as_torch)
    with pytest.raises(ValueError):
        geom.align_up_and_front((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), as_torch=as_torch)
    with pytest.raises(ValueError):
        geom.align_up_and_front((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), as_torch=as_torch)


# ---------------------------------------------------------------------------
# random_quat_uniform: no oracle, so the moments a Haar-uniform sampler must have are pinned.
# ---------------------------------------------------------------------------


def test_random_quat_uniform_draws_unit_quaternions_of_the_requested_dtype():
    q = geom.random_quat_uniform(512)
    assert torch.allclose(q.norm(dim=-1), torch.ones(512), atol=ATOL)
    assert geom.random_quat_uniform(8, dtype=torch.float64).dtype == torch.float64


def test_random_quat_uniform_is_haar_uniform():
    """Uniform on SO(3) => uniform on the 3-sphere: every component has mean 0
    and second moment 1/4 (the four must share the unit norm equally).
    """
    torch.manual_seed(0)
    q = geom.random_quat_uniform(200_000, dtype=torch.float64)
    assert torch.allclose(q.mean(dim=0), torch.zeros(4, dtype=torch.float64), atol=0.01)
    assert torch.allclose(q.square().mean(dim=0), torch.full((4,), 0.25, dtype=torch.float64), atol=0.01)
    # E[|w|] = 4/(3*pi) for a uniform point on S^3 (marginal density (2/pi)*sqrt(1-w^2)).
    assert q[:, 0].abs().mean().item() == pytest.approx(4.0 / (3.0 * torch.pi), abs=0.01)


def test_random_quat_uniform_rotates_an_axis_uniformly():
    """The image of a fixed axis must be uniform on S^2: mean 0, per-axis
    second moment 1/3. This is the check that catches a sampler biased toward
    the identity (which would leave the axis near its original direction).
    """
    torch.manual_seed(0)
    q = geom.random_quat_uniform(200_000, dtype=torch.float64)
    axis = gg.transform_by_quat(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(q.shape[0], 3), q)
    assert torch.allclose(axis.mean(dim=0), torch.zeros(3, dtype=torch.float64), atol=0.01)
    assert torch.allclose(axis.square().mean(dim=0), torch.full((3,), 1.0 / 3.0, dtype=torch.float64), atol=0.01)


def test_quat_unique_has_nonnegative_real_part_and_same_rotation():
    torch.manual_seed(21)
    q = _rand_quats(32)
    q[:16] = -q[:16]
    u = geom.quat_unique(q)
    assert (u[:, 0] >= 0).all()
    assert torch.allclose(geom.quat_error_magnitude(u, q), torch.zeros(32), atol=1e-5)


@pytest.mark.parametrize("given", ["both", "t12", "q12", "neither"])
def test_combine_frame_transforms_matches_oracle(given):
    """Either operand may be omitted, standing for zero translation / identity rotation."""
    torch.manual_seed(23)
    t01, q01 = torch.randn(16, 3), _rand_quats(16)
    t12 = torch.randn(16, 3) if given in ("both", "t12") else None
    q12 = _rand_quats(16) if given in ("both", "q12") else None
    t_g, q_g = geom.combine_frame_transforms(t01, q01, t12, q12)
    t_r, q_r = _ref_combine_frame_transforms(t01, q01, t12, q12)
    assert torch.allclose(t_g, t_r, atol=ATOL)
    assert torch.allclose(q_g, q_r, atol=ATOL)


def test_combine_frame_transforms_is_genesis_pose_transform():
    """The header-table mapping: ``combine(t01, q01, t12, q12) == transform_pos_quat_by_trans_quat(t12, q12, ...)``."""
    torch.manual_seed(24)
    t01, q01, t12, q12 = torch.randn(16, 3), _rand_quats(16), torch.randn(16, 3), _rand_quats(16)
    t_g, q_g = geom.combine_frame_transforms(t01, q01, t12, q12)
    t_r, q_r = gg.transform_pos_quat_by_trans_quat(t12, q12, t01, q01)
    assert torch.allclose(t_g, t_r, atol=ATOL)
    assert torch.allclose(q_g, q_r, atol=ATOL)


def test_combine_frame_transforms_inverts_subtract():
    """``combine(T_01, subtract(T_01, T_02)) == T_02``."""
    torch.manual_seed(25)
    t01, q01 = torch.randn(16, 3), _rand_quats(16)
    t02, q02 = torch.randn(16, 3), _rand_quats(16)
    t_c, q_c = geom.combine_frame_transforms(t01, q01, *geom.subtract_frame_transforms(t01, q01, t02, q02))
    assert torch.allclose(t_c, t02, atol=ATOL)
    assert torch.allclose(geom.quat_error_magnitude(q_c, q02), torch.zeros(16), atol=1e-4)


# ---------------------------------------------------------------------------
# quat_swing_twist
# ---------------------------------------------------------------------------

_Y = np.array([0.0, 1.0, 0.0])


def _about(axis, angle):
    """The wxyz quaternion of ``angle`` about the unit ``axis`` (hand-written oracle)."""
    return np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * np.asarray(axis, dtype=np.float64))])


@pytest.mark.parametrize(
    "quat, twist, swing",
    [
        (_about(_Y, 0.7), _about(_Y, 0.7), _about(_Y, 0.0)),  # pure twist about the axis
        (_about([1.0, 0.0, 0.0], 0.7), _about(_Y, 0.0), _about([1.0, 0.0, 0.0], 0.7)),  # pure swing
        # Exactly 180 degrees about an axis perpendicular to the twist axis (w == 0, written out:
        # cos(pi/2) is 6e-17, not 0): no twist component at all, and a plain renormalisation of
        # the zero projection gives NaN.
        (np.array([0.0, 1.0, 0.0, 0.0]), _about(_Y, 0.0), np.array([0.0, 1.0, 0.0, 0.0])),
        (_about(_Y, 0.0), _about(_Y, 0.0), _about(_Y, 0.0)),  # identity
    ],
)
def test_quat_swing_twist_known_decompositions(quat, twist, swing):
    got_twist, got_swing = geom.quat_swing_twist(quat[None], _Y)
    np.testing.assert_allclose(got_twist[0], twist, atol=1e-12)
    np.testing.assert_allclose(got_swing[0], swing, atol=1e-12)


def test_quat_swing_twist_recomposes_with_a_twist_about_the_axis():
    rng = np.random.default_rng(31)
    quat = rng.normal(size=(64, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    twist, swing = geom.quat_swing_twist(quat, _Y)
    np.testing.assert_allclose(gg.transform_quat_by_quat(swing, twist), quat, atol=1e-12)  # twist ⊗ swing
    np.testing.assert_allclose(twist[:, [1, 3]], 0.0, atol=1e-12)  # twist's vector part lies on the axis
    np.testing.assert_allclose(swing[:, 2], 0.0, atol=1e-12)  # and the swing has no rotation about it


def test_quat_swing_twist_torch_matches_numpy():
    rng = np.random.default_rng(32)
    quat = rng.normal(size=(16, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    twist_np, swing_np = geom.quat_swing_twist(quat, _Y)
    twist_t, swing_t = geom.quat_swing_twist(torch.from_numpy(quat), torch.from_numpy(_Y))
    np.testing.assert_allclose(twist_t.numpy(), twist_np, atol=1e-12)
    np.testing.assert_allclose(swing_t.numpy(), swing_np, atol=1e-12)
