"""Geometry helpers: axis alignment, point transforms, rotation conversions."""

import genesis as gs
import numpy as np
import torch

# Re-exported from genesis, so ``from eden.utils.geom import inv_transform_by_quat`` resolves to the
# same object as the genesis import. It is every genesis quaternion / pose helper that takes a batch
# of torch tensors, not only the subset this module calls: a partial set turns the next natural guess
# into an ImportError. That covers building and decomposing a 4x4 transform too; *applying* one is
# Eden's own ``transform_by_T`` / ``transform_by_R`` (numerically genesis', kept for their docstrings).
# Left out on purpose: the euler converters, whose convention differs from this module's
# ``quat_from_euler_xyz`` behind the same import path, and anything without a batched torch path.
# ``tests/utils/test_geom.py`` classifies every genesis helper in the family and pins the list.
from genesis.utils.geom import (  # noqa: F401 -- re-exported, see above
    R_to_quat,
    R_to_rot6d,
    R_to_T,
    T_to_quat,
    T_to_trans,
    T_to_trans_quat,
    axis_angle_to_quat,
    inv_quat,
    inv_transform_by_quat,
    inv_transform_by_T,
    inv_transform_by_trans_quat,
    inv_transform_pos_quat_by_trans_quat,
    normalize,
    quat_to_R,
    rot6d_to_R,
    slerp,
    trans_quat_to_T,
    trans_R_to_T,
    trans_to_T,
    transform_by_quat,
    transform_by_trans_quat,
    transform_by_trans_R,
    transform_pos_quat_by_trans_quat,
    transform_quat_by_quat,
)


def align_up_to_z(up, as_torch=False):
    """
    Aligns the given 'up' vector to the Z-up direction (0, 0, 1).

    Parameters
    ----------
    up : list | tuple | torch.Tensor
        A 3D vector to align to Z-up. If list/tuple, it's converted to a tensor.
    as_torch : bool
        If True, the returned rotation matrix is a torch.Tensor. Otherwise, it's a numpy.ndarray.

    Returns
    -------
    R : torch.Tensor | np.ndarray
        A 3x3 rotation matrix that aligns 'up' to the Z-axis.
    """
    if as_torch:
        if not isinstance(up, torch.Tensor):
            up = torch.tensor(up, dtype=gs.tc_float, device=gs.device)

        norm = torch.linalg.norm(up)
        if norm == 0.0:
            raise ValueError("`up` must be a non-zero vector")
        up = up / norm

        target = torch.tensor([0.0, 0.0, 1.0], dtype=gs.tc_float, device=gs.device)
        if torch.allclose(up, target, atol=1e-6):
            return torch.eye(3, dtype=gs.tc_float, device=gs.device)

        if torch.allclose(up, -target, atol=1e-6):
            # Proper 180-degree rotation about X (det=+1); see the numpy branch
            # below for why diag(1, 1, -1) would be wrong (reflection).
            return torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=gs.tc_float,
                device=gs.device,
            )

        axis = torch.cross(up, target)
        axis = axis / torch.linalg.norm(axis)

        dot_product = torch.dot(up, target)
        angle = torch.acos(torch.clamp(dot_product, min=-1.0, max=1.0))

        K = torch.zeros((3, 3), dtype=gs.tc_float, device=gs.device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]

        R = torch.eye(3, dtype=gs.tc_float, device=gs.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    else:
        up = np.array(up, dtype=np.float64)
        norm = np.linalg.norm(up)
        if norm == 0.0:
            raise ValueError("`up` must be a non-zero vector")
        up /= norm

        target = np.array([0, 0, 1])
        if np.allclose(up, target, atol=1e-6):
            return np.eye(3)

        if np.allclose(up, -target, atol=1e-6):
            # Proper 180-degree rotation around X (det=+1). NOTE: diag(1, 1, -1)
            # would be a REFLECTION (det=-1), not a rotation — R_to_quat of it
            # is undefined and yields garbage orientations.
            return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

        axis = np.cross(up, target)
        angle = np.arccos(np.dot(up, target))

        axis /= np.linalg.norm(axis)

        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R


def align_up_and_front(up, front, as_torch=False):
    """Return a rotation matrix R that realigns an asset's canonical axes to world axes.

    The asset's canonical up and front axes are given by `up` and `front` (in the
    asset's own mesh frame) so that after applying R::

        R @ up    = (0, 0, 1)   (asset's up aligned to world Z)
        R @ front = (1, 0, 0)   (asset's front aligned to world X)

    Parameters
    ----------
    up : list | tuple | torch.Tensor
        The asset's "up" direction in its own canonical/mesh frame.
    front : list | tuple | torch.Tensor
        The asset's "front" direction in its own canonical/mesh frame.
    as_torch : bool
        If True, the returned rotation matrix is a torch.Tensor. Otherwise, it's a numpy.ndarray.

    Returns
    -------
    R : torch.Tensor | np.ndarray
        A 3x3 rotation matrix.
    """
    if as_torch:
        if not isinstance(up, torch.Tensor):
            up = torch.tensor(up, dtype=gs.tc_float, device=gs.device)
        if not isinstance(front, torch.Tensor):
            front = torch.tensor(front, dtype=gs.tc_float, device=gs.device)

        norm_up = torch.linalg.norm(up)
        norm_front = torch.linalg.norm(front)
        if norm_up == 0.0 or norm_front == 0.0:
            raise ValueError("`up` and `front` must be non-zero vectors")
        up = up / norm_up
        front = front / norm_front

        R1 = align_up_to_z(up, as_torch=True)

        front_transformed = R1 @ front

        front_xy = front_transformed.clone()
        front_xy[2] = 0.0

        norm_front_xy = torch.linalg.norm(front_xy)
        if norm_front_xy < 1e-6:  # front parallel to up: no yaw to resolve
            return R1

        front_xy /= norm_front_xy

        angle_of_front_xy = torch.atan2(front_xy[1], front_xy[0])

        cos_val = torch.cos(-angle_of_front_xy)
        sin_val = torch.sin(-angle_of_front_xy)

        R2 = torch.eye(3, dtype=gs.tc_float, device=gs.device)
        R2[0, 0] = cos_val
        R2[0, 1] = -sin_val
        R2[1, 0] = sin_val
        R2[1, 1] = cos_val

        return R2 @ R1

    else:
        up = np.array(up, dtype=np.float64)
        front = np.array(front, dtype=np.float64)

        # Normalize the vectors (zero vectors would otherwise reach
        # align_up_to_z as NaN and bypass its guard)
        norm_up = np.linalg.norm(up)
        norm_front = np.linalg.norm(front)
        if norm_up == 0.0 or norm_front == 0.0:
            raise ValueError("`up` and `front` must be non-zero vectors")
        up /= norm_up
        front /= norm_front

        # First align "up" to (0,0,1), then project the rotated front onto the XY plane
        R1 = align_up_to_z(up, as_torch=False)
        front_transformed = R1 @ front
        front_xy = front_transformed.copy()
        front_xy[2] = 0

        norm_front_xy = np.linalg.norm(front_xy)
        if norm_front_xy < 1e-6:  # front parallel to up: no yaw to resolve
            return R1

        front_xy /= norm_front_xy

        # Yaw about Z that takes front_xy to (1,0,0)
        angle = np.arctan2(front_xy[1], front_xy[0])
        R2 = np.array(
            [
                [np.cos(-angle), -np.sin(-angle), 0],
                [np.sin(-angle), np.cos(-angle), 0],
                [0, 0, 1],
            ]
        )

        return R2 @ R1


@torch.jit.script
def transform_by_R(pos: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Transform 3D points by a 3x3 rotation matrix or a batch of matrices.

    Parameters
    ----------
    pos: torch.Tensor
        A torch tensor of 3D points. Can be a single point
        (3,), a batch of points (B, 3), or a batched batch of points (B, N, 3).
    R: torch.Tensor
        The 3x3 rotation matrix or a batch of B rotation
        matrices of shape (B, 3, 3).

    Returns
    -------
    torch.Tensor
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3

    dim_added = False
    if R.ndim == 2:
        R = R[None]
        dim_added = True
    if pos.ndim == 3:
        new_pos = (R @ pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    elif pos.ndim == 2:
        new_pos = (R @ pos[:, :, None])[..., 0]
    else:
        new_pos = (R @ pos[None, :, None])[..., 0]
        if dim_added:
            new_pos = new_pos[0]
    return new_pos


@torch.jit.script
def transform_by_T(pos: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Transform 3D points by a 4x4 transformation matrix or a batch of matrices.

    Parameters
    ----------
    pos: torch.Tensor
        A torch tensor of 3D points. Can be a single point
        (3,), a batch of points (B, 3), or a batched batch of points (B, N, 3).
    T: torch.Tensor
        The 4x4 transformation matrix or a batch of B transformation
        matrices of shape (B, 4, 4).

    Returns
    -------
    torch.Tensor
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3, "Input positions must have 3 dimensions"

    if T.ndim == 2:
        T = T.reshape(1, 4, 4)

    if pos.ndim > 1:
        ones_shape = pos.shape[:-1] + (1,)
        pos_hom = torch.cat([pos, torch.ones(ones_shape, dtype=pos.dtype, device=pos.device)], dim=-1)
    else:
        pos_hom = torch.cat([pos, torch.tensor([1.0], dtype=pos.dtype, device=pos.device)])

    if pos_hom.ndim == 1:
        pos_hom = pos_hom.reshape(1, 1, -1)
    elif pos_hom.ndim == 2:
        assert T.shape[0] == 1 or T.shape[0] == pos.shape[0], f"{T.shape}, {pos.shape}"
        pos_hom = pos_hom.reshape(-1, 1, 4)

    pos_hom_t = pos_hom.swapaxes(-1, -2)  # (..., N, 4) -> (..., 4, N)
    transformed_hom = T @ pos_hom_t
    transformed_hom = transformed_hom.swapaxes(-1, -2)[..., :3]

    if pos.ndim == 1:
        transformed_hom = transformed_hom.reshape(-1)
    elif pos.ndim == 2:
        transformed_hom = transformed_hom.reshape(-1, 3)
    return transformed_hom


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
    R = rot6d_to_R(d6)
    return R_to_quat(R)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 6D rotation rep via genesis ``R_to_rot6d``: the first two *rows* of ``R``.

    Layout is ``[R00, R01, R02, R10, R11, R12]``. Two other 6D layouts exist in
    this repo and are **not** interchangeable with it at a trained policy's
    input — they are different scalars, not a reordering:

    * first two *columns*, row-major (``quat_to_R(q)[..., :2].flatten(-2)`` =
      ``[R00, R01, R10, R11, R20, R21]``) — what IsaacLab's
      ``matrix_from_quat(q)[..., :2].reshape(-1, 6)`` produces, used by the
      SONIC port and by ``motion_future_links_ori_b(representation="rot6d")``.
    * :func:`quat_to_tan_norm` (``[R·x̂, R·ẑ]``) — the ProtoMotions rep.
    """
    R = quat_to_R(quat)
    return R_to_rot6d(R)


def R_to_rot6d_cols(R: torch.Tensor) -> torch.Tensor:
    """6D rotation rep as the first two *columns* of ``R``, flattened row-major.

    Layout ``[R00, R01, R10, R11, R20, R21]`` — what IsaacLab's
    ``matrix_from_quat(q)[..., :2].reshape(-1, 6)`` produces, and what the SONIC
    deploy C++ writes as ``{R[0][0], R[0][1], R[1][0], R[1][1], R[2][0], R[2][1]}``.

    A column-stacked ``[R00, R10, R20, R01, R11, R21]`` would also round-trip
    through a Gram-Schmidt decoder but feeds a trained head *different scalars*,
    so the flatten order is part of the contract, not a detail. This is also NOT
    :func:`quat_to_rot6d` (genesis: the first two *rows*) — see that docstring.
    """
    return R[..., :2].flatten(-2)


def quat_to_rot6d_cols(quat: torch.Tensor) -> torch.Tensor:
    """:func:`R_to_rot6d_cols` from a wxyz quaternion ``(..., 4) -> (..., 6)``."""
    return R_to_rot6d_cols(quat_to_R(quat))


def R_to_tan_norm(R: torch.Tensor) -> torch.Tensor:
    """ProtoMotions ``quat_to_tan_norm`` 6D rotation rep from a rotation matrix ``(..., 3, 3)``.

    Returns ``[R·x̂, R·ẑ]`` concatenated — columns 0 and 2 of ``R`` — as ``(..., 6)``.
    Takes ``R`` (not a quat) for the callers that already built the matrix; see
    :func:`quat_to_tan_norm` for the quaternion entry point.

    This is a *different* 6D layout from :func:`quat_to_rot6d` — see its
    docstring for the three variants in play.
    """
    return torch.cat([R[..., 0], R[..., 2]], dim=-1)


def quat_to_tan_norm(quat: torch.Tensor) -> torch.Tensor:
    """ProtoMotions ``quat_to_tan_norm`` 6D rotation rep from a wxyz quaternion ``(..., 4)``.

    Returns ``[R(q)·x̂, R(q)·ẑ]`` as ``(..., 6)``. Both halves are of the form
    ``R(q) @ v`` for a fixed canonical vector, so the output scalars are the
    same whether the source stack encodes ``q`` wxyz or xyzw — only the *input*
    ordering matters, and Eden is wxyz throughout.
    """
    return R_to_tan_norm(quat_to_R(quat))


def transform_by_quat_yaw_approx(v: torch.Tensor, quat: torch.Tensor):
    """Rotate ``v`` by an *approximate* yaw-only quaternion (cheap path).

    The heading is approximated by zeroing ``qx``/``qy`` and renormalising,
    which is fast but only equals the true yaw when roll/pitch are ~0. For the
    exact atan2 heading use :func:`quat_apply_yaw` (note the swapped argument
    order). The two are deliberately NOT aliases — see
    :func:`yaw_quat` and ``test_yaw_quat_differs_from_zeroing_qx_qy_shortcut``.
    """
    quat_yaw = quat.clone()
    quat_yaw[..., 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return transform_by_quat(v, quat_yaw)


def xyzw_to_wxyz(q):
    """Reorder the last axis of a quaternion from (x, y, z, w) to (w, x, y, z).

    Accepts either ``numpy.ndarray`` or ``torch.Tensor``; fancy indexing on the
    last axis is backend-agnostic.
    """
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q):
    """Reorder the last axis of a quaternion from (w, x, y, z) to (x, y, z, w).

    Accepts either ``numpy.ndarray`` or ``torch.Tensor``.
    """
    return q[..., [1, 2, 3, 0]]


def segment_closest_points(p0, p1, q0, q1):
    """Closest points between segment batches ``[p0, p1]`` and ``[q0, q1]``, all ``(..., 3)`` numpy arrays.

    Returns ``(on_p, on_q)``, each ``(..., 3)``. A zero-length segment is a point, in either slot; parallel
    segments return one of their equally close pairs.
    """
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    eps = 1e-12
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.maximum(denom, eps), 0.0, 1.0), 0.0)
    t = np.where(e > eps, (b * s + f) / np.maximum(e, eps), 0.0)
    # Re-clamp t, then recompute s for the clamped t. A point second segment (e = 0) fixes t = 0 unclamped, so s is
    # recomputed there too: left at 0, it read p0 for any point.
    t_clamped = np.clip(t, 0.0, 1.0)
    s = np.where(
        (t != t_clamped) | (e <= eps),
        np.where(a > eps, np.clip((b * t_clamped - c) / np.maximum(a, eps), 0.0, 1.0), 0.0),
        s,
    )
    return p0 + s[..., None] * d1, q0 + t_clamped[..., None] * d2


def planar_rigid_transform(a: np.ndarray, b: np.ndarray, min_spread: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares planar rigid motion ``b ~ a @ R.T + t`` of ``(n, 2)`` point sets: ``(R (2, 2), t (2,))``.

    Translation only when the points are too close together to show a rotation (rms distance from their centroid
    under ``min_spread``): for coincident points the fit is ``atan2(0, 0)``, an arbitrary angle (three probes sharing
    one source point would otherwise read a planted foot as turned 180 deg).
    """
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    if len(a) < 2 or np.sqrt(((a - ca) ** 2).sum(axis=1).mean()) < min_spread:
        return np.eye(2), cb - ca
    m = (b - cb).T @ (a - ca)
    theta = np.arctan2(m[1, 0] - m[0, 1], m[0, 0] + m[1, 1])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rot, cb - ca @ rot.T


def skew(v):
    """``(..., 3)`` vectors to ``(..., 3, 3)`` cross-product matrices: ``skew(a) @ b == cross(a, b)``.

    Accepts either ``numpy.ndarray`` or ``torch.Tensor``.
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x) if isinstance(v, torch.Tensor) else np.zeros_like(x)
    rows = ((zero, -z, y), (z, zero, -x), (-y, x, zero))
    if isinstance(v, torch.Tensor):
        return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


# ---------------------------------------------------------------------------
# Quaternion / frame operations (wxyz)
# ---------------------------------------------------------------------------
# ``isaac_math`` below names Isaac Lab's ``isaaclab.utils.math`` (BSD-3-Clause): the wxyz
# convention, names, and signatures here match its functions, and every function is pinned
# equivalent to a vendored copy of its counterpart in ``tests/utils/test_geom.py``.
#
# Only the ops that ``genesis.utils.geom`` does NOT cover exactly with a torch
# path are defined here; the exact ones are genesis' own functions, imported from
# either module (re-exported at the top of this file):
#
#   isaac_math.matrix_from_quat      -> gs.utils.geom.quat_to_R
#   isaac_math.quat_conjugate        -> gs.utils.geom.inv_quat
#   isaac_math.quat_from_angle_axis  -> gs.utils.geom.axis_angle_to_quat
#   isaac_math.quat_apply            -> gs.utils.geom.transform_by_quat       (vec, quat)
#   isaac_math.quat_apply_inverse    -> gs.utils.geom.inv_transform_by_quat   (vec, quat)
#
# One more is a thin wrapper over a genesis op, kept for isaac's signature and optional operands:
#
#   isaac_math.combine_frame_transforms -> gs.utils.geom.transform_pos_quat_by_trans_quat  (t12, q12, t01, q01)
#
# The ops below are kept because genesis either has no torch equivalent
# (euler/axis-angle converters are numba/numpy-only or degree-based), composes
# differently, or — for ``quat_mul`` — reverses the operand order in a way that
# is a silent footgun at every call site (``transform_quat_by_quat`` applies its
# *left* operand on the left).


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``q1 ∘ q2`` (wxyz), matching isaac_math's argument order.

    ``genesis.transform_quat_by_quat(v, u)`` applies ``u`` on the left, so the
    arguments are swapped here to reproduce ``q1 ∘ q2``. Kept as a named wrapper
    (rather than calling genesis at the use sites) precisely because both
    operands are quaternions, so the order swap is an invisible footgun. genesis
    renormalises the product (drift-resistant): identical to
    ``isaac_math.quat_mul`` for unit inputs; for non-unit inputs it returns a
    unit quaternion rather than the scaled product.
    """
    return transform_quat_by_quat(q2, q1)


# Norm floor for quat_nlerp. Unit inputs never get near it (after the hemisphere flip the blend
# keeps a norm of at least sqrt(1/2)); it only keeps zero-norm inputs from dividing by zero.
_NLERP_EPS = 1e-12


def quat_nlerp(q0, q1, t):
    """Shortest-path normalized linear interpolation (NLERP) of two quaternions.

    Flips ``q1`` into ``q0``'s hemisphere, blends linearly by ``t``, and
    renormalises. Batched over any leading dims; ``t`` is a scalar, one weight
    per quaternion (shape ``q0.shape[:-1]``), or already shaped ``(..., 1)``.
    Accepts ``numpy.ndarray`` or ``torch.Tensor`` (both operands the same type).
    Usable before ``gs.init()`` (its epsilon is not ``gs.EPS``, which is None
    until then), like the other helpers here that deployment hot loops call.

    NLERP is a component-wise blend, so it is convention-agnostic — valid for
    both wxyz and scalar-last (xyzw) quaternions. It is cheaper than a true
    spherical interpolation and matches the reference SONIC pose resampler's
    interpolant; for exact SLERP use ``genesis.utils.geom.slerp`` instead. There
    is no NLERP in ``genesis.utils.geom``, which is why this lives here.
    """
    if isinstance(q0, torch.Tensor):
        if isinstance(t, torch.Tensor) and t.dim() == q0.dim() - 1:
            t = t.unsqueeze(-1)  # one weight per quaternion -> broadcast over its 4 components
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        q = (1.0 - t) * q0 + t * q1
        return q / q.norm(dim=-1, keepdim=True).clamp_min(_NLERP_EPS)
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    t = np.asarray(t)
    if t.ndim == q0.ndim - 1:
        t = t[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    q = (1.0 - t) * q0 + t * q1
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _NLERP_EPS)


# Below this norm the twist projection has vanished: a 180-degree rotation about an axis
# perpendicular to the twist axis, whose twist is the identity.
_SWING_TWIST_EPS = 1e-9


def quat_swing_twist(quat, axis):
    """Swing-twist decomposition of wxyz quaternions about ``axis``: ``quat = twist ⊗ swing``.

    ``twist`` is the rotation about ``axis`` (the quaternion's vector part projected onto
    it, renormalised); ``swing`` is the remainder, ``inv(twist) ⊗ quat``. A 180-degree
    rotation about an axis perpendicular to ``axis`` has no twist component at all (the
    projection is zero); the twist is then the identity and the swing carries the whole
    rotation, rather than the NaN a plain renormalisation gives.

    Batched over leading dims; ``axis`` is a unit 3-vector. Accepts ``numpy.ndarray`` or
    ``torch.Tensor``. The product and inverse are ``genesis.utils.geom``'s, whose numpy
    path needs ``gs.init()``.
    """
    if isinstance(quat, torch.Tensor):
        axis = torch.as_tensor(axis, dtype=quat.dtype, device=quat.device)
        proj = (quat[..., 1:] * axis).sum(dim=-1, keepdim=True) * axis
        twist = torch.cat([quat[..., :1], proj], dim=-1)
        norm = twist.norm(dim=-1, keepdim=True)
        identity = torch.zeros_like(twist)
        identity[..., 0] = 1.0
        twist = torch.where(norm > _SWING_TWIST_EPS, twist / norm.clamp_min(_SWING_TWIST_EPS), identity)
    else:
        quat = np.asarray(quat, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        proj = (quat[..., 1:] @ axis)[..., None] * axis
        twist = np.concatenate([quat[..., :1], proj], axis=-1)
        norm = np.linalg.norm(twist, axis=-1, keepdims=True)
        identity = np.zeros_like(twist)
        identity[..., 0] = 1.0
        twist = np.where(norm > _SWING_TWIST_EPS, twist / np.maximum(norm, _SWING_TWIST_EPS), identity)
    return twist, transform_quat_by_quat(quat, inv_quat(twist))


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Quaternion ``(..., 4)`` wxyz from extrinsic XYZ Euler angles (radians): ``R = Rz(yaw) Ry(pitch) Rx(roll)``.

    The inverse of :func:`euler_xyz_from_quat`.

    Ported verbatim (torch) from ``isaac_math.quat_from_euler_xyz``. genesis
    ``euler_to_quat`` is numpy-only and expects degrees, so it is not a drop-in
    for the vectorised torch term code.
    """
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


def euler_xyz_from_quat(
    quat: torch.Tensor, wrap_to_2pi: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll/pitch/yaw (radians, extrinsic XYZ) from a wxyz quaternion ``(..., 4)``.

    Ported verbatim (torch) from ``isaac_math.euler_xyz_from_quat``. genesis
    ``quat_to_xyz`` is numpy-only and returns degrees.
    """
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(
        torch.abs(sin_pitch) >= 1,
        torch.copysign(torch.full_like(sin_pitch, torch.pi / 2.0), sin_pitch),
        torch.asin(sin_pitch),
    )

    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw


def intrinsic_xyz_from_quat(quat: torch.Tensor) -> torch.Tensor:
    r"""Intrinsic ``X-Y-Z`` Euler angles ``(..., 3)`` (radians) of a wxyz quaternion ``(..., 4)``.

    The angles ``(a, b, c)`` with ``R(q) = Rx(a) @ Ry(b) @ Rz(c)`` — the coordinates of three stacked hinge
    joints about the body's x, y, then z axes (the MJCF/Genesis way of writing a 3-DOF ball, e.g. PHC's SMPL
    humanoid). ``b`` is in ``[-pi/2, pi/2]``; ``a`` and ``c`` in ``(-pi, pi]``.

    Not :func:`euler_xyz_from_quat`, which is the *extrinsic* XYZ (roll/pitch/yaw) decomposition. It is the
    torch form of ``genesis.utils.geom.quat_to_xyz(q, rpy=False)``, which is numba/numpy-only.
    """
    R = quat_to_R(quat)
    a = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    b = torch.asin(R[..., 0, 2].clamp(-1.0, 1.0))
    c = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack([a, b, c], dim=-1)


def _check_lock_band(lock_band: float) -> None:
    if not 0.0 <= lock_band < 1.0:
        raise ValueError(
            f"lock_band={lock_band!r} is a bound on |cos b|, so it must be in [0, 1): 0 for the exact angles, "
            "e.g. 0.3 to damp within about 17 deg of gimbal lock."
        )


def intrinsic_xyz_nearest(
    abc: torch.Tensor,
    near: torch.Tensor,
    lock_band: float = 0.0,
) -> torch.Tensor:
    r"""The intrinsic ``X-Y-Z`` angles ``(..., 3)`` of the rotation ``abc`` names that lie nearest ``near``.

    A rotation has two intrinsic X-Y-Z decompositions, ``(a, b, c)`` and ``(a + pi, pi - b, c + pi)``, and each
    angle is defined only up to whole turns. :func:`intrinsic_xyz_from_quat` returns the one with ``b`` in
    ``[-pi/2, pi/2]``, so a body turning past ``b = +-pi/2`` — a knee bending past 90 deg about y — jumps by
    ``pi`` in both ``a`` and ``c`` there. This returns, of both decompositions with every angle shifted by whole
    turns towards ``near``, the one at the smaller squared distance from it: the target three stacked hinges
    reach without swinging through unrelated rotations. ``abc`` and ``near`` broadcast against each other.

    Near ``b = +-pi/2`` (gimbal lock) the x and z hinges turn about nearly one axis, so ``v = a - sin(b) c`` is
    ill-conditioned: a rotation ``delta`` off that axis needs ``v`` to move by about ``delta / cos(b)``, up to
    ``pi`` from a twitch of the input, and at the lock ``v`` is not defined at all. With ``lock_band > 0``, where
    ``|cos b| < lock_band`` the result moves along ``(1, 0, -sin b)`` — ``v`` alone, to first order — by the
    fraction ``(1 - (cos(b) / lock_band)^2)^2`` of the way to ``near``'s ``v``: fully at the lock, where that
    names the same rotation, and not at all at the band's edge or outside it. The price is a rotation error
    within the band, from any ``near``, of just under ``lock_band / 2`` (0.14 rad at 0.3); ``0`` keeps the exact
    angles.
    """
    _check_lock_band(lock_band)
    abc, near = torch.broadcast_tensors(abc, near)
    dual = torch.stack([abc[..., 0] + torch.pi, torch.pi - abc[..., 1], abc[..., 2] + torch.pi], dim=-1)
    candidates = torch.stack([abc, dual])  # (2, ..., 3)
    candidates = candidates + 2 * torch.pi * torch.round((near - candidates) / (2 * torch.pi))
    dist = ((candidates - near) ** 2).sum(dim=-1)  # (2, ...)
    nearest = torch.where((dist[1] < dist[0])[..., None], candidates[1], candidates[0])
    if lock_band > 0.0:
        a, b, c = nearest.unbind(-1)
        sin_b = torch.sin(b)
        inside = (1.0 - (torch.cos(b) / lock_band) ** 2).clamp(min=0.0) ** 2
        dv = (near[..., 0] - a) - sin_b * (near[..., 2] - c)
        step = inside * dv / (1.0 + sin_b**2)  # along (1, 0, -sin b), which moves v by exactly ``inside * dv``
        nearest = torch.stack([a + step, b, c - sin_b * step], dim=-1)
    return nearest


def intrinsic_xyz_rates(abc: torch.Tensor, ang_vel: torch.Tensor, lock_band: float = 0.0) -> torch.Tensor:
    r"""Rates ``(..., 3)`` of the intrinsic ``X-Y-Z`` angles ``abc`` that turn the body at ``ang_vel``.

    ``ang_vel`` ``(..., 3)`` is the body-frame angular velocity of ``R = Rx(a) @ Ry(b) @ Rz(c)`` — ``w`` with
    ``R^T dR/dt = [w]x`` — so for three stacked hinges it is the child's angular velocity relative to the
    parent, in the child's axes; the result is what to write to the hinges' velocities. It inverts
    ``w = (cos b cos c, -cos b sin c, sin b) da + (sin c, cos c, 0) db + (0, 0, 1) dc``, whose determinant is
    ``cos b``: ``da = p / cos b`` with ``p = cos c w_x - sin c w_y``, so near gimbal lock a slow turn off the
    x/z axis spins the x and z hinges at about ``|w| / cos b``, without bound.

    With ``lock_band > 0``, where ``|cos b| < lock_band`` the ``1 / cos b`` becomes ``(1 - weight) / cos b``, with
    ``weight`` the fraction :func:`intrinsic_xyz_nearest` damps ``v`` by: at most ``1.09 / lock_band``, and 0 at
    the lock. ``dc`` still matches ``w_z``, so only ``p``, the part of ``w`` the hinges barely reach there, is
    given up. ``0`` keeps the exact rates (infinite at the lock).
    """
    _check_lock_band(lock_band)
    _, b, c = abc.unbind(-1)
    sin_b, cos_b, sin_c, cos_c = torch.sin(b), torch.cos(b), torch.sin(c), torch.cos(c)
    p = cos_c * ang_vel[..., 0] - sin_c * ang_vel[..., 1]
    db = sin_c * ang_vel[..., 0] + cos_c * ang_vel[..., 1]
    if lock_band == 0.0:
        da = p / cos_b
    else:
        x = cos_b / lock_band
        inside = x.abs() < 1.0
        # 1 - (1 - x^2)^2 = x^2 (2 - x^2), so inside the band (1 - weight) / cos b = x (2 - x^2) / lock_band.
        da = p * torch.where(inside, x * (2.0 - x**2) / lock_band, 1.0 / torch.where(inside, 1.0, cos_b))
    return torch.stack([da, db, ang_vel[..., 2] - sin_b * da], dim=-1)


def quat_unique(q: torch.Tensor) -> torch.Tensor:
    """The wxyz quaternion with a non-negative real part, of the two (``q``, ``-q``) naming one rotation.

    Ported verbatim from ``isaac_math.quat_unique``. Shape ``(..., 4)`` in and out. Accepts
    ``numpy.ndarray`` or ``torch.Tensor``.
    """
    if not isinstance(q, torch.Tensor):
        return np.where(q[..., 0:1] < 0, -q, q)
    return torch.where(q[..., 0:1] < 0, -q, q)


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Shortest-arc spherical interpolation from wxyz ``q0`` to ``q1`` at ``t``, PHC's ``torch_utils.slerp``.

    ``q0`` / ``q1`` are ``(..., 4)``; ``t`` broadcasts against them with a trailing axis of 1 (``(..., 1)``).
    Ported verbatim, quirks included, because PHC-trained policies are graded against references
    interpolated this way: when the two rotations are within ~0.1 deg (``sin(half angle) < 1e-3``) the result
    is their plain average whatever ``t`` is, and the output is not renormalized.
    """
    cos_half = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(cos_half < 0, -q1, q1)
    cos_half = cos_half.abs().clamp(max=1.0)
    half = torch.acos(cos_half)
    sin_half = torch.sqrt(1.0 - cos_half * cos_half)
    ratio_a = torch.sin((1 - t) * half) / sin_half.clamp(min=1e-8)
    ratio_b = torch.sin(t * half) / sin_half.clamp(min=1e-8)
    out = ratio_a * q0 + ratio_b * q1
    out = torch.where(sin_half.abs() < 0.001, 0.5 * q0 + 0.5 * q1, out)
    return torch.where(cos_half >= 1.0, q0, out)


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Axis-angle vector ``(..., 3)`` from a wxyz quaternion ``(..., 4)``.

    The vector magnitude is the rotation angle (radians), canonicalised to ``[0, pi]``:
    the input is first flipped to the ``w >= 0`` hemisphere, so a quaternion and its
    negation — the same rotation — map to the same vector. Ported (torch) from
    ``isaac_math.axis_angle_from_quat``, including the small-angle Taylor branch. genesis
    ``quat_to_rotvec`` is numba/numpy-only and lacks the Taylor expansion near zero, so it
    is not a torch drop-in; it also reads ``gs.EPS``, so it raises before ``gs.init()``.

    Accepts ``numpy.ndarray`` or ``torch.Tensor``. Composed with ``R_to_quat`` it is the
    rotation-matrix log map (``R -> rotation vector``).
    """
    quat = quat_unique(quat)
    if not isinstance(quat, torch.Tensor):
        half_angle = np.arctan2(np.linalg.norm(quat[..., 1:], axis=-1), quat[..., 0])
        angle = 2.0 * half_angle
        ratio = np.where(angle > eps, np.sin(half_angle) / np.maximum(angle, eps), 0.5 - angle * angle / 48)
        return quat[..., 1:4] / ratio[..., None]
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(  # clamped: torch.where evaluates both branches, and
        angle.abs() > eps,  # an unguarded 0/0 here is NaN in the *discarded* one, which kills grads
        torch.sin(half_angle) / angle.clamp(min=eps),
        0.5 - angle * angle / 48,
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


def quat_from_axis_angle(axis_angle: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Quaternion (wxyz) ``(..., 4)`` from an axis-angle vector ``(..., 3)``.

    Inverse of :func:`axis_angle_from_quat` (magnitude = angle in radians), with the same
    small-angle Taylor branch so a zero vector maps to identity instead of ``NaN``.

    The round trip preserves the *rotation* for any input, and the vector itself only for
    ``||v|| <= pi``: beyond that, ``axis_angle_from_quat`` canonicalises to the ``w >= 0``
    hemisphere and returns the equal-but-shorter ``(2*pi - ||v||)`` turn about the flipped axis.
    SMPL/AMASS consume the rotation, so the distinction does not matter there — but do not rely
    on the vector surviving a round trip unchanged.

    genesis ``axis_angle_to_quat`` takes the angle and a *separate unit axis*, so it is not a
    drop-in for the packed-vector form SMPL/AMASS pose data uses; genesis ``rotvec_to_R`` and
    ``rotvec_to_quat`` have no small-angle branch (``NaN`` for a zero vector before ``gs.init()``).

    Accepts ``numpy.ndarray`` or ``torch.Tensor``. Composed with ``quat_to_R`` it is the
    rotation-vector exp map (``rotation vector -> R``).
    """
    if not isinstance(axis_angle, torch.Tensor):
        angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
        half_angle = 0.5 * angle
        ratio = np.where(angle > eps, np.sin(half_angle) / np.maximum(angle, eps), 0.5 - angle * angle / 48)
        return np.concatenate([np.cos(half_angle), axis_angle * ratio], axis=-1)
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    sin_half_angle_over_angle = torch.where(  # sin(θ/2)/θ, Taylor-expanded near zero
        angle.abs() > eps, torch.sin(half_angle) / angle.clamp(min=eps), 0.5 - angle * angle / 48
    )
    return torch.cat([torch.cos(half_angle), axis_angle * sin_half_angle_over_angle], dim=-1)


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Heading-only (yaw) component of a wxyz quaternion, same shape in/out.

    Extracts the true yaw via ``atan2`` and rebuilds a pure-z quaternion. NOTE:
    this is NOT the same as zeroing qx/qy and renormalising
    (:func:`transform_by_quat_yaw_approx`), which only agrees when roll/pitch are zero.
    Ported from ``isaac_math.yaw_quat`` so the heading-invariant motion warp
    keeps identical behaviour.
    """
    shape = quat.shape
    q = quat.reshape(-1, 4)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    out = torch.zeros_like(q)
    out[:, 0] = torch.cos(yaw / 2)
    out[:, 3] = torch.sin(yaw / 2)
    return normalize(out).reshape(shape)


def yaw_from_quat(quat):
    r"""Heading angle in radians about world Z from a wxyz quaternion. ``(..., 4) -> (...)``.

    ``atan2(2(wz + xy), 1 - 2(yy + zz))`` — the same angle :func:`yaw_quat` builds
    its pure-z quaternion from, and identical to ``euler_xyz_from_quat(q)[2]``
    without computing and discarding roll and pitch.

    Accepts ``numpy.ndarray`` or ``torch.Tensor``. Deliberately hand-written
    rather than delegating to ``genesis.utils.geom.quat_to_xyz(q, rpy=True)[..., 2]``:
    that path is a numba kernel that reads ``gs.EPS``, so it raises a
    ``TypingError`` until ``gs.init()`` has run, and it decomposes all three
    Euler angles to return one. Callers here are offline tools and deployment
    hot loops that must not depend on simulator init order.

    NOTE: this is the *heading*. ``genesis``'s ``quat_to_xyz`` **defaults** to
    ``rpy=False``, an intrinsic ``Rx·Ry·Rz`` decomposition whose third angle is a
    body-z angle, not the heading — the two agree only when roll/pitch are ~0.
    """
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    atan2 = torch.atan2 if isinstance(quat, torch.Tensor) else np.arctan2
    return atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z))


def heading_from_quat(quat, forward=None):
    """Heading angle in radians about world Z of a body-frame ``forward`` axis. ``(..., 4) -> (...)``.

    Rotates ``forward`` (``(..., 3)``, broadcast against ``quat``'s batch shape; body ``+x`` when
    ``None``) into the world by the wxyz ``quat`` and returns ``atan2(y, x)`` of the result.
    Accepts ``numpy.ndarray`` or ``torch.Tensor``.

    With the default ``+x`` this is the angle :func:`yaw_from_quat` computes, but not the same
    floats: ``transform_by_quat`` divides by ``|q|²`` where ``yaw_from_quat`` assumes a unit
    quaternion, so the two drift apart by ``O(|q|² - 1)`` (most near pitch ±90°, where the rotated
    axis is almost vertical). This is the spelling a ported policy that took its heading from the
    rotated forward axis needs, to stay bit-exact.
    """
    is_torch = isinstance(quat, torch.Tensor)
    if forward is None:
        forward = torch.zeros_like(quat[..., :3]) if is_torch else np.zeros_like(quat[..., :3])
        forward[..., 0] = 1.0
    else:
        forward = (torch.broadcast_to if is_torch else np.broadcast_to)(forward, (*quat.shape[:-1], 3))
    forward_w = transform_by_quat(forward, quat)
    return (torch.atan2 if is_torch else np.arctan2)(forward_w[..., 1], forward_w[..., 0])


def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` ``(N, 3)`` by only the yaw component of the wxyz ``quat``.

    Composition ``transform_by_quat(vec, yaw_quat(quat))`` — mirrors
    ``isaac_math.quat_apply_yaw``.
    """
    return transform_by_quat(vec, yaw_quat(quat))


def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Box-minus (axis-angle difference) of two wxyz quaternions: ``log(q1 ∘ q2⁻¹)``.

    Composition mirroring ``isaac_math.quat_box_minus`` (which uses the
    conjugate; for the unit quaternions used here conjugate == inverse).
    Returns an axis-angle vector ``(N, 3)``.
    """
    return axis_angle_from_quat(quat_mul(q1, inv_quat(q2)))


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between two unit wxyz quaternions. Shape ``(...)``.

    The magnitude of the relative rotation ``q1 ∘ q2⁻¹``, computed as
    ``2·atan2(‖vec‖, |w|)``. Equivalent to isaac_math's
    ``‖axis_angle(q1 ∘ q2⁻¹)‖`` (verified in tests) but uses ``atan2`` directly,
    so it is robust at the identity without a Taylor-series branch.
    """
    q_diff = quat_mul(q1, inv_quat(q2))
    return 2.0 * torch.atan2(torch.linalg.norm(q_diff[..., 1:], dim=-1), q_diff[..., 0].abs())


def random_quat_uniform(
    n: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample ``n`` quaternions uniformly on SO(3). Returns ``(n, 4)`` wxyz.

    Shoemake's method: three uniform variates map to a rotation drawn from the
    Haar measure, so every orientation is equally likely (unlike normalising a
    Gaussian 4-vector by hand, which is also uniform but easy to get subtly
    wrong, or perturbing Euler angles, which is not).

    ``genesis.utils.geom.random_quaternion`` implements the same method but is
    numpy-and-CPU-only, so it is not a drop-in for a per-env sampler on device.

    Parameters
    ----------
    n : int
        Number of quaternions to sample.
    device : torch.device | str | None, optional
        Device to allocate on. Defaults to None (torch's current default).
    dtype : torch.dtype, optional
        Floating dtype of the result. Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Unit quaternions ``(n, 4)`` in ``(w, x, y, z)`` order.
    """
    u1, u2, u3 = torch.rand(3, n, device=device, dtype=dtype)
    sqrt1 = torch.sqrt(1.0 - u1)
    sqrt_u1 = torch.sqrt(u1)
    a = 2.0 * torch.pi * u2
    b = 2.0 * torch.pi * u3
    return torch.stack(
        (sqrt1 * torch.sin(a), sqrt1 * torch.cos(a), sqrt_u1 * torch.sin(b), sqrt_u1 * torch.cos(b)),
        dim=-1,
    )


def wrap_to_pi(angles):
    r"""Wrap angles (radians) to :math:`[-\pi, \pi]`. Same shape in/out.

    Ported verbatim from ``isaac_math.wrap_to_pi``; genesis has no equivalent. Accepts a
    ``torch.Tensor``, a ``numpy.ndarray`` or a Python float (returned as a float).
    """
    if isinstance(angles, torch.Tensor):
        wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
        return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)
    wrapped_angle = (np.asarray(angles) + np.pi) % (2 * np.pi)
    wrapped = np.where((wrapped_angle == 0) & (np.asarray(angles) > 0), np.pi, wrapped_angle - np.pi)
    return float(wrapped) if np.ndim(angles) == 0 else wrapped


def combine_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t12: torch.Tensor | None = None,
    q12: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose of frame 2 in frame 0: ``T_02 = T_01 · T_12`` (wxyz), the inverse of :func:`subtract_frame_transforms`.

    Mirrors ``isaac_math.combine_frame_transforms``. With both operands given it is genesis
    ``transform_pos_quat_by_trans_quat(t12, q12, t01, q01)``; kept as a named wrapper for isaac's
    frame-order signature and the optional operands: ``t12`` / ``q12`` default to zero translation /
    identity rotation. Returns ``(t02, q02)`` with shapes ``(N, 3)`` and ``(N, 4)``.
    """
    q02 = quat_mul(q01, q12) if q12 is not None else q01
    t02 = t01 + transform_by_quat(t12, q01) if t12 is not None else t01
    return t02, q02


def subtract_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor | None = None,
    q02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose of frame 2 in frame 1: ``T_12 = T_01⁻¹ · T_02`` (wxyz).

    Pure composition of the quaternion ops above — mirrors
    ``isaac_math.subtract_frame_transforms`` exactly. ``t02`` / ``q02`` default
    to zero translation / identity rotation. Returns ``(t12, q12)`` with shapes
    ``(N, 3)`` and ``(N, 4)``.
    """
    q10 = inv_quat(q01)
    q12 = quat_mul(q10, q02) if q02 is not None else q10
    t12 = transform_by_quat(t02 - t01, q10) if t02 is not None else transform_by_quat(-t01, q10)
    return t12, q12


def transform_points(
    points: torch.Tensor,
    pos: torch.Tensor | None = None,
    quat: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Transform points into a target frame: ``p' = R · p + t`` (wxyz quat).

    Ported verbatim (torch) from ``isaac_math.transform_points``. ``points`` is
    ``(N, P, 3)`` or ``(P, 3)``; ``pos`` is ``(N, 3)``/``(3,)`` (default: no
    translation); ``quat`` is wxyz ``(N, 4)``/``(4,)`` (default: identity). genesis
    has no batched point-cloud transform with this broadcasting contract.
    """
    points_batch = points.clone()
    is_batched = points_batch.dim() == 3
    if points_batch.dim() == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 2 or dim = 3: got shape {points.shape}")
    if not (pos is None or pos.dim() == 1 or pos.dim() == 2):
        raise ValueError(f"Expected pos to have dim = 1 or dim = 2: got shape {pos.shape}")
    if not (quat is None or quat.dim() == 1 or quat.dim() == 2):
        raise ValueError(f"Expected quat to have dim = 1 or dim = 2: got shape {quat.shape}")
    # -- rotation
    if quat is not None:
        rot_mat = quat_to_R(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]  # (3, 3) -> (1, 3, 3)
        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
        points_batch = points_batch.transpose_(1, 2)
    # -- translation
    if pos is not None:
        if pos.dim() == 1:
            pos = pos[None, None, :]  # (3,) -> (1, 1, 3)
        else:
            pos = pos[:, None, :]  # (N, 3) -> (N, 1, 3)
        points_batch = points_batch + pos
    if not is_batched:
        points_batch = points_batch.squeeze(0)  # (1, P, 3) -> (P, 3)
    return points_batch
