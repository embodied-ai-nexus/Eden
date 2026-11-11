"""Built-in dataset modifiers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import torch
from genesis.utils.geom import axis_angle_to_quat, transform_by_quat, transform_quat_by_quat

from eden.datasets.modifiers.base import DATASET_MODIFIER_REGISTRY, DatasetModifier, build_modifier
from eden.options.datasets import DatasetModifierOptions

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import BatchMotionSequence


@DATASET_MODIFIER_REGISTRY.register()
class Compose(DatasetModifier):
    """Chain multiple dataset modifiers in sequence.

    Parameters
    ----------
    modifiers : tuple[DatasetModifierOptions, ...] | list[DatasetModifierOptions]
        Ordered sequence of child modifier configurations. Each modifier's
        :meth:`apply` runs against the output of the previous one.

    Example
    -------
    ::

        Compose.configure(
            modifiers=[
                Translate.configure(offset=(0.0, 0.0, 0.05)),
                Translate.configure(
                    offset=(0.5, 0.0, 0.0),
                    motion_sequence_keys=("obj",),
                ),
            ]
        )
    """

    modifiers: tuple[DatasetModifierOptions, ...] | list[DatasetModifierOptions] = ()

    def __init__(self, options: DatasetModifierOptions):
        super().__init__(options=options)
        self._modifiers: list[DatasetModifier] = [build_modifier(mod_options) for mod_options in self.modifiers]

    def apply(self, motion_sequences):
        selected = {k: motion_sequences[k] for k in self._selected_keys(motion_sequences)}
        for mod in self._modifiers:
            selected = mod.apply(selected)
        merged = dict(motion_sequences)
        merged.update(selected)
        return merged


@DATASET_MODIFIER_REGISTRY.register()
class Translate(DatasetModifier):
    """Shift world-frame translations of every motion by a constant offset.

    The offset is added to ``base_pos_w`` and, if populated, ``links_pos_w``.
    Body-frame fields (``links_pos_b``) and all velocity fields are invariant
    under a pure translation and are left untouched.

    Parameters
    ----------
    offset : tuple[float, float, float]
        (x, y, z) offset in the dataset world frame. Typical use is a z-only
        offset to lift trajectories off the ground for a taller robot.
    """

    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def apply_to_sequence(self, motion_seq: "BatchMotionSequence") -> "BatchMotionSequence":
        if all(v == 0.0 for v in self.offset):
            return motion_seq
        if motion_seq.base_pos_w is None:
            return motion_seq

        device = motion_seq.base_pos_w.device
        dtype = motion_seq.base_pos_w.dtype
        offset = torch.tensor(self.offset, dtype=dtype, device=device)

        # ConcatBatchTensor stores frames in a flat (total_frames, 3) buffer;
        # a broadcast-add over the last dim is equivalent to adding the offset
        # to every frame of every motion.
        motion_seq.base_pos_w.concatenated_tensor.add_(offset)
        if motion_seq.links_pos_w is not None:
            motion_seq.links_pos_w.concatenated_tensor.add_(offset)
        return motion_seq


@DATASET_MODIFIER_REGISTRY.register()
class Rotate(DatasetModifier):
    """Rotate world-frame fields about a given axis through the world origin.

    Rotates every world-frame position, quaternion, linear velocity, and angular
    velocity by a single rigid rotation. Body-frame fields (``links_*_b``) are
    invariant under a world rotation and are left untouched.

    Because the rotation is about the world origin, combining :class:`Rotate`
    with :class:`Translate` is order-sensitive — rotate first, then translate
    to apply the rotation about the current origin; translate first, then
    rotate to orbit the motion about a point.

    Parameters
    ----------
    axis : tuple[float, float, float]
        Unit rotation axis in the world frame. Automatically normalized. Default
        is the Z axis, i.e. a yaw rotation.
    angle_deg : float
        Rotation angle in degrees (right-handed about ``axis``).
    """

    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    angle_deg: float = 0.0

    _VEC_FIELDS: ClassVar[tuple[str, ...]] = (
        "base_pos_w",
        "base_lin_vel_w",
        "base_ang_vel_w",
        "links_pos_w",
        "links_lin_vel_w",
        "links_ang_vel_w",
    )
    _QUAT_FIELDS: ClassVar[tuple[str, ...]] = ("base_quat_w", "links_quat_w")

    def apply_to_sequence(self, motion_seq: "BatchMotionSequence") -> "BatchMotionSequence":
        if self.angle_deg == 0.0:
            return motion_seq

        # No world-frame field populated: nothing to rotate.
        ref_field = next(
            (
                getattr(motion_seq, name)
                for name in self._VEC_FIELDS + self._QUAT_FIELDS
                if getattr(motion_seq, name) is not None
            ),
            None,
        )
        if ref_field is None:
            return motion_seq

        device = ref_field.device
        dtype = ref_field.dtype

        axis = torch.tensor(self.axis, dtype=dtype, device=device)
        if axis.norm().item() < 1e-8:
            raise ValueError(f"Rotate axis must be non-zero, got {self.axis}")
        q = axis_angle_to_quat(torch.tensor(math.radians(self.angle_deg), dtype=dtype, device=device), axis)

        for name in self._VEC_FIELDS:
            cbt = getattr(motion_seq, name)
            if cbt is None:
                continue
            t = cbt.concatenated_tensor  # (N, 3) or (N, L, 3)
            t.copy_(transform_by_quat(t, q))

        for name in self._QUAT_FIELDS:
            cbt = getattr(motion_seq, name)
            if cbt is None:
                continue
            t = cbt.concatenated_tensor  # (N, 4) or (N, L, 4)
            # transform_quat_by_quat(v, u) returns u * v — pass u=q to left-multiply.
            t.copy_(transform_quat_by_quat(t, q))

        return motion_seq


@DATASET_MODIFIER_REGISTRY.register()
class MirrorLR(DatasetModifier):
    """Mirror motion data across a world-frame plane and relabel symmetric parts.

    Mirroring is an improper transformation: vectors, pseudovectors, and unit
    quaternions all follow different sign rules, which this modifier applies
    automatically. What it cannot know is how the robot's DOFs and links map
    to each other under reflection — that is robot-specific, and the user
    provides it as explicit integer swap tables.

    For a mirror plane with normal on axis ``a`` (e.g. Y for left/right):

    * Positions and linear velocities flip component ``a``.
    * Angular velocities (pseudovectors under improper transform) flip the
      *other two* components.
    * Unit quaternions ``(w, x, y, z)`` flip every ``xyz`` component except
      the one on axis ``a``; ``w`` is unchanged.

    The sign pattern is applied to both world-frame and body-frame link fields.
    The body-frame rule follows from applying the same mirror consistently to
    the base frame; for perfectly symmetric robots this is exact, and for
    slightly asymmetric ones it is a close approximation — consumers requiring
    strict accuracy should recompute body-frame fields from world-frame fields
    downstream.

    Ordering within the modifier: axis sign flips → DOF sign flips → DOF swaps
    → link swaps. User-supplied ``dofs_sign_flip`` and ``dofs_swap`` both
    reference the *original* DOF indices (before swapping).

    Parameters
    ----------
    axis : int
        Mirror-plane normal index: 0 = X, 1 = Y (default, left-right), 2 = Z.
    dofs_swap : tuple[tuple[int, int], ...]
        Pairs of DOF indices to swap after mirroring. Example for a biped:
        ``((hip_l_idx, hip_r_idx), (knee_l_idx, knee_r_idx), ...)``.
    dofs_sign_flip : tuple[int, ...]
        DOF indices whose stored value is negated under mirroring — typically
        joints that rotate about an axis orthogonal to the mirror plane (e.g.
        a hip-roll axial joint when mirroring left/right).
    links_swap : tuple[tuple[int, int], ...]
        Pairs of link indices to swap after mirroring. Applied to every
        ``links_*`` field along the link dimension.
    """

    axis: int = 1
    dofs_swap: tuple[tuple[int, int], ...] = ()
    dofs_sign_flip: tuple[int, ...] = ()
    links_swap: tuple[tuple[int, int], ...] = ()

    _POS_LIKE_FIELDS: ClassVar[tuple[str, ...]] = (
        "base_pos_w",
        "base_lin_vel_w",
        "links_pos_w",
        "links_lin_vel_w",
        "links_pos_b",
        "links_lin_vel_b",
    )
    _ANG_VEL_FIELDS: ClassVar[tuple[str, ...]] = (
        "base_ang_vel_w",
        "links_ang_vel_w",
        "links_ang_vel_b",
    )
    _QUAT_FIELDS: ClassVar[tuple[str, ...]] = ("base_quat_w", "links_quat_w", "links_quat_b")
    _breaks_body_model: ClassVar[str | None] = (
        "mirroring swaps a hand's or body's side, which a body-model pose cannot follow by sign flips"
    )
    _DOF_FIELDS: ClassVar[tuple[str, ...]] = ("dofs_pos", "dofs_vel")
    _LINK_FIELDS: ClassVar[tuple[str, ...]] = (
        "links_pos_w",
        "links_pos_b",
        "links_quat_w",
        "links_quat_b",
        "links_lin_vel_w",
        "links_lin_vel_b",
        "links_ang_vel_w",
        "links_ang_vel_b",
    )

    def apply_to_sequence(self, motion_seq: "BatchMotionSequence") -> "BatchMotionSequence":
        if self.axis not in (0, 1, 2):
            raise ValueError(f"MirrorLR.axis must be 0, 1, or 2; got {self.axis}")

        # Pick a populated field to read device/dtype from; otherwise this is a no-op.
        ref = next(
            (
                getattr(motion_seq, name)
                for name in (*self._POS_LIKE_FIELDS, *self._ANG_VEL_FIELDS, *self._QUAT_FIELDS, *self._DOF_FIELDS)
                if getattr(motion_seq, name) is not None
            ),
            None,
        )
        if ref is None:
            return motion_seq
        device, dtype = ref.device, ref.dtype

        # Sign patterns — see docstring for derivation.
        pos_sign = torch.ones(3, dtype=dtype, device=device)
        pos_sign[self.axis] = -1.0
        ang_sign = -pos_sign  # flip the *other two* components
        quat_sign = torch.ones(4, dtype=dtype, device=device)
        quat_sign[1:] = ang_sign

        # 1. Axis sign flips (position-like, angular velocity, quaternion).
        for name in self._POS_LIKE_FIELDS:
            cbt = getattr(motion_seq, name)
            if cbt is not None:
                cbt.concatenated_tensor.mul_(pos_sign)
        for name in self._ANG_VEL_FIELDS:
            cbt = getattr(motion_seq, name)
            if cbt is not None:
                cbt.concatenated_tensor.mul_(ang_sign)
        for name in self._QUAT_FIELDS:
            cbt = getattr(motion_seq, name)
            if cbt is not None:
                cbt.concatenated_tensor.mul_(quat_sign)

        # 2. Per-DOF sign flips (original-index convention).
        if self.dofs_sign_flip:
            flip_idx = torch.tensor(list(self.dofs_sign_flip), dtype=torch.long, device=device)
            for name in self._DOF_FIELDS:
                cbt = getattr(motion_seq, name)
                if cbt is None:
                    continue
                cbt.concatenated_tensor[:, flip_idx] *= -1.0

        # 3. DOF swaps.
        if self.dofs_swap:
            for name in self._DOF_FIELDS:
                cbt = getattr(motion_seq, name)
                if cbt is None:
                    continue
                t = cbt.concatenated_tensor
                perm = list(range(t.shape[1]))
                for a, b in self.dofs_swap:
                    perm[a], perm[b] = perm[b], perm[a]
                idx = torch.tensor(perm, dtype=torch.long, device=t.device)
                t.copy_(t[:, idx])

        # 4. Link swaps.
        if self.links_swap:
            for name in self._LINK_FIELDS:
                cbt = getattr(motion_seq, name)
                if cbt is None:
                    continue
                t = cbt.concatenated_tensor  # (N, L, *)
                if t.shape[1] == 0:
                    continue
                perm = list(range(t.shape[1]))
                for a, b in self.links_swap:
                    perm[a], perm[b] = perm[b], perm[a]
                idx = torch.tensor(perm, dtype=torch.long, device=t.device)
                t.copy_(t[:, idx])

        return motion_seq
