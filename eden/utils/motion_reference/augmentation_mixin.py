"""Symmetry-augmentation mixin for motion-reference buffers."""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from functools import cached_property

import torch

# Single-source XZ-plane mirror primitives. ``SymmetryAugmentation`` (the flat
# policy obs/action variant) calls the same helpers; see eden/utils/symmetry.py.
from eden.utils.symmetry import (
    mirror_xz_ang_vel_,
    mirror_xz_joint,
    mirror_xz_quat_,
    mirror_xz_y_negate_,
)

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import (
        MotionReferenceData,
        MotionReferenceState,
    )
    from eden.options.managers.motions import MotionManagerOptions


class SymmetricAugmentationMixin:
    if TYPE_CHECKING:
        _options: MotionManagerOptions
        _env_symmetric_augmentation_mask: torch.Tensor

    @cached_property
    def _symmetric_augmentation_configured(self) -> bool:
        """Whether symmetric augmentation is fully configured (static config check).

        The per-env dynamic mask is handled inside _symmetric_augment_reference_data via early-return.
        """
        return (
            self._options.symmetric_augmentation
            and (self._options.symmetric_augmentation_joint_mapping is not None)
            and (self._options.symmetric_augmentation_joint_reverse_buf is not None)
            and (self._options.symmetric_augmentation_links_mapping is not None)
        )

    @cached_property
    def _joint_reverse_buf(self) -> torch.Tensor:
        """1-D tensor of per-joint sign multipliers for symmetric augmentation."""
        return torch.tensor(self._options.symmetric_augmentation_joint_reverse_buf, device=self.device)

    def _symmetric_augment_reference_data(
        self, data_buf: MotionReferenceData | MotionReferenceState, envs_idx: torch.Tensor
    ) -> None:
        """Symmetrically augment the reference buffer w.r.t x-z plane in the envs drawn to mirror, values changed in place.

        Every field :meth:`_mirror_rows` knows that the buffer holds is mirrored; one it lacks or holds as None is
        skipped — the initial reference state, for one, carries only the base and DOF fields.
        """
        envs_idx_to_augment = envs_idx[self._env_symmetric_augmentation_mask[envs_idx]]
        if len(envs_idx_to_augment) == 0:
            return
        for attr in (*_DOF_FIELDS, *_VECTOR_MIRRORS):
            value = getattr(data_buf, attr, None)
            if value is not None:
                value[envs_idx_to_augment] = self._mirror_rows(attr, value[envs_idx_to_augment])

    def _mirror_rows(self, attr: str, rows: torch.Tensor) -> torch.Tensor:
        """``rows`` of reference field ``attr`` mirrored across the x-z plane, possibly written into ``rows`` itself.

        ``rows`` has any leading dims; the field's data is in the last one or two.
        """
        if attr in _DOF_FIELDS:
            return mirror_xz_joint(rows, self._options.symmetric_augmentation_joint_mapping, self._joint_reverse_buf)
        if attr.startswith("links_"):
            rows = rows[..., self._options.symmetric_augmentation_links_mapping, :]
        return _VECTOR_MIRRORS[attr](rows)


#: The DOF fields mirror by swapping each joint with its bilateral counterpart (last dim) and flipping the signs that
#: mirror (:func:`mirror_xz_joint`).
_DOF_FIELDS = ("dofs_pos", "dofs_vel")

#: How every other reference field mirrors across the x-z plane: this primitive, on the vector or quaternion in its
#: last dim, after a ``links_*`` field has swapped each link with its bilateral counterpart (second-to-last dim).
_VECTOR_MIRRORS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "base_pos_w": mirror_xz_y_negate_,
    "base_quat_w": mirror_xz_quat_,
    "base_lin_vel_w": mirror_xz_y_negate_,
    "base_ang_vel_w": mirror_xz_ang_vel_,
    "links_pos_w": mirror_xz_y_negate_,
    "links_quat_w": mirror_xz_quat_,
    "links_pos_b": mirror_xz_y_negate_,
    "links_quat_b": mirror_xz_quat_,
    "links_lin_vel_b": mirror_xz_y_negate_,
    "links_ang_vel_b": mirror_xz_ang_vel_,
    "links_lin_vel_w": mirror_xz_y_negate_,
    "links_ang_vel_w": mirror_xz_ang_vel_,
}
