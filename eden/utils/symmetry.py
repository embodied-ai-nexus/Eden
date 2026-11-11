"""Bilateral (XZ-plane) symmetry primitives and a flat-tensor augmentation helper.

Eden has two consumers of XZ-plane symmetry:

1. ``SymmetricAugmentationMixin`` (``eden/utils/motion_reference/augmentation_mixin.py``)
   mirrors per-env fields of ``MotionReferenceData`` / ``MotionReferenceState`` —
   used by ``MotionManager`` so the policy sees both an original and a mirrored
   reference trajectory during motion-imitation training.

2. ``SymmetryAugmentation`` (this file) mirrors flat concatenated policy
   observations and actions — intended for RL training loops that double the
   batch by appending the mirrored copy (PPO/SAC symmetry augmentation).

The two operate on different data structures and have different lifecycle
semantics, so they remain distinct classes. The underlying tensor math —
joint-vector mirror, angular-velocity XZ negation, quaternion XZ negation,
y-axis negation — lives here as module-level free functions that both
consumers call. This keeps the math single-sourced.

Conventions:
- Quaternions are wxyz (Eden convention; see AGENTS.md).
- "XZ-plane mirror" = reflection across the body's sagittal plane.
- Angular velocity under XZ mirror: negate x and z components.
- Quaternion under XZ mirror (wxyz): negate the x and z components (indices 1 and 3).
- Linear velocity / position under XZ mirror: negate the y component.

Usage::

    from eden.utils.symmetry import SymmetryAugmentation

    sym = SymmetryAugmentation(
        dofs_name=robot_options.dofs_name,
        symmetry_joint_names={"left_hip_pitch_joint": "right_hip_pitch_joint", ...},
        flip_sign_joint_names=["left_hip_roll_joint", "right_hip_roll_joint", ...],
        obs_components={
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "dofs_pos": 29,
            "dofs_vel": 29,
            "last_action": 29,
            "sin_phase": 2,
            "cos_phase": 2,
            "command_lin_vel": 2,
            "command_ang_vel": 1,
        },
    )

    aug_obs = sym.augment_observations(obs_batch)    # (2*B, obs_dim)
    aug_act = sym.augment_actions(action_batch)      # (2*B, act_dim)
"""

from __future__ import annotations

import warnings
from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Shared XZ-plane mirror primitives
# ---------------------------------------------------------------------------


def mirror_xz_joint(joint_buf: torch.Tensor, joint_mapping: torch.Tensor, sign_mask: torch.Tensor) -> torch.Tensor:
    """Mirror a joint-indexed tensor along the XZ plane (reorder + sign flip on last dim).

    Returns a new tensor; does not modify the input. ``sign_mask`` is reshaped to
    broadcast over arbitrary leading dims.

    Parameters
    ----------
    joint_buf : torch.Tensor
        Tensor with last dimension equal to ``num_joints``.
    joint_mapping : torch.LongTensor
        Index map that swaps each joint with its bilateral counterpart, shape ``(num_joints,)``.
    sign_mask : torch.Tensor
        Per-joint sign multipliers (``±1``), shape ``(num_joints,)``.
    """
    num_dims = joint_buf.dim()
    if num_dims > 1:
        sign_mask = sign_mask.view(*([1] * (num_dims - 1)), -1)
    return joint_buf[..., joint_mapping] * sign_mask


def mirror_xz_ang_vel_(buf: torch.Tensor) -> torch.Tensor:
    """In-place XZ-plane mirror of an angular-velocity tensor.

    Last dim is ``(wx, wy, wz)``. XZ-plane reflection negates rotations about
    x and z; rotation about y is preserved. Returns the same tensor.
    """
    buf[..., 0] *= -1
    buf[..., 2] *= -1
    return buf


def mirror_xz_quat_(buf: torch.Tensor) -> torch.Tensor:
    """In-place XZ-plane mirror of a quaternion tensor (wxyz convention).

    Negates components at indices 1 and 3 (i.e. quat.x and quat.z). Returns the
    same tensor.
    """
    buf[..., 1] *= -1
    buf[..., 3] *= -1
    return buf


def mirror_xz_y_negate_(buf: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """In-place XZ-plane mirror of a position / linear-velocity tensor.

    Negates the y component (index 1) along ``axis``. Returns the same tensor.
    """
    if axis == -1 or axis == buf.dim() - 1:
        buf[..., 1] *= -1
    else:
        # Slice along the requested axis; rare path, kept simple.
        sl = [slice(None)] * buf.dim()
        sl[axis] = 1
        buf[tuple(sl)] *= -1
    return buf


# ---------------------------------------------------------------------------
# SymmetryAugmentation — flat policy obs/action mirror for RL training
# ---------------------------------------------------------------------------


class SymmetryAugmentation:
    """XZ-plane symmetry augmentation for bilateral robots.

    Operates on flat concatenated policy tensors. For motion-reference-data
    mirroring, see ``SymmetricAugmentationMixin`` in
    ``eden/utils/motion_reference/augmentation_mixin.py``.

    Parameters
    ----------
    dofs_name : list[str]
        Ordered DOF names matching the action/observation DOF ordering.
    symmetry_joint_names : dict[str, str]
        Mapping from each joint name to its bilateral counterpart.
        Both directions should be specified (left->right AND right->left).
    flip_sign_joint_names : list[str]
        Joint names whose values get negated under the mirror transform
        (e.g., roll and yaw joints).
    obs_components : dict[str, int]
        Ordered mapping of observation component name to its dimension.
        The concatenation order must match the actual observation tensor.
    device : torch.device | str
        Device of the index map and sign mask; must match the tensors being mirrored.
    """

    def __init__(
        self,
        dofs_name: Sequence[str],
        symmetry_joint_names: dict[str, str],
        flip_sign_joint_names: Sequence[str],
        obs_components: dict[str, int],
        device: torch.device | str = "cpu",
    ):
        self.device = device
        self.dofs_name = list(dofs_name)
        self.obs_components = obs_components
        self.n_dofs = len(dofs_name)

        name_to_idx = {name: i for i, name in enumerate(dofs_name)}

        # Build joint index mapping
        self.joint_index_map = torch.arange(self.n_dofs, device=device, dtype=torch.long)
        for j1, j2 in symmetry_joint_names.items():
            if j1 in name_to_idx and j2 in name_to_idx:
                self.joint_index_map[name_to_idx[j1]] = name_to_idx[j2]

        # Build sign flip mask
        flip_indices = {name_to_idx[n] for n in flip_sign_joint_names if n in name_to_idx}
        self.sign_flip_mask = torch.tensor(
            [-1.0 if i in flip_indices else 1.0 for i in range(self.n_dofs)],
            device=device,
            dtype=torch.float32,
        )

        # Precompute observation component slices and validate handlers
        self._obs_slices: dict[str, slice] = {}
        idx = 0
        for name, dim in obs_components.items():
            self._obs_slices[name] = slice(idx, idx + dim)
            idx += dim
            if not hasattr(self, f"_mirror_{name}"):
                warnings.warn(
                    f"SymmetryAugmentation: no _mirror_{name} handler found for observation "
                    f"component '{name}' (dim={dim}). It will be left unchanged under mirroring. "
                    f"If this is intentional (e.g. a scalar command), you can ignore this warning.",
                    stacklevel=2,
                )
        self._obs_dim = idx

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Mirror an action tensor. Shape (..., n_dofs)."""
        return mirror_xz_joint(actions, self.joint_index_map, self.sign_flip_mask)

    def mirror_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Mirror a concatenated observation tensor. Shape (B, obs_dim).

        Each named component is dispatched to the appropriate mirror function.
        """
        mirrored = obs.clone()
        for name, s in self._obs_slices.items():
            handler = getattr(self, f"_mirror_{name}", None)
            if handler is not None:
                mirrored[:, s] = handler(mirrored[:, s])
            # If no handler, leave component unchanged (e.g. scalars like stand_command)
        return mirrored

    def augment_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Concatenate original and mirrored observations. Returns (2*B, obs_dim)."""
        return torch.cat([obs, self.mirror_observations(obs)], dim=0)

    def augment_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Concatenate original and mirrored actions. Returns (2*B, n_dofs)."""
        return torch.cat([actions, self.mirror_actions(actions)], dim=0)

    # ------------------------------------------------------------------
    # Per-component mirror handlers
    # ------------------------------------------------------------------
    # The naming convention is _mirror_<component_name>. Only register
    # handlers for components that need non-trivial transforms.
    # Components without a handler are left unchanged.

    def _mirror_base_lin_vel(self, v: torch.Tensor) -> torch.Tensor:
        return mirror_xz_y_negate_(v)

    def _mirror_base_ang_vel(self, w: torch.Tensor) -> torch.Tensor:
        return mirror_xz_ang_vel_(w)

    def _mirror_projected_gravity(self, g: torch.Tensor) -> torch.Tensor:
        return mirror_xz_y_negate_(g)

    def _mirror_dofs_pos(self, q: torch.Tensor) -> torch.Tensor:
        return mirror_xz_joint(q, self.joint_index_map, self.sign_flip_mask)

    def _mirror_dofs_vel(self, dq: torch.Tensor) -> torch.Tensor:
        return mirror_xz_joint(dq, self.joint_index_map, self.sign_flip_mask)

    def _mirror_last_action(self, a: torch.Tensor) -> torch.Tensor:
        return mirror_xz_joint(a, self.joint_index_map, self.sign_flip_mask)

    def _mirror_command_lin_vel(self, cmd: torch.Tensor) -> torch.Tensor:
        """[vx_cmd, vy_cmd] -> [vx_cmd, -vy_cmd]."""
        cmd[:, 1] = -cmd[:, 1]
        return cmd

    def _mirror_command_ang_vel(self, cmd: torch.Tensor) -> torch.Tensor:
        """[wz_cmd] -> [-wz_cmd]."""
        return -cmd

    def _mirror_sin_phase(self, sp: torch.Tensor) -> torch.Tensor:
        """Swap left/right foot phases: [sin_left, sin_right] -> [sin_right, sin_left]."""
        if sp.shape[1] >= 2:
            sp[:, [0, 1]] = sp[:, [1, 0]]
        return sp

    def _mirror_cos_phase(self, cp: torch.Tensor) -> torch.Tensor:
        """Swap left/right foot phases: [cos_left, cos_right] -> [cos_right, cos_left]."""
        if cp.shape[1] >= 2:
            cp[:, [0, 1]] = cp[:, [1, 0]]
        return cp
