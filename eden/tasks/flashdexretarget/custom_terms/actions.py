"""FlashDexRetarget's actions: the residual hand action and the object assist wrench."""

from __future__ import annotations

import math
from typing import ClassVar

import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.tasks.flashdexretarget.custom_terms.common import HandTermMixin
from eden.utils.geom import axis_angle_from_quat, inv_quat, quat_mul, quat_to_R
from eden.utils.string import term_subject


@ACTION_TERM_REGISTRY.register()
class FDRHandAction(ImplicitPDController):
    """FlashDexRetarget's residual hand action: per joint, ``target = reference + a * scale`` (the forearm joints) or
    ``a`` in ``[-1, 1]`` mapped onto the joint range (the fingers), then clamped to the range.

    The action is clipped to ``[-1, 1]`` and, per joint group, low-pass filtered (an EMA with
    ``alpha = 2 pi fc dt / (1 + 2 pi fc dt)``; the first step after a reset passes through). The reference is the
    motion manager's current frame of the controlled entity (``reference_source`` must be ``MOTION_REF``).

    Parameters
    ----------
    residual_scale : dict[str, float]
        DOF-name pattern -> residual scale; DOFs no pattern matches are absolute (mapped onto their range).
    cutoff_hz : dict[str, float]
        DOF-name pattern -> low-pass cutoff (Hz); DOFs no pattern matches are not filtered.
    """

    residual_scale: dict[str, float] = {}
    cutoff_hz: dict[str, float] = {}

    def build(self) -> None:
        super().build()
        from eden.constants import ReferenceSource

        if self.reference_source != ReferenceSource.MOTION_REF:
            raise ValueError(
                f"{term_subject(self)} needs reference_source=ReferenceSource.MOTION_REF (its residual base)."
            )
        from eden.utils.string import name_matches

        def per_dof(table: dict[str, float], fill: float) -> torch.Tensor:
            out = torch.full((len(self.dofs_name),), fill, device=self.device)
            for pattern, value in table.items():
                cols = [i for i, n in enumerate(self.dofs_name) if name_matches(pattern, n)]
                if not cols:
                    raise ValueError(f"{term_subject(self)}: pattern {pattern!r} matches none of {self.dofs_name}.")
                out[cols] = value
            return out

        nan = float("nan")
        scale = per_dof(self.residual_scale, nan)
        self._residual = ~torch.isnan(scale)
        self._res_scale = torch.nan_to_num(scale, nan=0.0)
        dt = self._env.dt
        fc = per_dof(self.cutoff_hz, nan)
        w = 2.0 * math.pi * fc * dt
        self._alpha = torch.where(torch.isnan(fc), torch.ones_like(fc), w / (1.0 + w))
        limits = self.entity.get_dofs_limit(dofs_idx_local=self.genesis_dofs_idx)
        if limits[0].dim() > 1:  # per-env limits: FlashDexRetarget's are one table
            limits = (limits[0][0], limits[1][0])
        self._lower, self._upper = limits[0].to(self.device), limits[1].to(self.device)
        self._filtered = torch.zeros(self.num_envs, len(self.dofs_name), device=self.device)
        self._fresh = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.reference_used = torch.zeros(self.num_envs, len(self.dofs_name), device=self.device)

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        a = actions.clamp(-1.0, 1.0)
        self._filtered = torch.where(self._fresh, a, self._filtered)
        self._fresh[:] = False
        self._filtered = self._alpha * a + (1.0 - self._alpha) * self._filtered
        self.reference_used = self._motion_ref_offset()
        absolute = self._lower + (self._filtered + 1.0) * 0.5 * (self._upper - self._lower)
        target = torch.where(self._residual, self.reference_used + self._filtered * self._res_scale, absolute)
        self._processed_action = torch.clamp(target, self._lower, self._upper)

    def reset(self, envs_idx=None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        self._fresh[envs_idx] = True


@ACTION_TERM_REGISTRY.register()
class FDRObjectAssist(HandTermMixin, ActionTerm):
    """FlashDexRetarget's object assist: a world-frame PD wrench pulling each object toward its reference pose,
    re-applied every physics step (Genesis clears applied wrenches after each one), gravity left on.

    ``force = kp (p_ref - p) + kv (v_ref - v)``; ``torque = I_world (kp_rot axis_angle(q_ref q^-1) + kv_rot (w_ref -
    w))`` with ``kp_rot = kp * rot_stiffness_mult`` and ``kv_rot = kv * sqrt(rot_stiffness_mult)``; both clamped to
    ``+-force_range`` per axis. Per-env gains are set by :class:`FDRAssistCurriculum` (zero until then).

    Parameters
    ----------
    rot_stiffness_mult : float
        Rotational stiffness relative to the translational (FlashDexRetarget: 96).
    """

    rot_stiffness_mult: float = 96.0
    _acts_without_action_: ClassVar[bool] = True

    def build(self) -> None:
        import genesis as gs

        ActionTerm.build(self)  # no DOFs and no action columns: the wrench is the whole effect
        self._build_state()  # the inertia only: no SDF grids or keypoints
        self._com_frame = gs.link_ref_frame.link_COM
        self.kp = torch.zeros(self.num_envs, 1, device=self.device)
        self.kv = torch.zeros(self.num_envs, 1, device=self.device)
        self.force_range: float | None = None
        # One wrench per object entity: a shared object is pushed once, through its first side.
        self._entity_side = {}
        for i, s in enumerate(self.sides):
            self._entity_side.setdefault(self.objects[s], i)
        self._links = {
            name: torch.zeros(1, dtype=torch.long, device=self.device) + self._env.entities[name].link_start
            for name in self._entity_side
        }

    def compute(self, actions: torch.Tensor) -> None:
        pass

    def set_gains(self, kp: torch.Tensor, kv: torch.Tensor, force_range: float | None) -> None:
        """Per-env translational gains ``(N,)``; the rotational ones follow by :attr:`rot_stiffness_mult`."""
        self.kp[:, 0] = kp
        self.kv[:, 0] = kv
        self.force_range = force_range

    def apply_actions(self) -> None:
        st = self.state
        pos, quat = st.object_pose()
        lin, ang = st.object_vel()
        ref = st.ref_object(0)
        kp_rot, kv_rot = self.kp * self.rot_stiffness_mult, self.kv * math.sqrt(self.rot_stiffness_mult)
        for name, i in self._entity_side.items():
            force = self.kp * (ref["pos"][:, i] - pos[:, i]) + self.kv * (ref["lin_vel"][:, i] - lin[:, i])
            err = axis_angle_from_quat(quat_mul(ref["quat"][:, i], inv_quat(quat[:, i])))
            R = quat_to_R(quat[:, i])
            I_world = R @ st.inertia(i) @ R.transpose(-1, -2)
            accel = kp_rot * err + kv_rot * (ref["ang_vel"][:, i] - ang[:, i])
            torque = torch.einsum("nij,nj->ni", I_world, accel)
            if self.force_range is not None:
                force = force.clamp(-self.force_range, self.force_range)
                torque = torque.clamp(-self.force_range, self.force_range)
            self._env.rigid_solver.apply_links_external_wrench(
                force=force[:, None],
                torque=torque[:, None],
                links_idx=self._links[name],
                ref=self._com_frame,
                local=False,
            )
