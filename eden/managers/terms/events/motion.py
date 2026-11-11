"""Motion-reference-driven event terms (RESET-mode initial-state seeding)."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from genesis.utils.geom import inv_transform_by_quat

from eden.constants import EventMode
from eden.entities.rigid import StagedStateWrite
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.utils.geom import quat_from_euler_xyz, quat_mul
from eden.utils.misc import env_indices, is_env_mask, sanitize_envs_idx
from eden.utils.sample import sample_uniform
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.utils.motion_reference.motion_data import MotionReferenceData, MotionReferenceState


def _stage_free_joint(
    write: StagedStateWrite,
    entity: RigidEntity,
    dofs_idx_local,
    qs_idx_local,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    dofs_pos: torch.Tensor | None = None,
    dofs_vel: torch.Tensor | None = None,
) -> None:
    """Stage a free base, and a robot's actuated DOFs after it, from a fully-resolved reference state.

    The one home of the free-joint velocity convention, shared by every RSI flavour's reset and
    teleport. The free joint stores its angular DOF velocity in the base BODY frame (MuJoCo's qvel
    convention), but the reference's is world-frame. It is rotated world->body by ``base_quat``, the
    quaternion written to qpos, so the body frame is the reset pose's. Linear DOFs stay world-frame.
    """
    base_ang_vel_b = inv_transform_by_quat(base_ang_vel, base_quat)
    joints_vel = [] if dofs_vel is None else [dofs_vel]
    joints_pos = [] if dofs_pos is None else [dofs_pos]
    write.set_dofs_vel(entity, torch.cat([base_lin_vel, base_ang_vel_b, *joints_vel], dim=1), dofs_idx_local)
    write.set_qpos(entity, torch.cat([base_pos, base_quat, *joints_pos], dim=1), qs_idx_local)


class ReferenceStateInitializationBase(EventTerm):
    """Shared base for the motion-reference RSI (reference-state-initialization) events.

    Two concrete flavours seed initial state from the motion reference at RESET:
    :class:`ReferenceStateInitialization` (the robot — a floating *or* immobile base)
    and :class:`ObjectReferenceStateInitialization` (a free rigid object). This base
    exists so the motion-end resync path can find *every* flavour by ``isinstance``
    (not by matching one exact class name) and ask each to seed itself onto a resampled
    reference via :meth:`teleport` — so the branch knowledge lives in the term, in one
    place, instead of being re-derived in :class:`~eden.managers.motion.resync.MotionEndResync`.

    A flavour does not write the simulator itself: it stages its entity's writes
    (:meth:`_stage_reset`, :meth:`_stage_teleport`) into a shared
    :class:`~eden.entities.rigid.StagedStateWrite`, and :meth:`compute_batch` /
    :meth:`teleport_batch` apply every term's writes at once. The event manager fires
    adjacent RSI terms through one :meth:`compute_batch`, so a reset seeding N entities
    gathers the reference once and pays one Genesis write per kind and one forward-kinematics
    pass, not N of each.
    """

    mode = EventMode.RESET
    priority: int = 100
    #: Cap on the speed at which the contact solver may push the entity's base out of an overlap it was seeded into,
    #: m/s, or ``None`` for no cap — PhysX's ``max_depenetration_velocity`` (Weave uses 1.0). A reference pose rarely
    #: clears the collision geometry (feet in the floor, fingers in the object), and Genesis resolves the overlap at
    #: full stiffness, which can fling a light object several m/s. For ``depenetration_window_s`` after each seed
    #: (reset or motion-end resync; PhysX caps every contact, always), each sim step's push — what the solver added to
    #: the base's linear velocity beyond free fall, ``v_after - (v_before + g * sim_dt)`` — is scaled down just enough
    #: that the base's velocity stays within this norm of its seeded velocity. So pushes over several steps of a lasting
    #: overlap cannot stack past the cap, while gravity, and contact that slows the base back towards its seed (a
    #: floor holding a resting body up), pass untouched. Exact for a free object; an articulated base's joint-driven
    #: contact pushes count too. Angular velocity is left alone. Floating/free bases only.
    max_depenetration_velocity: float | None = None
    #: How long after a seed the cap stays active, in seconds (a 5 cm overlap cleared at 1 m/s takes 0.05).
    depenetration_window_s: float = 0.1

    #: The entity the cap clamps, or ``None`` when the cap is off.
    _depen_entity: RigidEntity | None = None

    @classmethod
    def compute_batch(
        cls, terms: Sequence[ReferenceStateInitializationBase], envs_idx: slice | torch.Tensor | None
    ) -> None:
        """Seed every term's entity from the init reference state, in one write (see the class docstring).

        The terms draw their perturbation noise in order, as they would one by one, and write disjoint
        state, so the result is what firing them one after another leaves.
        """
        env = terms[0]._env
        if envs_idx is None or isinstance(envs_idx, slice):
            envs_idx = sanitize_envs_idx(envs_idx, env.num_envs, prefer_slice=False)
        # The reference is gathered for the selected rows, through int ids: a sync per op through a mask and
        # none through ids. The auto-reset's mask has its ids taken once for the whole reset, so this
        # costs nothing. The write itself takes the mask, which Genesis applies on the device.
        envs_ids = env_indices(envs_idx)
        init_states = env.motion_manager.get_reset_reference_state(envs_ids, [term.entity_name for term in terms])
        write = StagedStateWrite()
        for term in terms:
            term._stage_reset(write, envs_ids, init_states[term.entity_name])
        write.apply(envs_idx)
        for term in terms:
            term._arm_depenetration(envs_ids)

    def compute(self, envs_idx: slice | torch.Tensor | None = None):
        type(self).compute_batch([self], envs_idx)

    @classmethod
    def teleport_batch(
        cls,
        terms: Sequence[ReferenceStateInitializationBase],
        wrap_idx: torch.Tensor,
        refs: Mapping[str, MotionReferenceData],
    ) -> None:
        """:meth:`teleport` every term's entity onto ``refs[entity_name]``, in one write.

        ``wrap_idx`` is int env ids, or a ``(num_envs,)`` bool mask: then every env's frame 0 is staged
        and the mask picks which are written, which Genesis does on the device — no ``nonzero()``, so a
        mask that selects no env costs no GPU->CPU sync either (the motion-end resync calls this every step).
        """
        rows = slice(None) if is_env_mask(wrap_idx) else wrap_idx
        write = StagedStateWrite()
        for term in terms:
            term._stage_teleport(write, rows, refs[term.entity_name])
        write.apply(wrap_idx)
        for term in terms:
            term._arm_depenetration(wrap_idx)

    def teleport(self, wrap_idx: torch.Tensor, ref: MotionReferenceData) -> None:
        """Snap ``wrap_idx`` envs onto ``ref`` verbatim (no reset perturbation).

        The motion-end-resync counterpart of :meth:`compute`, staged and written the same way,
        but it reads the already-filled ``ref`` (this entity's ``MotionReferenceData``) at
        frame 0 instead of recomputing the init reference state, and adds no noise (a wrap is
        mid-episode, so the entity snaps exactly onto the demo). ``wrap_idx`` is an int tensor
        of the env ids that wrapped, or the ``(num_envs,)`` bool mask of them.
        """
        type(self).teleport_batch([self], wrap_idx, {self.entity_name: ref})

    def _build_depenetration(self, entity: RigidEntity, base_dofs_idx: Sequence[int]) -> None:
        """Allocate the post-seed velocity cap and hook it around the sim steps (no-op when disabled)."""
        if self.max_depenetration_velocity is None:
            return
        if self.max_depenetration_velocity <= 0.0 or self.depenetration_window_s <= 0.0:
            raise ValueError(
                f"{term_subject(self)}: max_depenetration_velocity="
                f"{self.max_depenetration_velocity} and depenetration_window_s={self.depenetration_window_s} must both be "
                f"> 0; pass max_depenetration_velocity=None to disable the cap."
            )
        self._depen_entity = entity
        self._depen_lin_idx = torch.as_tensor(list(base_dofs_idx)[:3], dtype=torch.long, device=self.device)
        self._depen_dt = float(self._env.env_options.sim_dt)
        self._depen_steps = max(1, round(self.depenetration_window_s / self._depen_dt))
        self._depen_left = torch.zeros(self._env.num_envs, dtype=torch.long, device=self.device)
        self._depen_vel_before = torch.zeros(self._env.num_envs, 3, device=self.device)
        self._depen_vel_seed = torch.zeros(self._env.num_envs, 3, device=self.device)
        self._env.register_substep_callback(self._record_depenetration_vel, before=True)
        self._env.register_substep_callback(self._cap_depenetration)

    def _arm_depenetration(self, envs_idx: torch.Tensor) -> None:
        """Start the cap window for the envs just written: int env ids or a ``(num_envs,)`` bool mask (no sync)."""
        if self._depen_entity is None:
            return
        if is_env_mask(envs_idx):
            # Not ``self._depen_left[envs_idx] = ...``: indexing by a bool mask is a host sync.
            self._depen_left.masked_fill_(envs_idx, self._depen_steps)
        else:
            self._depen_left[envs_idx] = self._depen_steps

    def _record_depenetration_vel(self) -> None:
        """Before a sim step: keep the base linear velocity it starts from, and on a window's first step the seed's."""
        self._depen_vel_before.copy_(self._depen_entity.get_dofs_vel(dofs_idx_local=self._depen_lin_idx))
        # Nothing moves the base between the seed's write and the next sim step, so this is the seeded velocity.
        first = (self._depen_left == self._depen_steps).unsqueeze(-1)
        torch.where(first, self._depen_vel_before, self._depen_vel_seed, out=self._depen_vel_seed)

    def _cap_depenetration(self) -> None:
        """After a sim step: scale its push so the base stays within the cap of its seeded velocity (in-window envs).

        With ``d = free - seed`` (where free fall alone leaves the base, relative to its seed) and ``p`` the push, the
        largest ``s`` in [0, 1] with ``|d + s p| <= max(cap, |d|)`` is taken: the push may not carry the base further
        from its seed than the cap, nor further than gravity alone already has.

        Only the in-window rows are written, through the bool mask, which Genesis applies on the device — no sync,
        and every other env's velocity is left exactly as the solver produced it. (Under hibernation the setter turns
        the mask into ids first, one sync per step.)
        """
        active = self._depen_left > 0
        vel = self._depen_entity.get_dofs_vel(dofs_idx_local=self._depen_lin_idx)
        free = self._depen_vel_before + self._env.rigid_solver.get_gravity() * self._depen_dt
        push = vel - free
        drift = free - self._depen_vel_seed
        dd, dp, pp = (drift * drift).sum(-1), (drift * push).sum(-1), (push * push).sum(-1)
        room = torch.clamp(dd, min=self.max_depenetration_velocity**2) - dd  # >= 0
        # The larger root of pp s^2 + 2 dp s + (dd - R^2) = 0; s = 0 always satisfies the bound since |d| <= R.
        root = (torch.sqrt(dp * dp + pp * room) - dp) / pp.clamp_min(1e-12)
        scale = torch.where(pp > 1e-12, root.clamp(0.0, 1.0), torch.ones_like(pp)).unsqueeze(-1)
        self._depen_entity.set_dofs_vel(free + push * scale, dofs_idx_local=self._depen_lin_idx, envs_idx=active)
        self._depen_left.sub_(active.long())

    def _stage_reset(self, write: StagedStateWrite, envs_idx: torch.Tensor, init_state: MotionReferenceState) -> None:
        """Stage this term's entity onto ``init_state`` (``envs_idx`` rows), perturbed as configured."""
        raise NotImplementedError

    def _stage_teleport(
        self, write: StagedStateWrite, wrap_idx: slice | torch.Tensor, ref: MotionReferenceData
    ) -> None:
        """Stage this term's entity onto frame 0 of ``ref`` for the ``wrap_idx`` rows, with no perturbation.

        ``wrap_idx`` selects the rows of ``ref`` to stage: the wrapped envs' int ids, or ``slice(None)`` —
        every env's row — when the write goes through the wrap mask (:meth:`teleport_batch`).
        """
        raise NotImplementedError


@EVENT_TERM_REGISTRY.register()
class ReferenceStateInitialization(ReferenceStateInitializationBase):
    """On reset, initialise robot state from the motion reference (RSI).

    Reads ``env.motion_manager.get_reset_reference_state(envs_idx)[entity_name]``
    and writes it onto the entity's qpos/qvel, with optional per-DoF and
    base-pose perturbations. ``overall_noise_scale`` multiplies all configured
    perturbation ranges (pose, velocity, dofs_position) so a curriculum can
    anneal init noise from a single knob.
    """

    entity_name: str = "robot"
    # None auto-detects the free base joint by joint TYPE (works for MJCF and URDF robots
    # alike); name it only to disambiguate a multi-free-joint entity.
    base_joint_name: str | None = None
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    dofs_position_range: tuple[float, float] = (-0.1, 0.1)
    overall_noise_scale: float = 1.0

    #: Columns of ``soft_dofs_pos_limits`` matching the reference's DOF order, or ``None``
    #: when the two already agree. Only the immobile-base path can disagree: it writes
    #: through name-resolved indices, so its reference columns follow the MOTION order,
    #: whereas ``soft_dofs_pos_limits`` rows always follow the entity's ``dofs_name`` order.
    #: The floating-base path writes via ``set_qpos``, which consumes ``dofs_name`` order.
    _soft_limits_cols: torch.Tensor | None = None

    def build(self):
        self.robot = self.resolve_entity(self.entity_name)
        # ``has_immobile_base`` covers both a fixed-base entity and one welded onto another
        # (a composite's dexterous hand): Genesis ``attach`` removes the child's root joint,
        # so there is no base to seed there either.
        if self.robot.has_immobile_base:
            if self.max_depenetration_velocity is not None:
                raise ValueError(
                    f"{term_subject(self)} sets max_depenetration_velocity, but "
                    f"the entity has no base to cap (it is fixed or welded to another entity). Drop the option here; "
                    f"set it on the RSI term of the entity carrying the free base."
                )
            # No floating base joint to seed — only the actuated DOFs are
            # written (see the immobile-base branch in ``_stage_reset``). The write
            # goes through name-resolved Genesis indices in the MOTION dof
            # order, so no qpos-declaration-order assumption is needed.
            from eden.utils.motion_reference.lookup import get_robot_dofs_idx, untracked_entity_error

            motion_dofs_name = self._env.motion_manager.get_target_dofs_name(self.entity_name)
            if motion_dofs_name is None:
                raise untracked_entity_error(self._env, self.entity_name)
            if not motion_dofs_name:
                raise ValueError(
                    f"{term_subject(self)} (immobile base) requires the motion reference to expose DOFs for it."
                )
            self._motion_dofs_idx = get_robot_dofs_idx(self._env, self.entity_name, motion_dofs_name)
            # Host copies for the staged write, which reads its indices in Python every reset.
            self._motion_dofs_host = self._motion_dofs_idx.tolist()
            # A ball joint's three DOFs make one quaternion, so the reference covers all of them or none.
            StagedStateWrite.check_dofs_pos(self.robot, self._motion_dofs_host)
            self._entity_dofs_host = range(self.robot.n_dofs)
            # See ``_soft_limits_cols``: gather the rows matching the motion DOF order so the
            # clip in ``_dofs_pos_with_noise`` can't silently apply another joint's range.
            self._soft_limits_cols = self.robot.find_named_dofs(motion_dofs_name).columns
        else:
            from eden.utils.motion_reference.lookup import assert_motion_field_present

            # The base is seeded from the reference's base pose, so an untracked entity, or a sequence
            # without one, is a config error: raise it here rather than at the first reset.
            assert_motion_field_present(self._env, self.entity_name, "base_pos_w")
            assert_motion_field_present(self._env, self.entity_name, "base_quat_w")
            base_dofs_index = self.robot.find_free_base_dofs_idx_local(self.base_joint_name)
            # Host lists for the staged write, which reads its indices in Python every reset: the free base's
            # six DOFs then the actuated ones, and the qs ``set_qpos`` writes by default (dofs_name order).
            self._base_and_dofs_host = [*base_dofs_index, *self.robot.dofs_idx_local.tolist()]
            self._qs_host = self.robot.qs_idx_local.tolist()
            self._build_depenetration(self.robot, base_dofs_index)

        range_list = [self.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_ranges = torch.tensor(range_list, device=self.device)
        range_list = [self.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_ranges = torch.tensor(range_list, device=self.device)

    def _dofs_pos_with_noise(self, dofs_pos: torch.Tensor, envs_idx: torch.Tensor) -> torch.Tensor:
        """Apply the configured DOF-position perturbation and soft-limit clip.

        ``soft_dofs_pos_limits`` has shape ``(1, n_dofs, 2)`` when every env shares one row (the
        common case) and ``(num_envs, n_dofs, 2)`` when rows differ per env (a heterogeneous entity
        whose variants have different limits). Only slice in the per-env case; otherwise let
        broadcasting handle the partial reset.
        """
        dofs_pos = dofs_pos + sample_uniform(
            lower=self.dofs_position_range[0] * self.overall_noise_scale,
            upper=self.dofs_position_range[1] * self.overall_noise_scale,
            size=dofs_pos.shape,
            device=dofs_pos.device,
        )
        soft_limits = self.robot.soft_dofs_pos_limits
        if soft_limits.shape[0] != 1:
            soft_limits = soft_limits[envs_idx]
        if self._soft_limits_cols is not None:
            soft_limits = soft_limits[:, self._soft_limits_cols]
        return torch.clip(dofs_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

    def _stage_dofs_only(self, write: StagedStateWrite, dofs_pos: torch.Tensor, dofs_vel: torch.Tensor) -> None:
        """Stage only the actuated DOFs (an immobile/welded base has no base to seed).

        The name-resolved ``_motion_dofs_idx`` puts value ``i`` on the motion reference's
        ``i``-th DOF regardless of qpos declaration order.
        """
        # The entity's every DOF: zero, then the reference's — the state ``set_dofs_pos(zero_velocity=True)``
        # followed by ``set_dofs_vel`` would leave.
        velocity = torch.zeros((dofs_vel.shape[0], self.robot.n_dofs), dtype=dofs_vel.dtype, device=dofs_vel.device)
        velocity[:, self._motion_dofs_idx] = dofs_vel
        write.set_dofs_pos(self.robot, dofs_pos, self._motion_dofs_host)
        write.set_dofs_vel(self.robot, velocity, self._entity_dofs_host)

    def _stage_reset(self, write: StagedStateWrite, envs_idx: torch.Tensor, init_state: MotionReferenceState) -> None:
        # Must match the branch ``build`` took — that is what resolved the host indices.
        if self.robot.has_immobile_base:
            # No base to perturb; seed the DOFs (with noise) only.
            self._stage_dofs_only(write, self._dofs_pos_with_noise(init_state.dofs_pos, envs_idx), init_state.dofs_vel)
            return

        scale = self.overall_noise_scale

        # Pose perturbation. Sample for all envs, then take the ``envs_idx`` rows (int ids), the
        # rows ``init_state`` was gathered for.
        rand_pose = sample_uniform(
            self.pose_ranges[:, 0] * scale,
            self.pose_ranges[:, 1] * scale,
            (self._env.num_envs, 6),
            device=self.device,
        )[envs_idx]
        root_pos = init_state.base_pos_w + rand_pose[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_pose[:, 3], rand_pose[:, 4], rand_pose[:, 5])
        root_ori = quat_mul(orientations_delta, init_state.base_quat_w)

        # Velocity perturbation
        rand_vel = sample_uniform(
            self.velocity_ranges[:, 0] * scale,
            self.velocity_ranges[:, 1] * scale,
            (self._env.num_envs, 6),
            device=self.device,
        )[envs_idx]
        root_lin_vel = init_state.base_lin_vel_w + rand_vel[:, :3]
        root_ang_vel = init_state.base_ang_vel_w + rand_vel[:, 3:]

        dofs_pos = self._dofs_pos_with_noise(init_state.dofs_pos, envs_idx)

        _stage_free_joint(
            write,
            self.robot,
            self._base_and_dofs_host,
            self._qs_host,
            root_pos,
            root_ori,
            root_lin_vel,
            root_ang_vel,
            dofs_pos=dofs_pos,
            dofs_vel=init_state.dofs_vel,
        )

    def _stage_teleport(
        self, write: StagedStateWrite, wrap_idx: slice | torch.Tensor, ref: MotionReferenceData
    ) -> None:
        # Same branch split as ``_stage_reset``/``build``; frame 0 of the just-filled reference,
        # no perturbation. An immobile/welded child seeds its DOFs only (no N+6 base payload).
        if self.robot.has_immobile_base:
            self._stage_dofs_only(write, ref.dofs_pos[wrap_idx, 0], ref.dofs_vel[wrap_idx, 0])
            return
        _stage_free_joint(
            write,
            self.robot,
            self._base_and_dofs_host,
            self._qs_host,
            ref.base_pos_w[wrap_idx, 0],
            ref.base_quat_w[wrap_idx, 0],
            ref.base_lin_vel_w[wrap_idx, 0],
            ref.base_ang_vel_w[wrap_idx, 0],
            dofs_pos=ref.dofs_pos[wrap_idx, 0],
            dofs_vel=ref.dofs_vel[wrap_idx, 0],
        )


@EVENT_TERM_REGISTRY.register()
class ObjectReferenceStateInitialization(ReferenceStateInitializationBase):
    """On reset, set a free rigid object from the motion reference.

    The object counterpart of :class:`ReferenceStateInitialization` for
    manipulation mimic tasks: writes the reference base pose and base
    velocities of a DOF-less free-floating entity (e.g. the manipulated
    object of a hand-object clip). Optional uniform XY perturbation matches
    ConTrack's object domain randomization.
    """

    entity_name: str = "object"
    xy_range: float = 0.0

    def build(self):
        from eden.utils.motion_reference.lookup import assert_motion_field_present

        self.obj = self.resolve_entity(self.entity_name)
        if self.obj.has_immobile_base:
            reason = "is welded to another entity" if self.obj.is_attached else "is fixed-base"
            raise ValueError(
                f"{term_subject(self)} writes a base pose, so it requires a "
                f"free-floating entity; it {reason} and has no base DOFs."
            )
        assert_motion_field_present(self._env, self.entity_name, "base_pos_w")
        assert_motion_field_present(self._env, self.entity_name, "base_quat_w")
        # A free rigid body's 6 base DOFs and 7 base qs are its only ones; index them directly.
        self._base_dofs_host = range(6)
        self._base_qs_host = range(7)
        self._build_depenetration(self.obj, self._base_dofs_host)

    def _stage_reset(self, write: StagedStateWrite, envs_idx: torch.Tensor, init_state: MotionReferenceState) -> None:
        pos = init_state.base_pos_w
        if self.xy_range > 0.0:
            # Only xy is perturbed, so the reset quat is the reference quat verbatim.
            pos = pos.clone()
            pos[:, :2] += sample_uniform(-self.xy_range, self.xy_range, (pos.shape[0], 2), device=pos.device)
        _stage_free_joint(
            write,
            self.obj,
            self._base_dofs_host,
            self._base_qs_host,
            pos,
            init_state.base_quat_w,
            init_state.base_lin_vel_w,
            init_state.base_ang_vel_w,
        )

    def _stage_teleport(
        self, write: StagedStateWrite, wrap_idx: slice | torch.Tensor, ref: MotionReferenceData
    ) -> None:
        # Frame 0 of the just-filled reference, no xy perturbation (a wrap is mid-episode).
        _stage_free_joint(
            write,
            self.obj,
            self._base_dofs_host,
            self._base_qs_host,
            ref.base_pos_w[wrap_idx, 0],
            ref.base_quat_w[wrap_idx, 0],
            ref.base_lin_vel_w[wrap_idx, 0],
            ref.base_ang_vel_w[wrap_idx, 0],
        )
