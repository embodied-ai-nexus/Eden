"""Reset and domain-randomization events for in-hand cube reorientation.

Port of ``wuji-mjlab``'s ``event_impl/`` package. Three of upstream's sixteen startup
randomizations have no per-env Genesis equivalent and are not ported -- per-env collision
geom scaling, per-env contact solver parameters, and per-env link inertia -- and the cube
COM offset is deliberately left off. See the task config for what stands in their place. The
rest map onto Eden's built-in DR events (``RandomizeFrictionRatio``, ``RandomizeKpKdGains``,
``RandomizeLinkMassScale``, ``RandomizeDofsPosOffset``); what remains here is the handful
Eden has no term for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import transform_by_quat

from eden.constants import EventMode
from eden.entities.dofs_default_mixin import DofSelection
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.tasks.reorient.constants import PALM_LINK, REORIENT_CUBE_IN_PALM_POS
from eden.utils.geom import quat_from_euler_xyz, random_quat_uniform
from eden.utils.misc import as_genesis_index, assign_over_envs, is_env_mask, sanitize_envs_idx
from eden.utils.sample import sample_uniform
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.events import EventTermOptions


@EVENT_TERM_REGISTRY.register()
class ResetObjectInPalm(EventTerm):
    """Place the object in the palm with a uniformly random orientation.

    The spawn is composed through the palm's *live* pose rather than written as a world
    constant (which is what upstream does), so it stays correct under both the per-env
    scene offset and the randomized wrist pitch -- and it keeps working if the hand's
    mounting pose is ever changed.

    Parameters
    ----------
    hand_entity_name:
        Scene entity name of the hand.
    object_entity_name:
        Scene entity name of the object to place.
    palm_link:
        Link pattern selecting the palm.
    pos_in_palm:
        Spawn position in the palm frame.
    pos_noise:
        Half-width [m] of the uniform jitter added to the spawn position.
    """

    mode: EventMode = EventMode.RESET
    hand_entity_name: str = "robot"
    object_entity_name: str = "cube"
    palm_link: str = PALM_LINK
    pos_in_palm: tuple[float, float, float] = REORIENT_CUBE_IN_PALM_POS
    pos_noise: float = 0.01

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self._palm_idx: torch.Tensor | None = None
        self._pos_in_palm = torch.tensor(self.pos_in_palm, device=self.device)

    def build(self) -> None:
        hand = self._env.entities[self.hand_entity_name]
        self._env.entities[self.object_entity_name]
        _, palm_idx = hand.find_named_links_idx_local(self.palm_link, preserve_order=True)
        if len(palm_idx) != 1:
            raise ValueError(
                f"ResetObjectInPalm: palm_link={self.palm_link!r} matched {len(palm_idx)} links on entity "
                f"'{self.hand_entity_name}'; expected exactly one."
            )
        self._palm_idx = torch.as_tensor(palm_idx, dtype=torch.long, device=self.device)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        hand = self.resolve_entity(self.hand_entity_name)
        obj = self.resolve_entity(self.object_entity_name)

        palm_pos = hand.get_links_pos(ls_idx_local=self._palm_idx, envs_idx=envs_idx).squeeze(1)
        palm_quat = hand.get_links_quat(ls_idx_local=self._palm_idx, envs_idx=envs_idx).squeeze(1)
        pos = palm_pos + transform_by_quat(self._pos_in_palm.expand_as(palm_pos), palm_quat)
        if self.pos_noise > 0.0:
            pos = pos + sample_uniform(-self.pos_noise, self.pos_noise, pos.shape, device=self.device)

        # ``relative=False`` on both setters: these are solver/world-frame values read back
        # from the palm in the same frame. (``set_quat`` defaults to relative=True.)
        obj.set_pos(pos, envs_idx=envs_idx)
        obj.set_quat(random_quat_uniform(pos.shape[0], device=self.device), envs_idx=envs_idx, relative=False)
        obj.zero_all_dofs_velocity(envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class ResetDofsByPatternOffset(EventTerm):
    """Reset DOFs to the default pose plus a per-pattern uniform offset.

    ``events.reset_dofs_by_offset`` takes a single range for the whole entity; this task
    needs the flexion joints started more extended than the abduction joints, so it needs
    one range per joint group. Offsets are clamped to the soft DOF limits.

    Parameters
    ----------
    entity_name:
        Scene entity name of the entity to reset.
    dofs_pos_range:
        Map of DOF-name pattern to ``(low, high)`` offset range [rad]. The patterns must
        *partition* the entity's DOFs: a DOF matched by two patterns raises at build (without
        that check the duplicate index makes the later range silently win the write), and an
        unmatched DOF simply gets no jitter.
    """

    mode: EventMode = EventMode.RESET
    entity_name: str = "robot"
    dofs_pos_range: dict[str, tuple[float, float]] = {}

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        if not self.dofs_pos_range:
            raise ValueError(f"{term_subject(self)}: dofs_pos_range is empty; use events.reset_dofs_by_offset.")
        # Resolve each pattern once, into both spaces, with (low, high) per DOF beside it.
        selections: list[DofSelection] = []
        low: list[float] = []
        high: list[float] = []
        seen: dict[str, str] = {}
        for pattern, (lo, hi) in self.dofs_pos_range.items():
            matched = entity.find_named_dofs(pattern)
            for name in dict.fromkeys(matched.names):  # a ball joint names one joint per DOF
                if name in seen:
                    raise ValueError(
                        f"{term_subject(self)}: DOF {name!r} is matched by both "
                        f"{seen[name]!r} and {pattern!r}; the patterns must partition the DOFs, or "
                        f"the later range silently wins the write."
                    )
                seen[name] = pattern
            selections.append(matched)
            low.extend([float(lo)] * len(matched))
            high.extend([float(hi)] * len(matched))
        # The patterns' selections end to end: columns read the property tensors, local indices go to the setter,
        # and both stay in the order ``low`` / ``high`` were built in.
        self._dofs = DofSelection(
            entity_name=entity.name,
            names=tuple(name for matched in selections for name in matched.names),
            columns=torch.cat([matched.columns for matched in selections]),
            dofs_idx_local=torch.cat([matched.dofs_idx_local for matched in selections]),
        )
        self._low = torch.tensor(low, device=self.device)
        self._high = torch.tensor(high, device=self.device)

    def _rows(self, prop: torch.Tensor, envs_idx: slice | torch.Tensor) -> torch.Tensor:
        """Select ``envs_idx`` from a DOF property that may or may not be per-env.

        With ``EnvOptions(batch_dofs_info=True)`` -- which this task needs for its per-env
        armature/damping randomization -- ``default_dofs_pos`` and ``soft_dofs_pos_limits``
        come back with a leading ``num_envs`` axis instead of a broadcast 1. On a *partial*
        reset that silently mismatches the selected rows, so the selection has to be applied
        to the property too rather than relying on broadcasting.
        """
        return prop[envs_idx] if prop.shape[0] == self._env.num_envs else prop

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        entity = self.resolve_entity(self.entity_name)
        default = self._rows(entity.default_dofs_pos, envs_idx)[:, self._dofs.columns]
        soft = self._rows(entity.soft_dofs_pos_limits, envs_idx)[:, self._dofs.columns, :]
        offset = sample_uniform(self._low, self._high, (self._env.num_envs, len(self._dofs)), device=self.device)[
            envs_idx
        ]
        target = (default + offset).clamp(soft[..., 0], soft[..., 1])
        entity.set_dofs_pos(target, dofs_idx_local=self._dofs.dofs_idx_local, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeGravityTilt(EventTerm):
    """Per-env random tilt of the gravity vector -- the fixed-base stand-in for wrist DR.

    Upstream randomizes the wrist pitch through ``reset_root_state_uniform``, which needs a
    free base. This hand is fixed-base *with collision geometry*, and Genesis refuses a
    per-env root orientation for exactly that case ("Impossible to set env-specific quat for
    fixed links with at least one geometry"). Tilting gravity instead is the physically
    equivalent perturbation: for a hand rigidly held in place, rotating the hand by ``R`` and
    rotating gravity by ``R^-1`` produce the same dynamics for the hand and the cube it
    holds.

    It matters more than it looks. With the hand fixed and every policy observation taken in
    the wrist-tag frame, the *only* thing a wrist rotation changes for the policy is which
    way gravity pulls the cube -- toward the fingertips, out of them, or across them. That is
    the whole content of the randomization, and it is what this term delivers.

    Two consequences of the substitution, both harmless here but worth knowing: the palm's
    world orientation no longer varies (so the critic's ``palm_rot_6d`` is constant per run
    rather than per env), and the tilt applies to the whole scene rather than to the hand
    alone -- which is fine, because the hand and the cube are the whole scene.

    Parameters
    ----------
    roll_range, pitch_range, yaw_range:
        Uniform tilt ranges [rad] about the world x/y/z axes, applied to gravity. A yaw
        rotation of gravity is a no-op (gravity is along z), so leave it at zero; it is
        accepted only so the ranges read the same way as an orientation randomization.
    gravity:
        The nominal gravity vector the tilt is applied to.
    """

    mode: EventMode = EventMode.RESET
    roll_range: tuple[float, float] = (0.0, 0.0)
    pitch_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self._gravity = torch.tensor(self.gravity, device=self.device)
        self._ranges = torch.tensor([self.roll_range, self.pitch_range, self.yaw_range], device=self.device)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        rpy = sample_uniform(self._ranges[:, 0], self._ranges[:, 1], (self._env.num_envs, 3), device=self.device)
        # Rotate gravity by the INVERSE of the hand rotation being emulated, so a positive
        # pitch here corresponds to pitching the hand by the same positive angle.
        tilt = quat_from_euler_xyz(-rpy[:, 0], -rpy[:, 1], -rpy[:, 2])
        gravity = transform_by_quat(self._gravity.expand(self._env.num_envs, 3), tilt)
        self._env.scene.sim.set_gravity(gravity[envs_idx], envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class ObjectVelocityDisturbance(EventTerm):
    """Kick the object with a random impulse on an interval, ramped by training progress.

    Upstream's ``apply_velocity_disturbance``. The kick magnitude ramps from ``min_speed``
    to ``max_speed`` over training (a linear schedule on the global step count, optionally
    scaled by an adaptive curriculum), and is suppressed for the first ``warmup_time_s`` of
    each episode so the cube is settled in the hand before it gets hit.

    The term owns the two buffers the critic reads (:attr:`force_dir`,
    :attr:`velocity_cache`) -- upstream keeps them in an env-level cache object; here they
    belong to the producer, which is also what gives them correct reset semantics.

    Parameters
    ----------
    object_entity_name:
        Scene entity name of the object to disturb.
    min_speed, max_speed:
        Impulse magnitude range [m/s] at the start and end of the ramp.
    warmup_time_s:
        Per-episode time before disturbances start.
    warmup_frac, rampup_frac:
        Fractions of ``total_steps`` at which the ramp starts and reaches full strength.
    total_steps:
        Expected total training steps (``max_iterations * num_steps_per_env``). Required --
        upstream reads it back from its PPO config at runtime, which couples the term to a
        specific training setup; passing it explicitly keeps the schedule inspectable.
    adaptive_curriculum_name:
        Optional curriculum term whose scalar value additionally scales the ramp, so
        disturbances only grow while the policy is actually surviving episodes.
    """

    mode: EventMode = EventMode.INTERVAL
    interval_range_s: tuple[float, float] | None = (0.6, 1.8)
    object_entity_name: str = "cube"
    min_speed: float = 0.05
    max_speed: float = 0.15
    warmup_time_s: float = 3.0
    warmup_frac: float = 0.05
    rampup_frac: float = 0.80
    total_steps: int = 0
    adaptive_curriculum_name: str = ""

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.force_dir = torch.zeros(self.num_envs, 3, device=self.device)
        self.velocity_cache = torch.zeros(self.num_envs, 6, device=self.device)

    def build(self) -> None:
        obj = self._env.entities[self.object_entity_name]
        base_dofs_idx = obj.find_free_base_dofs_idx_local()
        if len(base_dofs_idx) != 6:
            raise ValueError(
                f"ObjectVelocityDisturbance: '{self.object_entity_name}' has {len(base_dofs_idx)} free base "
                f"DOFs, expected 6 -- the object must be a free-floating body to be disturbed."
            )
        # Only the three linear DOFs; the impulse is purely translational.
        self._base_lin_dofs_idx = torch.as_tensor(base_dofs_idx[:3], dtype=torch.long, device=self.device)
        # Genesis applies a bool `envs_idx` itself, on the device, when handed every env's row and a `range` of
        # DOFs (`RigidEntity.setters_take_every_row`); selecting the rows would sync per read and write.
        self._write_every_row = obj.setters_take_every_row and isinstance(
            as_genesis_index(self._base_lin_dofs_idx), range
        )
        if self.total_steps <= 0:
            raise ValueError(
                "ObjectVelocityDisturbance: total_steps must be > 0 (set it to "
                "runner_options.max_iterations * runner_options.num_steps_per_env)."
            )
        # The curriculum manager is constructed before the event manager (``_manager_order``),
        # so a bad name raises at env construction.
        self._curriculum = None
        if self.adaptive_curriculum_name:
            if self._env.curriculum_manager is None:
                raise ValueError(
                    f"ObjectVelocityDisturbance: adaptive_curriculum_name="
                    f"{self.adaptive_curriculum_name!r} but the task declares no curriculum terms."
                )
            self._curriculum = self._env.curriculum_manager.get_term(self.adaptive_curriculum_name)

    def _ramp(self) -> float:
        """Linear 0 -> 1 over ``[warmup_frac, rampup_frac] * total_steps``."""
        step = float(self._env.common_step_counter)
        warmup = self.warmup_frac * self.total_steps
        rampup = self.rampup_frac * self.total_steps
        if step <= warmup:
            return 0.0
        if step >= rampup:
            return 1.0
        return (step - warmup) / max(rampup - warmup, 1.0)

    def _adaptive_scale(self) -> float:
        """The curriculum's difficulty scalar, or 1.0 when no curriculum is configured.

        Read off the term, not ``curriculum_manager.get_state()``: curricula are frozen whole at
        eval, so ``get_state()`` raises for that entire rollout; a ``None``-means-1.0 fallback
        would silently run every eval at *maximum* disturbance where training starts at
        ``min_scale``. The term's own level is correct in both modes.
        """
        if self._curriculum is None:
            return 1.0
        return float(self._curriculum.value)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        obj = self.resolve_entity(self.object_entity_name)

        self.force_dir[envs_idx] = 0.0
        self.velocity_cache[envs_idx] = 0.0

        # Suppress the kick for envs still inside their per-episode warmup, without an
        # index round-trip: sample for everyone, then zero the inactive rows.
        active = (self._env.episode_length_buf * self._env.dt >= self.warmup_time_s).unsqueeze(-1)

        curr_max_speed = self.min_speed + (self.max_speed - self.min_speed) * self._ramp() * self._adaptive_scale()
        direction = torch.randn(self._env.num_envs, 3, device=self.device)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        speed = sample_uniform(
            self.min_speed, max(self.min_speed, curr_max_speed), (self._env.num_envs, 1), device=self.device
        )
        velocity = torch.where(active, direction * speed, torch.zeros_like(direction))

        # Written through the interval tick's bool mask with `torch.where`, not gathered through it (no sync).
        assign_over_envs(self.force_dir, envs_idx, torch.where(active, direction, torch.zeros_like(direction)))
        assign_over_envs(self.velocity_cache, envs_idx, torch.cat([velocity, torch.zeros_like(velocity)], dim=-1))

        # Written through the free base joint's DOFs rather than a velocity setter (neither
        # the Eden wrapper nor Genesis' RigidEntity has one). Genesis stores a free joint's
        # three *linear* DOF velocities in the world frame, so a world-frame impulse goes
        # straight in with no frame conversion -- unlike the rotational DOFs, which are
        # body-frame (see events.PushByVelocity).
        rows = slice(None) if self._write_every_row and is_env_mask(envs_idx) else envs_idx
        lin_vel = obj.get_dofs_vel(dofs_idx_local=self._base_lin_dofs_idx, envs_idx=rows)
        obj.set_dofs_vel(lin_vel + velocity[rows], dofs_idx_local=self._base_lin_dofs_idx, envs_idx=envs_idx)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self.force_dir[envs_idx] = 0.0
        self.velocity_cache[envs_idx] = 0.0


class _PerEnvDofsPropertyDR(EventTerm):
    """Shared base for the two per-env DOF-property randomizations.

    Both scale the configured value by a log-uniform factor, once per env at startup. They
    need ``EnvOptions(batch_dofs_info=True)``: Genesis stores DOF properties once for the
    whole batch otherwise, and a per-env write is rejected -- so this is checked at build
    rather than surfacing on the first fire.

    Parameters
    ----------
    entity_name:
        Scene entity name of the entity to randomize.
    scale_range:
        Multiplicative range applied to the configured per-DOF value, sampled
        log-uniformly so that e.g. (0.5, 2.0) is symmetric in ratio.
    """

    mode: EventMode = EventMode.STARTUP
    entity_name: str = "robot"
    scale_range: tuple[float, float] = (0.5, 2.0)

    _default_attr: str = ""
    _setter: str = ""

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        if not entity.has_batched_dofs_info:
            raise ValueError(
                f"{term_subject(self)} writes per-env DOF properties and needs "
                f"EnvOptions(batch_dofs_info=True); Genesis rejects a per-env write otherwise."
            )
        if self.scale_range[0] <= 0.0:
            raise ValueError(f"{term_subject(self)}: scale_range must be positive, got {self.scale_range}.")
        # The configured ``default_dofs_*`` take its columns; the setter below takes its Genesis-local indices.
        self._dofs = entity.find_named_dofs()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        del envs_idx  # startup: always the whole batch.
        entity = self.resolve_entity(self.entity_name)
        default = getattr(entity, self._default_attr)[:, self._dofs.columns]
        log_lo, log_hi = (torch.tensor(self.scale_range, device=self.device)).log()
        scale = sample_uniform(log_lo, log_hi, (self._env.num_envs, len(self._dofs)), device=self.device).exp()
        getattr(entity, self._setter)(default * scale, dofs_idx_local=self._dofs.dofs_idx_local)


@EVENT_TERM_REGISTRY.register()
class RandomizeDofsArmature(_PerEnvDofsPropertyDR):
    """Per-env log-uniform scaling of DOF armature (rotor inertia)."""

    scale_range: tuple[float, float] = (0.75, 1.3)
    _default_attr: str = "default_dofs_armature"
    _setter: str = "set_dofs_armature"


@EVENT_TERM_REGISTRY.register()
class RandomizeDofsDamping(_PerEnvDofsPropertyDR):
    """Per-env log-uniform scaling of DOF damping.

    The Wuji URDF declares zero joint damping, so a *scale* would be a no-op. When every
    configured default is zero this term therefore draws an absolute damping uniformly from
    ``absolute_range`` instead -- which is upstream's intent (it randomizes MuJoCo ``dof_damping``,
    whose model default is also 0 for this hand, by scaling the *model* value that its own
    XML sets).
    """

    scale_range: tuple[float, float] = (0.3, 3.0)
    absolute_range: tuple[float, float] = (0.0, 0.002)
    _default_attr: str = "default_dofs_damping"
    _setter: str = "set_dofs_damping"

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        del envs_idx
        entity = self.resolve_entity(self.entity_name)
        default = getattr(entity, self._default_attr)[:, self._dofs.columns]
        if bool((default == 0.0).all()):
            damping = sample_uniform(
                self.absolute_range[0],
                self.absolute_range[1],
                (self._env.num_envs, len(self._dofs)),
                device=self.device,
            )
        else:
            log_lo, log_hi = (torch.tensor(self.scale_range, device=self.device)).log()
            damping = (
                default
                * sample_uniform(log_lo, log_hi, (self._env.num_envs, len(self._dofs)), device=self.device).exp()
            )
        entity.set_dofs_damping(damping, dofs_idx_local=self._dofs.dofs_idx_local)


__all__ = [
    "ObjectVelocityDisturbance",
    "RandomizeDofsArmature",
    "RandomizeDofsDamping",
    "RandomizeGravityTilt",
    "ResetDofsByPatternOffset",
    "ResetObjectInPalm",
]
