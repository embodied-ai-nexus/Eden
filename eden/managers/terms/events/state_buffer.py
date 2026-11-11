"""RESET event restoring each resetting env to a row of a :class:`~eden.utils.buffers.reset_states.ResetStateBuffer`."""

from __future__ import annotations

from dataclasses import dataclass

import genesis as gs
import torch

from eden.constants import EventMode
from eden.entities.rigid import StagedStateWrite
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.managers.resets.sampling import CategoryWeightedSampler, ResetConfigSampler, resolve_reset_config_sampler
from eden.utils.buffers.reset_states import (
    EntityStates,
    ResetStateBuffer,
    free_base_joint,
    state_joints,
    stores_root_pose,
)
from eden.utils.geom import inv_quat, inv_transform_by_quat, quat_mul, transform_by_quat
from eden.utils.misc import env_indices, is_env_mask, sanitize_envs_idx


@dataclass
class _EntityPlan:
    """Everything one entity's restore needs, resolved once in ``build()``."""

    name: str
    states: EntityStates
    #: Entity-local qs of the free joint (root position + wxyz quaternion), or ``None`` for an immobile base.
    root_qs: list[int] | None
    #: Entity-local DOFs the staged velocity covers: the free joint's six (if any), then the buffer's joints.
    vel_dofs: list[int]
    #: Entity-local DOFs of the buffer's joints, in its column order.
    joint_dofs: list[int]
    #: Per-env authored->solver offset of the root: solver origin in the authored frame, and the rotation between.
    offset_pos: torch.Tensor | None = None
    offset_quat: torch.Tensor | None = None
    #: A fixed base posed per env (``batch_fixed_verts``): its root is written through the base-link setters.
    fixed_root: bool = False


@EVENT_TERM_REGISTRY.register()
class ResetFromStateBuffer(EventTerm):
    """Restore each resetting env to one stored row of a reset-state buffer file.

    The buffer (:class:`~eden.utils.buffers.reset_states.ResetStateBuffer`, written by ``save()``) names the
    entities it covers. Each reset writes those entities' full state, meaning the root pose and velocity in the
    authored frame and every joint by name, through one staged write; a fixed base built with
    ``batch_fixed_verts=True`` (a robot mount or fixture placed per row) gets its root pose too. Entities it does
    not name keep what the scene reset and the other RESET events leave. Which row each env gets is the sampler's
    call: uniform by default, or the curriculum term named by ``sampler`` (e.g. Success Guided Sampling), bound to
    this buffer's size at build.

    RESET events run in ascending ``priority`` number, so give it a larger ``priority`` number than any RESET event
    that writes the same entities; one that runs after it overwrites the restored state. Task data stored with the
    rows (a goal) is read with :meth:`current_column`.

    Parameters
    ----------
    path: str | None
        The buffer file (``ResetStateBuffer.save``). Loaded and checked against the built entities in ``build()``.
    sampler: str | None
        Config key of a curriculum term subclassing ``ResetConfigSampler``, or ``None`` for uniform sampling.
    category_weights: dict[str, float] | None
        Draw a row's category by these weights (keyed by ``buffer.category_names``; a category left out is never
        drawn), then the row uniformly within it: :class:`CategoryWeightedSampler`. Needs a labelled buffer
        (``ResetStateBuffer.from_families``) and no ``sampler``.
    """

    mode = EventMode.RESET
    path: str | None = None
    sampler: str | None = None
    category_weights: dict[str, float] | None = None

    def build(self) -> None:
        if not self.path:
            raise ValueError(
                "ResetFromStateBuffer needs path=: the reset-state buffer file to restore from. "
                "Fix: ResetFromStateBuffer.configure(path='<buffer>.pt'); write one with ResetStateBuffer.save()."
            )
        # In the solver's float dtype, so the reset math never mixes precisions (``precision='64'``).
        self.buffer = ResetStateBuffer.load(self.path, device=self.device).to(self.device, dtype=gs.tc_float)
        self._plans = [self._plan(name, states) for name, states in self.buffer.entities.items()]
        self._dry_run_stage()
        consumer = f"ResetFromStateBuffer(path='{self.path}')"
        self._sampler: ResetConfigSampler = self._resolve_sampler(consumer)
        self._sampler.bind(self.buffer.num_states, consumer=consumer)
        self._buffer_generation = self.buffer.generation
        #: Each env's current row. Our own copy: the sampler's buffer is live and only valid at the call.
        self._current = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resolve_sampler(self, consumer: str) -> ResetConfigSampler:
        if self.category_weights is None:
            return resolve_reset_config_sampler(self._env, self.sampler, consumer=consumer)
        names = self.buffer.category_names
        if self.sampler is not None:
            raise ValueError(
                f"{consumer} sets both sampler='{self.sampler}' and category_weights; one picks the rows. "
                "Fix: drop one of them."
            )
        if self.buffer.category is None:
            raise ValueError(
                f"{consumer} sets category_weights, but the buffer labels no categories. "
                "Fix: build it with ResetStateBuffer.from_families, or drop category_weights for uniform rows."
            )
        unknown = sorted(set(self.category_weights) - set(names))
        if unknown:
            raise ValueError(
                f"{consumer}: category_weights names {unknown}, which the buffer does not have. "
                f"Its categories: {list(names)}."
            )
        weights = torch.tensor([float(self.category_weights.get(name, 0.0)) for name in names])
        return CategoryWeightedSampler(self.num_envs, self.device, self.buffer.category, weights)

    def _plan(self, name: str, states: EntityStates) -> _EntityPlan:
        """Resolve one buffer entity against the built scene, raising on anything a restore could not write."""
        from eden.entities.rigid import RigidEntity

        where = f"ResetFromStateBuffer(path='{self.path}')"
        if name not in self._env.entities:
            raise ValueError(
                f"{where} restores entity '{name}', which is not in the scene.\n"
                f"scene entities: {', '.join(self._env.entities)}\n"
                "Fix: regenerate the buffer for this scene, or rename the scene entity to match."
            )
        entity = self.resolve_entity(name)
        if not isinstance(entity, RigidEntity):
            raise TypeError(
                f"{where} restores entity '{name}', a {type(entity).__name__}; only a RigidEntity's state can be "
                "written per env."
            )
        root_qs = vel_dofs = None
        offset_pos = offset_quat = None
        fixed_root = False
        if entity.has_immobile_base:
            if states.has_root and not stores_root_pose(entity):
                if entity.is_attached:
                    reason, fix = "is welded to another entity, so it has no root to write", ""
                else:
                    reason = (
                        "is fixed-base without batch_fixed_verts=True, so Genesis holds one pose for it in every env"
                    )
                    fix = f" Or set batch_fixed_verts=True on '{name}' to restore a pose per env."
                raise ValueError(
                    f"{where} stores a root pose for entity '{name}', but '{name}' {reason}. Fix: regenerate the "
                    f"buffer with this scene (capture_reset_states stores no root for it).{fix}"
                )
            if not states.has_root and stores_root_pose(entity):
                raise ValueError(
                    f"{where} stores no root pose for entity '{name}', which is fixed-base with batch_fixed_verts=True, "
                    "so Genesis holds a pose for it per env: every reset would leave it wherever the scene reset put "
                    "it. Fix: store root_pos and root_quat for it (capture_reset_states does)."
                )
            if states.root_lin_vel is not None or states.root_ang_vel is not None:
                raise ValueError(
                    f"{where} stores a root velocity for entity '{name}', which is fixed-base and cannot move. "
                    "Fix: drop root_lin_vel / root_ang_vel for it from the buffer."
                )
            free_base_joint(entity, consumer=where)  # raises for a free body inside it, which no row can hold
            fixed_root = states.has_root
            vel_dofs = []
        else:
            if not states.has_root:
                raise ValueError(
                    f"{where} stores no root pose for entity '{name}', which has a free base: every reset would "
                    "leave it wherever the scene reset put it. Fix: store root_pos and root_quat for it."
                )
            free_joint = free_base_joint(entity, consumer=where)
            root_qs = list(free_joint.qs_idx_local)
            vel_dofs = list(free_joint.dofs_idx_local)
            # The authored frame is a rigid offset from the solver frame Genesis keeps in qpos (CoM re-origin and
            # morph offset). Read it once from Genesis' own two getters, per env (heterogeneous variants differ).
            authored_pos, authored_quat = entity.get_pos(relative=True), entity.get_quat(relative=True)
            offset_pos = inv_transform_by_quat(entity.get_pos() - authored_pos, authored_quat)
            offset_quat = quat_mul(inv_quat(authored_quat), entity.get_quat())

        joints = {joint.name: joint for joint in state_joints(entity)}
        available = ", ".join(f"{j.name} ({j.n_dofs} DOF)" for j in joints.values()) or "(none)"
        unknown = [j for j in states.joints_name if j not in joints]
        missing = [j for j in joints if j not in states.joints_name]
        if unknown or missing:
            raise ValueError(
                f"{where}: the joints stored for entity '{name}' do not match the entity's.\n"
                f"stored but not on the entity: {unknown or '(none)'}\n"
                f"on the entity but not stored: {missing or '(none)'}\n"
                f"entity joints: {available}\n"
                "Fix: regenerate the buffer with capture_reset_states on this scene. Every joint is required: a "
                "joint left out would keep the previous episode's value under a partial restore."
            )
        joint_dofs = []
        for joint_name, n_dofs in zip(states.joints_name, states.joints_n_dofs, strict=True):
            joint = joints[joint_name]
            if n_dofs != joint.n_dofs:
                raise ValueError(
                    f"{where}: entity '{name}' joint '{joint_name}' has {joint.n_dofs} DOFs, but the buffer stores "
                    f"{n_dofs}. Fix: regenerate the buffer for this asset."
                )
            joint_dofs.extend(joint.dofs_idx_local)
        vel_dofs = vel_dofs + joint_dofs
        return _EntityPlan(name, states, root_qs, vel_dofs, joint_dofs, offset_pos, offset_quat, fixed_root)

    def _dry_run_stage(self) -> None:
        """Stage one restore without applying it, so anything the write would refuse raises at build.

        ``StagedStateWrite`` refuses an IPC-articulated entity and a ball joint staged without all three of its
        angles. Both would otherwise surface on the first reset.
        """
        self._stage(StagedStateWrite(), torch.zeros(self.num_envs, dtype=torch.long, device=self.device), slice(None))

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx = sanitize_envs_idx(envs_idx, self.num_envs, prefer_slice=False)
        if self.buffer.generation != self._buffer_generation:
            # write_rows may have relabelled rows: regroup, or the weighted sampler draws by the old families.
            if isinstance(self._sampler, CategoryWeightedSampler):
                self._sampler.relabel(self.buffer.category)
            self._buffer_generation = self.buffer.generation
        assignment = self._sampler.sample(envs_idx)
        if is_env_mask(envs_idx):
            torch.where(envs_idx, assignment, self._current, out=self._current)
            rows = slice(None)  # every env's row; the mask picks which are written, on the device
        else:
            self._current[envs_idx] = assignment[envs_idx]
            rows = envs_idx
        # Fixed roots first: the staged write's forward pass then places their links from the new base pose.
        self._write_fixed_roots(self._current, envs_idx)
        write = StagedStateWrite()
        self._stage(write, self._current, rows)
        write.apply(envs_idx)

    def _write_fixed_roots(self, current: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Place every per-env fixed base at its row's authored pose; Genesis keeps no qpos for a fixed root.

        Written as the selected envs' rows and their ids, never every env's row under the mask: Genesis' base-pose
        setters take that form only on their zero-copy path, which hibernation, a disabled zero-copy and a relative
        write onto a morph offset all leave. The ids are free under the auto-reset's shared mask.
        """
        fixed = [plan for plan in self._plans if plan.fixed_root]
        if not fixed:
            return
        ids = env_indices(envs_idx)
        selected = current.index_select(0, ids)
        for plan in fixed:
            entity = self.resolve_entity(plan.name)
            entity.set_pos(plan.states.root_pos.index_select(0, selected), envs_idx=ids, relative=True)
            entity.set_quat(plan.states.root_quat.index_select(0, selected), envs_idx=ids, relative=True)

    def _stage(self, write: StagedStateWrite, current: torch.Tensor, rows: slice | torch.Tensor) -> None:
        """Stage every buffer entity's row ``current[env]``, keeping the env rows ``rows`` selects."""
        for plan in self._plans:
            entity = self.resolve_entity(plan.name)
            states = plan.states
            velocity = []
            if plan.root_qs is not None:
                pos_a = states.root_pos.index_select(0, current)
                quat_a = states.root_quat.index_select(0, current)
                lin_a = _gather_or_zeros(states.root_lin_vel, current, 3)
                ang_w = _gather_or_zeros(states.root_ang_vel, current, 3)
                offset_w = transform_by_quat(plan.offset_pos, quat_a)
                quat_s = quat_mul(quat_a, plan.offset_quat)
                # Free joint: world linear velocity of the solver origin, body-frame angular velocity.
                velocity += [lin_a + torch.cross(ang_w, offset_w, dim=-1), inv_transform_by_quat(ang_w, quat_s)]
                write.set_qpos(entity, torch.cat([pos_a + offset_w, quat_s], dim=-1)[rows], plan.root_qs)
            if plan.joint_dofs:
                write.set_dofs_pos(entity, states.dofs_pos.index_select(0, current)[rows], plan.joint_dofs)
                velocity.append(_gather_or_zeros(states.dofs_vel, current, len(plan.joint_dofs)))
            if plan.vel_dofs:  # a fixed base with no joints has its root pose and nothing else to write
                write.set_dofs_vel(entity, torch.cat(velocity, dim=-1)[rows], plan.vel_dofs)

    @property
    def current_config(self) -> torch.Tensor:
        """``(num_envs,)`` row index each env was last reset to. Read it; don't write it."""
        return self._current

    def current_column(self, name: str) -> torch.Tensor:
        """Task column ``name`` at each env's current row, ``(num_envs, ...)``; a goal stored with the states."""
        if name not in self.buffer.columns:
            raise KeyError(
                f"ResetFromStateBuffer(path='{self.path}') has no task column '{name}'. "
                f"Columns: {', '.join(self.buffer.columns) or '(none)'}."
            )
        return self.buffer.columns[name].index_select(0, self._current)


def _gather_or_zeros(column: torch.Tensor | None, current: torch.Tensor, width: int) -> torch.Tensor:
    """``column`` at the ``current`` rows, or zeros of that shape for a column the buffer leaves out.

    The zeros are in the solver's float dtype, like the loaded columns: not torch's default dtype, which a script
    can set apart from Genesis', and ``torch.cross`` does not promote a float32 operand to meet a float64 one.
    """
    if column is None:
        return torch.zeros(current.shape[0], width, dtype=gs.tc_float, device=current.device)
    return column.index_select(0, current)
