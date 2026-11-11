"""Rigid (articulated/actuated) entity wrapper with DOF, IK, and contact helpers.

DOF-ordering gotcha: ``set_qpos`` / ``get_dofs_limit`` / ``get_jacobian`` all use
**dofs_name order** (the configured ``dofs_name`` list, a pattern expanding in Genesis' joint
order; not the asset's declaration order), whereas ``find_named_dofs_idx_local``
returns indices in Genesis kinematic-tree order (breadth-first in genesis 1.1.x,
depth-first in 1.2.0) — use the latter for Genesis-level API calls, **not** for indexing
into ``get_jacobian`` columns or the ``default_dofs_*`` property tensors. ``get_jacobian``
already reorders its columns Genesis->dofs_name, so no manual reordering is needed. ``find_named_dofs``
resolves names into both spaces at once, and ``get_default_dofs_*`` reads the property tensors by name
(``eden.entities.dofs_default_mixin``).

For actuated entities under the **explicit** ``Euler`` integrator, ``post_build`` adds
``kd * substep_dt`` to the armature for numerical stability, so the effective armature
differs from the configured value. Genesis' default ``approximate_implicitfast`` and MuJoCo's
``implicitfast`` both integrate the PD controller's damping implicitly and get no bump —
``_needs_explicit_damping_armature`` is the gate. Read ``get_dofs_armature`` for what the
solver actually holds. "The PD controller" is Genesis' own (position/velocity control): a
torque-mode action term (``control_dofs_force``) computes its damping in Python, which is
explicit under every integrator.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Literal

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
from genesis.typing import UnitVec4FType, Vec2FType, Vec3FType

import eden as en
from eden.utils.contacts import aggregate_grouped_contacts, resolve_local_link_idx
from eden.entities.base import Entity
from eden.entities.dofs_default_mixin import DofDefaultsMixin
from eden.entities.support_mixin import SupportSurfaceMixin
from eden.options.actuators import ActuatorSpecOptions
from eden.options.entities import RESOLVED_DOF_TENSOR_FIELD_NAMES, RIGID_CONSTRAINT_FIELD_NAMES
from eden.utils.misc import (
    DofPermutation,
    DofRuns,
    as_genesis_control_index,
    as_genesis_index,
    env_indices,
    is_env_mask,
    sanitize_envs_idx,
)
from eden.utils.string import resolve_matching_names, resolve_matching_names_values

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidGeom, RigidJoint

# Config-level names for the per-link reference frames Genesis accepts on velocity reads and
# external wrenches. Genesis takes ``gs.link_ref_frame`` members only (strings raise), and
# dropped ``root_COM`` for these calls, so the map is the validator: an unknown name fails
# fast with the supported spellings.
LINK_REF_FRAMES: dict[str, gs.link_ref_frame] = {
    "link_origin": gs.link_ref_frame.link_origin,
    "link_com": gs.link_ref_frame.link_COM,
}


# The ``RigidGeom`` setter of each per-link friction field, and of each material friction coefficient.
_GEOM_FRICTION_SETTERS = {
    "default_geoms_friction": "set_friction",
    "default_geoms_friction_torsional": "set_friction_torsional",
    "default_geoms_friction_rolling": "set_friction_rolling",
}
_MATERIAL_FRICTION_SETTERS = {
    "friction": "set_friction",
    "friction_torsional": "set_friction_torsional",
    "friction_rolling": "set_friction_rolling",
}


def resolve_link_ref_frame(ref: str | gs.link_ref_frame) -> gs.link_ref_frame:
    """Resolve a reference-frame name (or an already-resolved member) to ``gs.link_ref_frame``.

    Parameters
    ----------
    ref : str | gs.link_ref_frame
        One of ``"link_origin"`` / ``"link_com"``, or a ``gs.link_ref_frame`` member.

    Returns
    -------
    gs.link_ref_frame
        The resolved enum member.
    """
    if isinstance(ref, gs.link_ref_frame):
        return ref
    try:
        return LINK_REF_FRAMES[ref]
    except KeyError:
        raise ValueError(f"Unknown link reference frame {ref!r}; expected one of {sorted(LINK_REF_FRAMES)}.") from None


def _row_major_selection(values: torch.Tensor, selection) -> torch.Tensor:
    """``values`` read for an explicit ``selection``, laid out row-major whatever form the selection took.

    Genesis returns a ``range`` (slice) selection env-minor — a strided view of its DOF-major state — and an
    index-tensor selection as a fresh row-major copy. The values are the same, but ``torch.rand_like`` keeps
    strides, so observation noise would land on other elements, and ``.view()`` refuses the layout. The
    ``dofs_idx_local=None`` defaults hand Genesis a ``range``, so without this a default read would come back
    env-minor. ``selection=None`` (every link) keeps Genesis' own layout.
    """
    return values if selection is None else values.contiguous()


#: ``(id(index tensor), control) -> (weakref, _version, resolved selector)`` for :func:`_genesis_selection`;
#: ``None`` for a tensor with no cheaper form, which is handed on as it is.
_RESOLVED_SELECTIONS: dict[tuple[int, bool], tuple[weakref.ref, int, range | DofPermutation | DofRuns | None]] = {}
#: ``(indices, control, device) -> resolved selector`` for the list / tuple selections: keyed by value.
_RESOLVED_LISTS: dict[tuple[tuple[int, ...], bool, str], range | DofRuns | torch.Tensor] = {}


def _genesis_selection(selection, *, control: bool = False):
    """An explicit DOF / q / link ``selection`` in the form Genesis serves cheapest: a ``range`` when contiguous.

    Terms resolve their link and DOF indices once, in ``build()``, and hand the same index tensor to
    every call. Genesis reads a one-element index tensor back with ``.item()`` on every call and uploads
    a Python list of several on every call — a GPU->CPU sync either way — and a ``control_dofs_*``
    setter syncs twice on any index tensor. So a tensor is read on the host the first time it is
    seen and remembered by identity: the ``range`` it equals when contiguous (with ``control``, the
    :class:`~eden.utils.misc.DofPermutation` / :class:`~eden.utils.misc.DofRuns` of a reordered or
    gapped one), else the tensor itself. The cache holds the tensor only weakly, so its entry goes with
    it — though an index built afresh on every call is read back on every call, so build it once. Its
    ``_version`` is part of the check, so an in-place write to it is picked up; an inference tensor
    tracks no version and passes through untouched. A list or tuple is remembered by value, and when it
    has no ``range`` form it becomes a device index tensor, built once — Genesis would upload it per
    call. An index counted from the end (``-1``) is handed on as it is: its ``range`` would select
    something else. The selected entries and their order never change, only the path Genesis takes.
    """
    if isinstance(selection, torch.Tensor):
        if selection.dtype == torch.bool or selection.ndim != 1:
            return selection
        try:
            version = selection._version
        except RuntimeError:
            return selection
        key = (id(selection), control)
        cached = _RESOLVED_SELECTIONS.get(key)
        if cached is not None and cached[0]() is selection and cached[1] == version:
            return selection if cached[2] is None else cached[2]
        # A permutation's `order` goes where the values are, the sim device, whatever device the index is on:
        # on the index's own (a CPU one), every call would copy the values to the host and back.
        resolved = as_genesis_control_index(selection, device=gs.device) if control else as_genesis_index(selection)
        _RESOLVED_SELECTIONS[key] = (
            weakref.ref(selection, lambda _, key=key, cache=_RESOLVED_SELECTIONS: cache.pop(key, None)),
            version,
            # Not `selection` itself: a strong reference here would keep it alive, its weakref would never
            # fire, and every index tensor built per call would stay in the cache for good.
            None if resolved is selection else resolved,
        )
        return resolved
    if (
        isinstance(selection, (list, tuple))
        and selection
        and all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) for i in selection)
    ):
        key = (tuple(int(i) for i in selection), control, str(gs.device))
        if min(key[0]) < 0:
            # Counted from the end, so no `range` form; Genesis takes the list as it always did. A device
            # tensor would cost more: a one-element one is `.item()`-ed per call, the list is not.
            return selection
        resolved = _RESOLVED_LISTS.get(key)
        if resolved is None:
            # Explicitly on the CPU, whatever torch's default device is: a CUDA copy read back here would be
            # the very sync this avoids.
            host = torch.tensor(key[0], dtype=torch.long, device="cpu")
            resolved = as_genesis_control_index(host, device=gs.device) if control else as_genesis_index(host)
            if isinstance(resolved, torch.Tensor):
                resolved = resolved.to(gs.device)
            _RESOLVED_LISTS[key] = resolved
        return resolved
    return selection


#: ``(start, stop, step, device) -> device index tensor`` for :func:`_kernel_selection`.
_KERNEL_RANGES: dict[tuple[int, int, int, str], torch.Tensor] = {}


def _kernel_selection(selection):
    """A DOF ``range`` as the device index tensor it equals, for a setter Genesis only runs as a kernel.

    Genesis' ``set_dofs_position`` has no zero-copy path: it converts the DOF argument with
    ``torch.as_tensor(..., device=gs.device)``, which for a ``range`` is a fresh host->device upload, a
    GPU->CPU sync, on every call — and the reset events call it every step the auto-reset fires.
    A device tensor passes through that conversion untouched, so it is built once per ``range``.
    """
    if not isinstance(selection, range):
        return selection
    key = (selection.start, selection.stop, selection.step, str(gs.device))
    tensor = _KERNEL_RANGES.get(key)
    if tensor is None:
        tensor = _KERNEL_RANGES[key] = torch.arange(*key[:3], dtype=gs.tc_int, device=gs.device)
    return tensor


def _control_dofs(write, values, selection, envs_idx) -> None:
    """``write(values, selection, envs_idx)`` for a Genesis ``control_dofs_*`` ``write``, one ``range`` per call.

    A :class:`~eden.utils.misc.DofPermutation` is one write through its ``range`` with the value columns
    permuted, a :class:`~eden.utils.misc.DofRuns` one write per run with that run's columns, and any other
    selection one write as given. ``values`` takes what Genesis takes: a single number (a Python one or a
    0-dim array) sets every selected DOF and goes to each write whole, and a list is made a tensor once,
    since ``[..., columns]`` cannot slice it.
    """
    if not isinstance(selection, (DofPermutation, DofRuns)):
        write(values, selection, envs_idx)
        return
    single = isinstance(values, (int, float)) or np.ndim(values) == 0
    if isinstance(selection, DofPermutation):
        if not single:
            values = torch.as_tensor(values, device=selection.order.device).index_select(-1, selection.order)
        write(values, selection.dofs, envs_idx)
        return
    if not single and not isinstance(values, (torch.Tensor, np.ndarray)):
        values = torch.as_tensor(values, device=gs.device)
    for dofs, columns in selection:
        write(values if single else values[..., columns], dofs, envs_idx)


#: One staged write: its scene-global indices (on the host) and its values, ``(n_rows, len(indices))``.
_StagedEntry = tuple[tuple[int, ...], torch.Tensor]


@dataclass
class _SolverStage:
    """One solver's staged writes, each kind in staging order.

    A DOF position is keyed by the q it moves: ``dofs_pos`` holds hinge and slide offsets from ``qpos0``,
    ``dofs_qpos`` the free and ball joints' qpos.
    """

    dofs_vel: list[_StagedEntry] = field(default_factory=list)
    dofs_pos: list[_StagedEntry] = field(default_factory=list)
    dofs_qpos: list[_StagedEntry] = field(default_factory=list)
    qpos: list[_StagedEntry] = field(default_factory=list)


def _every_env_row(values: torch.Tensor, envs_idx) -> bool:
    """Whether ``values`` holds every env's row alongside the bool mask ``envs_idx`` (not just the selected ones)."""
    return is_env_mask(envs_idx) and values.shape[0] == envs_idx.shape[0]


def _solver_takes_every_row(solver) -> bool:
    """Whether ``solver``'s state setters write every env's row through a bool mask on the device.

    See :attr:`RigidEntity.setters_take_every_row`; the DOF (q) selection must be a ``range`` too.
    """
    return gs.use_zerocopy and not getattr(solver, "_use_hibernation", False)


def _mask_rows(values, envs_idx, *, on_device: bool):
    """``(values, envs_idx)`` to hand a Genesis state setter, for a bool-mask ``envs_idx``.

    ``on_device``: whether the setter writes every env's row through the mask on the device
    (:func:`_solver_takes_every_row`, with a ``range`` selection). Then a single row is spread over
    every env — the kinematic solver would scatter it into the masked rows instead of broadcasting
    it. Otherwise every env's row is cut to the selected ones and the mask's ids, one ``nonzero()``
    (free for the auto-reset's mask, :func:`~eden.utils.misc.sharing_env_indices`). Anything but a
    mask with a 2-D tensor value passes through.
    """
    if not (is_env_mask(envs_idx) and isinstance(values, torch.Tensor) and values.ndim == 2):
        return values, envs_idx
    num_envs = envs_idx.shape[0]
    if values.shape[0] == num_envs and not on_device:
        ids = env_indices(envs_idx)
        return values[ids], ids
    if values.shape[0] == 1 and on_device:
        return values.expand(num_envs, -1), envs_idx
    return values, envs_idx


def _global_indices(start: int, idx_local) -> tuple[int, ...]:
    return tuple(start + i for i in idx_local)


#: Columns of a staged value: ``None`` for all of them, else a slice or a device index tensor.
_Columns = slice | torch.Tensor | None


@dataclass(frozen=True)
class _DofsPosPlan:
    """How one ``set_dofs_pos`` index list becomes qpos (see :func:`_dofs_pos_plan`).

    ``scalar``: the hinge and slide DOFs' columns and the q each moves (``q = qpos0 + value``).
    ``translations``: a free joint's position DOFs' columns and qs (``q = value``). ``rotations``: per
    free or ball joint, its three Euler-angle columns (x, y, z) and the four qs of its quaternion.
    """

    scalar: tuple[_Columns, tuple[int, ...]]
    translations: tuple[_Columns, tuple[int, ...]]
    rotations: tuple[tuple[_Columns, tuple[int, ...]], ...]


#: Per Eden entity: ``tuple(dofs_idx_local) -> _DofsPosPlan`` (see :func:`_dofs_pos_plan`).
_DOFS_POS_PLANS: weakref.WeakKeyDictionary[RigidEntity, dict[tuple[int, ...], _DofsPosPlan]] = (
    weakref.WeakKeyDictionary()
)
#: Per solver: its ``qpos0``, ``(n_envs, n_qs)``. A view under zero-copy and a copy otherwise: it is fixed once the
#: scene is built.
_QPOS0: weakref.WeakKeyDictionary[object, torch.Tensor] = weakref.WeakKeyDictionary()
#: Per solver: ``(indices) -> device index tensor``: the ``qpos0`` gather in :meth:`StagedStateWrite.apply` and the
#: column order in :meth:`StagedStateWrite._join`.
_DEVICE_INDICES: weakref.WeakKeyDictionary[object, dict[tuple[int, ...], torch.Tensor]] = weakref.WeakKeyDictionary()


def _columns(columns: list[int], n_columns: int, device) -> _Columns:
    """``columns`` of an ``n_columns``-wide value as the cheapest selector: all, a slice, or an index tensor."""
    if columns == list(range(n_columns)):
        return None
    if columns == list(range(columns[0], columns[0] + len(columns))):
        return slice(columns[0], columns[0] + len(columns))
    return torch.tensor(columns, dtype=torch.long, device=device)


def _select(values: torch.Tensor, columns: _Columns) -> torch.Tensor:
    return values if columns is None else values[:, columns]


def _dofs_pos_plan(entity: RigidEntity, dofs_idx_local) -> _DofsPosPlan:
    """How a DOF position staged on ``entity``'s ``dofs_idx_local`` becomes qpos, as Genesis ``set_dofs_position``
    turns DOF state into qpos: a hinge or slide is ``qpos0 + value``, a free joint's position is the value itself,
    and a free or ball joint's three Euler angles are its quaternion (:func:`genesis.utils.geom.xyz_to_quat`,
    Genesis' kernel conversion). A quaternion takes all three angles, so a joint's rotation DOFs are staged
    together or not at all. Cached per entity and index list.
    """
    key = tuple(dofs_idx_local)
    plans = _DOFS_POS_PLANS.setdefault(entity, {})
    plan = plans.get(key)
    if plan is not None:
        return plan
    if unknown := sorted({i for i in key if not 0 <= i < entity.n_dofs}):
        raise ValueError(
            f"StagedStateWrite.set_dofs_pos: entity '{entity.name}' has {entity.n_dofs} DOFs, so the entity-local "
            f"DOF indices {unknown} select none of them."
        )
    column_of = {entity.dof_start + i: column for column, i in enumerate(key)}
    scalar_columns, scalar_qs, translation_columns, translation_qs, rotations = [], [], [], [], []
    for joint in entity._entity.joints:
        if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
            if joint.dof_start in column_of:
                scalar_columns.append(column_of[joint.dof_start])
                scalar_qs.append(joint.q_start)
            continue
        if joint.type == gs.JOINT_TYPE.FREE:
            for k in range(3):
                if joint.dof_start + k in column_of:
                    translation_columns.append(column_of[joint.dof_start + k])
                    translation_qs.append(joint.q_start + k)
            angles, quat_start = joint.dof_start + 3, joint.q_start + 3
        elif joint.type == gs.JOINT_TYPE.SPHERICAL:
            angles, quat_start = joint.dof_start, joint.q_start
        else:
            continue
        staged = [k for k in range(3) if angles + k in column_of]
        if not staged:
            continue
        if len(staged) < 3:
            local = [angles + k - entity.dof_start for k in range(3)]
            raise ValueError(
                f"StagedStateWrite.set_dofs_pos: entity '{entity.name}' joint '{joint.name}' has rotation DOFs "
                f"{local} (entity-local), and only {[local[k] for k in staged]} are staged. Its qpos is a "
                f"quaternion built from all three Euler angles, so stage all of {local} in the same set_dofs_pos "
                f"call, or write the joint's quaternion with set_qpos."
            )
        rotations.append(([column_of[angles + k] for k in range(3)], tuple(range(quat_start, quat_start + 4))))
    device, n_columns = gs.device, len(key)
    plan = plans[key] = _DofsPosPlan(
        scalar=(_columns(scalar_columns, n_columns, device) if scalar_columns else None, tuple(scalar_qs)),
        translations=(
            _columns(translation_columns, n_columns, device) if translation_columns else None,
            tuple(translation_qs),
        ),
        rotations=tuple((_columns(columns, n_columns, device), qs) for columns, qs in rotations),
    )
    return plan


def _qpos0(solver) -> torch.Tensor:
    qpos0 = _QPOS0.get(solver)
    if qpos0 is None:
        from genesis.utils.misc import qd_to_torch

        qpos0 = _QPOS0[solver] = qd_to_torch(solver.qpos0, transpose=True, copy=not gs.use_zerocopy)
    return qpos0


def _device_index(solver, indices: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """``indices`` as a device index tensor, built once per solver."""
    cached = _DEVICE_INDICES.setdefault(solver, {})
    index = cached.get(indices)
    if index is None:
        index = cached[indices] = torch.tensor(indices, dtype=torch.long, device=device)
    return index


class StagedStateWrite:
    """Several rigid entities' state writes on one ``envs_idx``, applied as one Genesis call per kind.

    **Why.** Every Genesis state setter resets the selected envs' collider and constraint state and,
    unless told to skip it, runs forward kinematics over every link of those envs, whichever entity it
    wrote. N entities written one by one pay N of each where one would do.

    **What.** ``set_dofs_pos`` / ``set_dofs_vel`` / ``set_qpos`` stand in for the
    :class:`RigidEntity` setters of the same names: each only records its entity's values, and
    :meth:`apply` writes them all. The values mean what they mean to those setters (``set_dofs_pos``
    writes a DOF's state as Genesis holds it, ``set_qpos`` the coordinate itself), but nothing here zeroes a
    velocity on the side: stage the zeros where the entity setter's ``zero_velocity=True`` would. A free
    or ball joint's rotation DOFs are Euler angles, so ``set_dofs_pos`` takes all three of them or none.

    **Order.** :meth:`apply` writes per solver (a kinematic-material entity lives in its own):

    1. velocities, ``skip_forward=True``, so the forward pass after them sees the fresh ones;
    2. positions, as one qpos write and its forward pass. A staged DOF position becomes the qpos
       Genesis ``set_dofs_position`` would make of it (``qpos0 + value`` for a hinge or slide, the
       quaternion of a free or ball joint's Euler angles), placed before the staged qpos, so a qpos
       staged on the same coordinate wins.

    Nothing goes through Genesis ``set_dofs_position``. That setter re-derives *every* entity's qpos in
    the selected envs from its DOF state, which holds a free joint's orientation as Euler angles, so a
    quaternion written earlier, by this write or by any before it, would come back as ``-q``. Here only
    the joints a ``set_dofs_pos`` names are converted.

    **Envs.** ``envs_idx`` is int env ids or the bool mask they come from; a staged value has one row
    per selected env, in ascending env order for a mask — or, with a mask, every env's row, the mask
    picking which are written (stage every value of the write that way, then). A mask is what Genesis'
    setters take on the device (``masked_scatter_``, or ``torch.where`` for every env's row): handed ids
    they would write their error flag through them, a GPU->CPU sync per setter. Every env's row
    needs no ``nonzero()`` at all, so a write that may select no env can run unconditionally (the
    motion-end resync); where Genesis cannot take the rows on the device (hibernation, an index tensor
    selection) the selected ones are gathered here.

    **Indices** are entity-local and on the host (a ``range`` or ints), since they are read in Python on
    every call: resolve them once, in ``build()``. An index staged twice keeps the later value.
    """

    def __init__(self) -> None:
        self._stages: dict[object, _SolverStage] = {}

    def _stage_of(self, entity: RigidEntity) -> _SolverStage:
        return self._stages.setdefault(entity._entity.solver, _SolverStage())

    @staticmethod
    def _refuse_ipc_articulation(entity: RigidEntity, setter: str) -> None:
        """The guard Genesis' own entity setter applies, which the solver-level write in :meth:`apply` bypasses."""
        if entity._env._has_ipc_coupler and getattr(entity.material, "coup_type", None) == "external_articulation":
            raise ValueError(
                f"StagedStateWrite.{setter}: entity '{entity.name}' is IPC-coupled with "
                f"coup_type='external_articulation', so its joint state cannot be written: the IPC constraint "
                f"system drives its joints."
            )

    @staticmethod
    def check_dofs_pos(entity: RigidEntity, dofs_idx_local) -> None:
        """Raise what :meth:`set_dofs_pos` would raise for ``dofs_idx_local``; call it in ``build()``."""
        _dofs_pos_plan(entity, dofs_idx_local)

    def set_dofs_pos(self, entity: RigidEntity, position: torch.Tensor, dofs_idx_local) -> None:
        """Stage :meth:`RigidEntity.set_dofs_pos` (``zero_velocity=False``) for ``entity``."""
        self._refuse_ipc_articulation(entity, "set_dofs_pos")
        plan = _dofs_pos_plan(entity, dofs_idx_local)
        stage = self._stage_of(entity)
        scalar_columns, scalar_qs = plan.scalar
        if scalar_qs:
            stage.dofs_pos.append((scalar_qs, _select(position, scalar_columns)))
        translation_columns, translation_qs = plan.translations
        if translation_qs:
            stage.dofs_qpos.append((translation_qs, _select(position, translation_columns)))
        for columns, qs in plan.rotations:
            stage.dofs_qpos.append((qs, gu.xyz_to_quat(_select(position, columns))))

    def set_dofs_vel(self, entity: RigidEntity, velocity: torch.Tensor, dofs_idx_local) -> None:
        """Stage :meth:`RigidEntity.set_dofs_vel` for ``entity``; :meth:`apply` skips its forward pass."""
        self._stage_of(entity).dofs_vel.append((_global_indices(entity.dof_start, dofs_idx_local), velocity))

    def set_qpos(self, entity: RigidEntity, qpos: torch.Tensor, qs_idx_local) -> None:
        """Stage :meth:`RigidEntity.set_qpos` (``zero_velocity=False``) for ``entity``."""
        self._refuse_ipc_articulation(entity, "set_qpos")
        self._stage_of(entity).qpos.append((_global_indices(entity._entity.q_start, qs_idx_local), qpos))

    def apply(self, envs_idx: torch.Tensor) -> None:
        """Write everything staged to ``envs_idx`` (int env ids or a bool mask), in the class docstring's order.

        Empties the stage.
        """
        for solver, stage in self._stages.items():
            if stage.dofs_vel:
                velocity, dofs_idx = self._join(solver, stage.dofs_vel)
                selection = _genesis_selection(dofs_idx)
                on_device = isinstance(selection, range) and _solver_takes_every_row(solver)
                velocity, envs = _mask_rows(velocity, envs_idx, on_device=on_device)
                solver.set_dofs_velocity(velocity, selection, envs, skip_forward=True)
            positions = [*stage.dofs_qpos, *stage.qpos]
            if stage.dofs_pos:
                offsets, qs_idx = self._join(solver, stage.dofs_pos)
                qpos0 = _qpos0(solver)
                columns = _device_index(solver, tuple(qs_idx), qpos0.device)
                if _every_env_row(offsets, envs_idx):
                    base = qpos0.index_select(1, columns)
                else:
                    # The selected envs' rows, through their ids: free for the auto-reset's mask
                    # (`sharing_env_indices`).
                    base = qpos0[env_indices(envs_idx).unsqueeze(-1), columns]
                positions = [(tuple(qs_idx), offsets + base), *positions]
            if positions:
                qpos, qs_idx = self._join(solver, positions)
                selection = _genesis_selection(qs_idx)
                on_device = isinstance(selection, range) and _solver_takes_every_row(solver)
                qpos, envs = _mask_rows(qpos, envs_idx, on_device=on_device)
                solver.set_qpos(qpos, selection, envs, skip_forward=False)
        self._stages.clear()

    @staticmethod
    def _join(solver, staged: list[_StagedEntry]) -> tuple[torch.Tensor, list[int]]:
        """One kind's values and indices as a single write, in ascending index order.

        Ascending so that entities adjacent in the scene, and an entry staged in another order (a motion's
        DOF order), join into one contiguous selection, which :func:`_genesis_selection` hands Genesis as a
        ``range`` (its sync-free path) instead of a tensor. An index staged twice keeps its later value, as
        setters called in staging order would. The values come out in the solver's dtype: Genesis' mask
        path (``masked_scatter_``) does not cast.
        """
        staged_indices = [i for entry_indices, _ in staged for i in entry_indices]
        last_column = {i: column for column, i in enumerate(staged_indices)}
        indices = sorted(last_column)
        columns = tuple(last_column[i] for i in indices)
        values = torch.cat([values.to(gs.tc_float) for _, values in staged], dim=-1)
        if columns != tuple(range(len(staged_indices))):
            values = values.index_select(-1, _device_index(solver, columns, values.device))
        return values, indices


class RigidEntity(Entity, SupportSurfaceMixin, DofDefaultsMixin):
    """Entity subclass for rigid and kinematic materials.

    Provides DOF/link/geom access, FK/IK, Jacobians, contacts,
    and support-surface utilities on top of the base :class:`Entity`.

    Notes
    -----
    **DOF ordering.** Every ``*_dofs_*`` accessor below indexes DOFs in *dofs_name
    order* — the entity's configured ``dofs_name`` list, a pattern expanding in Genesis' joint
    order, not the asset's declaration order — because the default ``dofs_idx_local`` is
    :attr:`dofs_idx_local`, which is name-resolved in that order.
    :meth:`find_named_dofs_idx_local` returns indices in Genesis' kinematic-tree order
    instead: that is what Genesis-level APIs want, but it is *not* a column index into
    :meth:`get_jacobian`, which already permutes its columns back to dofs_name order.
    Passing an explicit ``dofs_idx_local`` returns values in the order *you* supplied.

    **DOF defaults.** The configured per-DOF tensors (``default_dofs_*``, ``soft_dofs_pos_limits``)
    are laid out in dofs_name order too, which no ``find_named_dofs_idx_local`` index addresses.
    Read them by name with ``get_default_dofs_*`` / :meth:`get_soft_dofs_pos_limits`; to slice or
    write one, resolve once with :meth:`find_named_dofs`, whose selection carries the columns *and* the
    matching ``dofs_idx_local`` — see :class:`~eden.entities.dofs_default_mixin.DofDefaultsMixin`.

    **Quaternions** are ``(w, x, y, z)``.

    **Batching.** Shapes below are written with the leading ``n_envs`` axis they carry; where
    an accessor takes ``envs_idx`` (int / slice / index tensor / bool mask), that axis is
    narrowed to the selection. (:meth:`get_mass` is the exception that proves the rule — it
    reports the entity's *total* mass, which is not a per-DOF or per-link quantity at all.)

    For the *info* getters that axis is Eden's contract, not Genesis'. Genesis splits per-DOF data in two: *state*
    (position, velocity, force) is always env-batched, while *info* — the nine
    model-parameter reads :meth:`get_dofs_kp`, :meth:`get_dofs_kd`,
    :meth:`get_dofs_stiffness`, :meth:`get_dofs_damping`, :meth:`get_dofs_armature`,
    :meth:`get_dofs_frictionloss`, :meth:`get_dofs_invweight`, :meth:`get_dofs_limit`
    and :meth:`get_dofs_force_range` — is batched only when ``EnvOptions.batch_dofs_info``
    is set, and rejects ``envs_idx`` outright when it is not. Since the shape a caller
    got would otherwise depend on the task config (and Eden ships tasks on both
    settings), the info getters normalize: when DOF info is unbatched every environment
    shares the one stored row, so broadcasting it over environments is exact.

    Note that ``EnvOptions.batch_dofs_info`` only reaches the *rigid* solver.
    ``KinematicSolver`` keeps unbatched DOF info unless it holds a heterogeneous entity,
    so a kinematic entity's info reads are otherwise broadcast however the env is configured.
    Either solver batches by itself once it holds one (:attr:`has_batched_dofs_info`).
    """

    dofs_name: list[str] = []
    default_dofs_pos: dict[str, float] = {}
    default_dofs_vel: dict[str, float] = {}
    default_dofs_stiffness: dict[str, float] = {}
    default_dofs_damping: dict[str, float] = {}
    default_dofs_armature: dict[str, float] = {}
    default_dofs_frictionloss: dict[str, float] = {}
    default_dofs_spring_ref: dict[str, float] = {}
    default_dofs_kp: dict[str, float] = {}
    default_dofs_kd: dict[str, float] = {}
    default_dofs_pos_limits: dict[str, Vec2FType] = {}
    soft_dofs_pos_limits: dict[str, Vec2FType] = {}
    default_dofs_force_limits: dict[str, float] = {}
    default_equalities_sol_params: tuple[float, ...] | None = None
    default_geoms_sol_params: dict[str, tuple[float, ...]] = {}
    default_geoms_friction: dict[str, float] = {}
    default_geoms_friction_torsional: dict[str, float] = {}
    default_geoms_friction_rolling: dict[str, float] = {}

    is_articulated: bool = False
    is_actuated: bool = False
    is_support_enabled: bool = False

    support_sample_method: Literal["uniform", "grid"] = "uniform"
    support_minimum_clearance: float = 0.05
    support_shrink: float = 0.1
    support_num_sample_points: int = 300
    support_grid_size: float = 0.01
    support_links_name: list[str] = []

    @property
    def fixed(self) -> bool:
        return self.is_fixed_base and not self.is_articulated

    @property
    def _dof_defaults_num_envs(self) -> int:
        return self._env.num_envs

    @property
    def dofs_spec(self):
        # A group's own ``dofs_spec`` is usually the ClassVar default; its variants hold the real one.
        if self.is_heterogeneous:
            return self._options.shared_dofs_spec
        return self._options.dofs_spec

    # ------------------------------------------------------------------------------------
    # ----------------------------------- post_build -------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def _parse_dofs_info(
        self,
        dofs_info: dict[str, float | Vec2FType],
        default_info: torch.Tensor,
    ) -> torch.Tensor:
        info = default_info.clone()
        idx, _, dofs_values = resolve_matching_names_values(
            dofs_info, self.dofs_name, preserve_order=True, literal_names=self._every_joint_name
        )

        for i, val in zip(idx, dofs_values):
            # ``as_tensor`` because torch rejects a python sequence here, and range-per-DOF
            # properties (``default_dofs_pos_limits``) pass a 2-tuple.
            info[i] = torch.as_tensor(val, dtype=info.dtype, device=info.device)

        return info.to(self.device)

    @gs.assert_built
    def _init_dofs_property(
        self,
        attr_name: str,
        setter_name: str | None,
        default_info: torch.Tensor,
        *,
        apply_to_sim: bool = True,
        force_broadcast: bool = False,
        variant_default_info: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Initialize a DOF property: parse user dict, store as tensor, optionally apply to sim.

        Works for both scalar-per-DOF (shape ``[1, num_dofs]``) and range-per-DOF
        (shape ``[1, num_dofs, 2]``) properties — ``_parse_dofs_info`` infers the shape
        from the value type (float vs tuple).

        Parameters
        ----------
        attr_name : str
            Attribute name on self, e.g. ``"default_dofs_stiffness"``.
        setter_name : str | None
            Method name to write values to sim, or None to skip.
        default_info : torch.Tensor
            Used as per-key fallback inside ``_parse_dofs_info`` when the user dict
            is set, and as the full silent fallback when the user dict is unset. Both
            branches return a tensor on ``self.device``, whatever device this one is on.
        apply_to_sim : bool
            Whether to call the setter after parsing.
        force_broadcast : bool
            Always repeat for all envs, bypassing the ``batch_dofs_info`` check.
            Use for properties whose setter always expects a full ``(num_envs, ...)`` tensor.
        variant_default_info : torch.Tensor | None
            Heterogeneous only: ``default_info`` per present variant, ``(n_variants, ...)``, for a property
            whose sim default can differ between variants (each variant's initial DOF positions, its default
            armature).
        """
        if attr_name in self._variant_fields or variant_default_info is not None:
            return self._init_variant_dofs_property(
                attr_name,
                setter_name,
                default_info,
                variant_default_info,
                apply_to_sim=apply_to_sim,
                force_broadcast=force_broadcast,
            )
        user_dict = getattr(self, attr_name)
        if user_dict:
            parsed = self._parse_dofs_info(user_dict, default_info=default_info).unsqueeze(0)
            self._warn_zero_gain_fallthrough(attr_name, user_dict, parsed[0])

            if force_broadcast:
                broadcast = parsed.repeat(self._env.num_envs, *([1] * (parsed.ndim - 1)))
                if apply_to_sim and setter_name is not None:
                    getattr(self, setter_name)(broadcast)
                return broadcast
            if apply_to_sim and setter_name is not None:
                getattr(self, setter_name)(parsed[0])
            return parsed
        else:
            # Moved like the parsed branch's result (`_parse_dofs_info` ends in `.to(self.device)`): otherwise a
            # property whose dict is left unset -- `default_dofs_vel`, `{}` by default -- would stay a CPU
            # tensor on a GPU backend, while one with any entry set is a device tensor.
            return default_info.unsqueeze(0).to(self.device)

    def _init_variant_dofs_property(
        self,
        attr_name: str,
        setter_name: str | None,
        default_info: torch.Tensor,
        variant_default_info: torch.Tensor | None,
        *,
        apply_to_sim: bool,
        force_broadcast: bool,
    ) -> torch.Tensor:
        """``_init_dofs_property`` for a heterogeneous entity: one row per variant, then one per env.

        Each present variant's value (``_variant_fields``, else the mirrored one) is parsed over its own
        default. Rows that all agree give the homogeneous ``(1 | num_envs, ...)`` result. Rows that differ are
        gathered to ``(num_envs, ...)`` through ``_env_variant_idx``, and writing them to the sim needs
        ``EnvOptions(batch_dofs_info=True)`` unless the setter is per env anyway (``force_broadcast``).
        """
        variant_values = self._variant_fields.get(attr_name)
        user_dicts = [
            getattr(self, attr_name) if variant_values is None else variant_values[v]
            for v in range(len(self._variant_first_envs))
        ]
        rows = []
        for v, user_dict in enumerate(user_dicts):
            row_default = default_info if variant_default_info is None else variant_default_info[v]
            if user_dict:
                row = self._parse_dofs_info(user_dict, default_info=row_default)
                self._warn_zero_gain_fallthrough(attr_name, user_dict, row)
            else:
                row = row_default.to(self.device)
            rows.append(row)
        rows = torch.stack(rows)
        writes = apply_to_sim and setter_name is not None

        if bool((rows == rows[:1]).all()):
            value = rows[:1]
            if force_broadcast:
                value = value.repeat(self._env.num_envs, *([1] * (value.ndim - 1)))
            if writes and any(user_dicts):
                getattr(self, setter_name)(value if force_broadcast else value[0])
            return value

        value = rows[self._env_variant_idx]
        # As above, and as for an ungrouped entity: the sim already holds each variant's own default.
        if writes and any(user_dicts):
            if not force_broadcast:
                self._require_batched_dofs_info(attr_name)
            getattr(self, setter_name)(value)
        return value

    def _require_batched_dofs_info(self, attr_name: str) -> None:
        """Raise unless per-env DOF info is on, which a heterogeneous entity's per-variant ``attr_name`` needs."""
        if self.has_batched_dofs_info:
            return
        values = self._variant_fields.get(attr_name)
        raise ValueError(
            f"scene_options.{self.name} gives its variants different {attr_name} ({values}, one per variant), which "
            f"is written to the simulator per env, and per-env DOF values need EnvOptions(batch_dofs_info=True). "
            f"Fix: set EnvOptions(batch_dofs_info=True), or give every variant the same {attr_name} (set it on the "
            f"GroupedEntityOptions and remove it from the variants)."
        )

    def _warn_zero_gain_fallthrough(self, attr_name: str, user_dict: dict, parsed: torch.Tensor) -> None:
        """Warn about silent control-gain fallthrough for kp/kd.

        A partial override dict replaces the robot's default dict entirely, and unmatched DOFs fall back to the
        sim's current gain. For robots whose XML has no <actuator> block (e.g. YAM), that fallback is 0, leaving
        the DOF uncontrolled — commands to it silently do nothing. ``parsed`` is the one resolved row.
        """
        if attr_name not in ("default_dofs_kp", "default_dofs_kd") or parsed.ndim != 1:
            return
        matched_idx, _, _ = resolve_matching_names_values(
            user_dict, self.dofs_name, preserve_order=True, literal_names=self._every_joint_name
        )
        matched_set = set(matched_idx)
        unmatched_zero = [
            self.dofs_name[i]
            for i in range(len(self.dofs_name))
            if i not in matched_set and float(parsed[i].item()) == 0.0
        ]
        if unmatched_zero:
            gain_kind = "kp" if attr_name.endswith("kp") else "kd"
            en.logger.warning(
                f"`{attr_name}` resolves to {gain_kind}=0 for DOFs {unmatched_zero} "
                f"(not covered by the override dict and no sim-level default set). "
                f"Position commands to these DOFs will produce no torque. "
                f"If this is intentional (passive/free-spinning DOFs), you can ignore "
                f"this message; otherwise add entries for these DOFs to `{attr_name}`."
            )

    def _variant_rows(self, read: Callable[[int], torch.Tensor]) -> torch.Tensor | None:
        """``read(env)`` at the first env of each present variant, stacked; ``None`` unless heterogeneous."""
        if not self.is_heterogeneous:
            return None
        return torch.stack([read(env) for env in self._variant_first_envs])

    @property
    def is_kinematic(self) -> bool:
        """Whether this entity is kinematic: not simulated, and owning no collision geometry.

        A kinematic material puts the entity in Genesis' *kinematic* solver, which builds
        visual geoms only — ``KinematicEntity`` exposes ``vgeoms`` and has no ``geoms``
        attribute at all. Anything that reaches for collision geometry has nothing to work
        with on such an entity.
        """
        return isinstance(self.material, en.materials.KinematicMaterialOptions)

    def pre_build(self) -> None:
        self._downgrade_kinematic_collision_vis_mode()
        super().pre_build()
        self._refuse_variant_density_without_density_mass()
        self._validate_opt_in_friction_axes()
        self._apply_collision_link_mask()

    def _downgrade_kinematic_collision_vis_mode(self) -> None:
        """Render a kinematic entity's visual mesh when collision geometry was asked for.

        Genesis' rasterizer walks the kinematic solver's entities together with the rigid
        solver's and reads ``entity.geoms`` for every non-``"visual"`` vis mode
        (``rasterizer_context.on_rigid``), so a kinematic entity set to ``"collision"`` /
        ``"sdf"`` aborts ``scene.build()`` with ``AttributeError: 'KinematicEntity' object
        has no attribute 'geoms'``. That is a rasterizer blind spot, not a rejected config:
        ``Scene.add_entity`` accepts both modes for a kinematic material, and the raytracer
        guards the same access with an ``isinstance(entity, RigidEntity)`` check.

        Collision-vis tooling sweeps *whole* scenes (``eden view --collision``,
        ``examples/recipes/rendering/pickup.py``) and scenes legitimately mix rigid bodies
        with kinematic ones — goal markers, ReplicaCAD furniture. Drawing the visual mesh
        and naming the entity in a warning keeps those tools usable on such a scene; the
        authored options are left alone, so ``env.config`` still round-trips what the task
        declared. Modes Genesis itself rejects for a kinematic material (``"particle"``,
        ``"recon"``) are deliberately left to fail in ``add_entity``.
        """
        if not self.is_kinematic or self.vis_mode not in ("collision", "sdf"):
            return
        en.logger.warning(
            f"'{self.name}' is kinematic and has no collision geometry, so "
            f"vis_mode='{self.vis_mode}' has nothing to render; drawing its visual mesh instead."
        )
        self.vis_mode = "visual"

    def _validate_opt_in_friction_axes(self) -> None:
        """Reject torsional/rolling friction coefficients the scene has no axes to apply them on.

        Both axes are opt-in per scene (``EnvOptions.enable_torsional_friction`` /
        ``enable_rolling_friction`` add constraint rows to every contact). With the switch
        off Genesis stores the coefficient and never reads it, so a task could tune a value
        that does nothing. Raise here — before ``scene.build()`` — instead.
        """
        # Only a rigid material carries the coefficients: a kinematic one has no such field, and an
        # entity may carry no material options at all. Read them by attribute rather than by built-up
        # name so a field rename fails loudly here instead of silently voiding this check.
        # A heterogeneous entity's variants each count: their own materials and per-link coefficients.
        materials = [
            material
            for material in [self.material, *(self._variant_materials or [])]
            if isinstance(material, en.materials.RigidMaterialOptions)
        ]

        # Every variant, not only those the envs hold, so a config valid at one env count is valid at all.
        variant_values = self._options.variant_values() if self.is_heterogeneous else {}

        def set_anywhere(name: str) -> bool:
            return any(variant_values.get(name, [getattr(self, name)]))

        # (axis, every switch that must be on for the solver to read it, whether a material-wide
        # coefficient is set, whether per-link coefficients are). Rolling names the torsional switch too
        # because genesis rejects ``enable_rolling_friction`` on its own — naming only the rolling switch
        # would hand the caller a fix that raises again one build later.
        axes = (
            (
                "torsional",
                ("enable_torsional_friction",),
                any(material.friction_torsional is not None for material in materials),
                set_anywhere("default_geoms_friction_torsional"),
            ),
            (
                "rolling",
                ("enable_rolling_friction", "enable_torsional_friction"),
                any(material.friction_rolling is not None for material in materials),
                set_anywhere("default_geoms_friction_rolling"),
            ),
        )

        # Report both axes at once: a build is expensive enough that fixing one and rediscovering the
        # other is a wasted round trip.
        problems = []
        for axis, switches, material_set, geoms_patterns in axes:
            missing = [switch for switch in switches if not getattr(self._env.env_options, switch)]
            if not missing:
                continue
            set_by = []
            if material_set:
                set_by.append(f"`material.friction_{axis}`")
            if geoms_patterns:
                set_by.append(f"`default_geoms_friction_{axis}`")
            if set_by:
                needs = ", ".join(f"`EnvOptions.{switch}=True`" for switch in missing)
                problems.append(f"{' and '.join(set_by)} set but {axis} friction is disabled (needs {needs})")

        if problems:
            raise ValueError(
                f"Entity '{self.name}': {'; '.join(problems)}. Enable the switch to give every contact the "
                f"constraint axis the coefficient scales, or drop the coefficient."
            )

    def _apply_collision_link_mask(self) -> None:
        """Restrict collision to geoms on links matching ``collision_link_patterns``.

        Runs after the entity is added to the scene but before ``scene.build()``,
        because Genesis bakes the collision-pair candidate list from per-geom
        ``contype``/``conaffinity`` at build time. Geoms on matching links keep
        collision (``contype = conaffinity = 1``, optionally with
        ``collision_friction``); every other link's geoms are disabled
        (``contype = conaffinity = 0``).

        Genesis defaults ``is_local_collision_mask=True`` for MJCF entities, which
        makes ``contype``/``conaffinity`` a *local self-collision* mask only — so
        zeroing them would NOT stop collisions with other entities (e.g. a grasped
        cube). We set it ``False`` so the mask filters cross-entity pairs (see
        ``collider.py``: ``con_skip = (same_entity | ~has_local_mask) & (con_match == 0)``).
        """
        patterns = getattr(self._options, "collision_link_patterns", None)
        if not patterns:
            return
        if self.is_kinematic:
            en.logger.warning(
                f"collision_link_patterns {patterns} ignored on kinematic entity '{self.name}': "
                f"kinematic entities have no collision geometry to mask."
            )
            return
        geoms = self._entity.geoms
        unique_links = sorted({g.link.name for g in geoms})
        _, matched = resolve_matching_names(patterns, unique_links, preserve_order=True)
        keep_links = set(matched)
        if not keep_links:
            en.logger.warning(
                f"collision_link_patterns {patterns} matched no links on '{self.name}' "
                f"(links: {unique_links}); leaving collision unchanged."
            )
            return
        # Treat contype/conaffinity as a global (cross-entity) filter, not a local one.
        self._entity._is_local_collision_mask = False
        fric = self._options.collision_friction
        for g in geoms:
            if g.link.name in keep_links:
                g._contype = 1
                g._conaffinity = 1
                if fric is not None:
                    g.set_friction(fric)
            else:
                g._contype = 0
                g._conaffinity = 0
        en.logger.info(
            f"[collision mask] '{self.name}': only links {sorted(keep_links)} collide "
            f"({len(keep_links)}/{len(unique_links)} links)."
        )

    @property
    def _needs_explicit_damping_armature(self) -> bool:
        """Whether the PD derivative gain must be folded into the armature for stability.

        Genesis-internal API: reads ``rigid_solver._integrator``. Only ``Euler`` integrates the PD
        controller explicitly. ``approximate_implicitfast`` handles it implicitly, and so does
        ``implicitfast``, whose implicit-damping pass takes the position/velocity actuators' ``kv``
        into ``M + h*D`` exactly as MuJoCo's does (``func_implicit_damping``); folding ``kd*dt`` in
        under either would only inflate the effective inertia, and count the damping twice.

        Both implicit passes see only the ``kv`` Genesis holds for a DOF in position or velocity mode.
        A torque-mode action term (``control_dofs_force``: ``ExplicitPDController``,
        ``OperationalSpaceController``, ...) computes ``-kd*qdot`` itself, so its damping is explicit
        under every integrator. This gate is per entity and cannot tell which term drives a DOF, so
        such a term is folded under ``Euler`` and left unstabilized under the implicit integrators.
        """
        return self._env.rigid_solver._integrator == gs.integrator.Euler

    @gs.assert_built
    def post_build(self) -> None:
        # First: every step below branches on ``is_fixed_base``, so a declaration the asset
        # contradicts must stop the build before any of them runs on the wrong branch.
        self._check_mjcf_base_matches_is_fixed_base()

        is_kinematic = self.is_kinematic
        self._build_dof_properties(is_kinematic)
        self._init_qs_idx_local()
        self._init_control_gains_and_damping(is_kinematic)
        self._apply_dofs_spring_ref(is_kinematic)
        self._apply_constraint_sol_params()
        self._apply_variant_densities()

        # Set initial root pose. Skipped for a welded child: ``attach`` already consumed its
        # ``default_root_pos``/``_quat`` as the mounting transform and made its world pose FK
        # output, so there is no root pose left to write here.
        if not self._is_attached:
            self._set_initial_root_pose()

        if self.is_support_enabled and is_kinematic:
            # Same shape as _apply_constraint_sol_params' kinematic guard: warn and skip rather
            # than drop the configuration silently. A kinematic entity produces no contacts, so
            # nothing could rest on the surface anyway, and the option can arrive by inheritance
            # (a ghost of a support-enabled table) rather than by intent.
            en.logger.warning(
                f"is_support_enabled ignored on kinematic entity '{self.name}': support surfaces "
                f"are extracted for contact-producing rigid entities only."
            )
        elif self.is_support_enabled:
            self._prepare_support(
                minimum_clearance=self.support_minimum_clearance,
                pre_sample_points=True,
                pre_num_sample_points=self.support_num_sample_points,
                pre_shrink=self.support_shrink,
                pre_sample_method=self.support_sample_method,
                pre_grid_size=self.support_grid_size,
                support_links_name=self.support_links_name,
            )

    def _build_dof_properties(self, is_kinematic: bool) -> None:
        """Resolve DOF ordering and batch-aware default DOF properties.

        (Named ``_build_dof_properties`` to avoid confusion with the per-property builder
        ``_init_dofs_property``, which it calls repeatedly.) Reads as an outline: articulated/
        actuated entities resolve DOF ordering then fill per-DOF defaults — positions always,
        and the simulated dynamics/limits only when non-kinematic; everything else clears them
        to ``None``.
        """
        if not (self.is_articulated or self.is_actuated):
            self._clear_dof_properties()
            return

        self._resolve_dof_ordering()
        self._init_dof_pos_defaults()
        if not is_kinematic:
            self._init_dof_dynamics_defaults()
            self._init_soft_pos_limits()
            self._init_actuator_spec_props()
            self._init_dof_force_limits()

    def _resolve_dof_ordering(self) -> None:
        """Resolve dofs_name order, the name/index maps, and the Jacobian column permutation."""
        self.dofs_name, dofs_idx_local = self.find_named_dofs_idx_local(self.dofs_name, preserve_order=True)
        self.dofs_idx_map = {idx: i for i, idx in enumerate(dofs_idx_local)}
        self.dofs_idx_local = torch.as_tensor(dofs_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        # What every ``dofs_idx_local=None`` accessor below hands Genesis: the same DOFs, as a ``range``
        # when they are contiguous (the common case), which is Genesis' cheap, sync-free path.
        self._genesis_dofs_idx = as_genesis_index(self.dofs_idx_local)

        # -- Jacobian column reorder: Genesis kinematic-tree → dofs_name --
        # get_jacobian() returns columns in Genesis kinematic-tree order (BFS in
        # genesis 1.1.x, DFS in 1.2.0), but all other DOF APIs (set_qpos,
        # get_dofs_limit, …) use dofs_name order.  Build a permutation so
        # get_jacobian() returns columns in dofs_name order. Derived from the
        # name-resolved dofs_idx_local, so it adapts to either internal order.
        #
        # ``has_immobile_base`` (not ``is_fixed_base``) is the right question: Genesis
        # ``attach`` removes a welded child's 6 base DOFs, so its Jacobian has no base
        # columns either, and a floating-base ``arange(6)`` prefix would make the permutation
        # over-long on a welded gripper/hand (as in ``_init_qs_idx_local``).
        if self.has_immobile_base:
            self._jac_perm = self.dofs_idx_local.clone()
        else:
            self._jac_perm = torch.cat(
                [
                    torch.arange(6, dtype=gs.tc_int, device=self.device),
                    self.dofs_idx_local,
                ]
            )

    def _init_dof_pos_defaults(self) -> None:
        """Initialize the default DOF positions and (hard) position limits.

        Always present for articulated/actuated entities, kinematic or not.
        """
        dofs_pos_init = self.get_dofs_pos(envs_idx=0)
        self.default_dofs_pos = self._init_dofs_property(
            "default_dofs_pos",
            "set_dofs_pos",
            default_info=dofs_pos_init.squeeze(0),
            force_broadcast=True,
            # Each variant starts from its own asset's initial joint positions.
            variant_default_info=self._variant_rows(lambda env: self.get_dofs_pos(envs_idx=env).squeeze(0)),
        )
        assert self.default_dofs_pos.ndim == 2, f"default_dofs_pos.shape: {self.default_dofs_pos.shape}"

        # --- pos limits (always read, no setter) ---
        def pos_limits_at(env: int) -> torch.Tensor:
            return torch.stack(self.get_dofs_limit(envs_idx=env), dim=-1).squeeze(0)

        self.default_dofs_pos_limits = self._init_dofs_property(
            "default_dofs_pos_limits",
            None,
            default_info=pos_limits_at(0),
            # Each variant's own range, which Genesis simulates per env
            # (https://github.com/Genesis-Embodied-AI/Genesis/pull/3497).
            variant_default_info=self._variant_rows(pos_limits_at),
        )
        assert self.default_dofs_pos_limits.ndim == 3, (
            f"default_dofs_pos_limits.shape: {self.default_dofs_pos_limits.shape}"
        )

    def _init_dof_dynamics_defaults(self) -> None:
        """Initialize the simulated per-DOF dynamics defaults: zero velocity plus batch-aware
        stiffness / damping / armature from each DOF's actuator spec.
        """
        # --- dofs_vel: zero default, always broadcast to all envs ---
        self.default_dofs_vel = self._init_dofs_property(
            "default_dofs_vel",
            "set_dofs_vel",
            default_info=torch.zeros(self.num_dofs, device=self.device),
            force_broadcast=True,
        )

        # --- batch-aware DOF properties ---
        # Each defaults to what the sim holds, read at env 0 or, for a heterogeneous entity, at the first
        # env of each variant: Genesis simulates every variant with its own asset's joint parameters and
        # default armature (https://github.com/Genesis-Embodied-AI/Genesis/pull/3497).
        # The info getters always carry the env axis, so ``envs_idx`` reads one row whatever
        # ``batch_dofs_info`` says; ``force_broadcast`` still tracks it, because that governs the *setter* shape.
        batch_info = self.has_batched_dofs_info

        def init_dynamics(attr_name: str, setter_name: str, getter: Callable, spec_values: dict) -> torch.Tensor:
            def default_at(env: int) -> torch.Tensor:
                return self._parse_dofs_info(spec_values, default_info=getter(envs_idx=env).squeeze(0))

            return self._init_dofs_property(
                attr_name,
                setter_name,
                default_info=default_at(0),
                force_broadcast=batch_info,
                variant_default_info=self._variant_rows(default_at),
            )

        specs = self.dofs_spec.items()
        self.default_dofs_stiffness = init_dynamics(
            "default_dofs_stiffness", "set_dofs_stiffness", self.get_dofs_stiffness, {k: s.STIFFNESS for k, s in specs}
        )
        self.default_dofs_damping = init_dynamics(
            "default_dofs_damping", "set_dofs_damping", self.get_dofs_damping, {k: s.DAMPING for k, s in specs}
        )
        self.default_dofs_armature = init_dynamics(
            "default_dofs_armature", "set_dofs_armature", self.get_dofs_armature, {k: s.ARMATURE() for k, s in specs}
        )
        self.default_dofs_frictionloss = init_dynamics(
            "default_dofs_frictionloss", "set_dofs_frictionloss", self.get_dofs_frictionloss, {}
        )

    def _init_soft_pos_limits(self) -> None:
        """Shrink the hard joint range to a soft range (mjlab/IsaacLab parity)."""
        # By 0.9 (mjlab/IsaacLab's soft_joint_pos_limit_factor) about the midpoint: the threshold the
        # dofs_pos_limits penalty is computed against, rather than the hard limits.
        # One row, or one per env when a heterogeneous entity's variants differ in their hard limits.
        _hard_limits = self.default_dofs_pos_limits.clone()
        _mid = 0.5 * (_hard_limits[..., 0] + _hard_limits[..., 1])
        _half_range = 0.5 * (_hard_limits[..., 1] - _hard_limits[..., 0])
        _soft_factor = 0.9
        _soft_limits = torch.stack([_mid - _soft_factor * _half_range, _mid + _soft_factor * _half_range], dim=-1)
        # An unbounded DOF (ball joint, or a hinge with no model ``range``) has -/+inf hard limits,
        # so the arithmetic above is ``inf - inf`` = NaN — which reaches the ``dofs_pos_limits``
        # reward/termination and the ``reset_dofs_by_*`` clamps, where -/+inf is instead benign.
        # Tested per *row*: a half-bounded DOF has an infinite half-range too, so it passes through
        # whole rather than carrying a 0.9 margin on its finite side.
        _unbounded = ~_hard_limits.isfinite().all(dim=-1, keepdim=True)
        _soft_limits = torch.where(_unbounded, _hard_limits, _soft_limits)
        self.soft_dofs_pos_limits = self._init_dofs_property(
            "soft_dofs_pos_limits",
            None,
            default_info=_soft_limits[0],
            variant_default_info=None if _soft_limits.shape[0] == 1 else _soft_limits[self._variant_first_envs],
        )

    def _init_actuator_spec_props(self) -> None:
        """Stash each actuator-spec scalar (torque/speed/friction limits) as a ``_<name>`` tensor."""
        _SPEC_PROPS = [
            "FULL_TORQUE_SPEED",
            "NO_LOAD_SPEED",
            "DRIVING_TORQUE_LIMIT",
            "BRAKING_TORQUE_LIMIT",
            "STATIC_FRICTION",
            "DYNAMIC_FRICTION",
            "FRICTION_ACTIVATION_SPEED",
        ]

        for key in _SPEC_PROPS:
            parsed = self._parse_dofs_info(
                {dofs_key: getattr(spec, key) for dofs_key, spec in self.dofs_spec.items()},
                default_info=torch.zeros(self.num_dofs, device=self.device) + getattr(ActuatorSpecOptions, key),
            ).unsqueeze(0)
            setattr(self, f"_{key.lower()}", parsed)

    def _init_dof_force_limits(self) -> None:
        """Initialize per-DOF effort limits and apply them to the sim as a symmetric force range."""
        spec_effort_limit = {key: spec.EFFORT_LIMIT for key, spec in self.dofs_spec.items()}

        def force_limits_at(env: int) -> torch.Tensor:
            return self._parse_dofs_info(
                spec_effort_limit, default_info=self.get_dofs_force_range(envs_idx=env)[1].squeeze(0)
            )

        self.default_dofs_force_limits = self._init_dofs_property(
            "default_dofs_force_limits",
            None,
            default_info=force_limits_at(0),
            apply_to_sim=False,
            # Each variant's own effort limit, which Genesis simulates per env
            # (https://github.com/Genesis-Embodied-AI/Genesis/pull/3497).
            variant_default_info=self._variant_rows(force_limits_at),
        )
        # The solver's flag: Genesis may batch a heterogeneous entity's DOF info whatever the option says.
        if self.has_batched_dofs_info:
            upper_limits = self.default_dofs_force_limits.expand(self._env.num_envs, -1).contiguous()
        else:
            if self.default_dofs_force_limits.shape[0] > 1:  # a heterogeneous entity's variants differ
                self._require_batched_dofs_info("default_dofs_force_limits")
            upper_limits = self.default_dofs_force_limits[0]
        self.set_dofs_force_range(-upper_limits, upper_limits)

    def _clear_dof_properties(self) -> None:
        """Null out all DOF-property attributes for entities without articulated/actuated DOFs."""
        self.dofs_idx_map = {}
        self.dofs_idx_local = None
        self._genesis_dofs_idx = None
        self.__dict__.pop("_first_geom_idx_cache", None)
        self._jac_perm = None
        # Every per-DOF tensor ``post_build`` would have resolved, from the one canonical list —
        # so a field added to ``_DOF_FIELD_NAMES`` is cleared here without a second edit.
        for field_name in RESOLVED_DOF_TENSOR_FIELD_NAMES:
            setattr(self, field_name, None)
        # actuator spec properties
        self._full_torque_speed = None
        self._no_load_speed = None
        self._driving_torque_limit = None
        self._braking_torque_limit = None
        self._static_friction = None
        self._dynamic_friction = None
        self._friction_activation_speed = None

    def _init_qs_idx_local(self) -> None:
        """Map DOF indices to q-coordinate indices (a floating base prepends 7 root qs).

        Gated on ``has_immobile_base``, not ``is_fixed_base``: Genesis ``attach`` removes
        ALL root joints of a non-fixed base link (the child is carried by the parent
        link), so the 7 root qs no longer exist after the weld. The floating-base +7
        offsets would make ``get_qpos``/``set_qpos`` on a welded gripper/hand index past
        the entity's q range (a CUDA device-side assert on GPU).
        """
        if self.has_immobile_base:
            self.qs_idx_local = self.dofs_idx_local
        else:
            if self.dofs_idx_local is not None:
                self.qs_idx_local = (
                    torch.cat(
                        [torch.arange(7, device=self.dofs_idx_local.device), self.dofs_idx_local + 1]
                    )  # why + 1: dofs_idx_local has 6 base free dofs -> to add 7 offset, we just add 1
                    .to(self.device)
                    .contiguous()
                )
            else:
                self.qs_idx_local = torch.arange(7, device=self.device)
        # The ``qs_idx_local=None`` counterpart of ``_genesis_dofs_idx``.
        self._genesis_qs_idx = None if self.qs_idx_local is None else as_genesis_index(self.qs_idx_local)

    def _hinge_qs(self, dofs_idx_local) -> range | None:
        """The local q ``range`` of a ``range`` of hinge / slide DOFs whose coordinates follow in order, else ``None``.

        For these joints Genesis keeps ``qpos = qpos0 + dofs_pos``, so the DOFs' positions can be written
        as those coordinates (:meth:`set_dofs_pos`). A free or ball joint's DOFs have none of their own.
        """
        if not isinstance(dofs_idx_local, range) or not dofs_idx_local:
            return None
        memo = self.__dict__.setdefault("_hinge_qs_memo", {})
        key = dofs_idx_local  # a range hashes by value
        if key not in memo:
            scalar = (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
            scalar_qs = {joint.dof_start: joint.q_start for joint in self._entity.joints if joint.type in scalar}
            dof_start, q_start = self.dof_start, self._entity.q_start
            qs = [scalar_qs.get(dof_start + dof) for dof in dofs_idx_local]
            first = qs[0]
            in_order = None not in qs and qs == list(range(first, first + len(qs)))
            memo[key] = range(first - q_start, first - q_start + len(qs)) if in_order else None
        return memo[key]

    @cached_property
    def root_qpos_is_authored_pose(self) -> bool:
        """Whether the free base's 7 root coordinates *are* its authored pose (``get_pos/get_quat(relative=True)``).

        They are whenever Genesis moves the root link by no morph offset or inertial alignment, so a
        ``relative=True`` pose can be written straight into qpos with :meth:`set_qpos` — Genesis' bool-mask
        path, no GPU->CPU sync — where ``set_pos`` / ``set_quat`` would go through the offset composition.
        Read once, comparing Genesis' own relative and solver-frame getters (one sync, on first use).
        ``False`` for an entity with no free base, and under IPC coupling, where Genesis refuses a qpos
        write to an IPC-articulated entity that its pose setters may still take.
        """
        if self.has_immobile_base or self._env._has_ipc_coupler:
            return False
        authored_quat, solver_quat = self.get_quat(relative=True), self.get_quat()
        same_pos = torch.allclose(self.get_pos(relative=True), self.get_pos(), atol=1e-6, rtol=0)
        # Equal up to the sign a quaternion carries for free: q or -q as a whole, not component by component.
        quat_gap = torch.minimum(
            (authored_quat - solver_quat).abs().amax(-1), (authored_quat + solver_quat).abs().amax(-1)
        )
        return same_pos and bool((quat_gap < 1e-6).all())

    def _qpos0_rows(self, qs: range, envs_idx, n_rows: int) -> torch.Tensor:
        """``qpos0`` of the local ``qs`` coordinates, as rows matching a value of ``n_rows`` written to ``envs_idx``.

        Every env's row alongside a bool mask, unless the value holds just the selected rows, which are
        gathered through the mask's ids (free for the auto-reset's mask: :func:`~eden.utils.misc.sharing_env_indices`).
        """
        q_start = self._entity.q_start
        qpos0 = _qpos0(self.solver)[:, q_start + qs.start : q_start + qs.stop]
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        if isinstance(envs_idx, slice):
            return qpos0[envs_idx]
        if envs_idx.dtype == torch.bool and n_rows in (1, self._env.num_envs):
            return qpos0  # every env's row, the mask picking which are written
        return qpos0[env_indices(envs_idx)]

    def _init_control_gains_and_damping(self, is_kinematic: bool) -> None:
        """Read PD control gains (kp/kd) and fold implicit damping into the armature.

        Actuated entities read kp/kd from the solver and (for explicit integrators only) add
        ``kd*substep_dt`` to the armature; non-actuated floating bases instead get a small
        damping bump on their free joints for numerical stability.
        """
        if not is_kinematic and self.is_actuated:
            # NOTE: kp, kd is for control (vs. stiffness/damping for simulation)
            def _read_dof_gains(getter, attr_name: str, env: int) -> torch.Tensor:
                """Read env ``env``'s per-DOF kp/kd from the solver, falling back to zeros on rejection.

                The solver rejects the read when, e.g., MJCF tendon-approximated
                actuators are not PD-reducible.

                The returned tensor is the base for ``_parse_dofs_info``; any DOFs
                listed in the user's ``default_dofs_{kp,kd}`` dict overwrite the
                zeros in place. A zero gain on an unspecified DOF matches the
                passive-follower semantics of tendon-driven joints.
                """
                try:
                    return getter(envs_idx=env).squeeze(0)
                except gs.GenesisException as e:
                    if "non-PD-reducible" not in str(e):
                        raise
                    en.logger.debug(
                        f"Entity '{self.name}': non-PD-reducible actuator detected; "
                        f"using zero fallback for `{attr_name}` where not explicitly set."
                    )
                    return torch.zeros(len(self.dofs_name), device=self.device)

            # Each variant's own asset gains, which Genesis simulates per env
            # (https://github.com/Genesis-Embodied-AI/Genesis/pull/3497).
            self.default_dofs_kp = self._init_dofs_property(
                "default_dofs_kp",
                "set_dofs_kp",
                default_info=_read_dof_gains(self.get_dofs_kp, "default_dofs_kp", 0),
                force_broadcast=self.has_batched_dofs_info,
                variant_default_info=self._variant_rows(
                    lambda env: _read_dof_gains(self.get_dofs_kp, "default_dofs_kp", env)
                ),
            )
            self.default_dofs_kd = self._init_dofs_property(
                "default_dofs_kd",
                "set_dofs_kd",
                default_info=_read_dof_gains(self.get_dofs_kd, "default_dofs_kd", 0),
                force_broadcast=self.has_batched_dofs_info,
                variant_default_info=self._variant_rows(
                    lambda env: _read_dof_gains(self.get_dofs_kd, "default_dofs_kd", env)
                ),
            )

            # Emulate implicit damping of the PD controller's derivative gain by folding `kd * substep_dt` into the
            # armature — a first-order correction (M_eff = M + kd*dt) that makes the otherwise-explicit `-kd*qd` term
            # unconditionally stable. Only needed under `Euler`, the one integrator that integrates the actuator
            # explicitly: `approximate_implicitfast` and `implicitfast` already handle the PD controller's damping
            # implicitly, so adding this there only inflates effective inertia (see `_needs_explicit_damping_armature`).
            effective_armature = self.default_dofs_armature
            if self._needs_explicit_damping_armature:
                effective_armature = effective_armature + self.default_dofs_kd * self._env.scene.sim._substep_dt
            assert effective_armature.ndim == 2
            self.set_dofs_armature(effective_armature.squeeze(0))
        elif not is_kinematic and self._entity.n_dofs > 0 and not self.has_immobile_base:
            self._stabilize_free_joints()

    def _stabilize_free_joints(self) -> None:
        """Add a little damping to a non-actuated floating base's free joints for numerical stability.

        A welded child has no free joints (``attach`` removed its root joint), so it takes the
        same no-op path as a fixed base rather than searching for root DOFs. The fully implicit
        Mochi solver needs no such bump (``MochiRigidEntity`` overrides this to a no-op).
        """
        free_dofs_idx = [idx for joint in self.free_joints for idx in joint.dofs_idx_local]
        if len(free_dofs_idx) > 0:
            # Every env's row when DOF info is per env (a heterogeneous entity's variants each bring their
            # own, https://github.com/Genesis-Embodied-AI/Genesis/pull/3497), else the one shared row: the setters take
            # the per-mode shape.
            damping = self.get_dofs_damping(dofs_idx_local=free_dofs_idx) + 0.0002
            self.set_dofs_damping(damping if self.has_batched_dofs_info else damping[0], dofs_idx_local=free_dofs_idx)

    def _apply_dofs_spring_ref(self, is_kinematic: bool) -> None:
        """Re-anchor joint springs at ``default_dofs_spring_ref`` instead of the initial qpos.

        Genesis ignores MJCF ``springref`` (``mj.qpos_spring``) and anchors every joint
        spring at ``qpos0``, silently dropping any spring preload — e.g. the Robotiq
        2F85's spring links (``stiffness=0.05, springref=2.62``) lose the +0.13 N*m
        preload that keeps the passive pad chain taut. Re-anchoring is implemented by
        moving the spring into the position servo: the sim spring stiffness is zeroed
        and the DOF gets ``kp = stiffness`` with the reference as its position target
        (``tau = stiffness * (ref - q)`` — exactly MuJoCo's spring force). The control
        target persists across steps and resets; these DOFs must not also be driven by
        an action term.
        """
        if not self.default_dofs_spring_ref:
            return
        if is_kinematic or self.default_dofs_stiffness is None:
            raise ValueError(
                f"Entity '{self.name}': `default_dofs_spring_ref` requires a simulated articulated "
                "entity (kinematic entities have no joint springs)."
            )
        idx, names, refs = resolve_matching_names_values(
            self.default_dofs_spring_ref, self.dofs_name, preserve_order=True, literal_names=self._every_joint_name
        )
        dofs_idx_local = self.dofs_idx_local[idx]
        # One value per DOF, unless a heterogeneous entity's variants differ in it: then one row per env, which
        # the stiffness write already required batched DOF info for.
        stiffness = self.default_dofs_stiffness[:, idx]
        if bool((stiffness == stiffness[:1]).all()):
            stiffness = stiffness[0]
        if bool((stiffness <= 0).any()):
            slack = [name for name, zero in zip(names, (stiffness <= 0).reshape(-1, len(names)).any(0)) if zero]
            raise ValueError(
                f"Entity '{self.name}': `default_dofs_spring_ref` set for DOFs {slack} whose spring "
                "stiffness is zero — set the joint `stiffness` in the asset or `default_dofs_stiffness`."
            )
        self.set_dofs_stiffness(torch.zeros_like(stiffness), dofs_idx_local=dofs_idx_local)
        self.default_dofs_stiffness[..., idx] = 0.0
        self.set_dofs_kp(stiffness, dofs_idx_local=dofs_idx_local)
        if self.default_dofs_kp is not None:
            if stiffness.ndim == 2 and self.default_dofs_kp.shape[0] == 1:
                self.default_dofs_kp = self.default_dofs_kp.repeat(self._env.num_envs, 1)
            self.default_dofs_kp[..., idx] = stiffness
        targets = torch.as_tensor(refs, dtype=gs.tc_float, device=self.device)
        self.control_dofs_pos(targets, dofs_idx_local=dofs_idx_local)

    def _apply_constraint_sol_params(self) -> None:
        """Apply the entity's declared equality/geom constraint-solver parameters and geom friction.

        All come from options and are written once at build. Genesis clamps every
        solref ``timeconst`` to ``2 * substep_dt`` (with a warning), so a stiff value
        only takes full effect when ``sim_dt / sim_substeps`` is small enough. Contact
        pairs mix the *mean* of the two geoms' parameters (geom ``priority`` is not
        honored), so partner geoms should be stiffened too — see
        ``default_geoms_sol_params`` in the options docs. Friction, in contrast, combines as
        ``max(a, b)`` on all three axes, so one side of the pair carrying the value is enough.

        Every write below goes through ``self.geoms`` or ``self._entity.equalities``, neither of
        which a kinematic entity has (see :attr:`is_kinematic`), so a kinematic entity carrying any
        of these options is warned about and skipped rather than aborting the build with a bare
        ``AttributeError``. Skipping loses nothing — the kinematic solver produces no contacts and
        holds no constraints — and the options can arrive by inheritance rather than by intent: a
        ghost (``GHOST_DEFAULTS`` puts a kinematic material on a robot's own options class) of a
        gripper that declares ``default_geoms_sol_params`` is a legitimate scene.
        """
        if self.is_kinematic:
            inapplicable = [
                name for name in RIGID_CONSTRAINT_FIELD_NAMES if getattr(self, name) or name in self._variant_fields
            ]
            if inapplicable:
                en.logger.warning(
                    f"{inapplicable} ignored on kinematic entity '{self.name}': kinematic entities "
                    f"have no contacts or constraints to apply them to."
                )
            return

        if self.default_equalities_sol_params is not None:
            equalities = self._entity.equalities
            if len(equalities) == 0:
                raise ValueError(
                    f"Entity '{self.name}': `default_equalities_sol_params` is set but the entity "
                    "has no equality constraints."
                )
            for equality in equalities:
                equality.set_sol_params(self.default_equalities_sol_params)

        # A heterogeneous entity's variants may each have their own material friction, which Genesis could not
        # build (it takes one material per entity); written first, so the per-link coefficients below win, as
        # they win over the material's for any entity.
        self._apply_variant_material_friction()

        for patterns, geoms, scope in self._geom_field_targets("default_geoms_sol_params"):
            for pattern, sol_params in patterns.items():
                matched = self._match_geoms("default_geoms_sol_params", pattern, geoms, scope)
                self._env.set_sol_params(sol_params, geoms_idx=[geom.idx for geom in matched])

        # The three per-geom friction coefficients: sliding (unitless), plus the opt-in torsional and
        # rolling ones (in meters, and only read by the solver when the matching EnvOptions switch is
        # on — checked in ``_validate_opt_in_friction_axes``).
        for option_name, setter_name in _GEOM_FRICTION_SETTERS.items():
            for patterns, geoms, scope in self._geom_field_targets(option_name):
                for pattern, value in patterns.items():
                    for geom in self._match_geoms(option_name, pattern, geoms, scope):
                        getattr(geom, setter_name)(value)

    def _match_geoms(
        self, option_name: str, pattern: str, geoms: list[RigidGeom] | None = None, scope: str = ""
    ) -> list[RigidGeom]:
        """Collision geoms among ``geoms`` (default: all) on the links matching ``pattern``; raises if none carry geoms.

        ``scope`` names the variants ``geoms`` belong to, for the error.
        """
        geoms = self.geoms if geoms is None else geoms
        link_names, _ = self.find_named_links_idx_local(pattern)
        matched_links = set(link_names)
        matched = [g for g in geoms if g.link.name in matched_links]
        if not matched:
            raise ValueError(
                f"Entity '{self.name}': `{option_name}` pattern '{pattern}' matches "
                f"links {sorted(matched_links)} but none carry collision geoms{scope}; links with geoms: "
                f"{sorted({g.link.name for g in geoms})}."
            )
        return matched

    def _variant_geoms(self, variant: int) -> list[RigidGeom]:
        """The collision geoms Genesis built for ``variant`` of a heterogeneous entity, active in its envs only."""
        geoms = []
        for link in self._entity.links:
            start, end = link._variant_geom_ranges[variant]
            geoms.extend(geom for geom in link.geoms if start <= geom.idx < end)
        return geoms

    def _geom_field_targets(self, name: str) -> list[tuple[object, list[RigidGeom], str]]:
        """``(value, geoms, scope)`` for geom field ``name``: its value over every geom, or, when a heterogeneous
        entity's variants differ in it, each distinct value over the geoms of the variants holding it.
        """
        values = self._variant_fields.get(name)
        if values is None:
            return [(getattr(self, name), self.geoms, "")]
        variants_by_value: list[tuple[object, list[int]]] = []
        for variant, value in enumerate(values):
            for held, variants in variants_by_value:
                if held == value:
                    variants.append(variant)
                    break
            else:
                variants_by_value.append((value, [variant]))
        return [
            (
                value,
                [geom for variant in variants for geom in self._variant_geoms(variant)],
                f" in scene_options.{self.name}.grouped_entities {variants}",
            )
            for value, variants in variants_by_value
        ]

    def _apply_variant_material_friction(self) -> None:
        """Write each variant's own material friction coefficients over its geoms (``GroupedEntityOptions``)."""
        if self._variant_materials is None:
            return
        for variant, material in enumerate(self._variant_materials):
            for name, setter_name in _MATERIAL_FRICTION_SETTERS.items():
                value = getattr(material, name)
                if value is not None and value != getattr(self.material, name):
                    for geom in self._variant_geoms(variant):
                        getattr(geom, setter_name)(value)

    def _apply_variant_densities(self) -> None:
        """Give each env's variant of a heterogeneous entity its own density (``GroupedEntityOptions``).

        Genesis built every variant at the group's density. A mesh or primitive variant's mass and inertia are
        linear in its density (``_refuse_variant_density_without_density_mass`` holds every variant with its own
        to those), so scaling each env's mass by ``variant_rho / group_rho``, inertia along with it, is the body
        it would have built.
        """
        if self._variant_materials is None or self.is_kinematic:
            return
        group_rho = self.material.rho
        rhos = [material.rho for material in self._variant_materials]
        if all(rho == group_rho for rho in rhos):
            return
        if all(link.is_fixed for link in self._entity.links):
            # A body fixed to the world moves no mass, so its density is never read -- as for an ungrouped fixed
            # entity's ``rho`` -- and Genesis' ``set_mass`` refuses an entity with no moving link.
            return
        # Per env: Genesis stores a heterogeneous scene's link info per env whatever ``batch_links_info`` says.
        scale = torch.tensor(rhos, dtype=gs.tc_float, device=self.device)[self._env_variant_idx] / group_rho
        self._entity.set_mass(self._entity.get_mass() * scale)

    def _refuse_variant_density_without_density_mass(self) -> None:
        """Raise before the build if a variant with its own ``rho`` is not a mesh or primitive.

        Genesis computes a mesh's or primitive's mass from its density, but a URDF / MJCF / USD body's from the
        inertial its file authors, which no density changes, so ``_apply_variant_densities`` could not scale it.
        """
        if self._variant_materials is None or self.is_kinematic:
            return
        for variant, (material, morph) in enumerate(zip(self._variant_materials, self.morph)):
            if material.rho != self.material.rho and not isinstance(morph, (gs.morphs.Mesh, gs.morphs.Primitive)):
                options = self._options.grouped_entities[variant]
                raise ValueError(
                    f"scene_options.{self.name}.grouped_entities[{variant}] ({type(options).__name__}) has its own "
                    f"rho={material.rho}, but it loads as a {type(morph).__name__}, whose mass Genesis reads from "
                    f"the inertial its file authors rather than computing it from a density, so the per-variant "
                    f"density rescale cannot apply to it. Fix: remove rho from that variant's material and set the "
                    f"body's mass in its asset, or load the variant from a mesh file (.obj / .stl / .glb) instead."
                )

    def _check_mjcf_base_matches_is_fixed_base(self) -> None:
        """Raise when an MJCF's base link disagrees with the declared ``is_fixed_base``.

        URDF / USD / mesh morphs take ``fixed=``, so they are built to match the flag. An MJCF is not:
        Genesis has no ``fixed`` switch for it, and the file's own joints decide. Eden reads the flag
        anyway — ``has_immobile_base``, ``qs_idx_local``, the Jacobian column order and the initial-pose
        write all branch on it — so a mismatch builds an entity whose q indices run past its q range.

        The question is asked of the *base link*, by joint type: Genesis' root-pose setters address
        that link alone, so a free joint elsewhere in a multi-body file is not a floating base Eden
        can pose. A base link carrying a free joint needs ``is_fixed_base=False``; one that is welded
        to the world, or moved by any other joint of its own (hinge, slide or ball), needs ``True``.
        A welded child is skipped: ``attach`` removed its base joints, so it has no floating base
        whatever it declares.
        """
        if self._is_attached or not isinstance(self.morph, gs.morphs.MJCF):
            return
        base_link = self._entity.base_link
        base_joints = [joint for joint in base_link.joints if joint.type != gs.JOINT_TYPE.FIXED]
        has_free_base = any(joint.type == gs.JOINT_TYPE.FREE for joint in base_joints)
        if has_free_base != self.is_fixed_base:
            return

        prefix = f"Entity '{self.name}': `is_fixed_base={self.is_fixed_base}`, but the base link '{base_link.name}'"
        file = self._options.file
        if self.is_fixed_base:
            raise ValueError(
                f"{prefix} of {file!r} carries a free joint, and Genesis has no `fixed` switch for MJCF, so the "
                f"base would float while Eden treats it as welded. Declare `is_fixed_base=False`, or remove the "
                f"free joint from the MJCF (or convert the asset to URDF) to pin it."
            )
        if base_joints:
            joints = ", ".join(f"'{joint.name}' ({joint.type.name.lower()})" for joint in base_joints)
            message = (
                f"{prefix} of {file!r} is moved by its own {joints} joint(s), not a free joint, so there is no "
                f"floating base to pose. Declare `is_fixed_base=True`; those joints stay ordinary DOFs."
            )
        else:
            message = (
                f"{prefix} of {file!r} carries no joint, so Genesis welds it to the world. Declare "
                f"`is_fixed_base=True`, or add a `<freejoint/>` to the root body to float it."
            )
        if self.free_joints:
            message += (
                f" The file's free joint(s) {[joint.name for joint in self.free_joints]} sit on a body other than "
                f"the base link, and Eden's root pose and base DOFs address the base link only: load that body as "
                f"its own entity."
            )
        raise ValueError(message)

    def _set_initial_root_pose(self) -> None:
        """Write the entity's default root pose, honoring IPC-coupling write restrictions."""
        default_root_pos, default_root_quat = self.get_default_root_pose()
        needs_coup = getattr(self.material, "needs_coup", True)
        coup_type = getattr(self.material, "coup_type", None)
        is_ipc_coupled = self._env._has_ipc_coupler and needs_coup

        if is_ipc_coupled and (coup_type == "external_articulation" or self.is_fixed_base):
            # IPC cases where Genesis rejects direct root-pose writes:
            # - external_articulation (base driven by IPC constraint system)
            # - any IPC coup_type on fixed-base (Genesis set_pos/set_quat guard)
            # Root pose is baked into morph.pos at morph creation.
            pass
        elif is_ipc_coupled and not self.is_fixed_base:
            # Floating-base with IPC coupling (two_way_soft_constraint / ipc_only), written through
            # set_qpos on the free-root prefix (first 7 qs: xyz + wxyz). NOTE: qpos is the *solver*
            # frame, so for a body Genesis re-origins this places the centre of mass (and the principal
            # axes) on the default pose — the one exception to the authored-frame rule (see
            # docs/guides/entities.md).
            self.set_qpos(
                torch.cat([default_root_pos, default_root_quat], dim=-1),  # (B, 7)
                qs_idx_local=torch.arange(7, device=self.device),
            )
        elif self.is_fixed_base and not self.is_heterogeneous:
            # Fixed, single-morph base: the root pose (pos + quat) is baked into the morph
            # at creation time, so there is nothing to write post-build. Crucially, a fixed
            # URDF whose base link has no geometry has *no* settable base pose — calling
            # set_quat/set_pos on it raises in Genesis (``Expected shape: (0,)``). Skipping
            # the write is both correct and necessary for such assets.
            #
            # Heterogeneous (grouped) entities are excluded: their per-env pose is the group
            # root pose composed with each variant's offset (see get_default_root_pose), and
            # that composition is applied *only* through the post-build write below — the
            # grouped morphs bake each variant's own pose, not the group transform.
            pass
        else:
            # Floating base (non-IPC), or a grouped fixed base. Both writes are in the *authored* frame
            # (``relative=True``): ``default_root_pos`` / ``default_root_quat`` become the root body's own
            # pose — replacing an MJCF root body's authored ``pos`` / ``quat`` rather than composing with it,
            # which is why a floating base is not baked into the morph — and not Genesis' solver origin,
            # which sits on the centre of mass for a free single body. Orientation goes first: the
            # solver position of an offset body depends on its orientation, so the position write must see
            # the final quaternion.
            self.set_quat(default_root_quat, relative=True)
            self.set_pos(default_root_pos, relative=True)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ attach ----------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_unbuilt
    def attach_to(
        self,
        entity: Entity,
        link_name: str,
        attach_offset: Vec3FType | None = None,
        attach_quat: UnitVec4FType | None = None,
    ) -> None:
        """Attach this entity to another entity.

        The mounting transform — this entity's base link pose relative to the parent link —
        defaults to the entity's ``default_root_pos`` / ``default_root_quat``. Pass
        ``attach_offset`` / ``attach_quat`` to override it: after attachment the base link's world
        pose is ``parent_link_pose ∘ (attach_offset, attach_quat)``. Genesis composes the transform
        with the morph frame offset (up/front alignment) exactly as it does for the morph pose at
        load time. (Mirrors the ``attach_offset`` / ``attach_quat`` naming on ``CameraOptions``.)

        Parameters
        ----------
        entity: Entity
            The entity to attach to.
        link_name: str
            The name of the link to attach this entity to.
        attach_offset: array-like[float, float, float], optional
            Mounting position of this entity's base link in the parent link frame.
            Defaults to the entity's ``default_root_pos``.
        attach_quat: array-like[float, float, float, float], optional
            Mounting orientation (w, x, y, z) of this entity's base link in the parent link
            frame. Defaults to the entity's ``default_root_quat``.
        """
        # Genesis's ``attach`` writes one link-local pose shared by all envs, so a grouped
        # child's per-variant root offsets can't be expressed — reject rather than silently
        # mount every env at the same pose.
        if self.is_heterogeneous:
            raise NotImplementedError(
                f"attach_to is not supported for heterogeneous (GroupedEntityOptions) entity "
                f"'{self.name}': the mounting transform is a single pose shared by all envs and "
                f"would silently drop the per-variant root offsets. Attach a non-grouped entity, "
                f"or add per-variant mount support."
            )

        mount_pos = self._options.default_root_pos if attach_offset is None else attach_offset
        mount_quat = self._options.default_root_quat if attach_quat is None else attach_quat

        def _do_attach():
            child = self._entity
            # MJCF entities may have multiple kinematic trees (e.g. a world link
            # in one tree and the actual robot base in another).  Genesis's
            # ``attach`` assumes every link shares the same ``root_idx`` as
            # ``links[0]``.  Merge secondary trees into the first so the
            # assertion holds.
            if len(child.links) > 1:
                base_root = child.links[0].root_idx
                roots_to_merge = set()
                for link in child.links[1:]:
                    if link.root_idx != base_root and link.root_idx != -1:
                        roots_to_merge.add(link.root_idx)
                if roots_to_merge:
                    for link in child.links:
                        if link.root_idx in roots_to_merge:
                            link._root_idx = base_root

            # Forward kinematics interprets the base link's local pose relative to the new
            # parent link, so ``(mount_pos, mount_quat)`` acts as the mounting transform.
            # Genesis composes it with the morph frame offset (``_offset_pos``/``_offset_quat``)
            # internally. Always pass an explicit pose — the morph alone only carries
            # ``default_root_pos`` for a floating base, so relying on the morph would silently
            # drop ``default_root_quat``.
            child.attach(
                entity._entity,
                link_name,
                pos=np.asarray(mount_pos, dtype=gs.np_float),
                quat=np.asarray(mount_quat, dtype=gs.np_float),
            )

        self._env.register_pre_build_hook(_do_attach)
        self._is_attached = True
        self._attachment_parent = entity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties ------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def dofs_order(self) -> dict[str, int]:
        """Get the mapping from dofs name to its local index."""
        if self.is_built:
            return {dof_name: i for i, dof_name in enumerate(self.dofs_name)}
        raise RuntimeError("Entity is not built yet")

    @property
    def joints_name(self) -> list[str]:
        """Every joint's name, in Genesis' joint order (not the asset's): the free root and any fixed or passive
        joints as well as the controlled ones in ``dofs_name``.
        """
        return [joint.name for joint in self._entity.joints]

    @property
    def joints(self) -> list[RigidJoint]:
        """Every Genesis joint of this entity, in declaration order: free, fixed and DOF-carrying alike.

        Each joint's ``name``, ``type``, ``n_dofs``, ``dofs_idx_local`` and ``qs_idx_local`` together describe
        the whole simulated state. :attr:`dofs_name` does not: it is the *controlled* subset, and it leaves out
        passive DOFs such as a Robotiq 2F85's followers. Read this for a full state snapshot.
        """
        return list(self._entity.joints)

    @property
    def _every_joint_name(self) -> list[str]:
        """Every joint's name, controlled or not: what a key that is exactly a joint's name is judged literal against.

        So the exact name of a passive ``pin.1`` selects nothing among the controlled joints, rather than a
        controlled ``pin_1`` it would match as a regex. Every DOF-keyed config dict and name lookup uses it.
        """
        return [joint.name for joint in self._entity.joints]

    @property
    def free_joints(self) -> list[RigidJoint]:
        """All free joints, detected by joint *type* — never by name."""
        return [joint for joint in self._entity.joints if joint.type == gs.JOINT_TYPE.FREE]

    @property
    def num_dofs(self):
        """Return the number of controllable DOFs specified by ``dofs_name``.

        This may differ from the total number of DOFs of the entity; uncontrollable
        DOFs such as free base joints are not included.
        """
        if self.is_built:
            return len(self.dofs_name)
        raise RuntimeError("Entity is not built yet")

    @property
    def n_dofs(self):
        return self._entity.n_dofs

    @property
    def n_qs(self):
        return self._entity.n_qs

    @property
    def num_links(self):
        return self._entity.n_links

    @property
    def links(self):
        return self._entity.links

    @property
    def link_start(self):
        return self._entity.link_start

    @property
    def link_end(self):
        return self._entity.link_end

    @property
    def geoms(self):
        """This entity's collision geoms, as an entity-local list.

        The list is local but each ``geom.idx`` is **scene-global** — a Genesis ``RigidGeom`` has
        no ``idx_local`` (unlike a link, which carries both ``idx`` and ``idx_local``). So
        ``entity.geoms[geom.idx]`` is wrong on any entity with geoms before it in the scene
        (``geom_start != 0``); the local index is ``geom.idx - entity.geom_start``.
        """
        return self._entity.geoms

    @property
    def geom_start(self):
        """Scene-global index of this entity's first geom: ``geom.idx - geom_start`` indexes :attr:`geoms`."""
        return self._entity.geom_start

    @property
    def geom_end(self):
        """Scene-global index one past this entity's last geom (``geoms`` spans ``geom_start:geom_end``)."""
        return self._entity.geom_end

    @property
    def dof_start(self):
        return self._entity.dof_start

    @property
    def dof_end(self):
        return self._entity.dof_end

    # ------------------------------------------------------------------------------------
    # ------------------------------- find named indices ---------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def find_named_dofs_idx_local(
        self,
        dofs_name: list[str] | str,
        name_scope: list[str] | None = None,
        preserve_order: bool = True,
    ) -> tuple[list[str], list[int]]:
        """Find DOF indices by DOF/joint names or patterns.

        Returns *local* indices, Genesis' own: hand them to a ``dofs_idx_local=`` argument and nowhere
        else. For the controlled joints, :meth:`find_named_dofs` resolves the same query into these and
        the matching columns of the property tensors at once.

        Supports:
        - Exact names: "joint_1", taken literally even when they contain a pattern character
          ("pin.1" selects ``pin.1`` alone, not also ``pin_1``). Judged against every joint of the
          entity, so the exact name of a joint outside ``name_scope`` selects nothing rather than a
          neighbour inside it
        - Glob patterns: "joint_*", "arm?[LR]_*" (fnmatch semantics)
        - Regex patterns: any valid regex (uses fullmatch). If a pattern isn't an exact
          match or glob, we attempt to treat it as a regex.

        Parameters
        ----------
        dofs_name: list[str] | str
            The names of the DOFs to find.
        name_scope: list[str] | None
            The names of the joints to search within. If None, all joints will be searched; an
            empty list searches none, so any key raises.
        preserve_order: bool
            Whether to preserve the order of the DOFs. If True, the DOFs are returned in the order of the keys in
            ``dofs_name``.
        """
        all_joint_names = self._every_joint_name
        joint_names = all_joint_names if name_scope is None else name_scope

        _, names = resolve_matching_names(
            dofs_name, joint_names, preserve_order=preserve_order, literal_names=all_joint_names
        )

        res_dofs_name: list[str] = []
        res_dofs_idx: list[int] = []
        for name in names:
            joint = self._entity.get_joint(name)
            res_dofs_name.extend([name] * len(joint.dofs_idx_local))
            res_dofs_idx.extend(joint.dofs_idx_local)

        return res_dofs_name, res_dofs_idx

    @gs.assert_built
    def find_free_base_dofs_idx_local(self, joint_name: str | None = None) -> list[int]:
        """Find the six local DOF indices of the floating base's free joint.

        With ``joint_name=None`` (the default) the free joint is detected by joint
        *type* — the only convention-free spelling: an MJCF ``<freejoint>`` can carry
        any name (Unitree's G1 uses ``floating_base_joint``) while Genesis synthesizes
        ``root_joint`` for a URDF floating base, so any fixed name default breaks one
        of the two formats. Raises unless exactly one free joint exists. Pass an
        explicit ``joint_name`` to pick one on the rare multi-free-joint entity; it
        must name a free joint, so a hinge that happens to match a naming convention
        can never silently receive base-DOF writes.

        Parameters
        ----------
        joint_name: str | None
            Exact name of the free joint to select, or ``None`` to auto-detect by
            joint type.
        """
        free_joints = self.free_joints
        if joint_name is None:
            if len(free_joints) != 1:
                raise ValueError(
                    f"Entity '{self.name}': cannot auto-detect the floating base — expected exactly one "
                    f"free joint, found {len(free_joints)} ({[joint.name for joint in free_joints]}). "
                    + (
                        "Set `joint_name` to pick one."
                        if free_joints
                        else "The entity has no floating base (fixed or welded); this lookup does not apply to it."
                    )
                )
            return list(free_joints[0].dofs_idx_local)
        for joint in free_joints:
            if joint.name == joint_name:
                return list(joint.dofs_idx_local)
        raise ValueError(
            f"Entity '{self.name}': joint_name='{joint_name}' does not name a free joint. "
            f"Free joints: {[joint.name for joint in free_joints]}; "
            f"all joints: {[joint.name for joint in self._entity.joints]}. "
            f"Pass None to auto-detect the free base joint by type."
        )

    @gs.assert_built
    def find_named_links_idx_local(
        self,
        links_name: list[str] | str,
        name_scope: list[str] | None = None,
        preserve_order: bool = True,
    ) -> tuple[list[str], list[int]]:
        """Find link indices by link names or patterns.

        Supports:
        - Exact names: "link_1"
        - Glob patterns: "link_*", "arm?[LR]_*" (fnmatch semantics)
        - Regex patterns: any valid regex (uses fullmatch). If a pattern isn't an exact
          match or glob, we attempt to treat it as a regex.

        Parameters
        ----------
        links_name: list[str] | str
            The names of the links to find.
        name_scope: list[str] | None
            The names of the links to search within. If None, all links will be searched; an
            empty list searches none, so any key raises.
        preserve_order: bool
            Whether to preserve the order of the links. If True, the links are returned in the order of the keys in
            ``links_name``.
        """
        every_link_name = [link.name for link in self._entity.links]
        all_links_name = every_link_name if name_scope is None else name_scope
        _, names = resolve_matching_names(
            links_name, all_links_name, preserve_order=preserve_order, literal_names=every_link_name
        )

        res_links_name: list[str] = []
        res_links_idx: list[int] = []
        for name in names:
            link = self._entity.get_link(name)
            res_links_name.extend([name])
            res_links_idx.extend([link.idx_local])

        return res_links_name, res_links_idx

    # ------------------------------------------------------------------------------------
    # --------------------------------- FK / IK / Jacobian ------------------------------
    # ------------------------------------------------------------------------------------

    def forward_kinematics(self, qpos, qs_idx_local=None, links_idx_local=None, envs_idx=None):
        """Compute forward kinematics for a single link or multiple links."""
        return self._entity.forward_kinematics(qpos, qs_idx_local, links_idx_local, envs_idx)

    def inverse_kinematics(
        self,
        links_name,
        poss=None,
        quats=None,
        local_points=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for a single link or multiple links.

        Parameters
        ----------
        links_name: str | list[str]
            The name of the link to compute inverse kinematics for.
        poss: array_like | list[array_like], optional
            The target position of each link. If None, position error is not considered. Defaults to None.
        quats: array_like | list[array_like], optional
            The target orientation (w, x, y, z) of each link. If None, orientation error is not considered.
            Defaults to None.
        local_points: array_like | list[array_like], optional
            The point in each link's *local* frame that ``poss`` / ``quats`` place, one per link. Use it to steer a
            frame the solver does not own -- a tool tip, or a gripper welded ahead of the flange -- without solving
            for the link origin and correcting afterwards. If None, each link's origin is placed. Defaults to None.
        init_qpos: array_like, optional
            Initial joint positions used to seed the solver. Defaults to None.
        respect_joint_limit: bool, optional
            Whether to clamp the solution to the joint limits. Defaults to True.
        max_samples: int, optional
            Maximum number of random restarts attempted by the solver. Defaults to 50.
        max_solver_iters: int, optional
            Maximum number of iterations per sample. Defaults to 20.
        damping: float, optional
            Damping factor for the damped least-squares update. Defaults to 0.01.
        pos_tol: float, optional
            Position tolerance for convergence, in meters. Defaults to 5e-4.
        rot_tol: float, optional
            Rotation tolerance for convergence, in radians. Defaults to 5e-3.
        pos_mask: list[bool], optional
            Per-axis ``(x, y, z)`` mask selecting which position components to match.
            Defaults to ``[True, True, True]``.
        rot_mask: list[bool], optional
            Per-axis ``(x, y, z)`` mask selecting which rotation components to match.
            Defaults to ``[True, True, True]``.
        max_step_size: float, optional
            Maximum joint update applied per iteration. Defaults to 0.5.
        dofs_idx_local: array_like, optional
            Local indices of the DOFs the solver is allowed to move. If None, all DOFs are used. Defaults to None.
        return_error: bool, optional
            If True, also return the final residual error alongside the solution. Defaults to False.
        envs_idx: array_like, optional
            The indices of the environments to solve for. If None, all environments are used. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_qs,) or (n_envs, n_qs) or (len(envs_idx), n_qs)
            Solver qpos (the entity's generalized coordinates).
        error_pose : array_like, shape (..., 6)
            Pose error ``[pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]`` per target; returned only when
            ``return_error`` is True.
        """
        if isinstance(links_name, str):
            return self._entity.inverse_kinematics(
                link=self.get_link(links_name),
                pos=poss,
                quat=quats,
                local_point=local_points,
                init_qpos=init_qpos,
                respect_joint_limit=respect_joint_limit,
                max_samples=max_samples,
                max_solver_iters=max_solver_iters,
                damping=damping,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                pos_mask=pos_mask,
                rot_mask=rot_mask,
                max_step_size=max_step_size,
                dofs_idx_local=dofs_idx_local,
                return_error=return_error,
                envs_idx=envs_idx,
            )
        else:
            return self._entity.inverse_kinematics_multilink(
                links=[self.get_link(name) for name in links_name],
                poss=poss,
                quats=quats,
                local_points=local_points,
                init_qpos=init_qpos,
                respect_joint_limit=respect_joint_limit,
                max_samples=max_samples,
                max_solver_iters=max_solver_iters,
                damping=damping,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                pos_mask=pos_mask,
                rot_mask=rot_mask,
                max_step_size=max_step_size,
                dofs_idx_local=dofs_idx_local,
                return_error=return_error,
                envs_idx=envs_idx,
            )

    def get_jacobian(self, link, local_point=None):
        """
        Get the spatial Jacobian for a point on a target link.

        Columns are returned in ``dofs_name`` order (matching ``set_qpos`` /
        ``get_dofs_limit``), **not** the Genesis kinematic-tree order.

        Parameters
        ----------
        link : RigidLink
            The target link.
        local_point : torch.Tensor or None, shape (3,)
            Coordinates of the point in the link's *local* frame.
            If None, the link origin is used.

        Returns
        -------
        jacobian : torch.Tensor
            The Jacobian matrix of shape (n_envs, 6, entity.n_dofs) or (6, entity.n_dofs) if n_envs == 0.
        """
        J = self._entity.get_jacobian(link, local_point)
        if self._jac_perm is not None:
            J = J[..., self._jac_perm]
        return J

    # ------------------------------------------------------------------------------------
    # ------------------------------------ getters --------------------------------------
    # ------------------------------------------------------------------------------------

    def get_link(self, link_name: str):
        return self._entity.get_link(link_name)

    @property
    def base_link_name(self) -> str:
        """Name of this entity's root link (the base of its kinematic tree).

        The frame a mounted entity's pose is expressed in, and the link an asset is rooted
        at — so a caller can reference it without hard-coding an asset-specific name
        (``"base"``, ``"right_hand_C_MC"``, …) or reaching through ``_entity``. Note that
        for an *attached* entity Genesis reparents this link under the parent, so it is no
        longer a free root.
        """
        return self._entity.base_link.name

    def get_AABB(self):
        return self._entity.get_AABB()

    def get_qpos(self, qs_idx_local=None, envs_idx=None):
        """Get the entity's generalized coordinates.

        These are q-coordinates, not DOFs: a floating base contributes 7 root qs
        (position + ``(w, x, y, z)`` quaternion) against 6 root DOFs, so ``n_qs`` and
        ``n_dofs`` differ for a free-base entity. A welded (attached) child has no root
        qs at all.

        Parameters
        ----------
        qs_idx_local : None | array_like, optional
            Local q indices to read. If None, reads the entity's own q range in
            dofs_name order (root qs first for a floating base). Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        qpos : torch.Tensor, shape (n_qs,) or (n_envs, n_qs)
            The entity's generalized coordinates.
        """
        if qs_idx_local is None:
            qs_idx_local = self._genesis_qs_idx
        else:
            qs_idx_local = _genesis_selection(qs_idx_local)
        return _row_major_selection(self._entity.get_qpos(qs_idx_local, envs_idx=envs_idx), qs_idx_local)

    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None):
        """Get the force the controller applied to the entity's dofs over the last physics step.

        This is the actuator/PD contribution only, already clipped to the force range —
        use :meth:`get_dofs_force` for the total internal force. A target set since that step
        shows in the force of the next one.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The control force on the entity's dofs.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        return _row_major_selection(
            self._entity.get_dofs_control_force(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx), dofs_idx_local
        )

    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None):
        """Get the total internal force acting on the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The total force on the entity's dofs.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        return _row_major_selection(
            self._entity.get_dofs_force(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx), dofs_idx_local
        )

    def get_dofs_vel(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' velocity.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        velocity : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' velocity.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        return _row_major_selection(self._entity.get_dofs_velocity(dofs_idx_local, envs_idx), dofs_idx_local)

    def get_dofs_pos(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' position.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        position : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' position.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        return _row_major_selection(self._entity.get_dofs_position(dofs_idx_local, envs_idx), dofs_idx_local)

    @property
    def has_batched_dofs_info(self) -> bool:
        """Whether the solver holding this entity stores per-environment DOF info.

        The solver's flag, not ``EnvOptions.batch_dofs_info``: that option configures the *rigid* solver only,
        and Genesis turns it on by itself in either solver once it holds a heterogeneous entity, whose variants
        each bring their own joint parameters (https://github.com/Genesis-Embodied-AI/Genesis/pull/3497).
        """
        return bool(self.solver._options.batch_dofs_info)

    @property
    def has_batched_links_info(self) -> bool:
        """Whether the solver holding this entity stores per-environment link info.

        The solver's flag, not ``EnvOptions.batch_links_info``: Genesis turns it on by itself for a scene that
        holds a heterogeneous entity, whose variants differ in mass and inertia.
        """
        return bool(self.solver._options.batch_links_info)

    @property
    def setters_take_every_row(self) -> bool:
        """Whether ``set_dofs_vel`` / ``set_qpos`` take every env's row alongside a bool ``envs_idx``.

        On Genesis' zero-copy path they write the masked rows through ``torch.where``, on the device,
        as long as the DOF (q) argument is a ``range`` (:func:`eden.utils.misc.contiguous_range`) —
        no ``nonzero()`` sync to turn the mask into indices. The selected rows are accepted in every
        case, so pass those otherwise. Not under hibernation: ``set_dofs_velocity`` first wakes the
        entities it touches, which turns the mask into indices and then rejects the unselected envs'
        rows. Reads the solver's own flag rather than ``EnvOptions.use_hibernation``: a
        ``rigid_options`` handed to the env constructor can switch hibernation on behind that option,
        and Genesis switches it off under ``requires_grad``. ``set_dofs_pos`` takes every row too when
        its DOFs are hinges or slides (it writes them through ``set_qpos``); the ``control_dofs_*``
        setters never do. A kinematic entity's solver never hibernates.
        """
        return _solver_takes_every_row(self.solver)

    def _get_dofs_info(self, getter, dofs_idx_local, envs_idx):
        """Read a per-DOF *info* field, always with a leading environment dimension.

        Genesis stores DOF info unbatched unless :attr:`has_batched_dofs_info`, and then
        refuses ``envs_idx`` outright. In that mode every environment shares the one stored
        row, so broadcasting it over environments is exact — and it spares callers a shape
        that changes with the task config. See the class ``Batching`` note.

        Parameters
        ----------
        getter : Callable
            The Genesis entity getter to read, e.g. ``self._entity.get_dofs_armature``.
            May return a tuple of tensors (``get_dofs_limit`` / ``get_dofs_force_range``),
            in which case every element is normalized.
        dofs_idx_local : None | array_like
            Local DOF indices. If None, all of the entity's dofs in dofs_name order.
        envs_idx : None | int | slice | array_like | torch.Tensor
            The environment indices. If None, all environments are considered.

        Returns
        -------
        info : torch.Tensor | tuple[torch.Tensor, ...]
            The requested field, shaped ``(n_envs, n_dofs, ...)`` and row-major in either mode.
            Never a live view into the solver.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        if self.has_batched_dofs_info:
            # Row-major, as `_row_major_selection` gives the state getters: a `range` selection (the
            # default) comes back env-minor, and a noise `rand_like` would follow that layout.
            out = getter(dofs_idx_local, envs_idx)
            if isinstance(out, tuple):
                return tuple(item.contiguous() for item in out)
            return out.contiguous()
        out = getter(dofs_idx_local, None)  # (n_dofs, ...) — shared by every env
        if isinstance(out, tuple):
            return tuple(self._expand_over_envs(item, envs_idx) for item in out)
        return self._expand_over_envs(out, envs_idx)

    def _expand_over_envs(self, per_dof: torch.Tensor, envs_idx) -> torch.Tensor:
        """Broadcast one shared per-DOF row over environments, then select ``envs_idx``."""
        return per_dof.unsqueeze(0).expand(self._env.num_envs, *per_dof.shape)[envs_idx].contiguous()

    def get_dofs_kp(self, dofs_idx_local=None, envs_idx=None):
        """Get the positional (proportional) gain of the entity's PD controllers.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        kp : torch.Tensor, shape (n_envs, n_dofs)
            The positional gain of the entity's dofs. Carries the environment axis in either
            ``batch_dofs_info`` mode — see the class ``Batching`` note.
        """
        return self._get_dofs_info(self._entity.get_dofs_kp, dofs_idx_local, envs_idx)

    def get_dofs_kd(self, dofs_idx_local=None, envs_idx=None):
        """Get the velocity (derivative) gain of the entity's PD controllers.

        Genesis names this gain ``kv``; Eden exposes it as ``kd`` throughout.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        kd : torch.Tensor, shape (n_envs, n_dofs)
            The velocity gain of the entity's dofs. Carries the environment axis in either
            ``batch_dofs_info`` mode — see the class ``Batching`` note.
        """
        return self._get_dofs_info(self._entity.get_dofs_kv, dofs_idx_local, envs_idx)

    def get_dofs_force_range(self, dofs_idx_local=None, envs_idx=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the force range (min and max limits) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_envs, n_dofs)
            The lower limit of the force range for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_envs, n_dofs)
            The upper limit of the force range for the entity's dofs.
        """
        return self._get_dofs_info(self._entity.get_dofs_force_range, dofs_idx_local, envs_idx)

    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None):
        """Get the positional limits of the entity's dofs.

        Returns the two bounds as a *tuple of tensors*, not one stacked tensor —
        unpack it as ``lower, upper = entity.get_dofs_limit()``. Unlimited dofs report
        ``-inf`` / ``+inf``. Both tensors carry the environment axis.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_envs, n_dofs)
            The lower positional limit of the entity's dofs.
        upper_limit : torch.Tensor, shape (n_envs, n_dofs)
            The upper positional limit of the entity's dofs.
        """
        return self._get_dofs_info(self._entity.get_dofs_limit, dofs_idx_local, envs_idx)

    def get_qpos_limit(self, envs_idx=None):
        """Get the positional limits of the entity's generalized coordinates, in :meth:`get_qpos` order.

        A hinge / slide coordinate takes its DOF's limit and a free joint's position its translational
        DOFs'. A free or ball joint's ``(w, x, y, z)`` quaternion takes the unit bounds ``[-1, 1]``, since
        it is normalized rather than limited. The limits are read off the solver, so each environment
        reports those of the heterogeneous variant it carries.

        Parameters
        ----------
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_envs, n_qs)
            The lower positional limit of the entity's generalized coordinates.
        upper_limit : torch.Tensor, shape (n_envs, n_qs)
            The upper positional limit of the entity's generalized coordinates.
        """
        # Every DOF, in Genesis order, then every q of the entity, joint by joint.
        lower, upper = self._get_dofs_info(self._entity.get_dofs_limit, range(self._entity.n_dofs), envs_idx)
        unit = torch.ones((lower.shape[0], 4), dtype=lower.dtype, device=lower.device)
        qs_lower, qs_upper = [lower[:, :0]], [upper[:, :0]]
        for joint in self._entity.joints:
            dof_start = joint.dof_start - self._entity.dof_start
            if joint.type in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.SPHERICAL):
                translation = slice(dof_start, dof_start + joint.n_qs - 4)  # 3 for a free joint, none for a ball
                qs_lower += [lower[:, translation], -unit]
                qs_upper += [upper[:, translation], unit]
            elif joint.type != gs.JOINT_TYPE.FIXED:
                qs_lower.append(lower[:, dof_start : dof_start + joint.n_dofs])
                qs_upper.append(upper[:, dof_start : dof_start + joint.n_dofs])
        qs_lower, qs_upper = torch.cat(qs_lower, dim=1), torch.cat(qs_upper, dim=1)
        if self.qs_idx_local is None:
            return qs_lower, qs_upper
        return qs_lower[:, self.qs_idx_local], qs_upper[:, self.qs_idx_local]

    def get_dofs_stiffness(self, dofs_idx_local=None, envs_idx=None):
        """Get the joint-spring stiffness of the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        stiffness : torch.Tensor, shape (n_envs, n_dofs)
            The spring stiffness of the entity's dofs. Carries the environment axis in either
            ``batch_dofs_info`` mode — see the class ``Batching`` note.
        """
        return self._get_dofs_info(self._entity.get_dofs_stiffness, dofs_idx_local, envs_idx)

    def get_dofs_invweight(self, dofs_idx_local=None, envs_idx=None):
        """Get the inverse effective inertia (``invweight``) of the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        invweight : torch.Tensor, shape (n_envs, n_dofs)
            The inverse effective inertia of the entity's dofs. Carries the environment axis in either
            ``batch_dofs_info`` mode — see the class ``Batching`` note.
        """
        return self._get_dofs_info(self._entity.get_dofs_invweight, dofs_idx_local, envs_idx)

    def get_dofs_armature(self, dofs_idx_local=None, envs_idx=None):
        """Get the armature (added rotor inertia) of the entity's dofs.

        For an actuated entity under ``EnvOptions.integrator="euler"`` this is *not* the configured
        value: ``post_build`` folds ``kd * substep_dt`` into the armature for implicit-damping
        stability. Under ``approximate_implicitfast`` (the default) and ``implicitfast`` it is.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        armature : torch.Tensor, shape (n_envs, n_dofs)
            The armature of the entity's dofs. Carries the environment axis in either
            ``batch_dofs_info`` mode — see the class ``Batching`` note.
        """
        return self._get_dofs_info(self._entity.get_dofs_armature, dofs_idx_local, envs_idx)

    def get_dofs_damping(self, dofs_idx_local=None, envs_idx=None):
        """Get the viscous damping of the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        damping : torch.Tensor, shape (n_envs, n_dofs)
            The damping of the entity's dofs. Carries the environment axis in either
            ``batch_dofs_info`` mode — see the class ``Batching`` note.
        """
        return self._get_dofs_info(self._entity.get_dofs_damping, dofs_idx_local, envs_idx)

    def get_dofs_frictionloss(self, dofs_idx_local=None, envs_idx=None):
        """Get the dry-friction loss (stick band) of the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments will be considered. Defaults to None.

        Returns
        -------
        frictionloss : torch.Tensor, shape (n_envs, n_dofs)
            The dry-friction loss of the entity's dofs. Carries the environment axis in either
            ``batch_dofs_info`` mode — see the class ``Batching`` note.
        """
        return self._get_dofs_info(self._entity.get_dofs_frictionloss, dofs_idx_local, envs_idx)

    def get_mass_mat(self, envs_idx=None, decompose=False):
        return self._entity.get_mass_mat(envs_idx, decompose)

    def get_pos(self, envs_idx=None, *, relative: bool = False):
        """Return the position of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            Whether to report the Genesis user-frame position with the morph pose offset and inertial alignment
            stripped. Defaults to False, which returns the solver/world-frame position.

        Returns
        -------
        pos : torch.Tensor, shape (3,) or (n_envs, 3)
            The base-link position.
        """
        return self._entity.get_pos(envs_idx=envs_idx, relative=relative)

    def get_quat(self, envs_idx=None, *, relative: bool = False):
        """Return the quaternion of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            Whether to report the Genesis user-frame orientation with the morph pose offset and inertial alignment
            stripped. Defaults to False, which returns the solver/world-frame orientation.

        Returns
        -------
        quat : torch.Tensor, shape (4,) or (n_envs, 4)
            The base-link quaternion in ``(w, x, y, z)`` format.
        """
        return self._entity.get_quat(envs_idx=envs_idx, relative=relative)

    def get_T(self, envs_idx=None, *, relative: bool = False):
        """Return the entity's base-link pose as a 4x4 homogeneous transform matrix.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            Whether to use Genesis's user-frame pose with the morph pose offset and inertial alignment stripped.
            Defaults to False, which returns the solver/world-frame transform.

        Returns
        -------
        transform : torch.Tensor, shape (4, 4) or (n_envs, 4, 4)
            The base-link homogeneous transform.
        """
        pos = self.get_pos(envs_idx=envs_idx, relative=relative)
        quat = self.get_quat(envs_idx=envs_idx, relative=relative)
        return gu.trans_quat_to_T(pos, quat)

    def get_links_pos(self, ls_idx_local=None, envs_idx=None, *, ref="link_origin", relative: bool = False):
        """Return the positions of the entity's links.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        ref : {"link_origin", "link_com"} | gs.link_ref_frame, optional
            Point on each link whose position is reported: the link frame origin, or the link's center of mass.
            Defaults to "link_origin".
        relative : bool, optional
            Whether to report Genesis user-frame positions with the morph pose offset stripped. Defaults to False,
            which returns solver/world-frame positions. Only affects ``ref="link_origin"``.

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The link positions.
        """
        pos = self._entity.get_links_pos(
            links_idx_local=_genesis_selection(ls_idx_local),
            envs_idx=envs_idx,
            ref=resolve_link_ref_frame(ref),
            relative=relative,
        )
        return _row_major_selection(pos, ls_idx_local)

    def get_links_quat(self, ls_idx_local=None, envs_idx=None, *, relative: bool = False):
        """Return the orientations of the entity's links.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            Whether to report Genesis user-frame orientations with the morph pose offset stripped. Defaults to False,
            which returns solver/world-frame orientations.

        Returns
        -------
        quat : torch.Tensor, shape (n_links, 4) or (n_envs, n_links, 4)
            The link quaternions in ``(w, x, y, z)`` format.
        """
        quat = self._entity.get_links_quat(
            links_idx_local=_genesis_selection(ls_idx_local), envs_idx=envs_idx, relative=relative
        )
        return _row_major_selection(quat, ls_idx_local)

    def get_links_vel(self, ls_idx_local=None, envs_idx=None, *, ref="link_origin", relative: bool = False):
        """Return the linear velocities of the entity's links, in world coordinates.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        ref : {"link_origin", "link_com"} | gs.link_ref_frame, optional
            Point on each link at which the velocity is expressed. A rotating link gives
            different answers at its origin and its center of mass. Defaults to "link_origin".
        relative : bool, optional
            Whether to report the velocity of the Genesis user-frame link origin, with the morph pose offset and
            inertial alignment stripped. Defaults to False, which returns the velocity of the solver/world-frame
            origin :meth:`get_links_pos` reports. The two differ by the transport term ``omega x d`` over the offset
            ``d``, and only for ``ref="link_origin"``.

        Returns
        -------
        vel : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The link linear velocities.
        """
        vel = self._entity.get_links_vel(
            links_idx_local=_genesis_selection(ls_idx_local),
            envs_idx=envs_idx,
            ref=resolve_link_ref_frame(ref),
            relative=relative,
        )
        return _row_major_selection(vel, ls_idx_local)

    def get_links_ang(self, ls_idx_local=None, envs_idx=None):
        """Return the angular velocities of the entity's links, in world coordinates.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The link angular velocities, as rotation vectors in rad/s.
        """
        return _row_major_selection(
            self._entity.get_links_ang(links_idx_local=_genesis_selection(ls_idx_local), envs_idx=envs_idx),
            ls_idx_local,
        )

    def get_links_acc(self, ls_idx_local=None, envs_idx=None, *, relative: bool = False):
        """Return the linear accelerations of the entity's link origins, in world coordinates.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            Whether to report the acceleration of the Genesis user-frame link origin, with the morph pose offset and
            inertial alignment stripped. Defaults to False, which returns the acceleration of the solver/world-frame
            origin :meth:`get_links_pos` reports.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The link linear accelerations.
        """
        acc = self._entity.get_links_acc(
            links_idx_local=_genesis_selection(ls_idx_local), envs_idx=envs_idx, relative=relative
        )
        return _row_major_selection(acc, ls_idx_local)

    def get_links_acc_ang(self, ls_idx_local=None, envs_idx=None):
        """Return the angular accelerations of the entity's links, in world coordinates.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.

        Returns
        -------
        acc_ang : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The link angular accelerations, in rad/s^2.
        """
        acc_ang = self._entity.get_links_acc_ang(links_idx_local=_genesis_selection(ls_idx_local), envs_idx=envs_idx)
        return _row_major_selection(acc_ang, ls_idx_local)

    def get_links_inertial_mass(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Return the per-link mass the solver currently uses.

        This is each link's *own* mass — unrelated to :meth:`get_mass`, which returns the
        entity's **total** mass. Genesis holds a single per-link mass channel, so
        mass-randomization events write here too: :meth:`set_mass_shift` moves this value,
        and the pre-randomization nominal is what :meth:`get_mass_shift` subtracts.

        The leading environment dimension is always present, and ``envs_idx`` always
        works. Genesis, underneath, stores link info unbatched unless
        ``EnvOptions.batch_links_info`` is set and then refuses ``envs_idx`` outright; in
        that mode every environment shares the one stored value, so broadcasting it is
        exact and callers get the same shape either way — the one the per-link DR readback
        getters (:meth:`get_mass_shift` and friends) return.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments are considered. Defaults to None.

        Returns
        -------
        inertial_mass : torch.Tensor, shape (n_envs, n_links)
            The mass of each requested link, in kg. Always a copy, never a live
            view into the solver.
        """
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        ls_idx_local = _genesis_selection(ls_idx_local)
        if self.has_batched_links_info:
            return self._entity.get_links_mass(ls_idx_local, envs_idx)
        per_link = self._entity.get_links_mass(ls_idx_local)  # (n_links,)
        return per_link.unsqueeze(0).expand(self._env.num_envs, *per_link.shape)[envs_idx].contiguous()

    def get_links_invweight(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Return the inverse effective inertia (``invweight``) of the entity's links.

        Two numbers per link — translational and rotational — giving the acceleration
        response of the link to a unit force / torque applied at its center of mass with
        the rest of the articulated body attached. Genesis derives them from link masses,
        poses, inertia matrices, joint placements and DOF armatures, so they are read-only:
        the per-link counterpart of :meth:`get_dofs_invweight`.

        Carries a leading environment dimension and accepts ``envs_idx`` in either
        ``EnvOptions.batch_links_info`` mode — see :meth:`get_links_inertial_mass` for why.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            The environment indices (int / slice / index tensor / bool mask). If None, all
            environments are considered. Defaults to None.

        Returns
        -------
        invweight : torch.Tensor, shape (n_envs, n_links, 2)
            The translational and rotational inverse effective inertia of each requested
            link. Always a copy, never a live view into the solver.
        """
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        if self.has_batched_links_info:
            return self._entity.get_links_invweight(ls_idx_local, envs_idx)
        per_link = self._entity.get_links_invweight(ls_idx_local)  # (n_links, 2)
        return per_link.unsqueeze(0).expand(self._env.num_envs, *per_link.shape)[envs_idx].contiguous()

    def get_links_net_contact_force(self):
        """Return the net contact force on each of the entity's links, in world coordinates.

        Covers direct external contacts only — constraint and actuator forces are not
        included. Values come from the most recent ``scene.step()``. Unlike the other
        link accessors this one takes no indices: it always returns every link, for
        every environment.

        Returns
        -------
        force : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The net contact force on each link, in N.
        """
        return self._entity.get_links_net_contact_force()

    def get_contacts(
        self, with_entity: Entity | None = None, exclude_self_contact: bool = False, *, is_padded: bool = False
    ):
        """Return contact information computed during the most recent ``scene.step()``.

        If ``with_entity`` is provided, only returns contact information involving the caller and the specified entity.
        Otherwise, returns all contact information involving the caller entity.
        When ``with_entity`` is ``self``, it will return the self-collision only.

        The returned dict contains the following keys (a contact pair consists of two geoms: A and B):

        - 'geom_a'     : The global geom index of geom A in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_a])
        - 'geom_b'     : The global geom index of geom B in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_b])
        - 'link_a'     : The global link index of link A (that contains geom A) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_a])
        - 'link_b'     : The global link index of link B (that contains geom B) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_b])
        - 'position'   : The contact position in world frame.
        - 'force_a'    : The contact force applied to geom A.
        - 'force_b'    : The contact force applied to geom B.
        - 'valid_mask' : A boolean mask indicating whether the contact information is valid.
                        (Only when scene is parallelized)

        The shape of each entry is (n_envs, n_contacts, ...) for scene with parallel envs
                               and (n_contacts, ...) for non-parallelized scene.

        Parameters
        ----------
        with_entity : Entity, optional
            The entity to check contact with. Defaults to None.
        exclude_self_contact: bool
            Exclude the self collision from the returning contacts. Defaults to False.
        is_padded: bool
            Return every contact slot of each env, invalid ones masked out by ``valid_mask`` and their
            floating-point fields zeroed, instead of trimming to the most contacts any env has. Trimming reads
            that count back to the host (a GPU->CPU sync) and sizes the result by it; the padded form does
            neither, at the cost of masking the full capacity. While the env's graded phase runs under a CUDA
            graph (``EnvOptions.cuda_graph_options``) the padded form is returned regardless (see
            :meth:`~eden.entities.base.Entity._query_contacts`): a reader that respects ``valid_mask`` (which
            every reader must, the trimmed form masking out other entities' contacts too) gets the same
            contacts. Defaults to False.

        Returns
        -------
        contact_info : dict
            The contact information.
        """
        if isinstance(with_entity, Entity):
            with_entity = with_entity._entity
        return self._query_contacts(
            lambda padded: self._entity.get_contacts(
                with_entity=with_entity, exclude_self_contact=exclude_self_contact, is_padded=padded
            ),
            is_padded,
        )

    def get_grouped_contacts(
        self,
        other_entity: "RigidEntity",
        *,
        links_a_idx_local: torch.Tensor | list[int] | None = None,
        links_b_idx_local: torch.Tensor | list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Force-weighted contact aggregation per (self link, other link) pair.

        Layered on top of :meth:`get_contacts`: per (link_a, link_b) pair,
        compute the contact-force-norm-weighted mean of contact positions
        across every contact between *this* entity (``self``) and
        ``other_entity`` in the most recent ``scene.step()``.

        The output canonicalises orientation: axis 1 always indexes
        ``self``'s links and axis 2 always indexes ``other_entity``'s
        links, regardless of which side Genesis happened to label "a" or
        "b" for any given contact.

        Port of ``dexmachina/envs/contacts.get_grouped_contact_pos``.

        Parameters
        ----------
        other_entity:
            The contact partner. Must be a different entity (``other is not
            self``); self-contact is not supported by this layer — use
            :meth:`get_contacts` directly with ``with_entity=self``.
        links_a_idx_local:
            Local link indices on ``self`` to include on axis 1. Defaults
            to all of ``self``'s links (``[0, num_links)``). For per-step
            consumers, pre-resolve once at ``build()`` time and pass a
            cached ``torch.long`` tensor on ``self.device`` to avoid a
            tiny H2D copy each call (no-op early-return path).
        links_b_idx_local:
            Local link indices on ``other_entity`` to include on axis 2.
            Defaults to all of ``other_entity``'s links. Same hot-path
            note as above.

        Returns
        -------
        dict with:

        - ``position`` : ``(num_envs, n_a, n_b, 3)`` — contact-force-norm
          weighted mean position in world frame; zero where ``valid`` is
          False.
        - ``valid`` : ``(num_envs, n_a, n_b)`` bool — True iff the pair's
          aggregate force-norm is strictly positive (``force_norm_sum >
          0``), matching the DexMachina reference. Pairs whose contacts
          all carry zero force — a quasi-static initial overlap or the
          frame of make/break — are reported invalid because no
          force-weighted mean position is defined for them. Callers that
          want pure geometric "are they touching" detection should
          bool-AND on the raw ``get_contacts(...)["valid_mask"]``
          themselves.
        - ``force_norm_sum`` : ``(num_envs, n_a, n_b)`` — sum of contact
          force magnitudes (``Σ ‖f_i‖``, **not** ``‖Σ f_i‖``;
          opposing-direction contacts do not cancel) across the pair.
          The denominator used to weight the mean; useful as a
          contact-strength feature in its own right.

        Notes
        -----
        Requires a parallelised scene (``num_envs >= 1``); the
        non-parallelised path is not implemented since the only consumers
        — RL reward / observation terms — always run parallelised.
        """
        if not isinstance(other_entity, RigidEntity):
            raise TypeError(f"other_entity must be a RigidEntity, got {type(other_entity).__name__}")
        if other_entity._entity.idx == self._entity.idx:
            raise ValueError(
                "get_grouped_contacts does not support self-contact; use get_contacts(with_entity=self) directly."
            )

        contacts = self.get_contacts(with_entity=other_entity)
        if "valid_mask" not in contacts:
            raise NotImplementedError(
                "get_grouped_contacts requires the env-leading contact layout "
                "(`valid_mask` field present in get_contacts output). Genesis "
                "scenes built with `n_envs == 0` return non-parallelised "
                "contacts without this field; rebuild with `num_envs >= 1` if "
                "you need this aggregation."
            )

        device = contacts["position"].device
        a_local = resolve_local_link_idx(links_a_idx_local, self.num_links, device)
        b_local = resolve_local_link_idx(links_b_idx_local, other_entity.num_links, device)
        a_global = a_local + self.link_start
        b_global = b_local + other_entity.link_start
        return aggregate_grouped_contacts(
            link_a=contacts["link_a"],
            link_b=contacts["link_b"],
            position=contacts["position"],
            force=contacts["force_a"],
            valid_mask=contacts["valid_mask"],
            a_global=a_global,
            b_global=b_global,
        )

    def detect_collision(self, env_idx=0):
        """Detect collision for the entity. This only supports a single environment."""
        return self._entity.detect_collision(env_idx=env_idx)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ setters --------------------------------------
    # ------------------------------------------------------------------------------------

    def set_pos(
        self,
        pos,
        envs_idx=None,
        *,
        relative: bool = False,
    ) -> None:
        """Set the position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            Whether ``pos`` is expressed in Genesis's user frame, with the morph pose offset and inertial alignment
            applied to reach the solver/world frame. Defaults to False: ``pos`` is the solver/world-frame position.
        """
        self._entity.set_pos(
            pos,
            envs_idx=envs_idx,
            relative=relative,
        )

    def set_quat(
        self,
        quat,
        envs_idx=None,
        *,
        relative: bool = True,
    ) -> None:
        """Set the entity's orientation.

        Parameters
        ----------
        quat : array_like
            The orientation to set in ``(w, x, y, z)`` format.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            If True, ``quat`` is composed with the morph pose offset in Genesis's user frame. If False, ``quat`` is
            written directly in the solver/world frame. Defaults to True.
        """
        self._entity.set_quat(
            quat,
            envs_idx=envs_idx,
            relative=relative,
        )

    def set_qpos(
        self,
        qpos,
        qs_idx_local=None,
        envs_idx=None,
        *,
        zero_velocity=True,
        skip_forward=False,
    ) -> None:
        """Teleport the entity by setting its generalized coordinates.

        Parameters
        ----------
        qpos : array_like, shape (n_qs,) or (len(envs_idx), n_qs)
            The generalized coordinates to write, ordered to match ``qs_idx_local``. With a bool-mask
            ``envs_idx``, every env's row ``(num_envs, n_qs)`` is accepted too, and the mask picks which
            are written — with :attr:`setters_take_every_row`, the sync-free form for a reset.
        qs_idx_local : None | array_like, optional
            Local q indices to write. If None, writes the entity's own q range in
            dofs_name order (root qs first for a floating base). Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the entity's velocities after the write, so the teleport does
            not leave stale momentum behind. Defaults to True.
        skip_forward : bool, optional
            Whether to skip the forward-kinematics pass that refreshes link poses. Set True
            only when a later setter in the same reset will trigger it anyway. Defaults to False.
        """
        if qs_idx_local is None:
            qs_idx_local = self._genesis_qs_idx
        else:
            qs_idx_local = _genesis_selection(qs_idx_local)
        qpos, envs_idx = _mask_rows(
            qpos, envs_idx, on_device=isinstance(qs_idx_local, range) and self.setters_take_every_row
        )
        self._entity.set_qpos(
            qpos,
            qs_idx_local,
            envs_idx=envs_idx,
            zero_velocity=zero_velocity,
            skip_forward=skip_forward,
        )

    def set_friction(self, friction: float) -> None:
        """Set the friction coefficient for all of the entity's geoms.

        The friction coefficient must be in range [1e-2, 5.0] for simulation stability.

        Parameters
        ----------
        friction : float
            The friction coefficient to set for all the entity's links.
        """
        self._entity.set_friction(friction)

    def set_friction_ratio(self, friction_ratio, ls_idx_local, envs_idx=None) -> None:
        """Set per-link friction ratio (Genesis).

        Parameters
        ----------
        friction_ratio : torch.Tensor
            Follows the Genesis contract — shaped ``(len(envs_idx), len(ls_idx_local))``
            (i.e. already indexed by ``envs_idx``), or any shape broadcastable to it.
        ls_idx_local : array_like
            Local link indices whose geoms receive the ratio.
        envs_idx : None | array_like, optional
            Environments to write; ``None`` writes all. Defaults to ``None``.
        """
        self._entity.set_friction_ratio(friction_ratio, ls_idx_local, envs_idx=envs_idx)

    # ------------------------------------------------------------------
    # Mass / COM shifts
    #
    # Genesis holds a single per-link mass and COM channel — the additive
    # ``mass_shift`` / ``COM_shift`` state buffers are gone — so the shift API
    # is defined against a *nominal* captured from the solver on first use:
    # after the build and any build-time mass fix-ups, before the first
    # randomization write. Setters write ``nominal + shift`` as absolute
    # values; the readback getters report ``current - nominal``, so
    # privileged-critic observation terms (see
    # ``eden.managers.terms.observations.privileged``) always see exactly the
    # randomized dynamics currently in effect. Per-env shifts require
    # ``EnvOptions.batch_links_info=True`` — link info is shared across the
    # batch otherwise, and Genesis rejects ``envs_idx`` on the write.
    # ------------------------------------------------------------------

    def _nominal_links_inertials(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-link (mass, COM) the shift APIs treat as nominal.

        Shapes are Genesis' raw ones: ``(n_links,)`` / ``(n_links, 3)``, with a leading
        ``n_envs`` axis when link info is batched (:attr:`has_batched_links_info`).
        """
        nominal = getattr(self, "_nominal_links_inertials_cache", None)
        if nominal is None:
            nominal = (self._entity.get_links_mass().clone(), self._entity.get_links_COM().clone())
            self._nominal_links_inertials_cache = nominal
        return nominal

    def links_with_anchored_inertials(self, ls_idx_local, quantity: Literal["mass", "COM"]) -> list[str]:
        """Names of the selected links whose ``quantity`` Genesis refuses to write after build.

        Genesis anchors the frame of an *aligned* free body (``EntityOptions.align``, on by default
        for one free rigid body with nothing articulated under it, and always for a primitive) on the
        centre of mass and principal axes of all its links, and then rejects a runtime write that
        would move the anchor: any COM write, and a mass write unless the body is that one link.
        Mirrors the guard in genesis' ``RigidSolver._set_links_info`` so a randomization event can
        refuse at ``build()`` rather than on its first write. That guard also skips a link index held
        on a GPU outside ``gs.debug`` (reading it back would stall the device), which is how Eden's
        events pass it, so on GPU the unsupported write would otherwise go through unchecked.

        Parameters
        ----------
        ls_idx_local : array_like
            Local link indices the caller will write.
        quantity : {"mass", "COM"}
            Which inertial property will be written.

        Returns
        -------
        list[str]
            The offending link names, in ``ls_idx_local`` order; empty when every write is allowed.
        """
        solver_links = self._env.rigid_solver.links

        def anchor_of(link):
            # A link hangs off its body's anchor through fixed joints only.
            while link.parent_idx != -1 and all(joint.type == gs.JOINT_TYPE.FIXED for joint in link.joints):
                link = solver_links[link.parent_idx]
            return link.idx if link.aligned else None

        anchors = [anchor_of(link) for link in solver_links]
        blocked = []
        for i_l in torch.as_tensor(ls_idx_local).reshape(-1).tolist():
            link = self._entity.links[i_l]
            anchor = anchors[link.idx]
            if anchor is None or (quantity == "mass" and anchors.count(anchor) == 1):
                continue
            blocked.append(link.name)
        return blocked

    def _select_links_info(self, info: torch.Tensor, ls_idx_local, envs_idx) -> torch.Tensor:
        """Select ``[envs_idx, ls_idx_local]`` from a cached links-info tensor.

        Mirrors the shapes Genesis expects on an info write: the leading env axis is
        present (and narrowed to ``envs_idx``) only when link info is batched.
        """
        ls = slice(None) if ls_idx_local is None else torch.as_tensor(ls_idx_local, device=self.device).reshape(-1)
        if not self.has_batched_links_info:
            return info[ls]
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        return info[envs_idx][:, ls]

    def _links_write_base(self, info: torch.Tensor, ls_idx_local, envs_idx) -> tuple[torch.Tensor, object, object]:
        """Nominal rows for a shift write, plus the exact link and env selectors to hand Genesis.

        One selector serves both the nominal lookup and the write, so the rows the shift is
        added to and the rows Genesis writes can never disagree. The links are always an index
        tensor: Genesis' ``set_links_mass`` runs a host-side check on any other form (list,
        tuple, int, ``range``, ``None``) that counts a link fixed to the world as massless and
        raises when every selected link is one, while the same selection as a tensor is written
        — so the Python type of the selection would decide the outcome. In batched mode
        the envs are normalized to an index tensor (bool masks included) rather than trusting
        Genesis to interpret the raw form identically. When link info is unbatched the raw
        ``envs_idx`` is passed through untouched, so a per-env write hits Genesis' own clear
        "non-batched links info" error.
        """
        if ls_idx_local is None:
            ls = torch.arange(self.num_links, device=self.device)
        else:
            ls = torch.as_tensor(ls_idx_local, device=self.device).reshape(-1)
        if not self.has_batched_links_info:
            return info[ls], ls, envs_idx
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        if envs_idx.dtype == torch.bool:
            envs_idx = torch.nonzero(envs_idx, as_tuple=True)[0]
        return info[envs_idx][:, ls], ls, envs_idx

    def set_mass_shift(self, mass_shift, ls_idx_local, envs_idx=None) -> None:
        """Set per-link mass shift, in kg, relative to the nominal (unrandomized) mass.

        The resulting mass (``nominal + mass_shift``) must stay strictly positive — neither
        Eden nor Genesis checks it here, since a device-side check would synchronize every
        write. The mass-randomization event terms validate their ranges at ``build()``.

        A link fixed to the world (``links[i].is_fixed``: every joint from it to the root is
        fixed, as on a fixed-base arm's base link, a mounted hand's palm or a ``fixed=True``
        prop) is written like any other, whatever form ``ls_idx_local`` takes, and
        :meth:`get_mass_shift` reads the value back — but the solver never reads that link's
        mass, so the shift changes no dynamics. The mass-randomization event terms reject such
        a link at ``build()``.

        Parameters
        ----------
        mass_shift : torch.Tensor
            Shaped ``(len(envs_idx), len(ls_idx_local))`` (already indexed by
            ``envs_idx``), or broadcastable to it. Per-env values require
            ``EnvOptions.batch_links_info=True``.
        ls_idx_local : array_like
            Local link indices to shift.
        envs_idx : None | array_like, optional
            Environments to write; ``None`` writes all. Defaults to ``None``.
        """
        nominal_mass, _ = self._nominal_links_inertials()
        base, ls_idx_local, envs_idx = self._links_write_base(nominal_mass, ls_idx_local, envs_idx)
        self._entity.set_links_mass(base + mass_shift, ls_idx_local, envs_idx=envs_idx)

    def set_COM_shift(self, com_shift, ls_idx_local, envs_idx=None) -> None:
        """Set per-link COM shift, in meters in the link frame, relative to the nominal COM.

        As with :meth:`set_mass_shift`, a link fixed to the world is written and reads back, but
        its COM changes no dynamics; :class:`~eden.managers.terms.events.domain_rand.RandomizeComShift`
        rejects such a link at ``build()``.

        Parameters
        ----------
        com_shift : torch.Tensor
            Shaped ``(len(envs_idx), len(ls_idx_local), 3)`` (already indexed by
            ``envs_idx``), or broadcastable to it. Per-env values require
            ``EnvOptions.batch_links_info=True``.
        ls_idx_local : array_like
            Local link indices to shift.
        envs_idx : None | array_like, optional
            Environments to write; ``None`` writes all. Defaults to ``None``.
        """
        _, nominal_com = self._nominal_links_inertials()
        base, ls_idx_local, envs_idx = self._links_write_base(nominal_com, ls_idx_local, envs_idx)
        self._entity.set_links_COM(base + com_shift, ls_idx_local, envs_idx=envs_idx)

    def _first_geom_per_link(self, ls_idx_local) -> torch.Tensor:
        """Global index of the first geom of each requested link (``None`` = all links).

        Genesis stores friction ratio per geom; ``set_friction_ratio`` repeats a
        single per-link value across every geom of the link, so the first geom
        recovers the per-link ratio.
        """
        # Memoized per resolved selection: reading an index tensor back with `.tolist()` and uploading the
        # geom ids would be a GPU->CPU sync each, every step a privileged observation reads friction.
        selection = range(self.num_links) if ls_idx_local is None else _genesis_selection(ls_idx_local)
        key = (selection.start, selection.stop, selection.step) if isinstance(selection, range) else None
        cache = self.__dict__.setdefault("_first_geom_idx_cache", {})
        if key is not None and key in cache:
            return cache[key]
        local = selection if key is not None else torch.as_tensor(selection).reshape(-1).tolist()
        geoms = torch.tensor([self.links[int(i)].geom_start for i in local], dtype=torch.long, device=self.device)
        if key is not None:
            cache[key] = geoms
        return geoms

    def get_mass_shift(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Per-env link mass shift (kg) vs the nominal (unrandomized) mass; 0 if none.

        Reads the live Genesis solver value and subtracts the nominal, so the result
        always reflects the randomization currently in effect.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            Local link indices to read; ``None`` reads all links. Defaults to ``None``.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            Environments to read (int / slice / index tensor / bool mask); ``None``
            reads all. Defaults to ``None``.

        Returns
        -------
        torch.Tensor
            Copy shaped ``(num_envs_or_subset, n_links_or_subset)``.
        """
        nominal_mass, _ = self._nominal_links_inertials()
        live = self.get_links_inertial_mass(ls_idx_local, envs_idx)  # (n_envs_sel, n_ls), a copy
        return live - self._select_links_info(nominal_mass, ls_idx_local, envs_idx)

    def get_friction_ratio(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Per-env link friction ratio applied by friction-randomization events; 1.0 if none.

        Reads the live Genesis solver state (per link, via the link's first geom).

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            Local link indices to read; ``None`` reads all links. Defaults to ``None``.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            Environments to read (int / slice / index tensor / bool mask); ``None``
            reads all. Defaults to ``None``.

        Returns
        -------
        torch.Tensor
            Copy shaped ``(num_envs_or_subset, n_links_or_subset)``.
        """
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        out = self._env.rigid_solver.get_geoms_friction_ratio(self._first_geom_per_link(ls_idx_local), envs_idx)
        return out.clone()

    def get_COM_shift(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Per-env link COM shift (m, link frame) vs the nominal (unrandomized) COM; 0 if none.

        Reads the live Genesis solver value and subtracts the nominal, so the result
        always reflects the randomization currently in effect.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            Local link indices to read; ``None`` reads all links. Defaults to ``None``.
        envs_idx : None | int | slice | array_like | torch.Tensor, optional
            Environments to read (int / slice / index tensor / bool mask); ``None``
            reads all. Defaults to ``None``.

        Returns
        -------
        torch.Tensor
            Copy shaped ``(num_envs_or_subset, n_links_or_subset, 3)``.
        """
        _, nominal_com = self._nominal_links_inertials()
        base = self._select_links_info(nominal_com, ls_idx_local, envs_idx)
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        ls_idx_local = _genesis_selection(ls_idx_local)
        if self.has_batched_links_info:
            return self._entity.get_links_COM(ls_idx_local, envs_idx) - base
        per_link = self._entity.get_links_COM(ls_idx_local)  # (n_links, 3)
        live = per_link.unsqueeze(0).expand(self._env.num_envs, *per_link.shape)[envs_idx]
        return live - base

    def set_dofs_force_range(self, lower, upper, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the force range (the drive's saturation cap) of the entity's dofs.

        Parameters
        ----------
        lower : array_like
            The lower force limit, ordered to match ``dofs_idx_local``.
        upper : array_like
            The upper force limit, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        self._entity.set_dofs_force_range(lower, upper, dofs_idx_local, envs_idx)

    def set_dofs_stiffness(self, stiffness, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the joint-spring stiffness of the entity's dofs.

        Parameters
        ----------
        stiffness : array_like
            The spring stiffness, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        self._entity.set_dofs_stiffness(stiffness, dofs_idx_local, envs_idx)

    def set_dofs_armature(self, armature, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the armature (added rotor inertia) of the entity's dofs.

        Under ``EnvOptions.integrator="euler"``, ``post_build`` already folded ``kd * substep_dt``
        into the armature of actuated entities; overwriting it here discards that implicit-damping
        term. Under the implicit integrators there is no fold to lose.

        Parameters
        ----------
        armature : array_like
            The armature, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        self._entity.set_dofs_armature(armature, dofs_idx_local, envs_idx)

    def set_dofs_damping(self, damping, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the viscous damping of the entity's dofs.

        Parameters
        ----------
        damping : array_like
            The damping, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        self._entity.set_dofs_damping(damping, dofs_idx_local, envs_idx)

    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the positional (proportional) gain of the entity's PD controllers.

        Parameters
        ----------
        kp : array_like
            The positional gain, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        self._entity.set_dofs_kp(kp, dofs_idx_local, envs_idx)

    def set_dofs_kd(self, kd, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the velocity (derivative) gain of the entity's PD controllers.

        Genesis names this gain ``kv``; Eden exposes it as ``kd`` throughout.

        Parameters
        ----------
        kd : array_like
            The velocity gain, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        self._entity.set_dofs_kv(kd, dofs_idx_local, envs_idx)

    def set_dofs_vel(
        self,
        velocity=None,
        dofs_idx_local=None,
        envs_idx=None,
        *,
        skip_forward=False,
    ) -> None:
        """Set the entity's dofs' velocity.

        Parameters
        ----------
        velocity : None | array_like, optional
            The velocity to write, ordered to match ``dofs_idx_local``. If None, zero.
            With a bool-mask ``envs_idx``, every env's row is accepted too (see :meth:`set_qpos`).
            Defaults to None.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        skip_forward : bool, optional
            Whether to skip the forward pass that propagates the new velocities. Set True
            only when a later setter in the same reset will trigger it anyway. Defaults to False.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        velocity, envs_idx = _mask_rows(
            velocity, envs_idx, on_device=isinstance(dofs_idx_local, range) and self.setters_take_every_row
        )
        self._entity.set_dofs_velocity(velocity, dofs_idx_local, envs_idx, skip_forward=skip_forward)

    def set_dofs_frictionloss(self, frictionloss, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the dry-friction loss (stick band) of the entity's dofs.

        Parameters
        ----------
        frictionloss : array_like
            The dry-friction loss, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        self._entity.set_dofs_frictionloss(frictionloss, dofs_idx_local, envs_idx)

    def set_dofs_pos(
        self,
        position,
        dofs_idx_local=None,
        envs_idx=None,
        *,
        zero_velocity=True,
    ) -> None:
        """Teleport the entity's dofs to a position.

        This writes state directly. To command a PD target instead, use
        :meth:`control_dofs_pos`.

        Parameters
        ----------
        position : array_like, shape (n_dofs,) or (len(envs_idx), n_dofs)
            The positions to write, ordered to match ``dofs_idx_local``. With a bool-mask
            ``envs_idx``, every env's row ``(num_envs, n_dofs)`` is accepted too, and the mask picks
            which are written — with :attr:`setters_take_every_row`, the sync-free form for a reset.
        dofs_idx_local : None | array_like, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the entity's velocities after the write, so the teleport does
            not leave stale momentum behind. Defaults to True.

        Notes
        -----
        Hinge and slide DOFs are written as their coordinates, ``qpos0 + position``, through
        ``set_qpos`` (which takes a bool mask without a GPU->CPU sync); a selection that reaches a
        free or ball joint's DOFs goes through Genesis ``set_dofs_position``. Either way the entity's
        DOF positions read back as ``position``.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local)
        qs = self._hinge_qs(dofs_idx_local) if self.setters_take_every_row else None
        if qs is None:
            # Genesis' kernel takes the selected rows only.
            position, envs_idx = _mask_rows(position, envs_idx, on_device=False)
            self._entity.set_dofs_position(
                position,
                _kernel_selection(dofs_idx_local),
                envs_idx,
                zero_velocity=zero_velocity,
            )
            return
        # Hinge / slide DOFs, written as the coordinates Genesis derives from them (``qpos0 + position``):
        # its ``set_qpos`` takes a bool mask on the device, with every env's row or the selected ones,
        # where ``set_dofs_position`` runs only as a kernel over int ids — a ``nonzero()`` of the mask
        # and an error-flag write through the ids, two GPU->CPU syncs per call. It also leaves the
        # other entities alone: ``set_dofs_position`` re-derives every entity's qpos in those envs.
        position = torch.as_tensor(position, dtype=gs.tc_float, device=gs.device)
        n_rows = position.shape[0] if position.ndim == 2 else 1
        qpos = self._qpos0_rows(qs, envs_idx, n_rows) + position
        self._entity.set_qpos(qpos, qs, envs_idx, zero_velocity=zero_velocity)

    def control_dofs_force(self, force, dofs_idx_local=None, envs_idx=None) -> None:
        """Command a feed-forward force on the entity's dofs for the next step.

        Bypasses the PD controller: the commanded force is applied directly, still
        clipped to the force range. The command persists until it is overwritten.

        Parameters
        ----------
        force : array_like, shape (n_dofs,) or (len(envs_idx), n_dofs)
            The force to command, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like | DofRuns, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. A
            :class:`~eden.utils.misc.DofPermutation` or :class:`~eden.utils.misc.DofRuns`
            (``as_genesis_control_index``) writes a reordered or gapped selection through ``range`` s,
            sync-free; an index tensor or list is resolved to one of those itself. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local, control=True)
        _control_dofs(self._entity.control_dofs_force, force, dofs_idx_local, envs_idx)

    def control_dofs_vel(self, velocity, dofs_idx_local=None, envs_idx=None) -> None:
        """Command a velocity target for the entity's PD controllers.

        Sets a target, not state — the solver drives toward it through the ``kd`` gain,
        clipped to the force range. Use :meth:`set_dofs_vel` to write velocity directly.

        Parameters
        ----------
        velocity : array_like, shape (n_dofs,) or (len(envs_idx), n_dofs)
            The velocity target, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like | DofRuns, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. A
            :class:`~eden.utils.misc.DofPermutation` or :class:`~eden.utils.misc.DofRuns`
            (``as_genesis_control_index``) writes a reordered or gapped selection through ``range`` s,
            sync-free; an index tensor or list is resolved to one of those itself. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local, control=True)
        _control_dofs(self._entity.control_dofs_velocity, velocity, dofs_idx_local, envs_idx)

    def control_dofs_pos(self, position, dofs_idx_local=None, envs_idx=None) -> None:
        """Command a position target for the entity's PD controllers.

        Sets a target, not state — the solver drives toward it through the ``kp``/``kd``
        gains, clipped to the force range. Use :meth:`set_dofs_pos` to teleport instead.
        This is the setter behind the joint-position action terms.

        Parameters
        ----------
        position : array_like, shape (n_dofs,) or (len(envs_idx), n_dofs)
            The position target, ordered to match ``dofs_idx_local``.
        dofs_idx_local : None | array_like | DofRuns, optional
            Local DOF indices. If None, all of the entity's dofs in dofs_name order. A
            :class:`~eden.utils.misc.DofPermutation` or :class:`~eden.utils.misc.DofRuns`
            (``as_genesis_control_index``) writes a reordered or gapped selection through ``range`` s,
            sync-free; an index tensor or list is resolved to one of those itself. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self._genesis_dofs_idx
        else:
            dofs_idx_local = _genesis_selection(dofs_idx_local, control=True)
        _control_dofs(self._entity.control_dofs_position, position, dofs_idx_local, envs_idx)

    def zero_all_dofs_velocity(self, envs_idx=None, *, skip_forward: bool = False) -> None:
        """Zero the velocity of every one of the entity's dofs, including a floating base.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        skip_forward : bool, optional
            Skip the forward-kinematics pass. Only when a later write to the same envs runs one (for example a
            ``set_dofs_pos``) before anything reads link state. Defaults to False.
        """
        self._entity.zero_all_dofs_velocity(envs_idx, skip_forward=skip_forward)

    # ------------------------------------------------------------------------------------
    # ---------------------------- support surface related -------------------------------
    # ------------------------------------------------------------------------------------
    def _run_broad_phase(self) -> None:
        """Run Genesis' rigid-collider broad phase to refresh the broad collision-pair state.

        Genesis-internal API: reaches into the rigid solver's private state/config buffers
        (``rigid_info``, ``rigid_config``, ``collider.collider_state``/``collider_info``/
        ``collider_config``, ``_errno``). Isolated here so a Genesis collider refactor breaks
        this one named method rather than the body of :meth:`place_on_to`. The argument list
        mirrors the ``func_broad_phase`` call in Genesis' ``func_detection``; the kernel that wraps it
        lives in :func:`eden.entities.support_mixin._broad_phase_kernel`.
        """
        from eden.entities.support_mixin import _broad_phase_kernel

        _broad_phase_kernel()(
            self.solver.geoms_init_AABB,
            self.solver.dyn_state,
            self.solver.collider.collider_state,
            self.solver.constraint_solver.constraint_state,
            self.solver.dyn_info,
            self.solver.rigid_info,
            self.solver.collider.collider_info,
            self.solver.rigid_config,
            self.solver.collider.collider_config,
            self.solver._errno,
        )

    def place_on_to(
        self,
        entity: RigidEntity,
        sampler: Callable,
        envs_idx: slice | torch.Tensor | None = None,
    ):
        """Place this entity onto the support surface of another entity.

        Parameters
        ----------
        entity: RigidEntity
            the entity on which this object will be placed
        sampler: Callable
            the sampler to use for location sampling
        envs_idx: None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self._env.num_envs, return_n_envs=True)

        AABB = self.get_AABB()
        AABB = AABB[envs_idx]
        # NOTE: support bool envs_idx
        if AABB.shape[0] == 0:
            return
        clearance = (AABB[:, 1, 2] - AABB[:, 0, 2]).max(dim=0)[0]
        support = entity.current_support(
            minimum_clearance=clearance,
            gravity=self._env.rigid_solver.get_gravity(envs_idx=envs_idx),
            envs_idx=envs_idx,
        )
        # NOTE: force to update collision state
        entity.solver.collider.clear()
        # NOTE: update slightly bigger AABBs
        entity._kernel_update_aabbs(
            self.solver.dyn_state.geoms.pos,
            self.solver.dyn_state.geoms.quat,
            self.solver.geoms_init_AABB,
            self.solver.dyn_state.geoms.aabb_min,
            self.solver.dyn_state.geoms.aabb_max,
        )
        # NOTE: use broad phase to check the occupied space on the support surface
        self._run_broad_phase()
        tensor = torch.zeros(
            (self._env.num_envs, self.solver.n_entities, 2, 3),
            dtype=gs.tc_float,
            device=self.device,
        )
        tensor[..., 0, :] = torch.inf
        tensor[..., 1, :] = -torch.inf
        valid_mask = torch.zeros(
            (self._env.num_envs, self.solver.n_entities),
            dtype=gs.tc_int,
            device=self.device,
        )
        # NOTE: get set of AABBs that is already occupied
        entity._kernel_filter_detection(
            tensor,
            valid_mask,
            self.solver.collider.collider_state.n_broad_pairs,
            self.solver.collider.collider_state.broad_collision_pairs,
            self.solver.dyn_info.geoms.link_idx,
            self.solver.dyn_info.links.entity_idx,
            self.solver.dyn_info.entities.geom_start,
            self.solver.dyn_info.entities.geom_end,
            self.solver.dyn_state.geoms.aabb_min,
            self.solver.dyn_state.geoms.aabb_max,
        )
        AABBs = tensor[:, (valid_mask == 1).any(dim=0)]
        AABBs = AABBs[envs_idx]

        pos, offset_direction, valid_mask = sampler(
            support,
            asset_AABB=AABB,
            holes=AABBs,
            batch_size=n_envs,
        )
        pos += torch.sum(self.offset * offset_direction, dim=-1, keepdim=True) * offset_direction
        if self.is_attached:
            raise RuntimeError(
                f"Cannot place '{self.name}': it is welded onto another entity, so its pose is "
                f"determined by the parent link and any write here is discarded by the next "
                f"forward-kinematics pass. Place the entity it is attached to instead."
            )
        if self.is_fixed_base and not self.morph.batch_fixed_verts:
            self.set_pos(pos)
        else:
            if valid_mask is not None:
                pos = pos[valid_mask]
                if isinstance(envs_idx, slice):
                    envs_idx = torch.arange(*envs_idx.indices(self._env.num_envs))[valid_mask.cpu()]
                else:
                    envs_idx = envs_idx[valid_mask]
            if isinstance(envs_idx, torch.Tensor):
                envs_idx = envs_idx.to(self.device)
            self.set_pos(pos, envs_idx=envs_idx)

    @cached_property
    def offset(self):
        if not self.is_heterogeneous:
            offset = torch.zeros(3, device=self.device, dtype=gs.tc_float)
            if self.is_fixed_base and not self.morph.batch_fixed_verts:
                pos_ = self.get_pos()
                self.set_pos(torch.zeros(self._env.num_envs, 3, device=self.device))
            else:
                pos_ = self.get_pos(envs_idx=0)
                self.set_pos(torch.zeros(1, 3, device=self.device), envs_idx=0)
            AABB = self.get_AABB()
            offset[:2] = -AABB[0, :, :2].mean(dim=-2)
            offset[2] = -AABB[0, 0, 2] + 0.001
            if self.is_fixed_base and not self.morph.batch_fixed_verts:
                self.set_pos(pos_)
            else:
                self.set_pos(pos_, envs_idx=0)
            return offset.unsqueeze(0)
        else:
            offset = torch.zeros(self._env.num_envs, 3, device=self.device, dtype=gs.tc_float)
            pos_ = self.get_pos()
            self.set_pos(torch.zeros(self._env.num_envs, 3, device=self.device))
            AABB = self.get_AABB()
            offset[:, :2] = -AABB[:, :, :2].mean(dim=-2)
            offset[:, 2] = -AABB[:, 0, 2] + 0.001
            self.set_pos(pos_)
            return offset
