"""An indexed table of simulator states that a RESET event restores.

Row ``i`` of a :class:`ResetStateBuffer` is one *reset configuration*: the full state of some entities, plus
optional task data (a goal, a terrain cell) and an optional category label. ``ResetFromStateBuffer``
(``eden/managers/terms/events/state_buffer.py``) owns one, asks its ``ResetConfigSampler`` for a row per
resetting env, and writes those rows into the sim. Rows get into a buffer three ways:

- **precollected**: :meth:`ResetStateBuffer.load` a file a generation script wrote (OmniReset's reset families);
- **generated**: build one in Python, from procedural poses, IK, or :func:`capture_reset_states` after running
  a task's own randomized resets, then :meth:`~ResetStateBuffer.save` it;
- **harvested**: :func:`capture_reset_states` on the live sim, written into slots with
  :meth:`~ResetStateBuffer.write_rows`. That is the primitive a ring-buffer harvester (PSI) is built on.

Frames and orders, the part a hand-written buffer gets wrong:

- ``root_pos`` / ``root_quat`` are the **authored** root frame, ``get_pos(relative=True)`` /
  ``get_quat(relative=True)``, which is the frame ``default_root_pos`` is in, not the centre-of-mass frame Genesis
  re-origins a free body onto. Quaternions are wxyz.
- ``root_lin_vel`` is the world-frame velocity of that authored origin, and ``root_ang_vel`` the world-frame
  angular velocity. Neither is the free joint's DOF velocity, whose angular half is in the body frame.
- A **fixed base** has a root pose to store only when it is built with ``batch_fixed_verts=True``: Genesis then
  keeps its pose per env (a robot base or fixture placed differently in each row), otherwise one pose for every
  env. It never has a root velocity. :func:`stores_root_pose` is the rule.
- DOFs are stored **per joint name**, joint-major, for every non-free joint that carries DOFs
  (:attr:`RigidEntity.joints`). That is not ``dofs_name``, which is the *controlled* subset and omits a
  closed-loop gripper's passive followers. Values are the DOF state as Genesis holds it (``get_dofs_pos``):
  ``qpos - qpos0`` for a hinge or slide, Euler angles for a ball joint.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from eden.utils.misc import env_indices, sanitize_envs_idx

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidJoint

    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase

#: The ``format`` tag :meth:`ResetStateBuffer.save` writes and :meth:`ResetStateBuffer.load` requires.
FORMAT = "eden.reset_state_buffer"
#: Bumped whenever the on-disk layout changes meaning; a file from another version is refused, not reinterpreted.
FORMAT_VERSION = 1

_ROOT_WIDTHS = {"root_pos": 3, "root_quat": 4, "root_lin_vel": 3, "root_ang_vel": 3}
#: How far a stored ``root_quat``'s norm may sit from 1: float32 rounding passes, an unnormalized quaternion does not.
_QUAT_NORM_ATOL = 1e-3
_TENSOR_FIELDS = ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "dofs_pos", "dofs_vel")


@dataclass(frozen=True)
class EntityStates:
    """One entity's columns of a :class:`ResetStateBuffer`, one row per stored state.

    A free-base entity carries ``root_pos`` and ``root_quat``; an immobile one (fixed or welded) carries neither.
    Velocities are optional and read as zero when absent. ``joints_name`` / ``joints_n_dofs`` name the joints the
    ``dofs_*`` columns hold, in column order. See the module docstring for every frame.
    """

    root_pos: torch.Tensor | None = None
    root_quat: torch.Tensor | None = None
    root_lin_vel: torch.Tensor | None = None
    root_ang_vel: torch.Tensor | None = None
    joints_name: tuple[str, ...] = ()
    joints_n_dofs: tuple[int, ...] = ()
    dofs_pos: torch.Tensor | None = None
    dofs_vel: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # Tuples whatever was passed: layout() compares them, and a list never equals a tuple.
        object.__setattr__(self, "joints_name", tuple(self.joints_name))
        object.__setattr__(self, "joints_n_dofs", tuple(self.joints_n_dofs))

    @property
    def has_root(self) -> bool:
        return self.root_pos is not None

    def tensors(self) -> dict[str, torch.Tensor]:
        """The columns this entity carries, by field name."""
        return {name: getattr(self, name) for name in _TENSOR_FIELDS if getattr(self, name) is not None}

    def map(self, fn) -> EntityStates:
        """These states with ``fn`` applied to every column (a gather, a device move)."""
        return replace(self, **{name: fn(value) for name, value in self.tensors().items()})

    def layout(self) -> tuple:
        """What two buffers must agree on to share rows: which columns exist, and the joint layout."""
        return (tuple(self.tensors()), self.joints_name, self.joints_n_dofs)


def _column_kind(value: torch.Tensor) -> str:
    """A task column's dtype as :meth:`ResetStateBuffer.layout` compares it: any float precision is ``'float'``."""
    return "float" if value.is_floating_point() else str(value.dtype).removeprefix("torch.")


def _describe(key: str, layout: Any) -> str:
    """One side of a :meth:`ResetStateBuffer.layout_differences` line, readable. Total: never raises."""
    if key.startswith("entity") and isinstance(layout, tuple) and len(layout) == 3:
        columns, joints, n_dofs = layout
        joint_text = ", ".join(f"{j}:{n}" for j, n in zip(joints, n_dofs)) or "none"
        return f"columns [{', '.join(columns)}], joints [{joint_text}]"
    return repr(layout)


def _check_entity_states(name: str, states: EntityStates) -> int | None:
    """Validate one entity's columns; return their row count (``None`` when it holds no column at all)."""
    where = f"ResetStateBuffer entity '{name}'"
    if (states.root_pos is None) != (states.root_quat is None):
        raise ValueError(
            f"{where} has only one of root_pos / root_quat. A free base's pose needs both; an immobile base needs "
            "neither. Fix: pass both, or drop both."
        )
    if not states.has_root and (states.root_lin_vel is not None or states.root_ang_vel is not None):
        raise ValueError(
            f"{where} has root velocities but no root pose. Fix: pass root_pos and root_quat, or drop the velocities."
        )
    if len(states.joints_name) != len(states.joints_n_dofs):
        raise ValueError(
            f"{where}: joints_name has {len(states.joints_name)} names but joints_n_dofs has "
            f"{len(states.joints_n_dofs)} counts. Fix: give one DOF count per joint, in the same order."
        )
    if len(set(states.joints_name)) != len(states.joints_name):
        repeated = sorted({j for j in states.joints_name if states.joints_name.count(j) > 1})
        raise ValueError(f"{where} names joint(s) {repeated} more than once. Fix: list each joint once.")
    if states.joints_name and states.dofs_pos is None:
        raise ValueError(
            f"{where} names joints {list(states.joints_name)} but has no dofs_pos. Fix: pass dofs_pos, one column per "
            "DOF of those joints."
        )
    if not states.joints_name and (states.dofs_pos is not None or states.dofs_vel is not None):
        raise ValueError(
            f"{where} has DOF columns but names no joints. Fix: pass joints_name and joints_n_dofs for the columns."
        )
    width = sum(states.joints_n_dofs)
    rows = None
    for field_name, value in states.tensors().items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"{where}: {field_name} must be a 2-D tensor (rows, width), got {shape}.")
        if not value.is_floating_point():
            raise ValueError(
                f"{where}: {field_name} has dtype {value.dtype}; entity columns hold floats, and an integer one would "
                "truncate every row written into it. Fix: build it as floats, e.g. torch.tensor([[0.0, 0.0, 1.0]]) "
                "or value.float()."
            )
        expected = _ROOT_WIDTHS.get(field_name, width)
        if value.shape[1] != expected:
            what = "its DOF count, sum(joints_n_dofs)" if field_name.startswith("dofs") else "fixed"
            raise ValueError(f"{where}: {field_name} has width {value.shape[1]}, expected {expected} ({what}).")
        if rows is None:
            rows = value.shape[0]
        elif value.shape[0] != rows:
            raise ValueError(
                f"{where}: {field_name} has {value.shape[0]} rows but the entity's other columns have {rows}. "
                "Every column holds one row per stored state."
            )
    for field_name, value in states.tensors().items():
        bad_rows = (~torch.isfinite(value)).any(dim=-1).nonzero().flatten()
        if bad_rows.numel():
            raise ValueError(
                f"{where}: {field_name} holds NaN or inf in {bad_rows.numel()} row(s), first {bad_rows[:8].tolist()}. "
                "Fix: drop or regenerate those states; a non-finite state written into the sim spreads to every body "
                "it touches."
            )
    if states.root_quat is not None:
        norms = states.root_quat.norm(dim=-1)
        bad_rows = ((norms - 1.0).abs() > _QUAT_NORM_ATOL).nonzero().flatten()
        if bad_rows.numel():
            raise ValueError(
                f"{where}: root_quat is not unit-norm in {bad_rows.numel()} row(s), first {bad_rows[:8].tolist()} "
                f"(norms {[round(float(n), 4) for n in norms[bad_rows[:8]]]}). Genesis writes it into qpos as given. "
                "Fix: normalize it where the buffer is generated, root_quat / root_quat.norm(dim=-1, keepdim=True)."
            )
    return rows


def _row_indices(rows: torch.Tensor | Sequence[int], num_states: int, device: torch.device, where: str) -> torch.Tensor:
    """``rows`` (row indices, or a ``(num_states,)`` bool mask over the rows) as a long index tensor on ``device``."""
    rows = torch.as_tensor(rows, device=device)
    if rows.dtype == torch.bool:
        if rows.shape != (num_states,):
            raise ValueError(
                f"{where}: a bool mask of shape {tuple(rows.shape)} for a buffer of {num_states} rows; pass a "
                f"({num_states},) mask, or the row indices."
            )
        return rows.nonzero()[:, 0]
    if rows.numel() and (rows.is_floating_point() or rows.is_complex()):
        raise ValueError(
            f"{where}: row indices must be integers or a bool mask, got dtype {rows.dtype}. Fix: pass them as a long "
            "tensor; casting would truncate them onto other rows."
        )
    return rows.long()


class ResetStateBuffer:
    """N reset states: per-entity sim state, task columns, and an optional category label per row.

    Parameters
    ----------
    entities : Mapping[str, EntityStates]
        The entities a reset from this buffer writes, by scene name. Entities not listed keep whatever the scene
        reset and the other RESET events leave them at.
    columns : Mapping[str, torch.Tensor] | None
        Task data, one leading row per state (a goal pose, a terrain cell id). Opaque to the buffer; a consumer
        reads it through ``ResetFromStateBuffer.current_column``.
    category : torch.Tensor | None
        ``(N,)`` integer family label per row (OmniReset's Reaching / Stable-Grasp / Near-Goal), indexing
        ``category_names``. :meth:`from_families` builds both.
    category_names : Sequence[str]
        The family names ``category`` indexes.
    """

    def __init__(
        self,
        entities: Mapping[str, EntityStates],
        columns: Mapping[str, torch.Tensor] | None = None,
        category: torch.Tensor | None = None,
        category_names: Sequence[str] = (),
    ) -> None:
        if not entities:
            raise ValueError(
                "ResetStateBuffer needs at least one entity: a reset that writes nothing restores nothing."
            )
        counts: dict[str, int] = {}
        for name, states in entities.items():
            if not isinstance(states, EntityStates):
                raise TypeError(
                    f"ResetStateBuffer entity '{name}' must be an EntityStates, got {type(states).__name__}."
                )
            rows = _check_entity_states(name, states)
            if rows is None:
                raise ValueError(
                    f"ResetStateBuffer entity '{name}' holds no column. Fix: give it a root pose or joint states, "
                    "or leave it out of the buffer."
                )
            counts[f"entity '{name}'"] = rows
        columns = dict(columns or {})
        for name, value in columns.items():
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise ValueError(f"ResetStateBuffer column '{name}' must be a tensor with a leading row axis.")
            counts[f"column '{name}'"] = value.shape[0]
        category_names = tuple(category_names)
        if category is not None:
            if category.ndim != 1 or category.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"ResetStateBuffer category must be a 1-D integer tensor, got shape {tuple(category.shape)} "
                    f"dtype {category.dtype}."
                )
            if not category_names:
                raise ValueError("ResetStateBuffer category is set but category_names is empty: name each family.")
            if category.numel() and not (0 <= int(category.min()) and int(category.max()) < len(category_names)):
                raise ValueError(
                    f"ResetStateBuffer category values span [{int(category.min())}, {int(category.max())}] but there "
                    f"are {len(category_names)} category_names {list(category_names)}."
                )
            counts["category"] = category.shape[0]
        elif category_names:
            raise ValueError("ResetStateBuffer category_names is set but category is None: label every row.")
        if len(set(counts.values())) != 1:
            listed = ", ".join(f"{key}: {n}" for key, n in counts.items())
            raise ValueError(f"ResetStateBuffer columns disagree on the number of states. {listed}.")
        self._num_states = next(iter(counts.values()))
        if self._num_states == 0:
            raise ValueError("ResetStateBuffer holds 0 states; a reset needs at least one to draw from.")
        self.entities: dict[str, EntityStates] = dict(entities)
        self.columns: dict[str, torch.Tensor] = columns
        self.category = None if category is None else category.long()
        self.category_names: tuple[str, ...] = category_names
        #: Bumped by every :meth:`write_rows`, so a consumer holding data derived from the rows (a sampler's grouping
        #: by category) can tell they changed without reading them.
        self.generation = 0

    # -- shape and identity ----------------------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._num_states

    def __len__(self) -> int:
        return self._num_states

    @property
    def device(self) -> torch.device:
        return next(iter(next(iter(self.entities.values())).tensors().values())).device

    def layout(self) -> dict[str, Any]:
        """What two buffers must agree on to share rows (:meth:`write_rows`, :meth:`concat`); entity order is free.

        A float column's precision is not part of it: rows captured at one precision are cast to the other on write,
        as the entity columns are, so a ``precision='64'`` env can take rows harvested at 32 bits. A float column
        never shares rows with an integer one, which would truncate it.
        """
        return {
            **{f"entity '{name}'": states.layout() for name, states in self.entities.items()},
            **{
                f"column '{name}'": (tuple(value.shape[1:]), _column_kind(value))
                for name, value in self.columns.items()
            },
            "category_names": self.category_names if self.category is not None else None,
        }

    def layout_differences(self, other: ResetStateBuffer) -> list[str]:
        """One line per way ``other``'s layout differs from this one's; empty when they can share rows."""
        mine, theirs = self.layout(), other.layout()
        lines = []
        for key in sorted(mine.keys() | theirs.keys()):
            if key not in theirs:
                lines.append(f"{key}: only in the first")
            elif key not in mine:
                lines.append(f"{key}: only in the second")
            elif mine[key] != theirs[key]:
                lines.append(f"{key}: {_describe(key, mine[key])} vs {_describe(key, theirs[key])}")
        return lines

    def __repr__(self) -> str:
        parts = []
        for name, states in self.entities.items():
            kinds = (["root"] if states.has_root else []) + (
                [f"{len(states.joints_name)} joints"] * bool(states.joints_name)
            )
            parts.append(f"{name}({', '.join(kinds)})")
        extra = f", columns={list(self.columns)}" if self.columns else ""
        cats = f", categories={list(self.category_names)}" if self.category_names else ""
        return f"ResetStateBuffer({self._num_states} states: {', '.join(parts)}{extra}{cats})"

    # -- transforms ------------------------------------------------------------------------------

    def _map(self, fn) -> ResetStateBuffer:
        return ResetStateBuffer(
            {name: states.map(fn) for name, states in self.entities.items()},
            {name: fn(value) for name, value in self.columns.items()},
            None if self.category is None else fn(self.category),
            self.category_names,
        )

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> ResetStateBuffer:
        """This buffer on ``device``; with ``dtype``, its float columns cast too (integer columns keep theirs)."""

        def move(value: torch.Tensor) -> torch.Tensor:
            return value.to(device, dtype=dtype if dtype is not None and value.is_floating_point() else value.dtype)

        return self._map(move)

    def select(self, rows: torch.Tensor | Sequence[int]) -> ResetStateBuffer:
        """The states at ``rows`` (an index tensor or sequence, or a bool mask over the rows), as a new buffer."""
        rows = _row_indices(rows, self._num_states, self.device, "ResetStateBuffer.select")
        return self._map(lambda value: value.index_select(0, rows))

    def write_rows(self, rows_idx: torch.Tensor, rows: ResetStateBuffer) -> None:
        """Overwrite the states at ``rows_idx`` with ``rows``, in place: the slot write a harvester's ring needs.

        ``rows`` must have this buffer's :meth:`layout` (same entities, columns, joints) and one row per index, and
        the indices must be distinct slots in ``[0, num_states)``, or a ``(num_states,)`` bool mask (its slots in
        ascending order). Checking them costs one device-to-host read per call: unchecked, an index out of range is
        a CUDA device-side assert that names no buffer, and a repeated one writes its rows in an unspecified order.
        """
        if differences := self.layout_differences(rows):
            listed = "\n".join(f"  {line}" for line in differences)
            raise ValueError(
                f"ResetStateBuffer.write_rows: the rows' layout differs from this buffer's (first = this buffer, "
                f"second = the rows):\n{listed}\n"
                "Fix: build the buffer from the same source as the rows: capture_reset_states with the same entity "
                "names and task columns, which stores every velocity column."
            )
        rows_idx = _row_indices(rows_idx, self._num_states, self.device, "ResetStateBuffer.write_rows")
        if rows_idx.shape != (rows.num_states,):
            raise ValueError(
                f"ResetStateBuffer.write_rows: {tuple(rows_idx.shape)} indices for {rows.num_states} rows; pass one "
                "index per row."
            )
        if rows_idx.numel():
            ordered = rows_idx.sort().values
            in_range = (ordered[0] >= 0) & (ordered[-1] < self._num_states)
            if not bool(in_range & (ordered[1:] != ordered[:-1]).all()):
                out_of_range = rows_idx[(rows_idx < 0) | (rows_idx >= self._num_states)].unique().tolist()
                repeated = ordered[1:][ordered[1:] == ordered[:-1]].unique().tolist()
                raise ValueError(
                    f"ResetStateBuffer.write_rows: slot indices must be distinct and in [0, {self._num_states}). "
                    f"Out of range: {out_of_range[:8] or 'none'}; repeated: {repeated[:8] or 'none'}. Fix: wrap a ring "
                    "pointer with % buffer.num_states, and write a slot once per call (keep its latest row)."
                )
        for name, states in self.entities.items():
            for field_name, value in states.tensors().items():
                value.index_copy_(0, rows_idx, getattr(rows.entities[name], field_name).to(value))
        for name, value in self.columns.items():
            value.index_copy_(0, rows_idx, rows.columns[name].to(value))
        if self.category is not None:
            self.category.index_copy_(0, rows_idx, rows.category.to(self.category))
        self.generation += 1

    @classmethod
    def concat(cls, buffers: Sequence[ResetStateBuffer]) -> ResetStateBuffer:
        """Stack buffers of one :meth:`layout` into one, in order."""
        if not buffers:
            raise ValueError("ResetStateBuffer.concat needs at least one buffer.")
        first = buffers[0]
        for i, other in enumerate(buffers[1:], start=1):
            if differences := first.layout_differences(other):
                listed = "\n".join(f"  {line}" for line in differences)
                raise ValueError(
                    f"ResetStateBuffer.concat: buffer {i} has a different layout from buffer 0 (first = buffer 0, "
                    f"second = buffer {i}):\n{listed}\n"
                    "Fix: capture every buffer with the same entities, joints and task columns."
                )
        device = first.device

        def cat(get) -> torch.Tensor:
            return torch.cat([get(buffer).to(device) for buffer in buffers])

        entities = {
            name: replace(
                states,
                **{
                    field_name: cat(lambda buffer, n=name, f=field_name: getattr(buffer.entities[n], f))
                    for field_name in states.tensors()
                },
            )
            for name, states in first.entities.items()
        }
        columns = {name: cat(lambda buffer, n=name: buffer.columns[n]) for name in first.columns}
        category = None if first.category is None else cat(lambda buffer: buffer.category)
        return cls(entities, columns, category, first.category_names)

    @classmethod
    def from_families(cls, families: Mapping[str, ResetStateBuffer]) -> ResetStateBuffer:
        """Concatenate per-family buffers, labelling each row with its family's name as its category.

        The OmniReset shape: one buffer per reset strategy (``{"reaching": ..., "near_goal": ...}``), one table out,
        so a sampler sees every row and per-family statistics stay recoverable.
        """
        labelled = [buffer for buffer in families.values() if buffer.category is not None]
        if labelled:
            raise ValueError(
                "ResetStateBuffer.from_families: a family buffer already carries a category; families are labelled "
                f"here, by their key. Families: {list(families)}."
            )
        names = list(families)
        buffers = [
            cls(
                buffer.entities,
                buffer.columns,
                torch.full((buffer.num_states,), i, dtype=torch.long, device=buffer.device),
                names,
            )
            for i, buffer in enumerate(families.values())
        ]
        return cls.concat(buffers)

    # -- persistence -----------------------------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        """Write the buffer to ``path``, atomically: a reader never sees a half-written file.

        Written from every process that calls it. Under DDP, the caller decides which rank writes, or gives each
        rank its own path; a silent no-op on non-main ranks would drop those ranks' harvest.
        """
        path = os.fspath(path)
        payload: dict[str, Any] = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "entities": {
                name: {
                    "joints_name": list(states.joints_name),
                    "joints_n_dofs": list(states.joints_n_dofs),
                    **{field_name: value.detach().cpu() for field_name, value in states.tensors().items()},
                }
                for name, states in self.entities.items()
            },
            "columns": {name: value.detach().cpu() for name, value in self.columns.items()},
            "category": None if self.category is None else self.category.cpu(),
            "category_names": list(self.category_names),
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike, device: torch.device | str = "cpu") -> ResetStateBuffer:
        """Read a buffer :meth:`save` wrote, onto ``device``. Tensors and plain containers only (``weights_only``)."""
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Reset-state buffer file not found: '{path}'. Fix: write one with ResetStateBuffer.save(path) "
                "(capture_reset_states builds the rows), or point path= at an existing file."
            )
        payload = torch.load(path, map_location=device, weights_only=True)
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        if found != FORMAT:
            raise ValueError(
                f"'{path}' is not a reset-state buffer: format {found!r}, expected {FORMAT!r}. "
                "Fix: convert it to ResetStateBuffer and save() it; a raw state dict has no frames or joint names to "
                "validate against."
            )
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"'{path}' is reset-state buffer format version {payload.get('version')!r}; this Eden reads version "
                f"{FORMAT_VERSION}. Fix: regenerate the buffer with this Eden."
            )
        entities = {}
        for name, raw in payload["entities"].items():
            entities[name] = EntityStates(
                joints_name=tuple(raw.get("joints_name", ())),
                joints_n_dofs=tuple(int(n) for n in raw.get("joints_n_dofs", ())),
                **{field_name: raw[field_name] for field_name in _TENSOR_FIELDS if field_name in raw},
            )
        return cls(entities, payload.get("columns") or {}, payload.get("category"), payload.get("category_names", ()))


def state_joints(entity: RigidEntity) -> list:
    """The joints a reset-state row stores for ``entity``: every non-free joint that carries DOFs, in declaration order."""
    import genesis as gs

    return [joint for joint in entity.joints if joint.n_dofs > 0 and joint.type != gs.JOINT_TYPE.FREE]


def free_base_joint(entity: RigidEntity, consumer: str) -> RigidJoint | None:
    """The free joint a reset-state row stores ``entity``'s root through, or ``None`` for an immobile base.

    Raises for any free joint a row cannot hold. A row holds one root per entity and :func:`state_joints` stores no
    free joint, so a second free body in a mobile entity, or any free body in a fixed or welded one, would keep the
    previous episode's pose on every restore.
    """
    free_joints = entity.free_joints
    if entity.has_immobile_base:
        if free_joints:
            kind = "welded onto another entity" if entity.is_attached else "fixed-base"
            raise ValueError(
                f"{consumer}: entity '{entity.name}' is {kind}, yet holds free joint(s) "
                f"{[joint.name for joint in free_joints]} on other bodies, which a reset-state row does not store: "
                "those bodies would keep the previous episode's pose on every restore. Fix: load each free body as "
                "its own scene entity."
            )
        return None
    if not free_joints:
        raise ValueError(
            f"{consumer}: entity '{entity.name}' is neither fixed nor welded, yet has no free joint, so a row has no "
            f"root to write it through. Fix: set is_fixed_base=True on '{entity.name}' if its base is meant to be fixed."
        )
    if len(free_joints) > 1:
        raise ValueError(
            f"{consumer}: entity '{entity.name}' has {len(free_joints)} free joints "
            f"{[joint.name for joint in free_joints]}, but a reset-state row stores one free root per entity. "
            "Fix: split each free body into its own scene entity. Picking one of the joints is not a fix: the others "
            "would keep the previous episode's pose on every restore."
        )
    return free_joints[0]


def stores_root_pose(entity: RigidEntity) -> bool:
    """Whether a reset-state row carries ``entity``'s root pose: a free base, or a fixed one posed per env.

    A fixed base is posed per env only when built with ``batch_fixed_verts=True``; one welded onto another entity
    follows its parent link and has no root of its own.
    """
    if not entity.has_immobile_base:
        return True
    # The built entity's flag: a per-env-variant group's ``morph`` is a list of its variants', and Genesis batches the
    # group by its first.
    return not entity.is_attached and bool(entity.entity.main_morph.batch_fixed_verts)


def capture_reset_states(
    env: EnvBase,
    entities_name: Sequence[str],
    envs_idx: torch.Tensor | Sequence[int] | None = None,
    columns: Mapping[str, torch.Tensor] | None = None,
) -> ResetStateBuffer:
    """Read the live state of ``entities_name`` in ``envs_idx`` (env ids or a ``(num_envs,)`` bool mask; default: every
    env) into a new buffer, one row per selected env, in env order for a mask.

    Reads every joint of each entity (:func:`state_joints`), so a restore writes the whole state, a closed-loop
    gripper's passive DOFs included. ``columns`` are task data for the same rows, copied like the states: a live
    tensor passed in (a command's buffer) does not drag the captured rows along when the env steps on. Not for the
    hot path: it reads Genesis state with ordinary getters.
    """
    from eden.entities.rigid import RigidEntity
    from eden.utils.geom import transform_by_quat

    envs_idx = env_indices(sanitize_envs_idx(envs_idx, env.num_envs, prefer_slice=False))
    entities: dict[str, EntityStates] = {}
    for name in entities_name:
        if name not in env.entities:
            raise KeyError(
                f"capture_reset_states: no entity '{name}' in the scene. Scene entities: {', '.join(env.entities)}."
            )
        entity = env.entities[name]
        if not isinstance(entity, RigidEntity):
            raise TypeError(
                f"capture_reset_states: entity '{name}' is a {type(entity).__name__}; only a RigidEntity's state can be "
                "captured and restored (particle and FEM states have no per-env setters)."
            )
        free_joint = free_base_joint(entity, consumer="capture_reset_states")
        root: dict[str, torch.Tensor] = {}
        if free_joint is None and stores_root_pose(entity):
            root = {
                "root_pos": entity.get_pos(envs_idx=envs_idx, relative=True),
                "root_quat": entity.get_quat(envs_idx=envs_idx, relative=True),
            }
        elif free_joint is not None:
            authored_pos = entity.get_pos(envs_idx=envs_idx, relative=True)
            solver_pos, solver_quat = entity.get_pos(envs_idx=envs_idx), entity.get_quat(envs_idx=envs_idx)
            base_vel = entity.get_dofs_vel(dofs_idx_local=list(free_joint.dofs_idx_local), envs_idx=envs_idx)
            # Free joint: linear DOFs are the solver origin's world velocity, angular DOFs are body-frame.
            ang_vel_w = transform_by_quat(base_vel[:, 3:], solver_quat)
            authored_lin_vel = base_vel[:, :3] - torch.cross(ang_vel_w, solver_pos - authored_pos, dim=-1)
            root = {
                "root_pos": authored_pos,
                "root_quat": entity.get_quat(envs_idx=envs_idx, relative=True),
                "root_lin_vel": authored_lin_vel,
                "root_ang_vel": ang_vel_w,
            }
        joints = state_joints(entity)
        joint_states: dict[str, Any] = {}
        if joints:
            dofs = [i for joint in joints for i in joint.dofs_idx_local]
            joint_states = {
                "joints_name": tuple(joint.name for joint in joints),
                "joints_n_dofs": tuple(joint.n_dofs for joint in joints),
                "dofs_pos": entity.get_dofs_pos(dofs_idx_local=dofs, envs_idx=envs_idx),
                "dofs_vel": entity.get_dofs_vel(dofs_idx_local=dofs, envs_idx=envs_idx),
            }
        tensors = {**root, **{k: v for k, v in joint_states.items() if isinstance(v, torch.Tensor)}}
        entities[name] = EntityStates(
            joints_name=joint_states.get("joints_name", ()),
            joints_n_dofs=joint_states.get("joints_n_dofs", ()),
            **{key: value.detach().clone().float() for key, value in tensors.items()},
        )
    copied = {
        name: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for name, value in (columns or {}).items()
    }
    return ResetStateBuffer(entities, copied)


__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "EntityStates",
    "ResetStateBuffer",
    "capture_reset_states",
    "free_base_joint",
    "state_joints",
    "stores_root_pose",
]
