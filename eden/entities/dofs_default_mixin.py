"""Name-based access to a rigid entity's DOFs and its configured per-DOF properties.

Two index spaces address an entity's DOFs, and only one of them indexes the property tensors:

- **column** — position in the entity's ``dofs_name``. The configured property tensors
  (``default_dofs_pos``, ``default_dofs_kp``, ``soft_dofs_pos_limits``, …) are laid out this way,
  and so is every full-width ``get_dofs_*()`` read.
- **local** — Genesis' DOF index within the entity: what ``find_named_dofs_idx_local`` returns and
  what every ``dofs_idx_local=`` argument takes. A floating base offsets it by six, an uncontrolled
  joint shifts it, and the kinematic tree can permute it. A local index used as a column therefore
  reads another joint's value, or, once it passes the last column, out of bounds (an asynchronous
  CUDA assert, far from the call site).

:meth:`DofDefaultsMixin.find_named_dofs` resolves names into both spaces at once, as a :class:`DofSelection`
whose fields say which space they belong to, and the ``get_default_dofs_*`` getters read the property
tensors by name — so no bare index from either space passes through the caller's hands.
"""

from __future__ import annotations

import reprlib
from collections.abc import Sequence
from dataclasses import dataclass

import genesis as gs
import torch

from eden.utils.misc import sanitize_envs_idx
from eden.utils.string import key_selects


@dataclass(frozen=True, eq=False)
class DofSelection:
    """Some of an entity's controlled DOFs, resolved into both index spaces at once.

    Made by :meth:`DofDefaultsMixin.find_named_dofs`, in ``build()``, and kept. The three fields list the
    same DOFs in the same order, one entry per DOF (a ball joint's name repeats once per DOF):

    .. code-block:: python

        dofs = robot.find_named_dofs([".*_knee_joint"])          # in build()
        robot.soft_dofs_pos_limits[:, dofs.columns]          # a property tensor: columns
        robot.get_dofs_pos(dofs_idx_local=dofs.dofs_idx_local)  # a Genesis-level call: local indices
        robot.get_default_dofs_kp(dofs)                      # a getter takes the selection itself

    Both tensors live on the entity's device; ``len(dofs)`` is the number of DOFs.

    Attributes
    ----------
    entity_name : str
        The entity it was resolved on; a getter refuses a selection from another one, whose columns would
        address that entity's layout.
    names : tuple[str, ...]
        The joint each DOF belongs to.
    columns : torch.Tensor
        ``long``; positions in the entity's ``dofs_name``. Index ``default_dofs_*``,
        ``soft_dofs_pos_limits`` or a full-width ``get_dofs_*()`` read with it.
    dofs_idx_local : torch.Tensor
        Genesis' local indices of the same DOFs. Pass it as ``dofs_idx_local=``, and nowhere else.
    """

    entity_name: str
    names: tuple[str, ...]
    columns: torch.Tensor
    dofs_idx_local: torch.Tensor

    def __len__(self) -> int:
        return len(self.names)


class DofDefaultsMixin:
    """Name-based access to the controlled DOFs and the configured per-DOF property tensors.

    :meth:`find_named_dofs` resolves names once, into a :class:`DofSelection` that carries the DOFs'
    columns and local indices together. ``get_default_dofs_<property>`` (and
    :meth:`get_soft_dofs_pos_limits`) is ``get_`` plus the config field it reads. It takes names, or a
    selection, where the ``get_dofs_*`` getters take ``dofs_idx_local``, and the same ``envs_idx``:

    .. code-block:: python

        robot.get_default_dofs_kp()                         # (num_envs, num_dofs), dofs_name order
        robot.get_default_dofs_kp(["left_knee_joint"])      # (num_envs, 1)
        robot.get_default_dofs_pos(envs_idx=[0])            # (1, num_dofs)
        robot.get_default_dofs_pos_limits(".*_knee_joint")  # (num_envs, 2, 2): lower, upper

    The ``default_*`` value is what the entity was *configured* with, resolved at build; the
    ``get_dofs_*`` getter beside it reads what the solver holds now. They differ wherever something
    writes the solver afterwards: domain randomization, the explicit-integrator armature bump, a
    deployment.

    **Selection.** ``dofs_name`` takes names, globs or regexes (``None`` for every DOF), matched
    against the joints in the entity's ``dofs_name`` — the controlled ones, which are the only
    joints that have a column. A free base or a passive joint is not one of them, and raises. A key
    that is exactly a joint's name — any joint's, controlled or not — selects that joint literally,
    metacharacters and all, so a passive ``pin.1`` never selects a controlled ``pin_1``. The result
    follows the query's order, and a joint with several DOFs (a ball joint) selects all of them, in
    order; name each joint once.

    **Shape.** Always ``(n_envs, n_selected[, 2])``, with ``envs_idx`` narrowing the env axis
    exactly as it does for the ``get_dofs_*`` info getters. The stored tensors are not that
    uniform: one row or one per env, depending on the property, whether the config set it,
    ``EnvOptions.batch_dofs_info`` and whether the entity is heterogeneous; and the rows can differ
    (a heterogeneous entity, ``RandomizeStartupDofsPosBias``) — so a caller that read row ``0`` of
    one was reading env ``0`` without saying so.

    **Ownership.** Every read is a copy, never a view of the stored tensor.

    **Hot paths and writes.** A getter allocates its ``(n_envs, n)`` copy on every call, so it is for
    ``build()``. To write into a property tensor, or to read one every step while keeping its stored
    layout (one broadcast row stays one row), index it with a :meth:`find_named_dofs` selection's
    ``columns``.

    The host class provides ``dofs_name`` (resolved: one entry per DOF), ``dofs_idx_local``,
    ``dofs_idx_map``, ``find_named_dofs_idx_local``, ``joints``, ``device``, ``is_built``, ``name``, the property
    attributes, and ``_dof_defaults_num_envs``.
    """

    @property
    def _dof_defaults_num_envs(self) -> int:
        """The number of environments the getters broadcast a single stored row over."""
        raise NotImplementedError

    @gs.assert_built
    def find_named_dofs(self, dofs_name: str | Sequence[str] | None = None) -> DofSelection:
        """Resolve names to a :class:`DofSelection`: the selected DOFs' columns *and* local indices.

        One resolution feeds both, so the two always describe the same DOFs, in the same order.
        Selection follows the class note: only the controlled joints (those in :attr:`dofs_name`) can
        be selected, through the same simulator-side resolver ``find_named_dofs_idx_local`` uses, so a
        query selects the same joints in both. Resolve in ``build()``, not every step.

        Parameters
        ----------
        dofs_name : str | Sequence[str] | None
            Names, globs or regexes of joints in :attr:`dofs_name`. ``None`` selects every
            controlled DOF, in column order.

        Returns
        -------
        DofSelection
            One name, column and local index per selected DOF.
        """
        if dofs_name is None:
            # Every controlled DOF, in column order: what ``dofs_name`` and ``dofs_idx_local`` already hold. An entity
            # that resolved no DOFs has an empty ``dofs_idx_map``, and its ``dofs_name`` is then still the raw config.
            names = list(self.dofs_name) if self.dofs_idx_map else []
            dofs_idx_local = self.dofs_idx_local.tolist() if names else []
        else:
            names, dofs_idx_local = self._resolve_controlled_dofs(dofs_name)
        # ``dofs_idx_map`` takes a controlled DOF's local index to its column.
        columns = [self.dofs_idx_map[int(i)] for i in dofs_idx_local]
        return DofSelection(
            entity_name=self.name,
            names=tuple(names),
            columns=torch.tensor(columns, dtype=torch.long, device=self.device),
            dofs_idx_local=torch.tensor(dofs_idx_local, dtype=gs.tc_int, device=self.device),
        )

    def _resolve_controlled_dofs(self, dofs_name: str | Sequence[str]) -> tuple[list[str], list[int]]:
        """``find_named_dofs_idx_local`` scoped to the controlled joints, with an error that says so."""
        keys = [dofs_name] if isinstance(dofs_name, str) else list(dofs_name)
        if not keys:
            return [], []
        # The controlled joints, once each: the only ones with a column (see ``find_named_dofs`` for the empty case).
        joints = list(dict.fromkeys(self.dofs_name)) if self.dofs_idx_map else []
        try:
            return self.find_named_dofs_idx_local(keys, name_scope=joints, preserve_order=True)
        except ValueError as e:
            # On one line: the resolver's own text breaks its key list and "Available strings" (the controlled joints,
            # the only ones with a column) over several.
            raise ValueError(
                f"Entity '{self.name}': find_named_dofs({dofs_name!r}) failed: {' '.join(str(e).split())} "
                f"{self._why_no_column(keys, joints)}"
            ) from e

    def _why_no_column(self, keys: list[str], joints: list[str]) -> str:
        """Why ``keys`` selected no column, and the fix for it; for the error message only."""
        if not joints:  # first: with no scope, nothing below would be evaluated against it
            return (
                "It controls no joint. Fix: build the entity articulated or actuated if it should carry per-DOF "
                "properties."
            )
        every_joint = [joint.name for joint in self.joints]
        literal = set(every_joint)  # exact names judged against every joint, as the resolver does
        selected = {key: [name for name in every_joint if key_selects(key, name, literal)] for key in keys}
        uncontrolled = {key: names for key, names in selected.items() if names and not set(names) & set(joints)}
        unknown = [key for key, names in selected.items() if not names]
        if uncontrolled:
            named = ", ".join(f"{key!r} selects {names}" for key, names in uncontrolled.items())
            return (
                f"{named}: joints this entity has but does not control. A free base or a passive joint has no "
                f"column. Fix: to address its DOFs in the simulator, pass "
                f"find_named_dofs_idx_local(...)'s indices as `dofs_idx_local`; to give it a column, add it to "
                f"the entity's `dofs_name`."
            )
        if unknown:
            return f"{unknown} match no joint of this entity. Fix: use one of the 'Available strings' above."
        return (
            "A joint is selected by more than one key (or named twice). Fix: name each joint once, e.g. "
            "narrow the overlapping pattern."
        )

    @gs.assert_built
    def _get_default_dofs(
        self, field_name: str, dofs_name: str | Sequence[str] | DofSelection | None, envs_idx
    ) -> torch.Tensor:
        """Read the property tensor ``field_name`` by name; selection, shape and ownership per the class note."""
        # ``None`` for an attribute a host does not carry: a deployment wrapper copies only the resolved tensors,
        # and its ``__getattr__`` would otherwise answer with a different error than an entity's.
        values = getattr(self, field_name, None)
        if not torch.is_tensor(values):
            raise ValueError(
                f"Entity '{self.name}' has no resolved {field_name}: it is {reprlib.repr(values)}, not a "
                f"tensor. Only an articulated or actuated entity resolves its per-DOF "
                f"properties at build; a kinematic one resolves default_dofs_pos and default_dofs_pos_limits "
                f"alone, and a non-actuated one has no default_dofs_kp / default_dofs_kd. Fix: read what the "
                f"solver holds with the matching get_dofs_* getter, or build the entity actuated (RobotOptions) "
                f"if it should carry configured values."
            )
        if dofs_name is None:
            columns = torch.arange(values.shape[1], device=values.device)
        else:
            dofs = dofs_name if isinstance(dofs_name, DofSelection) else self.find_named_dofs(dofs_name)
            if dofs.entity_name != self.name:
                raise ValueError(
                    f"Entity '{self.name}': a DofSelection resolved on entity '{dofs.entity_name}' was passed to "
                    f"get_{field_name}; its columns address that entity's `dofs_name`, not this one's. Fix: resolve "
                    f"it on this entity: env.entities[{self.name!r}].find_named_dofs({list(dict.fromkeys(dofs.names))!r})."
                )
            columns = dofs.columns.to(values.device)
        selected = values.index_select(1, columns)
        num_envs = self._dof_defaults_num_envs
        envs_idx = sanitize_envs_idx(envs_idx, num_envs, prefer_slice=True)
        if torch.is_tensor(envs_idx):  # built on Genesis' device, which a deployment wrapper's copies need not be on
            envs_idx = envs_idx.to(selected.device)
        return selected.expand(num_envs, *selected.shape[1:])[envs_idx].contiguous()

    def get_default_dofs_pos(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured default joint positions, ``(n_envs, n_dofs)``. See the class note."""
        return self._get_default_dofs("default_dofs_pos", dofs_name, envs_idx)

    def get_default_dofs_vel(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured default joint velocities, ``(n_envs, n_dofs)``. See the class note."""
        return self._get_default_dofs("default_dofs_vel", dofs_name, envs_idx)

    def get_default_dofs_kp(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured PD position gains, ``(n_envs, n_dofs)``; actuated entities only. See the class note."""
        return self._get_default_dofs("default_dofs_kp", dofs_name, envs_idx)

    def get_default_dofs_kd(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured PD velocity gains, ``(n_envs, n_dofs)``; actuated entities only. See the class note."""
        return self._get_default_dofs("default_dofs_kd", dofs_name, envs_idx)

    def get_default_dofs_stiffness(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured joint stiffness, ``(n_envs, n_dofs)``. See the class note."""
        return self._get_default_dofs("default_dofs_stiffness", dofs_name, envs_idx)

    def get_default_dofs_damping(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured joint damping, ``(n_envs, n_dofs)``. See the class note."""
        return self._get_default_dofs("default_dofs_damping", dofs_name, envs_idx)

    def get_default_dofs_armature(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured armature, ``(n_envs, n_dofs)``, before any explicit-integrator bump. See the class note."""
        return self._get_default_dofs("default_dofs_armature", dofs_name, envs_idx)

    def get_default_dofs_frictionloss(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured joint friction loss, ``(n_envs, n_dofs)``. See the class note."""
        return self._get_default_dofs("default_dofs_frictionloss", dofs_name, envs_idx)

    def get_default_dofs_pos_limits(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured hard position limits, ``(n_envs, n_dofs, 2)`` as (lower, upper). See the class note."""
        return self._get_default_dofs("default_dofs_pos_limits", dofs_name, envs_idx)

    def get_soft_dofs_pos_limits(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Soft position limits, ``(n_envs, n_dofs, 2)`` as (lower, upper); unbounded DOFs keep their hard range.

        See the class note.
        """
        return self._get_default_dofs("soft_dofs_pos_limits", dofs_name, envs_idx)

    def get_default_dofs_force_limits(
        self, dofs_name: str | Sequence[str] | DofSelection | None = None, envs_idx=None
    ) -> torch.Tensor:
        """Configured symmetric effort limits, ``(n_envs, n_dofs)``. See the class note."""
        return self._get_default_dofs("default_dofs_force_limits", dofs_name, envs_idx)
