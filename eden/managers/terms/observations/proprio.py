"""Proprioceptive observation terms (base pose/velocity; joint position, velocity and force)."""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import torch
from genesis.utils.geom import quat_to_xyz

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@OBSERVATION_TERM_REGISTRY.register()
def base_pos(
    env: EnvBase, *, entity_name: str, envs_idx: slice | torch.Tensor | None = None, relative: bool = False
) -> torch.Tensor:
    """Return the position of the entity's base link.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        The name of the entity.
    envs_idx : slice | torch.Tensor | None, optional
        The indices of the environments. If None, all environments will be considered. Defaults to None.
    relative : bool, optional
        Frame the position is reported in. ``False`` (the default) reads the solver frame, which
        Genesis re-origins onto a free rigid body's inertial frame; ``True`` reads the frame the
        asset was authored in. Pass ``True`` whenever the rest of the task reasons in the asset's own
        frame: the two differ by the body's centre-of-mass offset (6.8 cm on robosuite's hammer), and
        because that offset lives in the body frame it *rotates with the object* — so a policy handed
        this frame alongside a ``gripper_to_<obj>`` vector measured in the other one sees two
        different positions for one object. Same switch as
        :func:`~eden.managers.terms.metrics.containment.object_above_height`.
    """
    entity: Entity = env.get_entity(entity_name)
    return entity.get_pos(envs_idx=envs_idx, relative=relative)


@OBSERVATION_TERM_REGISTRY.register()
def base_quat(
    env: EnvBase, *, entity_name: str, envs_idx: slice | torch.Tensor | None = None, relative: bool = False
) -> torch.Tensor:
    """Return the quaternion of the entity's base link.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        The name of the entity.
    envs_idx : slice | torch.Tensor | None, optional
        The indices of the environments. If None, all environments will be considered. Defaults to None.
    relative : bool, optional
        Frame the orientation is reported in — see :func:`base_pos`. Defaults to ``False``, the
        solver frame. Pass ``True`` to report the frame the asset was authored in, which is the frame
        an ``object_offset`` taken from the asset's own geometry is measured in.
    """
    entity: Entity = env.get_entity(entity_name)
    return entity.get_quat(envs_idx=envs_idx, relative=relative)


@OBSERVATION_TERM_REGISTRY.register()
def base_rpy(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: slice | torch.Tensor | None = None,
    terms_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the roll, pitch, and yaw of the entity's base link.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        The name of the entity.
    envs_idx : slice | torch.Tensor | None, optional
        The indices of the environments. If None, all environments will be considered. Defaults to None.
    terms_idx : torch.Tensor | None, optional
        The indices of the terms to return. If None, all terms will be returned. Defaults to None.
    """
    entity: Entity = env.get_entity(entity_name)
    if terms_idx is not None:
        return quat_to_xyz(entity.get_quat(envs_idx=envs_idx), rpy=True)[..., terms_idx]
    return quat_to_xyz(entity.get_quat(envs_idx=envs_idx), rpy=True)


@OBSERVATION_TERM_REGISTRY.register()
def base_lin_vel(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: slice | torch.Tensor | None = None,
    frame: Literal["world", "body"] = "world",
) -> torch.Tensor:
    entity: Entity = env.get_entity(entity_name)
    return entity.get_vel(envs_idx=envs_idx, frame=frame)


@OBSERVATION_TERM_REGISTRY.register()
def base_ang_vel(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: slice | torch.Tensor | None = None,
    frame: Literal["world", "body"] = "world",
) -> torch.Tensor:
    asset: Entity = env.get_entity(entity_name)
    return asset.get_ang(envs_idx=envs_idx, frame=frame)


class _DofsObservation(ObservationTerm):
    """One per-DOF quantity of an entity, for every controlled DOF or a selection of them.

    The ``Dofs*`` terms below share this selection. It is resolved once in :meth:`build`, so a name
    that matches no controlled joint fails at env construction, and ``compute()`` re-resolves the
    entity each call, so a deployment hot-swap of ``env.entities[name]`` is read.

    Parameters
    ----------
    entity_name: str
        The entity whose DOFs are read.
    dofs_name: list[str] | None
        Names, globs or regexes of the entity's controlled joints, in the order the consumer expects
        (a policy server's state layout, say). See
        :meth:`~eden.entities.dofs_default_mixin.DofDefaultsMixin.find_named_dofs`. ``None`` reads every
        controlled DOF, in the entity's ``dofs_name`` order.
    terms_idx: list[int] | None
        Positions in the entity's ``dofs_name``, instead of names. A position silently names a different
        joint when the DOF layout changes (a hand attached, an asset update), so prefer ``dofs_name``. Not
        both.
    """

    entity_name: str = "robot"
    dofs_name: list[str] | None = None
    terms_idx: list[int] | None = None

    # These were function terms; a config saved then carries its parameters in ``params``, which load as fields.
    _function_era_params = frozenset({"entity_name", "terms_idx"})

    def build(self) -> None:
        # Touch the entity for fail-fast, but cache only the derived columns (see the class docstring).
        entity = self.resolve_entity(self.entity_name)
        columns = self._resolve_columns(entity)
        # Every DOF in dofs_name order is the full read; store None so compute() skips the per-step copy.
        if columns is not None and columns.tolist() == list(range(len(entity.dofs_name))):
            columns = None
        self._columns = columns

    def _resolve_columns(self, entity: Entity) -> torch.Tensor | None:
        """Positions in the entity's ``dofs_name`` to read, or ``None`` for every DOF."""
        if self.dofs_name is not None and self.terms_idx is not None:
            raise ValueError(
                f"{term_subject(self)} got both dofs_name={self.dofs_name!r} and terms_idx={self.terms_idx!r}; "
                f"they select the same thing two ways. Keep dofs_name and drop terms_idx."
            )
        if self.dofs_name is not None:
            if not self.dofs_name:
                raise ValueError(
                    f"{term_subject(self)}: dofs_name=[] selects no DOF. Name at least one, or pass None for every DOF."
                )
            return entity.find_named_dofs(list(self.dofs_name)).columns
        if self.terms_idx is not None:
            dofs = list(entity.dofs_name)
            bad = [idx for idx in self.terms_idx if not -len(dofs) <= idx < len(dofs)]
            if not self.terms_idx or bad:
                problem = f"positions {bad} are out of range" if bad else "it selects no DOF"
                raise ValueError(
                    f"{term_subject(self)}: terms_idx={self.terms_idx!r}: {problem} for its {len(dofs)} DOFs "
                    f"{dofs}. Select joints by name instead, e.g. dofs_name={dofs[:2]!r}."
                )
            n_dofs = len(dofs)
            return torch.tensor([idx % n_dofs for idx in self.terms_idx], dtype=torch.long, device=self.device)
        return None

    def _select(self, values: torch.Tensor) -> torch.Tensor:
        """The selected columns of a full-width ``(num_envs, n_dofs)`` read, in the selection's order."""
        return values if self._columns is None else values.index_select(-1, self._columns)


@OBSERVATION_TERM_REGISTRY.register()
class DofsPos(_DofsObservation):
    """Joint positions (angles; positions for linear actuators) of an entity's DOFs.

    Parameters
    ----------
    entity_name: str
        The entity whose DOFs are read.
    dofs_name: list[str] | None
        Joints to read, in output order; ``None`` for every controlled DOF (see :class:`_DofsObservation`).
    terms_idx: list[int] | None
        Positions in the entity's ``dofs_name`` instead of names; prefer ``dofs_name``.
    offset_from_default: bool
        Subtract the entity's default positions (``default_dofs_pos``) of the same DOFs.
    """

    offset_from_default: bool = False

    _function_era_params = frozenset({"entity_name", "offset_from_default", "terms_idx"})

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        dofs_pos = entity.get_dofs_pos()
        if self.offset_from_default:
            dofs_pos = dofs_pos - entity.default_dofs_pos
        return self._select(dofs_pos)


@OBSERVATION_TERM_REGISTRY.register()
class DofsVel(_DofsObservation):
    """Joint velocities of an entity's DOFs.

    Parameters
    ----------
    entity_name: str
        The entity whose DOFs are read.
    dofs_name: list[str] | None
        Joints to read, in output order; ``None`` for every controlled DOF (see :class:`_DofsObservation`).
    terms_idx: list[int] | None
        Positions in the entity's ``dofs_name`` instead of names; prefer ``dofs_name``.
    """

    def compute(self) -> torch.Tensor:
        return self._select(self.resolve_entity(self.entity_name).get_dofs_vel())


@OBSERVATION_TERM_REGISTRY.register()
class DofsForce(_DofsObservation):
    """Generalized force the solver applied to each of an entity's DOFs (``get_dofs_force``).

    Parameters
    ----------
    entity_name: str
        The entity whose DOFs are read.
    dofs_name: list[str] | None
        Joints to read, in output order; ``None`` for every controlled DOF (see :class:`_DofsObservation`).
    terms_idx: list[int] | None
        Positions in the entity's ``dofs_name`` instead of names; prefer ``dofs_name``.
    """

    # The function term took only entity_name; a saved config cannot carry terms_idx.
    _function_era_params = frozenset({"entity_name"})

    def compute(self) -> torch.Tensor:
        return self._select(self.resolve_entity(self.entity_name).get_dofs_force())


@OBSERVATION_TERM_REGISTRY.register()
class DofsControlForce(_DofsObservation):
    """Control force the actuators applied on each of an entity's DOFs over the last physics step
    (``get_dofs_control_force``).

    Parameters
    ----------
    entity_name: str
        The entity whose DOFs are read.
    dofs_name: list[str] | None
        Joints to read, in output order; ``None`` for every controlled DOF (see :class:`_DofsObservation`).
    terms_idx: list[int] | None
        Positions in the entity's ``dofs_name`` instead of names; prefer ``dofs_name``.
    """

    # The function term took only entity_name; a saved config cannot carry terms_idx.
    _function_era_params = frozenset({"entity_name"})

    def compute(self) -> torch.Tensor:
        return self._select(self.resolve_entity(self.entity_name).get_dofs_control_force())


@OBSERVATION_TERM_REGISTRY.register()
def links_pos(
    env: EnvBase,
    *,
    entity_name: str,
    ls_idx_local: torch.Tensor | None = None,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    robot: Entity = env.get_entity(entity_name)
    return robot.get_links_pos(ls_idx_local=ls_idx_local, envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def links_quat(
    env: EnvBase,
    *,
    entity_name: str,
    ls_idx_local: torch.Tensor | None = None,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    robot: Entity = env.get_entity(entity_name)
    return robot.get_links_quat(ls_idx_local=ls_idx_local, envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def links_vel(
    env: EnvBase,
    *,
    entity_name: str,
    ls_idx_local: torch.Tensor | None = None,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    robot: Entity = env.get_entity(entity_name)
    return robot.get_links_vel(ls_idx_local=ls_idx_local, envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def links_ang(
    env: EnvBase,
    *,
    entity_name: str,
    ls_idx_local: torch.Tensor | None = None,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    robot: Entity = env.get_entity(entity_name)
    return robot.get_links_ang(ls_idx_local=ls_idx_local, envs_idx=envs_idx)
