"""Robot state/command dataclasses and state-entity wrapper for deployment."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from genesis.utils.geom import inv_transform_by_quat, transform_by_quat

from eden.entities.dofs_default_mixin import DofDefaultsMixin
from eden.options.entities import RESOLVED_DOF_TENSOR_FIELD_NAMES
from eden.utils.geom import heading_from_quat, quat_mul, yaw_from_quat

if TYPE_CHECKING:
    from eden.entities.base import Entity


@dataclass
class RobotState:
    stamp: float
    base_quat: np.ndarray | None  # wxyz
    base_ang_vel: np.ndarray | None
    base_lin_acc: np.ndarray | None
    dofs_pos: np.ndarray
    dofs_vel: np.ndarray
    dofs_torque: np.ndarray
    full_dofs_pos: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device = "cpu") -> RobotState:
        return RobotState(
            stamp=self.stamp,
            base_quat=torch.from_numpy(self.base_quat).to(device) if self.base_quat is not None else None,
            base_ang_vel=torch.from_numpy(self.base_ang_vel).to(device) if self.base_ang_vel is not None else None,
            base_lin_acc=torch.from_numpy(self.base_lin_acc).to(device) if self.base_lin_acc is not None else None,
            dofs_pos=torch.from_numpy(self.dofs_pos).to(device),
            dofs_vel=torch.from_numpy(self.dofs_vel).to(device),
            dofs_torque=torch.from_numpy(self.dofs_torque).to(device),
            full_dofs_pos=torch.from_numpy(self.full_dofs_pos).to(device) if self.full_dofs_pos is not None else None,
            extra=self.extra,
        )


@dataclass
class RobotCommand:
    dofs_pos: np.ndarray
    dofs_vel: np.ndarray
    dofs_torque: np.ndarray
    dofs_kp: np.ndarray | None = None
    dofs_kd: np.ndarray | None = None


class RobotStateEntity(DofDefaultsMixin):
    """Entity-compatible wrapper that exposes RobotState through the Entity getter interface.

    Duck-types the subset of Entity getter methods used by observation terms,
    allowing the ObservationManager to compute observations from real robot state
    without any modification to existing terms.

    Static properties (default_dofs_pos, dofs_name, etc.) are copied from the sim
    Entity at construction time.  Mutable state is updated each step via ``update()``.
    The ``get_default_dofs_*`` getters (:class:`~eden.entities.dofs_default_mixin.DofDefaultsMixin`)
    read those copies.

    Notes
    -----
    - ``base_ang_vel`` in RobotState is assumed to be in **body frame** (IMU convention).
      ``get_ang(frame="world")`` rotates it into world frame, while
      ``get_ang(frame="body")`` returns it directly.
    - ``base_pos`` is initialized from the sim entity's spawn pose and is
      treated as **unknown** until ``RobotState.extra["base_pos"]`` is
      supplied (e.g. from an external estimator or motion capture). While
      base_pos is unknown, ``update()`` does **not** push a synthetic root
      position into the sim entity's FK so observation terms keep seeing the
      training spawn pose instead of an origin-anchored value. Once known,
      updates without a fresh reading (estimator dropout) hold the last
      known position.
    - ``base_lin_vel`` defaults to zeros and is updated from
      ``RobotState.extra["base_lin_vel"]`` when supplied (raw IMU world
      frame; the yaw offset is removed as the reading arrives, and the
      value is held — not re-rotated — through estimator dropouts).
      Alternatively an onboard estimator can push an already yaw-aligned
      velocity via :meth:`set_vel_estimate` after ``update()``.
    - A yaw offset can be installed via :meth:`set_yaw_offset` /
      :meth:`capture_yaw_offset`. Subsequent ``update()`` calls subtract that
      yaw from world-frame quantities: ``_quat`` on every update, ``_pos`` /
      ``_vel`` only on updates carrying a fresh estimator reading — a held
      value was already yaw-corrected when it arrived, so rotating it again
      would corrupt it. Body-frame quantities (``_ang_vel``) are unaffected. The spawn-pose
      fallback for ``_pos`` is left in the sim's training world frame and is
      not yaw-rotated, since the policy already saw it that way.
    """

    #: Every configured per-DOF tensor ``RigidEntity`` may resolve, taken from the canonical list
    #: in ``eden.options.entities`` rather than restated here, so a newly added property is covered
    #: too. ``__init__`` clones each one the sim entity carries; whatever it did not carry is
    #: blocked in ``__getattr__`` rather than forwarded, because these names are mirrored config
    #: fields: an entity ``post_build`` resolved nothing for (kinematic, or not actuated) still
    #: answers the attribute — with the raw configured value, a ``dict`` or ``None``, not a tensor.
    #:
    #: Cloning all of them (not only those a backend reads) gives every per-DOF read the same
    #: connect-time-snapshot semantics and keeps ``device`` meaningful as a constructor argument:
    #: nothing here may assume the caller's device is the sim entity's. In-tree, only
    #: ``default_dofs_pos`` is read through this wrapper — the stance each backend ramps to, and
    #: the baseline of the ``DofsPos`` observation term under ``offset_from_default=True`` (which
    #: re-resolves its entity on every ``compute()``, so the ``env.entities`` swap reaches it
    #: without the class-term repoint in ``DeploymentBase._repoint_entity_refs``).
    _DEFAULT_DOFS_ATTRS: tuple[str, ...] = RESOLVED_DOF_TENSOR_FIELD_NAMES

    #: The four a deployable robot must have resolved. Only ``default_dofs_pos`` is read off this
    #: wrapper (see above; the gains each backend publishes come from ``robot_options``), but a
    #: sim entity missing any of these is not a built, actuated robot at all — a broken
    #: deployment rather than an optional property, so it raises at connect time rather than at
    #: some first read.
    _REQUIRED_DEFAULT_DOFS_ATTRS: tuple[str, ...] = (
        "default_dofs_pos",
        "default_dofs_vel",
        "default_dofs_kp",
        "default_dofs_kd",
    )

    def __init__(self, sim_entity: Entity, device: torch.device) -> None:
        self.device = device
        # Kept as the kinematics oracle: link-level getters (get_links_pos, etc.)
        # delegate here, with the sim entity's qpos synced from real state in update().
        self._sim_entity = sim_entity

        # -- static properties copied from the sim entity --
        self.num_dofs: int = sim_entity.num_dofs
        self.dofs_name: list[str] = list(sim_entity.dofs_name)
        self.dofs_idx_local = sim_entity.dofs_idx_local.clone().to(device)
        self.dofs_idx_map: dict[int, int] = dict(sim_entity.dofs_idx_map)
        # ``_dofs_columns``' memo: an index tensor's columns by its ``id`` (with a weakref, so the entry goes
        # with the tensor), and a list's or range's by value.
        self._columns_by_tensor: dict[int, tuple[weakref.ref, int | None, torch.Tensor]] = {}
        self._columns_by_value: dict[tuple[int, ...] | range, torch.Tensor] = {}
        # Cloned onto ``device``, so nothing read through this wrapper aliases the sim entity's
        # memory (``_DEFAULT_DOFS_ATTRS`` says why a fall-through is wrong). Tolerant per property:
        # ``post_build`` resolves gains on actuated entities only and skips the dynamics properties
        # on kinematic ones. Env 0's row only, as for the base pose (``_initial_base_pos``): the
        # wrapper is one robot, so a property stored per sim env keeps the row of the env it stands
        # for, and every read through it has one env, like the state getters.
        for _attr_name in self._DEFAULT_DOFS_ATTRS:
            _value = getattr(sim_entity, _attr_name, None)
            if torch.is_tensor(_value):
                setattr(self, _attr_name, _value[:1].clone().to(device))
            elif _attr_name in self._REQUIRED_DEFAULT_DOFS_ATTRS:
                raise ValueError(
                    f"Sim entity '{getattr(sim_entity, 'name', sim_entity)}' has no resolved "
                    f"{_attr_name} tensor. Only a built, simulated, articulated and actuated "
                    f"entity resolves the per-DOF defaults, so this is not a robot a deployment "
                    f"can drive. Raised here rather than at the first read, which for these four "
                    f"would be mid-ramp with the robot already powered."
                )
        self.is_fixed_base: bool = getattr(sim_entity, "is_fixed_base", False)
        self.is_attached: bool = getattr(sim_entity, "is_attached", False)
        #: Mirrors ``Entity.has_immobile_base``: no base DOFs to read or write.
        self.has_immobile_base: bool = self.is_fixed_base or self.is_attached
        self.material = None

        fv = sim_entity.forward_vec
        if isinstance(fv, (tuple, list)):
            self._forward_vec = torch.tensor(fv, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            self._forward_vec = fv.clone().to(device)
            if self._forward_vec.dim() == 1:
                self._forward_vec = self._forward_vec.unsqueeze(0)

        # -- pre-allocated mutable state buffers (single env) --
        # Default ``_pos`` to the sim entity's spawn pose so that, when no
        # external base_pos estimator is wired up, the wrapper and the sim
        # entity agree on the root translation FK is computed at. This keeps
        # body-frame observation terms internally consistent and avoids
        # silently anchoring world-frame link positions at the origin.
        self._pos = self._initial_base_pos(sim_entity, device)
        self._base_pos_known: bool = False
        self._quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        self._vel = torch.zeros(1, 3, dtype=torch.float32, device=device)
        self._ang_vel = torch.zeros(1, 3, dtype=torch.float32, device=device)
        self._dofs_pos = torch.zeros(1, self.num_dofs, dtype=torch.float32, device=device)
        self._dofs_vel = torch.zeros(1, self.num_dofs, dtype=torch.float32, device=device)
        self._dofs_torque = torch.zeros(1, self.num_dofs, dtype=torch.float32, device=device)

        # Yaw offset removed from the IMU world frame so that deployment matches
        # the yaw-aligned world frame the policy was trained in. Identity by default.
        self._yaw_offset: float = 0.0
        self._yaw_remove_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    @staticmethod
    def _initial_base_pos(sim_entity: Entity, device: torch.device) -> torch.Tensor:
        """Read the sim entity's current base position and shape it to ``(1, 3)``.

        Falls back to zeros only when the sim entity doesn't expose
        ``get_pos`` (duck-typed stand-ins). Any error raised *inside*
        ``get_pos`` propagates so a broken entity fails at construction time
        instead of silently anchoring FK at the origin.
        """
        if not hasattr(sim_entity, "get_pos"):
            return torch.zeros(1, 3, dtype=torch.float32, device=device)
        pos = sim_entity.get_pos().detach().clone().to(dtype=torch.float32, device=device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        return pos[:1]

    @property
    def _dof_defaults_num_envs(self) -> int:
        # Deployment contains one environment; the clones keep env 0's row.
        return 1

    @property
    def forward_vec(self) -> torch.Tensor:
        return self._forward_vec

    @property
    def fixed(self) -> bool:
        return False

    @property
    def base_pos_known(self) -> bool:
        """Return True once a real base position has been supplied via ``RobotState.extra["base_pos"]``."""
        return self._base_pos_known

    def update(self, state: RobotState) -> None:
        """Update internal buffers from a real-robot RobotState reading.

        Also syncs the wrapped sim entity's qpos so that Genesis FK
        (consumed by ``get_links_pos`` / ``get_links_quat``) reflects the
        latest real-robot state when observation terms are computed.

        Any installed yaw offset (see :meth:`set_yaw_offset`) is applied to
        ``_quat`` on every call and to ``_pos`` / ``_vel`` when this call
        carries a fresh estimator reading, before the sim entity is synced,
        so FK is computed in the yaw-aligned world frame.
        """
        self._quat[0] = torch.as_tensor(state.base_quat, dtype=torch.float32, device=self.device)
        self._ang_vel[0] = torch.as_tensor(state.base_ang_vel, dtype=torch.float32, device=self.device)
        self._dofs_pos[0] = torch.as_tensor(state.dofs_pos, dtype=torch.float32, device=self.device)
        self._dofs_vel[0] = torch.as_tensor(state.dofs_vel, dtype=torch.float32, device=self.device)
        self._dofs_torque[0] = torch.as_tensor(state.dofs_torque, dtype=torch.float32, device=self.device)
        pos_fresh = "base_pos" in state.extra
        if pos_fresh:
            self._pos[0] = torch.as_tensor(state.extra["base_pos"], dtype=torch.float32, device=self.device)
            self._base_pos_known = True
        vel_fresh = "base_lin_vel" in state.extra
        if vel_fresh:
            self._vel[0] = torch.as_tensor(state.extra["base_lin_vel"], dtype=torch.float32, device=self.device)

        if self._yaw_offset != 0.0:
            self._quat[:] = quat_mul(self._yaw_remove_quat, self._quat)
            # Yaw-rotate _pos / _vel only when this update carries a fresh
            # estimator reading. A value held from a previous update was
            # already rotated when it arrived — rotating it again every step
            # would spin it around z. The spawn-pose fallback for _pos
            # likewise already lives in the training world frame.
            if pos_fresh:
                self._pos[:] = transform_by_quat(self._pos, self._yaw_remove_quat)
            if vel_fresh:
                self._vel[:] = transform_by_quat(self._vel, self._yaw_remove_quat)

        if not self.has_immobile_base:
            # Skip set_pos until a real base_pos has been supplied so the sim
            # entity's FK keeps using the training spawn pose instead of an
            # origin-anchored synthetic value; during estimator dropouts the
            # last known position is held. set_quat is always synced because
            # base_quat is always available from the IMU.
            if self._base_pos_known:
                self._sim_entity.set_pos(self._pos)
            self._sim_entity.set_quat(self._quat)
        self._sim_entity.set_dofs_pos(self._dofs_pos)
        self._sim_entity.set_dofs_vel(self._dofs_vel)
        if self.is_attached:
            # A welded body (the hand of an arm + hand rig) has no base reading of its own -- the
            # hand has no IMU, so ``state.base_quat`` is a placeholder. Its base sits wherever the
            # parent link carries it, which the sim entity's FK has just computed from the parent's
            # synced joints (a composite syncs the parent first). Read that back, so ``get_pos`` /
            # ``get_quat`` agree with the link getters instead of reporting the placeholder -- and so
            # do ``get_vel`` / ``get_ang``, its velocity the parent's synced joint velocities carry
            # through the weld (the angular one in the body frame ``base_ang_vel`` is read in).
            self._pos[0] = self._sim_entity.get_pos()[0].to(self._pos)
            self._quat[0] = self._sim_entity.get_quat()[0].to(self._quat)
            self._vel[0] = self._sim_entity.get_vel()[0].to(self._vel)
            self._ang_vel[0] = inv_transform_by_quat(self._sim_entity.get_ang()[0].to(self._ang_vel), self._quat[0])

    def set_vel_estimate(self, vel) -> None:
        """Install an estimated base linear velocity, **already** in the yaw-aligned world frame.

        Unlike ``RobotState.extra["base_lin_vel"]`` (which ``update()`` treats
        as a raw-IMU-world-frame reading and yaw-rotates as it arrives),
        values set here are stored as-is. Call after ``update()``, which syncs
        the FK the estimator reads and overwrites ``_vel`` whenever
        ``state.extra`` carries a velocity.
        """
        self._vel[0] = torch.as_tensor(vel, dtype=torch.float32, device=self.device)

    # -- yaw alignment ----------------------------------------------------

    @property
    def yaw_offset(self) -> float:
        """Captured world-frame yaw, in radians, that ``update()`` removes."""
        return self._yaw_offset

    def set_yaw_offset(self, yaw: float) -> None:
        """Install a yaw rotation (radians, around +z) that ``update()`` removes from world-frame quantities.

        The resulting world frame has the same z-axis but its x-axis is rotated
        by ``-yaw`` relative to the IMU's. Setting ``yaw=0`` restores the raw
        IMU world frame.
        """
        self._yaw_offset = float(yaw)
        # Build a yaw-only rotation around -z: q = (cos(-yaw/2), 0, 0, sin(-yaw/2)).
        half = -0.5 * float(yaw)
        self._yaw_remove_quat[0, 0] = float(np.cos(half))
        self._yaw_remove_quat[0, 1] = 0.0
        self._yaw_remove_quat[0, 2] = 0.0
        self._yaw_remove_quat[0, 3] = float(np.sin(half))

    def capture_yaw_offset(self, base_quat_wxyz) -> float:
        """Extract yaw from a (w, x, y, z) quaternion and install it as the offset.

        Returns the captured yaw in radians.
        """
        q = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(-1)
        if q.shape != (4,):
            raise ValueError(
                "capture_yaw_offset expects a single wxyz quaternion of 4 elements; got shape "
                f"{np.shape(base_quat_wxyz)}. Deployment is single-env — pass one base quaternion."
            )
        yaw = float(yaw_from_quat(q))
        self.set_yaw_offset(yaw)
        return yaw

    # -- base link getters ------------------------------------------------

    # TODO: Support relative=True after defining RobotState base frames/origins against Genesis user/solver frames
    # and adding non-identity pose-offset integration tests for root synchronization.
    @staticmethod
    def _reject_relative_pose(relative: bool) -> None:
        if relative:
            raise NotImplementedError(
                "RobotStateEntity does not support relative=True pose access until hardware state frames "
                "and offset-aware Genesis root synchronization are defined."
            )

    def get_pos(self, envs_idx=None, *, relative: bool = False) -> torch.Tensor:
        """Return the cached hardware base position in world frame.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            Accepted for Entity API compatibility. Deployment contains one environment. Defaults to None.
        relative : bool, optional
            Must be False. Relative hardware pose access is not yet supported. Defaults to False.

        Returns
        -------
        pos : torch.Tensor, shape (1, 3)
            The cached base position.

        Raises
        ------
        NotImplementedError
            If ``relative=True``.
        """
        self._reject_relative_pose(relative)
        return self._pos

    def get_quat(self, envs_idx=None, *, relative: bool = False) -> torch.Tensor:
        """Return the cached hardware base orientation in world frame.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            Accepted for Entity API compatibility. Deployment contains one environment. Defaults to None.
        relative : bool, optional
            Must be False. Relative hardware pose access is not yet supported. Defaults to False.

        Returns
        -------
        quat : torch.Tensor, shape (1, 4)
            The cached base quaternion in ``(w, x, y, z)`` format.

        Raises
        ------
        NotImplementedError
            If ``relative=True``.
        """
        self._reject_relative_pose(relative)
        return self._quat

    def get_T(self, envs_idx=None, *, relative: bool = False) -> torch.Tensor:
        """Return the synchronized Genesis base transform in world frame.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The environment indices. Defaults to None.
        relative : bool, optional
            Must be False. Relative hardware pose access is not yet supported. Defaults to False.

        Returns
        -------
        transform : torch.Tensor, shape (1, 4, 4)
            The solver/world-frame base transform from the synchronized Genesis entity.

        Raises
        ------
        NotImplementedError
            If ``relative=True``.
        """
        self._reject_relative_pose(relative)
        return self._sim_entity.get_T(envs_idx=envs_idx, relative=False)

    def get_heading(self, envs_idx=None) -> torch.Tensor:
        return heading_from_quat(self._quat, self._forward_vec)

    def get_vel(self, envs_idx=None, *, frame: Literal["world", "body"] = "world") -> torch.Tensor:
        if frame == "world":
            return self._vel
        elif frame == "body":
            return inv_transform_by_quat(self._vel, self._quat)
        raise ValueError(f"Invalid frame '{frame}'. Expected 'world' or 'body'.")

    def get_ang(self, envs_idx=None, *, frame: Literal["world", "body"] = "world") -> torch.Tensor:
        if frame == "body":
            return self._ang_vel
        elif frame == "world":
            return transform_by_quat(self._ang_vel, self._quat)
        raise ValueError(f"Invalid frame '{frame}'. Expected 'world' or 'body'.")

    # -- DOF getters ------------------------------------------------------

    def _select_dofs(self, data: torch.Tensor, dofs_idx_local) -> torch.Tensor:
        if dofs_idx_local is None:
            return data
        return data[:, self._dofs_columns(dofs_idx_local)]

    def _dofs_columns(self, dofs_idx_local) -> torch.Tensor:
        """``dofs_idx_local`` (the sim entity's indices) as columns of this wrapper's state, resolved once each.

        Terms resolve their indices in ``build()`` and pass the same tensor, list or range every step;
        translating it on every read cost a Python loop and a new device tensor per call. A tensor is
        remembered by identity for as long as it lives (and re-read if written in place), a list or range
        by value.
        """
        if not torch.is_tensor(dofs_idx_local):
            # A range hashes by value as it is; a list is frozen into a tuple.
            key = dofs_idx_local if isinstance(dofs_idx_local, range) else tuple(int(i) for i in dofs_idx_local)
            columns = self._columns_by_value.get(key)
            if columns is None:
                columns = self._columns_by_value[key] = self._columns_of(key)
            return columns
        try:
            version = dofs_idx_local._version
        except RuntimeError:  # an inference tensor tracks no version
            version = None
        key = id(dofs_idx_local)
        cached = self._columns_by_tensor.get(key)
        if cached is not None and cached[0]() is dofs_idx_local and cached[1] == version:
            return cached[2]
        columns = self._columns_of(dofs_idx_local.tolist())
        self._columns_by_tensor[key] = (
            weakref.ref(dofs_idx_local, lambda _, key=key, cache=self._columns_by_tensor: cache.pop(key, None)),
            version,
            columns,
        )
        return columns

    def _columns_of(self, dofs_idx_local: Sequence[int]) -> torch.Tensor:
        """The columns of ``dofs_idx_local``, raising for an index this robot holds no state for."""
        uncontrolled = [int(i) for i in dofs_idx_local if int(i) not in self.dofs_idx_map]
        if uncontrolled:
            raise ValueError(
                f"RobotStateEntity: dofs_idx_local {uncontrolled} are not DOFs this robot controls, so the "
                f"deployment holds no state for them. It reads the joints in `dofs_name` only "
                f"({self.dofs_name}), whose local indices are {sorted(self.dofs_idx_map)}. Fix: resolve the "
                f"indices with the entity's find_named_dofs(...).dofs_idx_local, which covers the controlled joints."
            )
        return torch.tensor([self.dofs_idx_map[int(i)] for i in dofs_idx_local], dtype=torch.long, device=self.device)

    def get_dofs_pos(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._select_dofs(self._dofs_pos, dofs_idx_local)

    def get_dofs_vel(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._select_dofs(self._dofs_vel, dofs_idx_local)

    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._select_dofs(self._dofs_torque, dofs_idx_local)

    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._select_dofs(self._dofs_torque, dofs_idx_local)

    # -- link getters (delegate to sim entity, kept in sync via update()) ----

    def find_named_links_idx_local(self, links_name, name_scope=None, preserve_order=True):
        return self._sim_entity.find_named_links_idx_local(
            links_name, name_scope=name_scope, preserve_order=preserve_order
        )

    def get_links_pos(self, ls_idx_local=None, envs_idx=None, *, relative: bool = False) -> torch.Tensor:
        """Return link positions from the synchronized Genesis entity.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The environment indices. Defaults to None.
        relative : bool, optional
            Must be False. Relative hardware pose access is not yet supported. Defaults to False.

        Returns
        -------
        pos : torch.Tensor
            The solver/world-frame link positions.

        Raises
        ------
        NotImplementedError
            If ``relative=True``.
        """
        self._reject_relative_pose(relative)
        return self._sim_entity.get_links_pos(ls_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_quat(self, ls_idx_local=None, envs_idx=None, *, relative: bool = False) -> torch.Tensor:
        """Return link orientations from the synchronized Genesis entity.

        Parameters
        ----------
        ls_idx_local : None | array_like, optional
            The local link indices. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The environment indices. Defaults to None.
        relative : bool, optional
            Must be False. Relative hardware pose access is not yet supported. Defaults to False.

        Returns
        -------
        quat : torch.Tensor
            The solver/world-frame link quaternions in ``(w, x, y, z)`` format.

        Raises
        ------
        NotImplementedError
            If ``relative=True``.
        """
        self._reject_relative_pose(relative)
        return self._sim_entity.get_links_quat(ls_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_vel(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._sim_entity.get_links_vel(ls_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_ang(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._sim_entity.get_links_ang(ls_idx_local=ls_idx_local, envs_idx=envs_idx)

    # -- catch-all delegation to the underlying sim entity -------------------
    #
    # Action terms (e.g. DifferentialIKController, OperationalSpaceController)
    # call kinematics/dynamics methods like ``get_link``, ``get_jacobian``,
    # ``get_mass_mat`` and ``control_dofs_pos`` / ``control_dofs_force``.
    # In normal use those run against the cached ``sim_entity`` reference
    # captured in ``ActionTerm.build()`` (before this wrapper replaces the
    # entities-dict entry), and ``update()`` has already synced the sim
    # entity's qpos from the real state so they return correct values.
    #
    # But anything that reaches the wrapper via ``env.entities[name]`` at
    # runtime would otherwise hit AttributeError. Falling through to the
    # sim entity preserves Genesis's behavior; the wrapper-specific
    # overrides above still take precedence because ``__getattr__`` is
    # only consulted when normal lookup fails.
    def __getattr__(self, name: str):
        if name.startswith("_"):
            # Avoid recursion via ``self._sim_entity`` before it's set in __init__.
            raise AttributeError(name)
        # The configured per-DOF properties must never fall through: reaching here means the sim
        # entity carried no resolved tensor, and it would answer with the raw configured value
        # (see ``_DEFAULT_DOFS_ATTRS``).
        if name in self._DEFAULT_DOFS_ATTRS:
            raise AttributeError(
                f"RobotStateEntity has no {name}: the sim entity carried no resolved {name} "
                f"tensor at connect time, so this wrapper has no copy of it. Forwarding to the "
                f"sim entity would hand back the raw configured value instead ({name} is a "
                f"mirrored options field), not a per-DOF tensor — only a built, simulated, "
                f"articulated and actuated entity resolves one."
            )
        try:
            sim_entity = object.__getattribute__(self, "_sim_entity")
        except AttributeError:
            raise AttributeError(name)
        return getattr(sim_entity, name)
