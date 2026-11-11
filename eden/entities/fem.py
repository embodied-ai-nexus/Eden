"""FEM (finite-element) soft-body entity wrapper."""

from __future__ import annotations

from typing import Literal

import torch

import eden as en
from eden.entities.base import Entity


class FEMEntity(Entity):
    """Entity subclass for FEM-based deformable materials (Elastic, Cloth, Muscle).

    FEM entities use vertex-based state (positions and velocities of mesh vertices)
    rather than rigid-body state. They have no DOFs, links, or joint-based control.

    The Genesis FEMEntity does not expose get_pos/get_vel/get_quat/get_ang, so this
    class overrides those methods to provide a compatible interface (mean vertex
    position, mean velocity, identity quaternion, zero angular velocity).
    """

    _identity_quat: torch.Tensor | None = None
    _zero_ang: torch.Tensor | None = None

    def get_pos(self, envs_idx=None, *, relative: bool = False):
        """Return mean vertex position (geometric centroid), shape (B, 3).

        Note: this is the unweighted mean of vertex positions, not the
        mass-weighted center of mass. ``relative`` must be False (see
        :meth:`~eden.entities.base.Entity._reject_relative`).
        """
        self._reject_relative(relative, "get_pos")
        state = self._entity.get_state()
        pos = state.pos  # (B, n_vertices, 3)
        centroid = pos.mean(dim=1)  # (B, 3)
        if envs_idx is not None:
            return centroid[envs_idx]
        return centroid

    def get_vel(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Return mean vertex velocity, shape (B, 3). Only world frame is supported."""
        if frame != "world":
            raise ValueError(f"FEMEntity only supports frame='world', got '{frame}'.")
        state = self._entity.get_state()
        vel = state.vel  # (B, n_vertices, 3)
        mean_vel = vel.mean(dim=1)  # (B, 3)
        if envs_idx is not None:
            return mean_vel[envs_idx]
        return mean_vel

    def get_quat(self, envs_idx=None, *, relative: bool = False):
        """Return identity quaternion (wxyz) for all envs, shape (B, 4).

        FEM entities have no rigid-body orientation. Returns a constant identity
        quaternion for interface compatibility. ``relative`` must be False (see
        :meth:`~eden.entities.base.Entity._reject_relative`).
        """
        self._reject_relative(relative, "get_quat")
        if self._identity_quat is None or self._identity_quat.shape[0] != self._env.num_envs:
            self._identity_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self._env.num_envs, -1).contiguous()
            )
        if envs_idx is not None:
            return self._identity_quat[envs_idx]
        return self._identity_quat

    def get_ang(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Return zero angular velocity for all envs, shape (B, 3). Only world frame is supported."""
        if frame != "world":
            raise ValueError(f"FEMEntity only supports frame='world', got '{frame}'.")
        if self._zero_ang is None or self._zero_ang.shape[0] != self._env.num_envs:
            self._zero_ang = torch.zeros(self._env.num_envs, 3, device=self.device)
        if envs_idx is not None:
            return self._zero_ang[envs_idx]
        return self._zero_ang

    def _set_position(self, pos, envs_idx):
        """Forward to Genesis ``set_position`` after rejecting ``envs_idx``."""
        if envs_idx is not None:
            raise NotImplementedError(
                f"FEMEntity '{self.name}' does not support per-env updates. Pass values for all envs instead."
            )
        self._entity.set_position(pos)

    def set_com_offset(self, offset, envs_idx=None):
        """Place the soft body's undeformed build-time shape, translated by ``offset``.

        Every vertex lands at its build-time position plus ``offset``, so the
        body is reset to its **undeformed** shape (any current deformation is
        discarded), and the offset is always measured from the build-time pose:
        repeated calls do not accumulate, like
        :meth:`~eden.entities.rigid.RigidEntity.set_COM_shift` is measured from
        the nominal COM. For an absolute centroid position use :meth:`set_pos`.

        Parameters
        ----------
        offset : array_like
            ``(3,)`` (applied to all envs) or ``(B, 3)`` (per-env) — an offset
            **from the build-time pose**, not an absolute world position. Moved
            to the sim device, so a host tensor, array or list is fine.
        envs_idx : None
            Per-env addressing is unsupported by Genesis FEM entities; must be None.

        Raises
        ------
        ValueError
            If ``offset`` is not rank-1/2 with a trailing dim of 3, or looks like
            per-vertex data (a ``(n_vertices, 3)`` array — use
            :meth:`set_vertex_positions`). The ``(num_envs, 3)`` vs
            ``(n_vertices, 3)`` shapes only collide when ``num_envs ==
            n_vertices``; then the per-env offset reading this method is named
            for is taken. (Genesis ``set_position``, and so :meth:`set_pos`,
            reads that shape the other way, as per-vertex positions.)
        """
        init_positions = self._entity.init_positions
        offset = torch.as_tensor(offset, dtype=init_positions.dtype, device=init_positions.device)
        # A (n_vertices, 3) array is per-vertex data, not an offset. The shape
        # only collides with a per-env (num_envs, 3) offset when the two counts
        # are equal, in which case we cannot tell them apart and take the offset
        # reading (Genesis would take the per-vertex one; see the forward below).
        n_vertices_differs_from_envs = self.n_vertices != self._env.num_envs
        looks_like_vertices = offset.ndim == 2 and offset.shape[0] == self.n_vertices and n_vertices_differs_from_envs
        if offset.shape[-1] != 3 or offset.ndim > 2 or looks_like_vertices:
            raise ValueError(
                f"set_com_offset expects (3,) or (num_envs, 3); got {tuple(offset.shape)}. "
                f"For absolute per-vertex positions (vertex dim {self.n_vertices}) use set_vertex_positions()."
            )
        # Genesis' own rank-1/2 branch adds the value to the *COM-subtracted* build-time vertices, i.e. it takes an
        # absolute centroid (that is ``set_pos``). Add the offset to the build-time vertices instead and forward
        # absolute per-vertex positions: (3,) -> (n_vertices, 3) tiled, (B, 3) -> (B, n_vertices, 3). Neither shape is
        # ambiguous to Genesis, so the num_envs == n_vertices collision never reaches its rank dispatch.
        self._set_position(init_positions + offset.unsqueeze(-2), envs_idx)

    def set_vertex_positions(self, pos, envs_idx=None):
        """Set absolute per-vertex world positions of the soft body.

        This is the rank-2 / rank-3 case of Genesis ``set_position``. Prefer
        this over :meth:`set_pos` when you have full vertex coordinates, so the
        absolute-vertex (vs. centroid) semantics is explicit at the call site.

        Parameters
        ----------
        pos : torch.Tensor
            ``(n_vertices, 3)`` (tiled to all envs) or ``(B, n_vertices, 3)``
            (full batched) absolute per-vertex positions.
        envs_idx : None
            Per-env addressing is unsupported by Genesis FEM entities; must be None.

        Raises
        ------
        ValueError
            If ``pos`` is not rank-2/3 with a trailing dim of 3 and a vertex
            dimension equal to ``n_vertices`` (for a COM offset use
            :meth:`set_com_offset`).
        """
        # For both accepted shapes the vertex count is the second-to-last axis;
        # ndim outside {2, 3} (a bare offset or a too-deep batch) has no vertex axis.
        vertex_dim = pos.shape[-2] if pos.ndim in (2, 3) else None
        if pos.shape[-1] != 3 or vertex_dim != self.n_vertices:
            raise ValueError(
                f"set_vertex_positions expects (n_vertices={self.n_vertices}, 3) or "
                f"(B, n_vertices, 3); got {tuple(pos.shape)}. For a COM offset use set_com_offset()."
            )
        self._set_position(pos, envs_idx)

    def set_pos(self, pos, envs_idx=None):
        """Set FEM state via Genesis ``set_position`` (shape-dispatched).

        Genesis interprets the tensor rank as either a centroid position or
        absolute per-vertex positions; this method only forwards. Because that dual
        meaning is invisible at the call site, **prefer the explicit
        :meth:`set_com_offset` / :meth:`set_vertex_positions`** instead. Kept for
        compatibility with the generic :class:`Entity` interface.

        Accepted shapes:

        - ``(3,)`` — world position of the vertex centroid, applied to all envs.
        - ``(n_vertices, 3)`` — absolute per-vertex positions, tiled to all envs.
        - ``(B, 3)`` — per-env world positions of the vertex centroid.
        - ``(B, n_vertices, 3)`` — full batched per-vertex absolute positions.

        ``(3,)`` and ``(B, 3)`` are **absolute**: they place the undeformed
        build-time shape so that its vertex centroid sits at that world
        position, which is what :meth:`get_pos` reads back. For an offset from
        the build-time pose use :meth:`set_com_offset`.

        When ``num_envs == n_vertices`` the ``(B, 3)`` and ``(n_vertices, 3)``
        shapes coincide, and Genesis checks ``(n_vertices, 3)`` first: a
        per-env centroid tensor is then silently read as per-vertex positions,
        collapsing the body. Use :meth:`set_vertex_positions` or
        :meth:`set_com_offset`, which name their meaning, in that case.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor (see shapes above).
        envs_idx : None
            Not supported by Genesis FEM entities. Must be None.

        Raises
        ------
        NotImplementedError
            If envs_idx is not None.
        """
        self._set_position(pos, envs_idx)

    def set_quat(self, quat, envs_idx=None):
        """FEM entities do not support set_quat.

        Raises
        ------
        NotImplementedError
            Always. Bake rotation into the morph (e.g. ``gs.morphs.Mesh(euler=...)``) instead.
        """
        raise NotImplementedError(
            f"FEMEntity '{self.name}' does not support set_quat. "
            "Bake rotation into the morph (e.g. gs.morphs.Mesh(euler=...)) instead."
        )

    def get_mass(self):
        """Return FEM entity mass. Falls back to 0 if Genesis does not support it."""
        try:
            return self._entity.get_mass()
        except (TypeError, AttributeError):
            en.logger.warning(f"FEMEntity '{self.name}': get_mass() not supported by Genesis FEM entity. Returning 0.")
            return 0

    # ------------------------------------------------------------------
    # FEM-specific state accessors
    # ------------------------------------------------------------------

    def get_vertex_positions(self, envs_idx=None):
        """Return per-vertex positions, shape (B, n_vertices, 3)."""
        state = self._entity.get_state()
        pos = state.pos
        if envs_idx is not None:
            return pos[envs_idx]
        return pos

    def get_vertex_velocities(self, envs_idx=None):
        """Return per-vertex velocities, shape (B, n_vertices, 3)."""
        state = self._entity.get_state()
        vel = state.vel
        if envs_idx is not None:
            return vel[envs_idx]
        return vel

    def set_actuation(self, actu):
        """Set muscle actuation signal (Muscle material only).

        Parameters
        ----------
        actu : torch.Tensor
            Actuation tensor. Accepted shapes:
            - (): scalar for all groups
            - (n_groups,): per-group actuation
            - (B, n_groups): batched per-group actuation
        """
        self._entity.set_actuation(actu)

    def set_muscle(self, muscle_group=None, muscle_direction=None):
        """Set muscle group IDs and/or fiber directions (Muscle material only).

        Parameters
        ----------
        muscle_group : array_like, optional
            Shape (n_elements,) — muscle group ID per element.
        muscle_direction : array_like, optional
            Shape (n_elements, 3) — unit direction vectors for muscle forces.
        """
        self._entity.set_muscle(muscle_group=muscle_group, muscle_direction=muscle_direction)

    def set_vertex_constraints(self, verts_idx_local, target_poss=None, link=None, **kwargs):
        """Pin vertices to target positions or to a rigid link.

        Thin wrapper around Genesis FEMEntity.set_vertex_constraints.

        Parameters
        ----------
        verts_idx_local : array_like
            Local vertex indices to constrain.
        target_poss : array_like, optional
            Target positions (len(verts_idx), 3). If None, uses current positions.
        link : RigidLink, optional
            Rigid link for vertices to follow.
        **kwargs
            Additional keyword arguments forwarded to Genesis (e.g., is_soft_constraint,
            stiffness, envs_idx).
        """
        self._entity.set_vertex_constraints(verts_idx_local, target_poss=target_poss, link=link, **kwargs)

    @property
    def n_vertices(self) -> int:
        return self._entity.n_vertices

    @property
    def n_elements(self) -> int:
        return self._entity.n_elements
