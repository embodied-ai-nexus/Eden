"""Entity wrappers for the Mochi implicit multi-physics solver.

Mochi (Meta's ``mochi`` engine, ported to Genesis as the ``MochiSolver`` Eden hosts in
:mod:`eden.engine.solvers.mochi`) assembles every body of a scene and a smooth penalty contact model
into one Newton system per substep. The solver hands out two entity families, and Eden wraps each:

* :class:`MochiRigidEntity` — rigid and articulated bodies (``MochiRigidMaterialOptions``). The solver's
  ``MochiEntity`` *is* a Genesis ``RigidEntity``, so the whole rigid API (DOFs, PD control, IK, ``attach_to``)
  carries over; only what the solver has no counterpart for is turned off here.
* :class:`MochiSoftEntity` — tetrahedral solids, thin shells and rods (``MochiElastic`` /
  ``MochiShell`` / ``MochiRodMaterialOptions``), a vertex-based body like :class:`FEMEntity`.

A scene holding one Mochi entity must hold only Mochi (and kinematic) entities — ``SceneOptions``
rejects the mix — and should run with ``en.init(precision="64")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import genesis as gs
import genesis.utils.geom as gs_utils_geom
import numpy as np
import torch

import eden as en
from eden.entities.base import Entity
from eden.entities.rigid import RigidEntity
from eden.utils.assets import get_asset_path

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink

    from eden.engine.solvers.mochi.mochi_entity import MochiEntity as GsMochiEntity
    from eden.engine.solvers.mochi.mochi_soft_entity import MochiSoftEntity as GsMochiSoftEntity

# A mounted soft body (``MochiSoftEntity.attach_to``) is tied to its link through the vertices of the face
# nearest the link: those within this distance (m) of the extreme along the body-centre-to-link line.
MOUNT_FACE_TOLERANCE = 1e-3
# Stiffness of the mount tie per vertex, as a multiple of the stiffness the body's own material gives one
# vertex (``E * h``, ``h`` the mean vertex spacing). Ten times keeps the tie from stretching under load
# without dominating the Newton system: at mochi's flat 1e6 N/m a 140-vertex foam sponge on a Franka took
# ~270 PCG iterations per Newton step and still missed tolerance; at ~1e4 N/m it takes ~90 (2.6x faster).
MOUNT_TIE_TO_MATERIAL = 10.0
# Damping of the tie on its violation rate: what the FinRay recipes mount their fingers with.
MOUNT_DAMPING = 1.0


def _get_link_or_raise(entity: Entity, link_name: str) -> RigidLink:
    links = [link.name for link in entity.entity.links]
    if link_name not in links:
        raise KeyError(
            f"Entity '{entity.name}' has no link '{link_name}' to mount on; its links are {links}. Fix the "
            f"link name in `attachments_dict`."
        )
    return entity.entity.get_link(link_name)


def _link_build_pose(link: RigidLink) -> tuple[np.ndarray, np.ndarray]:
    """World pose of ``link`` in the build configuration (every joint at zero), from the parsed link frames.

    The MochiSolver anchors an attachment in the link frame of the pose the scene is *built* in, and reports
    link poses only after the build — so a body to be mounted is placed by composing the parse-time frames.
    """
    pos, quat = gs_utils_geom.zero_pos(), gs_utils_geom.identity_quat()
    while link is not None:
        pos, quat = gs_utils_geom.transform_pos_quat_by_trans_quat(pos, quat, link.desc.pos, link.desc.quat)
        link = link.solver.links[link.parent_idx] if link.parent_idx >= 0 else None
    return pos, quat


def _mount_face(verts_link: np.ndarray) -> np.ndarray:
    """Indices of the vertices on the face of a mounted body nearest its link's origin."""
    toward_link = -verts_link.mean(axis=0)
    toward_link /= np.linalg.norm(toward_link)
    reach = verts_link @ toward_link
    return np.flatnonzero(reach >= reach.max() - MOUNT_FACE_TOLERANCE)


class MochiRigidEntity(RigidEntity):
    """A rigid or articulated body simulated by the MochiSolver.

    Everything :class:`RigidEntity` offers works unchanged except what the solver never reads:
    drive-force ranges, joint friction loss and constraint-solver parameters (rejected at options
    validation), link mass/inertia edits after build (``set_mass`` raises), support surfaces and
    AABB queries. Contact behaviour is a material property (``friction``, ``penalty_coefficient``,
    ...) and can be re-tuned per link with :meth:`set_contact_params`.
    """

    _entity: GsMochiEntity

    # ---- post_build seams --------------------------------------------------------------------

    @property
    def _needs_explicit_damping_armature(self) -> bool:
        """Mochi integrates the PD controller implicitly, so no ``kd * dt`` armature correction applies."""
        return False

    def _stabilize_free_joints(self) -> None:
        """No-op: the fully implicit solver needs no free-joint damping bump."""

    def _init_dof_force_limits(self) -> None:
        """Record the effort limits for observation/reward terms without writing a force range.

        Mochi never clamps drive forces, so there is no range to set; the limits still come from
        the actuator spec (or the asset) so ``default_dofs_force_limits`` stays readable.
        """
        spec_effort_limit = {key: spec.EFFORT_LIMIT for key, spec in self.dofs_spec.items()}
        self.default_dofs_force_limits = self._init_dofs_property(
            "default_dofs_force_limits",
            None,
            default_info=self._parse_dofs_info(
                spec_effort_limit,
                default_info=self.get_dofs_force_range(envs_idx=0)[1].squeeze(0),
            ),
            apply_to_sim=False,
        )

    def _apply_constraint_sol_params(self) -> None:
        """No-op: penalty contact has no constraint-solver parameters (the options are rejected up front)."""

    def set_mass(self, mass):
        raise NotImplementedError(
            f"MochiRigidEntity '{self.name}': link masses are fixed at build by the MochiSolver. Set "
            f"`MochiRigidMaterialOptions.rho` or give the asset an explicit mass instead."
        )

    # ---- Mochi-specific API ---------------------------------------------------------------------

    def get_contacts(
        self, with_entity: Entity | None = None, exclude_self_contact: bool = False, is_padded: bool = False
    ):
        """Contact records of this body, optionally filtered to the pair with ``with_entity``.

        Requires ``MochiOptions.record_contacts`` (the default). Each record carries both sides'
        entities/links/geoms (-1 where a side is deformable), the deformable-side vertices and
        barycentric weights, position, unit normal (away from side B), signed distance, force on
        A and quadrature weight. Shapes are ``(n_envs, n_contacts, ...)`` when batched.

        Parameters
        ----------
        with_entity : Entity, optional
            Restrict to contacts with this (Eden) entity. Defaults to None (all partners).
        exclude_self_contact : bool, optional
            Drop contacts between two links of this entity. Defaults to False.
        is_padded : bool, optional
            Return fixed-size padded arrays with a ``valid_mask``, the invalid slots' floating-point fields zeroed;
            taken regardless while the env's graded phase runs under a CUDA graph (see
            :meth:`~eden.entities.base.Entity._query_contacts`). Defaults to False.
        """
        partner = None if with_entity is None else with_entity._entity
        return self._query_contacts(
            lambda padded: self._entity.get_contacts(
                with_entity=partner, exclude_self_contact=exclude_self_contact, is_padded=padded
            ),
            is_padded,
        )

    def set_contact_params(self, *, links: str | list[str] | None = None, **params) -> None:
        """Re-tune the material's contact parameters after build, for all links or a named subset.

        Parameters
        ----------
        links : str or list[str], optional
            Link names or regex patterns (resolved with :meth:`find_named_links_idx_local`).
            Defaults to None (every link).
        **params
            Any of ``penalty_coefficient``, ``friction``, ``penalty_smoothing_half_distance``,
            ``penalty_threshold``, ``friction_falloff_vel``, ``viscous_friction``,
            ``normal_viscous_damping``.
        """
        links_idx_local = None
        if links is not None:
            _, links_idx_local = self.find_named_links_idx_local(links, preserve_order=True)
        self._entity.set_contact_params(links_idx_local=links_idx_local, **params)

    def set_friction(self, friction: float) -> None:
        """Set the Coulomb friction coefficient of every link."""
        self._entity.set_friction(friction)

    def set_has_gravity(self, has_gravity: bool) -> None:
        """Switch gravity on or off for this body."""
        self._entity.set_has_gravity(has_gravity)

    def set_contact_enabled(self, other: Entity, enabled: bool = True) -> None:
        """Enable or disable contact between this entity and ``other`` (post-build, symmetric)."""
        self._env.mochi_solver.enable_entity_contact(self._entity, other._entity, enabled)


class MochiSoftEntity(Entity):
    """A tetrahedral solid, thin shell or rod simulated by the MochiSolver.

    Vertex-based like :class:`FEMEntity`: no links, DOFs or rigid pose. ``get_pos`` is the vertex
    centroid, ``get_quat`` the identity, and ``set_quat`` raises — bake the orientation into
    ``default_root_quat`` (the morph) instead. Four calls are *pre-build only* because they change
    what the solver assembles: :meth:`attach_to_link`, :meth:`set_visual_mesh`,
    :meth:`set_contact_skin` and :meth:`set_rest_positions`. Register them with ``env.register_pre_build_hook``
    so they run after ``scene.add_entity`` (when :attr:`init_positions` exists) and before ``scene.build()``::

        env.register_pre_build_hook(lambda: finger.attach_to_link(gripper, "finger_link", verts_idx))
        env.build()

    :meth:`set_visual_mesh` goes through the morph transform, so it must precede
    :meth:`set_rest_positions` (it raises otherwise). :meth:`set_contact_skin` cannot be combined with
    :meth:`set_rest_positions` in either order (both raise). :meth:`attach_to_link` only records indices
    and can go anywhere in the hook.
    """

    _entity: GsMochiSoftEntity
    # Set by :meth:`attach_to`: the carrying entity, its link name, and this body's rest vertices in
    # that link's frame (what ``post_build`` carries to the link's post-build pose).
    _mount: tuple[MochiRigidEntity, str, np.ndarray] | None = None
    _identity_quat: torch.Tensor | None = None
    _zero_ang: torch.Tensor | None = None
    # Whether ``set_rest_positions`` has moved the rest geometry out of the morph frame, which
    # ``set_visual_mesh`` (placed through that frame) can no longer be embedded against.
    _rest_positions_moved: bool = False
    # Whether ``set_contact_skin`` has placed a skin through the morph frame. The solver embeds it in the
    # tetrahedra only at build, so it cannot follow a later ``set_rest_positions`` either.
    _has_contact_skin: bool = False

    # ---- generic Entity interface --------------------------------------------------------------

    def get_pos(self, envs_idx=None):
        """Mean vertex position (unweighted centroid), shape ``(B, 3)``."""
        pos = self.get_vertex_positions()
        centroid = pos.mean(dim=1)
        return centroid if envs_idx is None else centroid[envs_idx]

    def get_vel(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Mean vertex velocity, shape ``(B, 3)``. Only the world frame is defined."""
        if frame != "world":
            raise ValueError(f"MochiSoftEntity only supports frame='world', got '{frame}'.")
        vel = self.get_vertex_velocities().mean(dim=1)
        return vel if envs_idx is None else vel[envs_idx]

    def get_quat(self, envs_idx=None):
        """Identity quaternion (wxyz) for every env, shape ``(B, 4)``: a soft body has no rigid orientation."""
        if self._identity_quat is None or self._identity_quat.shape[0] != self._env.num_envs:
            self._identity_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self._env.num_envs, -1).contiguous()
            )
        return self._identity_quat if envs_idx is None else self._identity_quat[envs_idx]

    def get_ang(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Zero angular velocity for every env, shape ``(B, 3)``."""
        if frame != "world":
            raise ValueError(f"MochiSoftEntity only supports frame='world', got '{frame}'.")
        if self._zero_ang is None or self._zero_ang.shape[0] != self._env.num_envs:
            self._zero_ang = torch.zeros(self._env.num_envs, 3, device=self.device)
        return self._zero_ang if envs_idx is None else self._zero_ang[envs_idx]

    def set_pos(self, pos, envs_idx=None):
        """Move the body with Genesis ``set_position`` semantics.

        ``pos`` is a ``(3,)`` / ``(B, 3)`` offset of the rest positions, or full ``(n_vertices, 3)`` /
        ``(B, n_vertices, 3)`` vertex positions. Because that dual meaning is invisible at the call site,
        prefer the explicit :meth:`set_com_offset` / :meth:`set_vertex_positions`.
        """
        self._entity.set_position(pos, envs_idx=envs_idx)

    def set_com_offset(self, offset, envs_idx=None):
        """Translate the body by an offset of its rest positions, shape ``(3,)`` or ``(B, 3)``."""
        offset = torch.as_tensor(offset)
        if offset.shape[-1] != 3 or offset.ndim > 2 or (offset.ndim == 2 and offset.shape[0] == self.n_vertices):
            raise ValueError(
                f"set_com_offset expects (3,) or (num_envs, 3); got {tuple(offset.shape)}. For absolute "
                f"per-vertex positions (vertex dim {self.n_vertices}) use set_vertex_positions()."
            )
        self._entity.set_position(offset, envs_idx=envs_idx)

    def set_vertex_positions(self, pos, envs_idx=None):
        """Set absolute world vertex positions, shape ``(n_vertices, 3)`` or ``(B, n_vertices, 3)``."""
        pos = torch.as_tensor(pos)
        if pos.ndim < 2 or pos.shape[-2] != self.n_vertices or pos.shape[-1] != 3:
            raise ValueError(
                f"set_vertex_positions expects ({self.n_vertices}, 3) or (num_envs, {self.n_vertices}, 3); "
                f"got {tuple(pos.shape)}."
            )
        self._entity.set_position(pos, envs_idx=envs_idx)

    def set_quat(self, quat, envs_idx=None):
        raise NotImplementedError(
            f"MochiSoftEntity '{self.name}' does not support set_quat. Bake the orientation into "
            f"`default_root_quat` (the morph) instead."
        )

    def get_mass(self):
        """Total mass (kg) from the material density and the rest geometry."""
        return torch.as_tensor(float(self._entity.mass), dtype=gs.tc_float, device=self.device)

    def set_mass(self, mass):
        raise NotImplementedError(
            f"MochiSoftEntity '{self.name}': mass follows the material density; set `rho` on the material."
        )

    # ---- vertex state ---------------------------------------------------------------------------

    def get_vertex_positions(self, envs_idx=None) -> torch.Tensor:
        """World vertex positions, shape ``(B, n_vertices, 3)``."""
        return self._with_env_axis(self._entity.get_vertices_position(envs_idx))

    def get_vertex_velocities(self, envs_idx=None) -> torch.Tensor:
        """Vertex velocities, shape ``(B, n_vertices, 3)``."""
        return self._with_env_axis(self._entity.get_vertices_velocity(envs_idx))

    def get_vertex_contact_forces(self, envs_idx=None) -> torch.Tensor:
        """Net contact force on each vertex, shape ``(B, n_vertices, 3)``."""
        return self._with_env_axis(self._entity.get_vertices_contact_force(envs_idx))

    def _with_env_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        # Genesis drops the env axis for an unbatched scene (``n_envs == 0``); Eden always keeps it.
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor

    def set_vertex_velocities(self, vel, envs_idx=None) -> None:
        """Set one velocity for every vertex (``(3,)`` / ``(B, 3)``) or one per vertex (``(n_vertices, 3)`` /
        ``(B, n_vertices, 3)``).
        """
        self._entity.set_velocity(vel, envs_idx=envs_idx)

    def set_vertices_fixed(self, verts_idx_local, is_fixed: bool = True, envs_idx=None) -> None:
        """Pin (or release) vertices at their current positions (a Dirichlet condition)."""
        self._entity.set_vertices_fixed(verts_idx_local, is_fixed=is_fixed, envs_idx=envs_idx)

    def set_vertices_target(self, verts_idx_local, pos, envs_idx=None) -> None:
        """Drive pinned vertices to ``pos`` (``(n, 3)`` or ``(B, n, 3)``), reached exactly at the end of the next step.

        The vertices stay fixed afterwards until :meth:`set_vertices_fixed` releases them.
        """
        self._entity.set_vertices_target(verts_idx_local, pos, envs_idx=envs_idx)

    def get_contacts(
        self, with_entity: Entity | None = None, exclude_self_contact: bool = False, is_padded: bool = False
    ):
        """Contact records of this body; see :meth:`MochiRigidEntity.get_contacts`."""
        partner = None if with_entity is None else with_entity._entity
        return self._query_contacts(
            lambda padded: self._entity.get_contacts(
                with_entity=partner, exclude_self_contact=exclude_self_contact, is_padded=padded
            ),
            is_padded,
        )

    def set_contact_params(self, **params) -> None:
        """Re-tune the material's contact parameters after build (see :meth:`MochiRigidEntity.set_contact_params`)."""
        self._entity.set_contact_params(**params)

    def set_contact_enabled(self, other: Entity, enabled: bool = True) -> None:
        """Enable or disable contact between this entity and ``other`` (post-build, symmetric)."""
        self._env.mochi_solver.enable_entity_contact(self._entity, other._entity, enabled)

    # ---- mounting on a link (SceneOptions.attachments_dict) ---------------------------------------

    def attach_to(
        self,
        entity: Entity,
        link_name: str,
        attach_offset=None,
        attach_quat=None,
    ) -> None:
        """Mount this body on a link of a Mochi rigid entity — what ``SceneOptions.attachments_dict`` calls.

        The mounting transform has the meaning it has for :meth:`RigidEntity.attach_to`: the body's own
        frame (its ``default_root_pos`` / ``default_root_quat`` frame, in which a primitive is centred) sits
        at ``(attach_offset, attach_quat)`` in the link frame, defaulting to ``default_root_pos`` /
        ``default_root_quat``. The vertices of the face nearest the link origin — those within
        :data:`MOUNT_FACE_TOLERANCE` of the extreme along the line from the body centre to the link
        origin — are tied to the link by :meth:`attach_to_link`; the rest of the body hangs off them and
        deforms. A sponge mounted under a gripper is ``{"sponge": ("robot", "hand", (0, 0, 0.13), (1, 0, 0, 0))}``.
        Each tie is :data:`MOUNT_TIE_TO_MATERIAL` times as stiff as the body's material is at one vertex. Only
        tetrahedral solids mount this way; a shell or a rod raises.

        The solver anchors each tied vertex in the link frame of the *build* configuration (every joint at
        zero), so the rest geometry is placed there before the build, and :meth:`post_build` then carries the
        whole body to wherever the parent's default joint positions put the link.
        """
        if not isinstance(self.material, en.materials.MochiElasticMaterialOptions):
            raise NotImplementedError(
                f"MochiSoftEntity '{self.name}' is a {type(self.material).__name__} body; attachments_dict mounts "
                f"tetrahedral solids (MochiElasticMaterialOptions) only. Tie a shell or rod to a link with "
                f"`attach_to_link(...)` from `env.register_pre_build_hook(...)`, choosing the vertices yourself."
            )
        if not isinstance(entity, MochiRigidEntity):
            raise TypeError(
                f"MochiSoftEntity '{self.name}' can only be mounted on a Mochi rigid entity, but its parent "
                f"'{entity.name}' is a {type(entity).__name__}: a Mochi scene holds Mochi bodies only. Give "
                f"'{entity.name}' `material=MochiRigidMaterialOptions(...)`."
            )
        mount_pos = np.asarray(self._options.default_root_pos if attach_offset is None else attach_offset)
        mount_quat = np.asarray(self._options.default_root_quat if attach_quat is None else attach_quat)
        if np.linalg.norm(mount_pos) < MOUNT_FACE_TOLERANCE:
            raise ValueError(
                f"MochiSoftEntity '{self.name}' is mounted with its centre at the origin of link '{link_name}' "
                f"(offset {tuple(mount_pos.tolist())}), so no face of it is nearest the link and none can be "
                f"tied. Offset the body to sit against the link, e.g. attach_offset=(0, 0, <half its height>)."
            )

        def _do_attach():
            gs_link = _get_link_or_raise(entity, link_name)
            # Rest geometry in the body frame, the mesh as authored (scaled) about its own origin. The morph rotated
            # it about its vertex centroid, not about default_root_pos, so undo exactly that placement.
            rest_world = np.asarray(self.init_positions, dtype=gs.np_float)
            morph_pos = np.asarray(self._entity.morph.pos, dtype=gs.np_float)
            verts_body = (self._entity.world_to_morph_frame(rest_world) - morph_pos).astype(gs.np_float)
            verts_link = gs_utils_geom.transform_by_trans_quat(
                verts_body, mount_pos.astype(gs.np_float), mount_quat.astype(gs.np_float)
            )
            tied = _mount_face(verts_link)
            if len(tied) < 3:
                raise ValueError(
                    f"MochiSoftEntity '{self.name}': only {len(tied)} vertex(es) lie on the face nearest link "
                    f"'{link_name}', too few to hold the body. Refine the mesh, or rotate the body so a face "
                    f"(not an edge or corner) points at the link."
                )
            link_pos, link_quat = _link_build_pose(gs_link)
            self.set_rest_positions(gs_utils_geom.transform_by_trans_quat(verts_link, link_pos, link_quat))
            extent = verts_link.max(axis=0) - verts_link.min(axis=0)
            spacing = float(np.prod(extent) / len(verts_link)) ** (1.0 / 3.0)
            stiffness = MOUNT_TIE_TO_MATERIAL * self.material.E * spacing
            self._entity.attach_to_link(gs_link, tied, stiffness=stiffness, damping=MOUNT_DAMPING)
            self._mount = (entity, link_name, verts_link)

        self._env.register_pre_build_hook(_do_attach)
        self._is_attached = True
        self._attachment_parent = entity

    def post_build(self) -> None:
        """Carry a mounted body from the build pose of its link to the link's current pose."""
        if self._mount is None:
            return
        parent, link_name, verts_link = self._mount
        link = parent.get_link(link_name)
        verts = torch.as_tensor(verts_link, dtype=gs.tc_float, device=self.device)
        verts = verts.unsqueeze(0).expand(self._env.num_envs, -1, -1)
        pos = link.get_pos().unsqueeze(1).expand_as(verts)
        quat = link.get_quat().unsqueeze(1).expand(-1, verts.shape[1], -1)
        world = gs_utils_geom.transform_by_trans_quat(verts, pos, quat)
        self.set_vertex_positions(world)
        self.set_vertex_velocities(torch.zeros_like(world))

    # ---- pre-build-only ------------------------------------------------------------------------

    @gs.assert_unbuilt
    def attach_to_link(
        self, entity: MochiRigidEntity, link_name: str, verts_idx, stiffness: float = 1e6, damping: float = 0.0
    ) -> None:
        """Tie vertices rigidly to a link of a Mochi rigid entity (pre-build only).

        Each vertex is bound by a stiff penalty to the point of the link it coincides with at build
        time, inside the same implicit solve — so place the soft body at its mounted pose. Call from
        an ``env.register_pre_build_hook`` callback, where :attr:`init_positions` is available to
        pick ``verts_idx``.

        Parameters
        ----------
        entity : MochiRigidEntity
            The rigid/articulated entity carrying the link.
        link_name : str
            Name of the link to attach to.
        verts_idx : array_like
            Local indices of the vertices to attach.
        stiffness : float, optional
            Penalty stiffness of the tie. Defaults to 1e6.
        damping : float, optional
            Damping on the tie violation rate. Defaults to 0.0.
        """
        self._require_added("attach_to_link")
        self._entity.attach_to_link(entity.get_link(link_name), verts_idx, stiffness=stiffness, damping=damping)

    @gs.assert_unbuilt
    def set_rest_positions(self, positions) -> None:
        """Move the body's rest geometry before the scene is built (pre-build only).

        The build reads :attr:`init_positions` to fill the solver, and :meth:`attach_to_link` anchors each
        vertex in the frame of the link it coincides with *at build time* — so a soft body mounted on a link
        whose pose is only known once the model is parsed is placed here, from a pre-build hook, rather than
        through the morph. ``positions`` must have the body's own vertex count and ordering.

        Call :meth:`set_visual_mesh` *before* this, not after: it places the render mesh through the morph
        transform and binds it to the tetrahedra by barycentric coordinates, which then carry it along.

        Parameters
        ----------
        positions : array_like
            World rest positions, shape ``(n_vertices, 3)``.
        """
        self._require_added("set_rest_positions")
        if self._has_contact_skin:
            raise RuntimeError(
                f"MochiSoftEntity '{self.name}': set_rest_positions() cannot follow set_contact_skin(). The skin "
                f"is placed through the morph transform but embedded in the tetrahedra only at build, so moved "
                f"rest positions leave it outside the body and its contact samples act from the wrong place. "
                f"Swapping the two calls does not help either: place the body through the morph "
                f"(`default_root_pos` / `default_root_quat`) to keep the skin, or drop the skin."
            )
        positions = np.asarray(positions, dtype=self._entity.init_positions.dtype)
        if positions.shape != self._entity.init_positions.shape:
            raise ValueError(
                f"MochiSoftEntity '{self.name}'.set_rest_positions expects "
                f"{self._entity.init_positions.shape}, got {positions.shape}."
            )
        self._entity.init_positions = positions
        self._rest_positions_moved = True

    @gs.assert_unbuilt
    def set_visual_mesh(self, file: str, pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0), scale: float = 1.0) -> None:
        """Skin a detailed render mesh onto the simulation tetrahedra (solids only, pre-build only).

        ``pos`` / ``euler`` / ``scale`` place the render mesh in the *simulation mesh's file frame*; the
        entity's morph transform is then applied to it, and each of its vertices is bound to the tetrahedron
        it falls in by barycentric coordinates, so it follows the deformation. Because the placement goes
        through the morph, this must run *before* :meth:`set_rest_positions` — which raises here if it
        already ran, rather than leaving a render mesh embedded against geometry that has since moved away
        (the visual mesh then extrapolates far outside the body, and the body renders as nothing).

        Parameters
        ----------
        file : str
            Mesh file (resolved through :func:`eden.utils.assets.get_asset_path`).
        pos : array-like[float, float, float], optional
            Offset of the render mesh in the simulation mesh's file frame. Defaults to zero.
        euler : array-like[float, float, float], optional
            Rotation of the render mesh in degrees. Defaults to zero.
        scale : float, optional
            Uniform scale of the render mesh. Defaults to 1.0.
        """
        self._require_added("set_visual_mesh")
        if self._rest_positions_moved:
            raise RuntimeError(
                f"MochiSoftEntity '{self.name}': set_visual_mesh() must be called before "
                f"set_rest_positions(). The render mesh is placed through the morph transform and embedded "
                f"in the tetrahedra as they stand now, so embedding it after the rest geometry has moved "
                f"binds it to the wrong place. Swap the two calls: the barycentric embedding carries the "
                f"render mesh along when the rest positions move."
            )
        self._entity.set_visual_mesh(file=get_asset_path(file), pos=pos, euler=euler, scale=scale)

    @gs.assert_unbuilt
    def set_contact_skin(self, file: str, pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0), scale: float = 1.0) -> None:
        """Carry this solid's contact samples on a detailed triangle surface (solids only, pre-build only).

        The samples otherwise lie on the boundary triangles of the tetrahedral mesh, so a coarse mesh contacts
        through a coarse surface. A skin (fingertip pads, the authored collision surface) places them where the
        contact happens, at the cost of more samples. The skin only carries this body's samples: other bodies
        still collide against its tetrahedra. Placement works as in :meth:`set_visual_mesh`, but the solver
        embeds the skin in the tetrahedra only at build, so it cannot be combined with
        :meth:`set_rest_positions` (it raises if the rest geometry has already moved).

        Parameters
        ----------
        file : str
            Mesh file (resolved through :func:`eden.utils.assets.get_asset_path`).
        pos : array-like[float, float, float], optional
            Offset of the skin in the simulation mesh's file frame. Defaults to zero.
        euler : array-like[float, float, float], optional
            Rotation of the skin in degrees. Defaults to zero.
        scale : float, optional
            Uniform scale of the skin. Defaults to 1.0.
        """
        self._require_added("set_contact_skin")
        if self._rest_positions_moved:
            raise RuntimeError(
                f"MochiSoftEntity '{self.name}': set_contact_skin() cannot follow set_rest_positions(). The skin "
                f"is placed through the morph transform, so it would sit where the body was before the move. "
                f"Calling it first does not help either (the skin is embedded only at build): place the body "
                f"through the morph (`default_root_pos` / `default_root_quat`) to keep the skin, or drop the skin."
            )
        self._entity.set_contact_skin(file=get_asset_path(file), pos=pos, euler=euler, scale=scale)
        self._has_contact_skin = True

    def _require_added(self, method: str) -> None:
        if not hasattr(self, "_entity"):
            raise RuntimeError(
                f"MochiSoftEntity '{self.name}'.{method}() must run after the entity is added to the scene and "
                f"before it is built: call it from `env.register_pre_build_hook(...)`, not before `env.build()`."
            )

    # ---- geometry ------------------------------------------------------------------------------

    @property
    def init_positions(self) -> np.ndarray:
        """Rest vertex positions in the world frame, shape ``(n_vertices, 3)`` (available once added to the scene)."""
        self._require_added("init_positions")
        return self._entity.init_positions

    @property
    def elems(self) -> np.ndarray:
        """Element connectivity: ``(n, 4)`` tetrahedra, ``(n, 3)`` shell triangles or ``(n, 2)`` rod segments."""
        self._require_added("elems")
        return self._entity.elems

    @property
    def n_vertices(self) -> int:
        return self._entity.n_vertices

    @property
    def n_elements(self) -> int:
        return self._entity.n_elements

    @property
    def is_shell(self) -> bool:
        return isinstance(self.material, en.materials.MochiShellMaterialOptions)

    @property
    def is_rod(self) -> bool:
        return isinstance(self.material, en.materials.MochiRodMaterialOptions)
