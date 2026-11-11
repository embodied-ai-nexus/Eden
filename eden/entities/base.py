"""Base Entity wrapper providing shared pose/state accessors over a Genesis entity.

:class:`Entity` holds shared state (morph, material, pose getters/setters) and is
specialized by :class:`~eden.entities.rigid.RigidEntity` (articulated/actuated),
:class:`~eden.entities.particle.ParticleEntity` (MPM/SPH/PBD),
:class:`~eden.entities.fem.FEMEntity`, the Mochi entities in :mod:`eden.entities.mochi`, and
:class:`~eden.entities.terrain.Terrain`. ``eden.envs.base._entity_class_for_options`` picks the
subclass from the options (``TerrainOptions``) and the material type.

All quaternions are wxyz. Pose setters (``set_pos`` / ``set_quat``) take an ``envs_idx``
to target specific environments; note that ``ParticleEntity`` bakes its pose into the
morph at creation time and raises from ``set_pos`` / ``set_quat``, as does
``FEMEntity.set_quat``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
from genesis.engine.materials.base import Material
from genesis.typing import UnitVec4FType, Vec3FType
from genesis.utils.geom import inv_transform_by_quat, transform_by_quat

import eden as en
from eden._logging import collapse_repeated_genesis_warnings
from eden.engine.solvers.mochi import materials as mochi_materials
from eden.options import EntityOptions, GroupedEntityOptions, PrimitiveOptions
from eden.options.entities import MESH_FILE_SUFFIXES
from eden.options.materials import MaterialLike
from eden.options.surfaces import PlasticSurfaceOptions, SurfaceLike, resolve_surface
from eden.types import MorphLike, VisMode
from eden.utils.common import ConfigurableMixin, mirror_options_onto
from eden.utils.geom import align_up_and_front, heading_from_quat

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity as GenesisEntity
    from genesis.options.surfaces import Surface

    from eden.envs.base import EnvBase


class Entity(ConfigurableMixin[EntityOptions]):
    """
    Entity class for Eden, wrapping Genesis' Entity class.

    Parameters
    ----------
    default_root_pos: array-like[float, float, float]
        The default position of the entity's root link.
    default_root_quat: array-like[float, float, float, float]
        The default orientation of the entity's root link (w, x, y, z).
    """

    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)
    default_root_quat: UnitVec4FType = (
        1.0,
        0.0,
        0.0,
        0.0,
    )  # (w, x, y, z)

    forward_vec: Vec3FType = (1.0, 0.0, 0.0)

    if TYPE_CHECKING:
        morph: MorphLike
    material: MaterialLike | None = None
    surface: SurfaceLike | None = None
    visualize_contact: bool = False
    vis_mode: VisMode = "visual"

    is_fixed_base: bool = False

    def __init__(self, env: EnvBase, options: EntityOptions):
        self._options = options
        self._env = env
        self._is_attached = False
        self._attachment_parent: Entity | None = None
        self.is_heterogeneous = isinstance(self._options, GroupedEntityOptions)
        # Heterogeneous only: each env's own variant default pose, (num_envs, 3) / (num_envs, 4).
        self._variant_default_root_pos: torch.Tensor | None = None
        self._variant_default_root_quat: torch.Tensor | None = None
        # Heterogeneous only: the variant each env holds, (num_envs,) long, and the first env holding each variant
        # present (all of them, unless there are fewer envs than variants).
        self._env_variant_idx: torch.Tensor | None = None
        self._variant_first_envs: list[int] = []
        # Heterogeneous only: each present variant's value for a per-variant field they differ in; a field
        # they all resolve alike is mirrored as that one value instead. And each variant's material, present or
        # not, so the checks on it do not depend on the env count.
        self._variant_fields: dict[str, list] = {}
        self._variant_materials: list[MaterialLike] | None = None

        # Create morph from options if not already set
        if not self.is_heterogeneous:
            self._create_morph_from_options()
        else:
            self._create_grouped_morphs_from_options()

        # Mirror options onto the instance. Entity does not delegate __init__ to
        # ConfigurableMixin -- see mirror_options_onto's docstring for why, and for the
        # aliasing this protects against. Unlike the mixin, Entity sets no ``_uid``: its
        # ``uid`` is the wrapped Genesis entity's, so there is nothing to seed here.
        # SupportSurfaceMixin's attributes need no skip list -- get_parameter_names() walks
        # only ConfigurableMixin bases, so they never show up here.
        mirror_options_onto(self, self._options)
        if self.is_heterogeneous:
            self._resolve_variant_fields()

        # forward_vec is derived from the options' front direction (single source of truth);
        # entities without a front (e.g. primitives) keep the class default.
        self.forward_vec = getattr(self._options, "front", self.forward_vec)
        self.forward_vec = torch.tensor(self.forward_vec, dtype=gs.tc_float, device=self.device).unsqueeze(0)
        self.default_root_pos = torch.tensor(self.default_root_pos, dtype=gs.tc_float, device=self.device).unsqueeze(0)
        self.default_root_quat = torch.tensor(
            self.default_root_quat,
            dtype=gs.tc_float,
            device=self.device,
        ).unsqueeze(0)

    def _create_morph_from_options(self):
        """Create morph from EntityOptions fields."""
        if isinstance(self._options, PrimitiveOptions):
            self.morph = self._options.get_morph()
            return

        if not isinstance(self._options, EntityOptions):
            return

        # Resolve any runtime-only descriptor before locating the asset. The source
        # identity remains stored on the options so serialized configs stay portable.
        runtime_file = self._options.resolve_runtime_file()
        file = self._options.fetch_asset(runtime_file)

        # Compute the up/front alignment quaternion that rotates the asset's canonical
        # axes onto the world axes. We store it as the morph's *offset* rather than its
        # `quat`. Genesis >=1.1.1 reserves `offset_quat` for exactly this kind of fixed
        # body-frame frame conversion: the world pose becomes ``user_pose ∘ offset`` and
        # the offset is stripped back out by the relative pose accessors. Keeping the
        # alignment here (instead of in `quat`) means a later ``set_quat(q, relative=True)``
        # composes the user orientation *onto* the alignment (relative to qpos0) instead of
        # overwriting it.
        align_quat = tuple(float(v) for v in gu.R_to_quat(align_up_and_front(self._options.up, self._options.front)))

        # Prepare common options for morph creation
        common_options = dict(
            file=file,
            pos=self._options.default_root_pos,
            offset_quat=align_quat,
            batch_fixed_verts=self._options.batch_fixed_verts,
            align=self._options.align,
            scale=self._options.scale,
            visualization=self._options.visualization,
            collision=self._options.collision,
            convexify=self._options.convexify,
            decompose_robot_error_threshold=self._options.decompose_robot_error_threshold,
            decompose_object_error_threshold=self._options.decompose_object_error_threshold,
            decimate=self._options.decimate,
            decimate_face_num=self._options.decimate_face_num,
            decimate_aggressiveness=self._options.decimate_aggressiveness,
            coacd_options=gs.options.CoacdOptions(
                threshold=self._options.coacd_threshold,
                preprocess_resolution=self._options.coacd_preprocess_resolution,
                # NOTE: COACD's PCA preprocessing rotates the mesh to its principal axes
                # before decomposition and returns hulls in that rotated frame without
                # applying the inverse transform. That leaves the collision hulls misaligned
                # with the visual mesh. Default to False (matches Genesis CoacdOptions).
                pca=False,
                decimate=self._options.decimate,
            ),
        )

        # A fixed base cannot be re-posed after build (a fixed URDF with an empty base link has no
        # settable base pose at all), so bake its orientation into the morph — as PrimitiveOptions
        # already does. A floating base's orientation is not baked: its morph carries only ``pos``, which
        # composes with the asset's own root-body pose (an MJCF root ``<body pos=... quat=...>``; the G1
        # pelvis sits at z = 0.793), and the post-build write in ``RigidEntity._set_initial_root_pose``
        # then replaces the whole root pose, so ``default_root_pos`` / ``default_root_quat`` end up as the
        # root body's own pose. Vertex-based bodies (FEM / particle / Mochi soft) have no post-build
        # ``set_quat`` either, so their orientation is baked too — otherwise ``default_root_quat`` on a
        # file-mesh soft body would be dropped without a word.
        from eden.options.entities import NON_RIGID_MATERIAL_TYPES

        if self._options.is_fixed_base or isinstance(self._options.material, NON_RIGID_MATERIAL_TYPES):
            common_options["quat"] = self._options.default_root_quat

        # Create morph based on file extension, matched case-insensitively as Genesis' own ``Morph.is_format`` does
        file_lower = file.lower()
        if file_lower.endswith(".urdf"):
            self.morph = gs.morphs.URDF(
                fixed=self._options.is_fixed_base,
                links_to_keep=self._options.links_to_keep,
                default_armature=self._options.default_armature,
                merge_fixed_links=not hasattr(self._env, "_nyx_splat_options"),
                recompute_inertia=self._options.recompute_inertia,
                **common_options,
            )
        elif file_lower.endswith(".xml"):
            # Genesis's `gs.morphs.MJCF` has no `fixed` parameter: the free/fixed distinction is
            # encoded inside the MJCF (a `<freejoint/>` under the root body), so `is_fixed_base`
            # must *agree* with the asset. `RigidEntity.post_build` raises when it does not, reading
            # the built base link, where an `<include>` or a runtime rewrite can no longer hide it.
            self.morph = gs.morphs.MJCF(default_armature=self._options.default_armature, **common_options)
        elif file_lower.endswith((".usd", ".usda", ".usdc", ".usdz")):
            self.morph = gs.morphs.USD(
                fixed=self._options.is_fixed_base,
                default_armature=self._options.default_armature,
                **common_options,
            )
        elif file_lower.endswith(MESH_FILE_SUFFIXES):
            self.morph = gs.morphs.Mesh(
                fixed=self._options.is_fixed_base,
                contype=self._options.contype,
                conaffinity=self._options.conaffinity,
                recompute_inertia=self._options.recompute_inertia,
                **common_options,
            )
        else:
            raise ValueError(f"Invalid file type: {file}")

    def _create_grouped_morphs_from_options(self):
        """Create grouped morphs for heterogeneous entities, and record which variant each env holds."""
        from genesis.engine.solvers.kinematic_solver import _balanced_variant_mapping

        from eden.options.entities import GroupedEntityOptions

        if not isinstance(self._options, GroupedEntityOptions):
            return

        _SUPPORTED_MORPH_TYPES = (gs.morphs.Primitive, gs.morphs.Mesh, gs.morphs.URDF, gs.morphs.MJCF)

        morphs_heterogeneous = []
        variants_pos, variants_quat = [], []
        for entity_options in self._options.grouped_entities:
            # Prevent nested grouped entities, which would cause infinite recursion
            if isinstance(entity_options, GroupedEntityOptions):
                raise ValueError(
                    "Invalid configuration: GroupedEntityOptions cannot contain other GroupedEntityOptions."
                    " 'grouped_entities' must contain only non-grouped EntityOptions."
                )
            # Create a temporary entity to get its morph
            from eden.entities.rigid import RigidEntity

            temp_entity = RigidEntity(self._env, entity_options)

            if not isinstance(temp_entity.morph, _SUPPORTED_MORPH_TYPES):
                raise ValueError(
                    f"Only Primitive, Mesh, URDF, and MJCF morphs are supported for grouped entity, "
                    f"got {type(temp_entity.morph).__name__}."
                )

            variants_pos.append(temp_entity.default_root_pos.squeeze(0))
            variants_quat.append(temp_entity.default_root_quat.squeeze(0))

            # Override morph fixed property if needed (MJCF does not support fixed)
            if not isinstance(temp_entity.morph, gs.morphs.MJCF):
                if hasattr(temp_entity.morph, "fixed"):
                    temp_entity.morph.fixed = self._options.is_fixed_base

            morphs_heterogeneous.append(temp_entity.morph)

        self.morph = morphs_heterogeneous

        # Genesis hands the variants to the envs through this one mapping (per-variant init qpos,
        # geom ranges, pose offsets), so reuse it rather than inferring the mapping from geoms — a
        # variant can own several collision geoms, or none. ``scene.build`` gets ``n_envs=num_envs``.
        env_variant_np = _balanced_variant_mapping(len(morphs_heterogeneous), self._env.num_envs)
        env_variant = torch.as_tensor(env_variant_np, device=self.device)
        self._variant_default_root_pos = torch.stack(variants_pos)[env_variant]
        self._variant_default_root_quat = torch.stack(variants_quat)[env_variant]
        self._env_variant_idx = env_variant.to(torch.long)
        # The present variants are 0..k-1 in block order, so each one's first env is where its value first changes.
        self._variant_first_envs = np.unique(env_variant_np, return_index=True)[1].tolist()
        if self._env.num_envs < len(morphs_heterogeneous):
            en.logger.warning(
                f"Heterogeneous entity has {len(morphs_heterogeneous)} variants but the scene only {self._env.num_envs} "
                f"envs, so variants {list(range(self._env.num_envs, len(morphs_heterogeneous)))} are in no env."
            )

    @property
    def num_variants(self) -> int:
        """How many variants a heterogeneous (``GroupedEntityOptions``) entity was built from."""
        self._require_heterogeneous("num_variants")
        return len(self._options.grouped_entities)

    @property
    def env_variant_idx(self) -> torch.Tensor:
        """The variant each env holds, ``(num_envs,)`` long, indexing ``GroupedEntityOptions.grouped_entities``.

        Fixed at build: a reset re-poses an env's variant but never swaps it. This is the mapping Genesis itself
        distributes the variants by (contiguous, balanced env blocks), not an inference from the built geometry.
        """
        self._require_heterogeneous("env_variant_idx")
        return self._env_variant_idx

    def _require_heterogeneous(self, what: str) -> None:
        if not self.is_heterogeneous:
            raise TypeError(
                f"`{what}` is only defined for a heterogeneous entity (built from GroupedEntityOptions), but this "
                f"entity was built from {type(self._options).__name__}, so every env holds the same body."
            )

    def _resolve_variant_fields(self) -> None:
        """Resolve a heterogeneous entity's per-variant and shared fields over its mirrored group options.

        A shared DOF field is mirrored as the one value every variant takes. A per-variant field the present
        variants all resolve alike is mirrored as that value, so the homogeneous code path applies it; one they
        differ in is kept in ``_variant_fields``, a value per present variant, for ``RigidEntity.post_build`` to
        apply to each env's variant. Deep-copied, as the mirror does.
        """
        # The options checked these at construction; an edit made since (a TaskMod assigning a field) was not.
        self._options.check_variants()
        n_present = len(self._variant_first_envs)
        for name, value in self._options.shared_dof_fields().items():
            setattr(self, name, copy.deepcopy(value))
        for name, values in self._options.variant_values().items():
            values = values[:n_present]
            if all(value == values[0] for value in values[1:]):
                setattr(self, name, copy.deepcopy(values[0]))
            else:
                self._variant_fields[name] = copy.deepcopy(values)
        self._variant_materials = self._options.variant_materials

    def get_default_root_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the default root pose for all environments, ``(num_envs, 3)`` / ``(num_envs, 4)``.

        For a heterogeneous entity, each env's variant default pose is composed onto the
        GroupedEntityOptions default root pose as an additional offset.
        """
        default_pos = self.default_root_pos.repeat(self._env.num_envs, 1)
        default_quat = self.default_root_quat.repeat(self._env.num_envs, 1)
        if not self.is_heterogeneous:
            return default_pos, default_quat
        return (
            default_pos + transform_by_quat(self._variant_default_root_pos, default_quat),
            gu.transform_quat_by_quat(self._variant_default_root_quat, default_quat),
        )

    def pre_build(self) -> None:
        """Add the entity to the Genesis scene before the scene is built.

        Genesis' asset-level warnings repeat once per mesh (a G1 has nine non-watertight
        collision meshes), so the load collapses identical ones into a single counted line.
        """
        # Genesis falls back to an auto-generated name when the Eden scene key is absent
        # (entities constructed outside EnvBase).
        name = getattr(self, "_name", None)
        subject = f"entity {name!r}" if name is not None else f"a {type(self).__name__}"
        with collapse_repeated_genesis_warnings(subject):
            material = self._parse_material()
            if isinstance(material, mochi_materials.Base):
                # Genesis ships no MochiSolver: Eden hosts one (eden.engine.solvers.mochi), installed on the scene's
                # first Mochi entity, after which the scene's `add_entity` takes Mochi materials.
                self._env._install_mochi_solver(subject)
            self._entity: GenesisEntity = self._env.scene.add_entity(
                self.morph,
                material=material,
                surface=self._parse_surface(),
                visualize_contact=self.visualize_contact,
                vis_mode=self.vis_mode,
                name=name,
            )

    def _parse_surface(self) -> Surface | None:
        """Resolve the configured surface to a Genesis surface (after ``gs.init()``).

        Deferred :class:`SurfaceOptions` are materialized here. If no surface is set
        but the ``color`` shortcut is, a plastic surface with that color is created.
        """
        if self.surface is None and getattr(self._options, "color", None) is not None:
            self.surface = PlasticSurfaceOptions(color=self._options.color)
        return resolve_surface(self.surface)

    def post_build(self) -> None:
        """Finalize the entity after the scene is built (no-op for the base class)."""
        pass

    def _parse_material(self) -> Material:
        if self.material is None:
            self.material = en.materials.RigidMaterialOptions()
        return self.material.to_genesis_material()

    @property
    def name(self) -> str:
        """The entity's registered name."""
        return getattr(self, "_name", self._options.__class__.__name__)

    @property
    def entity(self):
        """The underlying Genesis entity this wrapper delegates to."""
        return self._entity

    @property
    def metadata(self):
        """User-defined metadata attached to the entity's options."""
        return self._options.metadata

    @property
    def solver(self):
        """The Genesis solver that owns this entity."""
        return self._entity._solver

    @property
    def env(self):
        """The environment this entity belongs to."""
        return self._env

    @property
    def idx(self):
        """The entity's index within its solver."""
        return self._entity.idx

    @property
    def uid(self):
        """The entity's globally unique identifier."""
        return self._entity.uid

    @property
    def is_built(self) -> bool:
        """Whether the owning scene has been built."""
        return self._env.is_built

    @property
    def device(self) -> torch.device:
        """The torch device the entity's tensors live on."""
        return self._env.device

    @property
    def is_attached(self) -> bool:
        """Whether this entity is welded onto a link of another entity.

        Genesis ``attach`` merges the pair into one kinematic tree and *removes* the
        child's root joint, so an attached entity has no base DOFs of its own even when
        ``is_fixed_base`` is False — see :attr:`has_immobile_base`. (Mirrors the
        ``is_attached`` property on ``Camera`` / ``Sensor``.)
        """
        return self._is_attached

    @property
    def attachment_parent(self) -> Entity | None:
        """Entity whose link carries this welded entity, or None when it is not attached."""
        return self._attachment_parent

    @property
    def has_immobile_base(self) -> bool:
        """Whether the entity's base link carries no DOFs of its own.

        True for a fixed-base entity and for an attached one (its base link is carried by
        the parent link). Prefer this over ``is_fixed_base`` wherever the question is "are
        there base DOFs to read or write", since ``is_fixed_base`` alone reports False for
        a welded gripper / dexterous hand.
        """
        return self.is_fixed_base or self.is_attached

    @property
    def fixed(self) -> bool:
        """Whether the entity has a fixed base (cannot move freely)."""
        return self.is_fixed_base

    # ------------------------------------------------------------------------------------
    # ------------------------------------- universal ------------------------------------
    # ------------------------------------------------------------------------------------

    def _query_contacts(self, query: Callable[[bool], dict[str, Any]], is_padded: bool) -> dict[str, Any]:
        """Run a contact query, ``query(is_padded)``, in the form ``is_padded`` asks for or the env requires.

        The padded form (every contact slot of each env, ``valid_mask`` marking the live ones) is taken when
        asked, and also whenever the env requires it (:attr:`~eden.envs.base.EnvBase.fixed_shape_queries`), as it
        does while the graded phase runs under a CUDA graph (``EnvOptions.cuda_graph_options``): the trimmed form
        reads the contact count back to the host and is sized by it, which a capture cannot record, and the eager
        runs that validate the graph must reduce over the same slots as the graph does, or a float sum over
        contacts can differ in its last bits. A padded result's invalid slots hold whatever an
        earlier step left there (a NaN force from a diverged env included), so their floating-point fields are
        zeroed: ``force * valid_mask`` and ``where(valid_mask, force, 0)`` then agree.
        """
        padded = is_padded or self._env.fixed_shape_queries
        contacts = query(padded)
        if not padded:
            return contacts
        valid = contacts["valid_mask"]
        return {
            key: (
                torch.where(valid.reshape(*valid.shape, *(1,) * (value.dim() - valid.dim())), value, 0.0)
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value
            )
            for key, value in contacts.items()
        }

    # NOTE: Eden's pose *and* velocity reads default to the solver frame (relative=False) instead of Genesis'
    # relative=True default. `relative=True` reports the authored link origin, which on a free
    # body Genesis re-origined onto its inertial frame is a different point; a velocity read there would not be the
    # time derivative of the position `get_pos` returns. Concrete entity classes own their pose-setter frame semantics.

    def _reject_relative(self, relative: bool, getter: str) -> None:
        """Refuse ``relative=True`` on an entity kind that has only one base frame.

        ``relative`` distinguishes Genesis' solver frame from the frame the asset was authored in,
        and the two differ only because Genesis re-origins a *free rigid body* onto its inertial
        frame (see :meth:`~eden.entities.rigid.RigidEntity.get_pos`). Nothing else has that offset:
        an FEM centroid and a particle cloud each report one frame and no other, so returning it
        for ``relative=True`` would answer a question this entity cannot answer.

        The argument still belongs on the shared signature, because the built-in pose terms
        (``observations.base_pos`` / ``base_quat``, ``metrics.object_above_height``,
        ``terminations.object_in_region``) are written against ``Entity`` and pass it through
        unconditionally — a getter that only ``RigidEntity`` accepted would narrow those four terms
        to rigid bodies and fail on a deformable at the ``False`` default. So the argument is
        accepted and only the value that would lie is refused, the way
        :meth:`~eden.entities.fem.FEMEntity.get_vel` refuses ``frame='body'``.
        """
        if relative:
            raise ValueError(
                f"{type(self).__name__}.{getter}(relative=True) is not meaningful: only a free rigid body is "
                f"re-origined onto its inertial frame, so this entity has one base frame and reports it at "
                f"relative=False."
            )

    def get_pos(self, envs_idx=None, *, relative: bool = False):
        """Return the position of the entity's base link in world frame.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            Must be False for this entity kind (see :meth:`_reject_relative`); only
            :class:`~eden.entities.rigid.RigidEntity` has a second frame to report.

        Returns
        -------
        pos : torch.Tensor, shape (3,) or (n_envs, 3)
            The base-link position.
        """
        self._reject_relative(relative, "get_pos")
        return self._entity.get_pos(envs_idx=envs_idx, relative=False)

    def get_quat(self, envs_idx=None, *, relative: bool = False):
        """Return the quaternion of the entity's base link in world frame.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The environment indices. If None, all environments are considered. Defaults to None.
        relative : bool, optional
            Must be False for this entity kind (see :meth:`_reject_relative`); only
            :class:`~eden.entities.rigid.RigidEntity` has a second frame to report.

        Returns
        -------
        quat : torch.Tensor, shape (4,) or (n_envs, 4)
            The base-link quaternion in ``(w, x, y, z)`` format.
        """
        self._reject_relative(relative, "get_quat")
        return self._entity.get_quat(envs_idx=envs_idx, relative=False)

    def get_heading(self, envs_idx=None):
        """Return the heading of the entity's base link in world frame."""
        return heading_from_quat(self.get_quat(envs_idx=envs_idx), self.forward_vec)

    def get_T(self, envs_idx=None):
        """Return the entity's base-link pose as a 4x4 homogeneous transform matrix.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are considered. Defaults to None.

        Returns
        -------
        transform : torch.Tensor, shape (4, 4) or (n_envs, 4, 4)
            The base-link homogeneous transform.
        """
        pos = self.get_pos(envs_idx=envs_idx)
        quat = self.get_quat(envs_idx=envs_idx)
        return gu.trans_quat_to_T(pos, quat)

    def get_vel(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Return the linear velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        frame : Literal["world", "body"], optional
            Coordinate frame for the returned linear velocity. Valid options:
            - "world": linear velocity expressed in world coordinates (default)
            - "body": linear velocity expressed in the body (base link) frame
        """
        vel_w = self._entity.get_vel(envs_idx=envs_idx, relative=False)
        if frame == "world":
            return vel_w
        elif frame == "body":
            quat_wb = self.get_quat(envs_idx=envs_idx)
            return inv_transform_by_quat(vel_w, quat_wb)
        else:
            raise ValueError(f"Invalid frame '{frame}'. Expected 'world' or 'body'.")

    def get_ang(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Return the angular velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        frame : Literal["world", "body"], optional
            Coordinate frame for the returned angular velocity. Valid options:
            - "world": angular velocity expressed in world coordinates (default)
            - "body": angular velocity expressed in the body (base link) frame
        """
        ang_w = self._entity.get_ang(envs_idx=envs_idx)

        if frame == "world":
            return ang_w
        elif frame == "body":
            quat_wb = self.get_quat(envs_idx=envs_idx)
            return inv_transform_by_quat(ang_w, quat_wb)
        else:
            raise ValueError(f"Invalid frame '{frame}'. Expected 'world' or 'body'.")

    def set_pos(self, pos, envs_idx=None) -> None:
        """Require concrete entity classes to define position-setter semantics."""
        raise NotImplementedError("set_pos must be implemented by a concrete Entity subclass.")

    def set_quat(self, quat, envs_idx=None) -> None:
        """Require concrete entity classes to define orientation-setter semantics."""
        raise NotImplementedError("set_quat must be implemented by a concrete Entity subclass.")

    def attach_to(
        self,
        entity: Entity,
        link_name: str,
        attach_offset: Vec3FType | None = None,
        attach_quat: UnitVec4FType | None = None,
    ) -> None:
        """Reject attachment for entity kinds that cannot be welded to a parent link.

        Overridden by :class:`~eden.entities.rigid.RigidEntity` (a weld) and
        :class:`~eden.entities.mochi.MochiSoftEntity` (a soft body tied to the link). Declared
        here — taking the mounting-transform kwargs that ``EnvBase`` always passes — so
        ``attachments_dict`` naming a deformable/particle child fails with an explanatory error
        instead of a bare ``AttributeError``/``TypeError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} '{self.name}' does not support attach_to (requested parent "
            f"'{entity.name}', link '{link_name}'). Only RigidEntity supports parent-child "
            "attachments (and MochiSoftEntity a mount on a Mochi link); remove this entity from `attachments_dict`."
        )

    def get_mass(self):
        """Return the total mass of the entity, in kg.

        For rigid entities this sums the links that are not fixed to the world, and returns
        a 0-dim tensor — or a ``(num_envs,)`` tensor when link info is batched
        (``EnvOptions.batch_links_info`` or a heterogeneous entity).
        """
        return self._entity.get_mass()

    def set_mass(self, mass):
        """Set the total mass of the entity.

        Parameters
        ----------
        mass : float
            The target total mass.
        """
        self._entity.set_mass(mass)
