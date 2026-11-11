"""Lift YCB from RGB (``manipulation_lift_ycb_rgb``): the RGB lift-box task over a bank of YCB objects.

Extends :class:`ManipulationLiftBoxRGBConfig` (wrist-RGB vision, spatial-softmax CNN policy,
asymmetric actor/critic) — the delta is the *object*: instead of one red box in
every environment, each env holds a different rigid YCB mesh (a can, a box, a
brick, a piece of fruit, ...), so the policy must learn to reach → grasp → lift
objects of varying shape, size, and appearance from sight alone.

How heterogeneity works
-----------------------
Genesis builds one scene shared across all vectorized envs, so the batch
dimension is *state* (pose/DOFs), never geometry — you cannot naively hand a
different mesh to each env. Eden's :class:`GroupedEntityOptions` is the
first-class mechanism: it registers a *list* of morphs under a single entity
name, and Genesis distributes the variants in contiguous blocks across the ``num_envs``
batch at build time (with ``num_envs=1024`` and 8 variants, 128 envs per
object). The variant an env holds is **fixed at build**; resets only re-pose the
object within its env, never swap its mesh.

Crucially the grouped entity is kept named ``"cube"`` — exactly the name the
inherited :class:`ManipulationLiftBoxConfig` terms/events/terminations bind to. Every one of
them reads ``env.entities["cube"].get_pos()`` (and friends), which returns a
per-env tensor spanning all variants, so reaching, grasping, goal-tracking, the
reset placement, the contact stiffening, and the fall termination all work over
the heterogeneous bank unchanged.

Per-env resting height
----------------------
The one thing a single scalar can't express across heterogeneous objects is the
lift gate. ``manipulation_lift_box`` uses ``rest_z = table_top + cube_half``; here each object
rests at a different origin height (a tall can's origin already clears a fixed
margin at rest, so a scalar gate would be farmable). :class:`ObjectRestHeight`
latches each env's *actual* resting z during the post-reset settle and exposes it
as a metric; the four rest_z-dependent lift rewards (``lift_height``,
``lifting_object``, ``object_goal_tracking`` and its fine copy) are retargeted to
that per-env metric via ``rest_z_metric`` (see
:func:`eden.tasks.manipulation.custom_terms.rewards._resolve_rest_z`).

Vision note: unlike the gray box (invisible on a gray table, hence the RGB lift-box task's
red recolor), YCB ``textured.obj`` meshes carry their own textures and are
visible to the madrona camera as-is — no recolor needed. The parent's darkened
table (for contrast) and workspace lights are kept.

Known rough edges:
* Spawn height is a single conservative float above the table; each object drops
  a few cm and settles under gravity, and ``ObjectRestHeight`` freezes each env's
  rest height the moment its object comes to rest (per-env velocity debounce, not a
  blind step count). ``_SPAWN_Z`` and ``ObjectRestHeight``'s ``settle_speed`` /
  ``settle_hold`` / ``max_settle_steps`` are the knobs if an object bounces or rolls
  before settling. A proper support-surface drop (``place_on_to``) is the eventual
  fix but crashes on partial resets today (see ``manipulation_lift_box``).
* The object set is curated to fit the Robotiq 2F85 (~85 mm) opening and stay
  roughly cube-scale; spheres (apple) are the hardest grasp.
* ``--object_properties ycb`` (:class:`LiftYCBObjectPropertiesMod`) gives each object its own
  mass and friction instead of the bank's one material: the published YCB mass (scaled
  with the mesh) and a friction by surface material. The default, ``uniform``, keeps one
  density and friction 2.0 for all eight.
* The task has **no success metric** yet. ``ObjectRestHeight`` is a shared-quantity
  vehicle and declares ``contributes_to_success=False``, so no ``Metrics/Success`` is
  logged at all — rather than a number that reads as a real success rate.
"""

from __future__ import annotations

from typing import Annotated, Literal

import torch

from eden.constants import MetricMode
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.options.entities import EntityOptions, GroupedEntityOptions
from eden.options.managers.metrics import MetricManagerOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.objects.ycb import (
    YCB002MasterChefCan,
    YCB005TomatoSoupCan,
    YCB007TunaFishCan,
    YCB008PuddingBox,
    YCB009GelatinBox,
    YCB010PottedMeatCan,
    YCB013Apple,
    YCB061FoamBrick,
)
from eden.tasks.manipulation import custom_terms
from eden.tasks.manipulation.lift_box.config import _CUBE_HOME, _GRASP, _lift_rewards, _TABLE_TOP
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.tasks.manipulation.lift_box_rgb.config import (
    _rgb_lift_scene,
    _rgb_runner_options,
    ManipulationLiftBoxRGBConfig,
)
from eden.tasks.registry import TASK_REGISTRY, TaskMod
from eden.terms import RewardTerm
from eden.utils.configs import RewardManagerOptions, SceneOptions

# Curated graspable set: cans / boxes / a brick / fruit that fit the Robotiq
# 2F85 (~85 mm) opening and stay near the 5 cm cube's scale. Distinct textures
# give the vision policy real appearance variety.
_YCB_VARIANTS = (
    YCB002MasterChefCan,
    YCB005TomatoSoupCan,
    YCB007TunaFishCan,
    YCB008PuddingBox,
    YCB009GelatinBox,
    YCB010PottedMeatCan,
    YCB061FoamBrick,
    YCB013Apple,
)

# Object spawn: reuse the cube's tabletop xy home, but float the origin a bit
# above the table so no variant interpenetrates it regardless of where each
# mesh's origin sits relative to its base — physics then settles each object
# onto the surface and ObjectRestHeight latches the per-env resting z.
#
# The spawn pose is set on the *group*; a per-variant ``default_root_pos`` composes onto it as an
# offset (``Entity.get_default_root_pose``), which none of the variants needs.
_OBJ_X = _CUBE_HOME[0]
_SPAWN_Z = _TABLE_TOP + 0.06
_SPAWN_POS = (_OBJ_X, 0.0, _SPAWN_Z)

_REST_METRIC = "object_rest_height"
# Lift gate params (mirrors _LIFT_GATE in lift_box/config.py) but sourcing a per-env
# resting height instead of the box's single scalar rest_z.
_LIFT_GATE_YCB = {
    "object_name": "cube",
    "rest_z_metric": _REST_METRIC,
    "lift_margin": 0.02,
    "grasp_threshold": 0.06,
}


# Each object's own physical properties for ``--object_properties ycb``: (published mass in g, from Table II of
# Calli et al., "The YCB Object and Model Set and Benchmarking Protocols" (arXiv:1502.03143); the volume in cm^3
# Genesis builds the class's convexified collision mesh at, at its ``scale``; sliding friction).
#
# The density reproduces the real object's at the mesh's scale, so a shrunken can keeps a can's density and weighs
# ``mass * scale**3`` (the master chef can at 0.7 comes out at 142 g, not 414 g). The volume was measured by building
# the bank at a known density (``get_mass() / rho``); for the master chef can it matches the real can's
# 102 x 139 mm cylinder to 0.5 %.
#
# Friction combines as ``max`` with the partner geom, and this scene's partners are the Robotiq pads (1.2) and the
# table (2.0). An object coefficient therefore shows in the grasp only between 1.2 and 2.0, and never on the
# tabletop, so the values are spread over that band in the order of their real surfaces: smooth plastic fruit,
# label-wrapped cans, cardboard boxes, foam.
_YCB_PHYSICAL: dict[type[EntityOptions], tuple[float, float, float]] = {
    YCB002MasterChefCan: (414.0, 391.6, 1.4),
    YCB005TomatoSoupCan: (349.0, 357.3, 1.4),
    YCB007TunaFishCan: (171.0, 83.7, 1.4),
    YCB008PuddingBox: (187.0, 180.3, 1.6),
    YCB009GelatinBox: (97.0, 179.2, 1.6),
    YCB010PottedMeatCan: (370.0, 377.1, 1.4),
    YCB061FoamBrick: (28.0, 197.4, 2.0),
    YCB013Apple: (68.0, 251.1, 1.2),
}

_ObjectProperties = Literal["uniform", "ycb"]


def _ycb_material(cls: type[EntityOptions]) -> RigidMaterialOptions:
    """``cls``'s own material: its real density, and its friction (see ``_YCB_PHYSICAL``).

    The density is the real object's, ``mass / real volume``, and the real volume is the measured one at the
    class's ``scale`` divided by ``scale**3``. It is the same at any instance ``scale``: the built mass scales with
    the mesh, as a smaller can of the same stuff would.
    """
    mass_g, volume_cm3, friction = _YCB_PHYSICAL[cls]
    scale = cls.model_fields["scale"].default
    return RigidMaterialOptions(rho=1000.0 * mass_g * scale**3 / volume_cm3, friction=friction)


def _ycb_object_bank() -> GroupedEntityOptions:
    """Heterogeneous ``cube`` entity: one YCB mesh per env, in contiguous blocks of envs."""
    return GroupedEntityOptions(
        grouped_entities=[cls(material=RigidMaterialOptions(friction=2.0)) for cls in _YCB_VARIANTS],
        default_root_pos=_SPAWN_POS,
        is_fixed_base=False,
    )


class LiftYCBObjectPropertiesMod(TaskMod):
    """Give each YCB object its own mass and friction, or one material for the whole bank."""

    def __init__(
        self,
        object_properties: Annotated[
            _ObjectProperties,
            "uniform: every object at the default density and friction 2.0. ycb: each at its published YCB mass "
            "(scaled with the mesh) and a friction by surface material.",
        ] = "uniform",
    ):
        self.object_properties = object_properties

    def apply(self, config):
        if self.object_properties == "uniform":
            return config
        bank = config.scene_options.cube
        unknown = [
            f"grouped_entities[{i}] ({type(variant).__name__})"
            for i, variant in enumerate(bank.grouped_entities)
            if type(variant) not in _YCB_PHYSICAL
        ]
        if unknown:
            raise ValueError(
                f"LiftYCBObjectPropertiesMod(object_properties={self.object_properties!r}) has no physical properties "
                f"for scene_options.cube's {', '.join(unknown)}; it knows "
                f"{sorted(cls.__name__ for cls in _YCB_PHYSICAL)}. Fix: add the class to _YCB_PHYSICAL in "
                f"{__name__} (its YCB Table II mass, the volume Genesis builds it at, a friction), or run with "
                f"--object_properties uniform. Skipping the variant would train it at the bank's material "
                f"while the run reads as per-object."
            )
        # Each variant edited in place of the bank, so whatever set the bank (a subclass's object set, its pose)
        # is kept. Rebuilt as a new group rather than assigned, so the group's checks run on the new materials.
        variants = [variant.updated(material=_ycb_material(type(variant))) for variant in bank.grouped_entities]
        kept = {name: getattr(bank, name) for name in bank.model_fields_set if name != "grouped_entities"}
        config.scene_options.cube = type(bank)(**kept, grouped_entities=variants)
        return config


@METRIC_TERM_REGISTRY.register()
class ObjectRestHeight(MetricTerm):
    """Per-env resting z of the grouped object, latched once the object comes to rest.

    Each env holds a differently-shaped YCB mesh (see :func:`_ycb_object_bank`),
    so each rests at a different origin height and a single scalar ``rest_z``
    can't gate the lift. After reset the object drops a few cm and settles; this
    tracks each env's object z while it is still moving and **freezes it the moment
    the object comes to rest** — detected per env, not on a blind step count, so a
    variant that bounces or takes longer to settle latches its *actual* resting
    height rather than a mid-fall one. The lift rewards read it via
    ``metric_manager.get_value``.

    Rest is a debounced low-speed test: an object must stay below ``settle_speed``
    for ``settle_hold`` consecutive control steps (a lone still frame at a bounce
    apex resets the counter, so it can't latch mid-air). ``max_settle_steps`` is a
    hard cap — an object that never fully settles (e.g. the apple rolling on the
    table, whose z is stable anyway) latches its current, on-surface height then.
    All three are tuning knobs.

    It is a vehicle for a shared quantity, not an evaluation signal, hence
    ``contributes_to_success=False``: a resting height trivially clears any threshold, so
    letting it into success accounting would mark every episode — and every episode the
    recorder labels — successful. With no success-contributing metric configured, the manager
    reports no ``Metrics/Success`` at all rather than a meaningless 0 or 1.
    """

    object_name: str = "cube"
    settle_speed: float = 0.02  # m/s; at or below this the object counts as momentarily still
    settle_hold: int = 3  # consecutive still control steps before the rest height is latched
    max_settle_steps: int = 30  # hard cap: latch the current height even if the object never settles
    metric_mode: MetricMode = MetricMode.INTERVAL
    contributes_to_success: bool = False

    def build(self) -> None:
        # Touch the entity once so a bad object_name fails at env construction rather than on
        # the first step; compute() re-resolves the handle (deployment hot-swaps env.entities,
        # see ManagerTermBase.resolve_entity), so only derived buffers are cached here.
        self.resolve_entity(self.object_name)
        self._rest_z = torch.zeros(self.num_envs, device=self.device)
        self._age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._still = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._age[envs_idx] = 0
        self._still[envs_idx] = 0
        self._latched[envs_idx] = False

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        self._age += 1
        obj = self.resolve_entity(self.object_name)
        obj_z = obj.get_pos()[:, 2]
        speed = torch.linalg.vector_norm(obj.get_vel(), dim=-1)
        # Debounced rest detection: count consecutive still steps, resetting on any motion,
        # so a single still frame (a bounce apex) can't trip the latch mid-air.
        self._still = torch.where(speed <= self.settle_speed, self._still + 1, torch.zeros_like(self._still))
        # Track the live height until the object rests (or the safety cap fires), then freeze it.
        tracking = ~self._latched
        self._rest_z = torch.where(tracking, obj_z, self._rest_z)
        self._latched |= (self._still >= self.settle_hold) | (self._age >= self.max_settle_steps)
        return self._rest_z


def _ycb_lift_scene() -> SceneOptions:
    """The RGB lift scene with the single red box swapped for the heterogeneous YCB bank.

    Built on the parent's scene factory, so the darkened table, workspace rig, and goal
    marker are inherited and only ``cube`` changes. The bank replaces the parent's *red* box
    outright: YCB ``textured.obj`` meshes carry their own textures, and a ``color`` on the
    group would tint all eight of them flat red.
    """
    scene = _rgb_lift_scene()
    scene.cube = _ycb_object_bank()
    return scene


# Retarget the four rest_z-dependent lift rewards to the per-env resting height. Weights,
# std, and deadband are unchanged from the lift-box task; only the rest_z source differs
# (single scalar -> per-env metric).
_YCB_LIFT_REWARDS = dict(
    lift_height=RewardTerm.configure(
        func=custom_terms.lift_height,
        weight=15.0,
        params={"object_name": "cube", "rest_z_metric": _REST_METRIC, "cap": 0.3},
    ),
    lifting_object=RewardTerm.configure(
        func=custom_terms.object_is_lifted,
        weight=5.0,
        params={**_GRASP, **_LIFT_GATE_YCB},
    ),
    object_goal_tracking=RewardTerm.configure(
        func=custom_terms.object_goal_distance,
        weight=25.0,
        params={
            "robot_name": "robot",
            **_GRASP,
            **_LIFT_GATE_YCB,
            "command_name": "lift_target",
            "std": 0.3,
        },
    ),
    object_goal_tracking_fine=RewardTerm.configure(
        func=custom_terms.object_goal_distance,
        weight=10.0,
        params={
            "robot_name": "robot",
            **_GRASP,
            **_LIFT_GATE_YCB,
            "command_name": "lift_target",
            "std": 0.05,
            "deadband": 0.02,
        },
    ),
)


@TASK_REGISTRY.register(name="manipulation_lift_ycb_rgb", modifiers=(LiftYCBObjectPropertiesMod,))
class ManipulationLiftYCBRGBConfig(ManipulationLiftBoxRGBConfig):
    """Every field is a declarative default — nothing is built in ``__init__``.

    The parent's wrist RGB camera, madrona lights, CNN actor/critic, and ``num_envs`` are
    inherited untouched; the deltas are the object bank, the per-env resting-height metric,
    the four lift rewards retargeted to it, and its own experiment name.
    """

    scene_options: SceneOptions = _ycb_lift_scene()

    metric_options: MetricManagerOptions = MetricManagerOptions(
        object_rest_height=ObjectRestHeight.configure(object_name="cube"),
    )

    reward_options: RewardManagerOptions = _lift_rewards(**_YCB_LIFT_REWARDS)

    # Own experiment name: the observation content and the object set differ from the parent's,
    # so sharing its ``logs/lift_box_rgb`` tree would interleave two tasks' checkpoints.
    runner_options: RslRlOnPolicyRunnerOptions = _rgb_runner_options(experiment_name="lift_ycb_rgb")
