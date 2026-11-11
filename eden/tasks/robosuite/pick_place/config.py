"""robosuite ``PickPlace``: move four grocery items from the source bin into their quadrants of the target bin.

Port of ``robosuite/environments/manipulation/pick_place.py`` on the ``BinsArena``. Two fixed bins
stand side by side (slab top 0.82 m, 5 cm walls): ``bin1`` at ``(0.1, -0.25)`` holds the milk carton,
bread, cereal box and soda can; ``bin2`` at ``(0.1, 0.28)`` is split by two divider walls into four
quadrants, one per object (milk -x/-y, bread +x/-y, cereal -x/+y, can +x/+y). The Panda stands on its
pedestal at ``(-0.5, -0.1)``. The mesh objects are robosuite's own MJCFs (``objects/milk.xml`` ...).

Reward, per object, from robosuite's ``staged_rewards`` multipliers (robosuite ``max``-es the stages
over the not-yet-placed objects; here every object carries its own set and the terms are summed):

* ``reach_<obj>``  — ``0.1 (1 - tanh(10 d))`` from the grasp point to the object.
* ``grasp_<obj>``  — 0.35 while both fingerpads touch it (``_check_grasp``).
* ``lift_<obj>``   — 0.5 once the object is 10 cm above its resting height, which clears the bin walls
  for every object (robosuite: a tanh ramp towards ``bin2_z + 0.25``).
* ``hover_<obj>``  — ``0.7 [lifted] (1 - tanh(10 d_xy))`` to the centre of its target quadrant.
* ``in_bin_<obj>`` — 1.0 for robosuite's ``objects_in_bins[i]``: root inside the quadrant box (z from
  0.8 up to the object's ``lift_<obj>`` threshold, see :func:`target_region`) **and released**.
  robosuite tests release as ``r_reach < 0.6`` (grasp point more than 4.24 cm away); here it is
  ``NOT PadsGrasping``, which is the same intent (the object is out of the fingers) without a distance
  the gripper could satisfy while still pinching the object.

Success is every ``in_bin_<obj>`` at once (robosuite ``_check_success``). ``--object <name>`` is
robosuite's ``PickPlace<Name>`` (``single_object_mode=2``): the other three objects and their terms
are removed and success is that object's ``in_bin``. robosuite's ``PickPlaceSingle`` (a random object
per reset) has no counterpart — a per-reset entity swap is not expressible in a config.

Deviations from robosuite. Placement: robosuite samples all four objects in one region of bin1 with
rejection of overlaps; Eden has no rejection sampler, so each object spawns in its own quadrant of
bin1 (mirroring its target quadrant), inset by its horizontal radius, and is dropped 1 cm above the
slab rather than 2 cm inside it. Object rest heights are measured from the meshes (robosuite's
declared ``bottom_site`` offsets are 1.5–2.5 cm below the true mesh bottoms). The transparent
"visual fiducial" copies robosuite renders at the target quadrants are not reproduced. The episode
is :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` long rather than 50 s, and an object falling
to the floor ends it early.
"""

from __future__ import annotations

from typing import Annotated, Literal

from eden import EventMode, events, metrics, observations, terminations
from eden.options.camera import CamerasOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.managers.events import EventTermOptions
from eden.options.managers.metrics import MetricTermOptions
from eden.options.managers.rewards import RewardTermOptions
from eden.options.managers.terminations import TerminationTermOptions
from eden.tasks.registry import TASK_REGISTRY, TaskMod
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite.common import (
    agentview_camera,
    arena_entities,
    arm_actions,
    arm_init_noise,
    fallen_termination,
    merged,
    object_observations,
    policy_and_critic,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    ROBOT_BASE_Z,
)
from eden.tasks.robosuite.staged_objects import staged_object_metrics, staged_object_rewards
from eden.tasks.robosuite.variants import drop_object
from eden.options.objects.robosuite import (
    RobosuiteBread,
    RobosuiteCan,
    RobosuiteCereal,
    RobosuiteMilk,
    RobosuiteObjectOptions,
    RobosuitePickPlaceBin1,
    RobosuitePickPlaceBin2,
)
from eden.terms import DoneTerm, EventTerm, MetricTerm, ObsTerm
from eden.utils.configs import (
    ActionManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

# --- arena (robosuite BinsArena) -------------------------------------------------------------------

#: Panda ``base_xpos_offset["bins"]``.
_BASE_POS = (-0.5, -0.1, ROBOT_BASE_Z)
_RIG = panda_rig("robot", _BASE_POS)

_BIN1_POS = (0.1, -0.25, 0.8)
_BIN2_POS = (0.1, 0.28, 0.8)
BIN_TOP_Z = 0.82  # slab top: body z + slab half-height 0.02
#: robosuite ``bin_size = (0.39, 0.49)``: the nominal footprint the quadrants are cut from.
_BIN_SIZE = (0.39, 0.49)
_QUADRANT_HALF = (_BIN_SIZE[0] / 4, _BIN_SIZE[1] / 4)  # (0.0975, 0.1225)
#: robosuite's placement range inside bin1 (``bin_size / 2 - 0.05``), before the per-object radius inset.
_PLACE_HALF = (0.145, 0.195)
_WALL_CLEARANCE = 0.01
_OBJECT_OPTIONS: dict[str, type[RobosuiteObjectOptions]] = {
    "milk": RobosuiteMilk,
    "bread": RobosuiteBread,
    "cereal": RobosuiteCereal,
    "can": RobosuiteCan,
}

# --- objects (robosuite MujocoXMLObject set) -------------------------------------------------------

#: robosuite object id order, which fixes the quadrant: (-x, -y), (+x, -y), (-x, +y), (+x, +y).
OBJECT_NAMES: tuple[str, ...] = ("milk", "bread", "cereal", "can")
_QUADRANT_SIGN = {"milk": (-1, -1), "bread": (1, -1), "cereal": (-1, 1), "can": (1, 1)}
#: Mesh bottom below the root body (measured from the STL at the XML mesh scale) and robosuite's
#: ``horizontal_radius``. Rest height on a slab = slab top + bottom.
_OBJECT_BOTTOM = {"milk": 0.061, "bread": 0.0233, "cereal": 0.075, "can": 0.0403}
_OBJECT_RADIUS = {"milk": 0.025, "bread": 0.03, "cereal": 0.04, "can": 0.025}
_DROP_HEIGHT = 0.01
_LIFT_ABOVE_REST = 0.10


def _rest_z(name: str) -> float:
    return BIN_TOP_Z + _OBJECT_BOTTOM[name]


def _lift_height(name: str) -> float:
    return _rest_z(name) + _LIFT_ABOVE_REST


def target_region(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The in-bin box robosuite tests for the object: its bin2 quadrant in xy, bin floor to lift height in z.

    robosuite writes the ceiling as the absolute ``bin2_z + 0.1``, which works there because its
    ``bottom_site`` placement puts every object's body origin at the bin surface. Here the origin sits
    :data:`_OBJECT_BOTTOM` *above* the surface (measured from the mesh), so a shared absolute ceiling
    leaves the tallest object — the cereal box, resting at 0.895 — barely 5 mm of headroom, and a box
    that settles on a neighbour or a divider lip reads as not-in-bin with no way to recover. Tracking
    the object's own rest height keeps robosuite's 10 cm of clearance for every object, and makes the
    ceiling exactly the ``lifted`` threshold: in the bin means in the quadrant and not lifted out of it.
    """
    sx, sy = _QUADRANT_SIGN[name]
    width, depth = 2 * _QUADRANT_HALF[0], 2 * _QUADRANT_HALF[1]
    x_low = _BIN2_POS[0] - width if sx < 0 else _BIN2_POS[0]
    y_low = _BIN2_POS[1] - depth if sy < 0 else _BIN2_POS[1]
    return (x_low, y_low, 0.8), (x_low + width, y_low + depth, _lift_height(name))


def target_xy(name: str) -> tuple[float, float]:
    """The quadrant centre (robosuite ``target_bin_placements[i]``)."""
    lo, hi = target_region(name)
    return ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)


def _spawn_span(name: str, axis: int) -> tuple[float, float]:
    """Centre range of the object along one axis, relative to bin1: its quadrant, inset by radius + clearance."""
    sign = _QUADRANT_SIGN[name][axis]
    radius = _OBJECT_RADIUS[name]
    inner = radius + _WALL_CLEARANCE  # clear of the quadrant boundary shared with the neighbour
    outer = min(2 * _QUADRANT_HALF[axis] - _WALL_CLEARANCE, _PLACE_HALF[axis]) - radius  # robosuite's outer bound
    low, high = sorted((sign * inner, sign * outer))
    return (low, high)


def _spawn_home(name: str) -> tuple[float, float, float]:
    xs, ys = _spawn_span(name, 0), _spawn_span(name, 1)
    return (_BIN1_POS[0] + (xs[0] + xs[1]) / 2, _BIN1_POS[1] + (ys[0] + ys[1]) / 2, _rest_z(name) + _DROP_HEIGHT)


def _spawn_jitter(name: str, axis: int) -> tuple[float, float]:
    lo, hi = _spawn_span(name, axis)
    half = (hi - lo) / 2
    return (-half, half)


def _object(name: str) -> RobosuiteObjectOptions:
    return _OBJECT_OPTIONS[name](default_root_pos=_spawn_home(name))


def _pick_place_scene() -> SceneOptions:
    return SceneOptions(
        robot=_RIG,
        **arena_entities(),
        mount=rethink_mount(_BASE_POS),
        bin1=RobosuitePickPlaceBin1(default_root_pos=_BIN1_POS),
        bin2=RobosuitePickPlaceBin2(default_root_pos=_BIN2_POS),
        **{name: _object(name) for name in OBJECT_NAMES},
    )


# --- per-object terms -------------------------------------------------------------------------------


def _object_metrics(name: str) -> dict[str, MetricTermOptions]:
    lo, hi = target_region(name)
    # robosuite objects_in_bins[i]: inside the quadrant box (and, via staged_object_metrics, released).
    return staged_object_metrics(
        _RIG,
        name,
        lift_height=_lift_height(name),
        placed_key=f"in_bin_{name}",
        placed_predicate=MetricTerm.configure(
            func=metrics.object_in_region,
            params={"entity_name": name, "region_min": list(lo), "region_max": list(hi), "relative": True},
        ),
    )


def _object_rewards(name: str) -> dict[str, RewardTermOptions]:
    return staged_object_rewards(
        _RIG, name, target_xy=target_xy(name), lift_height=_lift_height(name), placed_key=f"in_bin_{name}"
    )


def _object_terminations(name: str) -> dict[str, TerminationTermOptions]:
    return {f"{name}_fallen": fallen_termination(name, top_z=BIN_TOP_Z)}


def _object_events(name: str) -> dict[str, EventTermOptions]:
    # robosuite: x in [-0.145, 0.145], y in [-0.195, 0.195] about bin1 (inset by the object's radius),
    # yaw U[0, 2pi), rejection sampling against overlaps. Here: the object's own quadrant of bin1.
    return {
        f"place_{name}": EventTerm.configure(
            # ``_spawn_home`` is a body-origin number — the frame this term writes, not the centre of
            # mass Genesis re-origins a free MJCF body onto. The default orientation is identity here,
            # so the yaw range is unchanged.
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": name,
                "pos_x_range": _spawn_jitter(name, 0),
                "pos_y_range": _spawn_jitter(name, 1),
                "yaw_range": (-3.14159, 3.14159),
            },
        ),
    }


_OBSERVATIONS = dict(
    **proprio_observations(_RIG, prefix="robot0_"),
    **merged(lambda name: object_observations(_RIG, name), OBJECT_NAMES),
    last_action=ObsTerm.configure(func=observations.last_action),
)


# --- variants ---------------------------------------------------------------------------------------


class PickPlaceObjectMod(TaskMod):
    """robosuite's ``PickPlace<Object>`` variants (``--object``): keep one object, drop the other three.

    Removes the other objects' scene entities and every term keyed on them (observations, rewards,
    metrics, terminations, placement events), so success becomes the kept object's ``in_bin`` and
    the observation vector shrinks accordingly. Runs after ``--robot``; the term names it filters on
    do not depend on the rig.
    """

    def __init__(
        self,
        object: Annotated[
            Literal["all", "milk", "bread", "cereal", "can"],
            "Object set: all four (robosuite PickPlace) or a single one (PickPlaceMilk / Bread / Cereal / Can).",
        ] = "all",
    ):
        self.object = object

    def apply(self, config: RobosuitePickPlaceConfig) -> RobosuitePickPlaceConfig:
        if self.object == "all":
            return config
        for name in OBJECT_NAMES:
            if name != self.object:
                drop_object(config, name)
        config.runner_options.experiment_name = f"{config.runner_options.experiment_name}_{self.object}"
        return config


# --- the task ---------------------------------------------------------------------------------------


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod, PickPlaceObjectMod))
class RobosuitePickPlaceConfig(RobosuiteTaskConfig):
    env_options: EnvOptions = robosuite_env_options(max_collision_pairs=80)

    scene_options: SceneOptions = _pick_place_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG))

    metric_options: MetricManagerOptions = MetricManagerOptions(**merged(_object_metrics, OBJECT_NAMES))

    reward_options: RewardManagerOptions = RewardManagerOptions(**merged(_object_rewards, OBJECT_NAMES))

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        **merged(_object_terminations, OBJECT_NAMES),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        **merged(_object_events, OBJECT_NAMES),
        reset_robot=arm_init_noise(_RIG),
    )

    # robosuite's bins-arena agentview sits higher and further out than the table one: both bins in frame.
    cameras_options: CamerasOptions = agentview_camera(cam_pos=(1.45, 0.0, 1.95), cam_lookat=(0.0, 0.0, 0.8))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_pick_place")
