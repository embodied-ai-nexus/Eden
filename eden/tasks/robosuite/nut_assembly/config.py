"""robosuite ``NutAssembly``: thread the square nut onto the square peg and the round nut onto the round peg.

Port of ``robosuite/environments/manipulation/nut_assembly.py`` on the ``PegsArena``: a 0.8 m table
whose top is 0.82 m above the floor, with two fixed pegs 13 cm tall standing on it — the square peg
(3.2 cm square) at ``(0.23, 0.10)`` and the round peg (4 cm diameter) at ``(0.23, -0.10)``. The Panda
stands on its pedestal 0.56 m behind the table centre. The nuts are robosuite's own MJCFs
(``objects/square-nut.xml``, ``objects/round-nut.xml``): a 2 cm-thick ring with a handle, root at the
ring centre, spawned on the robot's side of the table (``x ≈ -0.11``) — the square nut at ``y ∈ [0.11,
0.225]``, the round nut at ``y ∈ [-0.225, -0.11]``, random yaw.

Reward, per nut, from robosuite's ``staged_rewards`` multipliers (robosuite ``max``-es the stages over
the nuts still off their pegs; here each nut carries its own set and the terms are summed):

* ``reach_<nut>``  — ``0.1 (1 - tanh(10 d))`` from the grasp point to the nut. robosuite measures to
  the ``handle_site``; MJCF sites are not exposed as Genesis links, so this is the ring centre — the
  handle is 5.4–6 cm further out, which only offsets the (saturating) tanh, it does not change where the
  fingers must close.
* ``grasp_<nut>``  — 0.35 while both fingerpads touch the nut (``_check_grasp``).
* ``lift_<nut>``   — 0.5 once the nut is 10 cm above the table (robosuite: a tanh ramp to ``table + 0.2``).
* ``hover_<nut>``  — ``0.7 [lifted] (1 - tanh(10 d_xy))`` to the matching peg's xy.
* ``on_peg_<nut>`` — 1.0 for robosuite's ``on_peg``: ``|dx| < 0.03 and |dy| < 0.03`` to the peg and
  ``z < table + 0.05``, **and released**. robosuite tests release as ``r_reach < 0.6`` (grasp point more
  than 4.24 cm from the nut); here it is ``NOT PadsGrasping`` — same intent, without a distance the
  gripper could satisfy while still pinching the handle.

Success is both ``on_peg_<nut>`` at once (robosuite ``_check_success``). ``--nut square`` / ``--nut
round`` are robosuite's ``NutAssemblySquare`` / ``NutAssemblyRound`` (``single_object_mode=2``): the
other nut and its terms are removed and success is the kept nut's ``on_peg``. ``NutAssemblySingle``
(a random nut per reset) has no counterpart — a per-reset entity swap is not expressible in a config.

Deviations from robosuite: nuts are spawned resting on the table (root 1 cm above it) instead of
6 cm in the air; the two nuts share robosuite's own non-overlapping y bands so no rejection sampling
is needed; the episode is :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` long rather than 50 s;
a nut falling off the table ends it early. The Eden pegs are plain box / cylinder primitives (robosuite's
brass / steel visuals are textures).
"""

from __future__ import annotations

from typing import Annotated, Literal

from eden import EventMode, events, observations, terminations
from eden.options.camera import CamerasOptions
from eden.options.entities import BoxOptions, CylinderOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.managers.events import EventTermOptions
from eden.options.managers.metrics import MetricTermOptions
from eden.options.managers.rewards import RewardTermOptions
from eden.options.managers.terminations import TerminationTermOptions
from eden.options.materials import RigidMaterialOptions
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
    PANDA_TABLE_XY,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    ROBOT_BASE_Z,
    table_options,
)
from eden.tasks.robosuite.nut_assembly.custom_terms import ObjectNearPointXY
from eden.tasks.robosuite.staged_objects import staged_object_metrics, staged_object_rewards
from eden.tasks.robosuite.variants import drop_object
from eden.options.objects.robosuite import RobosuiteObjectOptions, RobosuiteRoundNut, RobosuiteSquareNut
from eden.terms import DoneTerm, EventTerm, ObsTerm
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

# --- arena (robosuite PegsArena) -------------------------------------------------------------------

_BASE_POS = (PANDA_TABLE_XY[0], PANDA_TABLE_XY[1], ROBOT_BASE_Z)
_RIG = panda_rig("robot", _BASE_POS)

#: ``PegsArena(table_offset=(0, 0, 0.82))``.
_TABLE_TOP_Z = 0.82
#: Both pegs: body z 0.85, half-height 0.1 → they stand from 0.75 (inside the slab) to 0.95.
_PEG_Z = 0.85
_PEG_HEIGHT = 0.2
_SQUARE_PEG_HALF = 0.016
_ROUND_PEG_RADIUS = 0.02
_PEG_FRICTION = 1.0
_BRASS = (0.72, 0.58, 0.28, 1.0)
_STEEL = (0.62, 0.64, 0.66, 1.0)

# --- objects (robosuite SquareNutObject / RoundNutObject) -------------------------------------------

NUT_NAMES: tuple[str, ...] = ("square", "round")
_NUT_OPTIONS: dict[str, type[RobosuiteObjectOptions]] = {"square": RobosuiteSquareNut, "round": RobosuiteRoundNut}
_PEG_XY = {"square": (0.23, 0.10), "round": (0.23, -0.10)}
#: robosuite samplers: x ∈ [-0.115, -0.11] for both; y ∈ [0.11, 0.225] (square) / [-0.225, -0.11] (round).
_NUT_X_RANGE = (-0.115, -0.11)
_NUT_Y_RANGE = {"square": (0.11, 0.225), "round": (-0.225, -0.11)}
#: The nut geoms are 1 cm half-height boxes centred on the root: resting on the table puts the root 1 cm up.
_NUT_HALF_HEIGHT = 0.01
_NUT_REST_Z = _TABLE_TOP_Z + _NUT_HALF_HEIGHT
_LIFT_HEIGHT = _TABLE_TOP_Z + 0.10
#: robosuite ``on_peg``: per-axis 3 cm box around the peg, root below table + 5 cm.
_ON_PEG_RADIUS = 0.03
_ON_PEG_Z_MAX = _TABLE_TOP_Z + 0.05


def _nut_home(name: str) -> tuple[float, float, float]:
    y_lo, y_hi = _NUT_Y_RANGE[name]
    return (sum(_NUT_X_RANGE) / 2, (y_lo + y_hi) / 2, _NUT_REST_Z)


def _centred(span: tuple[float, float]) -> tuple[float, float]:
    half = (span[1] - span[0]) / 2
    return (-half, half)


def _nut(name: str) -> RobosuiteObjectOptions:
    return _NUT_OPTIONS[name](default_root_pos=_nut_home(name))


def _nut_assembly_scene() -> SceneOptions:
    return SceneOptions(
        robot=_RIG,
        **arena_entities(),
        mount=rethink_mount(_BASE_POS),
        table=table_options(top_z=_TABLE_TOP_Z),
        peg_square=BoxOptions(
            size=(2 * _SQUARE_PEG_HALF, 2 * _SQUARE_PEG_HALF, _PEG_HEIGHT),
            fixed=True,
            default_root_pos=(*_PEG_XY["square"], _PEG_Z),
            material=RigidMaterialOptions(friction=_PEG_FRICTION),
            color=_BRASS,
        ),
        peg_round=CylinderOptions(
            radius=_ROUND_PEG_RADIUS,
            height=_PEG_HEIGHT,
            fixed=True,
            default_root_pos=(*_PEG_XY["round"], _PEG_Z),
            material=RigidMaterialOptions(friction=_PEG_FRICTION),
            color=_STEEL,
        ),
        **{name: _nut(name) for name in NUT_NAMES},
    )


# --- per-nut terms ----------------------------------------------------------------------------------


def _nut_metrics(name: str) -> dict[str, MetricTermOptions]:
    # robosuite on_peg: within 3 cm of the peg per axis, down on the table (and released).
    return staged_object_metrics(
        _RIG,
        name,
        lift_height=_LIFT_HEIGHT,
        placed_key=f"on_peg_{name}",
        placed_predicate=ObjectNearPointXY.configure(
            object_name=name, point_xy=_PEG_XY[name], radius=_ON_PEG_RADIUS, z_max=_ON_PEG_Z_MAX
        ),
    )


def _nut_rewards(name: str) -> dict[str, RewardTermOptions]:
    return staged_object_rewards(
        _RIG, name, target_xy=_PEG_XY[name], lift_height=_LIFT_HEIGHT, placed_key=f"on_peg_{name}"
    )


def _nut_terminations(name: str) -> dict[str, TerminationTermOptions]:
    return {f"{name}_fallen": fallen_termination(name, top_z=_TABLE_TOP_Z)}


def _nut_events(name: str) -> dict[str, EventTermOptions]:
    # robosuite: UniformRandomSampler(x_range=[-0.115, -0.11], y_range=<band>, rotation=None) about the table.
    return {
        f"place_{name}": EventTerm.configure(
            # ``_nut_home`` is a body-origin number — the frame this term writes; the nut's centre of
            # mass sits ~1.1 cm along x from it. Default quat is identity, so the yaw range carries
            # over unchanged.
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": name,
                "pos_x_range": _centred(_NUT_X_RANGE),
                "pos_y_range": _centred(_NUT_Y_RANGE[name]),
                "yaw_range": (-3.14159, 3.14159),
            },
        ),
    }


_OBSERVATIONS = dict(
    **proprio_observations(_RIG, prefix="robot0_"),
    **merged(lambda name: object_observations(_RIG, name), NUT_NAMES),
    last_action=ObsTerm.configure(func=observations.last_action),
)


# --- variants ---------------------------------------------------------------------------------------


class NutAssemblyNutMod(TaskMod):
    """robosuite's ``NutAssemblySquare`` / ``NutAssemblyRound`` variants (``--nut``): keep one nut.

    Removes the other nut's scene entity and every term keyed on it (observations, rewards, metrics,
    termination, placement event), so success becomes the kept nut's ``on_peg``. Both pegs stay.
    Runs after ``--robot``; the term names it filters on do not depend on the rig.
    """

    def __init__(
        self,
        nut: Annotated[
            Literal["both", "square", "round"],
            "Nut set: both (robosuite NutAssembly) or a single one (NutAssemblySquare / NutAssemblyRound).",
        ] = "both",
    ):
        self.nut = nut

    def apply(self, config: RobosuiteNutAssemblyConfig) -> RobosuiteNutAssemblyConfig:
        if self.nut == "both":
            return config
        for name in NUT_NAMES:
            if name != self.nut:
                drop_object(config, name)
        config.runner_options.experiment_name = f"{config.runner_options.experiment_name}_{self.nut}"
        return config


# --- the task ---------------------------------------------------------------------------------------


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod, NutAssemblyNutMod))
class RobosuiteNutAssemblyConfig(RobosuiteTaskConfig):
    env_options: EnvOptions = robosuite_env_options(max_collision_pairs=60)

    scene_options: SceneOptions = _nut_assembly_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG))

    metric_options: MetricManagerOptions = MetricManagerOptions(**merged(_nut_metrics, NUT_NAMES))

    reward_options: RewardManagerOptions = RewardManagerOptions(**merged(_nut_rewards, NUT_NAMES))

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        **merged(_nut_terminations, NUT_NAMES),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        **merged(_nut_events, NUT_NAMES),
        reset_robot=arm_init_noise(_RIG),
    )

    cameras_options: CamerasOptions = agentview_camera(cam_pos=(1.35, 0.0, 1.6), cam_lookat=(0.0, 0.0, 0.87))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_nut_assembly")
