"""Scripted demonstrators for the SIMPLER Bridge tasks (``eden datagen --task <name>``).

The WidowX250 carries its own fingers, so the rig is read off the robot itself
(:func:`~eden.extensions.scripted_demos.rig.gripper_rig`). The table tasks' objects rest on the ground
plane (``z = 0``), placed by the reset at SIMPLER's layouts, within about ``x in [0.21, 0.41]``,
``y in [-0.07, 0.13]`` in front of the arm (:mod:`eden.tasks.simpler.bridge_table`); the eggplant starts
in the sink's basin (:mod:`eden.tasks.simpler.bridge_sink`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.extensions.scripted_demos.block_world import TowerController
from eden.extensions.scripted_demos.primitives import PickPlaceParams
from eden.extensions.scripted_demos.put_on_routine import Goal, Hand, put_on_routine
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.extensions.scripted_demos.routines import RoutineDemonstrator
from eden.tasks.simpler.bridge_sink.config import (
    BASKET_REGION,
    BASIN_FLOOR_Z,
    BASIN_SPAWN_X,
    BASIN_SPAWN_Y,
    SINK_RIM_Z,
)

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

_TABLE_Z = 0.0
#: Every tool target is clipped to this box: fingertips 2 mm off the ground, and no higher than the
#: camera keeps the hand in view (it sits 34 cm up, looking down onto the table). The fingers reach
#: ~1.9 cm past the pads, so the pads can come no lower than ~2.1 cm — 9 mm below a 3 cm cube's top.
_WORKSPACE = ((0.1, -0.25, 0.002), (0.45, 0.25, 0.2))
#: Where the hand retreats to after a place: above the middle of the placement area.
_EFFECTOR_BOUNDS = ((0.2, -0.1, 0.1), (0.3, 0.1, 0.16))
#: Free spots a block is moved to (to clear it, or by mistake): a box in front of the arm around the reset layouts.
_SPOT_BOUNDS = ((0.2, -0.15), (0.35, 0.15))
_PEAK_SPEED = 0.12
#: Hover height [m] above the pick / place pose: Bridge-scale, down from OCBench's table-scale 6–22 cm,
#: which carried the hand out of the camera's view for seconds at a time.
_DESCENT_HEIGHT = (0.04, 0.12)
#: Wrist tilt cap [rad], down from OCBench's 25 deg. The 6-DOF WidowX cannot hold a strongly tilted
#: hand everywhere in this workspace: the damped IK trades position for orientation and the hand
#: stalls short of its hover pose. Measured on 32 envs (MPR collision): 25 deg -> 91 % success,
#: 6 deg (this) -> 93 %, none -> 99 %.
_MAX_TILT = 0.1
#: Largest hand move asked for per 0.2 s step [m]. The WidowX's damped IK realizes only part of each
#: command, so the hand trails the plan by ~4.5 cm and following it takes commands near 5 cm: under
#: the default 5 cm clip a quarter of the vertical commands sat at it; at 10 cm about 1 % do.
_MAX_STEP_TRANSLATION = 0.1
#: The WidowX's fingers open to a ~5.4 cm gap and are ~1 cm thick; the hand clears the low camera's view
#: by going up and back toward the base.
_HAND = Hand(finger_reach=(0.027, 0.037), clear_view_pos=(0.15, 0.0, 0.18))


@SCRIPTED_DEMO_REGISTRY.register(name="simpler_stack_green_cube_on_yellow_cube")
def stack_green_cube_on_yellow_cube(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> TowerController:
    """The green 3 cm cube (``source``) onto the yellow one (``target``)."""
    return TowerController(
        env,
        ["target", "source"],
        table_z=_TABLE_Z,
        spot_bounds=_SPOT_BOUNDS,
        min_spot_distance=0.06,
        workspace=_WORKSPACE,
        effector_bounds=_EFFECTOR_BOUNDS,
        p_mistake=p_mistake,
        seed=seed,
        max_step_translation=_MAX_STEP_TRANSLATION,
        params=PickPlaceParams(peak_speed=_PEAK_SPEED, max_tilt=_MAX_TILT, descent_height=_DESCENT_HEIGHT),
    )


def _put_on(env: EnvBase, goal: Goal, *, seed: int, p_mistake: float, privileged: bool) -> RoutineDemonstrator:
    """The put-on routine for ``source`` in ``env``; camera-only through ``camera`` unless ``privileged``."""
    return RoutineDemonstrator(
        env,
        put_on_routine("source", goal, hand=_HAND, p_mistake=p_mistake, camera=None if privileged else "camera"),
        seed=seed,
        workspace=_WORKSPACE,
        peak_speed=_PEAK_SPEED,
        max_step_translation=_MAX_STEP_TRANSLATION,
        privileged=privileged,
        cameras=() if privileged else ("camera",),
    )


@SCRIPTED_DEMO_REGISTRY.register(name="simpler_put_spoon_on_table_cloth")
def put_spoon_on_table_cloth(
    env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1, privileged: bool = True
) -> RoutineDemonstrator:
    """The spoon (``source``) onto the folded table cloth (``target``) — a tool routine."""
    return _put_on(env, Goal(target="target"), seed=seed, p_mistake=p_mistake, privileged=privileged)


@SCRIPTED_DEMO_REGISTRY.register(name="simpler_put_carrot_on_plate")
def put_carrot_on_plate(
    env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1, privileged: bool = True
) -> RoutineDemonstrator:
    """The carrot (``source``) onto the plate (``target``) — a tool routine."""
    return _put_on(env, Goal(target="target"), seed=seed, p_mistake=p_mistake, privileged=privileged)


@SCRIPTED_DEMO_REGISTRY.register(name="simpler_put_eggplant_in_basket")
def put_eggplant_in_basket(
    env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1, privileged: bool = True
) -> RoutineDemonstrator:
    """The eggplant (``source``) out of the sink's basin, over its rim and into the basket — a tool routine.

    The basket is part of the fixed sink, so its box is a constant; the eggplant is laid in along
    the basket's long side.
    """
    goal = Goal(
        region=BASKET_REGION,
        object_yaw=0.0,
        obstacle_top=SINK_RIM_Z,
        # Mistakes drop the eggplant where the reset puts it: anywhere in the basin, it rolls toward -y
        # and can end wedged against that wall, where the fingers cannot get down to it.
        mistake_bounds=((BASIN_SPAWN_X[0], BASIN_SPAWN_Y[0]), (BASIN_SPAWN_X[1], BASIN_SPAWN_Y[1])),
        mistake_floor=BASIN_FLOOR_Z,
    )
    return _put_on(env, goal, seed=seed, p_mistake=p_mistake, privileged=privileged)


put_spoon_on_table_cloth.episode_length_s = put_carrot_on_plate.episode_length_s = 60.0
put_eggplant_in_basket.episode_length_s = 60.0
