"""Scripted demonstrators for the robosuite ports (``eden datagen --task robosuite_<name>``).

Each factory binds a block-world controller of :mod:`eden.extensions.scripted_demos` to one task's
scene: which boxes it moves, where the table is, and the reachable boxes the hand works in. The
robosuite arena is shared (:mod:`eden.tasks.robosuite.common`): a 0.8 m table centred at the origin
with its top at :data:`~eden.tasks.robosuite.common.TABLE_TOP_Z`, the arm 0.56 m behind its centre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eden.extensions.scripted_demos.block_world import LiftController, TowerController
from eden.extensions.scripted_demos.multi_arm import MultiArmDemonstrator
from eden.extensions.scripted_demos.primitives import PickPlaceParams
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.extensions.scripted_demos.routines import RoutineDemonstrator
from eden.tasks.robosuite.common import NARROW_TABLE_PLACEMENTS, TABLE_TOP_Z
from eden.tasks.robosuite.door.routine import door_routine
from eden.tasks.robosuite.pick_place.config import BIN_TOP_Z, OBJECT_NAMES
from eden.tasks.robosuite.pick_place.routine import pick_place_routine
from eden.tasks.robosuite.tool_hang.routine import tool_hang_routine
from eden.tasks.robosuite.wipe.routine import wipe_routine
from eden.tasks.robosuite.two_arm_transport.routine import two_arm_transport_routines

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

#: Every tool target is clipped to this box: above the table top, within the table's footprint.
_WORKSPACE = ((-0.35, -0.35, TABLE_TOP_Z + 0.005), (0.35, 0.35, TABLE_TOP_Z + 0.45))
#: Where the hand retreats to after a place: over the near half of the table, clear of the blocks.
_EFFECTOR_BOUNDS = ((-0.2, -0.15, TABLE_TOP_Z + 0.12), (0.05, 0.15, TABLE_TOP_Z + 0.3))
#: Free spots a block is moved to (to clear it, or by mistake): the robosuite spawn area, widened.
_SPOT_BOUNDS = ((-0.12, -0.15), (0.12, 0.15))
#: Nominal peak tool speed [m/s] — 0.4x OCBench's 0.3. The robosuite action is a 5 cm-per-step
#: delta-pose command through damped IK into a position-PD arm at 20 Hz, which realizes only a
#: fraction of each command per step; at OCBench's speed the arm lags the plan by several cm and
#: the command sits at its clamp ~10 % of the time, the saturated, bang-bang actions OCBench's
#: demonstrators exist to avoid. At 0.12 m/s about 1.5 % of commands clamp (Panda, Stack).
_PEAK_SPEED = 0.12


@SCRIPTED_DEMO_REGISTRY.register(name="robosuite_stack")
def robosuite_stack(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> TowerController:
    """Red ``cubeA`` onto green ``cubeB`` — a two-block tower."""
    return TowerController(
        env,
        ["cubeB", "cubeA"],
        table_z=TABLE_TOP_Z,
        spot_bounds=_SPOT_BOUNDS,
        workspace=_WORKSPACE,
        effector_bounds=_EFFECTOR_BOUNDS,
        p_mistake=p_mistake,
        seed=seed,
        params=PickPlaceParams(peak_speed=_PEAK_SPEED),
    )


@SCRIPTED_DEMO_REGISTRY.register(name="robosuite_lift")
def robosuite_lift(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> LiftController:
    """Pick the cube and hold it 10–20 cm above the table."""
    return LiftController(
        env,
        ["cube"],
        table_z=TABLE_TOP_Z,
        spot_bounds=_SPOT_BOUNDS,
        workspace=_WORKSPACE,
        effector_bounds=_EFFECTOR_BOUNDS,
        p_mistake=p_mistake,
        seed=seed,
        params=PickPlaceParams(peak_speed=_PEAK_SPEED),
    )


#: The pick-and-place arena: over both bins, fingertips above their floors, below the rims + a tall carry.
_BINS_WORKSPACE = ((-0.15, -0.55, BIN_TOP_Z + 0.005), (0.35, 0.58, 1.25))


@SCRIPTED_DEMO_REGISTRY.register(name="robosuite_pick_place")
def robosuite_pick_place(
    env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1, privileged: bool = True
) -> RoutineDemonstrator:
    """Sort the groceries present (all four, or ``--object``'s one) into their bin2 quadrants — a tool routine.

    ``privileged=False`` runs it camera-only, locating objects through ``agentview``.
    """
    objects = tuple(name for name in OBJECT_NAMES if name in env.entities)
    return RoutineDemonstrator(
        env,
        pick_place_routine(objects, p_mistake=p_mistake, camera=None if privileged else "agentview"),
        seed=seed,
        workspace=_BINS_WORKSPACE,
        peak_speed=_PEAK_SPEED,
        privileged=privileged,
        cameras=() if privileged else ("agentview",),
    )


#: Four objects at ~25 s each, mistakes and re-picks on top: 120 s on average (16 envs), 200 s leaves room.
robosuite_pick_place.episode_length_s = 200.0


#: The Door arena: the 0.8 x 0.3 m table at (-0.2, -0.35) and the door over it; the hand stays above the table.
_DOOR_WORKSPACE = ((-0.5, -0.6, TABLE_TOP_Z + 0.005), (0.25, 0.3, 1.45))


@SCRIPTED_DEMO_REGISTRY.register(name="robosuite_door")
def robosuite_door(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> RoutineDemonstrator:
    """Press the handle (with the latch), hook the lever from behind and pull the door open — a tool routine."""
    latch = "latch_joint" in env.entities["door"].dofs_name
    return RoutineDemonstrator(
        env,
        door_routine(latch=latch, p_mistake=p_mistake),
        seed=seed,
        workspace=_DOOR_WORKSPACE,
        peak_speed=_PEAK_SPEED,
    )


#: The Wipe arena: over the 0.5 x 0.8 m table (top at 0.9 m). The floor is 1 cm under its top: the
#: fingertip reach the floor is raised by is measured on the tilted home pose and reads long, and the
#: pad must be able to come down on the table (``wipe_routine``).
_WIPE_WORKSPACE = ((-0.1, -0.4, 0.89), (0.4, 0.4, 1.3))


@SCRIPTED_DEMO_REGISTRY.register(name="robosuite_wipe")
def robosuite_wipe(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> RoutineDemonstrator:
    """Sweep the closed fingers over every dirt marker in path order — a tool routine."""
    return RoutineDemonstrator(
        env, wipe_routine(p_mistake=p_mistake), seed=seed, workspace=_WIPE_WORKSPACE, peak_speed=_PEAK_SPEED
    )


#: ~30 s for a first-try opening, and a retry (a lever slipped off the palm mid-turn) costs another 30 s.
robosuite_door.episode_length_s = 120.0


#: Each arm over its own table and the gap between them; fingertips above the bin floors (0.81 m).
_TRANSPORT_WORKSPACES = {
    "robot0": ((-0.45, -0.75, TABLE_TOP_Z + 0.005), (0.45, 0.2, 1.3)),
    "robot1": ((-0.45, -0.2, TABLE_TOP_Z + 0.005), (0.45, 0.75, 1.3)),
}


@SCRIPTED_DEMO_REGISTRY.register(name="robosuite_two_arm_transport")
def robosuite_two_arm_transport(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.0) -> MultiArmDemonstrator:
    """robot0 uncovers the start bin and hands the hammer over; robot1 clears the trash and bins the hammer.

    Two tool routines, one per arm, coordinating through the env's blackboard. ``p_mistake`` is
    accepted for the generator's interface; these routines make no deliberate mistakes.
    """
    del p_mistake
    _require_opposed_arms(env)
    return MultiArmDemonstrator(
        env, two_arm_transport_routines(), seed=seed, workspace=_TRANSPORT_WORKSPACES, peak_speed=_PEAK_SPEED
    )


def _require_opposed_arms(env: EnvBase) -> None:
    """Raise unless the arms stand where ``--arrangement opposed`` puts them: the routines' layout."""
    for slot, (base_pos, _) in zip(("robot0", "robot1"), NARROW_TABLE_PLACEMENTS["opposed"]):
        placed = tuple(getattr(env.scene_options, slot).arm.default_root_pos)
        if not np.allclose(placed, base_pos):
            raise ValueError(
                f"The robosuite_two_arm_transport demonstrator is written for the opposed arrangement (arms facing "
                f"each other across the gap: {slot} at {base_pos}), but {slot} stands at {placed}. Run it without "
                f"--arrangement, or with --arrangement opposed; the hand-off point, retreats and hammer turn are "
                f"constants of that layout, so there is no parallel version to fall back to."
            )


#: The lid, the trash and the hand-off take 80-100 s (90 s on average over 32 episodes), past the
#: generator's 90 s default.
robosuite_two_arm_transport.episode_length_s = 150.0


#: The ToolHang arena: Lift's 0.8 m table; the hand reaches over the stand to 1.25 m.
_TOOL_HANG_WORKSPACE = ((-0.35, -0.35, TABLE_TOP_Z + 0.005), (0.35, 0.35, 1.25))


@SCRIPTED_DEMO_REGISTRY.register(name="robosuite_tool_hang")
def robosuite_tool_hang(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.0) -> RoutineDemonstrator:
    """Stand the frame in the stand's socket, then hang the wrench on its hook — a tool routine.

    ``p_mistake`` is accepted for the generator's interface; the routine makes no deliberate mistakes:
    every step of it is a precision placement, and a deliberate miss would end most episodes.
    """
    del p_mistake
    return RoutineDemonstrator(
        env, tool_hang_routine(), seed=seed, workspace=_TOOL_HANG_WORKSPACE, peak_speed=_PEAK_SPEED
    )


#: The frame takes ~70 s and the wrench ~40 s more, past the generator's 90 s default.
robosuite_tool_hang.episode_length_s = 180.0
