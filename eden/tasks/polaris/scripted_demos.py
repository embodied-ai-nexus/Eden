"""Scripted demonstrators for the PolaRiS DROID tasks (``eden datagen --task polaris_move_latte_cup_batched``, …).

Each is the put-on tool routine (:mod:`eden.extensions.scripted_demos.put_on_routine`) on the DROID
rig (Franka + Robotiq 2F85), one put-on per object the task's PolaRiS rubric scores
(:mod:`eden.tasks.polaris.rubric`), privileged: footprints come off the collision meshes.

They are registered under the ``*_batched`` tasks: ``eden datagen`` needs the RL surface (episode
ends, rewards) and many envs, and the single-env Polaris tasks are teleop / inference scenes without
either. The table top is ``z = 0`` in every scene (the splat is placed on the ground plane); the
mistake spots below are clear table next to each scene's objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.extensions.scripted_demos.put_on_routine import Goal, Hand, put_on_routine
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.extensions.scripted_demos.routines import RoutineDemonstrator

if TYPE_CHECKING:
    from collections.abc import Generator

    from eden.envs.base import EnvBase
    from eden.extensions.scripted_demos.routines import Routine
    from eden.extensions.scripted_demos.tools import Motion, Outcome, Tools

#: Every tool target is clipped to this box: fingertips 2 mm off the table, and inside the Franka's reach.
_WORKSPACE = ((0.2, -0.45, 0.002), (0.95, 0.45, 0.5))
_PEAK_SPEED = 0.15
#: The 2F85's pads open to its 8.5 cm stroke and its fingers are ~1.5 cm thick; the hand clears the
#: external camera's view by going up, where it starts.
_HAND = Hand(finger_reach=(0.0425, 0.058), clear_view_pos=(0.35, 0.0, 0.45))
#: The 2F85 closed on nothing settles at 0.998 of its stroke; on a cup's 3 mm wall it stops at 0.95.
_HOLDING_CLOSURE = 0.97
#: How far [m] from the base a container goal is aimed: the hand pointing down at a carry height
#: stalls short of targets past ~0.8 m.
_MAX_REACH = 0.75


def _put(source: str, goal: Goal, p_mistake: float, *, grasp: str = "pinch", clutter: tuple[str, ...] = ()) -> Routine:
    return put_on_routine(source, goal, hand=_HAND, p_mistake=p_mistake, camera=None, grasp=grasp, clutter=clutter)


def _in_order(*routines: Routine) -> Routine:
    """One routine running ``routines`` one after the other."""

    def run(tools: Tools) -> Generator[Motion, Outcome, None]:
        for routine in routines:
            yield from routine(tools)

    run.entity_names = tuple(name for routine in routines for name in getattr(routine, "entity_names", ()))
    return run


def _demonstrator(env: EnvBase, routine: Routine, seed: int) -> RoutineDemonstrator:
    return RoutineDemonstrator(
        env, routine, seed=seed, workspace=_WORKSPACE, peak_speed=_PEAK_SPEED, holding_closure=_HOLDING_CLOSURE
    )


@SCRIPTED_DEMO_REGISTRY.register(name="polaris_block_stack_kitchen_batched")
def block_stack_kitchen(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> RoutineDemonstrator:
    """The wood cube into the tray, clear of the tomato and the corn in it, then the green cube onto it.

    The cubes lie among the bowl and the plate, which settles tilted 5 cm tall: the fingers keep off
    them, and the cubes are carried over them.
    """
    clutter = ("pink_bowl", "purple_plate", "tomato", "uw_corn")
    common = {"mistake_bounds": ((0.3, -0.3), (0.55, -0.18)), "obstacle_top": 0.07}
    tray = Goal(target="green_tray", container=True, avoid=("tomato", "uw_corn"), **common)
    return _demonstrator(
        env,
        _in_order(
            _put("wood_cube", tray, p_mistake, clutter=(*clutter, "green_cube")),
            _put("green_cube", Goal(target="wood_cube", **common), p_mistake, clutter=clutter),
        ),
        seed,
    )


@SCRIPTED_DEMO_REGISTRY.register(name="polaris_food_bussing_batched")
def food_bussing(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> RoutineDemonstrator:
    """The ice cream, then the grapes, into the bowl: the two foods PolaRiS's rubric scores."""
    mistakes = {"mistake_bounds": ((0.3, 0.22), (0.5, 0.32))}
    return _demonstrator(
        env,
        _in_order(
            _put("ice_cream", Goal(target="yellow_bowl", container=True, **mistakes), p_mistake),
            _put("grapes", Goal(target="yellow_bowl", container=True, avoid=("ice_cream",), **mistakes), p_mistake),
        ),
        seed,
    )


@SCRIPTED_DEMO_REGISTRY.register(name="polaris_pan_clean_batched")
def pan_clean(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> RoutineDemonstrator:
    """The sponge into the pan's bowl (not over its handle), rubbed back and forth, and let go."""
    goal = Goal(target="frying_pan", target_body=True, scrub=0.03, mistake_bounds=((0.35, -0.25), (0.55, -0.12)))
    return _demonstrator(env, _put("sponge", goal, p_mistake), seed)


@SCRIPTED_DEMO_REGISTRY.register(name="polaris_move_latte_cup_batched")
def move_latte_cup(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> RoutineDemonstrator:
    """The latte cup, gripped by its rim (9 cm across, wider than the stroke), onto the cutting board."""
    goal = Goal(target="cutting_board", mistake_bounds=((0.45, 0.05), (0.6, 0.2)))
    return _demonstrator(env, _put("latte_art_cup", goal, p_mistake, grasp="rim"), seed)


@SCRIPTED_DEMO_REGISTRY.register(name="polaris_organize_tools_batched")
def organize_tools(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> RoutineDemonstrator:
    """The scissor into the near side of the large container, clear of the tape in it, over the bucket's rim.

    The container's middle is 0.89 m from the base, past the hand's reach at carry height.
    """
    goal = Goal(
        target="container_00",
        container=True,
        max_reach=_MAX_REACH,
        avoid=("tape",),
        obstacle_top=0.09,
        mistake_bounds=((0.35, 0.1), (0.55, 0.25)),
    )
    return _demonstrator(env, _put("scissor", goal, p_mistake), seed)


@SCRIPTED_DEMO_REGISTRY.register(name="polaris_tape_into_container_batched")
def tape_into_container(env: EnvBase, *, seed: int = 0, p_mistake: float = 0.1) -> RoutineDemonstrator:
    """The roll of tape, gripped by its wall (13 cm across), into the container."""
    goal = Goal(target="container_02", container=True, mistake_bounds=((0.3, -0.32), (0.5, -0.2)))
    return _demonstrator(env, _put("tape_00", goal, p_mistake, grasp="rim"), seed)


for _factory in (food_bussing, pan_clean, move_latte_cup, organize_tools, tape_into_container):
    _factory.episode_length_s = 60.0
#: Two objects among clutter, and either may be put down by mistake first: ~50 s on average.
block_stack_kitchen.episode_length_s = 90.0
