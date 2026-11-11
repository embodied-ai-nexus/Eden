"""Block-world demonstrators: build a tower, or lift a block — with OCBench's deliberate mistakes.

Port of ``ocbench/oracles/controllers/block.py`` (``BlockController``) onto
:class:`~eden.extensions.scripted_demos.controller.ScriptedPickPlaceController`. The objects are
boxes resting on a table; a subgoal moves one of them.

* **Goal subgoals** do the task: the lowest block of the tower that is not yet in place goes on
  the one below it — after whatever covers either of them has been moved out of the way.
* **Mistakes.** With probability ``p_mistake`` a subgoal is replaced by a wrong one: a random
  uncovered block goes to a random free spot on the table or, half the time, onto another
  uncovered block. The demonstrator recovers on its own (it re-plans from the resulting state),
  which is what makes the data suboptimal in the way human play data is: the high-level plan is
  noisy, not just the motion.

The decision logic is a per-env Python loop over a CPU snapshot, as in OCBench — it runs once per
completed subgoal, not per step.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from eden.extensions.scripted_demos.controller import ObjectSnapshot, ScriptedPickPlaceController, Subgoal
from eden.extensions.scripted_demos.rig import entity_entries
from eden.options.entities import BoxOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

#: Position tolerance [m] for "this block is resting where it should be".
REST_TOLERANCE = 0.015
#: Horizontal tolerance [m] for "this block is centred on the one below".
STACK_XY_TOLERANCE = 0.02


class BlockWorldController(ScriptedPickPlaceController):
    """Shared block-world geometry: sizes, what covers what, free table spots, and mistakes.

    Parameters
    ----------
    env : EnvBase
        The built environment.
    block_names : list[str]
        Scene entities that are boxes (:class:`~eden.options.entities.BoxOptions`) on the table.
    table_z : float
        Height [m] of the table top.
    spot_bounds : tuple[tuple[float, float], tuple[float, float]]
        Low / high xy corners of the free spots blocks are moved to.
    min_spot_distance : float
        A free spot keeps at least this distance [m] from every other block.
    p_mistake : float
        Probability a subgoal is replaced by a mistake.
    **kwargs
        Forwarded to :class:`~eden.extensions.scripted_demos.controller.ScriptedPickPlaceController`.
    """

    def __init__(
        self,
        env: EnvBase,
        block_names: list[str],
        *,
        table_z: float,
        spot_bounds: tuple[tuple[float, float], tuple[float, float]],
        min_spot_distance: float = 0.08,
        p_mistake: float = 0.1,
        **kwargs,
    ):
        if not 0.0 <= p_mistake <= 1.0:
            raise ValueError(f"p_mistake must be a probability in [0, 1], got {p_mistake}.")
        self.table_z = table_z
        self.spot_bounds = (np.asarray(spot_bounds[0], dtype=np.float64), np.asarray(spot_bounds[1], dtype=np.float64))
        self.min_spot_distance = min_spot_distance
        self.p_mistake = p_mistake
        self.half_heights = np.array([_box_half_extent(env, name) for name in block_names])
        super().__init__(env, block_names, **kwargs)

    # --- geometry on one env's snapshot ----------------------------------------------------------

    def _covering(self, pos: np.ndarray, block: int) -> list[int]:
        """Blocks resting on ``block`` (higher, and overlapping it in xy)."""
        h = self.half_heights
        return [
            other
            for other in range(len(pos))
            if other != block
            and pos[other, 2] > pos[block, 2]
            and np.linalg.norm(pos[other, :2] - pos[block, :2]) < h[other] + h[block]
        ]

    def _top_blocks(self, pos: np.ndarray) -> list[int]:
        return [block for block in range(len(pos)) if not self._covering(pos, block)]

    def _on_table(self, pos: np.ndarray, block: int) -> bool:
        return abs(pos[block, 2] - (self.table_z + self.half_heights[block])) < REST_TOLERANCE

    def _stacked_on(self, pos: np.ndarray, block: int, base: int) -> bool:
        h = self.half_heights
        return (
            np.linalg.norm(pos[block, :2] - pos[base, :2]) < STACK_XY_TOLERANCE
            and abs(pos[block, 2] - (pos[base, 2] + h[base] + h[block])) < REST_TOLERANCE
        )

    def _free_spot(self, pos: np.ndarray, block: int) -> np.ndarray:
        """A table position for ``block`` clear of every other block (best of 100 draws, as in OCBench)."""
        others = [pos[i, :2] for i in range(len(pos)) if i != block]
        low, high = self.spot_bounds
        for _ in range(100):
            xy = self.np_rng.uniform(low, high)
            if all(np.linalg.norm(xy - other) >= self.min_spot_distance for other in others):
                break
        return np.array([xy[0], xy[1], self.table_z + self.half_heights[block]])

    def _move_to_free_spot(self, pos: np.ndarray, block: int, kind: str) -> Subgoal:
        return Subgoal(block, self._free_spot(pos, block), float(self.np_rng.uniform(0.0, 2 * math.pi)), kind=kind)

    def _uncover(self, pos: np.ndarray, block: int) -> Subgoal | None:
        """Move the topmost block resting on ``block`` to a free spot, if any."""
        covering = self._covering(pos, block)
        if not covering:
            return None
        top = max(covering, key=lambda other: pos[other, 2])
        return self._move_to_free_spot(pos, top, kind="clear")

    def _mistake(self, pos: np.ndarray) -> Subgoal | None:
        """A wrong but plausible move: an uncovered block to a free spot, or onto another (OCBench
        ``_mistake_subgoal``).
        """
        top = self._top_blocks(pos)
        if not top:
            return None
        block = int(self.np_rng.choice(top))
        supports = [other for other in top if other != block]
        if supports and self.np_rng.uniform() < 0.5:
            base = int(self.np_rng.choice(supports))
            goal = pos[base] + np.array([0.0, 0.0, self.half_heights[base] + self.half_heights[block]])
            return Subgoal(block, goal, float(self.np_rng.uniform(0.0, 2 * math.pi)), kind="mistake")
        return self._move_to_free_spot(pos, block, kind="mistake")

    # --- decision ---------------------------------------------------------------------------------

    def goal_subgoal(self, env_id: int, pos: np.ndarray, yaw: np.ndarray) -> Subgoal | None:
        """The next step toward the task for one env, or ``None`` when it is complete."""
        raise NotImplementedError

    def choose_subgoals(self, env_ids: list[int], objects: ObjectSnapshot) -> list[Subgoal | None]:
        subgoals: list[Subgoal | None] = []
        for row, env_id in enumerate(env_ids):
            pos, yaw = objects.pos[row], objects.yaw[row]
            subgoal = self.goal_subgoal(env_id, pos, yaw)
            if subgoal is not None and self.np_rng.uniform() < self.p_mistake:
                subgoal = self._mistake(pos) or subgoal
            subgoals.append(subgoal)
        return subgoals


class TowerController(BlockWorldController):
    """Stack ``block_names`` into one tower, bottom to top, wherever the bottom block stands.

    robosuite ``Stack`` is the two-block case (``["cubeB", "cubeA"]``); OCBench's ``stack``
    tasks are the longer ones.
    """

    def goal_subgoal(self, env_id: int, pos: np.ndarray, yaw: np.ndarray) -> Subgoal | None:
        num_blocks = len(pos)
        level = 0
        while level < num_blocks and (
            self._on_table(pos, 0) if level == 0 else self._stacked_on(pos, level, level - 1)
        ):
            level += 1
        if level == num_blocks:
            return None
        if level == 0:
            # The base sits on another block: uncover it, then put it down.
            return self._uncover(pos, 0) or self._move_to_free_spot(pos, 0, kind="clear")
        base = level - 1
        for block in (base, level):
            blocking = [other for other in self._covering(pos, block) if other != level]
            if blocking:
                top = max(blocking, key=lambda other: pos[other, 2])
                return self._move_to_free_spot(pos, top, kind="clear")
        goal = pos[base] + np.array([0.0, 0.0, self.half_heights[base] + self.half_heights[level]])
        return Subgoal(level, goal, float(yaw[base]), kind="goal")


class LiftController(BlockWorldController):
    """Pick one block and hold it ``lift_height`` above the table (robosuite ``Lift``).

    The lift itself has no OCBench counterpart; mistakes are the block-world ones — the block
    is put down somewhere else first and the lift starts over from there.

    Parameters
    ----------
    lift_height : tuple[float, float]
        Range [m] the block's bottom is held above the table.
    lift_xy : float
        Largest horizontal drift [m] of the held pose from where the block was picked.
    **kwargs
        Forwarded to :class:`BlockWorldController`.
    """

    def __init__(self, env: EnvBase, block_names: list[str], *, lift_height=(0.1, 0.2), lift_xy=0.05, **kwargs):
        if len(block_names) != 1:
            raise ValueError(f"LiftController lifts one block, got block_names={block_names}.")
        self.lift_height = lift_height
        self.lift_xy = lift_xy
        super().__init__(env, block_names, **kwargs)

    def goal_subgoal(self, env_id: int, pos: np.ndarray, yaw: np.ndarray) -> Subgoal | None:
        # Done once the block is held off the table; a dropped block is lifted again.
        if pos[0, 2] - (self.table_z + self.half_heights[0]) > 0.5 * self.lift_height[0]:
            return None
        return Subgoal(
            0, self._held_pose(pos), float(yaw[0] + self.np_rng.uniform(-0.3, 0.3)), release=False, kind="goal"
        )

    def _held_pose(self, pos: np.ndarray) -> np.ndarray:
        xy = pos[0, :2] + self.np_rng.uniform(-self.lift_xy, self.lift_xy, size=2)
        z = self.table_z + self.half_heights[0] + self.np_rng.uniform(*self.lift_height)
        return np.array([xy[0], xy[1], z])


def _box_half_extent(env: EnvBase, name: str) -> float:
    """Half height [m] of the box entity ``name``, read off the scene options it was built from."""
    options = getattr(env.scene_options, name, None)
    if not isinstance(options, BoxOptions):
        raise TypeError(
            f"Block '{name}' must be a BoxOptions entity (its half height is read off its size), found "
            f"{type(options).__name__}. Scene entities: {entity_entries(env.scene_options)}."
        )
    return float(options.size[2]) / 2.0
