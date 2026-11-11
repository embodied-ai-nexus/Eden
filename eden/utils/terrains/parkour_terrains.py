"""Parkour / locomotion-obstacle terrain generators (ported from extreme-parkour).

These height-field generators carve corridor-style courses (stepping stones,
gaps, hurdles, steps) plus a few standalone obstacles.

Conventions: ``height_field_raw`` is indexed ``(x, y)`` where dim0 is the course
(forward) direction and dim1 is lateral; generators are shape-aware, so
non-square subterrains (e.g. 16 x 4 m lanes) place features correctly. The course
generators attach ``goals`` — an ``(N, 2)`` float array of waypoints in
**subterrain-local meters** (measured from the cell's ``[0, 0]`` corner) — and
``HalfSlopedTerrain`` attaches a normalized ``slope_vector``. Both are collected
by :func:`eden.utils.terrains.base.generate_terrain_fields` and surfaced world-frame
on the :class:`~eden.entities.terrain.Terrain` entity for goal-following command
terms.

``goals`` come out strictly ordered along the course axis and inside the walkable strip
(clear of the far-end :func:`_pad_borders` wall). The walk itself is not bounded by the
subterrain, so :func:`_fit_final_goal_x` pulls the last waypoint back when it overruns the
lane — preferring the course corridor over the bare far-end limit — and raises once the lane is
too short to hold an ordered sequence. :func:`_validate_goals` then checks the assembled array,
so the guarantee holds even for option values that make the walk step backwards rather than
being inferred from ``goals[-2]`` alone.

Three things it deliberately does not police:

* **Obstacle relief.** Height-field writes that overrun the strip are clipped (see
  :func:`_set_height_block`), so a short lane can still drop its last obstacle silently. A
  relocated final goal prefers the corridor but is not guaranteed it — see
  :func:`_pull_back_onto_corridor` for the lane shape that defeats the search.
* **Spacing between interior goals**, which is the generator's own layout, set by its
  ``x_range`` and stone/gap counts. ``tests/utils/terrains/test_parkour_terrains.py`` pins the minimum each
  generator actually produces at its shortest supported subterrain, and
  ``tests/tasks/extreme_parkour/teacher/test_config.py`` pins it for the shipped teacher lanes.
* **Lateral placement.** Only the course axis is bounded; a course whose ``y_range`` outruns
  a narrow lane can still put a waypoint in — or past — the lateral wall.
"""

from __future__ import annotations

import numpy as np
from genesis.ext.isaacgym import terrain_utils as tu
from scipy import interpolate

from eden.utils.terrains.base import TERRAIN_GENERATOR_REGISTRY, TerrainGenerator

#: Clearance (m) kept between the final goal and the far end of a subterrain. The generators
#: this was ported from clamped to ``width - 0.5 // horizontal_scale``; that 0.5 m is preserved
#: as a floor so the goal is never flush against the border wall at fine resolutions. See
#: :func:`_goal_margin_cells`.
MIN_GOAL_MARGIN_M = 0.5

#: Minimum room (m) that must remain ahead of ``goals[-2]`` for a *pulled-back* final goal to
#: be worth placing. Ordering alone is not enough: a goal-following command advances when the
#: robot is within its own threshold of the current goal (0.2 m for
#: ``eden/tasks/extreme_parkour``), so two goals closer than that are reached simultaneously and
#: the last one is degenerate. Below this the course is reported as not fitting instead.
#:
#: Scope: this bounds only the goal :func:`_fit_final_goal_x` relocates. Spacing between goals
#: the walk placed itself is the generator's own layout — its ``x_range`` and stone/gap counts —
#: and is deliberately not overridden here; a generator's designed course is not this check's to
#: retune. Those minima are pinned per generator in ``tests/utils/terrains/test_parkour_terrains.py`` instead.
MIN_GOAL_SPACING_M = 0.5


def _pad_width_cells(subterrain: tu.SubTerrain, pad_width_m: float) -> int:
    """Border-pad width in cells, clamped to at least one cell.

    Without the clamp a sub-cell ``pad_width_m`` at a coarse ``horizontal_scale``
    floors to zero, which turns :func:`_pad_borders`' ``[:, -pad_width:]`` into
    ``[:, 0:]`` and overwrites the entire height field.
    """
    return max(int(pad_width_m / subterrain.horizontal_scale), 1)


def _goal_margin_cells(subterrain: tu.SubTerrain, pad_width_m: float | None) -> int:
    """Cells at the far end of the x axis that the final goal must stay clear of.

    Two things reserve space, and the binding one wins:

    * the **border pad**, when the generator raises one — :func:`_pad_borders` turns the last
      ``pad_width`` cells into a ``pad_height`` wall, so a goal there is on unwalkable terrain;
    * :data:`MIN_GOAL_MARGIN_M` of **clearance** regardless. The pad alone is not enough: the
      shipped ``pad_width=0.1`` is a single cell either way — 0.25 m at the default resolution,
      0.08 m at the parkour lanes' 0.08 m — which would leave the final goal all but flush
      against a half-metre wall, and a quadruped has to *stop* there, within its goal-reached
      threshold. The upstream clamp guaranteed 0.5 m at any resolution; keep that as a floor.

    ``pad_width_m`` is ``None`` for a generator that does not pad (``ParkourDemoTerrain``), which
    only drops the first term — the clearance floor still applies, since the terrain simply ends.
    """
    pad_x = _pad_width_cells(subterrain, pad_width_m) if pad_width_m is not None else 0
    # At least one cell: round() floors 0.5 to 0 at horizontal_scale >= 1 m, which would drop
    # the clearance entirely for a generator that raises no pad.
    return max(pad_x, round(MIN_GOAL_MARGIN_M / subterrain.horizontal_scale), 1)


def _last_goal_cell(subterrain: tu.SubTerrain, generator: TerrainGenerator) -> int:
    """Last x cell a goal may occupy on this subterrain.

    ``-1`` because :func:`_goal_margin_cells` returns a *count* of reserved cells at the far
    end, and the last free index sits one below the first reserved one. Generators that raise
    a border wall declare ``pad_width``; ``ParkourDemoTerrain`` does not, and ``configure()``
    rejects fields a generator has not declared, so the attribute is present exactly when the
    generator really pads.
    """
    num_x = subterrain.height_field_raw.shape[0]
    return num_x - _goal_margin_cells(subterrain, getattr(generator, "pad_width", None)) - 1


def _run_out_column(subterrain: tu.SubTerrain, goal_y: float) -> np.ndarray:
    """The height profile along x at a goal's lateral position, as raw height units.

    ``goal_y`` is clamped into the field: a course whose lateral offsets outrun a very narrow
    lane can name a column that does not exist, and this only reads a *reference* height, so
    the nearest real column is the right answer rather than an error. Lateral placement itself
    is not policed here — see the module docstring.
    """
    num_y = subterrain.height_field_raw.shape[1]
    return subterrain.height_field_raw[:, int(np.clip(round(goal_y), 0, num_y - 1))]


def _pull_back_onto_corridor(
    subterrain: tu.SubTerrain, last_goal_x: int, first_goal_x: int, goal_y: float, corridor_raw: int
) -> int:
    """Farthest admissible x in ``[first_goal_x, last_goal_x]`` that sits on the course corridor.

    ``last_goal_x`` on its own is only guaranteed to be *on the strip*, not to be a place a robot
    can stop: the lane can end part-way through the course's last obstacle, which puts the
    relocated goal inside a dug gap (measured 0.99 m deep on ``ParkourGapTerrain``) or on top of a
    hurdle a few cells deep. Both are the failure the far-end clearance exists to prevent,
    displaced one obstacle earlier. So the search walks back from ``last_goal_x`` for the farthest
    cell at ``corridor_raw`` — usually ``last_goal_x`` itself, so a lane with a clean run-out is
    placed exactly where it was.

    ``corridor_raw`` comes from :meth:`RoughTerrainGenerator.corridor_height_raw`, i.e. from the
    generator that painted the relief, rather than being inferred from the height field. Inferring
    it cannot be made to work for all five courses at once: the flat ground is the most common
    height along the lane for the corridor courses but the *pit* is, for ``ParkourTerrain``.

    Falls back to ``last_goal_x`` when no admissible cell is on the corridor, which means the lane
    ended part-way through its last obstacle with no corridor left in reach. Raising there would
    be the honest reading, but it would reject lanes that otherwise build — it never fires on
    either shipped ``extreme_parkour`` grid yet does on lanes tuned to overrun — so the placement
    degrades to the far-end limit rather than to a crash. The goal stays ordered and on the
    strip either way.
    """
    column = _run_out_column(subterrain, goal_y)
    for x in range(last_goal_x, first_goal_x - 1, -1):
        if column[x] == corridor_raw:
            return x
    return last_goal_x


def _fit_final_goal_x(
    subterrain: tu.SubTerrain,
    final_dis_x: int,
    prev_goal_x: float,
    goal_y: float,
    generator: RoughTerrainGenerator,
) -> int:
    """Return the final goal's x, pulled back onto the strip when the walk overruns it.

    The randomized walk lays out its course independently of the subterrain extent, so its
    last leg can land past the far edge. That happens in two regimes, and they want opposite
    answers:

    * **Only the last leg overshoots** — the tail of a random walk, not a misconfiguration
      (the shipped ``extreme_parkour`` teacher grid hits it on ~4% of its cells). Pull the goal
      back to the farthest cell still on the course corridor
      (:func:`_pull_back_onto_corridor`): the goals stay ordered, clear of the far end, and on
      ground the robot can stop on — which is what a goal-following command needs.
    * **There is no room left to pull it back into.** The goal before it is at or past that
      limit, or within :data:`MIN_GOAL_SPACING_M` of it, so a pulled-back final goal would run
      backwards or land on top of its predecessor. Either way the command is handed an unusable
      course, so raise instead — with the shipped defaults an 8 m or 12 m subterrain lands here
      on 75% of seeds for each of the four randomized-walk generators.

    Scope: this decides the final **goal**, not the obstacle relief. The generators clip
    height-field writes that overrun the strip (see :func:`_set_height_block`), so a lane
    short enough to push the last obstacle off the edge still drops it silently. What is
    guaranteed here is the goal
    sequence, and :func:`_validate_goals` checks that guarantee on the assembled array rather
    than trusting this function's inputs.
    """
    hs = subterrain.horizontal_scale
    last_goal_x = _last_goal_cell(subterrain, generator)
    if final_dis_x <= last_goal_x:
        return int(final_dis_x)

    spacing_x = max(round(MIN_GOAL_SPACING_M / hs), 1)
    first_goal_x = int(np.ceil(prev_goal_x + spacing_x))
    if first_goal_x <= last_goal_x:
        corridor_raw = generator.corridor_height_raw(subterrain)
        return _pull_back_onto_corridor(subterrain, last_goal_x, first_goal_x, goal_y, corridor_raw)

    num_x = subterrain.height_field_raw.shape[0]
    margin_x = num_x - last_goal_x - 1
    raise ValueError(
        f"[{type(generator).__name__}] course does not fit: its last two goals are at "
        f"{prev_goal_x * hs:.1f} m and {final_dis_x * hs:.1f} m, but the last goal this "
        f"{num_x * hs:.1f} m subterrain can hold is at {last_goal_x * hs:.1f} m (keeping "
        f"{margin_x * hs:.2f} m clear at the far end), which leaves no room to pull the final "
        f"goal back and still keep it {MIN_GOAL_SPACING_M:.1f} m ahead of its predecessor. Give this "
        f"subterrain at least {(final_dis_x + margin_x + 1) * hs:.1f} m of x extent, or shorten the "
        "course by lowering this generator's num_stones / num_gaps or narrowing its x_range."
    )


def _validate_goals(subterrain: tu.SubTerrain, goals: np.ndarray, generator: TerrainGenerator) -> None:
    """Check the assembled goal array against the guarantee the module docstring advertises.

    :func:`_fit_final_goal_x` only ever inspects ``goals[-2]``, on the premise that the walk
    advances monotonically so that waypoint is the interior maximum. That premise is a property
    of each generator's option values, not of the code. A ``ParkourTerrain`` steps *backwards*
    whenever its per-stone advance collapses next to ``stone_len`` — from a large negative
    ``x_range`` floor, or from a ``horizontal_scale`` coarse enough to round the advance down to a
    single cell, which the shipped ``extreme_parkour`` difficulty-1.0 course does at 0.5 m. That
    both breaks the ordering and puts the interior maximum somewhere ``_fit_final_goal_x`` never
    looks, so unchecked such a course comes out with waypoints behind the strip origin, or with the
    last two goals on the same cell — the degeneracy :data:`MIN_GOAL_SPACING_M` rejects, reached
    through the path that returns early because the final goal already fits.

    Checking the whole array once, here, is what makes the advertised invariant true instead of
    assumed. This is the last chance to catch it: the caller converts to metres immediately
    after.
    """
    xs = goals[:, 0]
    last_goal_x = _last_goal_cell(subterrain, generator)
    if ((xs < 0) | (xs > last_goal_x)).any() or not (np.diff(xs) > 0).all():
        hs = subterrain.horizontal_scale
        raise ValueError(
            f"[{type(generator).__name__}] course is not a usable goal sequence: its goals must "
            f"advance strictly along x and stay inside [0, {last_goal_x * hs:.2f}] m of this "
            f"{subterrain.height_field_raw.shape[0] * hs:.1f} m subterrain, but came out at "
            f"{[round(float(v) * hs, 2) for v in xs]} m. A walk that steps backwards means its "
            "per-stone advance collapsed: either the x_range lower bound is negative and large "
            "next to the stone/gap length, or horizontal_scale is too coarse to resolve the "
            "advance. A walk that runs off the strip means the subterrain is too short."
        )


def _set_height_block(subterrain: tu.SubTerrain, x0: int, x1: int, y0: int, y1: int, values: np.ndarray) -> None:
    """Assign a 2D ``values`` block to ``[x0:x1, y0:y1]``, clipped to the terrain
    bounds (no-op if fully outside). ``values`` is indexed in the unclipped
    block's own frame (its ``[0, 0]`` corresponds to ``[x0, y0]``) and only the
    in-bounds sub-window is written, so the broadcast matches even when the
    block overruns a short subterrain. ``values`` must therefore be at least as
    large as the unclipped block; extra rows/cols beyond it are ignored.
    """
    rows, cols = subterrain.height_field_raw.shape
    cx0, cx1 = max(x0, 0), min(x1, rows)
    cy0, cy1 = max(y0, 0), min(y1, cols)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    subterrain.height_field_raw[cx0:cx1, cy0:cy1] = values[cx0 - x0 : cx1 - x0, cy0 - y0 : cy1 - y0]


def _pad_borders(subterrain: tu.SubTerrain, pad_width_m: float, pad_height_m: float) -> None:
    """Raise the outer border of the strip to ``pad_height`` (a wall the robot
    cannot walk off). The width comes from :func:`_pad_width_cells`, which
    :func:`_goal_margin_cells` also uses to keep the final goal clear of this wall.
    """
    pad_width = _pad_width_cells(subterrain, pad_width_m)
    pad_height = int(pad_height_m / subterrain.vertical_scale)
    subterrain.height_field_raw[:, :pad_width] = pad_height
    subterrain.height_field_raw[:, -pad_width:] = pad_height
    subterrain.height_field_raw[:pad_width, :] = pad_height
    subterrain.height_field_raw[-pad_width:, :] = pad_height


def _apply_roughness(
    subterrain: tu.SubTerrain,
    noise_range: tuple[float, float],
    noise_step: float,
    downsampled_scale: float,
    difficulty: float,
) -> None:
    """Add difficulty-scaled uniform roughness, sampled on a coarse grid and spline-upsampled.

    Heights are drawn from ``[-noise_range[0], (noise_range[1] - noise_range[0]) * difficulty
    + noise_range[0]]`` in ``noise_step`` increments (the extreme-parkour formulation) on a
    ``downsampled_scale`` grid, interpolated to the full resolution, and added in place.
    """
    hs, vs = subterrain.horizontal_scale, subterrain.vertical_scale
    num_x, num_y = subterrain.height_field_raw.shape
    size_x, size_y = num_x * hs, num_y * hs
    # RectBivariateSpline needs > 3 samples per axis for its default cubic order.
    down_x = max(int(size_x / downsampled_scale), 4)
    down_y = max(int(size_y / downsampled_scale), 4)

    max_height = (noise_range[1] - noise_range[0]) * difficulty + noise_range[0]
    step = max(int(noise_step / vs), 1)
    height_range = np.arange(int(-noise_range[0] / vs), int(max_height / vs) + step, step)

    coarse = np.random.choice(height_range, size=(down_x, down_y))
    spline = interpolate.RectBivariateSpline(np.linspace(0, size_x, down_x), np.linspace(0, size_y, down_y), coarse)
    fine = spline(np.linspace(0, size_x, num_x), np.linspace(0, size_y, num_y))
    subterrain.height_field_raw += np.rint(fine).astype(np.int16)


class RoughTerrainGenerator(TerrainGenerator):
    """Base for course generators with difficulty-scaled surface roughness (not registered)."""

    apply_roughness: bool = False
    noise_range: tuple[float, float] = (0.02, 0.06)
    noise_step: float = 0.005
    downsampled_scale: float = 0.075
    difficulty: float = 1.0

    def corridor_height_raw(self, subterrain: tu.SubTerrain) -> int:
        """Raw height of the flat ground this course walks along between its obstacles.

        The corridor generators (gap, hurdle, step, demo) carve their relief into a field that
        starts at zero and only paint ``platform_height`` onto the *start* platform, so zero is
        what the robot walks on and what the lane runs out onto. ``ParkourTerrain`` inverts that
        and overrides this.

        Used by :func:`_pull_back_onto_corridor` to place a relocated final goal on ground the
        robot can stop on. Declared by the generator rather than read off the height field
        because no single reading is right for all five courses.
        """
        return 0

    def _finish_course(
        self, subterrain: tu.SubTerrain, goals: np.ndarray, final_dis_x: int, final_dis_y: float
    ) -> None:
        """Land the final goal, check the sequence, publish it in metres, and add roughness.

        Every course generator ends this way; only whether a border pad follows differs, so
        :func:`_pad_borders` stays at the call sites that want it. Both goal steps run before
        roughness, so they read the course's own relief rather than relief plus noise.
        """
        goals[-1] = [_fit_final_goal_x(subterrain, final_dis_x, goals[-2][0], final_dis_y, self), final_dis_y]
        _validate_goals(subterrain, goals, self)
        subterrain.goals = goals * subterrain.horizontal_scale
        self._maybe_apply_roughness(subterrain)

    def _maybe_apply_roughness(self, subterrain: tu.SubTerrain) -> None:
        if self.apply_roughness:
            _apply_roughness(
                subterrain,
                noise_range=self.noise_range,
                noise_step=self.noise_step,
                downsampled_scale=self.downsampled_scale,
                difficulty=self.difficulty,
            )


@TERRAIN_GENERATOR_REGISTRY.register()
class GapParkourTerrain(TerrainGenerator):
    difficulty: float = 0.5
    platform_size: float = 4.0
    gap_depth: float = 2.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        gap_size = 0.1 + 0.3 * self.difficulty
        gap_size = int(gap_size / subterrain.horizontal_scale)
        platform_size = int(self.platform_size / subterrain.horizontal_scale)

        num_x, num_y = subterrain.height_field_raw.shape
        center_x = num_x // 2
        center_y = num_y // 2
        x1 = (num_x - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (num_y - platform_size) // 2
        y2 = y1 + gap_size

        gap_depth = -round(self.gap_depth / subterrain.vertical_scale)
        subterrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = gap_depth
        subterrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class ParkourTerrain(RoughTerrainGenerator):
    platform_len: float = 2.5
    platform_height: float = 0.0
    num_stones: int = 8
    x_range: tuple[float, float] = (1.8, 1.9)
    y_range: tuple[float, float] = (0.0, 0.1)
    z_range: tuple[float, float] = (-0.2, 0.2)
    stone_len: tuple[float, float] = (1.0, 1.0)
    stone_width: float = 0.6
    pad_width: float = 0.1
    pad_height: float = 0.5
    incline_height: float = 0.1
    last_incline_height: float = 0.6
    last_stone_len: float = 1.6
    pit_depth: tuple[float, float] = (0.5, 1.0)

    def corridor_height_raw(self, subterrain: tu.SubTerrain) -> int:
        """This course floods the lane with ``pit_depth`` and paints platforms on top of it.

        So unlike its siblings the walkable ground is ``platform_height``, not the field's initial
        value — the start platform and the run-out platform past the last stone are both painted
        there, and a goal pulled back onto the pit would be at the bottom of it.
        """
        return round(self.platform_height / subterrain.vertical_scale)

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        goals = np.zeros((self.num_stones + 2, 2))
        subterrain.height_field_raw[:] = -round(
            np.random.uniform(self.pit_depth[0], self.pit_depth[1]) / subterrain.vertical_scale
        )

        mid_y = subterrain.height_field_raw.shape[1] // 2
        stone_len = np.random.uniform(*self.stone_len)
        stone_len = 2 * round(stone_len / 2.0, 1)
        stone_len = round(stone_len / subterrain.horizontal_scale)
        dis_x_min = stone_len + round(self.x_range[0] / subterrain.horizontal_scale)
        dis_x_max = stone_len + round(self.x_range[1] / subterrain.horizontal_scale)
        dis_y_min = round(self.y_range[0] / subterrain.horizontal_scale)
        dis_y_max = round(self.y_range[1] / subterrain.horizontal_scale)
        dis_z_min = round(self.z_range[0] / subterrain.vertical_scale)
        dis_z_max = round(self.z_range[1] / subterrain.vertical_scale)
        # Ensure randint bounds stay valid even when ranges collapse after rounding.
        if dis_y_max <= dis_y_min:
            dis_y_max = dis_y_min + 1
        if dis_z_max <= dis_z_min:
            dis_z_max = dis_z_min + 1

        platform_len = round(self.platform_len / subterrain.horizontal_scale)
        platform_height = round(self.platform_height / subterrain.vertical_scale)
        subterrain.height_field_raw[0:platform_len, :] = platform_height

        stone_width = round(self.stone_width / subterrain.horizontal_scale)
        last_stone_len = round(self.last_stone_len / subterrain.horizontal_scale)

        incline_height = round(self.incline_height / subterrain.vertical_scale)
        last_incline_height = round(self.last_incline_height / subterrain.vertical_scale)

        dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
        goals[0] = [platform_len - stone_len // 2, mid_y]
        left_right_flag = np.random.randint(0, 2)
        dis_z = np.random.randint(dis_z_min, dis_z_max)

        for i in range(self.num_stones):
            dis_x += np.random.randint(dis_x_min, dis_x_max)
            pos_neg = round(2 * (left_right_flag - 0.5))
            dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
            if i == self.num_stones - 1:
                dis_x += last_stone_len // 4
                heights = (
                    np.tile(
                        np.linspace(-last_incline_height, last_incline_height, stone_width),
                        (last_stone_len, 1),
                    )
                    * pos_neg
                )
                x0 = dis_x - last_stone_len // 2
                x1 = dis_x + last_stone_len // 2
                y0 = dis_y - stone_width // 2
                y1 = dis_y + stone_width // 2
            else:
                heights = (
                    np.tile(
                        np.linspace(-incline_height, incline_height, stone_width),
                        (stone_len, 1),
                    )
                    * pos_neg
                )
                x0 = dis_x - stone_len // 2
                x1 = dis_x + stone_len // 2
                y0 = dis_y - stone_width // 2
                y1 = dis_y + stone_width // 2

            _set_height_block(subterrain, x0, x1, y0, y1, heights.astype(int) + dis_z)

            goals[i + 1] = [dis_x, dis_y]
            left_right_flag = 1 - left_right_flag
        final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // subterrain.horizontal_scale)
        subterrain.height_field_raw[final_platform_start:, :] = platform_height

        final_dis_x = dis_x + 2 * np.random.randint(dis_x_min, dis_x_max)
        self._finish_course(subterrain, goals, final_dis_x, mid_y)
        _pad_borders(subterrain, self.pad_width, self.pad_height)
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class ParkourGapTerrain(RoughTerrainGenerator):
    platform_len: float = 2.5
    platform_height: float = 0.0
    num_gaps: int = 8
    gap_size: float = 0.3
    x_range: tuple[float, float] = (1.6, 2.4)
    y_range: tuple[float, float] = (-1.2, 1.2)
    half_valid_width: tuple[float, float] = (0.6, 1.2)
    gap_depth: tuple[float, float] = (0.5, 1.0)
    pad_width: float = 0.1
    pad_height: float = 0.5
    flat: bool = False

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        goals = np.zeros((self.num_gaps + 2, 2))
        mid_y = subterrain.height_field_raw.shape[1] // 2

        dis_y_min = round(self.y_range[0] / subterrain.horizontal_scale)
        dis_y_max = round(self.y_range[1] / subterrain.horizontal_scale)

        platform_len = round(self.platform_len / subterrain.horizontal_scale)
        platform_height = round(self.platform_height / subterrain.vertical_scale)
        gap_depth = -round(np.random.uniform(self.gap_depth[0], self.gap_depth[1]) / subterrain.vertical_scale)

        half_valid_width = round(
            np.random.uniform(
                self.half_valid_width[0],
                self.half_valid_width[1],
            )
            / subterrain.horizontal_scale
        )

        subterrain.height_field_raw[0:platform_len, :] = platform_height

        gap_size = round(self.gap_size / subterrain.horizontal_scale)
        dis_x_min = round(self.x_range[0] / subterrain.horizontal_scale) + gap_size
        dis_x_max = round(self.x_range[1] / subterrain.horizontal_scale) + gap_size

        dis_x = platform_len
        goals[0] = [platform_len - 1, mid_y]
        last_dis_x = dis_x
        for i in range(self.num_gaps):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            dis_x += rand_x
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if not self.flat:
                subterrain.height_field_raw[dis_x - gap_size // 2 : dis_x + gap_size // 2, :] = gap_depth

            subterrain.height_field_raw[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = gap_depth
            subterrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width :] = gap_depth

            last_dis_x = dis_x
            goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
        self._finish_course(subterrain, goals, final_dis_x, mid_y)
        _pad_borders(subterrain, self.pad_width, self.pad_height)
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class ParkourHurdleTerrain(RoughTerrainGenerator):
    platform_len: float = 2.5
    platform_height: float = 0.0
    num_stones: int = 8
    stone_len: float = 0.3
    x_range: tuple[float, float] = (1.5, 2.4)
    y_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width: tuple[float, float] = (0.4, 0.8)
    hurdle_height_range: tuple[float, float] = (0.2, 0.3)
    pad_width: float = 0.1
    pad_height: float = 0.5
    flat: bool = False

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        goals = np.zeros((self.num_stones + 2, 2))
        mid_y = subterrain.height_field_raw.shape[1] // 2

        dis_x_min = round(self.x_range[0] / subterrain.horizontal_scale)
        dis_x_max = round(self.x_range[1] / subterrain.horizontal_scale)
        dis_y_min = round(self.y_range[0] / subterrain.horizontal_scale)
        dis_y_max = round(self.y_range[1] / subterrain.horizontal_scale)

        half_valid_width = round(
            np.random.uniform(
                self.half_valid_width[0],
                self.half_valid_width[1],
            )
            / subterrain.horizontal_scale
        )
        hurdle_height_max = round(self.hurdle_height_range[1] / subterrain.vertical_scale)
        hurdle_height_min = round(self.hurdle_height_range[0] / subterrain.vertical_scale)

        platform_len = round(self.platform_len / subterrain.horizontal_scale)
        platform_height = round(self.platform_height / subterrain.vertical_scale)
        subterrain.height_field_raw[0:platform_len, :] = platform_height

        stone_len = round(self.stone_len / subterrain.horizontal_scale)
        # Keep the hurdle at least one cell wide; at coarse horizontal_scale a thin
        # hurdle (e.g. 0.3 m) rounds to stone_len // 2 == 0 and draws nothing.
        half_stone_len = max(stone_len // 2, 1)

        dis_x = platform_len
        goals[0] = [platform_len - 1, mid_y]
        for i in range(self.num_stones):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            dis_x += rand_x
            if not self.flat:
                subterrain.height_field_raw[dis_x - half_stone_len : dis_x + half_stone_len, :] = np.random.randint(
                    hurdle_height_min, hurdle_height_max
                )
                subterrain.height_field_raw[
                    dis_x - half_stone_len : dis_x + half_stone_len,
                    : mid_y + rand_y - half_valid_width,
                ] = 0
                subterrain.height_field_raw[
                    dis_x - half_stone_len : dis_x + half_stone_len,
                    mid_y + rand_y + half_valid_width :,
                ] = 0
            goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
        self._finish_course(subterrain, goals, final_dis_x, mid_y)
        _pad_borders(subterrain, self.pad_width, self.pad_height)
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class ParkourStepTerrain(RoughTerrainGenerator):
    platform_len: float = 2.5
    platform_height: float = 0.0
    num_stones: int = 8
    x_range: tuple[float, float] = (0.2, 0.4)
    y_range: tuple[float, float] = (-0.15, 0.15)
    half_valid_width: tuple[float, float] = (0.45, 0.5)
    step_height: float = 0.2
    pad_width: float = 0.1
    pad_height: float = 0.5

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        goals = np.zeros((self.num_stones + 2, 2))
        mid_y = subterrain.height_field_raw.shape[1] // 2

        dis_x_min = round((self.x_range[0] + self.step_height) / subterrain.horizontal_scale)
        dis_x_max = round((self.x_range[1] + self.step_height) / subterrain.horizontal_scale)
        dis_y_min = round(self.y_range[0] / subterrain.horizontal_scale)
        dis_y_max = round(self.y_range[1] / subterrain.horizontal_scale)
        # Keep randint bounds valid after rounding small ranges.
        if dis_x_max <= dis_x_min:
            dis_x_max = dis_x_min + 1
        if dis_y_max <= dis_y_min:
            dis_y_max = dis_y_min + 1

        step_height = round(self.step_height / subterrain.vertical_scale)
        half_valid_width = round(
            np.random.uniform(
                self.half_valid_width[0],
                self.half_valid_width[1],
            )
            / subterrain.horizontal_scale
        )

        platform_len = round(self.platform_len / subterrain.horizontal_scale)
        platform_height = round(self.platform_height / subterrain.vertical_scale)
        subterrain.height_field_raw[0:platform_len, :] = platform_height

        dis_x = platform_len
        last_dis_x = dis_x
        stair_height = 0
        goals[0] = [platform_len - round(1 / subterrain.horizontal_scale), mid_y]
        for i in range(self.num_stones):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if i < self.num_stones // 2:
                stair_height += step_height
            elif i > self.num_stones // 2:
                stair_height -= step_height
            subterrain.height_field_raw[dis_x : dis_x + rand_x, :] = stair_height
            dis_x += rand_x
            subterrain.height_field_raw[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = 0
            subterrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width :] = 0

            last_dis_x = dis_x
            goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
        self._finish_course(subterrain, goals, final_dis_x, mid_y)
        _pad_borders(subterrain, self.pad_width, self.pad_height)
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class ParkourDemoTerrain(RoughTerrainGenerator):
    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        goals = np.zeros((8, 2))
        mid_y = subterrain.height_field_raw.shape[1] // 2

        platform_length = round(2 / subterrain.horizontal_scale)
        hurdle_depth = round(np.random.uniform(0.35, 0.4) / subterrain.horizontal_scale)
        hurdle_height = round(np.random.uniform(0.3, 0.36) / subterrain.vertical_scale)
        hurdle_width = round(np.random.uniform(1, 1.2) / subterrain.horizontal_scale)
        goals[0] = [platform_length + hurdle_depth / 2, mid_y]
        subterrain.height_field_raw[
            platform_length : platform_length + hurdle_depth,
            round(mid_y - hurdle_width / 2) : round(mid_y + hurdle_width / 2),
        ] = hurdle_height

        platform_length += round(np.random.uniform(1.5, 2.5) / subterrain.horizontal_scale)
        first_step_depth = round(np.random.uniform(0.45, 0.8) / subterrain.horizontal_scale)
        first_step_height = round(np.random.uniform(0.35, 0.45) / subterrain.vertical_scale)
        first_step_width = round(np.random.uniform(1, 1.2) / subterrain.horizontal_scale)
        goals[1] = [platform_length + first_step_depth / 2, mid_y]
        subterrain.height_field_raw[
            platform_length : platform_length + first_step_depth,
            round(mid_y - first_step_width / 2) : round(mid_y + first_step_width / 2),
        ] = first_step_height

        platform_length += first_step_depth
        second_step_depth = round(np.random.uniform(0.45, 0.8) / subterrain.horizontal_scale)
        second_step_height = first_step_height
        second_step_width = first_step_width
        goals[2] = [platform_length + second_step_depth / 2, mid_y]
        subterrain.height_field_raw[
            platform_length : platform_length + second_step_depth,
            round(mid_y - second_step_width / 2) : round(mid_y + second_step_width / 2),
        ] = second_step_height

        platform_length += second_step_depth
        gap_size = round(np.random.uniform(0.5, 0.8) / subterrain.horizontal_scale)

        platform_length += gap_size
        third_step_depth = round(np.random.uniform(0.25, 0.6) / subterrain.horizontal_scale)
        third_step_height = first_step_height
        third_step_width = round(np.random.uniform(1, 1.2) / subterrain.horizontal_scale)
        goals[3] = [platform_length + third_step_depth / 2, mid_y]
        subterrain.height_field_raw[
            platform_length : platform_length + third_step_depth,
            round(mid_y - third_step_width / 2) : round(mid_y + third_step_width / 2),
        ] = third_step_height

        platform_length += third_step_depth
        forth_step_depth = round(np.random.uniform(0.25, 0.6) / subterrain.horizontal_scale)
        forth_step_height = first_step_height
        forth_step_width = third_step_width
        goals[4] = [platform_length + forth_step_depth / 2, mid_y]
        subterrain.height_field_raw[
            platform_length : platform_length + forth_step_depth,
            round(mid_y - forth_step_width / 2) : round(mid_y + forth_step_width / 2),
        ] = forth_step_height

        platform_length += forth_step_depth
        gap_size = round(np.random.uniform(0.1, 0.4) / subterrain.horizontal_scale)
        platform_length += gap_size

        left_y = mid_y + round(np.random.uniform(0.15, 0.3) / subterrain.horizontal_scale)
        right_y = mid_y - round(np.random.uniform(0.15, 0.3) / subterrain.horizontal_scale)

        slope_height = round(np.random.uniform(0.15, 0.22) / subterrain.vertical_scale)
        slope_depth = round(np.random.uniform(0.75, 0.85) / subterrain.horizontal_scale)
        slope_width = round(1.0 / subterrain.horizontal_scale)

        platform_height = slope_height + np.random.randint(0, 0.2 / subterrain.vertical_scale)

        goals[5] = [platform_length + slope_depth / 2, left_y]
        heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1))
        _set_height_block(
            subterrain,
            platform_length,
            platform_length + slope_depth,
            left_y - slope_width // 2,
            left_y + slope_width // 2,
            heights.astype(int) + platform_height,
        )

        platform_length += slope_depth + gap_size
        goals[6] = [platform_length + slope_depth / 2, right_y]
        heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
        _set_height_block(
            subterrain,
            platform_length,
            platform_length + slope_depth,
            right_y - slope_width // 2,
            right_y + slope_width // 2,
            heights.astype(int) + platform_height,
        )

        platform_length += slope_depth + gap_size + round(0.4 / subterrain.horizontal_scale)
        self._finish_course(subterrain, goals, platform_length, left_y)
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class HalfSlopedTerrain(TerrainGenerator):
    wall_width: float = 4.0
    start2center: float = 0.7
    max_height: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        num_x = subterrain.height_field_raw.shape[0]
        wall_width_int = max(int(self.wall_width / subterrain.horizontal_scale), 1)
        max_height_int = int(self.max_height / subterrain.vertical_scale)
        slope_start = int(self.start2center / subterrain.horizontal_scale + num_x // 2)
        height2width_ratio = max_height_int / wall_width_int
        xs = np.arange(slope_start, num_x)
        heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
        subterrain.height_field_raw[slope_start:num_x, :] = heights[:, None]
        subterrain.slope_vector = np.array(
            [wall_width_int * subterrain.horizontal_scale, 0.0, self.max_height],
            dtype=np.float32,
        )
        subterrain.slope_vector /= np.linalg.norm(subterrain.slope_vector)
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class HalfPlatformTerrain(TerrainGenerator):
    start2center: float = 2.0
    max_height: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        num_x, num_y = subterrain.height_field_raw.shape
        max_height_int = int(self.max_height / subterrain.vertical_scale)
        offset = int(self.start2center / subterrain.horizontal_scale)
        slope_start_x = offset + num_x // 2
        slope_start_y = offset + num_y // 2
        subterrain.height_field_raw[:, :] = max_height_int
        subterrain.height_field_raw[-slope_start_x:slope_start_x, -slope_start_y:slope_start_y] = 0
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class CustomSteppingStonesTerrain(TerrainGenerator):
    stone_size: float = 1.0
    stone_distance: float = 1.0
    max_height: float = 0.2
    platform_size: float = 1.0
    depth: float = -1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        def get_rand_dis_int(scale: float) -> int:
            low = int(-scale / subterrain.horizontal_scale + 1)
            high = int(scale / subterrain.horizontal_scale)
            if high <= low:
                high = low + 1
            return np.random.randint(low, high)

        stone_size = int(self.stone_size / subterrain.horizontal_scale)
        stone_distance = int(self.stone_distance / subterrain.horizontal_scale)
        max_height = int(self.max_height / subterrain.vertical_scale)
        platform_size = int(self.platform_size / subterrain.horizontal_scale)
        height_range = np.arange(-max_height - 1, max_height, step=1)

        start_x = 0
        start_y = 0
        subterrain.height_field_raw[:, :] = int(self.depth / subterrain.vertical_scale)
        if subterrain.length >= subterrain.width:
            while start_y < subterrain.length:
                stop_y = min(subterrain.length, start_y + stone_size)
                start_x = np.random.randint(0, stone_size)
                stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
                subterrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(height_range)
                while start_x < subterrain.width:
                    stop_x = min(subterrain.width, start_x + stone_size)
                    subterrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                    start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
                start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
        elif subterrain.width > subterrain.length:
            while start_x < subterrain.width:
                stop_x = min(subterrain.width, start_x + stone_size)
                start_y = np.random.randint(0, stone_size)
                stop_y = max(0, start_y - stone_distance)
                subterrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(height_range)
                while start_y < subterrain.length:
                    stop_y = min(subterrain.length, start_y + stone_size)
                    subterrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                    start_y += stone_size + stone_distance
                start_x += stone_size + stone_distance

        x1 = (subterrain.width - platform_size) // 2
        x2 = (subterrain.width + platform_size) // 2
        y1 = (subterrain.length - platform_size) // 2
        y2 = (subterrain.length + platform_size) // 2
        subterrain.height_field_raw[x1:x2, y1:y2] = 0
        return subterrain
