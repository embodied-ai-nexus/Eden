"""Scripted demonstrators for the painting tasks (``eden datagen --task art_painting_sunset``, …).

One privileged :class:`ScriptedPainter` per task, registered under the task's name in
:data:`~eden.extensions.scripted_demos.registry.SCRIPTED_DEMO_REGISTRY`; it is also the proof that each work can be
painted (``tests/tasks/art/painting/test_scripted_demos.py``). It has no randomness and makes no deliberate mistakes,
so ``seed`` changes nothing it does and ``p_mistake`` must be 0. An ``eden datagen`` episode ends when the task does:
on success, which can come before the last stroke is finished.

The painter follows the work's tip path (``references.Painting.path``: every stroke, and a dip into the right well
before each stroke the plan dips for) through the task's own action, with the art suite's
:class:`~eden.tasks.art.scripted.TipFollower`: timed at ``draw_speed`` on and near the paper and in a dip (by
default the paintbrush's :attr:`~eden.tasks.art.workcell.ToolSpec.draw_speed`) and ``travel_speed`` between,
from and back to the arm's home; in a stroke or a dip the tip may trail the plan by ``careful_lag`` before the plan
waits for it, and by ``max_lag`` elsewhere.
"""

from __future__ import annotations

import numpy as np
import torch

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.managers.terms.utils import resolve_metric_term
from eden.tasks.art.painting.custom_terms import PaintingCopy
from eden.tasks.art.painting.references import TRAVEL_SPEED
from eden.tasks.art import workcell
from eden.tasks.art.plans import STOP
from eden.tasks.art.scripted import TipFollower, plan_timeline


class ScriptedPainter:
    """Batched privileged painter for a built painting env (``PaintingConfig`` and its tasks).

    ``act()`` returns the ``(num_envs, 6)`` action for the env's ``arm`` term; call ``reset(envs_idx)`` for the envs
    the env reset (all of them by default). ``done`` is True for an env whose plan has run out. For ``eden datagen``
    it is a demonstrator: :meth:`update_subgoals` returns ``done``, and :attr:`subgoal_counts` counts the strokes
    finished and the dips made.
    """

    def __init__(
        self,
        env,
        *,
        metric_name: str = "painting",
        action_name: str = "arm",
        draw_speed: float | None = None,
        travel_speed: float = TRAVEL_SPEED,
        max_lag: float = 0.008,
        careful_lag: float = 0.001,
        gain: float = 2.5,
        turn_radius: float = 0.005,
    ) -> None:
        self.env = env
        self.painting = resolve_metric_term(env, term="ScriptedPainter", metric_name=metric_name, expected=PaintingCopy)
        if draw_speed is None:
            draw_speed = workcell.TOOLS["paintbrush"].draw_speed
        path, owner = self.painting.work_spec.path()
        goal, careful, step_owner = plan_timeline(
            path,
            owner,
            draw_speed=draw_speed,
            travel_speed=travel_speed,
            dt=env.dt,
            home=(0.0, 0.0, -workcell.HOME_STANDOFF),
            turn_radius=turn_radius,
        )
        self.follower = TipFollower(
            env,
            self.painting,
            goal,
            careful,
            action_name=action_name,
            max_lag=max_lag,
            careful_lag=careful_lag,
            gain=gain,
            demonstrator="ScriptedPainter",
        )
        # Where each stroke and each dip ends: the first step off a run of its steps.
        ends = np.flatnonzero(step_owner[:-1] != step_owner[1:]) + 1
        finished = step_owner[ends - 1]
        device = env.device
        self._stroke_ends = torch.as_tensor(ends[finished >= 0], dtype=torch.long, device=device)
        self._dip_ends = torch.as_tensor(ends[finished == STOP], dtype=torch.long, device=device)
        self._counts = torch.zeros(2, dtype=torch.long, device=device)

    @property
    def clock(self) -> torch.Tensor:
        return self.follower.clock

    @property
    def done(self) -> torch.Tensor:
        return self.follower.done

    @property
    def subgoal_counts(self) -> dict[str, int]:
        """The strokes finished and the dips made so far, over every env and episode (``eden datagen``'s report)."""
        strokes, dips = self._counts.tolist()
        return {"stroke": strokes, "dip": dips}

    def update_subgoals(self) -> torch.Tensor:
        """``eden datagen``'s per-step check: :attr:`done` (the plan advances in :meth:`act`)."""
        return self.done

    def reset(self, envs_idx=None) -> None:
        self.follower.reset(envs_idx)

    def act(self) -> torch.Tensor:
        before = self.follower.clock.clone()
        action = self.follower.act()
        after = self.follower.clock[:, None]
        for k, ends in enumerate((self._stroke_ends, self._dip_ends)):
            self._counts[k] += ((before[:, None] < ends) & (ends <= after)).sum()
        return action


def _painter(env, task: str, *, p_mistake: float) -> ScriptedPainter:
    if p_mistake != 0.0:
        raise ValueError(
            f"The scripted demonstrator of '{task}' makes no deliberate mistakes, so p_mistake={p_mistake} cannot be "
            "honoured. Generate with --p_mistake 0 (or leave it out)."
        )
    return ScriptedPainter(env)


@SCRIPTED_DEMO_REGISTRY.register(name="art_painting_sunset")
def paint_sunset(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedPainter:
    """The sunset, stroke by stroke in the plan's order. ``seed`` changes nothing: the painter is exact."""
    return _painter(env, "art_painting_sunset", p_mistake=p_mistake)


@SCRIPTED_DEMO_REGISTRY.register(name="art_painting_fuji_study")
def paint_fuji_study(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedPainter:
    """The Fuji study, layer by layer in the plan's order. ``seed`` changes nothing: the painter is exact."""
    return _painter(env, "art_painting_fuji_study", p_mistake=p_mistake)


@SCRIPTED_DEMO_REGISTRY.register(name="art_painting_fuji")
def paint_fuji(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedPainter:
    """The Fuji, layer by layer in the plan's order. ``seed`` changes nothing: the painter is exact."""
    return _painter(env, "art_painting_fuji", p_mistake=p_mistake)


# Each task's own episode length (``config.py``), which the painter's plan fits in: the generator's default is 90 s.
paint_sunset.episode_length_s = 90.0
paint_fuji_study.episode_length_s = 800.0
paint_fuji.episode_length_s = 1200.0
