"""Scripted demonstrators for the calligraphy tasks (``eden datagen --task art_calligraphy_ei``, …).

One privileged :class:`ScriptedWriter` per task, registered under the task's name in
:data:`~eden.extensions.scripted_demos.registry.SCRIPTED_DEMO_REGISTRY`; it is also the proof that each work can be
written (``tests/tasks/art/calligraphy/test_scripted_demos.py``). It has no randomness and makes no deliberate
mistakes, so ``seed`` changes nothing it does and ``p_mistake`` must be 0. An ``eden datagen`` episode ends when the
task does: on success, which can come before the last stroke is finished.

The writer follows the work's tip path (``references.Plan.path``) through the task's own action, with the art suite's
:class:`~eden.tasks.art.scripted.TipFollower`: timed at ``draw_speed`` on and near the paper (by default its
tool's :attr:`~eden.tasks.art.workcell.ToolSpec.draw_speed`) and ``travel_speed`` between strokes, from and back
to the arm's home; within a stroke the tip may trail the plan by ``write_lag`` before the plan waits for it, and by
``max_lag`` elsewhere.
"""

from __future__ import annotations

import numpy as np
import torch

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.managers.terms.utils import resolve_metric_term
from eden.tasks.art.calligraphy.custom_terms import CalligraphyWriting
from eden.tasks.art import workcell
from eden.tasks.art.scripted import TipFollower, plan_timeline


class ScriptedWriter:
    """Batched privileged writer for a built calligraphy env (``CalligraphyConfig`` and its tasks).

    ``act()`` returns the ``(num_envs, 6)`` action for the env's ``arm`` term; call ``reset(envs_idx)`` for the envs
    the env reset (all of them by default). ``done`` is True for an env whose plan has run out. For ``eden datagen``
    it is a demonstrator: :meth:`update_subgoals` returns ``done``, and :attr:`subgoal_counts` counts the strokes
    finished.
    """

    def __init__(
        self,
        env,
        *,
        metric_name: str = "writing",
        action_name: str = "arm",
        draw_speed: float | None = None,
        travel_speed: float = 0.08,
        max_lag: float = 0.008,
        write_lag: float = 0.0008,
        gain: float = 2.5,
        turn_radius: float = 0.005,
    ) -> None:
        self.env = env
        self.writing = resolve_metric_term(
            env, term="ScriptedWriter", metric_name=metric_name, expected=CalligraphyWriting
        )
        work = self.writing.work_spec
        if draw_speed is None:
            draw_speed = workcell.TOOLS[work.plan.tool].draw_speed
        path, stroke_of = work.plan.path()
        goal, writing, _ = plan_timeline(
            path,
            stroke_of,
            draw_speed=draw_speed,
            travel_speed=travel_speed,
            dt=env.dt,
            home=(0.0, 0.0, -workcell.HOME_STANDOFF),
            turn_radius=turn_radius,
        )
        self.follower = TipFollower(
            env,
            self.writing,
            goal,
            writing,
            action_name=action_name,
            max_lag=max_lag,
            careful_lag=write_lag,
            gain=gain,
            demonstrator="ScriptedWriter",
        )
        # Where each stroke ends: the first step off a run of steps in a stroke.
        self._stroke_ends = torch.as_tensor(
            np.flatnonzero(writing[:-1] & ~writing[1:]) + 1, dtype=torch.long, device=env.device
        )
        self._strokes_finished = torch.zeros((), dtype=torch.long, device=env.device)

    @property
    def clock(self) -> torch.Tensor:
        return self.follower.clock

    @property
    def done(self) -> torch.Tensor:
        return self.follower.done

    @property
    def subgoal_counts(self) -> dict[str, int]:
        """The strokes finished so far, over every env and episode (``eden datagen``'s report)."""
        return {"stroke": int(self._strokes_finished)}

    def update_subgoals(self) -> torch.Tensor:
        """``eden datagen``'s per-step check: :attr:`done` (the plan advances in :meth:`act`)."""
        return self.done

    def reset(self, envs_idx=None) -> None:
        self.follower.reset(envs_idx)

    def act(self) -> torch.Tensor:
        before = self.follower.clock.clone()
        action = self.follower.act()
        passed = (before[:, None] < self._stroke_ends) & (self._stroke_ends <= self.follower.clock[:, None])
        self._strokes_finished += passed.sum()
        return action


def _writer(env, task: str, *, p_mistake: float) -> ScriptedWriter:
    if p_mistake != 0.0:
        raise ValueError(
            f"The scripted demonstrator of '{task}' makes no deliberate mistakes, so p_mistake={p_mistake} cannot be "
            "honoured. Generate with --p_mistake 0 (or leave it out)."
        )
    return ScriptedWriter(env)


@SCRIPTED_DEMO_REGISTRY.register(name="art_calligraphy_dai")
def write_dai(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedWriter:
    """大 with the brush, stroke by stroke in the plan's order. ``seed`` changes nothing: the writer is exact."""
    return _writer(env, "art_calligraphy_dai", p_mistake=p_mistake)


@SCRIPTED_DEMO_REGISTRY.register(name="art_calligraphy_san")
def write_san(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedWriter:
    """山 with the brush, stroke by stroke in the plan's order. ``seed`` changes nothing: the writer is exact."""
    return _writer(env, "art_calligraphy_san", p_mistake=p_mistake)


@SCRIPTED_DEMO_REGISTRY.register(name="art_calligraphy_ei")
def write_ei(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedWriter:
    """永 with the brush, stroke by stroke in the plan's order. ``seed`` changes nothing: the writer is exact."""
    return _writer(env, "art_calligraphy_ei", p_mistake=p_mistake)


@SCRIPTED_DEMO_REGISTRY.register(name="art_calligraphy_lettering_minimum")
def write_minimum(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedWriter:
    """*minimum* with the pointed pen, letter by letter, then the dots. ``seed`` changes nothing."""
    return _writer(env, "art_calligraphy_lettering_minimum", p_mistake=p_mistake)


# Each task's own episode length (``config.py``), which the writer's plan fits in: the generator's default is 90 s.
write_dai.episode_length_s = 60.0
write_san.episode_length_s = 60.0
write_ei.episode_length_s = 60.0
write_minimum.episode_length_s = 120.0
