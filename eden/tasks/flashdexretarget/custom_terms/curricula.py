"""FlashDexRetarget's adaptive assist curriculum."""

from __future__ import annotations

from collections import deque
from typing import ClassVar

import numpy as np
import torch

import eden as en
from eden.managers.curriculum_manager import CURRICULUM_TERM_REGISTRY, CurriculumTerm
from eden.tasks.flashdexretarget.custom_terms.actions import FDRObjectAssist


@CURRICULUM_TERM_REGISTRY.register()
class FDRAssistCurriculum(CurriculumTerm):
    """FlashDexRetarget's adaptive object-assist schedule (``xfrc_curriculum_adaptive``).

    Every env step it accumulates the mean weighted reward of each group of ``reward_terms`` and counts the episodes
    that ended by timing out (completed) or otherwise; every ``num_steps_per_env`` steps (an epoch) it records the
    epoch means. Once every group's mean over the last ``deque_len`` epochs reaches its ``rew_thresholds`` entry,
    the completion rate reaches ``completion_thres``, ``wait_epochs`` have passed and ``decay_cooldown_epochs`` have
    passed since the last change, it multiplies the gains' upper bound by ``upper_ratios`` (lower bound = upper *
    ``lower_ratios``); below ``kp_zero_thres`` they drop to zero. When instead the mean completion rate over the last
    ``deque_len`` epochs is under ``dialback_completion_thres`` and more than ``dialback_min_epochs`` have passed
    since the last change, it returns to the gains before the last decay scaled by ``dialback_ratios``, including
    from zero gains; once there, a still-low completion rate holds them rather than lowering them further. Each
    env's gains are drawn uniformly in ``[lower, upper]`` at every change.

    The dial-back is checked on its own. FlashDexRetarget checks it only once the decay gates have passed, which
    need a completion rate of at least ``completion_thres`` (0.65), above the dial-back's ceiling (0.5): its
    dial-back never fires.

    An env built with ``eval_mode=True`` (``eden inference``) starts, and stays, at zero gains: the assist is a
    training aid, so an evaluation grades the policy without it.

    Parameters
    ----------
    assist_term : str
        The :class:`FDRObjectAssist` action term.
    reward_terms : dict[str, list[str]]
        Gate group -> its reward terms (absent terms are skipped).
    rew_thresholds : dict[str, float]
        Gate group -> minimum mean weighted reward per step.
    kp_init, kv_init, force_range : float
        Starting gains and wrench clamp.
    wait_epochs, deque_len, num_steps_per_env, decay_cooldown_epochs, dialback_min_epochs : int
        Schedule timing, in epochs (``num_steps_per_env`` env steps each).
    completion_thres, dialback_completion_thres, kp_zero_thres, kp_lower_zero_thres : float
        Gates.
    upper_ratios, lower_ratios, dialback_ratios : dict[str, float]
        Per gain (``kp``, ``kv``, ``fr``) factors.
    """

    assist_term: str = "assist"
    reward_terms: dict[str, list[str]] = {}
    rew_thresholds: dict[str, float] = {}
    kp_init: float = 80.0
    kv_init: float = 5.0
    force_range: float = 50.0
    wait_epochs: int = 100
    deque_len: int = 30
    num_steps_per_env: int = 32
    decay_cooldown_epochs: int = 40
    kp_zero_thres: float = 0.05
    kp_lower_zero_thres: float = 0.1
    completion_thres: float = 0.65
    dialback_completion_thres: float = 0.5
    dialback_min_epochs: int = 500
    upper_ratios: dict[str, float] = {"kp": 0.9, "kv": 0.9, "fr": 1.0}
    lower_ratios: dict[str, float] = {"kp": 0.8, "kv": 0.8, "fr": 1.0}
    dialback_ratios: dict[str, float] = {"kp": 0.98, "kv": 0.98, "fr": 0.98}

    _GAINS: ClassVar[tuple[str, ...]] = ("kp", "kv", "fr")

    def build(self) -> None:
        self._assist = self._env.action_manager.get_term(self.assist_term)
        if not isinstance(self._assist, FDRObjectAssist):
            raise TypeError(
                f"FDRAssistCurriculum drives an FDRObjectAssist; {self.assist_term!r} is {type(self._assist).__name__}."
            )
        rm = self._env.reward_manager
        names = list(rm.active_terms)
        self._groups = {k: [names.index(t) for t in terms if t in names] for k, terms in self.reward_terms.items()}
        unknown = [t for terms in self.reward_terms.values() for t in terms if t not in names]
        if unknown and all(not v for v in self._groups.values()):
            raise ValueError(f"FDRAssistCurriculum: no reward term of {unknown} is active; active: {names}.")
        # The assist is a training aid: an eval env (inference, a held-out evaluation) grades the policy alone, with
        # no wrench on the objects, rather than at kp_init, the level the curriculum manager would otherwise freeze.
        training = self._env.training
        self.upper = {
            "kp": self.kp_init if training else 0.0,
            "kv": self.kv_init if training else 0.0,
            "fr": self.force_range if training else 0.0,
        }
        self.lower = dict(self.upper)
        self.history: list[tuple[dict, dict]] = []
        self.rew_deques = {k: deque(maxlen=self.deque_len) for k in self._groups}
        self.completion = deque(maxlen=self.deque_len)
        self.epochs_since_change = 0
        self._epoch = -1
        self._sum = torch.zeros(len(self._groups), device=self.device)
        self._steps = 0
        self._done_ok = torch.zeros((), dtype=torch.long, device=self.device)
        self._done_bad = torch.zeros((), dtype=torch.long, device=self.device)
        self._push_gains()

    def _push_gains(self) -> None:
        def draw(k: str) -> torch.Tensor:
            if self.upper[k] <= 0.0:
                return torch.zeros(self.num_envs, device=self.device)
            return torch.rand(self.num_envs, device=self.device) * (self.upper[k] - self.lower[k]) + self.lower[k]

        self._assist.set_gains(draw("kp"), draw("kv"), self.upper["fr"] if self.upper["fr"] > 0.0 else None)

    def compute(self) -> dict[str, float]:
        env = self._env
        step = env.reward_manager._step_reward
        for g, cols in enumerate(self._groups.values()):
            if cols:
                self._sum[g] += step[:, cols].sum(-1).mean()
        self._steps += 1
        tm = env.termination_manager
        self._done_ok += tm.timeouts.sum()
        self._done_bad += (tm.terminated & ~tm.timeouts).sum()

        epoch = int(env.common_step_counter) // self.num_steps_per_env
        if epoch != self._epoch and self._steps:
            self._epoch = epoch
            means = (self._sum / self._steps).tolist()
            ok, bad = int(self._done_ok), int(self._done_bad)
            for (k, cols), m in zip(self._groups.items(), means):
                if cols:
                    self.rew_deques[k].append(m)
            self.completion.append(ok / (ok + bad) if ok + bad else 1.0)
            self._sum.zero_()
            self._steps = 0
            self._done_ok.zero_()
            self._done_bad.zero_()
            self._update(epoch)
        return {
            "kp_upper": self.upper["kp"],
            "kp_lower": self.lower["kp"],
            "kv_upper": self.upper["kv"],
            "force_range": self.upper["fr"],
            "completion_rate": float(np.mean(self.completion)) if self.completion else 0.0,
            "epochs_since_change": float(self.epochs_since_change),
        }

    def _gates_pass(self, epoch: int) -> bool:
        if epoch < self.wait_epochs or self.epochs_since_change < self.decay_cooldown_epochs:
            return False
        for k, cols in self._groups.items():
            if not cols or k not in self.rew_thresholds:
                continue
            d = self.rew_deques[k]
            if len(d) < self.deque_len or float(np.mean(d)) < self.rew_thresholds[k]:
                return False
        return len(self.completion) >= self.deque_len and float(np.mean(self.completion)) >= self.completion_thres

    def _dialback_due(self) -> bool:
        return (
            bool(self.history)
            and len(self.completion) >= self.deque_len
            and float(np.mean(self.completion)) < self.dialback_completion_thres
            and self.epochs_since_change > self.dialback_min_epochs
        )

    def _update(self, epoch: int) -> None:
        self.epochs_since_change += 1
        dialback = self._dialback_due()
        if dialback:
            # Back to the gains before the last decay, scaled. The same target every time, so a dial-back that has
            # already happened holds its gains instead of compounding the ratios towards zero.
            upper, lower = self.history[-1]
            target = (
                {k: upper[k] * self.dialback_ratios.get(k, 1.0) for k in self._GAINS},
                {k: lower[k] * self.dialback_ratios.get(k, 1.0) for k in self._GAINS},
            )
            if target == (self.upper, self.lower):
                return
            self.upper, self.lower = target
        elif any(self.upper[k] > 0.0 for k in self._GAINS) and self._gates_pass(epoch):
            self.history.append((dict(self.upper), dict(self.lower)))
            self.upper = {k: self.upper[k] * self.upper_ratios.get(k, 1.0) for k in self._GAINS}
            self.lower = {k: self.upper[k] * self.lower_ratios.get(k, 1.0) for k in self._GAINS}
            if self.upper["kp"] < self.kp_zero_thres:
                self.upper = dict.fromkeys(self._GAINS, 0.0)
                self.lower = dict.fromkeys(self._GAINS, 0.0)
            elif self.lower["kp"] < self.kp_lower_zero_thres:
                self.lower = dict.fromkeys(self._GAINS, 0.0)
        else:
            return
        en.logger.info(
            f"[FDRAssistCurriculum] epoch {epoch}: {'dial-back' if dialback else 'decay'} -> "
            f"kp [{self.lower['kp']:.3g}, {self.upper['kp']:.3g}], kv [{self.lower['kv']:.3g}, {self.upper['kv']:.3g}]"
        )
        self._push_gains()
        for d in self.rew_deques.values():
            d.clear()
        self.completion.clear()
        self.epochs_since_change = 0
