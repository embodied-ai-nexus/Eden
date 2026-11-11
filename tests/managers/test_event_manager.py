"""Tests for ``EventManager``: interval countdowns (per env and global), the reset-mode throttle, the
per-term timer bookkeeping across mixed class and function terms, and which reset terms fire as one batch.

A class term counts its fires per env; the function term lifts the env's box by 1 m, so a fire of
either is visible in real state.
"""

from __future__ import annotations

import pytest
import torch

from eden.constants import EventMode
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventManager, EventTerm
from eden.options.managers.events import EventManagerOptions, EventTermOptions

NUM_ENVS = 4


@EVENT_TERM_REGISTRY.register()
class CountingEvent(EventTerm):
    """Counts how often it fired in each env."""

    def build(self) -> None:
        self.fires = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def compute(self, envs_idx=None) -> None:
        self.fires[slice(None) if envs_idx is None else envs_idx] += 1


@EVENT_TERM_REGISTRY.register()
def lift_box(env, *, envs_idx=None) -> None:
    """Raise the box of every selected env by 1 m."""
    pos = env.entities["box"].get_pos()
    pos[slice(None) if envs_idx is None else envs_idx, 2] += 1.0
    env.entities["box"].set_pos(pos)


def _interval(name: str, seconds: float, priority: int = 0, **kwargs) -> EventTermOptions:
    return EventTermOptions(
        name=name, mode=EventMode.INTERVAL, interval_range_s=(seconds, seconds), priority=priority, **kwargs
    )


def _lifts(env) -> list[float]:
    """How often ``lift_box`` fired in each env, from the box height (set to 0 by the test)."""
    return env.entities["box"].get_pos()[:, 2].tolist()


def test_a_reset_resamples_the_class_terms_timers_and_leaves_the_function_terms_alone(box_env, set_box_heights):
    """Three interval terms, class@0, func@1, class@2: each keeps its own countdown through a reset.

    Positional coupling between two separately sorted lists once resampled the function term's
    countdown in place of the class term's.
    """
    set_box_heights(box_env, [0.0] * NUM_ENVS)
    manager = EventManager(
        env=box_env,
        options=EventManagerOptions(
            class_a=_interval("counting_event", 10.0, priority=0),
            func_b=_interval("lift_box", 50.0, priority=1),
            class_c=_interval("counting_event", 30.0, priority=2),
        ),
    )
    class_a, class_c = (term for mode, name, term in manager.iter_terms() if name.startswith("class"))

    manager.compute(EventMode.INTERVAL, dt=25.0)  # class_a fires (and restarts at 10); 25 s and 5 s left for the others
    manager.reset(envs_idx=slice(None))  # class_a back to 10 s, class_c back to 30 s; func_b keeps its 25 s
    manager.compute(EventMode.INTERVAL, dt=26.0)

    assert class_a.fires.tolist() == [2] * NUM_ENVS
    assert _lifts(box_env) == [1.0] * NUM_ENVS
    assert class_c.fires.tolist() == [0] * NUM_ENVS


def test_a_partial_reset_restarts_only_the_given_envs_countdown(box_env):
    manager = EventManager(env=box_env, options=EventManagerOptions(tick=_interval("counting_event", 5.0)))
    term = next(term for _, _, term in manager.iter_terms())

    manager.compute(EventMode.INTERVAL, dt=3.0)  # 2 s left everywhere
    manager.reset(envs_idx=torch.tensor([0, 2]))  # 5 s again in envs 0 and 2
    manager.compute(EventMode.INTERVAL, dt=2.5)

    assert term.fires.tolist() == [0, 1, 0, 1]


def test_a_global_time_interval_fires_every_env_at_once_and_restarts(box_env):
    manager = EventManager(
        env=box_env, options=EventManagerOptions(tick=_interval("counting_event", 10.0, is_global_time=True))
    )
    term = next(term for _, _, term in manager.iter_terms())

    manager.compute(EventMode.INTERVAL, dt=6.0)
    assert term.fires.tolist() == [0] * NUM_ENVS
    manager.compute(EventMode.INTERVAL, dt=6.0)
    assert term.fires.tolist() == [1] * NUM_ENVS
    manager.compute(EventMode.INTERVAL, dt=6.0)  # restarted at 10 s: 4 s left
    assert term.fires.tolist() == [1] * NUM_ENVS


def test_reset_terms_throttle_on_their_own_min_step_count(box_env):
    """Each term's throttle bookkeeping is its own: 2 steps for one, 10 for the other."""
    manager = EventManager(
        env=box_env,
        options=EventManagerOptions(
            fast=CountingEvent.configure(mode=EventMode.RESET, min_step_count_between_reset=2, priority=0),
            slow=CountingEvent.configure(mode=EventMode.RESET, min_step_count_between_reset=10, priority=1),
        ),
    )
    fast, slow = (term for _, _, term in manager.iter_terms())

    for step in (0, 1, 3, 5):  # the first reset always fires; then at least 2 (or 10) steps apart
        manager.compute(EventMode.RESET, envs_idx=slice(None), global_env_step_count=step)

    assert fast.fires.tolist() == [3] * NUM_ENVS  # steps 0, 3, 5
    assert slow.fires.tolist() == [1] * NUM_ENVS  # step 0


def test_a_throttled_reset_fires_only_in_the_envs_it_is_given(box_env):
    manager = EventManager(
        env=box_env,
        options=EventManagerOptions(dr=CountingEvent.configure(mode=EventMode.RESET, min_step_count_between_reset=5)),
    )
    term = next(term for _, _, term in manager.iter_terms())

    manager.compute(EventMode.RESET, envs_idx=torch.tensor([1, 2]), global_env_step_count=0)
    manager.compute(EventMode.RESET, envs_idx=slice(None), global_env_step_count=3)  # first time for 0 and 3

    assert term.fires.tolist() == [1, 1, 1, 1]


def test_terms_are_listed_by_mode_in_priority_order(box_env):
    manager = EventManager(
        env=box_env,
        options=EventManagerOptions(
            late=_interval("counting_event", 1.0, priority=2),
            early=_interval("counting_event", 1.0, priority=0),
            on_reset=CountingEvent.configure(mode=EventMode.RESET),
        ),
    )

    assert dict(manager.mode_term_names) == {EventMode.INTERVAL: ["early", "late"], EventMode.RESET: ["on_reset"]}
    assert manager.active_terms == ["early", "late", "on_reset"]


def test_get_term_finds_a_term_by_key_across_modes_and_lists_them_on_a_miss(box_env):
    manager = EventManager(
        env=box_env,
        options=EventManagerOptions(
            tick=_interval("counting_event", 1.0), on_reset=CountingEvent.configure(mode=EventMode.RESET)
        ),
    )

    assert manager.get_term("on_reset") is next(term for _, name, term in manager.iter_terms() if name == "on_reset")
    with pytest.raises(
        KeyError, match="No active term named 'on_rset' in EventManager. Available options: tick, on_reset"
    ):
        manager.get_term("on_rset")


@EVENT_TERM_REGISTRY.register()
class BatchingEvent(EventTerm):
    """A kind that fires as a batch: every term records the terms it fired with, itself included, and the envs."""

    def build(self) -> None:
        self.fired_with: list[tuple[EventTerm, ...]] = []
        self.fired_on: list[slice | torch.Tensor] = []

    def compute(self, envs_idx=None) -> None:
        type(self).compute_batch([self], envs_idx)

    @classmethod
    def compute_batch(cls, terms, envs_idx) -> None:
        for term in terms:
            term.fired_with.append(tuple(terms))
            term.fired_on.append(envs_idx.clone() if isinstance(envs_idx, torch.Tensor) else envs_idx)


@EVENT_TERM_REGISTRY.register()
class SoloBatchingEvent(BatchingEvent):
    """Overrides ``compute`` alone, which its parent's ``compute_batch`` knows nothing of."""

    def compute(self, envs_idx=None) -> None:
        self.fired_with.append((self,))


def test_adjacent_reset_terms_of_a_batching_kind_fire_as_one_batch(box_env):
    """A run breaks at a term of another kind, at a subclass with its own compute, and at a throttled term."""
    reset = EventMode.RESET
    manager = EventManager(
        env=box_env,
        options=EventManagerOptions(
            a=BatchingEvent.configure(mode=reset, priority=0),
            b=BatchingEvent.configure(mode=reset, priority=0),
            counted=CountingEvent.configure(mode=reset, priority=5),
            c=BatchingEvent.configure(mode=reset, priority=10),
            solo=SoloBatchingEvent.configure(mode=reset, priority=10),
            d=BatchingEvent.configure(mode=reset, priority=10),
            throttled=BatchingEvent.configure(mode=reset, priority=10, min_step_count_between_reset=2),
        ),
    )
    terms = {name: term for _, name, term in manager.iter_terms()}
    names = {id(term): name for name, term in terms.items()}

    manager.compute(reset, envs_idx=slice(None), global_env_step_count=0)

    fired = {
        name: [[names[id(t)] for t in batch] for batch in term.fired_with]
        for name, term in terms.items()
        if name != "counted"
    }
    assert fired == {
        "a": [["a", "b"]],
        "b": [["a", "b"]],
        "c": [["c"]],
        "solo": [["solo"]],
        "d": [["d"]],
        "throttled": [["throttled"]],
    }
    assert terms["counted"].fires.tolist() == [1] * NUM_ENVS


def test_a_throttle_set_on_a_batched_term_after_the_build_is_honoured(box_env):
    """As for a lone term, whose throttle is read every reset: the batch splits and the throttled term waits."""
    reset = EventMode.RESET
    manager = EventManager(
        env=box_env,
        options=EventManagerOptions(a=BatchingEvent.configure(mode=reset), b=BatchingEvent.configure(mode=reset)),
    )
    a, b = (term for _, _, term in manager.iter_terms())
    manager.compute(reset, envs_idx=slice(None), global_env_step_count=1)
    a.min_step_count_between_reset = 10

    manager.compute(reset, envs_idx=slice(None), global_env_step_count=2)

    assert a.fired_with == [(a, b), (a,)] and b.fired_with == [(a, b), (b,)]
    assert not a.fired_on[-1].any()  # one step after its last fire, the throttled term selects no env


@EVENT_TERM_REGISTRY.register()
def noop_event(env, *, envs_idx=None) -> None:
    return None


def test_the_interval_tick_and_a_masked_reset_are_sync_free(cuda_stand_in_env, count_cuda_syncs):
    """The countdown ticks every step with a mask of the due envs, usually none."""
    manager = EventManager(
        env=cuda_stand_in_env,
        options=EventManagerOptions(noop=_interval("noop_event", 0.02)),
    )
    mask = torch.tensor([True, False, True, False], device="cuda")

    assert count_cuda_syncs(lambda: manager.compute(mode=EventMode.INTERVAL, dt=0.02)) == 0
    assert count_cuda_syncs(lambda: manager.reset(mask)) == 0
