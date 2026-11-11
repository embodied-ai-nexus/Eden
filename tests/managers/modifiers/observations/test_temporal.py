"""Tests for the built-in observation modifiers ``ExponentialMovingAverage`` and ``ObservationDelay``.

``compute(frame)`` is a read and ``advance(frame)`` the once-per-step commit, made on the
``update_history=True`` pass. Driven through a real ``ObservationManager`` on the built ``box_env``.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.modifiers.observations import ExponentialMovingAverage, ObservationDelay
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationManager, ObservationTerm
from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions

NUM_ENVS = 4


@OBSERVATION_TERM_REGISTRY.register()
class FedFrame(ObservationTerm):
    """``value`` in every entry of a ``(num_envs, *shape)`` frame of ``dtype``; a test sets ``value``."""

    shape: tuple = (2,)
    dtype: str = "float32"

    def build(self) -> None:
        self.value = 0.0

    def compute(self) -> torch.Tensor:
        return torch.full(
            (self.num_envs, *self.shape), self.value, dtype=getattr(torch, self.dtype), device=self.device
        )


def _manager(env, **terms) -> ObservationManager:
    return ObservationManager(env=env, options=ObservationManagerOptions(policy=ObservationGroupOptions(**terms)))


def _frame(*modifiers, **kwargs):
    return FedFrame.configure(modifiers=list(modifiers), **kwargs)


def _drive(mgr: ObservationManager, values) -> list[float]:
    """Push ``values`` through once per control step; return what env 0 was delivered in its first entry."""
    delivered = []
    for value in values:
        mgr.get_term("x").value = value
        obs = mgr.compute(update_history=True)["policy"]
        delivered.append(float(obs.reshape(NUM_ENVS, -1)[0, 0]))
    return delivered


# ---------------------------------------------------------------------------
# ExponentialMovingAverage
# ---------------------------------------------------------------------------


def test_ema_follows_its_recurrence_from_zero_without_backfill(box_env):
    mgr = _manager(box_env, x=_frame(ExponentialMovingAverage.configure(alpha=0.2, backfill=False)))

    # y_t = 0.2 x_t + 0.8 y_{t-1}, y_0 = 0
    assert _drive(mgr, [1.0, 2.0, 4.0, 8.0]) == pytest.approx([0.2, 0.56, 1.248, 2.5984])


def test_ema_with_backfill_delivers_the_first_frame_unchanged(box_env):
    """As if every frame before the first had equalled it: no ramp-up from zero."""
    mgr = _manager(box_env, x=_frame(ExponentialMovingAverage.configure(alpha=0.2)))

    assert _drive(mgr, [5.0, 10.0]) == pytest.approx([5.0, 6.0])


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["+inf", "-inf"])
@pytest.mark.parametrize(
    "alpha, backfill, poisons",
    [(0.2, True, True), (0.2, False, True), (0.8, True, True), (1.0, True, False)],
    ids=["small-alpha", "small-alpha-no-backfill", "large-alpha", "pass-through"],
)
def test_an_ema_carries_an_infinity_rather_than_delivering_nan(box_env, sign, alpha, backfill, poisons):
    """A NaN is what the term's ``clip`` cannot bound; ``torch.lerp`` meets ``inf - inf`` (or ``0 * inf``)."""
    mgr = _manager(box_env, x=_frame(ExponentialMovingAverage.configure(alpha=alpha, backfill=backfill)))
    infinity = sign * torch.inf

    assert _drive(mgr, [infinity, 2.0]) == [infinity, infinity if poisons else 2.0]


def test_a_reset_clears_an_infinity_out_of_the_average(box_env):
    mgr = _manager(box_env, x=_frame(ExponentialMovingAverage.configure(alpha=0.5)))
    assert _drive(mgr, [torch.inf]) == [torch.inf]

    mgr.reset()

    assert _drive(mgr, [2.0, 4.0]) == pytest.approx([2.0, 3.0])


@pytest.mark.parametrize(
    "envs_idx", [torch.tensor([False, True, False, True]), torch.tensor([1, 3])], ids=["mask", "index"]
)
def test_an_ema_restarts_only_the_reset_envs(box_env, envs_idx):
    mgr = _manager(box_env, x=_frame(ExponentialMovingAverage.configure(alpha=0.5, backfill=False)))
    _drive(mgr, [4.0])  # every env at 2.0

    mgr.reset(envs_idx)
    mgr.get_term("x").value = 4.0

    assert mgr.compute(update_history=True)["policy"][:, 0].tolist() == [3.0, 2.0, 3.0, 2.0]


def test_an_integer_frame_is_refused_by_the_average(box_env):
    """Weights held in int64 would be 0 without raising, delivering zeros from the second frame on."""
    with pytest.raises(ValueError, match="needs a floating-point frame, got torch.int64") as info:
        _manager(box_env, x=_frame(ExponentialMovingAverage.configure(alpha=0.5), dtype="int64"))
    assert any("observation term 'policy.x'" in note for note in info.value.__notes__)


# ---------------------------------------------------------------------------
# ObservationDelay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backfill, early", [(True, [1.0, 1.0]), (False, [0.0, 0.0])])
def test_a_delay_delivers_the_frame_from_k_control_steps_earlier(box_env, backfill, early):
    mgr = _manager(box_env, x=_frame(ObservationDelay.configure(control_steps=2, backfill=backfill)))

    assert _drive(mgr, [1.0, 2.0, 3.0, 4.0, 5.0]) == early + [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "envs_idx", [torch.tensor([False, True, False, False]), torch.tensor([1])], ids=["mask", "index"]
)
def test_a_delay_restarts_only_the_reset_envs(box_env, envs_idx):
    mgr = _manager(box_env, x=_frame(ObservationDelay.configure(control_steps=2)))
    _drive(mgr, [1.0, 2.0, 3.0])

    mgr.reset(envs_idx)
    mgr.get_term("x").value = 4.0

    # Env 1 starts over from this frame (backfill); the others deliver the frame from two pushes ago.
    assert mgr.compute(update_history=True)["policy"][:, 0].tolist() == [2.0, 4.0, 2.0, 2.0]


def test_a_delay_carries_an_integer_frame_unchanged(box_env):
    mgr = _manager(box_env, x=_frame(ObservationDelay.configure(control_steps=1), dtype="int64"))

    assert _drive(mgr, [1, 2, 3]) == [1.0, 1.0, 2.0]


def test_the_history_window_holds_the_delayed_frames(box_env):
    mgr = _manager(box_env, x=_frame(ObservationDelay.configure(control_steps=1), history_length=3))

    for value in [1.0, 2.0, 3.0, 4.0]:
        mgr.get_term("x").value = value
        obs = mgr.compute(update_history=True)["policy"]

    assert obs[0].reshape(3, 2)[:, 0].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Both
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modifier",
    [ExponentialMovingAverage.configure(alpha=0.5), ObservationDelay.configure(control_steps=1)],
    ids=["ema", "delay"],
)
def test_reads_before_the_push_see_what_the_push_delivers_and_change_nothing(box_env, modifier):
    """The phase-12 snapshot is such a read: the new frame, computed before the step's own push."""
    reads = _manager(box_env, x=_frame(modifier))
    once = _manager(box_env, x=_frame(modifier))  # the same frames, never read in between

    for value in [1.0, 3.0, 7.0]:
        reads.get_term("x").value = once.get_term("x").value = value
        snapshots = [reads.compute()["policy"].clone() for _ in range(3)]
        stepped = reads.compute(update_history=True)["policy"]
        assert all(torch.equal(snapshot, stepped) for snapshot in snapshots)
        assert torch.equal(once.compute(update_history=True)["policy"], stepped)


def test_each_stateful_modifier_advances_on_its_own_input(box_env):
    """An EMA then a delay: the delay pushes the EMA's output (0.5, 1.25, 2.625), one step late."""
    mgr = _manager(
        box_env,
        x=_frame(
            ExponentialMovingAverage.configure(alpha=0.5, backfill=False),
            ObservationDelay.configure(control_steps=1, backfill=False),
        ),
    )

    assert _drive(mgr, [1.0, 2.0, 4.0]) == [0.0, 0.5, 1.25]


def test_state_is_sized_from_the_sample_frame(box_env):
    """A ``(N,)`` frame, and a float64 ``(N, 3, 4)`` one."""
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                scalar=_frame(ExponentialMovingAverage.configure(alpha=0.5, backfill=False), shape=()),
                image=_frame(ObservationDelay.configure(control_steps=1), shape=(3, 4), dtype="float64"),
                concatenate_terms=False,
            )
        ),
    )
    for value in [2.0, 6.0]:
        for name in ("scalar", "image"):
            mgr.get_term(name).value = value
        obs = mgr.compute(update_history=True)["policy"]

    assert obs["scalar"].tolist() == [3.5] * NUM_ENVS  # 1, then 0.5 * 6 + 0.5 * 1
    assert torch.equal(obs["image"], torch.full((NUM_ENVS, 3, 4), 2.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "modifier, message",
    [
        (ExponentialMovingAverage.configure(), "alpha is required"),
        (ExponentialMovingAverage.configure(alpha=0.0), r"alpha must be in \(0, 1\]"),
        (ExponentialMovingAverage.configure(alpha=1.5), r"alpha must be in \(0, 1\]"),
        (ObservationDelay.configure(), "control_steps is required"),
        (ObservationDelay.configure(control_steps=0), "A delay of 0 is no delay"),
        (ObservationDelay.configure(control_steps=1.5), "must be an int >= 1"),
    ],
)
def test_a_bad_parameter_fails_at_construction_naming_the_term(box_env, modifier, message):
    with pytest.raises(ValueError, match=message) as info:
        _manager(box_env, x=_frame(modifier))
    assert any("ObservationManager term 'policy.x'" in note for note in info.value.__notes__)


def test_a_misspelled_parameter_fails_where_it_is_written():
    with pytest.raises(ValueError, match="alhpa"):
        ExponentialMovingAverage.configure(alhpa=0.2)


def test_an_auto_reset_step_costs_no_gpu_sync(cuda_stand_in_env, count_cuda_syncs):
    """Neither modifier adds to the per-step sync budget, the auto-reset mask included."""
    mgr = _manager(
        cuda_stand_in_env,
        x=_frame(ExponentialMovingAverage.configure(alpha=0.2), ObservationDelay.configure(control_steps=2)),
        y=_frame(ObservationDelay.configure(control_steps=1, backfill=False)),
    )
    mgr.compute(update_history=True)  # the delay allocates its storage on the first push
    mask = torch.arange(NUM_ENVS, device="cuda") % 2 == 0

    def step():
        mgr.compute()  # the phase-12 snapshot
        mgr.reset(mask)
        mgr.compute(update_history=True)

    assert count_cuda_syncs(step) == 0
