"""Tests for ``TerminationManager``: terminated vs. timed-out, per-term done masks, reset logging, and the
``compute_once`` pass around the term loop.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationManager, TerminationTerm
from eden.options.managers.terminations import TerminationManagerOptions


@TERMINATION_TERM_REGISTRY.register()
class ScriptedDone(TerminationTerm):
    """Done in the envs a test sets in ``done``."""

    def build(self) -> None:
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def compute(self) -> torch.Tensor:
        return self.done.clone()


@TERMINATION_TERM_REGISTRY.register()
class SharedDraw(TerminationTerm):
    """Records the draw it read through ``compute_once`` (a fresh random draw per production) in ``seen``,
    and raises after reading it while ``fail`` is set. Never done.
    """

    def build(self) -> None:
        self.fail = False
        self.seen = None

    def compute(self) -> torch.Tensor:
        manager = self._env.termination_manager
        self.seen = manager.compute_once(("termination_test_shared",), lambda: torch.rand(self.num_envs))
        if self.fail:
            raise RuntimeError("term failed mid-pass")
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


def test_terms_split_into_terminated_and_timed_out_and_keep_their_own_masks(box_env):
    manager = TerminationManager(
        env=box_env,
        options=TerminationManagerOptions(
            fell=ScriptedDone.configure(), out_of_time=ScriptedDone.configure(time_out=True)
        ),
    )
    manager.get_term("fell").done[:] = torch.tensor([True, False, False, False])
    manager.get_term("out_of_time").done[:] = torch.tensor([False, True, False, True])

    dones = manager.compute()

    assert dones.tolist() == [True, True, False, True]
    assert manager.terminated.tolist() == [True, False, False, False]
    assert manager.timeouts.tolist() == [False, True, False, True]
    # get_term is the base contract (the term); the mask is get_term_dones.
    assert isinstance(manager.get_term("fell"), ScriptedDone)
    assert manager.get_term_dones("out_of_time").tolist() == [False, True, False, True]
    extras = manager.reset(envs_idx=torch.tensor([False, True, False, True]))  # the auto-reset's mask
    assert (extras["Episode_Termination/fell"], extras["Episode_Termination/out_of_time"]) == (0, 2)


def test_a_flagged_env_is_truncated_and_counted_only_as_a_solver_error(box_env, flag_solver_error):
    """The simulator failed, not the task: the learner bootstraps the step (``time_outs``) instead of
    taking a 0-reward terminal a penalty-heavy policy would learn to provoke, the verdict a term reached on the
    corrupt state is dropped, and its own ``Episode_Termination`` key keeps a run hitting them from looking clean.
    """
    manager = TerminationManager(env=box_env, options=TerminationManagerOptions(fell=ScriptedDone.configure()))
    manager.get_term("fell").done[:] = torch.tensor([True, False, True, False])  # env 2 "fell" on garbage
    flag_solver_error(box_env, [2])

    assert manager.compute().tolist() == [True, False, True, False]
    assert manager.terminated.tolist() == [True, False, False, False]
    assert manager.timeouts.tolist() == [False, False, True, False]
    assert manager.solver_errors.tolist() == [False, False, True, False]
    assert manager.get_term_dones("fell").tolist() == [True, False, False, False]
    extras = manager.reset(envs_idx=torch.tensor([True, False, True, False]))
    assert (extras["Episode_Termination/fell"], extras["Episode_Termination/solver_error"]) == (1, 1)


def test_a_term_named_like_the_solver_error_count_fails_at_construction(box_env):
    with pytest.raises(ValueError, match="'solver_error' is reserved.*Rename the term"):
        TerminationManager(env=box_env, options=TerminationManagerOptions(solver_error=ScriptedDone.configure()))


def test_a_masked_reset_is_sync_free(cuda_stand_in_env, count_cuda_syncs):
    """The per-term counts are not an ``.item()`` each."""
    manager = TerminationManager(
        env=cuda_stand_in_env, options=TerminationManagerOptions(fell=ScriptedDone.configure())
    )
    mask = torch.tensor([True, False, True, True], device="cuda")

    assert count_cuda_syncs(lambda: manager.reset(mask)) == 0


@pytest.fixture(scope="module")
def sharing_env(make_box_env):
    """An env whose own termination manager holds two ``SharedDraw`` terms, found as ``env.termination_manager``."""
    return make_box_env(
        termination_options=TerminationManagerOptions(a=SharedDraw.configure(), b=SharedDraw.configure())
    )


def test_termination_terms_sharing_a_key_get_one_production_per_pass(sharing_env):
    manager = sharing_env.termination_manager
    a, b = manager.get_term("a"), manager.get_term("b")

    manager.compute()
    first = a.seen
    assert torch.equal(b.seen, first)

    manager.compute()  # a new pass starts empty
    assert not torch.equal(a.seen, first)
    assert torch.equal(b.seen, a.seen)


def test_a_termination_term_that_raises_mid_pass_does_not_leave_the_memo_behind(sharing_env):
    manager = sharing_env.termination_manager
    flaky = manager.get_term("b")
    flaky.fail = True
    try:
        with pytest.raises(RuntimeError, match="term failed mid-pass"):
            manager.compute()
    finally:
        flaky.fail = False

    assert manager.compute_once(("termination_test_shared",), lambda: "fresh") == "fresh"
