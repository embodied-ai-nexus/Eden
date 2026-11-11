"""Command manager for generating and updating commands."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from eden.managers.base import ConfigurableTermBase, FlatManagerBase
from eden.options.managers.commands import CommandManagerOptions, CommandTermOptions
from eden.utils.misc import mean_over_envs, sanitize_envs_idx
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class CommandTerm(ConfigurableTermBase[CommandTermOptions]):
    """
    Base class for command terms.

    Parameters
    ----------
    resampling_time_range: tuple[float, float]
        The range of time before commands are changed [s].
    debug_vis: bool
        Whether to visualize debug information. Defaults to False.
    """

    resampling_time_range: tuple[float, float] = None
    debug_vis: bool = False

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env=env, options=options)

        self.stats = dict()
        self.time_left = torch.zeros(self.num_envs, device=self.device)
        self.command_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    @property
    @abstractmethod
    def command(self):
        raise NotImplementedError

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, torch.Tensor]:
        envs_idx = sanitize_envs_idx(envs_idx, self.num_envs)
        extras = {}
        for stat_name, stat_value in self.stats.items():
            # A 0-dim tensor reduced on the device: gathering through the auto-reset's mask and `.item()`-ing
            # the mean were two GPU->CPU syncs per stat per step. The scalar fill stays sync-free.
            extras[stat_name] = mean_over_envs(stat_value, envs_idx)
            stat_value[envs_idx] = 0.0
        self.command_counter[envs_idx] = 0
        self._resample(envs_idx)
        return extras

    def compute(self, dt: float) -> None:
        """Tick the resampling clock, then resample and publish (phase 17 — the *preparation* epoch).

        In :meth:`RLEnvBase.step` this runs after the auto-reset and below the reward, so what it
        produces is the command the next action is chosen from and phase 19's observations carry —
        not the command this step was graded against. :meth:`_update_command` says what belongs
        here and what does not.

        On the deployment path it is one of **two** command hooks that run: ``DeploymentBase.step``
        drives :meth:`advance_reference` first — so a time-advancing reference keeps moving on
        hardware — then this method and :meth:`draw_vis`. :meth:`post_physics_refresh` does not run
        there at all. A deployed term must produce whatever the policy's observations read from here
        or from the advance — see ``docs/guides/terms.md``.
        """
        self.time_left -= dt
        resample_envs_mask = self.time_left <= 0.0
        # NOTE: no `.any()` gate here. `_resample` already counts the mask — its one GPU→CPU
        # sync — and returns early on an empty one, so a gate would only pay that sync twice.
        self._resample(resample_envs_mask)
        self._update_command()

    def _resample(self, envs_idx: slice | torch.Tensor) -> None:
        # NOTE: this runs at least once per command term per control step — `compute()` always
        # resamples, and the auto-reset path comes back here whenever an env terminates — and both
        # hand it a bool mask. On one it costs ONE GPU→CPU sync: the count behind the
        # empty-selection early-out (`sanitize_envs_idx` -> `idx.sum().item()`), which is what lets
        # every `_resample_command` assume a non-empty selection. The rest is sync-free on purpose:
        # a bool mask used as an index is sized on the host, a sync per gather or write, where
        # `torch.where` and `add_(mask)` stay on the device.
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return
        # One shape for `_resample_command` whatever reached here, and a fresh tensor, so an
        # implementation can never write into the caller's `reset_buf`.
        envs_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        envs_mask[envs_idx] = True

        new_time_left = torch.empty_like(self.time_left).uniform_(*self.resampling_time_range)
        self.time_left.copy_(torch.where(envs_mask, new_time_left, self.time_left))
        self.command_counter += envs_mask  # a selected env adds True, i.e. 1
        # Last, so nothing here reads the mask after the implementation has had it.
        self._resample_command(envs_mask)

    @abstractmethod
    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        """Resample the command for the environments selected by ``envs_idx``.

        ``envs_idx`` is always a ``(num_envs,)`` bool mask with at least one ``True``.
        :meth:`_resample` converts whatever selector reached it — the ``time_left <= 0`` mask from
        :meth:`compute`, the auto-reset's ``reset_buf``, a full-batch reset's ``slice``, an
        explicit index — into a fresh mask, and returns before calling this on an empty
        selection, so a guard for one here would be dead code. A term that calls this itself
        passes the same kind of mask (``MotionCommand``'s clip wrap). ``len(envs_idx)`` is
        therefore ``num_envs``, not the selected count.

        Write the new values through ``torch.where`` over a full-batch draw::

            r = torch.empty(self.num_envs, device=self.device)
            self.command[:, 0] = torch.where(envs_idx, r.uniform_(*self.some_range), self.command[:, 0])

        rather than through the mask as an index. ``self.command[envs_idx, 0] = r[envs_idx]`` is
        just as correct, but a bool mask used to index is sized on the host — a blocking GPU→CPU
        sync for the gather and another for the write — and this method runs every control step.
        A scalar write, ``self.flag[envs_idx] = False``, is a masked fill and costs none. To
        replace whole rows, broadcast the mask: ``torch.where(envs_idx[:, None], rows, self.command)``.

        See :class:`~eden.managers.terms.commands.velocity.UniformVelocityCommand` for the
        canonical implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def _update_command(self) -> None:
        """Publish this step's command value (phase 17, after the auto-reset).

        The **preparation** epoch: whatever is written here is what phase 19's observations carry
        and what the policy picks the next action from. Closing the loop of a resampling state
        machine belongs here too — deciding a goal has been reached and drawing the next one —
        because a switch is publication.

        **A measurement of what the step produced does not belong here.** A hold counter, a contact
        latch, a tracking error — anything derived from the post-physics robot — is a whole control
        step stale by the time it is read, because the reward manager runs at phase 9, above this
        call at 17. Worse, anything this call *clears* is never read at all: the step that earned
        the value is graded through the cleared one and silently pays zero. Compute it in
        :meth:`post_physics_refresh`, which runs at phase 5 of the very step the reward grades.

        The test is what the state *is*, not who reads it: a tracking reward reads the command
        itself, and the command still belongs here — grading the robot against a command resampled
        after the physics that produced its state is the arrangement this ordering exists to avoid.
        The rule, and the bug that produced it, are in ``docs/guides/terms.md`` under
        "Which hook: grade the step, or prepare the next one".
        """
        raise NotImplementedError

    def draw_vis(self) -> None:
        pass

    def post_physics_refresh(self) -> None:
        """Refresh state derived from the post-physics robot pose (phase 5 — the *grading* epoch).

        No-op by default; override to refresh caches. This runs after the physics sub-steps and
        before metric/termination/reward, which makes it the hook for **everything that grades
        this step**: pose and contact caches, progress counters, hold latches, tracking errors.
        A value written here is read by phase 9's reward in the same step — no lag, and nothing
        to snapshot across the auto-reset. Writing it in :meth:`_update_command` instead is the
        bug class ``docs/guides/terms.md`` documents under "Which hook: grade the step, or
        prepare the next one".

        Invoked by :meth:`CommandManager.post_physics_refresh`, which owns the call-time contract —
        which means it is **not** invoked on the deployment path: ``DeploymentBase.step`` drives
        :meth:`advance_reference` and :meth:`compute`, and nothing between them, because on
        hardware there are no physics sub-steps to bracket and no reward to grade. State a deployed
        term's observations read has to be produced from :meth:`compute` too; an idempotent cache
        refresh can simply live in both, the way
        :class:`~eden.managers.terms.commands.mimic.MotionCommand` refreshes its robot-state cache.
        """

    def pre_physics_advance(self) -> None:
        """Advance the command BEFORE the physics sub-steps.

        No-op by default. Overridden by terms that track a time-advancing reference
        and must advance it *before* physics so the post-physics robot is graded
        against the temporally-aligned frame ("advance-then-grade"; e.g.
        :class:`~eden.managers.terms.commands.mimic.MotionCommand`). Invoked once
        per control step by :meth:`CommandManager.pre_physics_advance` in
        :meth:`RLEnvBase.step`, before the decimation loop.

        The first of the term's three per-step hooks; the other two are
        :meth:`post_physics_refresh` (grade the step that just happened) and
        :meth:`_update_command` (publish the command the next step is chosen against). Unlike
        :meth:`post_physics_refresh`, this one also runs on deployment through
        :meth:`advance_reference` by default. A term whose simulation advance resets the robot
        must override that deployment hook with a clock-only advance.
        """

    def advance_reference(self) -> None:
        """Advance a deployed reference without resetting the robot.

        Preserve the existing advance hook by default. Terms that resample or teleport at
        clip boundaries in :meth:`pre_physics_advance` must provide a clock-only version here.
        """
        self.pre_physics_advance()

    def rewind_reference(self) -> None:
        """Restart a time-advancing reference at its first frame, for every env (no-op by default).

        The reset-side twin of :meth:`advance_reference`, and it exists for the deployment
        path alone: ``DeploymentBase.reset_motion_reference`` drives it at deploy reset, at the
        operator's START and on every policy switch, so a clip hands over from its beginning the
        way the manager path already does through ``MotionManager.rewind_to_start``. Without it
        a term that advances a clock resumes wherever the clock happened to stop — mid-clip, at a
        pose the robot standing in its INIT stance is nowhere near.

        Rewind the **clock only**. This runs on hardware, where there is no robot to place and no
        episode to randomize, so it must not reach for the reset-state initialization a wrap in
        :meth:`pre_physics_advance` performs in sim (``MotionCommand._resample_command`` teleports
        the robot and perturbs its pose). Clip *selection* is not resampled either: a deploy
        replays the same clip deterministically.
        """


# TermRegistry (not plain Registry) so CommandTermOptions gets the registry stamp for
# construction-time parameter-name validation. Command terms are class-based only.
COMMAND_TERM_REGISTRY = TermRegistry("COMMAND_TERM", CommandTerm, populated_by=("eden.managers.terms.commands",))


class CommandManager(FlatManagerBase[CommandManagerOptions]):
    _registry = COMMAND_TERM_REGISTRY

    def __init__(self, env: EnvBase, options: CommandManagerOptions):
        super().__init__(env=env, options=options)

        self._commands = dict()

    def summary(self) -> str:
        return self._format_summary_table(
            title="Active Command Terms",
            field_names=["Index", "Name", "Type"],
            rows=([index, name, term.__class__.__name__] for index, (name, term) in enumerate(self._terms.items())),
            align={"Name": "l"},
        )

    def draw_vis(self) -> None:
        for term in self._terms.values():
            if term.debug_vis:
                term.draw_vis()

    def post_physics_refresh(self) -> None:
        """Refresh every term's robot-state cache on the just-computed post-physics state.

        Called once per control step in :meth:`RLEnvBase.step` after the physics sub-steps
        but BEFORE metric/termination/reward, so terms that cache robot state (e.g. the
        motion-imitation command) grade the CURRENT step's state (S_N) rather than the
        previous step's (S_{N-1}). This method owns that ordering contract; individual terms
        override the no-op :meth:`CommandTerm.post_physics_refresh`.

        Being above the graded layer is also what makes this the right hook for a term's own
        per-step *grading* state, rather than :meth:`CommandTerm._update_command` below the
        reward — see that method, and ``docs/guides/terms.md``.
        """
        for term in self._terms.values():
            term.post_physics_refresh()

    def pre_physics_advance(self) -> None:
        """Advance every term's command BEFORE the physics sub-steps (no-op by default).

        Called once per control step in :meth:`RLEnvBase.step`, before the decimation
        loop, so terms tracking a time-advancing reference (e.g. the motion-imitation
        command) advance it ahead of physics — see :meth:`CommandTerm.pre_physics_advance`.
        """
        for term in self._terms.values():
            term.pre_physics_advance()

    def rewind_reference(self) -> None:
        """Restart every term's time-advancing reference at its first frame (no-op by default).

        Driven only by ``DeploymentBase.reset_motion_reference`` — see
        :meth:`CommandTerm.rewind_reference` for what a term may and may not do there.
        """
        for term in self._terms.values():
            term.rewind_reference()

    def advance_reference(self) -> None:
        """Advance each deployed reference; simulation still uses pre_physics_advance."""
        for term in self._terms.values():
            term.advance_reference()

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, torch.Tensor]:
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        extras = {}
        for name, term in self._terms.items():
            stats = term.reset(envs_idx=envs_idx)
            for stat_name, stat_value in stats.items():
                extras[f"Commands/{name}/{stat_name}"] = stat_value
        return extras

    def compute(self, dt: float) -> None:
        """Resample and publish every term's command (phase 17 of :meth:`RLEnvBase.step`).

        Called after the auto-reset and below the reward, so the command it leaves behind is the
        one phase 19's observations carry and the next action is chosen against — deliberately not
        the one this step was graded on, which is what keeps ``obs -> action -> reward`` aligned.
        See :meth:`CommandTerm._update_command`.
        """
        for term in self._terms.values():
            term.compute(dt)

    def get_command(self, name: str) -> torch.Tensor:
        """Return the command for the specified command term."""
        return self.get_term(name).command

    def swap_term(self, name: str, options: CommandTermOptions) -> CommandTerm:
        """Replace the built term ``name`` with one configured by ``options``; return the old one.

        The post-build counterpart of what the config does before build: a deployment
        substitutes a task's trained command generator for one an operator drives
        (:class:`~eden.managers.terms.commands.velocity.OperatorVelocityCommand`) or one that
        simply holds a value, the way it substitutes the simulated entity for a
        :class:`~eden.extensions.deployment.utils.state.RobotStateEntity`. The caller owns the
        returned term and can pass its options back here to undo the swap.

        The replacement is built and ``build()``-ed exactly as ``_prepare_terms`` would, and its
        command must have the **same shape** as the one it replaces: the observation buffers and
        the trained policy's input width were fixed by the old term, so a different width would
        reach the policy as a silently misaligned observation vector.

        Nothing here repoints terms that cached the *replaced* term in their own ``build()``
        (the pattern ``self._command = env.command_manager.get_term(...)``). A caller that needs
        those to follow the swap must repoint them — see
        :meth:`~eden.extensions.deployment.base.DeploymentBase._repoint_command_term_refs`.

        Raises
        ------
        KeyError
            If ``name`` is not an active term (the message lists the active ones).
        ValueError
            If the replacement's command shape differs from the replaced term's.
        """
        old_term = self.get_term(name)
        new_term = self._build_term(name, term_option=options)
        # ``build()`` has already run, so the replacement may own something by now — an
        # OperatorVelocityCommand holds a *process-global* keyboard grab. Release it if we
        # end up rejecting it, or a config error leaves that grab behind with nothing
        # holding a reference to close it.
        try:
            if new_term.command.shape != old_term.command.shape:
                raise ValueError(
                    f"Cannot swap command term '{name}': {type(new_term).__name__} commands "
                    f"{tuple(new_term.command.shape)} but {type(old_term).__name__} commands "
                    f"{tuple(old_term.command.shape)}. The observation width the policy was trained "
                    "on comes from the replaced term, so the two must agree."
                )
        except BaseException:
            close = getattr(new_term, "close", None)
            if close is not None:
                close()
            raise
        self._terms[name] = new_term
        return old_term

    def restore_term(self, name: str, term: CommandTerm) -> None:
        """Put back a term :meth:`swap_term` returned, undoing that swap.

        Deliberately narrow — it takes an already-built term rather than options, because the
        only correct argument is the exact object ``swap_term`` displaced. Rebuilding the
        original from its options instead would drop whatever state it had accumulated and
        leave any reference the caller repointed pointing at a third object.
        """
        if name not in self._terms:
            available = ", ".join(self._term_names) or "none"
            raise KeyError(f"No active term named '{name}' in {type(self).__name__}. Available options: {available}")
        self._terms[name] = term
