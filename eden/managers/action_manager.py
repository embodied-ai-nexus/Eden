"""Action manager and ActionTerm base: map policy actions to actuator commands.

The manager slices the policy action vector across its active :class:`ActionTerm`
instances and applies them every control step (re-applied across ``decimation``
physics sub-steps). ``ActionTerm.dofs_order`` maps a DOF name to its position in the term's
``dofs_name``; mind the DOF-ordering caveat in :mod:`eden.entities.rigid` when wiring actions to joints.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import genesis as gs
import torch

from eden.managers.base import ConfigurableTermBase, FlatManagerBase
from eden.options.managers.actions import ActionManagerOptions, ActionTermOptions
from eden.utils.misc import as_genesis_index
from eden.utils.registry import TermRegistry
from eden.utils.smoke import is_smoke
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


class ActionTerm(ConfigurableTermBase[ActionTermOptions]):
    """Base class for action terms."""

    entity_name: str = ""
    dofs_name: str | list[str] = []

    #: Declared by the term class, not configurable per-instance (leading underscore keeps it out
    #: of :meth:`ConfigurableMixin.get_parameter_names`, so it stays out of every serialized config).
    #: Read through the :attr:`sim_surrogate` property.
    _sim_surrogate_: ClassVar[bool] = False

    #: Declared by a term class that acts on the simulation through its own per-physics-step
    #: ``apply_actions`` rather than through the policy -- an external disturbance wrench, which has
    #: to be re-applied on every sub-step because Genesis clears applied forces after each one. Such
    #: a term draws no action column and commands no DOF, which is otherwise the signature of a
    #: misconfigured term (:meth:`ActionManager._reject_no_op_terms`); the flag is what exempts it.
    _acts_without_action_: ClassVar[bool] = False

    #: Declared by a term class that reads a ``NaN`` command as "no command this step" (a policy
    #: server that omitted the value) and records in :attr:`action` the value it acted on instead.
    #: :meth:`ActionManager.compute` then stores that value in its own buffers, so last-action
    #: observations, action-rate rewards and recorders never see the ``NaN``. Other terms are
    #: stored exactly as received.
    _accepts_missing_command_: ClassVar[bool] = False

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)

        self._entity = None
        self.dofs_idx_local = None
        # The memo behind ``genesis_dofs_idx``: its value, and the ``dofs_idx_local`` it was derived from.
        self._genesis_dofs_idx_source: torch.Tensor | None = None
        self._genesis_dofs_idx: range | torch.Tensor | None = None
        self._n_dofs = 0
        self._raw_action = None
        self._prev_action = None
        self._processed_action = None
        # Set once the first-step command-shape check has run (see ``verify_command_shape``).
        self._command_shape_checked = False

    def build(self) -> None:
        self._entity = self.resolve_entity(self.entity_name)
        self.dofs_name, dofs_idx_local = self.entity.find_named_dofs_idx_local(
            self.dofs_name, name_scope=self.entity.dofs_name, preserve_order=True
        )
        self._reject_multi_dof_joints()
        self.dofs_idx_local = torch.as_tensor(dofs_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self._n_dofs = len(self.dofs_idx_local)
        self._raw_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_action = torch.zeros(self.num_envs, self.num_dofs, device=self.device)

    def _reject_multi_dof_joints(self) -> None:
        """Refuse a ``dofs_name`` that matched a joint Genesis gives more than one DOF.

        An action term addresses one DOF per *name*. Its ``scale`` / ``offset`` dicts are keyed by
        joint name, and a deployment's motor map is name-keyed too — a free or ball joint breaks
        both, and does so silently rather than loudly: ``Entity.dofs_name`` lists such a joint once
        per DOF, and ``build`` hands that already-expanded list back as the ``name_scope``, so an
        n-DOF joint matches n times and re-expands to n\u00b2 entries over only n distinct DOFs.
        Verified on ``robots.G1Dofs29(dofs_name=['.*'])``: ``floating_base_joint`` (6 DOFs)
        resolves to 36 entries.

        Only reachable when the *entity* lists a multi-DOF joint in its own ``dofs_name`` — no
        shipped robot does; a ``.*`` pattern written there is the way in.
        """
        hits = Counter(self.dofs_name)
        offenders = [
            f"'{name}' ({self.entity.dofs_name.count(name)} DOFs, resolved to {count} columns)"
            for name, count in hits.items()
            if count > 1
        ]
        if not offenders:
            return
        raise ValueError(
            f"Action term {term_subject(self)} names "
            f"{', '.join(offenders)} — a free or ball joint. An action term commands one DOF per "
            f"name: its `scale` / `offset` dicts and a deployment's motor map are keyed by joint "
            f"name, so one value would stand for every DOF of that joint, silently.\n"
            f"Fix: list only single-DOF joints in this term's `dofs_name`. A `.*` in the "
            f"*entity's* `dofs_name` is the usual way a free base joint gets here. "
            f"See docs/guides/terms.md.\n"
            f"Do NOT try to reach a floating base by naming its joint here — no term can, for the "
            f"reason above. To *drive* one (a hand or gripper with no arm under it), use "
            f"`FreeBasePoseController`, which resolves the free joint itself and takes no "
            f"`dofs_name`. To *place* one at reset, write it from an event "
            f"(`events.reset_base_state_uniform`)."
        )

    @property
    def entity(self) -> Entity:
        return self._entity

    @property
    def genesis_dofs_idx(self) -> range | torch.Tensor:
        """``dofs_idx_local`` in the form to hand the entity's DOF accessors: a ``range`` when contiguous.

        Selects exactly the DOFs ``dofs_idx_local`` does, in the same order, but a ``range`` takes
        Genesis' slice path — sync-free and ~4x cheaper on the host than an index tensor, which
        matters in ``apply_actions()``, called once per physics substep (see
        :func:`eden.utils.misc.contiguous_range`). Derived from whichever tensor ``dofs_idx_local``
        currently holds, so a subclass that rebinds it in its own ``build()`` never reads a stale one.
        """
        if self._genesis_dofs_idx_source is not self.dofs_idx_local:
            self._genesis_dofs_idx = as_genesis_index(self.dofs_idx_local)
            self._genesis_dofs_idx_source = self.dofs_idx_local
        return self._genesis_dofs_idx

    @property
    def action(self) -> torch.Tensor:
        """The raw action this term received from the policy this step, shape ``(num_envs, action_dim)``."""
        return self._raw_action

    @property
    def prev_action(self) -> torch.Tensor:
        """The raw action this term received on the previous step, shape ``(num_envs, action_dim)``."""
        return self._prev_action

    @property
    def processed_action(self) -> torch.Tensor:
        """The command this term issued this step — what a deployment sends to the robot.

        For the joint PD controllers this is the position target the controller
        tracks — ``raw_action * scale + offset + reference_offset``, after any
        configured :class:`~eden.managers.modifiers.ActionModifier`'s control-step
        ``modify_processed_action``. Other terms command other quantities (a velocity,
        an effort, a binary weld flag), so both the units and the trailing width are
        term-specific; only the leading ``num_envs`` dimension is guaranteed.

        A physics-step modifier (``modify_applied_target`` — e.g.
        :class:`~eden.managers.modifiers.actions.actuators.ActionDelay`) is not folded in:
        under a delay the simulated actuator tracks an *earlier* ``processed_action``,
        exactly as the real robot tracks the command it received a latency ago.

        Read this rather than recomputing the target at the call site — a
        recomputation silently drops whatever control-step modifiers are configured,
        so the logged target diverges from the one the controller issued.
        """
        return self._processed_action

    @property
    def sim_surrogate(self) -> bool:
        """Whether this term only stands in, in simulation, for something the real robot does itself.

        ``True`` means the physical effect follows from another term's command (or from physics)
        on hardware, so a deployment has nothing to send for this term:
        :class:`~eden.managers.terms.actions.welding_actions.ParallelJawWelding` welds the object
        to the gripper because sim contact friction is unreliable, while on the robot the gripper
        joints closing — commanded by a separate joint term — hold it.

        ``False`` (the default) means the term commands something in its own right, and a
        deployment that cannot carry that command must say so rather than drop it:
        :class:`~eden.managers.terms.actions.welding_actions.SuctionCupWelding` is the vacuum
        on/off bit, which no other term supplies. See
        :func:`eden.extensions.deployment.base.build_term_action_maps`.
        """
        return self._sim_surrogate_

    @property
    def dofs_order(self) -> dict[str, int]:
        """Get the mapping from dofs name to its local index."""
        return {dof_name: i for i, dof_name in enumerate(self.dofs_name)}

    @property
    def action_dim(self) -> int:
        return self._n_dofs

    @property
    def num_dofs(self) -> int:
        return self._n_dofs

    def verify_command_shape(self, command: torch.Tensor) -> None:
        """Fail loud — naming this term — the first time a control command's width != the DOF count.

        A wrong-width joint command otherwise reaches the raw Genesis call and surfaces as an opaque
        ``GenesisException: Invalid input shape: (N, W)... required size D`` naming neither the term,
        the entity, nor the DOF set. Controllers call this from ``apply_actions``
        with the exact tensor they hand to ``entity.control_dofs_*`` — the only layer that knows both
        the term identity and the true (post-slice) command width.

        Checked once per term, not every step: the command width is structural (fixed by the term's
        ``dofs_name`` and controller), so a first-step check catches a misconfiguration without any
        steady-state cost. No-op for a shape-less command (a Python scalar Genesis handles itself).
        """
        if self._command_shape_checked:
            return
        self._command_shape_checked = True
        shape = getattr(command, "shape", None)
        if shape is None or len(shape) == 0:
            return
        width = shape[-1]
        if width != self._n_dofs:
            raise ValueError(
                f"Action term {term_subject(self)} produced a command of "
                f"width {width}, but it controls {self._n_dofs} DOFs ({self.dofs_name}). The command's last "
                f"dimension must equal the number of controlled DOFs — check the term's `dofs_name` and its "
                f"controller output size."
            )

    @abstractmethod
    def apply_actions(self) -> None:
        raise NotImplementedError


# TermRegistry (not plain Registry) so ActionTermOptions gets the registry stamp for
# construction-time parameter-name validation. Action terms are class-based only.
ACTION_TERM_REGISTRY = TermRegistry("ACTION_TERM", ActionTerm, populated_by=("eden.managers.terms.actions",))


class ActionManager(FlatManagerBase[ActionManagerOptions]):
    """Action manager for processing actions sent to the environment."""

    _registry = ACTION_TERM_REGISTRY

    def __init__(self, env: EnvBase, options: ActionManagerOptions):
        super().__init__(env=env, options=options)

        self._reject_no_op_terms()
        self._reject_overlapping_dofs()

        # Create action slices after terms are prepared
        self._term_slice = {}
        idx = 0
        for term_name, term in self._terms.items():
            self._term_slice[term_name] = slice(idx, idx + term.action_dim)
            idx += term.action_dim

        # Create buffers to store actions.
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

        # Cached once: the dict-action width check runs only in smoke mode (a process-level flag
        # set before construction), so real training pays nothing — not even the env lookup — per step.
        self._smoke = is_smoke()

    def _reject_no_op_terms(self) -> None:
        """Fail on a term that neither draws policy output nor commands a DOF.

        ``action_dim`` is what this manager reserves for a term in the action vector and
        ``num_dofs`` is what the term commands. Both zero means the term does nothing at all, and
        the cause is almost always an unset ``dofs_name`` — the option defaults to ``[]``, which
        resolves to *no* DOFs (an unmatched pattern already raises inside
        :meth:`ActionTerm.build`; an absent one does not). Left alone it builds, trains and
        deploys as a silent no-op: those joints go uncommanded and nothing ever names them.

        Zero on one side alone is legitimate and stays allowed, which is why the test is the
        conjunction rather than ``action_dim == 0``:

        * the welding terms command no DOF but own an action column — the weld / vacuum bit;
        * :class:`~eden.managers.terms.actions.joint_actions.NullJointAction` is the reverse, and
          the only joint term that reports ``action_dim == 0`` by design: it draws no action column
          *because* it pins its DOFs at their defaults for the whole episode.

        Zero on both sides is exempt only for a term class declaring ``_acts_without_action_``: an
        external disturbance applied on every physics step acts, just not through the action vector
        or a DOF.

        An action manager with no terms at all is also fine — a plain :class:`~eden.envs.base.EnvBase`
        used for replay or data collection has nothing to drive. Checked here rather than in
        ``ActionTerm.build`` because a term does not know its own config key, which is the name
        the user has to go and edit.
        """
        no_op = [
            f"{term_name!r} ({type(term).__name__} on entity {term.entity_name!r}, dofs_name={list(term.dofs_name)})"
            for term_name, term in self._terms.items()
            if term.action_dim == 0 and term.num_dofs == 0 and not term._acts_without_action_
        ]
        if no_op:
            raise ValueError(
                f"Action term(s) {', '.join(no_op)} reserve no columns of the action vector and "
                f"command no DOFs, so they would do nothing at all.\n"
                f"Fix: set `dofs_name` on each (it defaults to an empty list, which matches no DOF) "
                f"or drop the term. NullJointAction draws no action column but still needs `dofs_name`: "
                f"it commands the DOFs it pins.\n"
                f"Only a term class setting `_acts_without_action_ = True` may have neither: its "
                f"apply_actions() applies a per-physics-step disturbance (an external wrench) itself. "
                f"Do NOT set that flag to silence this error: a joint term would still command nothing."
            )

    def _reject_overlapping_dofs(self) -> None:
        """Fail on two terms of the same entity commanding the same DOF.

        Nothing downstream merges two commands on one DOF. Every term writes its own control
        target straight to Genesis in :meth:`apply_actions`, and each ``control_dofs_*`` call
        *replaces* that DOF's control mode and target rather than adding to it — so a shared DOF
        is driven by whichever term this manager applies last (declaration order) and the other
        term's command is discarded on every sub-step, with nothing raised anywhere. There is no
        legitimate overlap to opt into: the losing term is a no-op on those DOFs whatever it
        commands, while its slice of the action vector still costs the policy output width and
        still feeds the action-rate and action-smoothness rewards.

        Keyed on the resolved *local DOF index*, not the DOF name. The two agree — a term's names
        are unique, because :meth:`ActionTerm._reject_multi_dof_joints` refuses the one joint kind
        whose name repeats — but the index is what Genesis dispatches on, so the check does not
        rest on that separate invariant holding. Names are what the message reports, deduped.

        A term may command DOFs it has no name for: a free base is six DOFs of one joint, which
        ``_reject_multi_dof_joints`` refuses to let anyone spell in ``dofs_name``, so
        :class:`~eden.managers.terms.actions.FreeBasePoseController` leaves that list empty and
        resolves ``dofs_idx_local`` itself. Those DOFs are commanded like any other and collide
        like any other, so the walk is over the *indices*, and the message falls back to the local
        index for a DOF the term cannot name.

        Checked here rather than in ``ActionTerm.build``, for the same reason as
        :meth:`_reject_no_op_terms`: a term sees only itself, and the names the user has to go and
        edit are the config keys, which only the manager knows.
        """
        owner: dict[tuple[str, int], str] = {}
        shared: dict[tuple[str, str, str], list[str]] = {}
        for term_name, term in self._terms.items():
            if term.num_dofs == 0:
                # The welding terms, whose ``dofs_idx_local`` their own ``__init__`` fixes empty:
                # a term that commands no DOF cannot collide with one.
                continue
            # Walked by index, not `zip(dofs_idx_local, dofs_name)`: a term whose `dofs_name` is
            # empty because it commands DOFs Genesis names only as a joint (a free base) would
            # zip to nothing and register no ownership at all, so two of them on one entity
            # would pass this check and then fight, which is the whole failure it exists to stop.
            for position, dof_idx in enumerate(term.dofs_idx_local.tolist()):
                dof_name = term.dofs_name[position] if position < len(term.dofs_name) else f"DOF {dof_idx}"
                key = (term.entity_name, dof_idx)
                if key not in owner:
                    owner[key] = term_name
                    continue
                shared.setdefault((term.entity_name, owner[key], term_name), []).append(dof_name)
        if not shared:
            return
        lines = "\n".join(
            f"  '{first}' ({type(self._terms[first]).__name__}) and '{second}' "
            f"({type(self._terms[second]).__name__}) both command "
            f"{list(dict.fromkeys(dofs_name))} of entity '{entity_name}'"
            for (entity_name, first, second), dofs_name in shared.items()
        )
        raise ValueError(
            f"Action terms overlap on the DOFs they command:\n{lines}\n"
            f"A DOF takes one command: each term writes its own control target every physics "
            f"sub-step and the last write wins, so on each line above the first term's command "
            f"is discarded on the shared DOFs.\n"
            f"Fix: remove the listed joints from one of the two terms' `dofs_name`, or drop that "
            f"term from action_options. A DOF reported as 'DOF <i>' is one its term does not name "
            f"— a free base, which `FreeBasePoseController` resolves itself — so there dropping "
            f"one of the two terms is the only fix. See docs/guides/terms.md.\n"
            f"Do NOT reorder the terms so the one you want lands last — the other one keeps "
            f"drawing its slice of the action vector, so the policy goes on emitting (and the "
            f"action-rate rewards go on grading) columns that steer nothing."
        )

    def summary(self) -> str:
        return self._format_summary_table(
            title=f"Active Action Terms (shape: {self.total_action_dim})",
            field_names=["Index", "Name", "Dimension", "Slice"],
            rows=(
                [index, name, term.action_dim, self._term_slice[name]]
                for index, (name, term) in enumerate(self._terms.items())
            ),
            align={"Name": "l", "Dimension": "r", "Slice": "r"},
        )

    @cached_property
    def total_action_dim(self) -> int:
        return sum(self.action_term_dim)

    @cached_property
    def action_term_dim(self) -> list[int]:
        """The order corresponds to the .active_terms property."""
        return [term.action_dim for term in self._terms.values()]

    @cached_property
    def total_dofs_dim(self) -> int:
        return sum(self.dofs_term_dim)

    @cached_property
    def dofs_term_dim(self) -> list[int]:
        """The order corresponds to the .active_terms property."""
        return [term.num_dofs for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """The raw action vector received this step, shape ``(num_envs, total_action_dim)``."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The raw action vector received on the previous step, shape ``(num_envs, total_action_dim)``."""
        return self._prev_action

    @property
    def processed_actions(self) -> dict[str, torch.Tensor]:
        """Each term's :attr:`ActionTerm.processed_action`, keyed by term name.

        A mapping rather than one concatenated tensor (unlike :attr:`action`): terms
        command different quantities with different widths and dtypes, so there is no
        meaningful flat layout to concatenate them into. Hence the plural name — it
        keeps the type visible at the call site, so indexing it like the flat
        :attr:`action` fails with a clear ``AttributeError`` on the singular spelling
        rather than a puzzling ``KeyError`` from a tensor-style slice on a dict.
        """
        return {name: term.processed_action for name, term in self._terms.items()}

    def clear_action_history(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        """Zero the raw actions :meth:`reset` zeroes, the manager's and each term's, current and previous.

        For a caller that restarts what a policy observes without resetting the env: a deployment's
        START or policy switch, where the robot keeps running. A ``last_action`` observation then reads
        zeros, as it does after an env reset. :meth:`reset` itself does more, which such a caller must
        not: it resets every term's modifiers and an ``ACCUMULATE`` target, state a reset-mode DR event
        rewrites right after it in an env reset and that nothing rewrites on a robot.
        """
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._action[envs_idx] = 0.0
        for term in self._terms.values():
            if term._raw_action is not None:
                term._raw_action[envs_idx] = 0.0
                term._prev_action[envs_idx] = 0.0

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, float]:
        if envs_idx is None:
            envs_idx = slice(None)

        # Reset action history.
        # NOTE: use assignment here (it works for both slice and tensor index), not .zero_()
        self._prev_action[envs_idx] = 0.0
        self._action[envs_idx] = 0.0

        # Reset action terms.
        for term in self._terms.values():
            term.reset(envs_idx=envs_idx)
        return {}

    def _validate_dict_action(self, action: dict[str, torch.Tensor]) -> None:
        """Name the offending term on a dict-action key/width mismatch (smoke-only, see ``compute``).

        The flat (single-tensor) path is validated every step by the total-width check in
        ``compute``; the per-term dict path is checked here instead so real training carries no
        per-step cost (the sim-facing command width is still guarded once per term by
        :meth:`ActionTerm.verify_command_shape`).
        """
        for term_name, term_action in action.items():
            if term_name not in self._terms:
                raise KeyError(f"Unknown action term '{term_name}'. Active action terms: {list(self._terms)}.")
            expected = self._terms[term_name].action_dim
            if term_action.shape[-1] != expected:
                raise ValueError(
                    f"Action dict for term '{term_name}' has width {term_action.shape[-1]}, "
                    f"but the term expects {expected}."
                )

    def compute(self, action: torch.Tensor | dict[str, torch.Tensor]) -> None:
        if isinstance(action, dict):
            if self._smoke:
                # Smoke-only: name the offending term before term.compute turns a width mismatch
                # into a bare torch broadcast error (or an unknown key into a bare KeyError).
                self._validate_dict_action(action)
            for term_name, term_action in action.items():
                # The tensor form below copies into the device buffer, so a host-built action works there; the
                # dict form hands each term its tensor as given. Moved here (a no-op for one already on the
                # device) so both forms accept `torch.tensor([...])` — a CPU tensor, since Eden sets no torch
                # default device.
                term_action = term_action.to(self.device)
                term = self._terms[term_name]
                term.compute(term_action)
                if term._accepts_missing_command_:
                    term_action = torch.where(torch.isnan(term_action), term.action, term_action)
                self._prev_action[:, self._term_slice[term_name]] = self._action[:, self._term_slice[term_name]]
                self._action[:, self._term_slice[term_name]] = term_action
        else:
            if self.total_action_dim != action.shape[1]:
                # Name the per-term layout so the caller can see which term owns which columns —
                # the mismatch is otherwise reported only as two totals.
                layout = ", ".join(f"{name}={term.action_dim}" for name, term in self._terms.items())
                raise ValueError(
                    f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}. "
                    f"Expected per-term layout: [{layout}]."
                )
            self._prev_action[:] = self._action
            self._action[:] = action
            for term_name, term in self._terms.items():
                term_actions = self._action[:, self._term_slice[term_name]]
                term.compute(term_actions)
                if term._accepts_missing_command_:
                    self._action[:, self._term_slice[term_name]] = torch.where(
                        torch.isnan(term_actions), term.action, term_actions
                    )

    def apply_actions(self) -> None:
        for term in self._terms.values():
            term.apply_actions()
