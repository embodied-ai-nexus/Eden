"""Binary (open/close) gripper action terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.utils.string import resolve_matching_names, term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


@ACTION_TERM_REGISTRY.register()
class BinaryJointController(ActionTerm):
    """A binary joint controller for a parallel-jaw gripper.

    When the action is applied, the joints are opened or closed.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: list[str]
        The names of the DOFs to control.
    open_action: float | dict[str, float]
        The action to open the joints.
    close_action: float | dict[str, float]
        The action to close the joints.
    scale: float
        The scale to apply to the actions.

    Note
    ----
    action is a boolean tensor, True: close, False: open
    if float is used, >0: close, <=0: open
    """

    open_action: float | dict[str, float] = 1.0
    close_action: float | dict[str, float] = 0.0
    scale: float = 1.0

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self._open_action: torch.Tensor | None = None
        self._close_action: torch.Tensor | None = None

    def build(self) -> None:
        super().build()

        self._open_action = torch.zeros(1, self.num_dofs, device=self.device)
        self._close_action = torch.zeros(1, self.num_dofs, device=self.device)

        # Keys resolve to positions in this term's own `dofs_name` (the layout of the two rows), but a key that is
        # exactly a joint's name is literal against every joint of the entity: another term's or a passive
        # `pin.1` selects nothing here, rather than this term's `pin_1` as a regex.
        every_joint = [joint.name for joint in self.entity.joints]

        if isinstance(self.open_action, float):
            self._open_action[:] = self.open_action
        elif isinstance(self.open_action, dict):
            for dofs_name, open_action in self.open_action.items():
                term_idx, _ = resolve_matching_names(
                    dofs_name, self.dofs_name, preserve_order=True, literal_names=every_joint
                )
                self._open_action[:, term_idx] = open_action
        else:
            raise ValueError(f"{term_subject(self)}: unsupported open_action type {type(self.open_action)}.")

        if isinstance(self.close_action, float):
            self._close_action[:] = self.close_action
        elif isinstance(self.close_action, dict):
            for dofs_name, close_action in self.close_action.items():
                term_idx, _ = resolve_matching_names(
                    dofs_name, self.dofs_name, preserve_order=True, literal_names=every_joint
                )
                self._close_action[:, term_idx] = close_action
        else:
            raise ValueError(f"{term_subject(self)}: unsupported close_action type {type(self.close_action)}.")

    @property
    def open_targets(self) -> torch.Tensor:
        """``(1, num_dofs)`` joint targets sent to open the gripper.

        Column ``i`` is ``dofs_name[i]``, driven at ``genesis_dofs_idx[i]``.
        """
        return self._open_action

    @property
    def close_targets(self) -> torch.Tensor:
        """``(1, num_dofs)`` joint targets sent to close the gripper.

        Column ``i`` is ``dofs_name[i]``, driven at ``genesis_dofs_idx[i]``.
        """
        return self._close_action

    @property
    def action_dim(self) -> int:
        return 1

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        if actions.dtype == torch.bool:
            close_mask = actions
        else:
            close_mask = (actions * self.scale) > 0
        self._processed_action = (
            self._open_action.repeat(self.num_envs, 1) * (~close_mask)
            + self._close_action.repeat(self.num_envs, 1) * close_mask
        )

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)


GripperInput = Literal["sign", "open_sign", "closedness", "openness"]


@ACTION_TERM_REGISTRY.register()
class GripperDecisionController(BinaryJointController):
    """Gripper controller that decides open/close from a policy scalar.

    Policy servers describe the gripper in different units and conventions (``> 0`` means
    close for some, open for others; a closedness or an openness in ``[0, 1]`` for others),
    often with noise that wants hysteresis, a hold, or a latch. This term takes the raw
    scalar and decides, so a client only slices its action vector. With the defaults it
    behaves exactly like :class:`BinaryJointController`.

    Parameters
    ----------
    input: {"sign", "open_sign", "closedness", "openness"}
        Meaning of the scalar: ``sign`` closes on ``> 0`` (parent behaviour), ``open_sign``
        opens on ``> 0``, ``closedness`` / ``openness`` are ``[0, 1]`` amounts.
    threshold: float
        Binarisation point for ``closedness`` / ``openness``: a closedness strictly above it
        closes; an openness strictly above it opens.
    open_threshold, close_threshold: float | None
        Hysteresis on closedness: once closed, open only below ``open_threshold``; once
        open, close only above ``close_threshold``. Set both to enable (they replace
        ``threshold``); a value exactly on a threshold keeps the current state.
    hold_steps: int
        After a state change, the new state lasts at least this many decisions, counting the
        one that changed it (hysteresis mode only; ``0`` and ``1`` do not hold).
    sticky_repeats: int
        Fractal / Google-robot latch (SimplerEnv CogACT / SpatialVLA ``google_robot``): the
        relative command is ``prev - openness`` (``> 0`` closes), with ``prev`` taken from the
        first command and updated only when the latch triggers. A jump of more than 0.5
        triggers the latch, which repeats the jump for this many steps; otherwise the small
        relative command itself is sent. The command then drives ManiSkill's
        ``gripper_pd_joint_target_delta_pos`` controller: the target (in closedness) starts at
        the measured gripper, moves by ``1.31 * clip(command, -1, 1) / 1.3`` per step, clipped
        to the joint range ``[-0.0101, 1.31] / 1.3``, and holds when the move is below
        ``1e-3 / 1.3``. The fingers follow that target continuously. Needs an ``openness`` /
        ``closedness`` input. ``0`` disables.
    zero_holds: bool
        With a ``sign`` / ``open_sign`` input, a command of exactly ``0`` keeps the previous
        state instead of opening (robosuite's gripper steps by ``sign(action)``).
    continuous: bool
        Interpolate the DOF targets between ``open_action`` and ``close_action`` by the
        closedness instead of binarising; ``snap_threshold`` promotes closedness above it
        to fully closed.
    snap_threshold: float | None
        See ``continuous``.
    on_missing: {"hold", "measured"}
        What a ``NaN`` command (the server omitted the gripper) does: ``hold`` keeps the
        previous target; ``measured`` commands the gripper's measured closedness every such
        step, as the DROID Galaxea reference client does.

    Notes
    -----
    Under ``on_missing="hold"``, a ``NaN`` before any command was received is seeded from
    the measured DOF position (the gripper is where it physically is).
    In place of a ``NaN``, :attr:`action` (and so the ActionManager's buffers) records the
    command acted on, in the ``input``'s units: that measured seed, or the held previous command.
    Per-env decision state resets with the env (``reset(envs_idx)``).
    """

    input: GripperInput = "sign"
    threshold: float = 0.5
    open_threshold: float | None = None
    close_threshold: float | None = None
    hold_steps: int = 0
    sticky_repeats: int = 0
    zero_holds: bool = False
    continuous: bool = False
    snap_threshold: float | None = None
    on_missing: Literal["hold", "measured"] = "hold"

    #: A ``NaN`` command means "the server omitted the gripper" (see ``on_missing``).
    _accepts_missing_command_: ClassVar[bool] = True

    def build(self) -> None:
        super().build()
        if self.input not in ("sign", "open_sign", "closedness", "openness"):
            raise ValueError(
                f"{term_subject(self)}: input={self.input!r}; expected sign, open_sign, closedness or openness."
            )
        if (self.open_threshold is None) != (self.close_threshold is None):
            raise ValueError(f"{term_subject(self)}: set both open_threshold and close_threshold or neither.")
        if self.open_threshold is not None and self.close_threshold < self.open_threshold:
            raise ValueError(
                f"{term_subject(self)}: close_threshold ({self.close_threshold}) must be >= "
                f"open_threshold ({self.open_threshold})."
            )
        if self.hold_steps < 0 or self.sticky_repeats < 0:
            raise ValueError(f"{term_subject(self)}: hold_steps and sticky_repeats must be non-negative.")
        if self.continuous and self.input in ("sign", "open_sign"):
            raise ValueError(f"{term_subject(self)}: continuous mode needs a closedness or openness input.")
        if self.snap_threshold is not None and not 0.0 <= self.snap_threshold <= 1.0:
            raise ValueError(f"{term_subject(self)}: snap_threshold must be in [0, 1].")
        if self.on_missing not in ("hold", "measured"):
            raise ValueError(f"{term_subject(self)}: on_missing={self.on_missing!r}; expected hold or measured.")
        # Settings that the selected mode would silently ignore.
        hysteresis = self.open_threshold is not None
        if self.hold_steps > 0 and not hysteresis:
            raise ValueError(
                f"{term_subject(self)}: hold_steps only applies with open_threshold/close_threshold (hysteresis)."
            )
        if hysteresis and self.input in ("sign", "open_sign"):
            raise ValueError(
                f"{term_subject(self)}: hysteresis thresholds need a closedness or openness input, not {self.input!r}."
            )
        if self.snap_threshold is not None and not self.continuous:
            raise ValueError(f"{term_subject(self)}: snap_threshold only applies with continuous=True.")
        if self.continuous and (hysteresis or self.sticky_repeats > 0):
            raise ValueError(
                f"{term_subject(self)}: continuous mode cannot be combined with hysteresis or sticky_repeats."
            )
        if self.sticky_repeats > 0 and (hysteresis or self.input in ("sign", "open_sign")):
            raise ValueError(
                f"{term_subject(self)}: sticky_repeats needs an openness/closedness input and no hysteresis."
            )
        if self.zero_holds and self.input not in ("sign", "open_sign"):
            raise ValueError(f"{term_subject(self)}: zero_holds only applies to a sign or open_sign input.")
        if self.on_missing == "measured" and not self.continuous:
            # Re-deciding from a stalled grasp's measured position would re-binarize (or re-latch) it open.
            raise ValueError(f"{term_subject(self)}: on_missing='measured' needs continuous=True.")
        n = self.num_envs
        dev = self.device
        self._closed = torch.zeros(n, dtype=torch.bool, device=dev)  # starts open
        self._hold_remaining = torch.zeros(n, dtype=torch.long, device=dev)
        self._sticky_prev = torch.full((n,), float("nan"), device=dev)  # unset until the first command
        self._sticky_on = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sticky_count = torch.zeros(n, dtype=torch.long, device=dev)
        self._sticky_action = torch.zeros(n, device=dev)
        self._sticky_target = torch.zeros(n, device=dev)  # delta-controller target, in closedness
        self._seen_command = torch.zeros(n, dtype=torch.bool, device=dev)
        self._last_processed = self._open_action.repeat(n, 1).clone()
        # Fixed at build time; resolved once so the per-step measured read never syncs with the device.
        self._measure_span_zero = bool(torch.isclose(self._open_action[0, 0], self._close_action[0, 0]))
        self.last_closedness = torch.zeros(n, device=dev)
        self._last_raw: torch.Tensor | None = None
        self._last_valid: torch.Tensor | None = None

    # -- decision -------------------------------------------------------

    def _closedness_from_input(self, raw: torch.Tensor) -> torch.Tensor:
        if self.input == "sign":
            return ((raw * self.scale) > 0).float()
        if self.input == "open_sign":
            return ((raw * self.scale) <= 0).float()
        if self.input == "closedness":
            return raw
        return 1.0 - raw  # openness

    def _measured_closedness(self) -> torch.Tensor:
        """Closedness of the physical gripper, from the first controlled DOF's position."""
        pos = self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)[:, 0]
        if self._measure_span_zero:
            return torch.zeros_like(pos)
        open_pos, close_pos = self._open_action[0, 0], self._close_action[0, 0]
        return ((pos - open_pos) / (close_pos - open_pos)).clamp(0.0, 1.0)

    # ManiSkill2_real2sim google_robot gripper: joint range [-0.0001, 1.3] (URDF), action range
    # [-1.31, 1.31] (normalize_action), target clipped to the range +/- 0.01, |delta| < 1e-3 repeats.
    _GR_DELTA_SCALE = 1.31 / 1.3
    _GR_TARGET_LO = (-0.0001 - 0.01) / 1.3
    _GR_TARGET_HI = (1.3 + 0.01) / 1.3
    _GR_SMALL = 1e-3 / 1.3

    def _sticky_step(self, closedness: torch.Tensor, measured: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """SimplerEnv CogACT/SpatialVLA google_robot latch driving a delta joint target; returns closedness."""
        openness = 1.0 - closedness
        first = valid & torch.isnan(self._sticky_prev)
        self._sticky_prev = torch.where(first, openness, self._sticky_prev)  # relative 0 on the first command
        self._sticky_target = torch.where(first, measured, self._sticky_target)  # controller reset: target = qpos
        relative = self._sticky_prev - openness  # > 0 closes (google robot convention)
        trigger = (relative.abs() > 0.5) & ~self._sticky_on & valid
        self._sticky_on = self._sticky_on | trigger
        self._sticky_action = torch.where(trigger, relative, self._sticky_action)
        self._sticky_prev = torch.where(trigger, openness, self._sticky_prev)
        active = self._sticky_on & valid
        self._sticky_count = torch.where(active, self._sticky_count + 1, self._sticky_count)
        command = torch.where(active, self._sticky_action, relative)
        expired = (self._sticky_count >= self.sticky_repeats) & valid
        self._sticky_on = self._sticky_on & ~expired
        self._sticky_count = torch.where(expired, torch.zeros_like(self._sticky_count), self._sticky_count)
        self._sticky_action = torch.where(expired, torch.zeros_like(self._sticky_action), self._sticky_action)
        delta = self._GR_DELTA_SCALE * command.clamp(-1.0, 1.0)
        moved = (self._sticky_target + delta).clamp(self._GR_TARGET_LO, self._GR_TARGET_HI)
        update = valid & (delta.abs() >= self._GR_SMALL)
        self._sticky_target = torch.where(update, moved, self._sticky_target)
        return self._sticky_target.clamp(0.0, 1.0)

    def _decide(self, raw: torch.Tensor, closedness: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Update per-env state for ``valid`` envs and return the closed mask ``(N,)``."""
        prev = self._closed
        if self.input in ("sign", "open_sign"):
            closed = closedness > 0.5
            if self.zero_holds:
                closed = torch.where(raw == 0, prev, closed)
        elif self.open_threshold is None:
            if self.input == "openness":
                closed = ~(raw > self.threshold)  # open iff openness strictly above the threshold
            else:
                closed = closedness > self.threshold
        else:
            holding = self._hold_remaining > 0
            open_now = prev & (closedness < self.open_threshold)
            close_now = ~prev & (closedness > self.close_threshold)
            proposed = torch.where(prev, ~open_now, close_now)
            closed = torch.where(holding, prev, proposed)
            changed = (closed != prev) & ~holding & valid
            self._hold_remaining = torch.where(holding & valid, self._hold_remaining - 1, self._hold_remaining)
            if self.hold_steps > 0:
                self._hold_remaining = torch.where(
                    changed, torch.full_like(self._hold_remaining, self.hold_steps - 1), self._hold_remaining
                )
        return torch.where(valid, closed, prev)

    def compute(self, actions: torch.Tensor) -> None:
        if actions.dtype == torch.bool:
            actions = actions.float() * 2.0 - 1.0  # True -> +1 (close), False -> -1 (open) in sign terms
            if self.input != "sign":
                raise ValueError(f"{term_subject(self)}: boolean commands are only valid with input='sign'.")
        raw = actions[:, 0].to(torch.float32)
        self._prev_action[:] = self._raw_action
        valid = ~torch.isnan(raw)
        missing = ~valid

        # Missing command: the gripper is wherever it physically is (first miss, or every miss
        # under on_missing="measured"). Computed unconditionally to avoid a host sync per step.
        seed = ~valid if self.on_missing == "measured" else ~valid & ~self._seen_command
        measured = self._measured_closedness()
        if self.input == "closedness":
            seeded = measured
        elif self.input == "openness":
            seeded = 1.0 - measured
        else:  # sign conventions: raw * scale > 0 means close for "sign", open for "open_sign"
            closed_sign = torch.where(measured > 0.5, torch.ones_like(raw), -torch.ones_like(raw))
            closed_sign = closed_sign if self.scale >= 0 else -closed_sign
            seeded = closed_sign if self.input == "sign" else -closed_sign
        raw = torch.where(seed, seeded, raw)
        valid = valid | seed
        self._seen_command |= valid
        # Record the command acted on, in the input's units: the seeded measured value, else (a held
        # miss) the previous command. The ActionManager stores this in place of the NaN
        # (_accepts_missing_command_), so last-action observations and recorders see what was done.
        self._raw_action[:, 0] = torch.where(missing & ~seed, self._prev_action[:, 0], raw)

        closedness = self._closedness_from_input(raw)
        if self.sticky_repeats > 0:
            cmd = torch.where(valid, self._sticky_step(closedness, measured, valid), self.last_closedness)
            processed = self._open_action + (self._close_action - self._open_action) * cmd.unsqueeze(-1)
            self._closed = cmd > 0.5
        elif self.continuous:
            cmd = closedness.clamp(0.0, 1.0)
            if self.snap_threshold is not None:
                cmd = torch.where(cmd > self.snap_threshold, torch.ones_like(cmd), cmd)
            cmd = torch.where(valid, cmd, self.last_closedness)
            processed = self._open_action + (self._close_action - self._open_action) * cmd.unsqueeze(-1)
            self._closed = cmd > 0.5
        else:
            self._closed = self._decide(raw, closedness, valid)
            cmd = self._closed.float()
            mask = self._closed.unsqueeze(-1)
            processed = torch.where(
                mask, self._close_action.expand(self.num_envs, -1), self._open_action.expand(self.num_envs, -1)
            )
        processed = torch.where(valid.unsqueeze(-1), processed, self._last_processed)
        self._last_processed = processed.clone()
        self.last_closedness = cmd
        self._processed_action = processed
        self._last_raw = raw
        self._last_valid = valid

    @property
    def last_decision(self) -> dict[str, Any] | None:
        """Env 0's last decision, for debugging (reading it syncs with the device)."""
        if self._last_raw is None:
            return None
        closed = bool(self._closed[0])
        return {
            "raw": float(self._last_raw[0]),
            "valid": bool(self._last_valid[0]),
            "closedness": float(self.last_closedness[0]),
            "closed": closed,
            "intent": "close" if closed else "open",
            "hold_remaining": int(self._hold_remaining[0]),
        }

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        super().reset(envs_idx)
        idx = slice(None) if envs_idx is None else envs_idx
        self._closed[idx] = False
        self._hold_remaining[idx] = 0
        self._sticky_prev[idx] = float("nan")
        self._sticky_on[idx] = False
        self._sticky_count[idx] = 0
        self._sticky_action[idx] = 0.0
        self._sticky_target[idx] = 0.0
        self._seen_command[idx] = False
        self._last_processed[idx] = self._open_action
        self.last_closedness[idx] = 0.0
