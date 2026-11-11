"""Multi-arm demonstrators: one routine per arm, run side by side in every env, handing off through a blackboard.

A two-arm task is scripted as two routines, each written for its own arm against its own
:class:`~eden.extensions.scripted_demos.tools.Tools` handle, exactly like a single-arm routine. They
run concurrently — each arm resumes its routine when its own motion ends — and coordinate through
:attr:`Tools.shared <eden.extensions.scripted_demos.tools.Tools.shared>`, a dict both routines of an
env share for the episode. A hand-off is one routine writing a key and the other waiting for it::

    def giver(tools):
        ...  # carry the object to the hand-off pose
        tools.shared["offered"] = True
        while not tools.shared.get("taken"):
            yield tools.wait(0.1)
        yield tools.open_gripper()


    def taker(tools):
        while not tools.shared.get("offered"):
            yield tools.wait(0.1)
        ...  # close on the object
        tools.shared["taken"] = True

:class:`MultiArmDemonstrator` runs one
:class:`~eden.extensions.scripted_demos.routines.RoutineDemonstrator` per arm, merges their actions
into the task's, and has the interface ``eden datagen`` drives (``reset`` / ``update_subgoals`` /
``act`` / ``done`` / ``subgoal_counts`` / ``trace``). An env's task is complete when every arm's
routine has returned; an arm that finishes first holds still.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch

from eden.extensions.scripted_demos.routines import Routine, RoutineDemonstrator

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

#: Seeds of the arms' random streams are this far apart, so two arms do not draw the same variations.
_ARM_SEED_STRIDE = 1_000_003
_TRACE_TIME = re.compile(r"^\s*(-?\d+(?:\.\d+)?) s")


class MultiArmDemonstrator:
    """Runs one routine per arm in every env, batched, with a blackboard per env shared between the arms.

    Parameters
    ----------
    env : EnvBase
        A built environment holding each arm at its scene entry, each moved by one IK term and one
        gripper term, and no other action term.
    routines : dict[str, Routine]
        Scene entry of each arm -> the routine that drives it. Each is called with that arm's
        :class:`~eden.extensions.scripted_demos.tools.Tools` at the start of every episode.
    seed : int
        Seeds every arm's per-episode random streams (each arm its own).
    **arm_options : Any
        Passed to every arm's :class:`~eden.extensions.scripted_demos.routines.RoutineDemonstrator`
        (``workspace``, ``peak_speed``, ``holding_closure``, ...). An option that differs per arm is a
        ``dict`` keyed by the arm's scene entry, e.g. ``workspace={"robot0": ..., "robot1": ...}``.

    Raises
    ------
    ValueError
        Fewer than two arms, or a per-arm option that does not name exactly the arms.
    TypeError
        An action term no arm drives.
    """

    def __init__(self, env: EnvBase, routines: dict[str, Routine], *, seed: int = 0, **arm_options: Any):
        if len(routines) < 2:
            raise ValueError(
                f"A MultiArmDemonstrator drives two or more arms, got routines for {sorted(routines)}; a single arm "
                f"is a RoutineDemonstrator(env, routine, rig='<slot>')."
            )
        for name, value in arm_options.items():
            if isinstance(value, dict) and set(value) != set(routines):
                raise ValueError(
                    f"Per-arm option '{name}' is given for {sorted(value)}, but the arms are {sorted(routines)}; key it "
                    f"by every arm's scene entry, or pass one value for all of them."
                )
        self.env = env
        self.arms: dict[str, RoutineDemonstrator] = {
            slot: RoutineDemonstrator(
                env,
                routine,
                rig=slot,
                seed=seed + i * _ARM_SEED_STRIDE,
                exclusive=False,
                **{name: value[slot] if isinstance(value, dict) else value for name, value in arm_options.items()},
            )
            for i, (slot, routine) in enumerate(routines.items())
        }
        self._check_action_terms()
        self.num_envs = env.num_envs
        self.device = env.device

    def _check_action_terms(self) -> None:
        """Every action term must be one arm's IK or gripper term: the demonstrator has no command for any other."""
        driven = {
            term: slot
            for slot, arm in self.arms.items()
            for term in (arm.driver.arm_term_name, arm.driver.gripper_term_name)
        }
        undriven = sorted(set(self.env.action_manager.active_terms) - set(driven))
        if undriven:
            raise TypeError(
                f"Action terms {undriven} are driven by none of the arms {sorted(self.arms)}; a multi-arm demonstrator "
                f"commands only each arm's IK and gripper term ({driven}), and every term needs a command each step."
            )

    @property
    def done(self) -> torch.Tensor:
        """``(N,)`` bool: every arm's routine has returned — this env's task is complete."""
        done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for arm in self.arms.values():
            done = done & arm.done
        return done

    @property
    def subgoal_counts(self) -> dict[str, int]:
        """Motions ended so far, by ``<arm>/<tool>:<status>`` — the generation report's tally."""
        return {f"{slot}/{key}": count for slot, arm in self.arms.items() for key, count in arm.subgoal_counts.items()}

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        """Start fresh routines on every arm for ``envs_idx``, the arms of each env sharing one new blackboard.

        Call it when :meth:`~eden.extensions.scripted_demos.routines.RoutineDemonstrator.reset` would be called.
        """
        if envs_idx is None:
            env_ids = list(range(self.num_envs))
        else:
            env_ids = (
                envs_idx.nonzero(as_tuple=False).flatten() if envs_idx.dtype == torch.bool else envs_idx
            ).tolist()
        boards = {env_id: {} for env_id in env_ids}
        for arm in self.arms.values():
            arm.reset(envs_idx, blackboards=boards)

    def update_subgoals(self) -> torch.Tensor:
        """Advance every arm's routines (see :meth:`RoutineDemonstrator.update_subgoals`); returns :attr:`done`."""
        for arm in self.arms.values():
            arm.update_subgoals()
        return self.done

    def act(self) -> dict[str, torch.Tensor]:
        """The action dict for this control step: every arm's IK and gripper commands."""
        actions: dict[str, torch.Tensor] = {}
        for arm in self.arms.values():
            actions.update(arm.act())
        return actions

    def trace(self, env_id: int) -> list[str]:
        """This episode's record for ``env_id``: every arm's lines, prefixed with the arm, in time order."""
        lines = [(slot, line) for slot, arm in self.arms.items() for line in arm.trace(env_id)]

        def at(entry: tuple[str, str]) -> float:
            match = _TRACE_TIME.match(entry[1])
            return float(match.group(1)) if match else 0.0

        return [f"[{slot}] {line}" for slot, line in sorted(lines, key=at)]
