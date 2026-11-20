"""The deployment safety layer: a filter on the policy's joint targets before each send, and a guard on each reading.

:class:`~eden.extensions.deployment.base.DeploymentBase` builds one :class:`SafetyLayer` when its ``safety`` option
is set (``eden deploy --safety FILE``, a :class:`~eden.options.extensions.deployment.DeploySafetyOptions`), and
calls it from :meth:`~eden.extensions.deployment.base.DeploymentBase.step` on every control tick:

1. Before the send, on the ticks the policy drives, :meth:`SafetyLayer.filter` clamps each joint target into the
   built robot's joint range (shrunk by ``dofs_pos_margin``) and caps how far it may move from the joint's measured
   position and from the target sent on the previous tick. It also counts the consecutive ticks on which the
   policy's raw target was past a joint limit.
2. After the reading has been synced onto the scene, :meth:`SafetyLayer.check` compares it with the limits: each
   joint's tracking error, the interval since the previous reading, each link box, and the count from step 1. On a
   tick where the operator state machine moves the robot (the INIT ramp or the policy), a crossed limit raises
   :class:`~eden.extensions.deployment.base.DeployStop` with every limit crossed on that tick. The deploy loop does
   not catch it, so the run ends and ``close()`` settles the robot. In the STOP hold the layer records the values
   and stops nothing, so an operator can move a robot that boots outside a box back in by hand.

Each tick's values join ``state.extra`` under ``safety.*``, so ``--debug_record`` keeps them next to the command and
the reading. The record is where to tune a limit: run with the limit off and read how close the robot came.

The same layer runs under ``eden deploy --mujoco``, so a policy can be run through the limits in MuJoCo before it
runs on the robot.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import time

import numpy as np

from eden.extensions.deployment.utils.state import RobotState
from eden.options.extensions.deployment import DeploySafetyOptions
from eden.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from eden.extensions.deployment.base import DeploymentBase


def _per_dof(value: float | dict[str, float] | None, dofs_name: Sequence[str], field: str) -> np.ndarray | None:
    """A per-joint safety field as an array over ``dofs_name``: ``inf`` on joints it does not limit, ``None`` if unset."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return np.full(len(dofs_name), float(value))
    resolved = np.full(len(dofs_name), np.inf)
    try:
        idx, _, values = resolve_matching_names_values(value, list(dofs_name), preserve_order=True)
    except ValueError as exc:
        exc.add_note(
            f"[deploy] Raised resolving DeploySafetyOptions.{field} against the deployed joints {list(dofs_name)}."
        )
        raise
    resolved[idx] = values
    return resolved


def _fmt(vector: np.ndarray) -> str:
    return "(" + ", ".join("open" if not np.isfinite(v) else f"{v:.3f}" for v in vector) + ")"


@dataclass
class _LinkBox:
    """A :class:`~eden.options.extensions.deployment.LinkBoxOptions` resolved against the built scene."""

    name: str
    entity_name: str
    links_name: list[str]
    links_idx_local: list[int]
    lower: np.ndarray
    upper: np.ndarray


class SafetyLayer:
    """Clamp a deployment's policy targets and stop the run when a reading crosses a limit.

    Build it with :meth:`for_deployment`, which resolves every joint and link name against the built robot, so a
    misnamed one raises when the deployment is constructed rather than on the robot.

    Parameters
    ----------
    options : DeploySafetyOptions
        The limits.
    dofs_name : Sequence[str]
        The deployed joints, in the order of every command and reading.
    dofs_pos_limits : tuple[np.ndarray, np.ndarray]
        The built robot's joint range over ``dofs_name`` (``-inf`` / ``inf`` where a joint has none).
    link_boxes : list[_LinkBox]
        The resolved boxes.
    entities : dict
        The scene's entities by name (``env.entities``), read for the link positions on each check.
    clock : Callable[[], float]
        Seconds, for ``max_read_interval_s``. ``time.monotonic`` by default.
    """

    def __init__(
        self,
        options: DeploySafetyOptions,
        dofs_name: Sequence[str],
        dofs_pos_limits: tuple[np.ndarray, np.ndarray],
        link_boxes: list[_LinkBox],
        entities: Any,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.options = options
        self.dofs_name = list(dofs_name)
        self.lower = np.asarray(dofs_pos_limits[0], dtype=np.float64)
        self.upper = np.asarray(dofs_pos_limits[1], dtype=np.float64)
        margin = options.dofs_pos_margin
        too_narrow = np.flatnonzero(self.upper - self.lower < 2.0 * margin)
        if too_narrow.size:
            raise ValueError(
                f"DeploySafetyOptions.dofs_pos_margin={margin} leaves no range on joint(s) "
                f"{[self.dofs_name[i] for i in too_narrow]}, whose range is narrower than twice the margin: "
                f"{[(float(self.lower[i]), float(self.upper[i])) for i in too_narrow]}. Lower the margin."
            )
        self.clamp_lower = self.lower + margin
        self.clamp_upper = self.upper - margin
        self.step_from_measured = _per_dof(
            options.max_dofs_step_from_measured, dofs_name, "max_dofs_step_from_measured"
        )
        self.step_from_sent = _per_dof(options.max_dofs_step_from_sent, dofs_name, "max_dofs_step_from_sent")
        self.max_tracking_error = _per_dof(options.max_dofs_tracking_error, dofs_name, "max_dofs_tracking_error")
        self.link_boxes = link_boxes
        self.entities = entities
        self.clock = clock
        self.reset()

    @classmethod
    def for_deployment(cls, dep: DeploymentBase, options: DeploySafetyOptions) -> SafetyLayer:
        """The layer for ``dep``: its joints and their built range, and each link box resolved on its scene."""
        entities = dep._env.entities
        boxes = []
        for name, box in options.link_boxes.items():
            if box.entity_name not in entities:
                raise ValueError(
                    f"DeploySafetyOptions.link_boxes['{name}'] names entity_name='{box.entity_name}', which the scene "
                    f"does not have. Its entities: {sorted(entities.keys())}."
                )
            entity = entities[box.entity_name]
            try:
                links_name, links_idx_local = entity.find_named_links_idx_local(list(box.links))
            except ValueError as exc:
                exc.add_note(f"[deploy] Raised resolving DeploySafetyOptions.link_boxes['{name}'].links.")
                raise
            boxes.append(
                _LinkBox(
                    name=name,
                    entity_name=box.entity_name,
                    links_name=links_name,
                    links_idx_local=links_idx_local,
                    lower=np.array([-np.inf if v is None else v for v in box.lower], dtype=np.float64),
                    upper=np.array([np.inf if v is None else v for v in box.upper], dtype=np.float64),
                )
            )
        return cls(options, dep.dofs_name, dep.dofs_pos_limits(), boxes, entities)

    def reset(self) -> None:
        """Forget the previous ticks: the last target sent, the last reading's time, the count past a limit."""
        self._last_sent: np.ndarray | None = None
        self._last_read_time: float | None = None
        self._was_moving = False
        self._excess_ticks = 0
        self._excess = np.zeros(len(self.dofs_name))
        self._raw: np.ndarray | None = None

    # -- before the send --------------------------------------------------------------------------

    def filter(self, target: np.ndarray, measured: np.ndarray | None) -> np.ndarray:
        """The policy's ``target`` as it may be sent: clamped into range, then capped against ``measured`` and the last send.

        Call it on the ticks the policy drives, and only then: the hold, the INIT ramp and the START tick send the
        state machine's own commands. Returns a new array; ``target`` is left as it was.
        """
        raw = np.array(target, dtype=np.float64)
        self._raw = raw
        self._excess = np.maximum(np.maximum(self.lower - raw, raw - self.upper), 0.0)
        past = bool((self._excess > self.options.limit_excess_tolerance).any())
        self._excess_ticks = self._excess_ticks + 1 if past else 0
        out = np.clip(raw, self.clamp_lower, self.clamp_upper)
        if self.step_from_measured is not None and measured is not None:
            measured = np.asarray(measured, dtype=np.float64)
            out = np.clip(out, measured - self.step_from_measured, measured + self.step_from_measured)
        if self.step_from_sent is not None and self._last_sent is not None:
            out = np.clip(out, self._last_sent - self.step_from_sent, self._last_sent + self.step_from_sent)
        return out

    def sent(self, dofs_pos: np.ndarray) -> None:
        """Note the joint targets that went out this tick, whoever sent them (hold, INIT ramp or policy)."""
        self._last_sent = np.array(dofs_pos, dtype=np.float64)

    # -- after the reading ------------------------------------------------------------------------

    def check(self, state: RobotState, *, moving: bool, driven: bool) -> list[str]:
        """Compare this tick's reading, already synced onto the scene, with the limits; return what was crossed.

        ``moving`` is whether the state machine moves the robot this tick (the INIT ramp or the policy), and only
        then is a crossed limit returned. ``driven`` is whether :meth:`filter` ran this tick. Writes this tick's
        values into ``state.extra`` under ``safety.*`` either way.
        """
        crossed: list[str] = []
        nan_dofs = np.full(len(self.dofs_name), np.nan)

        now = self.clock()
        interval = np.nan if self._last_read_time is None else now - self._last_read_time
        limit = self.options.max_read_interval_s
        if moving and self._was_moving and limit is not None and interval > limit:
            crossed.append(f"{interval:.3f} s between two readings (max_read_interval_s={limit})")
        self._last_read_time = now
        self._was_moving = moving

        tracking = nan_dofs
        if self._last_sent is not None:
            tracking = np.abs(np.asarray(state.dofs_pos, dtype=np.float64) - self._last_sent)
            if moving and self.max_tracking_error is not None:
                over = np.flatnonzero(tracking > self.max_tracking_error)
                if over.size:
                    worst = over[np.argmax(tracking[over] - self.max_tracking_error[over])]
                    crossed.append(
                        f"joint '{self.dofs_name[worst]}' is {tracking[worst]:.3f} from its target (max_dofs_tracking_error "
                        f"{self.max_tracking_error[worst]:g}; {over.size} joint(s) over)"
                    )

        if not driven:
            self._excess_ticks = 0
            self._raw = None
        steps = self.options.limit_excess_steps
        if moving and steps is not None and self._excess_ticks >= steps:
            worst = int(np.argmax(self._excess))
            crossed.append(
                f"the policy's targets were past a joint's range by more than limit_excess_tolerance="
                f"{self.options.limit_excess_tolerance:g} on {self._excess_ticks} tick(s) in a row (limit_excess_steps="
                f"{steps}); on the last, joint '{self.dofs_name[worst]}' was {self._excess[worst]:.3f} past "
                f"[{self.lower[worst]:g}, {self.upper[worst]:g}]"
            )

        for box in self.link_boxes:
            pos = self.entities[box.entity_name].get_links_pos(box.links_idx_local)
            pos = np.asarray(pos.detach().cpu().numpy(), dtype=np.float64).reshape(len(box.links_idx_local), 3)
            margin = np.minimum(pos - box.lower, box.upper - pos)  # metres inside the box per link and axis
            state.extra[f"safety.box.{box.name}"] = np.array([margin.min()])
            if moving and margin.min() < 0.0:
                link = int(np.argmin(margin.min(axis=1)))
                crossed.append(
                    f"link '{box.links_name[link]}' of '{box.entity_name}' at {_fmt(pos[link])} m is outside box "
                    f"'{box.name}' (lower {_fmt(box.lower)}, upper {_fmt(box.upper)})"
                )

        state.extra["safety.target_raw"] = nan_dofs if self._raw is None else self._raw.copy()
        state.extra["safety.tracking_error"] = tracking
        state.extra["safety.read_interval_s"] = np.array([interval])
        state.extra["safety.stopped"] = np.array([1.0 if crossed else 0.0])
        return crossed
