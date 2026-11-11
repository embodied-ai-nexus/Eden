"""Deploy a composite rig -- an arm with a hand welded on, and kin -- as one robot.

A composite (:class:`~eden.options.entities.CompositeEntityOptions`, e.g.
:class:`~eden.options.robots.FrankaXHand`) sits in the task's scene under one name and expands into
several entities when the env builds -- the Franka arm as ``robot``, the XHand as ``hand``. Each
part is its own piece of hardware with its own SDK, so :class:`CompositeDeployment` drives each
through the backend that part's robot is bound to (:func:`resolve_deployment_for`) and presents the
set as one robot to everything above it: the action manager runs once per tick and its targets are
split across the parts, the parts' readings are joined into one :class:`RobotState`, and one
observation is computed from all of them.

The parts' DOFs are laid end to end parent first -- a welded part always after the part it is welded
to, otherwise in the rig's declaration order -- so ``dofs_name`` -- the layout of every command, of
``--disable_joints`` and of ``--debug_record`` -- is the arm's joints followed by the hand's.

The rig runs the one operator state machine every backend runs (boot in STOP, INIT, START, STOP,
KILL), once, for all of its parts: one START hands the whole rig to the policy.

Each part's wire -- ``connect``, ``read_state``, ``send_payload``, ``_idle_tick``, ``close`` --
runs on that part's own thread, every part at once, so a tick's I/O takes as long as its slowest
part rather than the sum of them all. The policy tick stays one: the loop waits for every part's
reading before it observes, and for every part's send before it moves on.

A part runs as it does deployed alone: its backend's own option defaults, its own wire, its own
hooks. The composite adds the joining and changes nothing on a part. The one thing it cannot keep
per part is a backend's loop setting that acts on the env every part shares
(``sync_before_commands``, ``advances_episode_clock``): the rig follows one only when every part
does, and otherwise runs the standard loop and says which parts it does not follow.
"""

from __future__ import annotations
from collections.abc import Callable
from concurrent.futures import Future, wait
from typing import TYPE_CHECKING, Any, ClassVar, get_args, get_origin
import ast
import inspect
import queue
import sys
import threading

import numpy as np

import eden as en
import eden.extensions.deployment.base as deployment_base
from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    DeploymentBase,
    resolve_deployment_for,
    scene_entity_options,
)
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.options.entities import CompositeEntityOptions, RobotOptions
from eden.utils.params import render_annotation

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.options.extensions.deployment import DeploymentOptions
    from eden.utils.configs import EdenRLConfig


#: Base options every part runs with the composite's value of, unless ``parts_options`` sets its own.
#: They act inside the part's own transport (its connect, its ``send_payload`` low-pass) or on the
#: part's own joints: ``deploy_kp_kd_scale`` scales that part's slice of every command, and each
#: backend also gives it its own meaning -- a Wuji Hand 2 scales the MIT gains it programs at
#: connect, an XHand (which cannot be softened that way) refuses anything but 1.0. So a body can be
#: commissioned at 0.2 next to a hand kept at 1.0: ``--part_option hand.deploy_kp_kd_scale=1.0``.
FORWARDED_OPTIONS: tuple[str, ...] = ("connect_timeout_s", "target_lpf", "deploy_kp_kd_scale")

#: Base options every part runs with the composite's value of, always. Every other base option is the
#: composite's alone (it runs the loop and the state machine, owns the recording), and
#: ``parts_options`` refuses it too.
_ALWAYS_FORWARDED: tuple[str, ...] = ("ignore_action_terms",)

#: Options only the part the rig's base reading comes from takes: the velocity estimator runs on its feet.
_BASE_PART_OPTIONS: tuple[str, ...] = ("estimate_lin_vel", "feet_links", "lin_vel_estimator_options")


#: The :class:`DeploymentBase` methods a part's backend may override: the ones its own construction
#: runs, and the hooks the composite calls on each part. Every other one belongs to the loop the
#: composite runs once for the whole rig (``step``, ``reset``, the hold and INIT payloads, the
#: observation), so a part's override of it would be skipped silently -- and is refused instead.
PART_HOOKS: frozenset[str] = frozenset(
    {
        # Its own construction.
        "_resolve_robot_options",
        "_resolve_control_dt",
        "_resolve_dofs",
        "_resolve_disabled_joints",
        "_resolve_options",
        "_resolve_feet_links",
        "_validate_init_target",
        "_validate_on_motion_end",
        "_build_term_action_maps",
        "_swap_in_state_entity",
        "_make_lin_vel_estimator",
        "_undo_swaps",
        # The wire, the per-tick hooks and the queries the composite fans out to its parts.
        "connect",
        "close",
        "read_state",
        "send_payload",
        "_before_step",
        "_idle_tick",
        "_prepare_reset",
        "_sync_state",
        "_update_lin_vel_estimate",
        "_remote_inputs",
        "_print_control_guide",
        "_motion_start_dofs",
        "programmed_gains",
        "command_gain_dofs",
        "blocks_for_the_period",
        "provides_base_lin_vel",
        "_publishes_base_lin_vel",
    }
)


def _skipped_overrides(cls: type[DeploymentBase]) -> list[str]:
    """The :class:`DeploymentBase` methods ``cls`` overrides that a composite never calls on a part."""
    skipped = []
    for name, member in vars(DeploymentBase).items():
        if name.startswith("__") or name in PART_HOOKS:
            continue
        if not (callable(member) or isinstance(member, (property, classmethod, staticmethod))):
            continue
        if inspect.getattr_static(cls, name) is not member:
            skipped.append(name)
    return sorted(skipped)


def _deploy_alone_hint(rig: CompositeEntityOptions, name: str) -> str:
    """How to deploy part ``name`` on its own -- possible from ``eden deploy`` for every part but the root.

    The root sits under the rig's own scene key, which ``--entity_name`` reads as the whole rig.
    """
    if name == rig.primary_name:
        return (
            f"'{name}' is the rig's root, whose name is the rig's own scene key, so `eden deploy --entity_name {name}` "
            f"means the whole rig: deploy it from a task whose scene places the {name} without the rest."
        )
    return f"Deploy '{name}' alone with `--entity_name {name}`; the rest of the rig cannot be deployed without it."


def _declared_type(cls: type, option: str) -> Any:
    """The type backend ``cls`` declares ``option`` as, resolved; ``Any`` where it declares none.

    The backends' modules use ``from __future__ import annotations``, so an annotation is the text
    written; it is evaluated in the module of the class that wrote it, as ``typing.get_type_hints``
    would, but for the one option only.
    """
    for owner in cls.__mro__:
        annotations = owner.__dict__.get("__annotations__") or {}
        if option not in annotations:
            continue
        declared = annotations[option]
        if isinstance(declared, str):
            declared = eval(declared, vars(sys.modules[owner.__module__]))  # noqa: S307 -- source text of a class
        return get_args(declared)[0] if get_origin(declared) is ClassVar else declared
    return Any


def _typed_part_option(cls: type[DeploymentBase], part: str, option: str, value: Any) -> Any:
    """``value`` as backend ``cls``'s declared type for ``option``; raises ``ValueError`` if it is not one.

    ``--part_option`` hands every value over as typed, and only the backend knows its type: a ``str``
    option keeps ``"12345678"`` as a serial number, an ``int`` one reads ``"2"`` as ``2``, a ``bool``
    reads ``false`` as ``False``, a tuple reads ``('a', 'b')``. A value that is none of the declared
    type is refused here, by name, rather than handed to a backend that would read the text
    ``"false"`` as true.
    """
    from pydantic import ConfigDict, TypeAdapter, ValidationError

    declared = _declared_type(cls, option)
    adapter = TypeAdapter(declared, config=ConfigDict(arbitrary_types_allowed=True))
    literal: Any = None
    has_literal = False
    if isinstance(value, str):
        try:
            literal, has_literal = ast.literal_eval(value), True
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass  # not a Python literal: the text itself is the only candidate
    # The literal exactly as declared (2 for an int, None for an Optional), else the value read
    # leniently (a str option's text, "false" for a bool), else the literal read leniently.
    attempts = (
        ([(literal, True)] if has_literal else []) + [(value, False)] + ([(literal, False)] if has_literal else [])
    )
    error: ValidationError | None = None
    for candidate, strict in attempts:
        try:
            return adapter.validate_python(candidate, strict=strict)
        except ValidationError as exc:
            error = exc
    expected = render_annotation(declared) or "Any"
    reason = error.errors()[0]["msg"] if error is not None and error.errors() else "invalid"
    raise ValueError(
        f"parts_options['{part}']['{option}'] = {value!r} is not a {expected}, the type {cls.__name__}.{option} "
        f"declares ({reason}). On the command line: --part_option {part}.{option}=<{expected}> (a bool takes "
        f"true / false)."
    )


def _parent_first(rig: CompositeEntityOptions) -> list[str]:
    """``rig``'s entity names with every welded part after the part it is welded to, else in declaration order.

    The order its parts are connected, read, synced and laid out in: a welded part's base pose is
    FK through its parent's joints, which must already hold this tick's reading.
    """
    parents = {child: spec[0] for child, spec in rig.attachments().items()}
    ordered: list[str] = []
    pending = list(rig.entity_names)
    while pending:
        ready = [name for name in pending if parents.get(name) not in pending]
        if not ready:
            raise ValueError(
                f"{type(rig).__name__}'s attachments() weld {pending} to one another in a cycle: "
                f"{ {name: parents[name] for name in pending} }. Each part must hang off one parent, the root off none."
            )
        ordered += ready
        pending = [name for name in pending if name not in ready]
    return ordered


def _slice(value: np.ndarray | None, part: slice) -> np.ndarray | None:
    return None if value is None else np.array(value[part], dtype=np.float64)


class _PartThread:
    """One part's own thread: every call on that part's wire runs there, one at a time, in the order submitted.

    A daemon, so a part whose SDK call hangs cannot hold the process open on exit. Always the same
    thread for the same part, so an SDK that cares which thread drives it sees one.
    """

    def __init__(self, name: str) -> None:
        self._calls: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name=f"[deploy] part {name}", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        future: Future = Future()
        self._calls.put((future, fn, args))
        return future

    def stop(self) -> None:
        """Finish the calls already submitted, then end the thread."""
        self._calls.put(None)

    def _run(self) -> None:
        while (call := self._calls.get()) is not None:
            future, fn, args = call
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # handed to the waiting loop, which raises it
                future.set_exception(exc)


@DEPLOYMENT_REGISTRY.register()
class CompositeDeployment(DeploymentBase):
    """Drive every part of a composite rig through its own backend, as one robot.

    ``eden deploy`` picks this backend for any composite (:func:`resolve_deployment_for`), so a task
    whose scene holds ``robot=FrankaXHand(...)`` deploys onto a Franka and an XHand with no extra
    flags, assuming both are connected. Each part with joints gets the backend its robot is bound to
    (``FrankaPandaDeployment``, ``RoboTeraXHandDeployment``), constructed against its own scene
    entity; a part with none (a mount) is not deployed.

    What the composite owns, and the parts do not:

    - **The loop and the state machine.** One action-manager pass and one observation per tick,
      and one INIT / START / STOP / KILL state machine for the whole rig: it boots in STOP holding
      every part's measured pose, INIT ramps every joint, and START hands the whole rig to the
      policy. The operator's inputs are every part's remote (the Oli's joystick, the G1's) plus
      one keyboard. :meth:`connect`, :meth:`read_state`, :meth:`send_payload`, ``_idle_tick``
      and :meth:`close` run on every part at once, each on the part's own thread, and the loop
      waits for all of them: a tick's I/O takes as long as its slowest part. The scene sync and
      the other per-tick hooks (``_before_step`` -- a Wuji hand's safety check --
      ``_prepare_reset``) run on the loop's thread, part by part, every welded part after its
      parent (whatever order ``entity_parts`` lists them in), so a welded part's FK reads its
      parent's fresh pose.
    - **The pace.** The parts are built with ``sync=False`` and the composite's rate limiter paces
      the loop -- unless a part's send itself waits out the control period (a Wuji hand's default
      ``stream_mode="sync"``, :attr:`~DeploymentBase.blocks_for_the_period`). Then that part paces
      the rig, as it paces itself deployed alone, and the rig runs no rate limiter on top; the
      parts' sends overlap, so two such hands still take one period, not two.
    - **The loop settings.** A backend's ``sync_before_commands`` / ``advances_episode_clock`` act
      on the env every part shares, so they cannot be kept per part: the rig follows one only when
      every part sets it, and otherwise runs the standard loop -- the order and clock a part
      without it runs deployed alone -- and warns which parts it does not follow. No part takes on
      another part's loop.
    - **The gains.** Commands carry the concatenation of the parts' robot gains, scaled by
      ``deploy_kp_kd_scale`` and zeroed on ``disable_joints``. A part whose robot ignores each
      command's gains (the Franka's controller, the XHand's and Wuji's firmware) cannot go limp that
      way, so a disabled joint there is refused. Each part takes the rig's scale unless its
      ``parts_options`` sets its own, and its backend also applies it its own way (a Wuji Hand 2
      programs its MIT gains from it) or refuses it (an XHand, anything but 1.0).
    - **The base reading.** Base orientation, angular velocity and acceleration come from the rig's
      root part (its primary entity), as does the velocity estimator (``estimate_lin_vel``), on the
      rig's feet (a :class:`~eden.options.robots.composite.base.HumanoidOptions`' ``feet_links``). A
      welded part's own base pose is read off FK through the weld, not off its (IMU-less) reading.
    - **Commands, recording, the motion clock** -- everything else :class:`DeploymentBase` does.

    Every other option of a part's backend keeps its own default, as when the part is deployed
    alone. Each part also runs the composite's :data:`FORWARDED_OPTIONS` (no backend changes their
    defaults); ``parts_options`` overrides those, and sets each backend's own options, per part::

        CompositeDeployment.configure(parts_options={
            "robot": {"robot_ip": "172.16.0.2"},
            "hand": {"protocol": "EtherCAT"},
        })

    (``eden deploy --part_option robot.robot_ip=172.16.0.2 --part_option hand.protocol=EtherCAT``.)

    Not supported yet, and refused at construction: a part whose backend overrides a method the
    composite never calls on a part (``step``, ``reset``, the hold payload -- anything outside
    :data:`PART_HOOKS`), two parts with a joint of the same name, and ``--mujoco`` sim-to-sim of a
    composite.

    Parameters
    ----------
    parts_options : dict[str, dict]
        Backend options per part, keyed by the part's scene entity name. A key may be any option of
        that part's backend except the composite-owned base options.
    """

    parts_options: dict = {}

    def __init__(self, env: RLEnvBase, options: DeploymentOptions) -> None:
        #: The deployed parts, by scene entity name, in rig order. Filled by :meth:`_resolve_dofs`.
        self.parts: dict[str, DeploymentBase] = {}
        self._part_slices: dict[str, slice] = {}
        #: Each part's own thread for its wire calls, started on first use (:meth:`_on_part_threads`).
        self._part_threads: dict[str, _PartThread] = {}
        try:
            super().__init__(env, options)
        except BaseException:
            # The parts swap their own entities as they are built, before the composite's own
            # rollback is armed; put back every one that got that far.
            self._undo_part_swaps()
            raise
        if paced := [name for name, part in self.parts.items() if part.blocks_for_the_period]:
            # Each such part waits out the period in its own send, on its own thread, as it does
            # deployed alone (where it runs no rate limiter either); a limiter on top would add
            # whatever the tick spent outside the send to every period.
            self._rate = None
            en.logger.info(
                f"[deploy] Part(s) {paced} wait out the control period in their own send and pace the rig, as they "
                "pace themselves deployed alone; the rig runs no rate limiter of its own."
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_robot_options(cls, cfg: EdenRLConfig, entity_name: str) -> CompositeEntityOptions:
        # The scene entry itself, not the env entity: under a composite's key the env holds its primary part.
        rig = getattr(cfg.scene_options, entity_name, None)
        if not isinstance(rig, CompositeEntityOptions):
            what = type(scene_entity_options(cfg.scene_options, entity_name)).__name__
            raise TypeError(
                f"CompositeDeployment drives a composite rig placed in the scene under entity_name, and "
                f"'{entity_name}' is a single {what}. Deploy it with the backend its robot is bound to "
                f"(resolve_deployment_for), which `eden deploy` picks on its own."
            )
        return rig

    def _resolve_dofs(self) -> None:
        """Build one backend per jointed part, and lay their DOFs, stances and gains end to end."""
        rig: CompositeEntityOptions = self.robot_options
        robots = self._jointed_parts(rig)
        if not robots:
            raise ValueError(f"{type(rig).__name__} has no part with joints, so there is nothing to deploy.")
        if skipped := [name for name in rig.entity_names if name not in robots]:
            en.logger.info(f"[deploy] {type(rig).__name__}: part(s) {skipped} have no joints and are not deployed.")
        self._check_parts_options(rig, robots)
        self._check_unique_dofs(rig, robots)
        backends = {name: self._part_backend(rig, name, options) for name, options in robots.items()}

        primary = rig.primary_name
        self._base_name = primary if primary in robots else next(iter(robots))
        offset = 0
        for name, cls in backends.items():
            try:
                part = cls(self._env, cls.configure(**self._part_options(name, cls)))
            except (ValueError, TypeError) as exc:
                # The backend's own message names its standalone fix; in a rig that fix is a part option.
                exc.add_note(
                    f"[deploy] Raised building part '{name}' of {type(rig).__name__} ({cls.__name__}). In a composite "
                    f"rig a part's backend options are set per part: --part_option {name}.<option>=<value> "
                    f"(parts_options={{'{name}': {{...}}}}); e.g. --part_option {name}.deploy_kp_kd_scale=1.0 keeps "
                    f"the rig's gain scale off it."
                )
                raise
            self.parts[name] = part
            self._part_slices[name] = slice(offset, offset + part.num_dofs)
            offset += part.num_dofs
            en.logger.info(f"[deploy] {type(rig).__name__} part '{name}': {cls.__name__} ({part.num_dofs} DOFs).")

        parts = list(self.parts.values())
        self.dofs_name = [dof for part in parts for dof in part.dofs_name]
        self.default_dof_pos = np.concatenate([part.default_dof_pos for part in parts])
        self._robot_dofs_kp = np.concatenate([part._robot_dofs_kp for part in parts])
        self._robot_dofs_kd = np.concatenate([part._robot_dofs_kd for part in parts])
        # A part whose robot ignores command gains still takes them from a slot switch for the rest.
        self.sends_command_gains = any(part.sends_command_gains for part in parts)
        self._resolve_loop_settings(rig)

    #: The loop settings a backend may turn on, and what the standard loop does instead.
    _LOOP_SETTINGS: ClassVar[dict[str, str]] = {
        "sync_before_commands": "the commands publish before this tick's reading is synced",
        "advances_episode_clock": "the env's episode clock does not advance",
    }

    def _resolve_loop_settings(self, rig: CompositeEntityOptions) -> None:
        """Follow each loop setting only when every part's backend sets it; else run the standard loop, and warn.

        They act on the env the parts share (when the commands publish, the episode clock), so one
        loop cannot run a part with a setting and a part without it each as it runs deployed alone.
        Following the setting would hand it to the parts that never run with it -- the Oli body of
        an ``OliWuji`` reading in the Wuji hands' order, its observations on the hands' clock -- so
        the rig keeps the standard loop, and the parts that set it are the ones told.
        """
        for setting, standard in self._LOOP_SETTINGS.items():
            on = {name: type(part).__name__ for name, part in self.parts.items() if getattr(part, setting)}
            setattr(self, setting, len(on) == len(self.parts))
            if on and len(on) < len(self.parts):
                off = {name: type(part).__name__ for name, part in self.parts.items() if name not in on}
                en.logger.warning(
                    f"[deploy] {type(rig).__name__}: {setting} is set by part(s) {on} and not by {off}. It acts on "
                    f"the env every part shares, so one loop cannot keep both as they run deployed alone: the rig "
                    f"runs the standard loop, as {list(off)} do alone, and for {list(on)} {standard}. Every part's "
                    f"wire is unchanged."
                )

    @staticmethod
    def _jointed_parts(rig: CompositeEntityOptions) -> dict[str, RobotOptions]:
        """The parts of ``rig`` with joints -- the ones deployed -- by name, parent first (:func:`_parent_first`)."""
        options = rig.entity_options()
        return {
            name: options[name]
            for name in _parent_first(rig)
            if isinstance(options[name], RobotOptions) and options[name].dofs_name
        }

    @classmethod
    def part_backends(cls, rig: CompositeEntityOptions) -> dict[str, type[DeploymentBase]]:
        """The backend each jointed part of ``rig`` deploys through, parent first.

        Raises as construction would for a part that has none or cannot run inside a rig. ``eden
        deploy`` asks before constructing, to hand a backend-specific flag (``--publish_rate``) to
        the parts that declare it.
        """
        return {name: cls._part_backend(rig, name, options) for name, options in cls._jointed_parts(rig).items()}

    def _check_parts_options(self, rig: CompositeEntityOptions, robots: dict[str, RobotOptions]) -> None:
        if unknown := sorted(set(self.parts_options) - set(robots)):
            raise ValueError(
                f"parts_options names {unknown}, which are not deployed parts of {type(rig).__name__}. "
                f"Parts: {list(robots)}."
            )
        owned = set(DeploymentBase.get_parameter_names()) - set(FORWARDED_OPTIONS)
        for name, options in self.parts_options.items():
            if clash := sorted(set(options) & owned):
                raise ValueError(
                    f"parts_options['{name}'] sets {clash}, which the composite sets for the whole rig: set it on "
                    f"CompositeDeployment itself (the `eden deploy` flag of the same name). Per part, only a "
                    f"backend's own options and {list(FORWARDED_OPTIONS)} can be set."
                )

    @staticmethod
    def _check_unique_dofs(rig: CompositeEntityOptions, robots: dict[str, RobotOptions]) -> None:
        owners: dict[str, list[str]] = {}
        for name, options in robots.items():
            for dof in options.dofs_name:
                owners.setdefault(dof, []).append(name)
        if shared := {dof: names for dof, names in owners.items() if len(names) > 1}:
            raise NotImplementedError(
                f"{type(rig).__name__}'s parts share joint names {dict(list(shared.items())[:5])}, and a composite "
                f"deployment addresses its DOFs by name (--disable_joints, a policy's write_dofs_targets, the debug "
                f"record). Deploying two parts with the same joint names (two identical arms) is not supported yet; "
                f"deploy one part with `--entity_name <part>`."
            )

    @staticmethod
    def _part_backend(rig: CompositeEntityOptions, name: str, options: RobotOptions) -> type[DeploymentBase]:
        """The backend ``name``'s robot is bound to, refusing one that cannot run inside a composite."""
        alone = _deploy_alone_hint(rig, name)
        try:
            cls = resolve_deployment_for(options)
        except KeyError:
            bound = sorted({robot.__name__ for robot in deployment_base._ROBOT_DEPLOYMENT_BINDINGS})
            raise ValueError(
                f"Part '{name}' of {type(rig).__name__} ({type(options).__name__}) has no deployment backend bound to "
                f"it, so the rig cannot be deployed whole. Robots with a backend: {bound}. Add one with "
                f"bind_deployment({type(options).__name__}, '<deployment_name>') in its backend's module. {alone}"
            ) from None
        if overridden := _skipped_overrides(cls):
            raise NotImplementedError(
                f"Part '{name}' of {type(rig).__name__} runs on {cls.__name__}, which replaces DeploymentBase."
                f"{'/'.join(f'{method}()' for method in overridden)}, and a composite runs one loop for every part "
                f"rather than each part's own, so that logic would be skipped. Move it into the hooks the composite "
                f"calls on each part (_before_step, _idle_tick, _prepare_reset, _sync_state, read_state, "
                f"send_payload; all of them: composite.PART_HOOKS). {alone}"
            )
        return cls

    def _part_options(self, name: str, cls: type[DeploymentBase]) -> dict[str, Any]:
        """The options ``name``'s backend ``cls`` is built with: forwarded, then ``parts_options``; the rest its own defaults."""
        options: dict[str, Any] = {option: getattr(self, option) for option in FORWARDED_OPTIONS}
        options.update(self._typed_part_options(cls, name, self.parts_options.get(name, {})))
        options.update({option: getattr(self, option) for option in _ALWAYS_FORWARDED})
        # At the composite's control rate, paced by the composite (or by a part that waits out the
        # period in its own send, as it does alone), and recorded by it.
        options.update(entity_name=name, control_dt=self.control_dt, sync=False)
        if name == self._base_name:
            options.update({option: getattr(self, option) for option in _BASE_PART_OPTIONS})
            feet = self._declared_feet()
            if not options["feet_links"] and feet:
                options["feet_links"] = feet
            elif not options["feet_links"] and feet == () and self.estimate_lin_vel:
                raise ValueError(
                    f"estimate_lin_vel=True needs foot links, and {type(self.robot_options).__name__} declares it has "
                    f"none (feet_links_name=()): what it stands on is not its "
                    f"{type(self.robot_options.entity_options()[name]).__name__}'s own feet, so the estimator must "
                    f"not use them. Name the links for this run: `eden deploy ... --estimate_lin_vel --feet_links "
                    f"<left> <right>`, or drop --estimate_lin_vel."
                )
        return options

    def _declared_feet(self) -> tuple[str, ...] | None:
        """The rig's own feet override (``HumanoidOptions.feet_links_name``): ``None`` inherits, ``()`` is none."""
        declared = getattr(self.robot_options, "feet_links_name", None)
        return None if declared is None else tuple(declared)

    @staticmethod
    def _typed_part_options(cls: type[DeploymentBase], part: str, options: dict[str, Any]) -> dict[str, Any]:
        """``options`` with each value checked against, and read as, its declared type (:func:`_typed_part_option`).

        An option the backend does not declare is left alone, for its ``configure`` to refuse by name.
        """
        declared = set(cls.get_parameter_names())
        return {
            option: _typed_part_option(cls, part, option, value) if option in declared else value
            for option, value in options.items()
        }

    def _resolve_feet_links(self) -> None:
        # The base part resolved (and checked) its own feet; the estimator runs there. A rig that
        # declares no feet has none, whatever its root robot's own options say.
        override = tuple(self.feet_links)
        self.feet_links = () if not override and self._declared_feet() == () else self._base.feet_links

    def _resolve_options(self) -> None:
        super()._resolve_options()
        self._check_gains_reach_the_parts()

    def _gain_scale(self) -> np.ndarray:
        """Each part's own ``deploy_kp_kd_scale`` on its slice of the rig's DOFs."""
        return np.concatenate([np.full(part.num_dofs, part.deploy_kp_kd_scale) for part in self.parts.values()])

    def _check_gains_reach_the_parts(self) -> None:
        """Refuse what acts through each command's kp/kd on a part whose robot ignores them.

        A disabled joint there would keep servoing rather than go limp, and the INIT ramp's gain
        scale would leave that part ramping at full gains -- each refused as the part's own backend
        refuses it standalone (a Wuji hand does both).
        """
        fixed = {name: part for name, part in self.parts.items() if not part.sends_command_gains}
        for name, part in fixed.items():
            if limp := [dof for dof in self.disabled_dofs if dof in part.dofs_name]:
                raise ValueError(
                    f"disable_joints matched {limp} on part '{name}', but {type(part).__name__} runs its own gains "
                    f"and ignores the kp/kd in each command, so those joints would keep servoing to the policy's "
                    f"targets rather than go limp. Remove them from disable_joints."
                )
        if fixed and (self.init_dofs_kp_scale != 1.0 or self.init_dofs_kd_scale != 1.0):
            backends = {name: type(part).__name__ for name, part in fixed.items()}
            raise ValueError(
                f"init_dofs_kp_scale={self.init_dofs_kp_scale:g} / init_dofs_kd_scale={self.init_dofs_kd_scale:g} "
                f"soften the INIT ramp through each command's kp/kd, which part(s) {backends} ignore: they would ramp "
                f"at full gains while the rest ramp softly. Leave both at 1.0 and soften the ramp with --init_steps."
            )

    def _swap_in_state_entity(self) -> None:
        """Nothing to swap here: each part swapped its own entity as it was built."""
        self._state_entity = self._base._state_entity
        return None

    def _make_lin_vel_estimator(self, sim_entity: Any) -> None:
        return None  # the base part runs it, on its own entity

    def _undo_swaps(self, sim_entity: Any = None) -> None:
        restored = self._restore_swapped_commands()
        self._undo_part_swaps()
        if restored:
            en.logger.warning(f"[deploy] CompositeDeployment construction failed; restored command term(s) {restored}.")

    def _undo_part_swaps(self) -> None:
        """Put back every part's sim entity (and the term handles it repointed), last part first. Idempotent."""
        for part in reversed(list(self.parts.values())):
            part._undo_swaps(part._state_entity._sim_entity)
        self.parts = {}

    def _build_term_action_maps(self):
        """Each part's term maps, shifted onto the part's slice of the rig's DOFs.

        A part maps only the terms that drive its own entity (``build_term_action_maps``), so between
        them the parts place every term, each exactly once.
        """
        return [
            (term, src_idx, dst_idx + self._part_slices[name].start)
            for name, part in self.parts.items()
            for term, src_idx, dst_idx in part._term_action_maps
        ]

    @property
    def _base(self) -> DeploymentBase:
        """The part the rig's base reading comes from: its primary entity, or the first part with joints."""
        return self.parts[self._base_name]

    # ------------------------------------------------------------------
    # The wire, fanned out to the parts, each on its own thread
    # ------------------------------------------------------------------

    def _on_part_threads(
        self, call: Callable[[str, DeploymentBase], Any], parts: dict[str, DeploymentBase] | None = None
    ) -> tuple[dict[str, Any], dict[str, BaseException]]:
        """Run ``call(name, part)`` for every part at once, each on its part's own thread, and wait for all.

        Returns the results and the failures, each by part in rig order. Every call has finished
        when this returns, a failed one's neighbours included, so nothing is left on a wire. Only
        the wire runs there: the scene sync and the other per-tick hooks stay on the loop's thread
        (Genesis is not driven from two threads), and the loop waits here, so a part's SDK is never
        driven from both at once.
        """
        parts = self.parts if parts is None else parts
        futures = {}
        for name, part in parts.items():
            if name not in self._part_threads:
                self._part_threads[name] = _PartThread(name)
            futures[name] = self._part_threads[name].submit(call, name, part)
        wait(futures.values())
        results: dict[str, Any] = {}
        errors: dict[str, BaseException] = {}
        for name, future in futures.items():
            if (exc := future.exception()) is not None:
                errors[name] = exc
            else:
                results[name] = future.result()
        return results, errors

    def _raise_part_error(self, errors: dict[str, BaseException], doing: str) -> None:
        """Raise the first part's failure (in rig order), naming the part; log every other one."""
        if not errors:
            return
        (name, first), *others = errors.items()
        for other, exc in others:
            en.logger.error(
                f"[deploy] Part '{other}' ({type(self.parts[other]).__name__}) also failed {doing}: {exc!r}"
            )
        first.add_note(
            f"[deploy] Raised by part '{name}' ({type(self.parts[name]).__name__}) of "
            f"{type(self.robot_options).__name__}, {doing}, on its own thread."
        )
        raise first

    def _stop_part_threads(self) -> None:
        for thread in self._part_threads.values():
            thread.stop()
        self._part_threads = {}

    def connect(self) -> None:
        """Connect every part at once. If one fails, the ones that came up are closed again, then it raises."""
        _, errors = self._on_part_threads(lambda name, part: part.connect())
        if errors:
            up = {name: part for name, part in self.parts.items() if name not in errors}
            _, unclosed = self._on_part_threads(lambda name, part: part.close(), up)
            for name, exc in unclosed.items():
                en.logger.error(
                    f"[deploy] Could not close part '{name}' ({type(self.parts[name]).__name__}) after a failed "
                    f"connect: {exc!r}"
                )
            self._raise_part_error(errors, "connecting")

    def close(self) -> None:
        """Close every part at once, each even if another raises; then raise the first failure."""
        errors: list[BaseException] = []
        try:
            self._teardown_common()
        except Exception as exc:
            errors.append(exc)
        _, failed = self._on_part_threads(lambda name, part: part.close())
        for name, exc in failed.items():
            en.logger.error(f"[deploy] Closing part '{name}' ({type(self.parts[name]).__name__}) failed: {exc!r}")
            errors.append(exc)
        self._stop_part_threads()
        if errors:
            raise errors[0]

    def read_state(self) -> RobotState:
        """Read every part at once and join the readings, with the base fields from the base part.

        Each part keeps its own reading as its ``state`` too, which its backend reads back (the
        Franka clamps its next command against it). A part's ``extra`` entries are prefixed with
        the part's name, so ``--debug_record`` keeps them apart.
        """
        states, errors = self._on_part_threads(lambda name, part: part.read_state())
        self._raise_part_error(errors, "reading its state")
        for name, part in self.parts.items():
            part.state = states[name]
        base = states[self._base_name]
        return RobotState(
            stamp=base.stamp,
            base_quat=base.base_quat,
            base_ang_vel=base.base_ang_vel,
            base_lin_acc=base.base_lin_acc,
            dofs_pos=np.concatenate([state.dofs_pos for state in states.values()]),
            dofs_vel=np.concatenate([state.dofs_vel for state in states.values()]),
            dofs_torque=np.concatenate([state.dofs_torque for state in states.values()]),
            extra={f"{name}.{key}": value for name, state in states.items() for key, value in state.extra.items()},
        )

    def send_payload(self, payload: RobotCommand) -> None:
        """Send every part its slice of ``payload`` at once; ``payload.dofs_pos`` then holds what was sent (filtered).

        Each part sends it as the rig does: a hold of the rig is a hold of every part
        (:attr:`~DeploymentBase.sending_hold`), so a Franka part keeps its gripper where the policy left it.
        """
        commands = {
            name: RobotCommand(
                dofs_pos=_slice(payload.dofs_pos, part_slice),
                dofs_vel=_slice(payload.dofs_vel, part_slice),
                dofs_torque=_slice(payload.dofs_torque, part_slice),
                dofs_kp=_slice(payload.dofs_kp, part_slice),
                dofs_kd=_slice(payload.dofs_kd, part_slice),
            )
            for name, part_slice in self._part_slices.items()
        }
        for part in self.parts.values():
            part._sending_hold = self._sending_hold
        try:
            _, errors = self._on_part_threads(lambda name, part: part.send_payload(commands[name]))
        finally:
            for part in self.parts.values():
                part._sending_hold = False
        self._raise_part_error(errors, "sending its command")
        sent = np.array(payload.dofs_pos, dtype=np.float64)
        for name, command in commands.items():
            sent[self._part_slices[name]] = command.dofs_pos
        payload.dofs_pos = sent

    # ------------------------------------------------------------------
    # The loop's hooks, fanned out to the parts
    # ------------------------------------------------------------------

    def _before_step(self) -> None:
        for part in self.parts.values():
            part._before_step()

    def _idle_tick(self) -> None:
        """Every part's idle tick at once, on its own thread: a hand that holds for the period (a Wuji's) is wire."""
        _, errors = self._on_part_threads(lambda name, part: part._idle_tick())
        self._raise_part_error(errors, "holding its last target")

    def _prepare_reset(self) -> None:
        for part in self.parts.values():
            part._lpf_prev = None  # each part low-passes its own slice
            part._prepare_reset()

    def _remote_inputs(self) -> list[Any]:
        """Every part's remote, in rig order: any of them drives the whole rig's state machine."""
        return [remote for part in self.parts.values() for remote in part._remote_inputs()]

    def _print_control_guide(self) -> None:
        """Each part's own cheat-sheet where it has a remote with chords of its own, else the keyboard's."""
        guided = [part for part in self.parts.values() if part._remote_inputs()]
        for part in guided:
            part._print_control_guide()
        if not guided:
            super()._print_control_guide()
        en.logger.info(
            f"[deploy] One state machine drives every part of {type(self.robot_options).__name__}: "
            f"{ {name: type(part).__name__ for name, part in self.parts.items()} }."
        )

    def _sync_state(self, state: RobotState) -> None:
        """Sync each part's slice of ``state`` onto its entity, in rig order (a welded part after its parent)."""
        for name, part in self.parts.items():
            part_state = self._part_state(name, state)
            part._sync_state(part_state)
            state.extra.update({f"{name}.{key}": value for key, value in part_state.extra.items()})

    def _observe(self, state: RobotState, *, post_physics_refresh: bool) -> dict[str, Any]:
        """Run the base part's velocity estimator on its slice of ``state``, then observe the rig."""
        base = self._base
        if base._lin_vel_estimator is not None:
            part_state = self._part_state(self._base_name, state)
            base._update_lin_vel_estimate(part_state)
            # Its estimate and contacts join the base part's extras; carry them into the record.
            state.extra.update({f"{self._base_name}.{key}": value for key, value in part_state.extra.items()})
        return super()._observe(state, post_physics_refresh=post_physics_refresh)

    def _part_state(self, name: str, state: RobotState) -> RobotState:
        """``name``'s slice of the rig's ``state``.

        The joints come from ``state``. The base part takes the base fields from ``state`` too; any
        other part keeps those of its own last reading (identity before it has one), which for a
        welded part FK replaces anyway. ``extra`` is the part's own, unprefixed -- a ``base_pos``
        from mocap is read from there -- and is shared, so what syncing adds to it reaches the
        record.
        """
        part_slice = self._part_slices[name]
        own = self.parts[name].state
        if name == self._base_name:
            base = (state.base_quat, state.base_ang_vel, state.base_lin_acc)
        elif own is not None:
            base = (own.base_quat, own.base_ang_vel, own.base_lin_acc)
        else:
            base = (np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))
        return RobotState(
            stamp=state.stamp if own is None else own.stamp,
            base_quat=base[0],
            base_ang_vel=base[1],
            base_lin_acc=base[2],
            dofs_pos=state.dofs_pos[part_slice],
            dofs_vel=state.dofs_vel[part_slice],
            dofs_torque=state.dofs_torque[part_slice],
            extra={} if own is None else own.extra,
        )

    def _motion_start_dofs(self) -> np.ndarray | None:
        """Frame 0 of each part's tracked reference, its default stance where it tracks none; ``None`` if none does."""
        starts = [part._motion_start_dofs() for part in self.parts.values()]
        if all(start is None for start in starts):
            return None
        return np.concatenate(
            [part.default_dof_pos if start is None else start for part, start in zip(self.parts.values(), starts)]
        )

    @property
    def provides_base_lin_vel(self) -> bool:
        return self._base.provides_base_lin_vel

    def command_gain_dofs(self) -> np.ndarray:
        """Each part's own answer on its slice: a body that takes each command's gains next to hands that do not."""
        return np.concatenate([part.command_gain_dofs() for part in self.parts.values()])

    def programmed_gains(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The programmed gains of every part that runs its own, end to end; ``None`` if one has none in our units.

        A part that takes each command's gains runs no fixed set, so its slice holds what it is
        commanded now; :meth:`command_gain_dofs` tells the two apart.
        """
        kp, kd = [], []
        for name, part in self.parts.items():
            part_slice = self._part_slices[name]
            gains = (
                (self.default_dof_kp[part_slice], self.default_dof_kd[part_slice])
                if part.sends_command_gains
                else part.programmed_gains()
            )
            if gains is None:
                return None
            kp.append(gains[0])
            kd.append(gains[1])
        return np.concatenate(kp), np.concatenate(kd)
