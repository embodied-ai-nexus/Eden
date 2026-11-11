"""Deploy against a real built env through an in-memory wire.

A deployment is constructed against a real, built env, as ``eden deploy`` does. The one double is
the transport (``connect`` / ``read_state`` / ``send_payload``, the hardware boundary):
:class:`LoopbackDeployment` records every payload it is sent and replays scripted
:class:`RobotState` readings. Entities, managers and terms are the real ones. A backend that
already has a transport seam is driven through it instead: :class:`ScriptedHandTransport` is the
Wuji hand's.
"""

from __future__ import annotations

import collections
import contextlib
from collections.abc import Iterator

import numpy as np
import torch

from eden.extensions.deployment.base import DeploymentBase
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.extensions.deployment.wuji_hand import NUM_JOINTS, HandTransport, StreamOptions, _checked


def robot_state(num_dofs: int, *, dofs_pos=None, base_quat=(1.0, 0.0, 0.0, 0.0), **fields) -> RobotState:
    """A reading of a robot standing still: zero velocities and torques, gravity on the accelerometer."""
    state = dict(
        stamp=0.0,
        base_quat=None if base_quat is None else np.asarray(base_quat, dtype=np.float64),
        base_ang_vel=np.zeros(3),
        base_lin_acc=np.array([0.0, 0.0, 9.81]),
        dofs_pos=np.zeros(num_dofs) if dofs_pos is None else np.asarray(dofs_pos, dtype=np.float64),
        dofs_vel=np.zeros(num_dofs),
        dofs_torque=np.zeros(num_dofs),
    )
    state.update(fields)
    return RobotState(**state)


def start_policy(dep) -> None:
    """START, then the handover tick, which holds while it reads the policy's first observation.

    The action a loop hands ``step`` on the START tick was computed before START, so it is never
    sent; the policy's first command goes out on the tick after. The handover's hold is dropped
    from :attr:`LoopbackDeployment.sent`, so a test sees only what the policy sent.
    """
    dep.request_start()
    dep.step(None)
    dep.sent.clear()


def assert_dofs_looked_up_by_name(dep, robot_options) -> None:
    """``write_dofs_targets`` and ``wire_gains`` map deployment DOFs by name on a connected backend.

    Both look a DOF name up in the base's name -> index map. Every SDK backend also keeps an
    Eden -> motor-slot map of its own, resolved at construction or ``connect()``; while the two
    shared an attribute, the backend's overwrote the base's, and both calls raised on the first
    extension-policy slot of a real run (an ``IndexError`` / ``TypeError`` from a string index).
    """
    n = dep.num_dofs
    dep.write_dofs_targets(torch.arange(float(n)), list(reversed(dep.dofs_name)))
    np.testing.assert_array_equal(dep.action_to_payload(None).dofs_pos, np.arange(float(n))[::-1])
    kp, kd = dep.wire_gains(robot_options)
    np.testing.assert_array_equal(kp, dep.default_dof_kp)
    np.testing.assert_array_equal(kd, dep.default_dof_kd)


class ScriptedOperator:
    """The operator's keyboard or remote: yields the commands a test presses, once each."""

    def __init__(self) -> None:
        self._pending: list[str] = []

    def press(self, *commands: str) -> None:
        self._pending.extend(commands)

    def poll(self) -> list[str]:
        pressed, self._pending = self._pending, []
        return pressed


class LoopbackDeployment(DeploymentBase):
    """A backend whose wire is in memory.

    ``send_payload`` low-passes the targets as every hardware backend does and records a copy in
    :attr:`sent`. ``read_state`` replays :attr:`readings` in order, then reports the robot standing
    still at the last commanded pose, as a perfectly tracking robot would, except that a joint named
    in :attr:`frozen` reads its pinned value whatever it is commanded (an encoder that stopped
    reporting).
    """

    def __init__(self, env, options) -> None:
        super().__init__(env, options)
        self.sent: list[RobotCommand] = []
        self.readings: collections.deque[RobotState] = collections.deque()
        self.frozen: dict[str, float] = {}
        self.operator = ScriptedOperator()

    def connect(self) -> None:
        self._setup_inputs()

    def close(self) -> None:
        self._teardown_common()

    def read_state(self) -> RobotState:
        if self.readings:
            state = self.readings.popleft()
        else:
            pose = self.sent[-1].dofs_pos if self.sent else self.default_dof_pos
            state = robot_state(self.num_dofs, dofs_pos=pose)
        if self.frozen:
            state.dofs_pos = np.array(state.dofs_pos, dtype=np.float64)
            for name, value in self.frozen.items():
                state.dofs_pos[self.dofs_name.index(name)] = value
        return state

    def send_payload(self, payload: RobotCommand) -> None:
        payload.dofs_pos = self._apply_target_lpf(np.asarray(payload.dofs_pos, dtype=np.float64))
        # Copied: ``action_to_payload`` hands out a view of the deployment's own target buffer.
        self.sent.append(
            RobotCommand(
                dofs_pos=np.array(payload.dofs_pos, dtype=np.float64),
                dofs_vel=np.array(payload.dofs_vel, dtype=np.float64),
                dofs_torque=np.array(payload.dofs_torque, dtype=np.float64),
                dofs_kp=None if payload.dofs_kp is None else np.array(payload.dofs_kp, dtype=np.float64),
                dofs_kd=None if payload.dofs_kd is None else np.array(payload.dofs_kd, dtype=np.float64),
            )
        )


class OperatorLoopbackDeployment(LoopbackDeployment):
    """A loopback backend whose operator is :attr:`operator`, pressing INIT/START/STOP as a test scripts it."""

    def _setup_inputs(self) -> None:
        self._inputs = [self.operator]


class FirmwareGainsLoopbackDeployment(LoopbackDeployment):
    """A loopback backend whose robot ignores each command's kp/kd, as the Wuji/XHand hands and the Franka do."""

    sends_command_gains = False


class ProgrammedGainsLoopbackDeployment(FirmwareGainsLoopbackDeployment):
    """...whose robot runs the env robot's gains, programmed once, as Wuji Hand 2 runs its nominal MIT gains."""

    def programmed_gains(self):
        return self.wire_gains()


class UnattendedLoopbackDeployment(LoopbackDeployment):
    """A loopback with no input source open, as in a headless run (the keyboard included, TTY or not)."""

    def _setup_inputs(self) -> None:
        self._inputs = []


class ScriptedHandTransport(HandTransport):
    """A Wuji hand's wire in memory, for ``WujiHandDeployment.use_transport``.

    :meth:`read` replays :attr:`reads` in order (the last one repeats) and :meth:`clock` the
    :attr:`times` stamps (then 0.0); ``fault`` / ``stale`` are what :meth:`fault` / :meth:`stale`
    report. :meth:`command` and :meth:`ease_to` validate their target as the real ones do, then record
    it in :attr:`sent` / :attr:`eased` rather than stream it: no sleeping, no sub-targets. :meth:`ease_to`
    runs its ``guard`` once, as the real one does before its first waypoint, and :meth:`hold` counts
    the periods it is asked to hold in :attr:`holds`.
    """

    def __init__(self, reads=(), clock=(), fault=None, stale=None, stream: StreamOptions | None = None) -> None:
        super().__init__(stream or StreamOptions())
        self.reads = [np.asarray(r, dtype=np.float64) for r in reads] or [np.zeros(NUM_JOINTS)]
        self.times = list(clock)
        self.sent: list[np.ndarray] = []
        self.eased: list[np.ndarray] = []
        self.holds = 0
        self._fault, self._stale = fault, stale
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def read(self) -> np.ndarray:
        return (self.reads.pop(0) if len(self.reads) > 1 else self.reads[0]).copy()

    def clock(self) -> float:
        return self.times.pop(0) if self.times else 0.0

    def fault(self) -> str | None:
        return self._fault

    def stale(self) -> str | None:
        return self._stale

    def command(self, target) -> None:
        self.sent.append(_checked(target).copy())

    def hold(self) -> None:
        self.holds += 1

    def ease_to(self, goal, duration_s, rate_hz, profile, guard=None) -> None:
        goal = _checked(goal)
        if guard is not None:
            guard()
        self.eased.append(goal.copy())

    def disconnect(self) -> None:
        self.connected = False


@contextlib.contextmanager
def deployments(env) -> Iterator:
    """Yield ``make(cls=LoopbackDeployment, *, connect=False, **options)``; undo every deployment it made on exit.

    ``make`` constructs ``cls(env, cls.configure(**options))`` (``sync`` defaults to ``False``) and, with
    ``connect=True``, connects it. On exit each deployment is closed, which also restores command
    overrides, so a backend's ``close()`` must tolerate being called again after a test closed it.
    Nothing in production hands an env back after a deploy, so there is no public undo of the entity
    swap: the teardown puts the sim entity and the terms repointed at the wrapper back by hand, then
    resets the env, so the next test sees the env as it was built.
    """
    made: list[DeploymentBase] = []

    def make(cls: type[DeploymentBase] = LoopbackDeployment, *, connect: bool = False, **options) -> DeploymentBase:
        options.setdefault("sync", False)
        deployment = cls(env, cls.configure(**options))
        made.append(deployment)
        if connect:
            deployment.connect()
        return deployment

    try:
        yield make
    finally:
        for deployment in reversed(made):
            deployment.close()
            # A composite swapped nothing itself: each of its parts swapped its own entity.
            for swapped in list(getattr(deployment, "parts", {}).values())[::-1] or [deployment]:
                sim_entity = swapped._state_entity._sim_entity
                env.entities[swapped.entity_name] = sim_entity
                for term, attr in swapped._repointed_refs:
                    setattr(term, attr, sim_entity)
        env.reset()


def robot_env_config(robot_options, dofs_name=None, *, observation_options=None, **action_options):
    """A plain RL task around one robot: one ``ImplicitPDController`` and a ``policy`` group reading ``dofs_pos``.

    ``dofs_name`` defaults to every DOF the robot declares; ``action_options`` reach the controller's
    ``configure`` (a ``modifier``, say). ``observation_options`` replaces the ``policy`` group. No
    commands and no motion reference.
    """
    from eden import actions, observations
    from eden.options import ObservationGroupOptions
    from eden.utils.configs import (
        ActionManagerOptions,
        EdenRLConfig,
        EnvOptions,
        ObservationManagerOptions,
        SceneOptions,
    )

    return EdenRLConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=robot_options),
        action_options=ActionManagerOptions(
            joint_pos=actions.ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=list(robot_options.dofs_name if dofs_name is None else dofs_name),
                **action_options,
            )
        ),
        observation_options=observation_options
        or ObservationManagerOptions(
            policy=ObservationGroupOptions(dofs_pos=observations.DofsPos.configure(entity_name="robot"))
        ),
    )


def robot_env(robot_options, dofs_name=None, *, observation_options=None, **action_options):
    """:func:`robot_env_config`, built and reset. Call it from a module-scoped fixture."""
    from tests.support.envs import build_env

    config = robot_env_config(robot_options, dofs_name, observation_options=observation_options, **action_options)
    return build_env(config, reset=True)
