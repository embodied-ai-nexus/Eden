"""Sim-twin config for XHand1 system identification.

A minimal ``EdenRLConfig`` that spawns a single fixed-base XHand1
controlled by an ``ExplicitPDController``. No rewards, no commands, no
termination logic: the config is only used as a rollout substrate for
``eden.extensions.sysid``. Explicit (rather than implicit) PD avoids the
``kd * substep_dt`` armature correction Genesis applies under implicit
damping, which would otherwise inject ~100× phantom inertia and bias
the identification.

The ``num_envs`` field is exposed so the identification script can spin
up a single env for serial fits (scipy least-squares, CMA-ES serial) or
an env sized to the CMA-ES population for batched evaluation.
"""

from __future__ import annotations

import argparse

import eden as en
from eden.constants import ReferenceSource
from eden.managers.terms.recorders.sysid import SysIDRecorder
from eden.options import ObservationGroupOptions
from eden.options.entities import RobotOptions
from eden.options.managers.recorders import RecorderManagerOptions
from eden.utils.configs import (
    ActionManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

DEFAULT_SIM_DT = 0.005663
DEFAULT_DECIMATION = 4


def make_xhand1_sim_config(
    side: str = "left",
    num_envs: int = 1,
    sim_dt: float = DEFAULT_SIM_DT,
    decimation: int = DEFAULT_DECIMATION,
    kp: float = 50.0,
    kd: float = 25.0,
) -> EdenRLConfig:
    """Build an ``EdenRLConfig`` wrapping a single XHand1 hand.

    The default ``sim_dt * decimation = 22.65 ms`` (≈44 Hz) matches the
    actual cadence at which RoboTera's RS485 SDK round-trips
    ``send_payload + read_state`` — picking the same cadence here keeps
    the rollout's chirp-frequency content aligned with what was recorded
    on the real hand. Override ``kp`` / ``kd`` to match whatever gains
    were used during data collection.
    """

    class _XHand1SysIDScene(en.scenes.Empty):
        robot: RobotOptions = (en.robots.XHand1_R if side == "right" else en.robots.XHand1_L)(
            is_fixed_base=True,
            default_root_pos=(0.0, 0.0, 0.1),  # raise the hand up to avoid collision with the floor
            # XHand1's RobotOptions declares default_dofs_pos={} (no safe-hold
            # pose). DeploymentBase needs a concretely-sized default pose for
            # DeploymentRecorder to add excitation offsets on top of — pin it
            # to the URDF's qpos0 (all zeros for XHand1 = fingers extended).
            default_dofs_pos={"*": 0.0},
            default_dofs_kp={"*": kp},
            default_dofs_kd={"*": kd},
        )

    class _XHand1SysIDConfig(EdenRLConfig):
        env_options: EnvOptions = EnvOptions(
            num_envs=num_envs,
            episode_length_s=120.0,
            sim_dt=sim_dt,
            decimation=decimation,
            sim_substeps=1,
            enable_multi_contact=False,
        )

        scene_options: SceneOptions = _XHand1SysIDScene()

        observation_options: ObservationManagerOptions = ObservationManagerOptions(
            policy=ObservationGroupOptions(
                dofs_pos=en.observations.DofsPos.configure(entity_name="robot"),
                dofs_vel=en.observations.DofsVel.configure(entity_name="robot"),
                enable_corruption=False,
            ),
        )

        action_options: ActionManagerOptions = ActionManagerOptions(
            joint_pos_controller=en.actions.ExplicitPDController.configure(
                entity_name="robot",
                dofs_name="*",
                scale=1.0,
                reference_source=ReferenceSource.DEFAULT,
            ),
        )

        reward_options: RewardManagerOptions = RewardManagerOptions()
        termination_options: TerminationManagerOptions = TerminationManagerOptions()
        event_options: EventManagerOptions = EventManagerOptions()
        recorder_options: RecorderManagerOptions = RecorderManagerOptions(
            sysid=SysIDRecorder.configure(
                entity_name="robot",
                include_torque=True,
                include_base=False,  # fixed-base hand — no floating-base signals.
            ),
        )

    return _XHand1SysIDConfig()


def make_argparser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--side", choices=["right", "left"], default="right", help="Which hand to drive")
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=DEFAULT_SIM_DT,
        help="Sim physics timestep for the XHand1 sysid twin.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=DEFAULT_DECIMATION,
        help="Control decimation for the XHand1 sysid twin. The replay step is sim_dt * decimation.",
    )
    return parser
