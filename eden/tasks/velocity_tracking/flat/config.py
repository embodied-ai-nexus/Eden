"""Flat-terrain velocity-tracking task configuration."""

from __future__ import annotations

import eden as en
from eden.options import SceneOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.velocity_tracking.base_config import (
    VelocityTrackingBaseConfig,
    VelocityTrackingObsMod,
    VelocityTrackingRobotMod,
)
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions


@TASK_REGISTRY.register(modifiers=(VelocityTrackingRobotMod, VelocityTrackingObsMod))
class VelocityTrackingFlatConfig(VelocityTrackingBaseConfig):
    """Velocity tracking for a bipedal humanoid on flat terrain.

    Robot is selected by :class:`VelocityTrackingRobotMod` (``--robot g1``
    default, ``--robot oli``). The class-level scene holds a G1 so the config
    instantiates without the mod, but the robot-bound term fields stay empty until
    the mod fills them, and building the env raises on them (``check_bindings``).
    The policy observation recipe is selected by :class:`VelocityTrackingObsMod`
    (``--obs_preset proprioceptive`` default, ``--obs_preset privileged``).

    Not yet implemented (relative to mjlab flat):

    - ``height_scan`` observation -- omitted because flat terrain always
      returns zero.
    - Critic-only obs (``foot_height``, ``foot_air_time``, ``foot_contact``,
      ``foot_contact_forces``) and contact-based rewards (``feet_air_time``,
      ``foot_swing_height``, ``soft_landing``) -- available via
      ``gs.sensors.Contact`` / ``gs.sensors.ContactForce``.
    - ``angular_momentum`` reward.
    - ``rel_forward_envs`` command option (not in Eden's
      ``UniformVelocityCommand``).
    """

    scene_options: SceneOptions = en.scenes.Empty(robot=en.robots.G1Dofs29())
    # Same runner as the base, only the experiment (log dir) name differs; the robot
    # mod prefixes the robot key -> "g1_velocity_tracking_flat". Shallow copy is fine:
    # pydantic deep-copies field defaults per instance at construction.
    runner_options: RslRlOnPolicyRunnerOptions = VelocityTrackingBaseConfig.model_fields[
        "runner_options"
    ].default.model_copy(update={"experiment_name": "velocity_tracking_flat"})
