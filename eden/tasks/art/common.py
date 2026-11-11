"""The config pieces both art families build their tasks from (``calligraphy/config.py``, ``painting/config.py``).

The timing (:func:`env_options`), the arm's action (:func:`arm_action`), the reset noise (:func:`reset_events`), the
plan-following rewards (:func:`plan_rewards`), the training runner (:func:`ppo_runner`), and the lookup of the
config's copy metric that a recording camera paints (:func:`copy_metric_key`). A family's config adds its work, its
own terms and its camera.
"""

from __future__ import annotations

from typing import Any

from eden import EventMode, events
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks.art import workcell
from eden.tasks.art.terms import CopyMetric, PlanProgressGain, ReachNextPlanPoint
from eden.terms import EventTerm
from eden.utils.configs import ActionManagerOptions, EdenRLConfig, EnvOptions, EventManagerOptions
from eden.utils.misc import to_snake_case

CONTROL_HZ = 20
SIM_DT = 0.005


def env_options(episode_length_s: float, *, num_envs: int) -> EnvOptions:
    """20 Hz control over a 5 ms / 2-substep step (the tool is welded: nothing rests on a friction grip)."""
    return EnvOptions(
        num_envs=num_envs,
        episode_length_s=episode_length_s,
        sim_dt=SIM_DT,
        sim_substeps=2,
        decimation=round(1.0 / (CONTROL_HZ * SIM_DT)),
    )


def arm_action(tool: workcell.ToolKind) -> ActionManagerOptions:
    """Differential IK at the held ``tool``'s undeformed tip, up to 1 cm and 0.05 rad a step: a rotation turns the
    tool about where it marks the paper.
    """
    return ActionManagerOptions(
        arm=DifferentialIKController.configure(
            entity_name="robot",
            ee_link_name="attachment",
            ee_local_pos=workcell.tool_tip_in_attachment(tool),
            dofs_name=list(workcell.TOOLS[tool].home),
            position_scale=0.01,
            orientation_scale=0.05,
            damping=0.05,
            max_dq=0.1,
        ),
    )


def reset_events() -> EventManagerOptions:
    """The arm's joints set off their home by up to 0.02 rad at each reset."""
    return EventManagerOptions(
        arm_noise=EventTerm.configure(
            func=events.reset_dofs_by_offset,
            mode=EventMode.RESET,
            params={"entity_name": "robot", "dofs_pos_range": (-0.02, 0.02)},
        ),
    )


def plan_rewards(metric_name: str) -> dict[str, Any]:
    """The plan-following rewards of the copy metric ``metric_name``, for a ``RewardManagerOptions``: the share of
    the plan's points first reached, and the pull toward the next point at its depth, broad and fine.
    """
    return {
        "progress": PlanProgressGain.configure(metric_name=metric_name, weight=20.0),
        "reach": ReachNextPlanPoint.configure(metric_name=metric_name, std=0.03, weight=0.05),
        "reach_fine": ReachNextPlanPoint.configure(metric_name=metric_name, std=0.005, weight=0.1),
    }


def ppo_runner(experiment_name: str) -> RslRlOnPolicyRunnerOptions:
    """The tasks' PPO runner: a 512-256-128 ELU actor and critic, 48 steps per env per iteration."""
    return RslRlOnPolicyRunnerOptions(
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[512, 256, 128]),
        num_steps_per_env=48,
        max_iterations=5000,
        save_interval=100,
        experiment_name=experiment_name,
        run_name="",
    )


def copy_metric_key(config: EdenRLConfig, term: type[CopyMetric], marks: str) -> str:
    """The key of ``config``'s one ``term`` metric, whatever it is: the metric whose ``marks`` (``"ink"``,
    ``"paint"``) its recording camera paints.
    """
    name = to_snake_case(term.__name__)
    metrics = {key: getattr(config.metric_options, key).name for key in config.metric_options.keys()}
    found = [key for key, metric in metrics.items() if metric == name]
    if len(found) != 1:
        raise ValueError(
            f"{type(config).__name__}.recording_camera() paints the {marks} of the config's {term.__name__} metric "
            f"('{name}'), but metric_options has {len(found)} of them (metrics: {metrics}). Keep exactly one, or "
            f"override recording_camera() to name the metric whose {marks} to paint."
        )
    return found[0]
