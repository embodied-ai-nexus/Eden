"""The whiteboard drawing tasks: RoboWorld ``whiteboard``'s three drawing tasks, on a Panda holding a marker.

Port of the ``write-text``, ``write-in-colour`` and ``draw-table`` tasks of ``gym-anything/robotics``'s
``whiteboard`` environment (contract ``whiteboard-tasks-v1``): the same references at the same sizes and
places on the same board, the same ink law, and the same verifiers (``grading.py`` says what is and is not
ported). The workcell differs (``common.py``): a Panda on a stand with the marker already in its Franka Hand,
where the original's G1 has to take a capped marker from the tray and uncap it.

* ``whiteboard_write_text`` — write HELLO (0.60 m) in black; success at a lettering F1 >= 0.85.
* ``whiteboard_write_in_colour`` — write WELCOME (0.70 m) in green; success at F1 >= 0.85.
* ``whiteboard_draw_table`` — draw the blank table (0.60 m: thick frame, header line, two column lines) in blue;
  success when every line is >= 90 % covered with no gap over 10 mm and <= 1 cm^2 of ink is off the table.

The original's "no other colour on the board" check is dropped: the hand holds one marker. Its text tasks put no
condition on the tools; neither does this port, beyond the marker staying in the hand (the episode ends when it
drops).

The arm acts through differential IK at the nib's tip, 20 times a second: up to 1 cm and 0.05 rad a step. The
policy sees the arm's joints, the nib's pose and press, the drawing's grade and the next reference points to ink.
``eden inference --rec`` records :meth:`WhiteboardDrawingConfig.recording_camera`: the arm and the whole board from
behind the robot's right shoulder, with the ink painted on the board (``PlanePaintOverlay``). The ink is the
metric's, not geometry in the scene, so a camera without the overlay shows a clean board. The training config
declares no camera.

Rewards pay for reference points first inked (100 for the whole drawing), charge visible ink off the drawing,
pull the nib toward the next point on the board's face, and pay a bonus on success. Every per-step term but the
small penalties is positive, so dropping the marker, which ends the episode, forfeits the rest of it.
"""

from __future__ import annotations

from eden import EventMode, events, observations, rewards, terminations
from eden.entities.camera_overlays import PlanePaintOverlay
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options import CameraOptions, ObservationGroupOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.whiteboard import common
from eden.tasks.whiteboard.custom_terms import (
    MarkerDropped,
    NextStrokePoints,
    NibForce,
    NibPose,
    PlaceMarkerInGrasp,
    ReachNextStrokePoint,
    StrayInkGain,
    StrokeCoverageGain,
    WhiteboardDrawing,
)
from eden.terms import DoneTerm, EventTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

CONTROL_HZ = 20
SIM_DT = 0.005
#: The original's lettering F1 threshold; the table's grade is 1.0 exactly when it passes.
TEXT_F1_THRESHOLD = 0.85


def _env_options(episode_length_s: float) -> EnvOptions:
    """20 Hz control over a 5 ms / 2-substep step, multi-contact and torsional friction on (the pads' grip)."""
    return EnvOptions(
        num_envs=512,
        episode_length_s=episode_length_s,
        sim_dt=SIM_DT,
        sim_substeps=2,
        decimation=round(1.0 / (CONTROL_HZ * SIM_DT)),
        enable_multi_contact=True,
        enable_torsional_friction=True,
    )


def _drawing_metric(drawing: str) -> MetricManagerOptions:
    threshold = 1.0 if drawing == "table" else TEXT_F1_THRESHOLD
    return MetricManagerOptions(drawing=WhiteboardDrawing.configure(drawing=drawing, success_threshold=threshold))


_OBSERVATIONS = {
    "arm_dofs_pos": observations.DofsPos.configure(entity_name="robot"),
    "arm_dofs_vel": observations.DofsVel.configure(entity_name="robot"),
    "nib_pose": NibPose.configure(),
    "nib_force": NibForce.configure(),
    "next_points": NextStrokePoints.configure(),
    "grade": observations.MetricObservation.configure(metric_name="drawing"),
    "last_action": ObsTerm.configure(func=observations.last_action),
}


class WhiteboardDrawingConfig(EdenRLConfig):
    """What every whiteboard drawing task shares; a task sets its marker, its drawing and its episode length."""

    env_options: EnvOptions = _env_options(90.0)

    scene_options: SceneOptions = SceneOptions(**common.whiteboard_scene("black"))

    metric_options: MetricManagerOptions = _drawing_metric("hello")

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        # No term configures noise, so neither group is corrupted: the policy sees what the critic sees.
        policy=ObservationGroupOptions(**_OBSERVATIONS, enable_corruption=False),
        critic=ObservationGroupOptions(**_OBSERVATIONS, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        arm=DifferentialIKController.configure(
            entity_name="robot",
            ee_link_name="attachment",
            # The control point is the nib's tip (uncompressed): a rotation turns the marker about where it writes.
            ee_local_pos=(0.0, 0.0, common.PandaFrankaHand().tcp_offset[2] + common.NIB_AHEAD_OF_GRIP),
            dofs_name=list(common.HOME),
            position_scale=0.01,
            orientation_scale=0.05,
            damping=0.05,
            max_dq=0.1,
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        inked=StrokeCoverageGain.configure(weight=100.0),
        stray=StrayInkGain.configure(weight=-5.0),
        reach=ReachNextStrokePoint.configure(std=0.05, weight=0.05),
        reach_fine=ReachNextStrokePoint.configure(std=0.01, weight=0.1),
        success=rewards.TaskSuccess.configure(weight=50.0),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        arm_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "robot", "max_vel": 1.0}
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        drawn=terminations.OnMetricSuccess.configure(metric_name="drawing"),
        marker_dropped=MarkerDropped.configure(),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        arm_noise=EventTerm.configure(
            func=events.reset_dofs_by_offset,
            mode=EventMode.RESET,
            params={"entity_name": "robot", "dofs_pos_range": (-0.02, 0.02)},
        ),
        # After the arm's reset, so the marker goes where the pads ended up.
        marker_in_hand=PlaceMarkerInGrasp.configure(mode=EventMode.RESET, priority=10),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
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
        experiment_name="whiteboard",
        run_name="",
    )

    def recording_camera(self) -> CameraOptions:
        """Behind the robot's right shoulder: the arm and the whole board, with the drawing's ink painted on in the
        marker's colour. The marker is the scene entity the drawing metric grades (its ``marker_name``), so the colour
        has the one source the task sets it in.
        """
        marker_name = self.metric_options.drawing.marker_name
        marker = getattr(self.scene_options, marker_name, None)
        if not isinstance(marker, common.WhiteboardMarkerOptions):
            raise ValueError(
                f"{type(self).__name__}.recording_camera() paints the ink in the marker's colour, read from the scene "
                f"entity the drawing metric grades (marker_name={marker_name!r}), but that entity is "
                f"{type(marker).__name__ if marker is not None else 'missing'} (scene entities: "
                f"{list(self.scene_options.keys())}). Make it a WhiteboardMarkerOptions, or override recording_camera() "
                "to pass the overlay its color."
            )
        ink = common.COLORS[marker.ink]
        return CameraOptions(
            cam_pos=(-0.30, -1.05, 1.68),
            cam_lookat=(0.45, 0.02, 1.18),
            cam_fov=42.0,
            cam_resolution=(960, 540),
            overlays=[PlanePaintOverlay.configure(metric_name="drawing", color=ink)],
        )


@TASK_REGISTRY.register()
class WhiteboardWriteTextConfig(WhiteboardDrawingConfig):
    """Write HELLO in black, where and as large as the reference shows (the original's ``write-text``)."""

    scene_options: SceneOptions = SceneOptions(**common.whiteboard_scene("black"))
    metric_options: MetricManagerOptions = _drawing_metric("hello")


@TASK_REGISTRY.register()
class WhiteboardWriteInColourConfig(WhiteboardDrawingConfig):
    """Write WELCOME on the whiteboard in green (the original's ``write-in-colour``)."""

    env_options: EnvOptions = _env_options(120.0)
    scene_options: SceneOptions = SceneOptions(**common.whiteboard_scene("green"))
    metric_options: MetricManagerOptions = _drawing_metric("welcome")


@TASK_REGISTRY.register()
class WhiteboardDrawTableConfig(WhiteboardDrawingConfig):
    """Draw the blank table in blue: frame, header line, two column lines (the original's ``draw-table``)."""

    env_options: EnvOptions = _env_options(240.0)
    scene_options: SceneOptions = SceneOptions(**common.whiteboard_scene("blue"))
    metric_options: MetricManagerOptions = _drawing_metric("table")
