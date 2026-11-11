"""The calligraphy tasks: copy a written work with a Chinese brush or a pointed pen, RoboPaint's writing rules.

RoboPaint's writing disciplines (calligraphy with a Chinese brush, lettering with a pointed pen), on works drawn
for Eden (``works.py``; RoboPaint's own exemplars are not released) with RoboPaint's tool models and rules:

* ``art_calligraphy_dai`` — 大 (big), kaisho (楷書), 3 strokes, with the brush.
* ``art_calligraphy_san`` — 山 (mountain), kaisho (楷書), 3 strokes, with the brush.
* ``art_calligraphy_ei`` — 永 (eternity), kaisho (楷書), 5 strokes, with the brush.
* ``art_calligraphy_lettering_minimum`` — *minimum* in round hand, 9 strokes, with the pointed pen.

A copy succeeds when the picture passes (IoU >= 0.8 with the target, boundary F >= 0.75, every stroke >= 80 %
covered, stray ink <= 4 % of the target; for lettering an IoU of 0.7, boundary F 0.8 and stroke coverage 0.75,
the *minimum* work's rule, :data:`~eden.tasks.art.calligraphy.works.LETTERING_RULE`) and so does the hand: no more
strokes than the work has, a path on the paper within 2.5 times the plan's, and a mean mark width at least a third
of the plan's. Placement counts: unlike RoboPaint, which registers the copy to the target within 5 mm first, the
copy must be where the target is.

The arm acts through differential IK at the tool's undeformed tip, 20 times a second: up to 1 cm and 0.05 rad a
step. The policy sees the arm's joints, the tip's pose, the tool's last mark, the next plan points, the strokes
left and the grade. Rewards pay for the target first inked (100 for all of it) and the plan's points first
reached, charge stray ink, pull the tip toward the next plan point at its depth, and pay a bonus on success.

``eden inference --rec`` records :meth:`CalligraphyConfig.recording_camera`: the sheet, the tool and the hand from
above the robot's right side, with the ink painted on the sheet (``PlanePaintOverlay``). The ink is the metric's,
not geometry in the scene, so a camera without the overlay shows a blank sheet. The training config declares no
camera.
"""

from __future__ import annotations

from eden import observations, rewards, terminations
from eden.entities.camera_overlays import PlanePaintOverlay
from eden.options import CameraOptions, ObservationGroupOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.tasks.art import common, workcell
from eden.tasks.art.calligraphy.custom_terms import CalligraphyWriting, StraySheetInkGain, TargetCoverageGain, ToolMark
from eden.tasks.art.calligraphy.grading import INK_COLOR
from eden.tasks.art.calligraphy.works import WORKS
from eden.tasks.art.terms import CopyGrade, NextPlanPoints, StrokesLeft, TipPose
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import DoneTerm, ObsTerm, RewardTerm
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

#: Where the recording camera stands from the sheet's centre (m): above the desk, out to the robot's right and a
#: little back toward it. Close enough that the pen's hairlines show; the arm above the wrist leaves the frame.
RECORDING_OFFSET = (-0.15, -0.45, 0.50)


def _scene(work: str) -> SceneOptions:
    spec = WORKS[work]
    return SceneOptions(**workcell.scene(spec.plan.tool, spec.sheet))


def _writing_metric(work: str) -> MetricManagerOptions:
    return MetricManagerOptions(writing=CalligraphyWriting.configure(work=work, **WORKS[work].rule))


_TERMINATIONS = TerminationManagerOptions(
    timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    written=terminations.OnMetricSuccess.configure(metric_name="writing"),
)


_OBSERVATIONS = {
    "arm_dofs_pos": observations.DofsPos.configure(entity_name="robot"),
    "arm_dofs_vel": observations.DofsVel.configure(entity_name="robot"),
    "tip_pose": TipPose.configure(metric_name="writing"),
    "tool_mark": ToolMark.configure(),
    "next_points": NextPlanPoints.configure(metric_name="writing"),
    "strokes_left": StrokesLeft.configure(metric_name="writing"),
    "grade": CopyGrade.configure(metric_name="writing"),
    "last_action": ObsTerm.configure(func=observations.last_action),
}


class CalligraphyConfig(EdenRLConfig):
    """What every writing task shares; a task sets its work (and with it the tool, the sheet and the rules)."""

    env_options: EnvOptions = common.env_options(60.0, num_envs=256)
    scene_options: SceneOptions = _scene("ei")
    metric_options: MetricManagerOptions = _writing_metric("ei")

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        # No term configures noise, so neither group is corrupted: the policy sees what the critic sees.
        policy=ObservationGroupOptions(**_OBSERVATIONS, enable_corruption=False),
        critic=ObservationGroupOptions(**_OBSERVATIONS, enable_corruption=False),
    )

    action_options: ActionManagerOptions = common.arm_action("brush")

    reward_options: RewardManagerOptions = RewardManagerOptions(
        inked=TargetCoverageGain.configure(weight=100.0),
        **common.plan_rewards("writing"),
        stray=StraySheetInkGain.configure(weight=-50.0),
        success=rewards.TaskSuccess.configure(weight=50.0),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        arm_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "robot", "max_vel": 1.0}
        ),
    )

    termination_options: TerminationManagerOptions = _TERMINATIONS
    event_options: EventManagerOptions = common.reset_events()

    runner_options: RslRlOnPolicyRunnerOptions = common.ppo_runner("calligraphy")

    def recording_camera(self) -> CameraOptions:
        """From above the robot's right side: the sheet, the tool and the hand, with the ink painted on in
        :data:`~eden.tasks.art.calligraphy.grading.INK_COLOR`, the ink of the config's
        :class:`~eden.tasks.art.calligraphy.custom_terms.CalligraphyWriting` metric, found by its term, whatever its
        key.
        """
        metric_name = common.copy_metric_key(self, CalligraphyWriting, "ink")
        sx, sy, sz = workcell.SHEET_CENTER
        return CameraOptions(
            cam_pos=(sx + RECORDING_OFFSET[0], sy + RECORDING_OFFSET[1], sz + RECORDING_OFFSET[2]),
            cam_lookat=(sx - 0.03, sy, sz + 0.06),
            cam_fov=40.0,
            cam_resolution=(960, 540),
            overlays=[PlanePaintOverlay.configure(metric_name=metric_name, color=INK_COLOR)],
        )


@TASK_REGISTRY.register()
class ArtCalligraphyDaiConfig(CalligraphyConfig):
    """Write 大 (big) with the Chinese brush: 3 strokes, kaisho."""

    scene_options: SceneOptions = _scene("dai")
    metric_options: MetricManagerOptions = _writing_metric("dai")


@TASK_REGISTRY.register()
class ArtCalligraphySanConfig(CalligraphyConfig):
    """Write 山 (mountain) with the Chinese brush: 3 strokes, kaisho."""

    scene_options: SceneOptions = _scene("san")
    metric_options: MetricManagerOptions = _writing_metric("san")


@TASK_REGISTRY.register()
class ArtCalligraphyEiConfig(CalligraphyConfig):
    """Write 永 (eternity) with the Chinese brush: 5 strokes, kaisho, the "eight principles" character."""

    scene_options: SceneOptions = _scene("ei")
    metric_options: MetricManagerOptions = _writing_metric("ei")


@TASK_REGISTRY.register()
class ArtCalligraphyLetteringMinimumConfig(CalligraphyConfig):
    """Write *minimum* in round hand with the pointed pen: 7 letters and 2 dots, 9 strokes."""

    env_options: EnvOptions = common.env_options(120.0, num_envs=256)
    scene_options: SceneOptions = _scene("minimum")
    metric_options: MetricManagerOptions = _writing_metric("minimum")
    action_options: ActionManagerOptions = common.arm_action("pen")
