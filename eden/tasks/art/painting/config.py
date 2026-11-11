"""The painting tasks: copy a painting in acrylic with a round brush, dipping it in a palette, RoboPaint's rules.

RoboPaint's acrylic discipline, on paintings made for Eden (``works.py``; RoboPaint's own plans are not released)
with a paint model of Eden's own (``paint.py``) and RoboPaint's picture and hand rules:

* ``art_painting_sunset`` — a sea, a sky and a low sun: 3 paints, 7 strokes.
* ``art_painting_fuji_study`` — a study of Hokusai's *Fine Wind, Clear Morning*: 5 paints in two layers.
* ``art_painting_fuji`` — the painting itself, as RoboPaint's ``acrylic_redfuji`` is set: 10 paints in four layers,
  brushes of 16 down to 3 mm.

The Hokusai works are planned from a picture the first time a task builds (some 10-20 s on the CPU), then read
from Eden's cache (``works.py``).

The brush's paint runs out and the brush holds one paint at a time, so it is dipped in the palette beside the sheet
for a new paint and for more of the same one. A copy succeeds when the picture passes (mean ΔE <= 2 and its 90th
percentile <= 6 after a 1 mm blur, lightness SSIM >= 0.92, edge correlation >= 0.8, the painted cells' IoU >= 0.95)
and so does the hand: at most 1.3 times the plan's strokes, a path on the paper within 3 times the plan's, and at
most a tenth of it in scribbles. Placement counts: unlike RoboPaint, which registers the copy to the target within
5 mm first, the copy must be where the target is.

The arm acts through differential IK at the brush's undeformed tip, 20 times a second: up to 1 cm and 0.05 rad a
step. The policy sees the arm's joints, the tip's pose, the paint on the brush, the paint the plan needs next and
its well, the next plan points, the strokes left and the grade. Rewards pay for the error the copy removes and the
plan's points first reached, pull the tip toward the next plan point at its depth, and pay a bonus on success.

``eden inference --rec`` records :meth:`PaintingConfig.recording_camera`: the sheet, the palette, the brush and the
hand from over the arm, the painting upright, with the paint painted on the sheet in its colours
(``PlanePaintOverlay``).
The paint is the metric's, not geometry in the scene, so a camera without the overlay shows a blank sheet. The
training config declares no camera.
"""

from __future__ import annotations

from eden import observations, rewards, terminations
from eden.entities.camera_overlays import PlanePaintOverlay
from eden.options import CameraOptions, ObservationGroupOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.tasks.art import common, workcell
from eden.tasks.art.painting.custom_terms import BrushPaint, NextPaint, PaintingCopy, PaintingErrorDrop
from eden.tasks.art.painting.palette import PaletteOptions
from eden.tasks.art.painting.works import WORKS
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.art.terms import CopyGrade, NextPlanPoints, StrokesLeft, TipPose
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

#: Where the recording camera stands from the sheet's centre (m), and where it looks: behind the sheet's bottom edge
#: and high over the arm, so the painting reads upright (its top is away from the robot), with the palette to its left.
RECORDING_OFFSET = (-0.36, 0.05, 0.46)
RECORDING_LOOKAT = (0.02, 0.04, 0.0)


def _scene(work: str) -> SceneOptions:
    return SceneOptions(**workcell.scene("paintbrush", WORKS[work].sheet), palette=PaletteOptions(work=work))


def _painting_metric(work: str) -> MetricManagerOptions:
    return MetricManagerOptions(painting=PaintingCopy.configure(work=work, **WORKS[work].rule))


_TERMINATIONS = TerminationManagerOptions(
    timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    painted=terminations.OnMetricSuccess.configure(metric_name="painting"),
)

_OBSERVATIONS = {
    "arm_dofs_pos": observations.DofsPos.configure(entity_name="robot"),
    "arm_dofs_vel": observations.DofsVel.configure(entity_name="robot"),
    "tip_pose": TipPose.configure(metric_name="painting"),
    "brush_paint": BrushPaint.configure(),
    "next_paint": NextPaint.configure(),
    "next_points": NextPlanPoints.configure(metric_name="painting"),
    "strokes_left": StrokesLeft.configure(metric_name="painting"),
    "grade": CopyGrade.configure(metric_name="painting"),
    "last_action": ObsTerm.configure(func=observations.last_action),
}


class PaintingConfig(EdenRLConfig):
    """What every painting task shares; a task sets its work (and with it the palette, the sheet and the rules)."""

    env_options: EnvOptions = common.env_options(90.0, num_envs=64)
    scene_options: SceneOptions = _scene("sunset")
    metric_options: MetricManagerOptions = _painting_metric("sunset")

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        # No term configures noise, so neither group is corrupted: the policy sees what the critic sees.
        policy=ObservationGroupOptions(**_OBSERVATIONS, enable_corruption=False),
        critic=ObservationGroupOptions(**_OBSERVATIONS, enable_corruption=False),
    )

    action_options: ActionManagerOptions = common.arm_action("paintbrush")

    reward_options: RewardManagerOptions = RewardManagerOptions(
        error_drop=PaintingErrorDrop.configure(weight=100.0),
        **common.plan_rewards("painting"),
        success=rewards.TaskSuccess.configure(weight=50.0),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        arm_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "robot", "max_vel": 1.0}
        ),
    )

    termination_options: TerminationManagerOptions = _TERMINATIONS
    event_options: EventManagerOptions = common.reset_events()

    runner_options: RslRlOnPolicyRunnerOptions = common.ppo_runner("painting")

    def recording_camera(self) -> CameraOptions:
        """From over the arm: the sheet (the painting upright), the palette, the brush and the hand, with the paint
        painted on the sheet in its own colours: the paint of the config's
        :class:`~eden.tasks.art.painting.custom_terms.PaintingCopy` metric, found by its term, whatever its key.
        """
        metric_name = common.copy_metric_key(self, PaintingCopy, "paint")
        sx, sy, sz = workcell.SHEET_CENTER
        return CameraOptions(
            cam_pos=(sx + RECORDING_OFFSET[0], sy + RECORDING_OFFSET[1], sz + RECORDING_OFFSET[2]),
            cam_lookat=(sx + RECORDING_LOOKAT[0], sy + RECORDING_LOOKAT[1], sz + RECORDING_LOOKAT[2]),
            cam_fov=34.0,
            cam_resolution=(960, 540),
            overlays=[PlanePaintOverlay.configure(metric_name=metric_name)],
        )


@TASK_REGISTRY.register()
class ArtPaintingSunsetConfig(PaintingConfig):
    """Paint a sea, a sky and a low sun: 3 paints, 7 strokes."""


@TASK_REGISTRY.register()
class ArtPaintingFujiStudyConfig(PaintingConfig):
    """Paint a study of Hokusai's *Fine Wind, Clear Morning*: 5 paints in two layers."""

    env_options: EnvOptions = common.env_options(800.0, num_envs=64)
    scene_options: SceneOptions = _scene("fuji_study")
    metric_options: MetricManagerOptions = _painting_metric("fuji_study")


@TASK_REGISTRY.register()
class ArtPaintingFujiConfig(PaintingConfig):
    """Paint Hokusai's *Fine Wind, Clear Morning*: 10 paints in four layers."""

    env_options: EnvOptions = common.env_options(1200.0, num_envs=64)
    scene_options: SceneOptions = _scene("fuji")
    metric_options: MetricManagerOptions = _painting_metric("fuji")
