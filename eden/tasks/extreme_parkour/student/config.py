"""Extreme Parkour student task configuration (Unitree Go2, depth distillation).

Student stage of the IsaacLab/legged_gym "Extreme Parkour with Legged Robots"
(Cheng et al. 2023) port: the same parkour environment as
:mod:`~eden.tasks.extreme_parkour.teacher.config`, extended with the forward
depth camera the vision policy is distilled onto. The policy itself is trained
by DAgger-style distillation from a teacher checkpoint — the runner options select
:class:`~eden.learning.wrappers.rsl_rl.rma.distillation.RmaDepthDistillationRunner`::

    eden train --task extreme_parkour_student --checkpoint logs/extreme_parkour_teacher/<run>/model_<it>.pt
    eden inference --task extreme_parkour_student --checkpoint logs/extreme_parkour_student/<run>/model_<it>.pt

Deltas vs the teacher task (all from the source student setup):

* **192 envs** (the depth camera dominates memory/compute; source value).
* **Depth camera** ``depth``: 106x60, 87 deg horizontal FOV, mounted on the
  ``base`` link at ``(0.33, 0, 0.08)`` looking forward pitched 20 deg down,
  rendered by the gs-madrona batch renderer (rasterizer mode — depth needs no
  lighting). Two extra observation groups: ``depth_camera`` (processed 58x87
  frame with one-interval latency) and ``delta_yaw_ok`` (the yaw-substitution
  gate). See the terms' docstrings for the processing contract.
* **Camera-extrinsics DR**: per-env mount pitch offset, uniform +-5 deg at
  startup (``RandomizeCameraMountPitch``).
* **Action delay**: the applied position target lags the policy output by one
  control step (20 ms), via ``ActionDelay(delay_range_s=(0.02, 0.02))`` — four
  physics steps. The observation still carries the undelayed action, as in the
  source.
* **Terrain**: coarser 0.1 m horizontal scale; corridor courses narrowed to
  ``y_range (-0.1, 0.1)``; per-row column mix follows the source proportions —
  4 gap, 4 hurdle, 1 flat, 4 step, 4 stepping-stones, 3 demo (the fixed
  showcase course, only present in the student phase).

Known deviations from the references (documented, deliberate):

* The depth render is metric **plane depth** (madrona z-buffer) like the
  original Isaac Gym implementation; the IsaacLab port's ray-cast camera
  returns euclidean ray length instead.
* Rewards are inherited from the teacher unchanged. Distillation never reads
  them — the source zeroes its student reward set to save compute — but keeping
  them gives meaningful episode-return logging during distillation.
* The source randomizes the camera mount once per training run; here the
  startup event re-fires per run the same way (identical semantics).
"""

from __future__ import annotations

import math

import numpy as np
from genesis.options.renderers import RendererOptions
from genesis.utils.geom import R_to_quat, euler_to_R

import eden as en
from eden.constants import ReferenceSource
from eden.learning.wrappers.rsl_rl.rma import (
    RslRlRmaDepthDistillationAlgorithmOptions,
    RslRlRmaDepthDistillationRunnerOptions,
)
from eden.managers.modifiers.actions import ActionDelay
from eden.options import BatchRendererOptions, ObservationGroupOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.extreme_parkour.custom_terms.events import RandomizeCameraMountPitch
from eden.tasks.extreme_parkour.custom_terms.observations import (
    _DEPTH_RAW_RESOLUTION,
    ParkourDeltaYawOk,
    ParkourDepthImage,
)
from eden.tasks.extreme_parkour.teacher.config import (
    _GO2,
    _ROWS,
    ExtremeParkourTeacherConfig,
    _course_generators,
    _make_env_options,
    _make_event_terms,
    _make_terrain_scene,
    _policy_observation_group,
    rma_actor_options,
)
from eden.utils.configs import (
    ActionManagerOptions,
    EnvOptions,
    EventManagerOptions,
    ObservationManagerOptions,
    SceneOptions,
)
from eden.utils.geom import align_up_and_front
from eden.utils.terrains import ParkourDemoTerrain

# ---------------------------------------------------------------------------
# Terrain grid (student mix)
# ---------------------------------------------------------------------------

# Source student terrain: horizontal scale 0.1 m (teacher 0.08) and corridor courses
# narrowed laterally so the camera keeps the goals in view.
_STUDENT_HORIZONTAL_SCALE = 0.1
_STUDENT_CORRIDOR_Y_RANGE = (-0.1, 0.1)


def _make_student_grid() -> list[list]:
    """10 difficulty rows x 20 columns with the source student proportions.

    Proportions gap/hurdle/step/stones 0.2 each, flat 0.05, demo 0.15 over 20 columns:
    ``[gap x4 | hurdle x4 | flat x1 | step x4 | stones x4 | demo x3]``. Repeated column
    objects are fine — ``randomize=True`` reseeds per cell, so repeats differ (same
    pattern as the teacher grid).
    """
    grid: list[list] = []
    for i in range(_ROWS):
        d = i / (_ROWS - 1)
        c = _course_generators(d, corridor_y_range=_STUDENT_CORRIDOR_Y_RANGE)
        demo = ParkourDemoTerrain.configure(apply_roughness=True, difficulty=d)
        grid.append([c["gap"]] * 4 + [c["hurdle"]] * 4 + [c["flat"]] + [c["step"]] * 4 + [c["stones"]] * 4 + [demo] * 3)
    return grid


# ---------------------------------------------------------------------------
# Depth camera (D435 on the Go2 head)
# ---------------------------------------------------------------------------

# Raw sensor: 87 deg HORIZONTAL FOV at the crop-fixed resolution the depth term owns
# (_DEPTH_RAW_RESOLUTION, imported above — the single source of the 106x60 contract).
# Genesis's `cam_fov` is the VERTICAL FOV with square pixels, so convert via the aspect.
_DEPTH_CAM_HFOV_DEG = 87.0  # the source sensor spec (D435)
_DEPTH_CAM_VFOV_DEG = math.degrees(  # ~56.5 deg
    2 * math.atan(math.tan(math.radians(_DEPTH_CAM_HFOV_DEG / 2)) * _DEPTH_RAW_RESOLUTION[1] / _DEPTH_RAW_RESOLUTION[0])
)

# Mount: (0.33, 0, 0.08) on the `base` link, looking forward pitched 20 deg down (decoded
# numerically from the IsaacLab port's ROS-convention offset quat). Genesis rendering
# cameras use the OpenGL convention (look down -Z, +Y up), so the attach quat is
# R = Ry(pitch_down) @ R_gl2flu, where R_gl2flu maps the OpenGL camera axes onto a
# forward-looking (+X fwd, +Z up, image-right = -Y) camera.
_DEPTH_CAM_PITCH_DOWN_DEG = 20.0
_R_GL_TO_FLU = align_up_and_front(up=(0.0, 1.0, 0.0), front=(0.0, 0.0, -1.0))


def _depth_camera_quat(pitch_down_deg: float) -> tuple[float, float, float, float]:
    """Attach quat (wxyz) for a forward camera pitched ``pitch_down_deg`` below horizontal."""
    return tuple(R_to_quat(euler_to_R(np.array([0.0, pitch_down_deg, 0.0])) @ _R_GL_TO_FLU).tolist())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@TASK_REGISTRY.register()
class ExtremeParkourStudentConfig(ExtremeParkourTeacherConfig):
    """Extreme Parkour student (teacher env + depth camera; see module docstring).

    Rewards, terminations, commands and the terrain-level curriculum are inherited from
    the teacher unchanged.
    """

    env_options: EnvOptions = _make_env_options(num_envs=192)

    scene_options: SceneOptions = _make_terrain_scene(_make_student_grid(), _STUDENT_HORIZONTAL_SCALE)

    # Depth-only rendering: the rasterizer branch is the faster madrona backend and needs no
    # declared lights (depth ignores shading). Typed like the base field (not narrowed): a
    # camera-free build of this task legitimately passes renderer_options=None, and a
    # narrowed required field would reject that at construction and on load.
    renderer_options: RendererOptions | None = BatchRendererOptions(use_rasterizer=True)

    cameras_options: CamerasOptions = CamerasOptions(
        depth=CameraOptions(
            cam_resolution=_DEPTH_RAW_RESOLUTION,
            cam_fov=_DEPTH_CAM_VFOV_DEG,
            attach_entity_name="robot",
            attach_link_name="base",
            attach_offset=(0.33, 0.0, 0.08),
            attach_quat=_depth_camera_quat(_DEPTH_CAM_PITCH_DOWN_DEG),
        ),
    )

    # The teacher's 753-d `policy` group plus the two student groups (order-independent:
    # ParkourDeltaYawOk reads ParkourProprio's delta-yaw cache, ticked before any group computes).
    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=_policy_observation_group(),
        depth_camera=ObservationGroupOptions(
            depth=ParkourDepthImage.configure(camera_name="depth", update_interval=5),
            enable_corruption=False,
        ),
        delta_yaw_ok=ObservationGroupOptions(
            ok=ParkourDeltaYawOk.configure(threshold=0.6),
            enable_corruption=False,
        ),
    )

    # Same controller as the teacher plus the source's one-control-step action delay
    # (a fixed 20 ms = 4 physics steps lag on the applied target; the observation keeps
    # the undelayed action).
    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=en.actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_GO2.dofs_name,
            scale=0.25,
            reference_source=ReferenceSource.DEFAULT,
            modifier=ActionDelay.configure(delay_range_s=(0.02, 0.02)),
        )
    )

    event_options: EventManagerOptions = EventManagerOptions(
        **_make_event_terms(),
        randomize_camera_mount=RandomizeCameraMountPitch.configure(
            camera_name="depth",
            pitch_range_deg=(-5.0, 5.0),
        ),
    )

    # DAgger distillation from a teacher checkpoint (source setup). The actor must be the
    # teacher's architecture — it is rebuilt to load the checkpoint into. One iteration
    # collects 24 x 5 steps: 24 depth-encoder passes at the depth term's 5-step cadence.
    runner_options: RslRlRmaDepthDistillationRunnerOptions = RslRlRmaDepthDistillationRunnerOptions(
        seed=1,
        num_steps_per_env=24 * 5,
        max_iterations=5_000,
        save_interval=100,
        clip_actions=4.8,
        experiment_name="extreme_parkour_student",
        wandb_project="extreme_parkour_student",
        obs_groups={"actor": ["policy"]},
        actor=rma_actor_options(),
        algorithm=RslRlRmaDepthDistillationAlgorithmOptions(
            learning_rate=1.0e-3,
            max_grad_norm=1.0,
            gru_hidden_dim=512,
            # Must equal the depth / proprio terms' update_interval; the runner checks it at build.
            update_interval=5,
            # The yaw head is Tanh-bounded; 1.5x stretches it over the +-1.6 rad heading range.
            yaw_scale=1.5,
            # [start, stop) of ParkourProprio's (delta_yaw, delta_next_yaw) slots.
            yaw_slots=[6, 8],
            num_pretrain_iters=0,
            depth_obs_group="depth_camera",
            yaw_gate_obs_group="delta_yaw_ok",
        ),
    )
