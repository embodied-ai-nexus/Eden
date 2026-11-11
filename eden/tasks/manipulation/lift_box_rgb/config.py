"""Lift box from RGB (``manipulation_lift_box_rgb``): the actor finds the cube in a wrist RGB camera.

Physics, actions, rewards, terminations, events, and curricula are inherited
from :class:`ManipulationLiftBoxConfig` unchanged; the delta is perception:

* A wrist RGB camera (128x72, on the Franka tool flange) rendered with the
  gs-madrona :class:`BatchRendererOptions` batch renderer.
* Asymmetric actor-critic observations: the actor **loses the privileged
  object pose** (``object_position`` / ``object_orientation``) and must find
  the cube in the wrist image; the critic keeps full privileged state. Both
  consume the shared ``camera`` group
  (``obs_groups={"actor": ["policy", "camera"], "critic": ["critic", "camera"]}``).
* CNN policy: a spatial-softmax conv head
  (:mod:`eden.options.learning.rsl_rl.spatial_softmax`) — one conv stack per 2D
  obs group ending in a soft-argmax that emits each channel's expected (x, y)
  image location, so the policy gets explicit object keypoints. This is the head
  that trains vision manipulation from scratch; a plain flatten plateaus at
  reach/hover (verified on this task).

Madrona rendering gotchas this config codifies (either one leaves the actor
effectively blind, with near-black or colorless frames):

* Madrona renders **ambient-only until lights are added**, and bakes the light
  count at build time — lights are declared on :class:`BatchRendererOptions`
  and applied before ``scene.build()``.
* ``BoxOptions`` defaults to a gray surface: a gray cube on a gray table is
  invisible to the camera. The cube is colored red and the table darkened,
  scoped to this task (the blind ``manipulation_lift_box`` is untouched).
"""

from __future__ import annotations

from genesis.options.renderers import RendererOptions

from eden import observations
from eden.options import BatchRendererOptions, ObservationGroupOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.learning.rsl_rl import (
    RslRlActorCNNOptions,
    RslRlCriticCNNOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks.manipulation.lift_box.config import (
    _droid_lift_scene,
    _lift_env_options,
    _Observations,
    ManipulationLiftBoxConfig,
)
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import ObsTerm
from eden.utils.configs import EnvOptions, ObservationManagerOptions, SceneOptions

# The actor sees no privileged object state — the wrist camera is its only
# source of cube pose. Everything else (proprio, goal, grasp point from FK,
# last action) is knowable on a real robot.
_ACTOR_OBS = {k: v for k, v in _Observations.items() if k not in ("object_position", "object_orientation")}

# Wrist camera image, CNN-ready: RGBA -> RGB, uint8 -> float [0, 1], HWC -> CHW.
_CAMERA_OBS = ObsTerm.configure(
    func=observations.rgb_image,
    params={"camera_name": "wrist", "normalize": True, "channels_first": True},
)

# Spatial-softmax CNN head: the conv stack ends in a
# soft-argmax that returns each channel's expected (x, y) image location, giving
# the policy explicit 2D keypoints of what it sees. This is the head that makes
# vision manipulation trainable from scratch — a plain flatten forces the MLP to
# recover object position from a raw feature vector and plateaus at reach/hover.
_SS_CNN_MODEL = "eden.options.learning.rsl_rl.spatial_softmax:SpatialSoftmaxCNNModel"
_CNN_CFG = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
}


def _rgb_lift_scene() -> SceneOptions:
    """The Droid lift scene, recolored so the wrist camera can see the cube.

    Scoped scene tweaks for camera visibility (see the module docstring): a red cube on a
    darkened table. A factory over :func:`_droid_lift_scene` — the house pattern — so the
    recolor stays a *field default* rather than an ``__init__`` mutation, which pydantic
    re-applies on every reconstruction and which therefore silently reverts a ``--config``
    override of either color.
    """
    scene = _droid_lift_scene()
    scene.cube.color = (0.85, 0.15, 0.15, 1.0)
    scene.table.color = (0.25, 0.25, 0.28, 1.0)
    return scene


# Two shadow-free directional lights over the workspace, so madrona doesn't render
# ambient-only. Madrona bakes the light count at build time, so these are declared here and
# applied before ``scene.build()``.
_WORKSPACE_LIGHTS = BatchRendererOptions(
    lights=[
        {
            "pos": (0.3, -0.6, 1.5),
            "dir": (0.0, 0.4, -1.0),
            "directional": True,
            "castshadow": False,
            "intensity": 3.0,
        },
        {
            "pos": (0.3, 0.6, 1.5),
            "dir": (0.0, -0.4, -1.0),
            "directional": True,
            "castshadow": False,
            "intensity": 3.0,
        },
    ]
)


def _rgb_runner_options(experiment_name: str = "lift_box_rgb") -> RslRlOnPolicyRunnerOptions:
    """The spatial-softmax CNN PPO runner, with a per-task ``experiment_name``.

    A factory so a subclass gets its own log/checkpoint directory: ``experiment_name`` is the
    only field a variant needs to change, and inheriting it verbatim would silently write two
    different tasks' runs into the same ``logs/<experiment_name>`` tree.
    """
    return RslRlOnPolicyRunnerOptions(
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,  # more exploration than the blind task's 0.005
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        actor=RslRlActorCNNOptions(
            class_name=_SS_CNN_MODEL,
            activation="elu",
            hidden_dims=[256, 256, 128],
            cnn_cfg=_CNN_CFG,
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticCNNOptions(
            class_name=_SS_CNN_MODEL,
            activation="elu",
            hidden_dims=[256, 256, 128],
            cnn_cfg=_CNN_CFG,
        ),
        obs_groups={"actor": ["policy", "camera"], "critic": ["critic", "camera"]},
        max_iterations=3000,
        save_interval=50,
        experiment_name=experiment_name,
        run_name="v1",
    )


# ``modifiers=()``: opt out of the LiftBoxManipulatorMod a lift-box subclass would otherwise inherit.
# The wrist camera below is calibrated to the DROID flange, and a non-default arm rewrites the
# experiment name to the blind task's ``lift_box_<arm>`` -- two runs of different tasks, one log tree.
@TASK_REGISTRY.register(modifiers=())
class ManipulationLiftBoxRGBConfig(ManipulationLiftBoxConfig):
    # The wrist-camera rollout storage (per-env RGB over num_steps_per_env) OOMs a 48 GB
    # GPU at the blind task's 4096 envs; 1024 fits and is what the from-scratch runs used.
    env_options: EnvOptions = _lift_env_options(num_envs=1024)

    scene_options: SceneOptions = _rgb_lift_scene()

    # Optional, like the base field: a camera-free / headless build of this task legitimately
    # passes renderer_options=None, and narrowing to a required field would reject that at
    # construction and on load while a plain attribute assignment still went through.
    renderer_options: RendererOptions | None = _WORKSPACE_LIGHTS

    cameras_options: CamerasOptions = CamerasOptions(
        # Wrist camera on the tool flange ("attachment"), calibrated to the real
        # DROID rig. Extrinsic from Stanford-TML/cloak's silhouette calibration
        # (mean cam_to_gripper over 77k DROID episodes): translation
        # [-0.070, +0.026, +0.013] m and XYZ-euler [-18.2, -1.1, -89.0] deg, in
        # the FR3 flange frame with an OpenCV camera. Ported here to eden's
        # "attachment" frame (= flange rotated 180 deg about z, per the axis match)
        # with the OpenCV->Genesis(OpenGL) camera flip diag(1,-1,-1): the camera
        # looks down the approach axis (+z toward the gripper) with the ~18 deg
        # forward tilt the real rig has.
        wrist=CameraOptions(
            cam_resolution=(128, 72),
            cam_fov=80,
            attach_entity_name="robot",
            attach_link_name="attachment",
            attach_offset=(0.070, -0.026, 0.013),
            attach_quat=(0.103966, 0.692961, 0.703320, 0.119734),
        ),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_ACTOR_OBS, enable_corruption=True),
        critic=ObservationGroupOptions(**_Observations, enable_corruption=False),
        # Single image term, so concatenation is the identity and the group
        # keeps its (N, C, H, W) shape — which is how ActorCriticCNN detects a
        # 2D obs group and routes it through a conv stack.
        camera=ObservationGroupOptions(wrist_rgb=_CAMERA_OBS, enable_corruption=False),
    )

    runner_options: RslRlOnPolicyRunnerOptions = _rgb_runner_options()
