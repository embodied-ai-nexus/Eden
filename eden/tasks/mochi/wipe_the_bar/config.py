"""``mochi_wipe_the_bar``: a YAM picks a foam sponge off the mat and wipes a marker stripe off a grab bar.

Eden's version of RoboSimStudio's ``wipe_the_bar`` demo (``examples/08_demos/wipe_the_bar``, the
``surface_prep`` task), at its layout measured off the clip: a 25 mm chrome tube with a 340 mm
straight run on two 32 mm elbows, its legs 50 mm into 76 x 80 x 70 mm blocks on a 10 mm mat; the stripe is the
middle 200 mm of its crown, 9 degrees either side (243 cells of 2.5 mm); the sponge is 50 x 75 x 30 mm of
RoboSimStudio's ``FIRM_FOAM`` (E 80 kPa, nu 0.3, 110 kg/m^3), lying on the mat 70 mm past the near block's outer
face. The stripe wears by Preston's law with RoboSimStudio's constants
(:class:`~eden.tasks.mochi.custom_terms.wipe.BarStripeWipe`), and the episode succeeds at 90 % of it clean, the
bar RoboSimStudio's test holds its takes to (its routine strokes on to 96 %).

Two YAMs stand at RoboSimStudio's mounts. The right one does the job: it takes the sponge off the mat with its
jaws across the 50 mm side, carries it over the near block and strokes the stripe. Its jaws squeeze at most 8 N
a finger, RoboSimStudio's cap, and grip the foam by contact; once that grip has settled, RoboSimStudio's latch
pins the foam under the pads to the hand until the jaws open
(:class:`~eden.tasks.mochi.custom_terms.grasp.SettledGraspLatch`). Friction alone lifts and carries the sponge,
but a stroke drags it out of the jaws (RoboSimStudio saw the same, and latched for it). The left YAM stands at
its ready posture: RoboSimStudio's left hand rests on the welded bar's far elbow without clamping it, which carries no
load, so it is left out of the job.

The right arm acts through differential IK on its tool centre's position (its tilt held: the hand points down,
jaws across the bench's x), and a binary gripper.

The stripe is the metric's, not geometry in the scene: ``eden inference --rec`` records
:meth:`MochiWipeTheBarConfig.recording_camera`, which paints it (``PlanePaintOverlay``). The training config
declares no camera.
"""

from __future__ import annotations

import math

import genesis as gs
import numpy as np

from eden import observations, rewards, terminations
from eden.engine.solvers.mochi.options import MochiOptions
from eden.entities.camera_overlays import PlanePaintOverlay
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options import CameraOptions, ObservationGroupOptions
from eden.options.entities import BoxOptions, EntityOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.materials import MochiElasticMaterialOptions, MochiRigidMaterialOptions
from eden.tasks.mochi import custom_terms
from eden.tasks.mochi.common import (
    CONTACT_RAMP,
    CONTACT_STIFFNESS,
    YAM_ARM_DOFS,
    YAM_FINGER_DOFS,
    YAM_FINGERS_CLOSED,
    YAM_FINGERS_OPEN,
    YAM_PLATE_DEPTH,
    YAM_TCP_OFFSET,
    bench,
    yam,
)
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import DoneTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    EdenRLConfig,
    EnvOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)
from eden.utils.procedural_meshes import box_tet_lattice_file, swept_tube_file

# ---- the layout: RoboSimStudio's ``bar.Layout``, measured off the clip against the mat's 25 mm grid ------------
# The ground plane is the 10 mm mat's top; RoboSimStudio's numbers are from the bench under it. Its arms stand on
# the bench beside the mat. Here the plane is the only surface, the arms' as well as the mat's, and their base
# plates stand on it, so the mounts are 20 mm higher than the demo's.
TUBE_RADIUS = 0.0125
STRAIGHT_HALF = 0.170  # half the straight run between the elbows
BEND_RADIUS = 0.032  # centreline radius of the elbows
LEG_DROP = 0.050  # bar axis to the blocks' top face
LEG_BURIED = 0.015  # how far each leg runs into its block
BLOCK_HALF = (0.038, 0.040, 0.035)  # 76 deep, 80 along the bar, 70 tall
BLOCK_INSET = (0.018, 0.020)  # block centre from its leg: +x, and outward along the bar
BAR_X, BAR_Y = -0.05, 0.05
BAR_Z = 2.0 * BLOCK_HALF[2] + LEG_DROP  # the axis height over the mat
SPONGE_SIZE = (0.050, 0.075, 0.030)  # long axis along the bar
SPONGE_GAP = 0.070  # clear mat between the near block's outer face and the sponge
SPONGE_FORWARD = 0.035  # the sponge lies this far past the bar's line, away from the arms
STRIPE_HALF = 0.100  # the marker line covers the middle 200 mm of the run
STRIPE_HALF_ANGLE_DEG = 9.0
#: The colour the recording camera paints the stripe's ink in: the near-black navy the README's wipe clip drew it in,
#: not a measured marker colour. Rendering only; grading never reads it.
STRIPE_COLOR = (0.02, 0.03, 0.10)

_LEG_U = STRAIGHT_HALF + BEND_RADIUS  # along the bar from its middle to a leg's axis
_NEAR_BLOCK = (BAR_X + BLOCK_INSET[0], BAR_Y - (_LEG_U + BLOCK_INSET[1]), BLOCK_HALF[2])
_FAR_BLOCK = (BAR_X + BLOCK_INSET[0], BAR_Y + (_LEG_U + BLOCK_INSET[1]), BLOCK_HALF[2])
SPONGE_AT = (
    BAR_X + SPONGE_FORWARD,
    _NEAR_BLOCK[1] - BLOCK_HALF[1] - SPONGE_GAP - 0.5 * SPONGE_SIZE[1],
    0.5 * SPONGE_SIZE[2],
)
# RoboSimStudio meshes the sponge 9 x 13 x 6 cells (980 nodes); 6 x 9 x 4 (350 nodes, ~8 mm apart) runs twice as
# fast under Mochi and the stripe's reach follows the spacing (custom_terms.wipe).
SPONGE_CELLS = (6, 9, 4)
_SPONGE_NODES = tuple(c + 1 for c in SPONGE_CELLS)
# The sponge's eight corners (lattice index (i * ny + j) * nz + k).
SPONGE_CORNERS = [
    (i * _SPONGE_NODES[1] + j) * _SPONGE_NODES[2] + k
    for i in (0, _SPONGE_NODES[0] - 1)
    for j in (0, _SPONGE_NODES[1] - 1)
    for k in (0, _SPONGE_NODES[2] - 1)
]
_STRIPE_REACH = 0.5 * math.hypot(*(s / c for s, c in zip(SPONGE_SIZE[:2], SPONGE_CELLS[:2])))


def _bar_path() -> np.ndarray:
    """The tube's centreline: up the near leg, round the near elbow, along the run, round the far one, down."""
    elbow = np.linspace(0.0, 0.5 * math.pi, 7)
    leg_bottom = BAR_Z - LEG_DROP - LEG_BURIED
    near = [(BAR_X, BAR_Y - _LEG_U, leg_bottom)]
    near += [
        (BAR_X, BAR_Y - STRAIGHT_HALF - BEND_RADIUS * math.cos(a), BAR_Z - BEND_RADIUS + BEND_RADIUS * math.sin(a))
        for a in elbow
    ]
    far = [(x, 2.0 * BAR_Y - y, z) for x, y, z in reversed(near)]
    return np.asarray(near + far)


_EPISODE_S = 15.0
CONTROL_DT = 1.0 / 30.0
_GRIP = {"grip_metric_name": "grip", "grip_threshold": 2.0}
_RIGHT_TCP = {"robot_name": "right", "link_name": "link_6", "link_offset": YAM_TCP_OFFSET}
# RoboSimStudio's jaws: at most 8 N a finger (a 25 mm finger travel left on the 50 mm foam at kp 320 N/m), damped
# at 100 N s/m.
_FINGER_KP, _FINGER_KD = 320.0, 100.0
# RoboSimStudio stiffens the working arm's servos for the press: at the robot file's gain the hand lags ~10 mm.
_ARM_KP, _ARM_KD = 1600.0, 70.0

_Observations = {
    "arm_dofs_pos": observations.DofsPos.configure(entity_name="right"),
    "arm_dofs_vel": observations.DofsVel.configure(entity_name="right"),
    "tool_centre": custom_terms.ToolCentre.configure(**_RIGHT_TCP),
    "sponge_corners": custom_terms.SoftBodyPoints.configure(
        entity_name="sponge", vertices=SPONGE_CORNERS, **_RIGHT_TCP
    ),
    "stripe_ink": custom_terms.StripeInk.configure(metric_name="cleaned"),
    "press": observations.MetricObservation.configure(metric_name="press", scale=0.25),
    "grip": observations.MetricObservation.configure(metric_name="grip", scale=0.25),
    "last_action": ObsTerm.configure(func=observations.last_action),
}


@TASK_REGISTRY.register()
class MochiWipeTheBarConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=256,
        episode_length_s=_EPISODE_S,
        sim_dt=CONTROL_DT,
        sim_substeps=1,
        decimation=1,
        mochi_options=MochiOptions(n_newton_iterations=4),
    )

    scene_options: SceneOptions = SceneOptions(
        # The mat: an EVA puzzle mat over the whole bench.
        plane=bench(friction=0.9),
        right=yam(
            "right",
            ready="jaws_x",
            finger_kp=_FINGER_KP,
            finger_kd=_FINGER_KD,
            contact_ramp=CONTACT_RAMP,
            arm_kp=_ARM_KP,
            arm_kd=_ARM_KD,
            base_height=YAM_PLATE_DEPTH,
        ),
        left=yam(
            "left",
            ready="jaws_x",
            finger_kp=_FINGER_KP,
            finger_kd=_FINGER_KD,
            contact_ramp=CONTACT_RAMP,
            base_height=YAM_PLATE_DEPTH,
        ),
        # Mochi combines two frictions by their geometric mean, and RoboSimStudio's SuperDex takes the foam's own
        # 0.9 against every collider: so 0.9 throughout.
        bar=EntityOptions(
            file=swept_tube_file("grab_bar", _bar_path(), TUBE_RADIUS),
            is_fixed_base=True,
            convexify=False,
            material=MochiRigidMaterialOptions(
                penalty_coefficient=CONTACT_STIFFNESS, friction=0.9, collider_type="sdf", sdf_cell_size=2e-3
            ),
            surface=gs.surfaces.Default(color=(0.8, 0.8, 0.8)),
        ),
        block_near=BoxOptions(
            size=tuple(2 * h for h in BLOCK_HALF),
            fixed=True,
            default_root_pos=_NEAR_BLOCK,
            material=MochiRigidMaterialOptions(penalty_coefficient=CONTACT_STIFFNESS, friction=0.9),
            surface=gs.surfaces.Default(color=(0.05, 0.05, 0.06)),
        ),
        block_far=BoxOptions(
            size=tuple(2 * h for h in BLOCK_HALF),
            fixed=True,
            default_root_pos=_FAR_BLOCK,
            material=MochiRigidMaterialOptions(penalty_coefficient=CONTACT_STIFFNESS, friction=0.9),
            surface=gs.surfaces.Default(color=(0.05, 0.05, 0.06)),
        ),
        sponge=EntityOptions(
            file=box_tet_lattice_file(SPONGE_SIZE, SPONGE_CELLS),
            default_root_pos=SPONGE_AT,
            material=MochiElasticMaterialOptions(
                E=8e4,
                nu=0.3,
                rho=110.0,
                stiffness_damping=8e-3,
                friction=0.9,
                # RoboSimStudio's FOAM: FIRM_FOAM with as much static friction as the regularised law gives (below
                # this slip, friction grows with the slip).
                friction_falloff_vel=1e-4,
                penalty_coefficient=CONTACT_STIFFNESS,
                # The pads press into the foam with the foam as the collider: its ramp, not the arm's, stops them.
                penalty_smoothing_half_distance=CONTACT_RAMP,
            ),
            # A kitchen sponge's yellow.
            surface=gs.surfaces.Default(color=(0.98, 0.82, 0.18)),
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        cleaned=custom_terms.BarStripeWipe.configure(
            entity_name="sponge",
            bar_axis=(BAR_X, BAR_Y, BAR_Z),
            radius=TUBE_RADIUS,
            half_length=STRIPE_HALF,
            half_angle_deg=STRIPE_HALF_ANGLE_DEG,
            reach=_STRIPE_REACH,
            # RoboSimStudio's test passes a take at 0.9 clean (its routine strokes on to 0.96, and stops there).
            success_threshold=0.9,
        ),
        press=custom_terms.BarPress.configure(metric_name="cleaned", contributes_to_success=False),
        grip=custom_terms.JawGrip.configure(entity_name="right", contributes_to_success=False),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Observations, enable_corruption=True),
        critic=ObservationGroupOptions(**_Observations, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        hand=DifferentialIKController.configure(
            entity_name="right",
            ee_link_name="link_6",
            dofs_name=YAM_ARM_DOFS,
            # 10 mm a step at 30 Hz: up to 0.3 m/s, against RoboSimStudio's 0.06 m/s strokes and 0.2 m/s moves. At
            # 0.45 m/s the policy swept onto the stripe at speed and touched down halfway along it.
            position_scale=0.01,
            # Hold the hand pointing down, jaws across the bench's x: the orientation half of the action is inert.
            orientation_scale=0.0,
            # Light damping: at 0.1 the hand lagged 5 mm under a 4 N press and drifted 1 cm off the bar's axis
            # in a stroke, leaving a row of the stripe dirty.
            damping=0.02,
            max_dq=0.1,
        ),
        # RoboSimStudio's latch: once the jaws' grip on the foam has settled, the foam under the pads rides on the
        # hand until they open (custom_terms.grasp).
        gripper=custom_terms.SettledGraspLatch.configure(
            entity_name="right",
            dofs_name=YAM_FINGER_DOFS,
            open_action=YAM_FINGERS_OPEN,
            close_action=YAM_FINGERS_CLOSED,
            soft_entity_name="sponge",
            free_width=SPONGE_SIZE[0],
            settle_steps=round(0.2 / CONTROL_DT),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        # Coarse and fine reaching for the sponge (it starts ~0.2 m from the tool centre); both pay in full while
        # the jaws grip, so taking hold never costs reward.
        reach_sponge=custom_terms.ReachSoftBody.configure(
            entity_name="sponge", **_RIGHT_TCP, **_GRIP, std=0.2, weight=0.25
        ),
        reach_sponge_fine=custom_terms.ReachSoftBody.configure(
            entity_name="sponge", **_RIGHT_TCP, **_GRIP, std=0.02, weight=0.25
        ),
        # Lifted in the jaws.
        held=custom_terms.GripHeld.configure(entity_name="sponge", **_GRIP, weight=0.25),
        # The sponge's wiping face to the stroke's next dirty cell (the first from the stripe's near end), only while
        # gripped: carries it over the block onto the stripe, and on along it as the stripe comes clean.
        reach_ink=custom_terms.ReachDirtyCell.configure(entity_name="sponge", **_GRIP, std=0.2, weight=0.2),
        reach_ink_mid=custom_terms.ReachDirtyCell.configure(entity_name="sponge", **_GRIP, std=0.05, weight=0.2),
        reach_ink_fine=custom_terms.ReachDirtyCell.configure(entity_name="sponge", **_GRIP, std=0.02, weight=0.2),
        # The whole stripe's ink is worth 1000 over an episode, against at most 0.6 a step for hovering by the ink:
        # at 200 against 1.0 the policy wiped the far third of the stripe and then hovered past its end.
        wipe_progress=custom_terms.WipeProgress.configure(metric_name="cleaned", weight=1000.0),
        # The bonus for crossing the success threshold, which the dense terms alone pay too little to reach.
        success=rewards.TaskSuccess.configure(weight=200.0),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        arm_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "right", "max_vel": 1.0}
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        wiped_clean=terminations.OnMetricSuccess.configure(metric_name="cleaned"),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
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
            hidden_dims=[256, 128, 64],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[256, 128, 64]),
        num_steps_per_env=24,
        max_iterations=3000,
        save_interval=100,
        experiment_name="mochi_wipe_the_bar",
        run_name="",
    )

    def recording_camera(self) -> CameraOptions:
        """In front of the bench, high on the right arm's side: the sponge, the bar and the stripe's ink painted on.

        The README wipe clip's view. The stripe is the metric's (``BarStripeWipe``), not geometry, so the overlay is
        what shows it wearing off; its 1 mm plane tolerance keeps the sponge's underside, pressed on the crown, clear.
        That tolerance holds the tube's 0.6 mm sag over the cells plus the rasterizer's depth error at this distance
        (checked); it is not checked against the batch renderer's depth. If the stripe's edges come and go there,
        widen ``max_gap`` toward 2 mm, past which the sponge's underside starts to take ink.
        """
        return CameraOptions(
            cam_pos=(0.45, -0.45, 0.70),
            cam_lookat=(-0.10, -0.05, 0.06),
            cam_fov=40.0,
            cam_resolution=(960, 540),
            overlays=[PlanePaintOverlay.configure(metric_name="cleaned", color=STRIPE_COLOR, max_gap=0.001)],
        )
