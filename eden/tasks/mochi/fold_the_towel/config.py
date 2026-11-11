"""``mochi_fold_the_towel``: two YAMs fold a terry towel in half, one pinching its far edge, the other pressing.

Eden's version of RoboSimStudio's ``fold_the_towel`` demo (``examples/08_demos/fold_the_towel``, the
``cloth_fold`` suite's ``towel_half`` task) at its numbers: a 30 cm square of terry, 500 g/m^2, meshed 30 x 30
cells, lying flat at ``(-0.05, +0.06)`` on the bench, the left arm's reach band. The crease is the towel's own
centreline across y; the far half (+y, toward the left arm) is carried over onto the near half.

The fold is scored with the suite's measures (:mod:`eden.tasks.mochi.custom_terms.fold`) and its goal: the span
across the crease at most 0.6 of the flat towel's, a fold score of at least 0.85 (SoftGym's ClothFold with the
drag term, forgiving one layer's 7.5 mm), the towel still (95th-percentile speed under 0.02 m/s), and both hands
clear of it (no particle within 2 cm of a tool centre).

Both arms act, as in the demo: the left takes the far edge between its jaws (closing along y, across the edge)
and carries it over; the right, its jaws shut, presses the near half to the bench so the flap does not drag it
along (without the press the demo's fold scored 0.18-0.45). Each arm moves its tool centre by differential IK
with its tilt held, and the left's jaws open and close. The jaws take the towel with the benchmark's deformable
grasp (:class:`~eden.tasks.mochi.custom_terms.grasp.FeatureGraspLatch`, RoboSimStudio's ``attach``): closing
seizes the edge strip within 3 cm of the tool centre. The demo pinches by friction alone, gathering the two faces
of its VBD slab; the jaws' tips press a Mochi shell, one surface, into the bench instead.

RoboSimStudio's cloth is a VBD slab (two faces 5 mm apart, its core softened to 3 kPa so it drapes); here it is a
Mochi shell with a 2.5 mm collider either side (a 5 mm thick layer to anything that touches it) and a bending
length near the softened terry's (~3 cm).
"""

from __future__ import annotations

import genesis as gs

from eden import observations, rewards, terminations
from eden.engine.solvers.mochi.options import MochiOptions
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options import ObservationGroupOptions
from eden.options.entities import EntityOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.materials import MochiShellMaterialOptions
from eden.tasks.mochi import custom_terms
from eden.tasks.mochi.common import (
    CONTACT_STIFFNESS,
    YAM_ARM_DOFS,
    YAM_FINGER_DOFS,
    YAM_FINGERS_CLOSED,
    YAM_FINGERS_OPEN,
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
from eden.utils.procedural_meshes import grid_sheet_file

# ---- the towel: the cloth_fold suite's, in the bench's work-surface frame ---------------------------------------
SIDE = 0.30
CELLS = 30  # a 10 mm cell
CENTER = (-0.05, 0.06)
# Each side of the shell collides 2.5 mm out: a 5 mm thick layer, the demo's slab.
COLLIDER_RADIUS = 0.0025
# The suite forgives one layer's contact thickness in the fold score: 7.5 mm for its terry.
LAYER_THICKNESS = 0.0075
# RoboSimStudio's arena YAMs stand on a gantry 22 mm over the bench.
BASE_HEIGHT = 0.022
# The demo's marks: the left hand pinches the far edge 30 mm toward the robot from the towel's middle, and the
# right presses the near half, here at (0.08, -0.04), 80 mm farther out than the demo's (0, -0.04). The arms
# collide here, and at the demo's press the folding arm stopped against the pressing one 45 mm short of its place.
PINCH_X = -0.08
PRESS = (0.08, -0.04)

_NODES = CELLS + 1


def _vertex(x: float, y: float) -> int:
    """The towel vertex nearest a bench point (grid_sheet lays its vertices out x-major)."""
    i = round((x - CENTER[0] + 0.5 * SIDE) / SIDE * CELLS)
    j = round((y - CENTER[1] + 0.5 * SIDE) / SIDE * CELLS)
    return i * _NODES + j


# The far edge where the demo pinches it, the vertex beside it either way along the edge, and one row in.
EDGE_GRASP = [_vertex(PINCH_X + dx, CENTER[1] + 0.5 * SIDE - dy) for dx in (-0.01, 0.0, 0.01) for dy in (0.0, 0.01)]
PRESS_POINT = [_vertex(*PRESS)]
# The suite's "grasp" feature: the far edge's 25 mm strip (selector_extreme(1, band=0.025, high=True)), what the
# jaws can seize.
GRASP_STRIP = [i * _NODES + j for i in range(_NODES) for j in range(_NODES) if j * SIDE / CELLS >= SIDE - 0.025 - 1e-9]
# A 5 x 5 lattice of the towel's vertices, corners included: its shape for the policy.
_LATTICE = (0, 8, 15, 22, CELLS)
KEYPOINTS = [i * _NODES + j for i in _LATTICE for j in _LATTICE]

_EPISODE_S = 12.0
CONTROL_DT = 1.0 / 30.0
_FOLD = {"entity_name": "towel", "fold_axis": 1}
_LEFT_TCP = {"robot_name": "left", "link_name": "link_6", "link_offset": YAM_TCP_OFFSET}
_RIGHT_TCP = {"robot_name": "right", "link_name": "link_6", "link_offset": YAM_TCP_OFFSET}
_HOLDS = {"grip_metric_name": "holds", "grip_threshold": 0.5}
# The i2RT jaws' own servo (kp 600 N/m): squeezing to 0.95 of their travel puts the pads 3.5 mm apart in the demo.
_FINGER_KP, _FINGER_KD = 600.0, 20.0


def _hand(robot: str) -> DifferentialIKController:
    return DifferentialIKController.configure(
        entity_name=robot,
        ee_link_name="link_6",
        dofs_name=YAM_ARM_DOFS,
        position_scale=0.015,
        orientation_scale=0.0,
        damping=0.02,
        max_dq=0.1,
    )


_Observations = {
    "left_dofs_pos": observations.DofsPos.configure(entity_name="left"),
    "left_dofs_vel": observations.DofsVel.configure(entity_name="left"),
    "right_dofs_pos": observations.DofsPos.configure(entity_name="right"),
    "right_dofs_vel": observations.DofsVel.configure(entity_name="right"),
    "left_tool": custom_terms.ToolCentre.configure(**_LEFT_TCP),
    "right_tool": custom_terms.ToolCentre.configure(**_RIGHT_TCP),
    "towel": custom_terms.SoftBodyPoints.configure(entity_name="towel", vertices=KEYPOINTS, **_LEFT_TCP),
    "fold_score": observations.MetricObservation.configure(metric_name="fold_score"),
    "fold_extent": observations.MetricObservation.configure(metric_name="fold_extent"),
    "holds": observations.MetricObservation.configure(metric_name="holds"),
    "last_action": ObsTerm.configure(func=observations.last_action),
}


@TASK_REGISTRY.register()
class MochiFoldTheTowelConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=256,
        episode_length_s=_EPISODE_S,
        sim_dt=CONTROL_DT,
        sim_substeps=1,
        decimation=1,
        # At most 100 conjugate-gradient iterations a Newton step: by default up to 500, and a policy swinging the
        # held towel about spent 1000-1100 a control step (10x the flat towel's), 60 s an iteration.
        mochi_options=MochiOptions(n_newton_iterations=4, n_pcg_iterations=100),
    )

    scene_options: SceneOptions = SceneOptions(
        # The bench pushes back at the shell's collider radius: against an analytic plane Mochi takes a shell's
        # vertices as points, and the towel would lie on its mid-plane.
        plane=bench(friction=0.8, penalty_threshold=COLLIDER_RADIUS),
        left=yam("left", ready="jaws_y", finger_kp=_FINGER_KP, finger_kd=_FINGER_KD, base_height=BASE_HEIGHT),
        # The pressing hand, shut so its closed jaws press flat.
        right=yam(
            "right",
            ready="jaws_y",
            finger_kp=_FINGER_KP,
            finger_kd=_FINGER_KD,
            base_height=BASE_HEIGHT,
            fingers_open=False,
        ),
        towel=EntityOptions(
            file=grid_sheet_file((SIDE, SIDE), (CELLS, CELLS)),
            # Flat, 1 mm over the bench beyond its collider: the demo settles it from 1 mm up.
            default_root_pos=(*CENTER, COLLIDER_RADIUS + 1e-3),
            material=MochiShellMaterialOptions(
                E=2e5,
                nu=0.3,
                rho=250.0,  # 500 g/m^2 over the 2 mm membrane
                thickness=2e-3,
                stiffness_damping=1e-3,
                friction=0.8,
                penalty_coefficient=CONTACT_STIFFNESS,
                collider_radius=COLLIDER_RADIUS,
                # A fold must not pass through itself.
                self_contact=True,
            ),
            surface=gs.surfaces.Default(color=(0.88, 0.85, 0.77)),
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        fold_score=custom_terms.ClothFoldScore.configure(
            **_FOLD, layer_thickness=LAYER_THICKNESS, success_threshold=0.85
        ),
        fold_extent=custom_terms.ClothFoldExtent.configure(**_FOLD, success_threshold=0.6),
        settled=custom_terms.SoftBodySettled.configure(entity_name="towel", success_threshold=0.02),
        hands_clear=custom_terms.HandsClear.configure(
            entity_name="towel", robots_name=["left", "right"], link_offset=YAM_TCP_OFFSET, success_threshold=0.02
        ),
        holds=custom_terms.LatchHolds.configure(action_name="left_gripper", contributes_to_success=False),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Observations, enable_corruption=True),
        critic=ObservationGroupOptions(**_Observations, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        left_hand=_hand("left"),
        right_hand=_hand("right"),
        # RoboSimStudio's deformable grasp: closing seizes the edge strip within 3 cm of the tool centre.
        left_gripper=custom_terms.FeatureGraspLatch.configure(
            entity_name="left",
            dofs_name=YAM_FINGER_DOFS,
            open_action=YAM_FINGERS_OPEN,
            close_action=YAM_FINGERS_CLOSED,
            soft_entity_name="towel",
            tcp_offset=YAM_TCP_OFFSET,
            feature_vertices=GRASP_STRIP,
            radius=0.03,
            max_vertices=128,
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        # Coarse and fine reaching for the far edge where the demo pinches it; both pay in full while the jaws grip.
        reach_edge=custom_terms.ReachSoftBody.configure(
            entity_name="towel", vertices=EDGE_GRASP, **_LEFT_TCP, **_HOLDS, std=0.2, weight=0.25
        ),
        reach_edge_fine=custom_terms.ReachSoftBody.configure(
            entity_name="towel", vertices=EDGE_GRASP, **_LEFT_TCP, **_HOLDS, std=0.02, weight=0.25
        ),
        # The right hand onto the press mark (80 mm farther out than the demo's: see PRESS).
        press=custom_terms.ReachSoftBody.configure(
            entity_name="towel", vertices=PRESS_POINT, **_RIGHT_TCP, grip_metric_name="", std=0.05, weight=0.1
        ),
        # Dense: the score rises as the far half closes onto the near one, and falls with any drag.
        fold_score=rewards.MetricReward.configure(metric_name="fold_score", weight=1.0),
        # Staged on the way to the goal: folded; folded and let go; folded and graded by how far both hands are from
        # it, up to the goal's 2 cm. Without the last two the policy folded the towel (score 0.79, extent 0.55) and
        # held on to it for 500 iterations: the goal wants the latch open, both hands away and the towel still.
        folded=custom_terms.MetricsHold.configure(metrics_name=["fold_score", "fold_extent"], weight=2.0),
        folded_released=custom_terms.MetricsHold.configure(
            metrics_name=["fold_score", "fold_extent"], zero_metrics_name=["holds"], weight=4.0
        ),
        folded_clear=custom_terms.MetricsHold.configure(
            metrics_name=["fold_score", "fold_extent"],
            scale_metric_name="hands_clear",
            scale_target=0.02,
            weight=8.0,
        ),
        # Every step the whole goal holds (let go and at rest included).
        success=rewards.TaskSuccess.configure(weight=50.0),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        left_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "left", "max_vel": 1.0}
        ),
        right_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "right", "max_vel": 1.0}
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        # No termination on success: the goal is a state to hold (a towel lying folded, a cable routed and let go),
        # and the success bonus pays every step it holds. Ended on success, the episode forfeited the dense terms'
        # remaining pay, worth more than a one-off bonus, and the policy learned to keep hold of the towel.
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
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
            hidden_dims=[256, 128, 64],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[256, 128, 64]),
        num_steps_per_env=24,
        max_iterations=3000,
        save_interval=100,
        experiment_name="mochi_fold_the_towel",
        run_name="",
    )
