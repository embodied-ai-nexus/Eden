"""``mochi_route_the_cable``: two YAMs route a cable from its anchor, past two spools and into a port.

Eden's version of RoboSimStudio's ``route_the_cable`` demo (``examples/08_demos/route_the_cable``, VABAR's
``cable_route/port_easy``) at its board's numbers: a 600 mm cable 7 mm across, welded under an anchor bracket at
its first end and lying straight along +y; two ManipulationNet F1 spools 240 mm apart (a 4.75 mm shaft between
12.5 mm flanges, 23.5 mm tall), which the cable must pass on the far side and on the robot's side in turn; and a
port 61 mm out from the second spool and 100 mm past it, a 10 mm channel between two 12.5 mm walls with a stop
at the back, facing the run.

The goal is the board's ``routed`` (:mod:`eden.tasks.mochi.custom_terms.route`): at least two of the cable's
fifty segments in each spool's seat, three in the port's box with the free end among them, both hands more than
50 mm from the cable, and the cable at rest (no node faster than 50 mm/s).

Both arms act, as in the demo, where the right hand makes the three placements and the left plugs the end: each
moves its tool centre by differential IK and turns it about the vertical (the jaws close across the cable's
tangent, and the end has to be lined up with the channel), its tilt held pointing down, and opens and closes its
jaws. The jaws hold the cable by friction, as the demo's do; their servo is the demo's 6000 N/m (at the i2RT
fingers' own 600 N/m a dragged cable pried them open).

RoboSimStudio's cable is a capsule chain with a bend stiffness per joint; here it is a Mochi discrete elastic rod
of 50 segments with the same bending rigidity (``EI`` = the joint stiffness times the segment length) and an axial
stiffness that keeps it all but inextensible, as rigid capsules are.
"""

from __future__ import annotations

import math

import genesis as gs

from eden import EventMode, observations, rewards, terminations
from eden.engine.solvers.mochi.options import MochiOptions
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options import ObservationGroupOptions
from eden.options.entities import BoxOptions, CylinderOptions, RodOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.materials import MochiRigidMaterialOptions, MochiRodMaterialOptions
from eden.tasks.mochi import custom_terms
from eden.tasks.mochi.common import (
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
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

# ---- the board: RoboSimStudio's board.py, in bench_075's work-surface frame -------------------------------------
LENGTH, RADIUS, SEGMENTS = 0.60, 0.0035, 50
START = (-0.18, -0.36)  # the welded end
PEGS = ((-0.075, -0.26), (-0.075, -0.02))  # the two spools
# Which side of each spool the cable must pass, in x: -1 is the robot's side. The spool nearer the welded end is
# passed away from the robot and the second toward it, so the cable crosses the spools' line twice before the port.
SIDES = (+1, -1)
ANCHOR = (START[0], START[1] + 0.008, math.pi / 2)  # the bracket over the welded end: x, y, yaw
PORT = (-0.014, 0.08, -math.pi / 2)  # the port, facing the run: x, y, yaw
SHAFT_R, FLANGE_R, GROOVE, HEIGHT = 0.00475, 0.0125, (0.0025, 0.019679), 0.0235  # ManipulationNet's F1
TERMINAL = 0.016  # half the terminals' 32 x 32 mm footprint
# The routed goal's boxes.
SEAT = (0.003, 0.045, 0.035)  # from 3 to 45 mm off a spool's axis on its side, 35 mm either way along the line
PORT_BOX = 0.045  # 45 mm every way from the port's centre
HEIGHT_BOX = 0.15  # every box runs from the bench to 150 mm over it
# The rod: the reference cable's bend (0.04 N m a joint on a 22 mm, 32.7 mm-pitch rod), scaled as RoboSimStudio
# scales it to 7 mm and a 12 mm pitch, as a rigidity EI = k * pitch.
_PITCH = LENGTH / SEGMENTS
_JOINT_BEND = 0.04 * (RADIUS / 0.011) ** 4 * ((0.9144 / 28) / _PITCH)
FLEXURAL_STIFFNESS = _JOINT_BEND * _PITCH
# Welded under the bracket: the first two nodes pinned (the second holds the twist).
ANCHORED_NODES = [0, 1]

_EPISODE_S = 20.0
CONTROL_DT = 1.0 / 30.0
_RIGHT_TCP = {"robot_name": "right", "link_name": "link_6", "link_offset": YAM_TCP_OFFSET}
_LEFT_TCP = {"robot_name": "left", "link_name": "link_6", "link_offset": YAM_TCP_OFFSET}
# Every contact with the cable starts at its radius: a rod touches through its centreline.
_BOARD = MochiRigidMaterialOptions(penalty_coefficient=CONTACT_STIFFNESS, friction=0.3, penalty_threshold=RADIUS)
_ORANGE, _METAL = (0.85, 0.35, 0.12), (0.38, 0.40, 0.43)


def _node(s: float) -> int:
    """The rod node ``s`` metres along the cable from its welded end."""
    return round(s / _PITCH)


def _spool(x: float, y: float) -> dict[str, CylinderOptions]:
    """The F1 as three cylinders: a foot flange, the shaft the cable seats against, a rim."""
    parts = (
        ("foot", FLANGE_R, 0.0, GROOVE[0]),
        ("shaft", SHAFT_R, GROOVE[0], GROOVE[1]),
        ("rim", FLANGE_R, GROOVE[1], HEIGHT),
    )
    return {
        part: CylinderOptions(
            radius=r,
            height=z1 - z0,
            fixed=True,
            default_root_pos=(x, y, 0.5 * (z0 + z1)),
            material=_BOARD,
            surface=gs.surfaces.Default(color=_ORANGE),
        )
        for part, r, z0, z1 in parts
    }


def _box(frame, centre, half, color) -> BoxOptions:
    """A box given in a terminal's own frame (local +x is the way the run leaves)."""
    x, y, yaw = frame
    c, s = math.cos(yaw), math.sin(yaw)
    return BoxOptions(
        size=tuple(2.0 * h for h in half),
        fixed=True,
        default_root_pos=(x + c * centre[0] - s * centre[1], y + s * centre[0] + c * centre[1], centre[2]),
        default_root_quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
        material=_BOARD,
        surface=gs.surfaces.Default(color=color),
    )


def _board() -> dict:
    hood = 0.5 * (0.0125 - 0.008)  # the hood's underside 8 mm up, just over the cable's 7 mm crown
    rail = (TERMINAL, 0.0055, 0.0025)
    wall = (TERMINAL, 0.0055, 0.00625)
    parts = {f"spool0_{k}": v for k, v in _spool(*PEGS[0]).items()}
    parts |= {f"spool1_{k}": v for k, v in _spool(*PEGS[1]).items()}
    parts |= {
        "anchor_rail_a": _box(ANCHOR, (0.0, 0.0105, 0.0025), rail, _METAL),
        "anchor_rail_b": _box(ANCHOR, (0.0, -0.0105, 0.0025), rail, _METAL),
        "anchor_hood": _box(ANCHOR, (-0.008, 0.0, 0.008 + hood), (0.008, TERMINAL, hood), _ORANGE),
        "anchor_saddle": _box(ANCHOR, (0.008, 0.0, 0.008 + hood), (0.0032, TERMINAL, hood), _ORANGE),
        "port_wall_a": _box(PORT, (0.0, 0.0105, 0.00625), wall, _ORANGE),
        "port_wall_b": _box(PORT, (0.0, -0.0105, 0.00625), wall, _ORANGE),
        "port_back": _box(PORT, (-0.0128, 0.0, 0.00625), (0.0032, 0.005, 0.00625), _ORANGE),
    }
    return parts


def seat_box(i: int) -> dict:
    """Spool ``i``'s seat box (``lower``, ``upper`` corners): 3 to 45 mm off its axis on the side the cable passes."""
    (x, y), side = PEGS[i], SIDES[i]
    near, far = sorted((x + side * SEAT[0], x + side * SEAT[1]))
    return {"lower": (near, y - SEAT[2], 0.0), "upper": (far, y + SEAT[2], HEIGHT_BOX)}


_PORT_BOX = {
    "lower": (PORT[0] - PORT_BOX, PORT[1] - PORT_BOX, 0.0),
    "upper": (PORT[0] + PORT_BOX, PORT[1] + PORT_BOX, HEIGHT_BOX),
}
# The routine's marks, where it lays each point of the cable down: (distance along the cable from its welded end,
# where on the bench). The route hugs the spools (about 0.53 m of the 0.60 m cable to the port's back): it starts on
# the robot's side of the line, crosses it to the first spool's far side and back to the second's near side. The first
# mark sits short of the first spool, so the cable from the welded end crosses the line clear of the spool's flange
# (29 mm short of its axis) rather than draping over it onto the robot's side.
MARKS = (
    (0.165, (PEGS[0][0] + 0.020, PEGS[0][1] - 0.012)),  # beside the first spool, on the far side
    (0.31, (PEGS[0][0] - 0.003, 0.5 * (PEGS[0][1] + PEGS[1][1]))),  # across the spools' line, between them
    (0.455, (PEGS[1][0] - 0.018, PEGS[1][1] + 0.020)),  # round the robot's side of the second
    (LENGTH, (PORT[0], PORT[1])),  # the free end, in the port
)


def _hand(robot: str) -> DifferentialIKController:
    return DifferentialIKController.configure(
        entity_name=robot,
        ee_link_name="link_6",
        dofs_name=YAM_ARM_DOFS,
        position_scale=0.015,
        # The tilt held (pointing down), the turn about the vertical free: 0.1 rad a step.
        orientation_scale=[0.0, 0.0, 0.1],
        damping=0.02,
        max_dq=0.1,
    )


def _gripper(robot: str) -> custom_terms.RateLimitedGripper:
    return custom_terms.RateLimitedGripper.configure(
        entity_name=robot, dofs_name=YAM_FINGER_DOFS, open_action=YAM_FINGERS_OPEN, close_action=YAM_FINGERS_CLOSED
    )


_CABLE_KEYPOINTS = list(range(0, SEGMENTS + 1, 5))

_Observations = {
    "right_dofs_pos": observations.DofsPos.configure(entity_name="right"),
    "right_dofs_vel": observations.DofsVel.configure(entity_name="right"),
    "left_dofs_pos": observations.DofsPos.configure(entity_name="left"),
    "left_dofs_vel": observations.DofsVel.configure(entity_name="left"),
    "right_tool": custom_terms.ToolCentre.configure(**_RIGHT_TCP),
    "left_tool": custom_terms.ToolCentre.configure(**_LEFT_TCP),
    # The cable's shape, from the right hand and from the left.
    "cable_right": custom_terms.SoftBodyPoints.configure(entity_name="cable", vertices=_CABLE_KEYPOINTS, **_RIGHT_TCP),
    "cable_left": custom_terms.SoftBodyPoints.configure(entity_name="cable", vertices=_CABLE_KEYPOINTS, **_LEFT_TCP),
    "right_grip": observations.MetricObservation.configure(metric_name="right_grip", scale=0.05),
    "left_grip": observations.MetricObservation.configure(metric_name="left_grip", scale=0.05),
    "last_action": ObsTerm.configure(func=observations.last_action),
}

# The routine's hand for each mark.
_TOOLS = (_RIGHT_TCP, _RIGHT_TCP, _RIGHT_TCP, _LEFT_TCP)


def _shaping() -> dict:
    """Per mark: the routine's hand reaching that point of the cable, and the point reaching its mark.

    The reach is paid by distance alone. Paid in full while the jaws gripped (as the other tasks' reaches are), it
    paid all three of the right hand's marks for one grip anywhere on the cable, and the policy held on and placed
    nothing. Each placement pays on a coarse kernel (the points start 10-30 cm from their marks) and a fine one.
    """
    terms = {}
    for k, ((s, (x, y)), tool) in enumerate(zip(MARKS, _TOOLS)):
        node = min(_node(s), SEGMENTS)
        terms[f"reach_{k}"] = custom_terms.ReachSoftBody.configure(
            entity_name="cable", vertices=[node], **tool, grip_metric_name="", std=0.1, weight=0.1
        )
        for name, std, weight in (("place", 0.15, 0.25), ("place_fine", 0.05, 0.5)):
            terms[f"{name}_{k}"] = custom_terms.RodNodeToPoint.configure(
                entity_name="cable", node=node, point=(x, y, RADIUS), std=std, weight=weight
            )
    return terms


@TASK_REGISTRY.register()
class MochiRouteTheCableConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=256,
        episode_length_s=_EPISODE_S,
        sim_dt=CONTROL_DT,
        sim_substeps=1,
        decimation=1,
        mochi_options=MochiOptions(n_newton_iterations=4),
    )

    scene_options: SceneOptions = SceneOptions(
        # The bench under the arms, and under the cable a bed in the same plane that pushes back at the cable's
        # radius (a rod touches through its centreline): Mochi takes the band from the collider, so one plane with
        # the cable's band would stop the fingertips at the cable's centreline, and the jaws would ride over it.
        plane=bench(friction=0.9),
        # Friction 0.3 against the cable (Mochi's geometric mean of 0.09 and the cable's 1.0). RoboSimStudio's
        # bench is slicker (0.02) but its contacts are sticky, which is static friction; at 0.02 a laid cable
        # slid out of the second spool's seat when the plug carried the end past it.
        cable_bed=bench(friction=0.09, penalty_threshold=RADIUS).model_copy(update={"visualization": False}),
        # The pads grip it hard (RoboSimStudio gives them the rod's 60): Mochi's mean of 9 and 1 is 3.
        right=yam(
            "right",
            ready="jaws_x",
            finger_kp=6000.0,
            finger_kd=120.0,
            friction=9.0,
            penalty_threshold=RADIUS,
            base_height=YAM_PLATE_DEPTH,
        ),
        left=yam(
            "left",
            ready="jaws_x",
            finger_kp=6000.0,
            finger_kd=120.0,
            friction=9.0,
            penalty_threshold=RADIUS,
            base_height=YAM_PLATE_DEPTH,
        ),
        cable=RodOptions(
            points=[(START[0], START[1] + i * _PITCH, RADIUS) for i in range(SEGMENTS + 1)],
            radius=RADIUS,
            material=MochiRodMaterialOptions(
                rho=1000.0,
                flexural_stiffness=FLEXURAL_STIFFNESS,
                # All but inextensible, as RoboSimStudio's rigid capsules are.
                axial_stiffness=2e3,
                stiffness_damping=1e-3,
                friction=1.0,
                # Below this slip Mochi's friction grows with the slip, so a lower falloff is closer to static
                # friction; at the default 10 mm/s a bent cable crept out of the port after it was let go.
                friction_falloff_vel=1e-4,
                penalty_coefficient=CONTACT_STIFFNESS,
            ),
            surface=gs.surfaces.Default(color=(0.08, 0.08, 0.09)),
        ),
        **_board(),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        seat0=custom_terms.RodShareInBox.configure(entity_name="cable", **seat_box(0), success_threshold=0.04),
        seat1=custom_terms.RodShareInBox.configure(entity_name="cable", **seat_box(1), success_threshold=0.04),
        port=custom_terms.RodShareInBox.configure(entity_name="cable", **_PORT_BOX, success_threshold=0.06),
        end_in_port=custom_terms.RodEndInBox.configure(entity_name="cable", **_PORT_BOX, success_threshold=1.0),
        hands_clear=custom_terms.HandsClear.configure(
            entity_name="cable", robots_name=["right", "left"], link_offset=YAM_TCP_OFFSET, success_threshold=0.05
        ),
        at_rest=custom_terms.SoftBodySettled.configure(entity_name="cable", quantile=1.0, success_threshold=0.05),
        right_grip=custom_terms.JawGrip.configure(entity_name="right", contributes_to_success=False),
        left_grip=custom_terms.JawGrip.configure(entity_name="left", contributes_to_success=False),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Observations, enable_corruption=True),
        critic=ObservationGroupOptions(**_Observations, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        right_hand=_hand("right"),
        right_gripper=_gripper("right"),
        left_hand=_hand("left"),
        left_gripper=_gripper("left"),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        **_shaping(),
        seat0=rewards.MetricReward.configure(metric_name="seat0", weight=5.0),
        seat1=rewards.MetricReward.configure(metric_name="seat1", weight=5.0),
        port=rewards.MetricReward.configure(metric_name="port", weight=5.0),
        # Staged on the way to the goal: routed whether or not let go, then routed and graded by how far both hands are
        # from it, up to the goal's 5 cm. Without the second the policy routed the cable and kept hold of it, its
        # hands 1-3 cm away; paid only once both were 5 cm clear, it was never found.
        routed=custom_terms.MetricsHold.configure(metrics_name=["seat0", "seat1", "port", "end_in_port"], weight=2.0),
        routed_clear=custom_terms.MetricsHold.configure(
            metrics_name=["seat0", "seat1", "port", "end_in_port"],
            scale_metric_name="hands_clear",
            scale_target=0.05,
            weight=8.0,
        ),
        # Every step the whole goal holds (let go and at rest included).
        success=rewards.TaskSuccess.configure(weight=50.0),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        right_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "right", "max_vel": 1.0}
        ),
        left_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "left", "max_vel": 1.0}
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        # No termination on success: the goal is a state to hold (a towel lying folded, a cable routed and let go),
        # and the success bonus pays every step it holds. Ended on success, the episode forfeited the dense terms'
        # remaining pay, worth more than a one-off bonus, and the policy learned to keep hold of the part.
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # Welded under the bracket; Mochi keeps pins outside the state a reset restores, so they are re-made each reset.
        weld_cable=custom_terms.PinSoftBodyVertices.configure(
            mode=EventMode.RESET, entity_name="cable", vertices=ANCHORED_NODES
        ),
        # Each bench plane carries only its own bodies: the cable lies on the bed, the arms stop on the bench.
        cable_off_bench=custom_terms.DisableMochiContact.configure(
            mode=EventMode.STARTUP, entity_name="plane", other_name="cable"
        ),
        right_off_bed=custom_terms.DisableMochiContact.configure(
            mode=EventMode.STARTUP, entity_name="cable_bed", other_name="right"
        ),
        left_off_bed=custom_terms.DisableMochiContact.configure(
            mode=EventMode.STARTUP, entity_name="cable_bed", other_name="left"
        ),
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
        experiment_name="mochi_route_the_cable",
        run_name="",
    )
