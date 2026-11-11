r"""OmniReset (UWLab, ICLR 2026): a UR5e + Robotiq 2F-85 assembles an object pair from recorded reset states.

``omnireset`` is UWLab's stage-1 training task (``OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0``):
the scene, observations, controller and 16 s episode the released expert (``eden.extensions.policies.omnireset``)
was trained in, each episode starting from one of four recorded reset families drawn at 0.25 each, UWLab's rewards
and startup randomization, and its PPO recipe. Train with ``eden train --task omnireset``.

``--robot ur5e_2f85`` (:class:`OmniResetRobotMod`) puts the same task on Eden's ``ur5e_2f85`` composite, for
training a policy of our own (:data:`~.constants.RIGS`, :func:`robot_entities`). It restores UWLab's states
retargeted onto that robot; run ``python -m eden.tasks.omnireset.retarget --pair peg`` once first.

UWLab evaluates (``...-State-Play-v0``) with every episode from :data:`~.constants.EVAL_FAMILY`
(``ObjectAnywhereEEAnywhere``). ``uv run python examples/policies/omnireset/main.py`` runs the released expert that
way. To evaluate a trained checkpoint the same way, override its reset weights::

    eden inference --task omnireset --checkpoint <run>/model_<n>.pt --num_envs 256 --episodes 256 \
        --override "event_options: {reset_states: {category_weights: {ObjectAnywhereEEAnywhere: 1.0,
        ObjectRestingEEGrasped: 0.0, ObjectAnywhereEEGrasped: 0.0, ObjectPartiallyAssembledEEGrasped: 0.0}}}"

Domain randomization follows UWLab's ``BaseEventCfg`` with two adaptations. Genesis combines a contact pair's
friction as the max where PhysX averages it, so only the insertive object's friction is randomized, over 0.9-1.6:
the released expert keeps its success there (94 % from ObjectAnywhereEEAnywhere), while 0.65-1.3, the span of
UWLab's averaged pairs, drops it to 87 % (Genesis' grasps slip more at low friction). And the gripper gains are scaled
uniformly over (0.5, 2), where UWLab draws the factor log-uniformly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from eden import actions, observations, terminations
from eden.managers.terms.events.domain_rand import (
    RandomizeFrictionRatio,
    RandomizeKpKdGains,
    RandomizeLinkMassScale,
    RandomizeMassShift,
)
from eden.options import EntityOptions, PlaneOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlHeteroscedasticGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots import OmniResetUR5e2f85
from eden.options.robots.composite.droid import _droid_gripper
from eden.options.robots.composite.ur_robotiq import UR5eRobotiq
from eden.options.robots.universal_ur5e import UR5e
from eden.options.scenes.empty import EmptySceneOptions
from eden.tasks.omnireset import constants as C
from eden.tasks.omnireset.custom_terms import (
    ArmOverspeed,
    JointPositionsByName,
    OmniResetStates,
    PoseInFrame,
    RelCartesianOSC,
)
from eden.tasks.omnireset.custom_terms.metrics import AssemblyError
from eden.tasks.omnireset.custom_terms.rewards import (
    AssemblyReward,
    GripperObjectProximity,
    JointVelL2Clamped,
    TerminationFired,
    action_l2_clamped,
    action_rate_l2_clamped,
)
from eden.tasks.registry import TASK_REGISTRY, TaskMod
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

if TYPE_CHECKING:
    from eden.options.entities import EntityOptionsLike
    from eden.options.managers.actions import ActionTermOptions


# ---------------------------------------------------------------------------
# Scene: UWLab's ``RlStateSceneCfg``
# ---------------------------------------------------------------------------


def robot_entities(rig: C.Rig) -> dict[str, EntityOptionsLike]:
    """The robot's scene entries: the arm's base is a fixed body each reset row places (``batch_fixed_verts``).

    On the composite (``"ur5e_2f85"``) two things change from the stock :class:`UR5eRobotiq`: the gripper is welded
    :data:`~.constants.COMPOSITE_MOUNT_OFFSET` off the flange, where UWLab's camera mount puts it, so UWLab's arm
    joint angles still put the grasp point where they recorded it; and the arm carries UWLab's system-identified
    motor armature, the real arm's reflected inertia, which keeps the 120 Hz explicit operational-space controller
    stable without PhysX's phantom gripper mass. Both rigs' robots, gripper pads included, carry the task's robot
    friction: the composite's pads would otherwise keep the 1.2 their MJCF authors, and since Genesis takes a contact
    pair's max friction, the insertive object's randomized friction would never reach below 1.2 at the grasp.
    """
    material = RigidMaterialOptions(gravity_compensation=1.0, friction=C.ROBOT_FRICTION)
    if rig.name == "uwlab":
        return {
            C.ROBOT: OmniResetUR5e2f85(
                batch_fixed_verts=True,
                default_root_pos=C.ROBOT_BASE_POS,
                material=material,
                default_geoms_sol_params={".*_inner_finger": C.GRASP_SOL_PARAMS},
                default_equalities_sol_params=C.GRASP_SOL_PARAMS,
            )
        }
    arm_joints = C.EDEN_ARM_JOINTS
    home = dict(zip(arm_joints, (0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708), strict=True))
    arm = UR5e(
        batch_fixed_verts=True,
        default_root_pos=C.ROBOT_BASE_POS,
        material=material,
        default_dofs_pos=home,
        # Torque-driven by RelCartesianOSC: no PD; UWLab's effort limits and measured armature.
        default_dofs_kp=dict.fromkeys(arm_joints, 0.0),
        default_dofs_kd=dict.fromkeys(arm_joints, 0.0),
        default_dofs_force_limits=dict(zip(arm_joints, C.OSC_TORQUE_LIMIT, strict=True)),
        default_dofs_armature=dict(zip(arm_joints, C.UR5E_SYSID_ARMATURE, strict=True)),
    )
    # The pads carry the robot friction too, as UWLab's do; left alone they keep the MJCF's 1.2.
    gripper = _droid_gripper().model_copy(update={"material": material})
    return {C.ROBOT: UR5eRobotiq(arm=arm, gripper=gripper, attach_offset=C.COMPOSITE_MOUNT_OFFSET)}


def scene_entity_options(pair: C.Pair, rig: C.RigName = "uwlab") -> dict[str, EntityOptions | PlaneOptions]:
    """Every entity of UWLab's ``RlStateSceneCfg`` for ``pair`` and robot ``rig``, keyed as the reset buffer is.

    The robot base, the mount plate and the receptive object are fixed bodies a reset state places per env, so
    they are built with ``batch_fixed_verts``. The pads, the objects and the gripper's couplings carry
    :data:`~.constants.GRASP_SOL_PARAMS`, which need :data:`~.constants.SIM_SUBSTEPS`.
    """
    base_x, base_y, base_z = C.ROBOT_BASE_POS
    return {
        # "plane" replaces the flat ground a preset scene adds at z = 0, which would cut through the table top.
        "plane": PlaneOptions(default_root_pos=(0.0, 0.0, C.GROUND_Z)),
        "table": EntityOptions(
            file=C.prop_file("pat_vention"),
            is_fixed_base=True,
            default_root_pos=C.TABLE_POS,
            default_root_quat=C.TABLE_QUAT,
            material=RigidMaterialOptions(friction=C.TABLE_FRICTION),
        ),
        C.MOUNT: EntityOptions(
            file=C.prop_file("ur5_metal_support"),
            is_fixed_base=True,
            batch_fixed_verts=True,
            default_root_pos=(base_x, base_y, base_z + C.MOUNT_OFFSET_Z),
        ),
        **robot_entities(C.RIGS[rig]),
        C.RECEPTIVE: EntityOptions(
            file=C.prop_file(pair.receptive),
            is_fixed_base=True,
            batch_fixed_verts=True,
            default_root_pos=(0.4, 0.1, 0.0022),
            color=(0.2, 0.45, 0.85, 1.0),  # cosmetic: UWLab's USDs carry no colours
            material=RigidMaterialOptions(friction=C.RECEPTIVE_FRICTION),
            default_geoms_sol_params={".*": C.GRASP_SOL_PARAMS},
        ),
        C.INSERTIVE: EntityOptions(
            file=C.prop_file(pair.insertive),
            default_root_pos=(0.4, -0.1, 0.002),
            color=(0.95, 0.5, 0.1, 1.0),
            material=RigidMaterialOptions(friction=C.INSERTIVE_FRICTION),
            default_geoms_sol_params={".*": C.GRASP_SOL_PARAMS},
        ),
    }


# ---------------------------------------------------------------------------
# Observation and action spaces: the released experts were trained on them, and the policy extension
# (``OmniResetPolicy``) hands them back for a task-less run rather than defining them.
# ---------------------------------------------------------------------------


def observation_group(rig: C.Rig) -> ObservationGroupOptions:
    """UWLab's policy group: last action, joint angles, then four relative poses as position + axis-angle."""

    def pose(target: str, target_link: str | None, root: str, root_link: str | None):
        return PoseInFrame.configure(
            target_entity_name=target, target_link_name=target_link, root_entity_name=root, root_link_name=root_link
        )

    joints = {
        f"joint_pos{'' if i == 0 else f'_{entity}'}": JointPositionsByName.configure(
            entity_name=entity, joints_name=list(names)
        )
        for i, (entity, names) in enumerate(rig.joint_observation)
    }
    return ObservationGroupOptions(
        concatenate_terms=True,
        enable_corruption=False,
        prev_arm_action=ObsTerm.configure(func=observations.last_action, params={"action_name": "arm"}),
        prev_gripper_action=ObsTerm.configure(func=observations.last_action, params={"action_name": "gripper"}),
        **joints,
        end_effector_pose=pose(rig.arm_entity, rig.ee_link, rig.arm_entity, rig.base_link),
        insertive_asset_pose=pose(C.INSERTIVE, None, rig.arm_entity, rig.ee_link),
        receptive_asset_pose=pose(C.RECEPTIVE, None, rig.arm_entity, rig.ee_link),
        insertive_asset_in_receptive_asset_frame=pose(C.INSERTIVE, None, C.RECEPTIVE, None),
    )


def action_terms(rig: C.Rig) -> dict[str, ActionTermOptions]:
    """UWLab's relative Cartesian OSC on the arm and its binary gripper."""
    return dict(
        arm=RelCartesianOSC.configure(
            entity_name=rig.arm_entity,
            dofs_name=list(rig.arm_joints),
            ee_link_name=rig.ee_link,
            base_link_name=rig.base_link,
            scale=C.OSC_SCALE,
            kp=C.OSC_KP,
            damping_ratio=C.OSC_DAMPING_RATIO,
            torque_limit=C.OSC_TORQUE_LIMIT,
        ),
        # Eden closes on a positive scaled action; UWLab closes on a negative one.
        gripper=actions.BinaryJointController.configure(
            entity_name=rig.gripper_entity,
            dofs_name=list(rig.gripper_drivers),
            open_action=C.GRIPPER_OPEN,
            close_action=rig.gripper_closed,
            scale=-1.0,
        ),
    )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

_PAIR = "peg"
#: UWLab trains from all four reset families, drawn evenly.
_TRAIN_MIX = dict.fromkeys(C.RESET_FAMILIES, 0.25)
#: Every link the robot moves (UWLab scales each body's mass); the base welded to the mount stays.
_MOVING_LINKS = {
    "uwlab": {
        "robot": [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_.*",
            "robotiq_base_link",
            ".*knuckle",
            ".*finger",
        ]
    },
    "ur5e_2f85": {
        "robot": ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_.*"],
        "gripper": ["base", ".*driver", ".*coupler", ".*spring_link", ".*follower", ".*pad"],
    },
}
#: UWLab draws the insertive mass from U(0.02, 0.2) kg; the asset carries the midpoint, so a shift around it.
_INSERTIVE_MASS_SHIFT = (0.02 - C.INSERTIVE_MASS, 0.2 - C.INSERTIVE_MASS)


def _env_options(rig: C.Rig, num_envs: int) -> EnvOptions:
    return EnvOptions(
        num_envs=num_envs,
        episode_length_s=C.EPISODE_LENGTH_S,
        sim_dt=C.SIM_DT,
        sim_substeps=rig.sim_substeps,
        decimation=C.DECIMATION,
        batch_dofs_info=True,  # per-env gripper gains
        batch_links_info=True,  # per-env masses
    )


def _scene_options(rig: C.Rig) -> SceneOptions:
    entities = scene_entity_options(C.PAIRS[_PAIR], rig.name)
    return EmptySceneOptions(robot=entities.pop("robot"), **entities)


def _event_options(rig: C.Rig) -> EventManagerOptions:
    """UWLab's ``BaseEventCfg`` around the reset that picks the episode's recorded state."""
    masses = {
        f"{entity}_mass": RandomizeLinkMassScale.configure(
            mode="startup", entity_name=entity, links_name=links, mass_scale_range=(0.7, 1.3)
        )
        for entity, links in _MOVING_LINKS[rig.name].items()
    }
    return EventManagerOptions(
        reset_states=OmniResetStates.configure(pair=_PAIR, rig=rig.name, category_weights=_TRAIN_MIX),
        insertive_friction=RandomizeFrictionRatio.configure(
            mode="startup", entity_name=C.INSERTIVE, links_name=[".*"], friction_range=(0.9, 1.6)
        ),
        **masses,
        insertive_mass=RandomizeMassShift.configure(
            mode="startup", entity_name=C.INSERTIVE, links_name=[".*"], mass_shift_range=_INSERTIVE_MASS_SHIFT
        ),
        gripper_gains=RandomizeKpKdGains.configure(
            entity_name=rig.gripper_entity, kp_range=(0.5, 2.0), kd_range=(0.5, 2.0)
        ),
    )


def _metric_options() -> MetricManagerOptions:
    pair = C.PAIRS[_PAIR]
    return MetricManagerOptions(
        assembly_position=AssemblyError.configure(
            pair=_PAIR, component="position", success_threshold=pair.success_position
        ),
        assembly_tilt=AssemblyError.configure(pair=_PAIR, component="tilt", success_threshold=pair.success_orientation),
    )


def _reward_options(rig: C.Rig) -> RewardManagerOptions:
    return RewardManagerOptions(
        action_magnitude=RewardTerm.configure(func=action_l2_clamped, weight=-1e-4),
        action_rate=RewardTerm.configure(func=action_rate_l2_clamped, weight=-1e-3),
        joint_vel=JointVelL2Clamped.configure(
            entity_name=rig.arm_entity, joints_name=list(rig.arm_joints), weight=-1e-2
        ),
        abnormal_robot=TerminationFired.configure(termination_name="abnormal_robot", weight=-100.0),
        ee_asset_distance=GripperObjectProximity.configure(
            entity_name=rig.grasp_entity, link_name=rig.grasp_link, offset=rig.grasp_offset, weight=0.1, std=1.0
        ),
        dense_success_reward=AssemblyReward.configure(kind="dense", std=1.0, weight=0.1),
        success_reward=AssemblyReward.configure(kind="success", weight=1.0),
    )


def _termination_options(rig: C.Rig) -> TerminationManagerOptions:
    return TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        abnormal_robot=ArmOverspeed.configure(entity_name=rig.arm_entity, velocity_limits=rig.arm_velocity_limits),
    )


def _runner_options(experiment_name: str) -> RslRlOnPolicyRunnerOptions:
    """UWLab's ``Base_PPORunnerCfg``: actor and critic both read the policy group."""
    return RslRlOnPolicyRunnerOptions(
        num_steps_per_env=32,
        max_iterations=40000,
        save_interval=100,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        experiment_name=experiment_name,
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.006,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128, 64],
            obs_normalization=True,
            distribution_cfg=RslRlHeteroscedasticGaussianDistributionOptions(init_std=1.0, std_type="log"),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[512, 256, 128, 64], obs_normalization=True),
    )


#: The rig the task is built on unless ``--robot`` names another: UWLab's own, which the released expert drives.
_DEFAULT_RIG: C.RigName = "uwlab"


def _rig_options(rig: C.Rig) -> dict[str, Any]:
    """Every option group that names the robot's entities, links or joints, built for ``rig``.

    :class:`OmniResetConfig` takes these defaults from it and :class:`OmniResetRobotMod` swaps all of it in, so the
    class and the mod cannot disagree on which groups follow the robot. A group that starts to name the robot
    belongs here, not in the class body, where the mod would leave it on the old robot.
    """
    suffix = "" if rig.name == _DEFAULT_RIG else f"_{rig.name}"
    return {
        "env_options": _env_options(rig, num_envs=4096),
        "scene_options": _scene_options(rig),
        "observation_options": ObservationManagerOptions(policy=observation_group(rig)),
        "action_options": ActionManagerOptions(**action_terms(rig)),
        "event_options": _event_options(rig),
        "reward_options": _reward_options(rig),
        "termination_options": _termination_options(rig),
        # Obs and action dims follow the rig, so a checkpoint does not transfer: keep the runs apart.
        "runner_options": _runner_options(f"omnireset_peg{suffix}"),
    }


_DEFAULT_OPTIONS = _rig_options(C.RIGS[_DEFAULT_RIG])


class OmniResetRobotMod(TaskMod):
    """Put OmniReset on another rig (``--robot``): every group of :func:`_rig_options` is rebuilt for it.

    The rig runs through every group, so the mod replaces whole groups rather than editing named terms in place as
    ``MoveForwardRobotMod`` does: one builder decides what follows the robot. A subclass's edits to those groups do
    not survive ``--robot ur5e_2f85``; make them after the build, or with ``--override``.
    """

    def __init__(
        self,
        robot: Annotated[
            C.RigName,
            "uwlab (UWLab's UR5e + 2F-85, which the released expert drives) or ur5e_2f85 (Eden's composite; run "
            "python -m eden.tasks.omnireset.retarget --pair peg once first).",
        ] = _DEFAULT_RIG,
    ):
        self.robot = robot

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        if self.robot != _DEFAULT_RIG:  # the class already holds the default rig's options
            for field, options in _rig_options(C.RIGS[self.robot]).items():
                setattr(config, field, options)
        return config


@TASK_REGISTRY.register(name="omnireset", modifiers=(OmniResetRobotMod,))
class OmniResetConfig(EdenRLConfig):
    env_options: EnvOptions = _DEFAULT_OPTIONS["env_options"]
    scene_options: SceneOptions = _DEFAULT_OPTIONS["scene_options"]
    observation_options: ObservationManagerOptions = _DEFAULT_OPTIONS["observation_options"]
    action_options: ActionManagerOptions = _DEFAULT_OPTIONS["action_options"]
    event_options: EventManagerOptions = _DEFAULT_OPTIONS["event_options"]
    metric_options: MetricManagerOptions = _metric_options()
    reward_options: RewardManagerOptions = _DEFAULT_OPTIONS["reward_options"]
    termination_options: TerminationManagerOptions = _DEFAULT_OPTIONS["termination_options"]
    runner_options: RslRlOnPolicyRunnerOptions = _DEFAULT_OPTIONS["runner_options"]
