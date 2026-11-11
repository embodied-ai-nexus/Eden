"""Shared arena, timing, camera and observation/action factories for the robosuite ports.

Numbers follow robosuite's defaults (``TableArena``, ``Panda.base_xpos_offset``, the ``OSC_POSE``
controller's per-step output limits, the ``agentview`` camera); the module docstring of each task
names what it changes.

Timing. robosuite steps MuJoCo at 500 Hz and the policy at 20 Hz. Eden runs the same 20 Hz control
rate over a 5 ms / 2-substep Genesis step (``decimation=10``). The 50 s robosuite horizon is not
kept: every task defaults to :data:`EPISODE_LENGTH_S`, which is plenty for one pick-and-place and
keeps a training iteration short; override it with ``--override 'env_options: {episode_length_s: 50}'``
for parity.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Annotated, ClassVar, Literal

import numpy as np
from genesis.utils import geom as gu

import eden as en

from eden import EventMode, events, observations
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import BaseEntityOptions, BoxOptions, PlaneOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.managers.actions import ActionTermOptions
from eden.options.managers.events import EventTermOptions
from eden.options.managers.observations import ObservationGroupOptions, ObservationTermOptions
from eden.options.managers.terminations import TerminationTermOptions
from eden.options.materials import KinematicMaterialOptions, RigidMaterialOptions
from eden.options.objects.libero import LiberoFloorWorkspace
from eden.options.objects.robosuite import RethinkMount
from eden.tasks.robosuite.custom_terms import (
    ArmDofsPosSinCos,
    ArmDofsVel,
    EefPos,
    EefQuat,
    entity_below_height,
    GripperDofsPos,
    GripperDofsVel,
    GripperToObject,
)
from eden.options.robots import ManipulatorOptions
from eden.tasks.registry import TaskMod
from eden.tasks.robosuite.rigs import replace_rig_placement, rig_of
from eden.terms import DoneTerm, EventTerm, ObsTerm
from eden.utils.configs import EdenRLConfig, EnvOptions, ObservationManagerOptions

# --- arena ----------------------------------------------------------------------------------------

#: robosuite ``TableArena`` default: 0.8 x 0.8 x 0.05 m slab whose top is 0.8 m above the floor.
TABLE_SIZE = (0.8, 0.8, 0.05)
TABLE_TOP_Z = 0.8
TABLE_FRICTION = 1.0
TABLE_COLOR = (0.75, 0.73, 0.70, 1.0)

#: Panda ``base_xpos_offset["table"]`` for the 0.8 m table: 0.16 m behind the table edge.
PANDA_TABLE_XY = (-0.56, 0.0)
#: robosuite mounts every arm on a ``RethinkMount`` pedestal: link0 sits 0.912 m above the floor, and the
#: mount's root body 1 cm above that (its top surface is at ``-0.01`` in the root frame).
ROBOT_BASE_Z = 0.912
MOUNT_ROOT_ABOVE_ARM = 0.01

#: Contact solver parameters (time constant, damping ratio, then the impedance curve) of an object a hand
#: picks up and carries. Genesis' default 0.02 s time constant is soft for objects of 50-250 g: the
#: robosuite gripper's 20 N pinch sinks its pads 6 mm into the TwoArmTransport hammer's 3 cm handle and the
#: hammer creeps out of the fingers within seconds; the ToolHang wrench, pinched, turns in the fingers
#: three times as far as it does on this time constant. 0.005 s (the floor Genesis allows at the ports'
#: substep, 2 x 2.5 ms) holds them. Stiffening the objects rather than the pads keeps every other
#: robosuite task's contacts as they were (AGENTS.md: stiffen graspable objects per object).
GRASPED_SOL_PARAMS = (0.005, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)

#: Every task terminates a fallen object a little below the table top.
FALLEN_MARGIN = 0.10


def arena_entities() -> dict[str, BaseEntityOptions]:
    """Robosuite's arena look plus Eden's ground collision, as two named scene entries.

    ``floor`` is LIBERO's floor workspace — robosuite's own arena MJCF (the 6 x 6 m tiled floor and the
    six visual walls) — loaded as a *kinematic* entity, so it renders but generates no contacts.
    ``plane`` is the ground the objects and robots actually rest on, with its rendering off so the two
    surfaces do not z-fight. Splat ``**arena_entities()`` into a ``SceneOptions``.
    """
    return {
        "floor": LiberoFloorWorkspace(material=KinematicMaterialOptions()),
        "plane": PlaneOptions(visualization=False),
    }


def table_options(
    size: tuple[float, float, float] = TABLE_SIZE,
    *,
    top_z: float = TABLE_TOP_Z,
    xy: tuple[float, float] = (0.0, 0.0),
    friction: float = TABLE_FRICTION,
) -> BoxOptions:
    """A fixed table slab whose top surface is at ``top_z`` (robosuite's ``table_offset``)."""
    return BoxOptions(
        size=size,
        fixed=True,
        default_root_pos=(xy[0], xy[1], top_z - size[2] / 2),
        material=RigidMaterialOptions(friction=friction),
        color=TABLE_COLOR,
    )


def rethink_mount(
    base_pos: tuple[float, float, float], base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
) -> RethinkMount:
    """Robosuite's pedestal under an arm whose ``link0`` is at ``base_pos`` (the mount top is 1 cm below its root)."""
    x, y, z = base_pos
    return RethinkMount(default_root_pos=(x, y, z + MOUNT_ROOT_ABOVE_ARM), default_root_quat=base_quat)


def rpy_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """The (w, x, y, z) quaternion of an rpy triple, as plain floats for a config field.

    Config-time spelling of :func:`genesis.utils.geom.xyz_to_quat` — the numpy adapter is the whole
    body. Rotations are authored through this and :func:`axis_angle_quat` rather than written out as
    ``cos``/``sin`` literals, per the geometry rule in ``AGENTS.md``: a hand-derived wxyz constant is
    unverifiable at a glance and wrong silently.
    """
    return tuple(float(v) for v in gu.xyz_to_quat(np.array([roll, pitch, yaw], dtype=np.float64), rpy=True))


def axis_angle_quat(angle: float, axis: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """The (w, x, y, z) quaternion of ``angle`` radians about ``axis`` (normalized), as plain floats."""
    return tuple(float(v) for v in gu.axis_angle_to_quat(np.array(angle), np.array(axis, dtype=np.float64)))


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """The (w, x, y, z) quaternion of a rotation by ``yaw`` radians about +z."""
    return rpy_quat(0.0, 0.0, yaw)


# --- timing / env -------------------------------------------------------------------------------

#: robosuite's control rate. ``decimation`` is derived from it and ``sim_dt`` rather than written
#: out, so the two cannot drift apart.
CONTROL_HZ = 20
SIM_DT = 0.005
EPISODE_LENGTH_S = 10.0


def robosuite_env_options(
    num_envs: int = 4096, *, episode_length_s: float = EPISODE_LENGTH_S, **overrides
) -> EnvOptions:
    """20 Hz control over a 5 ms / 2-substep Genesis step; ``overrides`` reach :class:`EnvOptions` directly."""
    options = dict(
        num_envs=num_envs,
        episode_length_s=episode_length_s,
        sim_dt=SIM_DT,
        sim_substeps=2,
        decimation=round(1.0 / (CONTROL_HZ * SIM_DT)),
        max_collision_pairs=40,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
    )
    options.update(overrides)
    return EnvOptions(**options)


def fit_substeps_to_rigs(options: EnvOptions, rigs: Iterable[ManipulatorOptions]) -> None:
    """Raise ``sim_substeps`` until the substep meets every rig's ``max_substep_dt``.

    A closed-loop gripper is only as stiff as the substep lets it be: Genesis clamps a ``solref``
    ``timeconst`` to ``2 * substep_dt``, so a Robotiq 2F85's four-bar closure and pinch contact go
    soft above 1 ms and the fingers slide off what they hold — silently, because the scene still
    builds and steps at any substep. The defaults above are robosuite's 20 Hz over ``sim_dt=0.005``
    with two substeps, which suits the Franka Hand the tasks are authored around (rigid, no loop);
    ``--robot`` swapping in a Robotiq rig refits the substep here rather than leaving behind a
    gripper that cannot grasp. It is a real cost — the physics runs 2.5x per step — and the honest
    one: the alternative is a task that trains against a gripper the policy cannot close.
    """
    limits = [rig.max_substep_dt for rig in rigs if rig.max_substep_dt is not None]
    if not limits:
        return
    needed = math.ceil(options.sim_dt / min(limits) - 1e-9)
    if needed > options.sim_substeps:
        en.logger.info(
            f"Raising sim_substeps {options.sim_substeps} -> {needed}: the rig needs a substep of "
            f"{min(limits)} s or finer to keep its closed-loop gripper stiff."
        )
        options.sim_substeps = needed


# --- camera ---------------------------------------------------------------------------------------


def agentview_camera(
    *,
    cam_pos: tuple[float, float, float] = (1.35, 0.0, 1.55),
    cam_lookat: tuple[float, float, float] = (0.0, 0.0, 0.85),
    resolution: tuple[int, int] = (640, 480),
) -> CamerasOptions:
    """The ``agentview`` vantage of robosuite: in front of the table, looking back at the robot."""
    return CamerasOptions(
        agentview=CameraOptions(cam_pos=cam_pos, cam_lookat=cam_lookat, cam_resolution=resolution, cam_fov=45.0)
    )


# --- observations ---------------------------------------------------------------------------------

_OBS_NOISE = UniformNoise.configure(n_min=-0.005, n_max=0.005)


def proprio_observations(rig: ManipulatorOptions, *, prefix: str = "") -> dict[str, ObservationTermOptions]:
    """The default proprioceptive observables of robosuite for one arm, keyed ``<prefix>joint_pos_sincos`` etc.

    Every one of them takes the rig's *slot* and reads the arm, gripper and mount link off the scene
    at build, so ``--robot`` never has to touch them.
    """
    slot = rig.arm_name
    return {
        f"{prefix}joint_pos_sincos": ArmDofsPosSinCos.configure(rig=slot),
        f"{prefix}joint_vel": ArmDofsVel.configure(rig=slot),
        f"{prefix}eef_pos": EefPos.configure(rig=slot),
        f"{prefix}eef_quat": EefQuat.configure(rig=slot),
        f"{prefix}gripper_qpos": GripperDofsPos.configure(rig=slot),
        f"{prefix}gripper_qvel": GripperDofsVel.configure(rig=slot),
    }


def object_pose_noise() -> UniformNoise:
    """The observation noise the object-pose terms carry in the policy group (0.5 cm uniform)."""
    return _OBS_NOISE


def object_pose_observations(name: str) -> dict[str, ObservationTermOptions]:
    """The ``<obj>_pos`` / ``<obj>_quat`` pair robosuite reports for one object, in the asset's own frame.

    ``relative=True`` is the whole point of this factory. Genesis re-origins a free rigid body's
    base link onto its inertial frame, and the root-pose getters default to that solver frame — so
    the frame these two terms report in is *not* the frame the rest of a robosuite task reasons in.
    Every reward, metric, termination and placement event here reads the MJCF body origin
    (``ObjectPointMixin``, ``ReachTanh``, ``entity_below_height``, ``object_above_height``,
    ``events.reset_base_state_uniform``), and so does robosuite's own ``object-state``.
    Measured on the shipped assets, the gap is 6.8 cm on the hammer and 5.8 cm on the ToolHang
    frame, and it *rotates with the object* — reporting the solver frame here would hand the policy
    ``<obj>_pos`` and ``eef_pos + gripper_to_<obj>`` as two different positions for one object.

    Every task routes its object pose observables through here rather than open-coding the pair, so
    the frame is decided once.
    """
    return {
        f"{name}_pos": ObsTerm.configure(
            func=observations.base_pos,
            params={"entity_name": name, "relative": True},
            noise=object_pose_noise(),
        ),
        f"{name}_quat": ObsTerm.configure(
            func=observations.base_quat,
            params={"entity_name": name, "relative": True},
            noise=object_pose_noise(),
        ),
    }


def object_observations(rig: ManipulatorOptions, name: str) -> dict[str, ObservationTermOptions]:
    """The per-object observables robosuite reports: the object's pose plus the gripper-to-object vector.

    The three terms every task declares for a manipulable object, in robosuite's order. Both halves
    read the frame the asset was authored in — see :func:`object_pose_observations` for why that has
    to be the same frame on both, and what it costs when it is not.
    """
    return {
        **object_pose_observations(name),
        f"gripper_to_{name}": GripperToObject.configure(rig=rig.arm_name, object_name=name),
    }


def policy_and_critic(observations: dict[str, ObservationTermOptions]) -> ObservationManagerOptions:
    """The two observation groups every robosuite task declares: same terms, corruption on the policy only.

    The critic sees the uncorrupted values; nothing else differs, so the two are built from one dict
    rather than written out twice per task (where they could silently drift apart).
    """
    return ObservationManagerOptions(
        policy=ObservationGroupOptions(**observations, enable_corruption=True),
        critic=ObservationGroupOptions(**observations, enable_corruption=False),
    )


def merged(factory: Callable[[str], dict], names: Iterable[str]) -> dict:
    """Flatten a per-object term factory over ``names``, keeping declaration order.

    Manager options are ordered dicts and metric terms are computed in declaration order, so the
    flattening has to preserve it: ``names`` first, then each factory's own keys.
    """
    return {key: value for name in names for key, value in factory(name).items()}


# --- events / terminations shared by every task ----------------------------------------------------

#: robosuite's "default" initialization noise on an arm. robosuite samples ``N(0, 0.02)`` per joint;
#: ``reset_dofs_by_offset`` is uniform, so this is the matching half-width rather than a sigma.
ARM_INIT_NOISE_RANGE = (-0.02, 0.02)


def arm_init_noise(rig: ManipulatorOptions) -> EventTermOptions:
    """Jitter one arm's joints about their home pose on reset (:data:`ARM_INIT_NOISE_RANGE`)."""
    return EventTerm.configure(
        func=events.reset_dofs_by_offset,
        mode=EventMode.RESET,
        params={"entity_name": rig.arm_name, "dofs_pos_range": ARM_INIT_NOISE_RANGE},
    )


def fallen_termination(name: str, *, top_z: float = TABLE_TOP_Z) -> TerminationTermOptions:
    """End the episode once ``name`` drops :data:`FALLEN_MARGIN` below the surface it started on.

    robosuite only terminates on the horizon; Eden's episodes are shorter, so an object that has
    fallen off the table ends the episode rather than spending the rest of it out of reach.
    """
    return DoneTerm.configure(
        func=entity_below_height,
        params={"entity_name": name, "height": top_z - FALLEN_MARGIN},
    )


# --- actions --------------------------------------------------------------------------------------


def arm_actions(rig: ManipulatorOptions) -> dict[str, ActionTermOptions]:
    """The 7-D robosuite action for one arm: 6-D delta end-effector pose (``OSC_POSE``) + binary gripper.

    The IK step limits are robosuite's ``output_max``: 5 cm and 0.5 rad per control step. Keys are
    ``<slot>_arm`` / ``<slot>_gripper`` so :class:`~eden.tasks.robosuite.rigs.RobosuiteRobotMod` can
    find them.

    The control point is the rig's ``ee_link`` — the arm's mounting flange — not the tool point
    between the fingers that robosuite's ``OSC_POSE`` acts at, and not the pad midpoint the
    ``eef_pos`` observable reports (:class:`~eden.tasks.robosuite.custom_terms.EefPos`). The two are
    ``tcp_offset`` apart (10.3 cm on the Franka Hand, 14.1 cm on a Robotiq), so a pure-rotation
    action swings the fingers through an arc of up to ``0.5 rad x tcp_offset`` — about 5 cm — where
    robosuite would turn them in place. Position deltas are unaffected. Moving the control point to
    ``tcp_offset`` would change the action space of every task, so it is left as a documented
    deviation (``docs/tasks/robosuite.md``) rather than changed under a port that claims parity.
    """
    return {
        f"{rig.arm_name}_arm": DifferentialIKController.configure(
            entity_name=rig.arm_name,
            ee_link_name=rig.ee_link,
            dofs_name=list(rig.arm_dofs),
            position_scale=0.05,
            orientation_scale=0.5,
            damping=0.1,
            max_dq=0.3,
        ),
        f"{rig.arm_name}_gripper": BinaryJointController.configure(
            entity_name=rig.gripper_name,
            dofs_name=list(rig.gripper_dofs),
            open_action=dict(rig.gripper_open),
            close_action=dict(rig.gripper_close),
        ),
    }


# --- runner ---------------------------------------------------------------------------------------


def ppo_runner_options(experiment_name: str, *, max_iterations: int = 3000) -> RslRlOnPolicyRunnerOptions:
    """The PPO recipe shared by the ports (``manipulation_lift_box``'s, which trains a from-scratch grasp)."""
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
        max_iterations=max_iterations,
        save_interval=50,
        experiment_name=experiment_name,
        run_name="v1",
    )


# --- config base ----------------------------------------------------------------------------------


class RobosuiteTaskConfig(EdenRLConfig):
    """Base of every robosuite task config: declares which scene keys hold a robot rig.

    :class:`~eden.tasks.robosuite.rigs.RobosuiteRobotMod` reads ``robot_slots`` to know which
    entities to swap; the single-arm tasks keep the default, the two-arm tasks declare
    ``("robot0", "robot1")``.
    """

    robot_slots: ClassVar[tuple[str, ...]] = ("robot",)

    def on_rigs_changed(self, rigs: list[ManipulatorOptions]) -> None:
        """Re-derive scene state that depends on the rigs beyond their names and placement.

        Both rig mods call this after they have written the new rig entities into the scene —
        :class:`~eden.tasks.robosuite.rigs.RobosuiteRobotMod` with the swapped embodiments,
        :class:`TwoArmArrangementMod` with the re-placed ones — in ``robot_slots`` order. The
        default does nothing; ``TwoArmPegInHole`` welds its objects to the hands and presets the
        fingers around them here.
        """


# --- two-arm placement --------------------------------------------------------------------------

Arrangement = Literal["opposed", "parallel"]
#: One arm's base placement: ``(position, wxyz quaternion)``.
Placement = tuple[tuple[float, float, float], tuple[float, float, float, float]]
#: Both arms' placements per arrangement, e.g. ``{"opposed": (p0, p1), "parallel": (p0, p1)}``.
Placements = dict[str, tuple[Placement, Placement]]

#: The Handover / Transport table: robosuite's ``table_true_size = (0.8, 0.3, 0.05)``, against the 0.8 m
#: square :data:`TABLE_SIZE` the single-arm tasks share. Transport sets two of them, one either side of the gap.
NARROW_TABLE_SIZE = (0.8, 0.3, 0.05)
#: robosuite ``TwoArmEnv`` placement for Handover and Transport: the ``"table"`` offset ``(-0.56, 0, 0)`` rotated
#: by +-90 degrees and pushed out by 0.25 m (opposed), or shifted sideways by 0.6 m (parallel).
NARROW_TABLE_PLACEMENTS: Placements = {
    "opposed": (
        ((0.0, -0.81, ROBOT_BASE_Z), yaw_quat(math.pi / 2)),
        ((0.0, 0.81, ROBOT_BASE_Z), yaw_quat(-math.pi / 2)),
    ),
    "parallel": (
        ((-0.56, -0.6, ROBOT_BASE_Z), yaw_quat(0.0)),
        ((-0.56, 0.6, ROBOT_BASE_Z), yaw_quat(0.0)),
    ),
}


class TwoArmArrangementMod(TaskMod):
    """robosuite's ``env_configuration`` for two single arms (``--arrangement``).

    ``opposed`` faces the arms across the workspace, ``parallel`` stands them side by side facing
    +x; the per-task base positions come in as ``placements`` at registration. The mod
    rebuilds both rig entities at the new placement — keeping whichever embodiment ``--robot``
    chose, so the two compose in either order — moves the pedestals ``mount0`` / ``mount1``, and
    hands the re-placed rigs to :meth:`RobosuiteTaskConfig.on_rigs_changed`.
    """

    def __init__(
        self,
        arrangement: Annotated[
            Arrangement, "Arm placement: opposed (facing each other across the workspace) or parallel (side by side)."
        ] = "opposed",
        *,
        placements: Placements | None = None,
    ):
        self.arrangement = arrangement
        self.placements = placements

    def apply(self, config: RobosuiteTaskConfig) -> RobosuiteTaskConfig:
        if self.placements is None:
            raise TypeError(
                f"{type(self).__name__} needs the task's base positions: register it as "
                f"{type(self).__name__}(placements={{'opposed': ..., 'parallel': ...}})."
            )
        rigs: list[ManipulatorOptions] = []
        for i, (slot, (pos, quat)) in enumerate(zip(type(config).robot_slots, self.placements[self.arrangement])):
            # Placement only — ``replace_rig_placement`` keeps the home pose the config authored,
            # which a bare ``make_rig`` would reset to the composite's registry default.
            rig = replace_rig_placement(rig_of(config, slot), slot, pos, quat)
            setattr(config.scene_options, slot, rig)
            setattr(config.scene_options, f"mount{i}", rethink_mount(pos, quat))
            rigs.append(rig)
        config.on_rigs_changed(rigs)
        if self.arrangement != "opposed":
            config.runner_options.experiment_name = f"{config.runner_options.experiment_name}_{self.arrangement}"
        return config
