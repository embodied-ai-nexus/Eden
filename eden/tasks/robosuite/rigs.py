"""Robot embodiments for the robosuite ports, and the ``--robot`` TaskMod that swaps them.

A robosuite task is authored around one *rig* per robot slot: a
:class:`~eden.options.robots.ManipulatorOptions` composite (an arm + a two-finger gripper,
welded, expanding into ``<slot>`` and ``<slot>_gripper`` entities at build) from
:data:`~eden.options.robots.COMPOSITE_ROBOT_REGISTRY`. The default is robosuite's own robot,
:class:`~eden.options.robots.PandaFrankaHand`; :data:`ROBOT_NAMES` lists every registered
alternative (Droid, UR5e / UR10e + Robotiq).

The composite already carries everything a term needs — ``arm_name`` / ``gripper_name``, ``ee_link``,
``arm_dofs`` / ``gripper_dofs``, ``grasp_links``, the open / close targets, the gripper's tool point
(``tcp_offset``) and ``stroke`` — and the port's terms read it off the scene themselves
(:class:`~eden.tasks.robosuite.custom_terms.common.RigMixin` takes the *slot*, not a copy of its
facts). So a swap is a scene edit: :class:`RobosuiteRobotMod` replaces the composite in each robot
slot, retargets the two built-in action terms that do name entities and DOFs explicitly, refits the
substep to the new gripper, and hands the rigs to
:meth:`~eden.tasks.robosuite.common.RobosuiteTaskConfig.on_rigs_changed`. Nothing else in the config
mentions the robot, which is why there is no rewrite pass here and nothing that can be left stale.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Literal

from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, ManipulatorOptions
from eden.tasks.registry import TaskMod

if TYPE_CHECKING:
    from eden.tasks.robosuite.common import RobosuiteTaskConfig

#: robosuite's default embodiment: the Panda arm with its own Franka Hand.
DEFAULT_ROBOT = "panda_franka_hand"
#: Every arm + two-finger-gripper composite the registry offers, ``--robot``'s choices.
ROBOT_NAMES: tuple[str, ...] = tuple(
    sorted(
        name
        for name in COMPOSITE_ROBOT_REGISTRY.keys()
        if issubclass(COMPOSITE_ROBOT_REGISTRY.get(name), ManipulatorOptions)
    )
)
RobotName = Literal[ROBOT_NAMES]  # type: ignore[valid-type]

#: robosuite ``Panda.init_qpos``: ``[0, pi/16, 0, -pi/2 - pi/3, 0, pi - 0.2, pi/4]``.
PANDA_HOME = {
    "joint1": 0.0,
    "joint2": math.pi / 16.0,
    "joint3": 0.0,
    "joint4": -math.pi / 2.0 - math.pi / 3.0,
    "joint5": 0.0,
    "joint6": math.pi - 0.2,
    "joint7": math.pi / 4.0,
}


def rig_name(rig: ManipulatorOptions) -> str:
    """The :data:`COMPOSITE_ROBOT_REGISTRY` key of ``rig``'s class (one of :data:`ROBOT_NAMES`)."""
    for name in ROBOT_NAMES:
        if COMPOSITE_ROBOT_REGISTRY.get(name) is type(rig):
            return name
    raise TypeError(
        f"{type(rig).__name__} is not a registered arm + two-finger-gripper composite; robosuite rigs are one of "
        f"{list(ROBOT_NAMES)}."
    )


#: robosuite's ``PandaGripper`` (``panda_gripper.xml``): each finger a position actuator with kp 1000 clipped
#: to +-20 N, on a slide joint with damping 100, armature 1.0 and friction loss 1.0; pads of friction 2.
#: The actuator has no velocity gain, so ``kd`` is 0 and the joint's damping is the passive
#: ``default_dofs_damping``, outside the 20 N clip (as a ``kd`` it would sit inside it).
#: Eden's ``FrankaHand`` defaults (Menagerie's kp 100) close with at most 4 N per finger — 1.4 N on a
#: 3 cm handle — which cannot lift robosuite's 0.25 kg lid and lets a held hammer twist out of the grip.
_ROBOSUITE_PANDA_FINGERS = {
    "default_dofs_kp": 1000.0,
    "default_dofs_kd": 0.0,
    "default_dofs_damping": 100.0,
    "default_dofs_force_limits": 20.0,
    "default_dofs_armature": 1.0,
    "default_dofs_frictionloss": 1.0,
}
_ROBOSUITE_PANDA_PAD_FRICTION = 2.0


def _robosuite_panda_gripper(rig: ManipulatorOptions) -> None:
    """Give a Panda rig's Franka Hand robosuite's finger actuation and pad friction.

    See :data:`_ROBOSUITE_PANDA_FINGERS` for the values and why the defaults fall short.
    """
    gripper = rig.gripper
    for field, value in _ROBOSUITE_PANDA_FINGERS.items():
        setattr(gripper, field, dict.fromkeys(rig.gripper_dofs, value))
    gripper.default_geoms_friction = {".*_gripper_pad": _ROBOSUITE_PANDA_PAD_FRICTION}


def make_rig(
    robot: str,
    slot: str,
    base_pos: tuple[float, float, float],
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ManipulatorOptions:
    """Build the composite named ``robot`` for scene slot ``slot``.

    Its arm entity takes the slot's name and its gripper ``<slot>_gripper``.

    robosuite's own embodiment (:data:`DEFAULT_ROBOT`) gets robosuite's gripper actuation; the other
    rigs keep their registry defaults, having no robosuite counterpart to match.
    """
    if robot not in ROBOT_NAMES:
        raise ValueError(f"Unknown robosuite robot '{robot}'; choose one of {list(ROBOT_NAMES)}.")
    composite_cls = COMPOSITE_ROBOT_REGISTRY.get(robot)
    rig = composite_cls(base_pos=base_pos, arm_name=slot, gripper_name=f"{slot}_gripper")
    rig.arm.default_root_quat = base_quat
    if robot == DEFAULT_ROBOT:
        _robosuite_panda_gripper(rig)
    return rig


def panda_rig(
    slot: str,
    base_pos: tuple[float, float, float],
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ManipulatorOptions:
    """The default embodiment at robosuite's home pose (:data:`PANDA_HOME`), fingers open."""
    rig = make_rig(DEFAULT_ROBOT, slot, base_pos, base_quat)
    rig.arm.default_dofs_pos = dict(PANDA_HOME)
    return rig


def replace_rig_placement(
    old: ManipulatorOptions,
    slot: str,
    base_pos: tuple[float, float, float],
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ManipulatorOptions:
    """Rebuild ``old``'s embodiment at a new base placement, keeping the home pose it was authored with.

    Re-running :func:`make_rig` alone would hand back the composite's *registry* defaults, dropping
    whatever the task set on the arm — for the Panda that is robosuite's ``init_qpos``
    (:data:`PANDA_HOME`, written by :func:`panda_rig`), so a placement-only mod would silently start
    the arm somewhere else than the single-arm tasks do. Only the home pose needs carrying: every
    other arm-level field a robosuite config touches is derived from the placement itself.
    """
    rig = make_rig(rig_name(old), slot, base_pos, base_quat)
    rig.arm.default_dofs_pos = dict(old.arm.default_dofs_pos)
    return rig


def rig_of(config: RobosuiteTaskConfig, slot: str) -> ManipulatorOptions:
    """The rig a config's scene slot holds, checked to be one (the scene entity *is* the rig)."""
    rig = getattr(config.scene_options, slot)
    if not isinstance(rig, ManipulatorOptions):
        raise TypeError(
            f"scene slot '{slot}' holds a {type(rig).__name__}; robosuite configs place a ManipulatorOptions "
            f"composite there (see make_rig)."
        )
    rig_name(rig)  # registered, so the mods can rebuild it by name
    if rig.arm_name != slot:
        raise ValueError(
            f"scene slot '{slot}' holds a rig whose arm_name is '{rig.arm_name}'. The ports key their terms on the "
            f"slot and read the rig off the scene under that key, so a rig must be built for the slot it sits in "
            f"(make_rig(robot, slot, ...) does this)."
        )
    return rig


# --- the TaskMod ----------------------------------------------------------------------------------


class RobosuiteRobotMod(TaskMod):
    """Swap the arm + gripper embodiment of a robosuite task (``--robot``).

    The config is authored around the Panda. For another rig this replaces the scene entity in
    every robot slot the config declares (``robot_slots``), retargets the ``<slot>_arm`` IK and
    ``<slot>_gripper`` controllers, refits the substep to what the new gripper needs
    (:func:`~eden.tasks.robosuite.common.fit_substeps_to_rigs`), hands the new rigs to
    :meth:`RobosuiteTaskConfig.on_rigs_changed`, and suffixes the experiment name. The MDP terms
    need no edit: they name the *slot* and read the rig off the scene at build.

    The base placement is kept, so the new arm stands where the Panda stood; obs dims follow the
    arm's DOF count, so checkpoints do not transfer across rigs.
    """

    def __init__(
        self,
        robot: Annotated[
            RobotName,
            "Arm + gripper composite from COMPOSITE_ROBOT_REGISTRY: panda_franka_hand (robosuite's default), droid, "
            "ur5e_2f85 or ur10e_2f85.",
        ] = DEFAULT_ROBOT,
    ):
        self.robot = robot

    def apply(self, config: RobosuiteTaskConfig) -> RobosuiteTaskConfig:
        slots = getattr(type(config), "robot_slots", None)
        if not slots:
            raise TypeError(
                f"{type(config).__name__} declares no robot_slots; RobosuiteRobotMod only applies to configs that "
                f"list the scene keys holding a rig (e.g. robot_slots = ('robot',))."
            )
        swapped = False
        rigs: list[ManipulatorOptions] = []
        for slot in slots:
            old = rig_of(config, slot)
            if rig_name(old) == self.robot:
                rigs.append(old)
                continue
            new = make_rig(self.robot, slot, tuple(old.arm.default_root_pos), tuple(old.arm.default_root_quat))
            # One scene entry per rig: the composite expands into its arm + gripper entities only at build time.
            setattr(config.scene_options, slot, new)
            self._retarget_actions(config, new)
            rigs.append(new)
            swapped = True
        if swapped:
            # Local import: ``common`` imports this module for the rig helpers its config base needs.
            from eden.tasks.robosuite.common import fit_substeps_to_rigs

            fit_substeps_to_rigs(config.env_options, rigs)
            config.on_rigs_changed(rigs)
            config.runner_options.experiment_name = f"{config.runner_options.experiment_name}_{self.robot}"
        return config

    @staticmethod
    def _retarget_actions(config: RobosuiteTaskConfig, new: ManipulatorOptions) -> None:
        arm = getattr(config.action_options, f"{new.arm_name}_arm")
        arm.entity_name = new.arm_name
        arm.ee_link_name = new.ee_link
        arm.dofs_name = list(new.arm_dofs)
        gripper = getattr(config.action_options, f"{new.arm_name}_gripper", None)
        if gripper is None:  # a gripper-less task (TwoArmPegInHole welds the objects to the hands)
            return
        gripper.entity_name = new.gripper_name
        gripper.dofs_name = list(new.gripper_dofs)
        gripper.open_action = dict(new.gripper_open)
        gripper.close_action = dict(new.gripper_close)
