"""Robotiq 2F-85 parallel-jaw gripper configuration."""

from __future__ import annotations
from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions, SolParamsType

# Loop-closure / pad-contact solver parameters, stiffened from the MJCF's
# solref "0.005 1" / solimp "0.95 0.99": timeconst 0.002 and dmax 0.999 make the
# four-bar closure and the pinch contact near-rigid. Genesis clamps timeconst to
# 2*substep_dt (with a warning), so full stiffness needs sim_dt/sim_substeps <= 0.001;
# at larger steps the values degrade gracefully to the clamp.
_STIFF_SOL_PARAMS: SolParamsType = (0.002, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)


@ROBOT_REGISTRY.register()
class Robotiq2f85(RobotOptions):
    """Robotiq 2F85 parallel-jaw gripper, tuned for stable grasping.

    The 2F85 is an underactuated four-bar linkage: MuJoCo closes it with
    ``<equality><connect>`` constraints and drives it through a ``<tendon>``
    actuator. Genesis has no runtime tendon (the actuator is approximated per
    driver joint at parse time) and solves the loop closures softly, so the
    stock asset penetrates/oscillates instead of pinching. These options bake in
    the validated compensation (tuned by SpaceMouse-teleop A/B against the
    loop-free Franka Hand; pinned by the grasp-and-lift regression in
    ``tests/options/robots/test_robotiq_2f85.py``):

    - **Pads**: flat box collision geoms with high friction (a face contact that
      resists rotation), instead of the original 5-sphere cluster. Pair with
      ``collision_link_patterns=[".*pad"]`` so only the pads collide and the
      object isn't wedged against the inner linkage.
    - **Drive** (``default_dofs_force_limits``): the kp=100 servo commanded to
      the closed target would develop ~40 N·m and crush through a small object;
      the cap reproduces MuJoCo's tendon ``forcerange -5 5`` × wrap coef 0.485
      = 2.4 N·m per driver.
    - **Loop closure** (``default_equalities_sol_params``): the connects +
      driver coupling are stiffened to near-rigid so grip force transmits
      through the four-bar to the pads instead of flexing it.
    - **Contact** (``default_geoms_sol_params``): pad contact stiffened to
      match. Genesis mixes a pair's params as the mean of its two geoms (geom
      ``priority`` is ignored), so graspable objects should carry comparable
      sol params via their own ``default_geoms_sol_params`` — never via a
      global ``EnvOptions.constraint_timeconst``, which over-stiffens every
      rigid contact and ejects mesh objects on contact.
    - **Spring preload** (``default_dofs_spring_ref``): Genesis drops MJCF
      ``springref``; re-anchoring the spring-link springs at 2.62 rad restores
      the preload that keeps the passive pad chain taut.
    - **Dissipation** (``damping``/``frictionloss``/``armature``): at contact
      the drivers are torque-saturated, so kp/kd have no authority over contact
      chatter; always-on joint damping (~critical for the pinch mode) and a dry
      frictionloss stick band absorb it instead.

    Full constraint stiffness needs an effective substep dt <= 0.001 (e.g.
    ``sim_dt=0.004, sim_substeps=4``); Genesis clamps solref timeconst to
    ``2*substep_dt`` otherwise.
    """

    file: str = "grippers/robotiq_2f85/2f85.xml"

    is_fixed_base: bool = True
    links_to_keep: list[str] = ["gripper_base_link"]
    # No default_root_quat: the -90° z mounting rotation (mjlab's Droid tool frame)
    # is baked into the asset's ``base`` body. Duplicating it here would double-apply, since
    # ``attach_to`` uses default_root_quat as the mounting transform.

    dofs_name: list[str] = [
        "left_driver_joint",
        "left_spring_link_joint",
        "left_follower",
        "right_driver_joint",
        "right_spring_link_joint",
        "right_follower_joint",
    ]
    default_dofs_pos: dict[str, float] = {
        "left_driver_joint": 0.0,
        "left_spring_link_joint": 0.0,
        "left_follower": 0.0,
        "right_driver_joint": 0.0,
        "right_spring_link_joint": 0.0,
        "right_follower_joint": 0.0,
    }
    default_dofs_kp: dict[str, float] = {
        "left_driver_joint": 100.0,
        "right_driver_joint": 100.0,
    }
    default_dofs_kd: dict[str, float] = {
        "left_driver_joint": 10.0,
        "right_driver_joint": 10.0,
    }
    # MuJoCo-equivalent drive cap: tendon forcerange 5 N x wrap coef 0.485 (see docstring).
    default_dofs_force_limits: dict[str, float] = {
        "left_driver_joint": 2.4,
        "right_driver_joint": 2.4,
    }
    # Joint-level dissipation for contact chatter (the saturated servo can't damp it):
    # driver damping ~critical for the pinch mode (contact stiffness reflected at the
    # driver ~45 N*m/rad over ~0.005 kg*m^2), frictionloss well under the 2.4 N*m drive;
    # the passive pad chain (XML: damping 0.00125/0) gets light damping + a stick band
    # under the 0.13 N*m spring preload, and a touch more armature to slow its
    # near-massless modes.
    default_dofs_damping: dict[str, float] = {
        ".*_driver_joint": 1.0,
        ".*_spring_link_joint": 0.05,
        "left_follower": 0.05,
        "right_follower_joint": 0.05,
    }
    default_dofs_frictionloss: dict[str, float] = {
        ".*_driver_joint": 0.1,
        ".*_spring_link_joint": 0.01,
        "left_follower": 0.01,
        "right_follower_joint": 0.01,
    }
    default_dofs_armature: dict[str, float] = {
        ".*_spring_link_joint": 0.002,
        "left_follower": 0.002,
        "right_follower_joint": 0.002,
    }
    # Restore the MJCF springref=2.62 preload Genesis drops (see docstring).
    default_dofs_spring_ref: dict[str, float] = {
        ".*_spring_link_joint": 2.62,
    }
    default_equalities_sol_params: SolParamsType | None = _STIFF_SOL_PARAMS
    default_geoms_sol_params: dict[str, SolParamsType] = {
        ".*pad": _STIFF_SOL_PARAMS,
    }

    actuated_dofs_name: ClassVar[list[str]] = ["left_driver_joint", "right_driver_joint"]

    open_dofs_pos: ClassVar[dict[str, float]] = {
        "left_driver_joint": 0.0,
        "right_driver_joint": 0.0,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "left_driver_joint": 0.8,
        "right_driver_joint": 0.8,
    }

    # Pad midpoint along the gripper base's z axis, measured on the built pad geoms (the pads span
    # 0.127-0.155 m); the 2F85's 85 mm stroke.
    tcp_offset: ClassVar[tuple[float, float, float]] = (0.0, 0.0, 0.1413)
    stroke: ClassVar[float] = 0.085

    # The 0.002 timeconst of ``_STIFF_SOL_PARAMS`` survives Genesis' ``2 * substep_dt`` clamp only at
    # this substep or finer (see the class docstring): ``sim_dt=0.004`` with ``sim_substeps=4``, say.
    max_substep_dt: ClassVar[float] = 0.001
