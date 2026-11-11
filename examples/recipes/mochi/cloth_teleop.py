"""Keyboard or SpaceMouse teleoperation of a Franka grabbing two stacked Mochi cloths draped over cubes.

---
tier: recipe
capabilities: [solver.mochi, robot.franka, teleop.keyboard, teleop.spacemouse]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/cloth_teleop.py
---

Eden port of the Genesis fork's ``examples/mochi/cloth_teleop.py`` (itself the Mochi counterpart of the IPC
robot-cloth teleoperation example). Two self-colliding shells lie over a 4x4 grid of fixed cubes; the teleop
device drives the end-effector frame through a ``DifferentialIKController`` while a ``BinaryJointController``
opens and closes the fingers. The arm, both cloths, their self-contact and the penalty contact all enter one
implicit Newton solve at four Newton iterations per step, so grabbing a fold and dragging it is stable at a
20 ms step.

Viewer-first: it runs with the window open unless ``--no-vis`` is passed. ``decimation=1`` keeps the teleop
input live. Keys: ``wasd`` translate, ``qe`` raise/lower, ``zc`` yaw, ``space`` toggles the gripper, ``x``
resets. ``--spacemouse`` drives the same command from a 3Dconnexion SpaceMouse instead (the puck is the
delta pose, button 0 toggles the gripper, button 1 resets), which suits dragging a fold across the grid far
better than the keyboard's one-axis-at-a-time keys. It needs ``spnav``, whose install recipe is in
``eden/utils/devices/spacemouse.py``; the keyboard path has no extra dependency.

    uv run python examples/recipes/mochi/cloth_teleop.py
    uv run python examples/recipes/mochi/cloth_teleop.py --spacemouse
"""

import argparse
import threading

import genesis as gs
from _mochi_common import FRANKA_ARM_DOFS, FRANKA_FINGER_DOFS, MOCHI_GRAVITY, init, sheet_mesh

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiShellMaterialOptions
from eden.utils.devices.keyboard import SE3Keyboard
from eden.utils.devices.spacemouse import SE3Spacemouse

CUBE_SIZE, GRID_SPACING = 0.05, 0.15


def teleop_device(use_spacemouse: bool) -> tuple[SE3Keyboard | SE3Spacemouse, threading.Event]:
    """Open the requested teleop device and bind its reset control.

    Both devices emit the same seven-component command — a delta pose plus the gripper toggle — so the
    control loop does not care which is in use. Only the reset binding differs: a keyboard callback is
    keyed by key name, a SpaceMouse callback by button number.

    Either device runs its callbacks on its own listener thread, so the reset control only raises the
    returned event and the control loop resets between steps: an ``env.reset()`` from the callback would
    race the ``env.step()`` running on the main thread.
    """
    if use_spacemouse:
        # Importing the module is safe without spnav; only constructing the device raises, with the
        # install recipe in the message.
        device, reset_control = SE3Spacemouse(is_toggle_btn0=True), 1
    else:
        device, reset_control = SE3Keyboard(SE3Keyboard.wasd_key_bindings()), "x"
    reset_requested = threading.Event()
    device.add_callback(reset_control, lambda on_press: reset_requested.set() if on_press else None)
    return device, reset_requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-vis", action="store_false", dest="vis", help="Run headless (teleop needs the viewer)")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("--spacemouse", action="store_true", help="Teleoperate with a SpaceMouse, not the keyboard")
    args = parser.parse_args()
    init(args)

    # Two stacked sheets over the cubes; light cloth takes a softer contact stiffness (see the README).
    cloth = {"nu": 0.3, "rho": 200.0, "thickness": 1e-3, "friction": 1.0, "penalty_coefficient": 1e7}
    cubes = {
        f"cube_{i}_{j}": BoxOptions(
            size=(CUBE_SIZE,) * 3,
            default_root_pos=((i + 1.7) * GRID_SPACING, (j - 1.5) * GRID_SPACING, 0.5 * CUBE_SIZE),
            is_fixed_base=True,
            material=MochiRigidMaterialOptions(friction=0.5),
            surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
        )
        for i in range(4)
        for j in range(4)
    }

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.02,
            decimation=1,  # teleop drives the env with env.step(action), so decimation is live
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=4),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            franka=en.robots.FrankaEmikaPanda(
                default_root_pos=(0.0, 0.0, 0.005),
                # Rubber-pad friction (the FinRay pads of the Schunk model use the same value). No gravity on the
                # arm: the differential IK sets each step's joint targets from the measured pose, so a sag under
                # gravity would accumulate step after step (a held key sinks the hand ~6 cm/s and slows a lift).
                material=MochiRigidMaterialOptions(friction=3.0, viscous_friction=1.0, has_gravity=False),
            ),
            sheet_large=EntityOptions(
                file=sheet_mesh(20, 0.5)[0],
                default_root_pos=(0.5, 0.0, 0.1),
                material=MochiShellMaterialOptions(E=6e4, collider_radius=1.2e-2, self_contact=True, **cloth),
                surface=gs.surfaces.Plastic(color=(0.3, 0.1, 0.8, 1.0)),
            ),
            sheet_small=EntityOptions(
                file=sheet_mesh(12, 0.3)[0],
                default_root_pos=(0.5, 0.0, 0.14),
                material=MochiShellMaterialOptions(E=6e4, collider_radius=1.2e-2, self_contact=True, **cloth),
                surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
            ),
            **cubes,
        ),
        action_options=ActionManagerOptions(
            ee_controller=DifferentialIKController.configure(
                entity_name="franka",
                ee_link_name="hand",
                dofs_name=FRANKA_ARM_DOFS,
                position_scale=0.05,
                orientation_scale=0.05,
            ),
            gripper_controller=BinaryJointController.configure(
                entity_name="franka",
                dofs_name=FRANKA_FINGER_DOFS,
                open_action=0.04,
                # Stop the pads a few millimeters apart: squeezing further ejects the fold.
                close_action=0.002,
            ),
        ),
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()
    env.reset()
    en.logger.info(env.action_manager.summary())

    device, reset_requested = teleop_device(args.spacemouse)

    while True:
        if reset_requested.is_set():
            reset_requested.clear()
            env.reset()
        command = device.get_command()
        env.step(
            {
                "ee_controller": command[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
                "gripper_controller": command[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            }
        )
        if en.is_smoke():
            break


if __name__ == "__main__":
    main()
