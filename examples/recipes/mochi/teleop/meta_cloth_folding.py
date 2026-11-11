"""Bimanual YAM t-shirt folding in Mochi, teleoperated with Meta Quest controllers.

---
tier: recipe
capabilities: [solver.mochi, robot.yam, teleop.metaquest]
hardware: gpu-fp64
optional_deps: [metaquest]
run: uv run python examples/recipes/mochi/teleop/meta_cloth_folding.py
---

The Mochi build of ``examples/recipes/teleop/metaquest/controller_teleop.py``: the same controller
stack (Quest controller pose -> ``ClutchPoseMapper`` -> inverse kinematics -> joint targets, the
triggers on the grippers, a head-tracked stereo view streamed back to the headset) driving a task
the rigid solver cannot do. The garment is the 3593-node shell of ``../cloth_tshirt.py``; the two
arms, the cloth, its self-contact and the penalty contact all enter one implicit Newton solve per
step, so a gripper can pinch a shoulder, drag it across the body and have the fold land on cloth
and stay there.

A Mochi scene is all-Mochi, which shapes the options here:

* every entity carries a Mochi material, the ground plane and the table included;
* ``en.robots.Yam`` ships ``default_dofs_force_limits``, which Eden rejects on a Mochi material
  (the solver clamps no drive force), so the recipe clears it;
* ``gravity_compensation`` is a *rigid* material field that ``Mochi.Rigid`` refuses. Its Mochi
  equivalent is the per-body ``has_gravity=False``, and that is what keeps the PD-tracked arms
  from sagging below their IK targets.

The mesh stands in the x-z plane, a metre across, with its origin at a corner;
:func:`flat_shirt_mesh` scales it, lays it flat with the shoulder line along y (one shoulder per
arm) and centres it, so ``default_root_pos`` is simply where the garment goes. It is dropped a
centimetre above the table and settles for ``--settle`` steps before the teleop loop opens; that
settled state is what the X button restores.

Controls (per hand -> per arm, both independent, so it is genuinely bimanual):

* **grip = clutch** — hold and that arm's end-effector follows your hand's *relative* motion;
  release to reposition your hand without moving the arm,
* **trigger = gripper** — analog; a full squeeze stops the pads ``FINGER_PINCH`` apart, which
  pinches cloth. Closing them the whole way squeezes the shell out from between the pads,
* **A** (right controller) — home both arms,
* **X** (left controller) — drop the garment back to its settled state.

Every five seconds it logs the garment's footprint (the area of its xy bounding box) and its height
above the table: folding one side across the body takes the footprint down by about a quarter and
raises the garment by a couple of centimetres.

It is not a real-time scene. Measured on an RTX 3080 with the garment settled: 90 ms per step at the
default four Newton iterations, 51 ms at two, plus 22 ms for the stereo pair — so the loop runs near
9 fps and the sim in slow motion. The garment is what costs; the two arms are worth about 3 ms of it.
``--newton-iters 2`` (mochi's own setting for this t-shirt) buys responsiveness at the price of a
less tightly converged pinch, and a smaller ``--width`` / ``--height`` makes the render cheaper.

Run (after starting the bridge in controller mode):
    eden metaquest bridge --tracking controller
    uv run python examples/recipes/mochi/teleop/meta_cloth_folding.py
    uv run python examples/recipes/mochi/teleop/meta_cloth_folding.py -c
"""

import argparse
import os
import sys
import time

import genesis as gs
import numpy as np
import torch
import trimesh
import zmq
from genesis.utils.geom import R_to_quat

import eden as en  # must be imported before genesis / torch
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import BoxOptions, EntityOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiShellMaterialOptions
from eden.utils.assets import get_asset_path
from eden.utils.procedural_meshes import derived_file_path, write_atomically
from eden.utils.devices.metaquest import protocol as P
from eden.utils.devices.metaquest.pose_mapping import ClutchPoseMapper

# Neither helper module is a sibling of this script: the Mochi conventions (fp64 init, the fork's
# gravity) live one directory up, the XR stereo helpers with the Quest recipes.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), os.path.join(os.path.dirname(_HERE), "..", "teleop", "metaquest")]

from _mochi_common import MOCHI_GRAVITY, init, newton_iterations  # noqa: E402
from demo import aim_cameras, render_rgb_np  # noqa: E402

EE_LINK = "link_6"  # YAM end-effector link
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGERS = ["left_finger", "right_finger"]
# YAM gripper finger targets (from eden.options.robots.i2rt_yam). The closed pose is the pinch of
# ``../cloth_teleop.py`` rather than the robot's own 0.0: a shell has no volume to arrest the pads,
# so jaws driven together squeeze the cloth out from between them instead of holding it.
FINGER_OPEN = np.array([0.037524, -0.037524])
FINGER_PINCH = np.array([0.004, -0.004])

# Scene geometry (metres, Genesis world frame: z up, x forward, y left). The arms are mounted on the
# table surface, as in the rigid controller recipe; both they and the table are fixed, so the contact
# where a base rests on the top is inert.
TABLE_TOP_Z = 0.20
TABLE_SIZE = (0.60, 0.90, TABLE_TOP_Z)
TABLE_POS = (0.30, 0.0, TABLE_TOP_Z / 2)  # spans x in [0, 0.6]
ARM_BASES = {"left_arm": (0.0, 0.22, TABLE_TOP_Z), "right_arm": (0.0, -0.22, TABLE_TOP_Z)}
CONTROLLER_TO_ARM = {"left": "left_arm", "right": "right_arm"}

SHIRT_MESH = "objects/mochi/tshirt_3593.obj"  # in the Kashu7100/eden_assets Hub dataset
# 0.42 of the benchmark garment: 0.42 m across, so a shoulder sits under each arm, on a 0.6 m table.
SHIRT_SCALE = 0.42
SHIRT_XY = (0.30, 0.0)
SHIRT_DROP = 0.01  # released this far above the table, then left to settle
# Self-contact spheres at the vertices. The benchmark's 1.5 cm is for the full-size shirt; this is
# the same fraction of the garment, and still under the ~8 mm edge length of the scaled mesh.
SHIRT_COLLIDER_RADIUS = 6e-3


def flat_shirt_mesh(scale: float) -> str:
    """Write the benchmark t-shirt scaled, laid flat and centred on the origin; return the path.

    Baking the transform into the mesh keeps it out of the morph, where a rotation is taken about the
    mesh origin — a corner of this garment — and so would displace it as well as turn it.
    """
    source = get_asset_path(SHIRT_MESH)
    mesh = trimesh.load(source, process=False)
    verts = np.asarray(mesh.vertices, dtype=float) * scale
    # (x, y, z) -> (-z, x, -y): the garment plane comes down onto the table, shoulder line along y.
    # A proper rotation, so the winding stays consistent, as Mochi shells require.
    verts = np.stack([-verts[:, 2], verts[:, 0], -verts[:, 1]], axis=-1)
    verts[:, :2] -= 0.5 * (verts[:, :2].min(axis=0) + verts[:, :2].max(axis=0))
    verts[:, 2] -= verts[:, 2].min()
    path = derived_file_path("procedural_meshes", source, f"tshirt_flat_{scale:g}.obj")
    write_atomically(
        path, trimesh.exchange.obj.export_obj(trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False))
    )
    return path


def ctrl_pose(pose_msg, side):
    """Controller world pose (pos (3,), wxyz quat (4,)); None if tracking dropped."""
    mat = np.asarray(pose_msg[f"{side}_wrist"], dtype=float)
    if np.abs(mat).sum() < 1e-6:
        return None
    return mat[:3, 3].copy(), R_to_quat(np.ascontiguousarray(mat[:3, :3]))


def build_env(args):
    """Construct the all-Mochi folding scene; return (env, shirt, arms, (cam_left, cam_right))."""
    init(args)  # fp64: Mochi's 1e9 Pa/m default contact stiffness needs it

    arms = {
        name: en.robots.Yam(
            default_root_pos=pos,
            # The MochiSolver clamps no drive force, so Eden rejects the robot's own force limits.
            default_dofs_force_limits={},
            material=MochiRigidMaterialOptions(friction=1.5, viscous_friction=1.0, has_gravity=False),
        )
        for name, pos in ARM_BASES.items()
    }
    eye_w = args.width // 2
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / args.physics_hz,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=args.newton_iters),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            table=BoxOptions(
                size=TABLE_SIZE,
                default_root_pos=TABLE_POS,
                is_fixed_base=True,
                material=MochiRigidMaterialOptions(friction=0.8),
                surface=gs.surfaces.Plastic(color=(0.60, 0.45, 0.30)),
            ),
            shirt=EntityOptions(
                file=flat_shirt_mesh(SHIRT_SCALE),
                default_root_pos=(*SHIRT_XY, TABLE_TOP_Z + SHIRT_DROP),
                material=MochiShellMaterialOptions(
                    E=6e4,
                    nu=0.3,
                    rho=200.0,
                    thickness=5e-4,
                    friction=0.2,
                    # Light cloth takes a softer contact than the 1e9 default (see the mochi README).
                    penalty_coefficient=1e7,
                    collider_radius=SHIRT_COLLIDER_RADIUS,
                    self_contact=True,
                ),
                # surface=gs.surfaces.Plastic(color=(0.85, 0.30, 0.35, 1.0)),
            ),
            **arms,
        ),
        cameras_options=CamerasOptions(
            cam_left=CameraOptions(
                cam_resolution=(eye_w, args.height),
                cam_pos=(-0.45, 0.03, 0.75),
                cam_lookat=(*SHIRT_XY, TABLE_TOP_Z),
                cam_fov=args.fov,
            ),
            cam_right=CameraOptions(
                cam_resolution=(eye_w, args.height),
                cam_pos=(-0.45, -0.03, 0.75),
                cam_lookat=(*SHIRT_XY, TABLE_TOP_Z),
                cam_fov=args.fov,
            ),
        ),
        show_viewer=args.vis,
    )
    env.build()
    arms = {name: env.entities[name] for name in ARM_BASES}
    cams = (env.cameras["cam_left"], env.cameras["cam_right"])
    return env, env.entities["shirt"], arms, cams


class ArmController:
    """Per-arm IK driver: caches DOF/link indices and the home qpos, drives joints."""

    def __init__(self, arm, device):
        self.arm = arm
        self.device = device
        _, self.arm_idx = arm.find_named_dofs_idx_local(ARM_JOINTS, preserve_order=True)
        _, self.finger_idx = arm.find_named_dofs_idx_local(FINGERS, preserve_order=True)
        _, self.ee_idx = arm.find_named_links_idx_local([EE_LINK], preserve_order=True)
        # YAM is fixed-base with only revolute joints, so dofs_pos is the full configuration
        # (get_qpos misindexes when a free body shares the scene).
        self.home_dofs = arm.get_dofs_pos().clone()
        self.mapper = ClutchPoseMapper()  # R = identity: poses already in world frame

    def ee_pose(self):
        """Current end-effector world pose as numpy (pos (3,), wxyz quat (4,))."""
        pos = self.arm.get_links_pos(ls_idx_local=self.ee_idx)[0, 0].detach().cpu().numpy()
        quat = self.arm.get_links_quat(ls_idx_local=self.ee_idx)[0, 0].detach().cpu().numpy()
        return pos.astype(float), quat.astype(float)

    def hold(self):
        """Command the arm to hold its current measured pose (PD setpoint = now)."""
        self.arm.control_dofs_pos(self.arm.get_dofs_pos(dofs_idx_local=self.arm_idx), dofs_idx_local=self.arm_idx)

    def reset_home(self):
        self.arm.set_dofs_pos(self.home_dofs)
        self.arm.control_dofs_pos(self.home_dofs[:, self.arm_idx], dofs_idx_local=self.arm_idx)
        self.mapper.disengage()

    def drive_to(self, target_pos, target_quat):
        """Solve IK to the world-frame EE target and command the arm joints."""
        tp = torch.as_tensor(target_pos, dtype=gs.tc_float, device=self.device).reshape(1, 3)
        tq = torch.as_tensor(target_quat, dtype=gs.tc_float, device=self.device).reshape(1, 4)
        # IK self-seeds from the current configuration. Its qpos is dofs_name order; for YAM's serial
        # chain that matches the genesis-tree indices from find_named_dofs_idx_local, so arm_idx both
        # slices the solution and selects the controlled DOFs.
        qpos = self.arm.inverse_kinematics(links_name=EE_LINK, poss=tp, quats=tq, dofs_idx_local=self.arm_idx)
        self.arm.control_dofs_pos(qpos[:, self.arm_idx], dofs_idx_local=self.arm_idx)

    def set_gripper(self, close_frac):
        """Analog gripper: 0 = open, 1 = pinched shut on cloth (interpolate finger targets)."""
        f = (1.0 - close_frac) * FINGER_OPEN + close_frac * FINGER_PINCH
        tgt = torch.as_tensor(f, dtype=gs.tc_float, device=self.device).reshape(1, 2)
        self.arm.control_dofs_pos(tgt, dofs_idx_local=self.finger_idx)


def fold_metrics(shirt):
    """Garment footprint (area of the xy bounding box, m^2) and height above the table (m)."""
    verts = shirt.get_vertex_positions()[0]
    lo, hi = verts.amin(dim=0), verts.amax(dim=0)
    span = (hi - lo).tolist()
    return span[0] * span[1], float(hi[2]) - TABLE_TOP_Z


def settle(env, shirt, controllers, steps):
    """Let the garment drape onto the table with the arms holding, and return the settled state."""
    for ctl in controllers.values():
        ctl.hold()
        ctl.set_gripper(0.0)
    for _ in range(steps):
        env.scene.step()
    footprint, height = fold_metrics(shirt)
    en.logger.info(f"garment settled after {steps} steps: footprint {footprint:.3f} m^2, height {height * 100:.1f} cm")
    return shirt.get_vertex_positions().clone()


def main():
    parser = argparse.ArgumentParser(description="Bimanual YAM Mochi cloth folding, Quest-controller teleop")
    parser.add_argument("--width", type=int, default=1280, help="full stereo frame width (both eyes)")
    parser.add_argument("--height", type=int, default=480, help="frame height")
    parser.add_argument("--fov", type=float, default=90.0, help="vertical fov per eye (deg)")
    parser.add_argument("--fps", type=float, default=30.0, help="streaming fps")
    parser.add_argument("--physics-hz", type=float, default=50.0, help="physics dt = 1/physics_hz")
    parser.add_argument("--settle", type=int, default=90, help="steps the garment drapes before teleop opens")
    parser.add_argument(
        "--newton-iters",
        type=int,
        default=4,
        help="Newton iterations per step: 4 holds a pinch, 2 nearly doubles the loop rate",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("--bridge-host", default="127.0.0.1", help="host running the XR bridge")
    parser.add_argument("-v", "--vis", action="store_true", help="also open the local Genesis viewer")
    args = parser.parse_args()

    env, shirt, arms, (cam_left, cam_right) = build_env(args)
    device = env.device

    controllers = {side: ArmController(arms[name], device) for side, name in CONTROLLER_TO_ARM.items()}
    settled_verts = settle(env, shirt, controllers, args.settle)

    ctx = zmq.Context.instance()
    frame_sock = P.frame_push(ctx, host=args.bridge_host)
    pose_sock = P.pose_sub(ctx, host=args.bridge_host)

    en.logger.info(
        f"streaming {args.width}x{args.height} stereo to {args.bridge_host}; subscribing to controller state. "
        f"Hold a GRIP to clutch that arm; TRIGGER pinches its gripper; A homes both arms; X re-drops the shirt."
    )

    period = 1.0 / max(args.fps, 1.0)
    last_pose = None
    frames = 0
    last_log = time.time()

    while True:
        loop_start = time.time()

        pose = P.recv_pose(pose_sock, flags=zmq.NOBLOCK)
        if pose is not None:
            if pose.get("mode") != "controller":
                en.logger.warning(
                    "received a non-controller pose stream; start the bridge with "
                    "`eden metaquest bridge --tracking controller`."
                )
            last_pose = pose

        if last_pose is not None and last_pose.get("ready"):
            aim_cameras(cam_left, cam_right, last_pose["head"])

            # A (right controller) homes the arms; X (left controller) re-drops the garment.
            if last_pose.get("right_a_button"):
                for ctl in controllers.values():
                    ctl.reset_home()
            if last_pose.get("left_a_button"):
                shirt.set_vertex_positions(settled_verts)
                shirt.set_vertex_velocities((0.0, 0.0, 0.0))

            for side, ctl in controllers.items():
                # Gripper tracks the trigger every tick (works clutched or not).
                ctl.set_gripper(float(last_pose[f"{side}_trigger_value"]))

                cp = ctrl_pose(last_pose, side)
                grip = bool(last_pose[f"{side}_squeeze"]) and cp is not None
                ee_pos, ee_quat = ctl.ee_pose()
                if grip and not ctl.mapper.engaged:
                    ctl.mapper.engage(cp[0], cp[1], ee_pos, ee_quat)  # rising clutch edge
                elif not grip and ctl.mapper.engaged:
                    ctl.mapper.disengage()

                if ctl.mapper.engaged:
                    out = ctl.mapper.target(cp[0], cp[1], ee_pos, ee_quat)  # reach-limited vs current EE
                    if out is not None:
                        ctl.drive_to(out[0], out[1])
                # disengaged: leave the last PD setpoint in place (arm holds).

        env.scene.step()

        stereo = np.concatenate([render_rgb_np(cam_left), render_rgb_np(cam_right)], axis=1)
        P.send_frame(frame_sock, stereo)
        frames += 1

        now = time.time()
        if now - last_log > 5.0:
            footprint, height = fold_metrics(shirt)
            if last_pose and last_pose.get("ready"):
                clutched = [s for s, c in controllers.items() if c.mapper.engaged] or ["—"]
                en.logger.info(
                    f"frames: {frames} | clutched: {','.join(clutched)} | "
                    f"footprint {footprint:.3f} m^2 height {height * 100:4.1f} cm | "
                    f"newton iterations={newton_iterations(env)}"
                )
            else:
                en.logger.info(f"frames sent: {frames} | headset tracking: False")
            last_log = now

        dt = time.time() - loop_start
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()
