"""Bimanual i2rt YAM teleop with Meta Quest controllers (clutch-relative + IK).

---
tier: recipe
capabilities: [teleop.metaquest, robot.yam]
hardware: cpu
optional_deps: [metaquest]
run: uv run python examples/recipes/teleop/metaquest/controller_teleop.py
---

Each Quest controller drives one 6-DOF YAM arm to pick up / move a few objects on
a table; the triggers drive the parallel grippers. This is the "real arm" build of
the controller stack: instead of teleporting a marker cube (``controller_demo.py``),
the controller pose becomes an end-effector *target* that is tracked through
inverse kinematics, and the mapper's reach limit does real work because the arm
(joint limits, IK, PD tracking) genuinely lags the hand.

The arms are mounted on the table surface with ``gravity_compensation`` so the PD
position controller holds the IK target without gravity sag.

Controls (per hand → per arm; both run independently, so it's truly bimanual):
    * **grip = clutch** — hold a grip and that arm's end-effector follows your
      hand's *relative* motion (position + orientation); release to reposition
      your hand without moving the arm, then grab again,
    * **trigger = gripper** — analog: squeeze to close the fingers, release to open,
    * **A button** (right controller) — reset both arms to the home pose,
    * **X button** (left controller) — reset the objects to their start poses.

Pipeline per engaged hand, per tick:
    controller pose → ClutchPoseMapper.target(.., current EE pose) → reach-limited
    EE target → RigidEntity.inverse_kinematics(link_6) → control_dofs_pos(arm joints)

Controller poses arrive already in the Genesis/robot world frame, and Genesis IK
works in world frame too, so the mapper runs with R = identity ("arm base" = world).

It renders a head-tracked stereo view back to the bridge, like the other demos.

Run (after starting the bridge in controller mode):
    eden metaquest bridge --tracking controller
    uv run python examples/recipes/teleop/metaquest/controller_teleop.py            # GPU
    uv run python examples/recipes/teleop/metaquest/controller_teleop.py -c
"""

import argparse
import time

import eden as en  # must be imported before genesis / torch
import genesis as gs
import numpy as np
import torch
import zmq

from eden.envs.base import EnvBase
from eden.options import BoxOptions, EnvOptions, PlaneOptions, SceneOptions, SphereOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.materials import RigidMaterialOptions
from eden.utils.configs import EdenConfig
from eden.utils.devices.metaquest import protocol as P
from eden.utils.devices.metaquest.pose_mapping import ClutchPoseMapper
from genesis.utils.geom import R_to_quat

# Shared helpers live in the hand-tracking demo (same directory on sys.path[0]).
from demo import aim_cameras, render_rgb_np

EE_LINK = "link_6"  # YAM end-effector link
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGERS = ["left_finger", "right_finger"]
# YAM gripper finger targets (from eden.options.robots.i2rt_yam): open vs closed.
FINGER_OPEN = np.array([0.037524, -0.037524])
FINGER_CLOSED = np.array([0.0, 0.0])

# Scene geometry (metres, Genesis world frame: z up, x forward, y left). The arms
# are mounted ON the table surface (base z = table top); the YAM reaches the whole
# table workspace (~0.25-0.45 m forward, table level up to ~0.4 m) from there.
TABLE_TOP_Z = 0.20
TABLE_POS = (0.30, 0.0, TABLE_TOP_Z / 2)
TABLE_SIZE = (0.70, 0.9, TABLE_TOP_Z)  # spans x in [-0.05, 0.65]; bases sit on the back
ARM_BASES = {"left": (0.0, 0.22, TABLE_TOP_Z), "right": (0.0, -0.22, TABLE_TOP_Z)}
CONTROLLER_TO_ARM = {"left": "left_arm", "right": "right_arm"}

# Objects to interact with, spread across both arms' reach: (name, kind, size, color, pos).
# "size" is the box half-extents tuple or the sphere radius; pos rests them on the table.
OBJECT_SPECS = [
    ("cube", "box", (0.05, 0.05, 0.05), (0.90, 0.70, 0.10), (0.34, 0.00, TABLE_TOP_Z + 0.026)),
    ("block", "box", (0.045, 0.045, 0.09), (0.85, 0.25, 0.20), (0.31, 0.16, TABLE_TOP_Z + 0.046)),
    ("brick", "box", (0.08, 0.04, 0.04), (0.20, 0.45, 0.85), (0.31, -0.16, TABLE_TOP_Z + 0.021)),
    ("ball", "sphere", 0.03, (0.30, 0.75, 0.35), (0.41, 0.06, TABLE_TOP_Z + 0.031)),
]


def _object_options(kind, size, color, pos):
    if kind == "box":
        return BoxOptions(size=size, default_root_pos=pos, color=color)
    return SphereOptions(radius=size, default_root_pos=pos, color=color)


def ctrl_pose(pose_msg, side):
    """Controller world pose (pos (3,), wxyz quat (4,)); None if tracking dropped."""
    mat = np.asarray(pose_msg[f"{side}_wrist"], dtype=float)
    if np.abs(mat).sum() < 1e-6:
        return None
    return mat[:3, 3].copy(), R_to_quat(np.ascontiguousarray(mat[:3, :3]))


def build_env(args):
    """Construct the bimanual scene; return (env, objects-dict, arms-dict, (cam_l, cam_r))."""
    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    # gravity_compensation=1.0 cancels the arms' self-weight so the PD position
    # controller holds the IK target without gravity sag (steady-state droop).
    grav_comp = RigidMaterialOptions(gravity_compensation=1.0)
    objects = {name: _object_options(kind, size, color, pos) for name, kind, size, color, pos in OBJECT_SPECS}

    scene_options = SceneOptions(
        plane=PlaneOptions(),
        table=BoxOptions(
            size=TABLE_SIZE,
            fixed=True,
            default_root_pos=TABLE_POS,
            color=(0.6, 0.45, 0.3),
        ),
        left_arm=en.robots.Yam(default_root_pos=ARM_BASES["left"], material=grav_comp),
        right_arm=en.robots.Yam(default_root_pos=ARM_BASES["right"], material=grav_comp),
        **objects,
    )

    eye_w = args.width // 2
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / args.physics_hz, decimation=1),
        scene_options=scene_options,
        cameras_options=CamerasOptions(
            cam_left=CameraOptions(
                cam_resolution=(eye_w, args.height),
                cam_pos=(-0.45, 0.03, 0.75),
                cam_lookat=(0.35, 0.03, 0.22),
                cam_fov=args.fov,
            ),
            cam_right=CameraOptions(
                cam_resolution=(eye_w, args.height),
                cam_pos=(-0.45, -0.03, 0.75),
                cam_lookat=(0.35, -0.03, 0.22),
                cam_fov=args.fov,
            ),
        ),
    )

    env = EnvBase.from_config(config, show_viewer=args.vis)
    env.build()
    arms = {"left_arm": env.entities["left_arm"], "right_arm": env.entities["right_arm"]}
    objects = {name: env.entities[name] for name, *_ in OBJECT_SPECS}
    cams = (env.cameras["cam_left"], env.cameras["cam_right"])
    return env, objects, arms, cams


class ArmController:
    """Per-arm IK driver: caches DOF/link indices and the home qpos, drives joints."""

    def __init__(self, arm, device):
        self.arm = arm
        self.device = device
        _, self.arm_idx = arm.find_named_dofs_idx_local(ARM_JOINTS, preserve_order=True)
        _, self.finger_idx = arm.find_named_dofs_idx_local(FINGERS, preserve_order=True)
        _, self.ee_idx = arm.find_named_links_idx_local([EE_LINK], preserve_order=True)
        # YAM is fixed-base with only revolute joints, so dofs_pos is the full
        # configuration (get_qpos misindexes when a free body shares the scene).
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
        tp = torch.as_tensor(target_pos, dtype=torch.float32, device=self.device).reshape(1, 3)
        tq = torch.as_tensor(target_quat, dtype=torch.float32, device=self.device).reshape(1, 4)
        # IK self-seeds from the current configuration. Its qpos is dofs_name
        # order; for YAM's serial chain that matches the genesis-tree indices from
        # find_named_dofs_idx_local, so arm_idx both slices the solution and
        # selects the controlled DOFs.
        qpos = self.arm.inverse_kinematics(links_name=EE_LINK, poss=tp, quats=tq, dofs_idx_local=self.arm_idx)
        self.arm.control_dofs_pos(qpos[:, self.arm_idx], dofs_idx_local=self.arm_idx)

    def set_gripper(self, close_frac):
        """Analog gripper: 0 = open, 1 = closed (interpolate finger targets)."""
        f = (1.0 - close_frac) * FINGER_OPEN + close_frac * FINGER_CLOSED
        tgt = torch.as_tensor(f, dtype=torch.float32, device=self.device).reshape(1, 2)
        self.arm.control_dofs_pos(tgt, dofs_idx_local=self.finger_idx)


def reset_objects(objects, starts, device):
    """Teleport every object back to its start pose and zero its velocity."""
    for name, obj in objects.items():
        pos, quat = starts[name]
        obj.set_pos(torch.as_tensor(pos, dtype=torch.float32, device=device).reshape(1, 3))
        obj.set_quat(torch.as_tensor(quat, dtype=torch.float32, device=device).reshape(1, 4), relative=False)
        obj.set_dofs_vel()  # zero the free body's linear+angular velocity


def main():
    parser = argparse.ArgumentParser(description="Bimanual YAM Quest-controller teleop (Eden side)")
    parser.add_argument("--width", type=int, default=1280, help="full stereo frame width (both eyes)")
    parser.add_argument("--height", type=int, default=480, help="frame height")
    parser.add_argument("--fov", type=float, default=90.0, help="vertical fov per eye (deg)")
    parser.add_argument("--fps", type=float, default=30.0, help="streaming fps")
    parser.add_argument("--physics-hz", type=float, default=200.0, help="physics dt = 1/physics_hz")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("--bridge-host", default="127.0.0.1", help="host running the XR bridge")
    parser.add_argument("-v", "--vis", action="store_true", help="also open the local Genesis viewer")
    args = parser.parse_args()

    env, objects, arms, (cam_left, cam_right) = build_env(args)
    device = env.device
    object_starts = {
        name: (np.array(pos, dtype=float), np.array([1.0, 0.0, 0.0, 0.0])) for name, *_, pos in OBJECT_SPECS
    }

    controllers = {side: ArmController(arms[name], device) for side, name in CONTROLLER_TO_ARM.items()}
    for ctl in controllers.values():  # hold the home pose until the operator clutches
        ctl.hold()
        ctl.set_gripper(0.0)

    ctx = zmq.Context.instance()
    frame_sock = P.frame_push(ctx, host=args.bridge_host)
    pose_sock = P.pose_sub(ctx, host=args.bridge_host)

    en.logger.info(
        f"streaming {args.width}x{args.height} stereo to {args.bridge_host}; subscribing to controller state. "
        f"Hold a GRIP to clutch that arm; TRIGGER closes its gripper; A homes both arms; X resets the objects."
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

            # A (right controller) homes the arms; X (left controller) resets the objects.
            if last_pose.get("right_a_button"):
                for ctl in controllers.values():
                    ctl.reset_home()
            if last_pose.get("left_a_button"):
                reset_objects(objects, object_starts, device)

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
            if last_pose and last_pose.get("ready"):
                clutched = [s for s, c in controllers.items() if c.mapper.engaged] or ["—"]
                en.logger.info(
                    f"frames: {frames} | clutched: {','.join(clutched)} | "
                    f"L grip={last_pose['left_squeeze_value']:.2f} trig={last_pose['left_trigger_value']:.2f} | "
                    f"R grip={last_pose['right_squeeze_value']:.2f} trig={last_pose['right_trigger_value']:.2f}"
                )
            else:
                en.logger.info(f"frames sent: {frames} | headset tracking: False")
            last_log = now

        dt = time.time() - loop_start
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()
