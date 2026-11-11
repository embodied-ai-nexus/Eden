"""Bimanual i2rt YAM teleop with PICO controllers (clutch-relative + IK).

---
tier: recipe
capabilities: [teleop.pico, robot.yam]
hardware: cpu
optional_deps: [pico]
run: uv run python examples/recipes/teleop/pico/controller_teleop.py
---

The PICO port of the Meta Quest ``controller_teleop.py``: each controller drives
one 6-DOF YAM arm to pick up / move a few objects on a table, and the triggers
drive the parallel grippers. This is the "real arm" build of the PICO stack —
instead of teleporting the marker cube (``demo.py``), the controller pose becomes
an end-effector *target* tracked through inverse kinematics, so the mapper's reach
limit does real work because the arm (joint limits, IK, PD tracking) genuinely
lags the hand.

The arms are mounted on the table surface with ``gravity_compensation`` so the PD
position controller holds the IK target without gravity sag.

Control frame: unlike ``demo.py`` (whose cube *translation* follows your gaze),
the arm **mirrors your hand** — the end-effector tracks the controller's own
motion in the world (``ClutchPoseMapper`` with ``R = identity``), which is what
you want when a real arm should copy your hand rather than move relative to the
view. Only the camera you see through follows your head (or the ``--viewer-camera``
desktop view).

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

It renders a head-tracked stereo view back to the bridge, like ``demo.py``;
``--viewer-camera`` uses the interactive Genesis viewer as the camera instead (for
driving from a screen — the controllers still come from the PICO).

Run (after starting the bridge):
    eden pico bridge
    uv run python examples/recipes/teleop/pico/controller_teleop.py            # GPU
    uv run python examples/recipes/teleop/pico/controller_teleop.py --backend cpu
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

# The PICO bridge shares the Meta Quest ZMQ protocol and controller schema.
from eden.utils.devices.metaquest import protocol as P
from eden.utils.devices.metaquest.pose_mapping import ClutchPoseMapper
from genesis.utils.geom import R_to_quat

# Shared helpers live in the PICO hand demo (same directory on sys.path[0]).
from demo import aim_cameras, render_rgb_np, viewer_head_world

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
    en.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu)

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

    env = EnvBase.from_config(config, show_viewer=args.viewer)
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
        # R = identity: the EE mirrors the controller's own world motion (hand-following),
        # which is what you want for a real arm — see the module docstring.
        self.mapper = ClutchPoseMapper()

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
    ap = argparse.ArgumentParser(description="Bimanual YAM PICO-controller teleop (Eden side)")
    ap.add_argument("--width", type=int, default=1280, help="full stereo frame width (both eyes)")
    ap.add_argument("--height", type=int, default=480, help="frame height")
    ap.add_argument("--fov", type=float, default=90.0, help="vertical fov per eye (deg)")
    ap.add_argument("--fps", type=float, default=30.0, help="streaming fps")
    ap.add_argument("--physics-hz", type=float, default=200.0, help="physics dt = 1/physics_hz")
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"], help="Genesis compute backend")
    ap.add_argument("--bridge-host", default="127.0.0.1", help="host running the PICO bridge")
    ap.add_argument("--viewer", action="store_true", help="also open the local Genesis viewer")
    ap.add_argument(
        "--viewer-camera",
        action="store_true",
        help="use the interactive viewer camera (orbit with the mouse) as the streamed view "
        "instead of the headset — for driving from a screen (implies --viewer)",
    )
    args = ap.parse_args()
    if args.viewer_camera:
        args.viewer = True  # the viewer must be open to read its camera

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
        f"streaming {args.width}x{args.height} stereo to {args.bridge_host}; subscribing to PICO controller state. "
        f"Hold a GRIP to clutch that arm; TRIGGER closes its gripper; A homes both arms; X resets the objects."
    )

    period = 1.0 / max(args.fps, 1.0)
    last_pose = None
    frames = 0
    last_log = time.time()
    warned_device = False
    remaining = 5 if en.is_smoke() else -1

    while remaining != 0:
        loop_start = time.time()

        pose = P.recv_pose(pose_sock, flags=zmq.NOBLOCK)
        if pose is not None:
            if pose.get("device") != "pico" and not warned_device:
                en.logger.warning(f"pose stream reports device={pose.get('device')!r}; expected `eden pico bridge`.")
                warned_device = True
            last_pose = pose

        if last_pose is not None and last_pose.get("ready"):
            # Camera the operator sees through: the desktop viewer, else the headset.
            head_world = viewer_head_world(env.scene) if args.viewer_camera else last_pose["head"]
            aim_cameras(cam_left, cam_right, head_world)

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
        remaining -= 1 if remaining > 0 else 0

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
