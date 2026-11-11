"""Eden-side Meta Quest XR hand-tracking teleop demo (built on Eden's env/config API).

---
tier: recipe
capabilities: [teleop.metaquest]
hardware: cpu
optional_deps: [metaquest]
run: uv run python examples/recipes/teleop/metaquest/demo.py
---

Builds a small Eden environment (``EnvBase.from_config`` over Genesis), then:
    * receives Quest head / wrist / hand poses over ZMQ (from ``eden metaquest
      bridge``),
    * renders the 25 tracked joints of each hand as small spheres,
    * grabs a cube when you pinch near it,
    * renders a stereo camera view that follows the headset and streams it back
      to the bridge for display in the Quest.

This is the Eden port of the original Genesis-direct ``eden_demo.py``: it uses
Eden options (``SceneOptions`` / ``BoxOptions`` / ``SphereOptions`` /
``CameraOptions``) and the wrapped ``env.entities`` / ``env.cameras`` API instead
of touching ``gs.Scene`` directly, and it imports the shared wire protocol from
``eden.utils.devices.metaquest.protocol`` (the same one the bridge uses) rather
than a local copy.

Poses arrive already in the Genesis/robot world frame (z up, x forward, y left,
metres, origin at the XR tracking origin), so no further basis conversion is
needed here. Because the head pose is in that frame, its local axes are the
robot-convention axes: local +X = forward (gaze), +Z = up, +Y = left.

To teleoperate a *real* Eden task instead of this toy scene, replace
``build_env()`` with a registered task --- ``RLEnvBase.from_config(cfg)`` --- grab
its ``env.cameras`` (add a stereo pair via ``cameras_options``), and keep the
same pose->action / render->stream loop.

Run (after starting the bridge with ``eden metaquest bridge``):
    uv run python examples/recipes/teleop/metaquest/demo.py            # GPU
    uv run python examples/recipes/teleop/metaquest/demo.py -c
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
from eden.utils.configs import EdenConfig
from eden.utils.devices.metaquest import protocol as P

IPD = 0.063  # interpupillary distance (m) -> stereo camera separation
GRAB_DIST = 0.06  # pinch-point-to-cube distance (m) under which a pinch grabs the cube
N_HAND_JOINTS = 25
# WebXR hand-joint indices (immersive-web XRHand): 0 wrist, 4 thumb-tip, 9 index-tip.
WRIST_IDX, THUMB_TIP_IDX, INDEX_TIP_IDX = 0, 4, 9


def head_axes(mat):
    """Return (forward, up, right) world directions from a robot-frame head pose.

    The pose is already in the Genesis/robot frame, so the head's neutral local
    axes are: +X forward, +Z up, +Y left (hence rightward = -Y). Each axis is just
    a column of the rotation block (forward = col 0, up = col 2, right = -col 1).
    """
    r = np.asarray(mat, dtype=float)[:3, :3]
    return r[:, 0], r[:, 2], -r[:, 1]


def build_env(args):
    """Construct the toy teleop env; return (env, cube, joints, (cam_left, cam_right))."""
    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    # Hand joints are fixed (kinematic) spheres we reposition each frame via set_pos.
    hands = {}
    for side, color in (("lhand", (0.2, 0.4, 1.0)), ("rhand", (1.0, 0.3, 0.3))):
        for i in range(N_HAND_JOINTS):
            radius = 0.014 if i == WRIST_IDX else 0.008
            hands[f"{side}_{i:02d}"] = SphereOptions(
                radius=radius,
                fixed=True,
                default_root_pos=(0.3, 0.0, 0.9 + i * 0.001),
                color=color,
            )

    scene_options = SceneOptions(
        plane=PlaneOptions(),
        table=BoxOptions(
            size=(0.4, 0.6, 0.7),
            fixed=True,
            default_root_pos=(0.45, 0.0, 0.35),
            color=(0.6, 0.45, 0.3),
        ),
        cube=BoxOptions(
            size=(0.05, 0.05, 0.05),
            default_root_pos=(0.45, 0.0, 0.75),
            color=(0.9, 0.7, 0.1),
        ),
        **hands,
    )

    eye_w = args.width // 2
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / args.physics_hz, decimation=1),
        scene_options=scene_options,
        cameras_options=CamerasOptions(
            cam_left=CameraOptions(
                cam_resolution=(eye_w, args.height),
                cam_pos=(0.0, 0.03, 1.6),
                cam_lookat=(1.0, 0.03, 1.6),
                cam_fov=args.fov,
            ),
            cam_right=CameraOptions(
                cam_resolution=(eye_w, args.height),
                cam_pos=(0.0, -0.03, 1.6),
                cam_lookat=(1.0, -0.03, 1.6),
                cam_fov=args.fov,
            ),
        ),
    )

    env = EnvBase.from_config(config, show_viewer=args.vis)
    env.build()

    joints = {
        "left": [env.entities[f"lhand_{i:02d}"] for i in range(N_HAND_JOINTS)],
        "right": [env.entities[f"rhand_{i:02d}"] for i in range(N_HAND_JOINTS)],
    }
    cams = (env.cameras["cam_left"], env.cameras["cam_right"])
    return env, env.entities["cube"], joints, cams


def aim_cameras(cam_left, cam_right, head_mat):
    """Place the two eye cameras at the head pose, offset by IPD along head-right."""
    head = np.asarray(head_mat, dtype=float)
    pos = head[:3, 3]
    forward, up, right = head_axes(head)
    left_pos = pos - right * (IPD / 2.0)
    right_pos = pos + right * (IPD / 2.0)
    cam_left.set_pose(pos=left_pos, lookat=left_pos + forward, up=up)
    cam_right.set_pose(pos=right_pos, lookat=right_pos + forward, up=up)


def set_world_pos(entity, p, device):
    """Teleport a (num_envs=1) entity to world point ``p`` (array-like, shape (3,))."""
    entity.set_pos(torch.as_tensor(p, dtype=torch.float32, device=device).reshape(1, 3))


def update_hand(joints, positions, device):
    """Move the 25 joint spheres; skip if tracking momentarily dropped (all zeros)."""
    pts = np.asarray(positions, dtype=float)
    if pts.shape != (N_HAND_JOINTS, 3) or np.abs(pts).sum() < 1e-6:
        return None
    for sphere, p in zip(joints, pts):
        set_world_pos(sphere, p, device)
    return pts


def render_rgb_np(cam):
    """Render one camera to a contiguous (H, W, 3) uint8 numpy array."""
    rgb = cam.render_rgb()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:  # (num_envs, H, W, C) -> single env
        rgb = rgb[0]
    return np.ascontiguousarray(rgb[..., :3].astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Eden <-> Quest XR teleop demo (Eden side)")
    parser.add_argument("--width", type=int, default=1280, help="full stereo frame width (both eyes)")
    parser.add_argument("--height", type=int, default=480, help="frame height")
    parser.add_argument("--fov", type=float, default=90.0, help="vertical fov per eye (deg)")
    parser.add_argument("--fps", type=float, default=30.0, help="streaming fps")
    parser.add_argument("--physics-hz", type=float, default=100.0, help="physics dt = 1/physics_hz")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("--bridge-host", default="127.0.0.1", help="host running the XR bridge")
    parser.add_argument("-v", "--vis", action="store_true", help="also open the local Genesis viewer")
    args = parser.parse_args()

    env, cube, joints_for, (cam_left, cam_right) = build_env(args)
    device = env.device

    ctx = zmq.Context.instance()
    frame_sock = P.frame_push(ctx, host=args.bridge_host)
    pose_sock = P.pose_sub(ctx, host=args.bridge_host)

    en.logger.info(
        f"streaming {args.width}x{args.height} stereo to {args.bridge_host}; "
        f"subscribing to poses. Pinch (thumb+index) near the cube to grab it."
    )

    grabbed = {"left": False, "right": False}
    period = 1.0 / max(args.fps, 1.0)
    last_pose = None
    frames = 0
    last_log = time.time()

    while True:
        loop_start = time.time()

        pose = P.recv_pose(pose_sock, flags=zmq.NOBLOCK)
        if pose is not None:
            last_pose = pose

        if last_pose is not None and last_pose.get("ready"):
            aim_cameras(cam_left, cam_right, last_pose["head"])

            cube_pos = cube.get_pos()[0].detach().cpu().numpy()[:3]
            for side in ("left", "right"):
                pts = update_hand(joints_for[side], last_pose[f"{side}_hand_pos"], device)
                if pts is None:
                    continue
                # Pinch point = midpoint of thumb tip and index tip.
                pinch_pt = (pts[THUMB_TIP_IDX] + pts[INDEX_TIP_IDX]) / 2.0
                pinching = bool(last_pose[f"{side}_pinch"])
                if not pinching:
                    grabbed[side] = False
                elif not any(grabbed.values()) and np.linalg.norm(pinch_pt - cube_pos) < GRAB_DIST:
                    grabbed[side] = True
                if grabbed[side]:
                    set_world_pos(cube, pinch_pt, device)

        env.scene.step()

        stereo = np.concatenate([render_rgb_np(cam_left), render_rgb_np(cam_right)], axis=1)
        P.send_frame(frame_sock, stereo)
        frames += 1

        now = time.time()
        if now - last_log > 5.0:
            tracking = bool(last_pose and last_pose.get("ready"))
            en.logger.info(f"frames sent: {frames} | headset tracking: {tracking}")
            last_log = now

        dt = time.time() - loop_start
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()
