"""Eden-side Meta Quest *controller* teleop demo with clutch-relative mapping.

---
tier: recipe
capabilities: [teleop.metaquest]
hardware: cpu
optional_deps: [metaquest]
run: uv run python examples/recipes/teleop/metaquest/controller_demo.py
---

The hand-tracking sibling (``demo.py``) tracks 25 finger joints and grabs on a
pinch. This demo consumes the **motion-controller** stream published by
``eden metaquest bridge --tracking controller`` and drives a cube — standing in
for a robot end-effector — through ``ClutchPoseMapper`` (the clutch-relative
mapper ported from vr-teleop-kit). It exercises every part of the controller
state:

    * **pose**     — a marker sphere follows each controller (blue = left, red =
      right) with a tip showing its forward (-Z grip) direction,
    * **grip = clutch** — hold a grip and the cube follows that hand's *relative*
      motion (position + orientation); release to reposition your hand without
      moving the cube, then grab again. Only one hand drives at a time.
    * **trigger = precision** — hold to scale the response down for fine work,
    * **A button** — (either controller) sends the cube back to its home pose.

To make the mapper's **reach limit** ("slipping clutch") visible in sim, the
cube chases the mapper's target at a capped linear/angular speed — like a real
arm's velocity limit. Flick your hand fast and the target runs ahead, hits the
reach limit, and the excess is absorbed: reverse and the cube bites immediately
instead of retracing the overshoot. (With a real arm you pass its measured EE
pose instead of this simulated follower.)

It renders a stereo camera view that follows the headset and streams it back to
the bridge, exactly like ``demo.py``; the shared helpers (camera aiming,
rendering, world-space teleport) are imported from ``demo``.

Controller poses arrive already in the Genesis/robot world frame (z up, x
forward, y left, metres), so the mapper's XR->arm rotation R is identity here.

Run (after starting the bridge in controller mode):
    eden metaquest bridge --tracking controller
    uv run python examples/recipes/teleop/metaquest/controller_demo.py            # GPU
    uv run python examples/recipes/teleop/metaquest/controller_demo.py -c
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
from eden.utils.devices.metaquest.pose_mapping import ClutchPoseMapper
from genesis.utils.geom import (
    R_to_quat,
    inv_quat,
    quat_to_rotvec,
    rotvec_to_quat,
    transform_quat_by_quat,
)

# Shared helpers live in the hand-tracking demo (same directory on sys.path[0]).
from demo import aim_cameras, render_rgb_np, set_world_pos

CUBE_HOME_POS = (0.45, 0.0, 0.75)  # cube reset position (A button)
CUBE_HOME_QUAT = (1.0, 0.0, 0.0, 0.0)  # cube reset orientation (wxyz)
TIP_LEN = 0.08  # length (m) of the forward-direction tip marker off each controller
PRECISION_SCALE = 0.25  # response gain while the trigger (precision modifier) is held
EE_MAX_LIN = 0.75  # m/s cap on how fast the cube chases the target (simulated arm lag)
EE_MAX_ANG = 3.0  # rad/s cap on the cube's angular tracking


def ctrl_axes(mat):
    """Return the controller's forward world direction from its robot-frame pose.

    The grip pose is already in the Genesis/robot frame. WebXR grip space points
    "into the handle" along local -Z, so forward = -col 2 of the rotation block.
    """
    r = np.asarray(mat, dtype=float)[:3, :3]
    return -r[:, 2]


def mat_to_pos_quat(mat):
    """4x4 robot-frame pose -> (pos (3,), wxyz quat (4,)). None if tracking dropped."""
    mat = np.asarray(mat, dtype=float)
    if np.abs(mat).sum() < 1e-6:
        return None
    return mat[:3, 3].copy(), R_to_quat(np.ascontiguousarray(mat[:3, :3]))


def set_world_quat(entity, q, device):
    """Set a (num_envs=1) entity's world-frame orientation to wxyz quaternion ``q``."""
    # relative=False to match get_quat()'s world-frame readout (round-trip consistent).
    entity.set_quat(torch.as_tensor(q, dtype=torch.float32, device=device).reshape(1, 4), relative=False)


def step_toward(cur_pos, cur_quat, tgt_pos, tgt_quat, dt):
    """Move a pose toward a target at capped linear/angular speed (arm-lag model)."""
    dp = np.asarray(tgt_pos, float) - cur_pos
    dp_norm = float(np.linalg.norm(dp))
    max_lin = EE_MAX_LIN * dt
    new_pos = cur_pos + dp * (max_lin / dp_norm) if dp_norm > max_lin else np.asarray(tgt_pos, float)

    # shortest-arc error tgt ⊗ cur⁻¹ (transform_quat_by_quat(v, u) == u ⊗ v).
    e = quat_to_rotvec(transform_quat_by_quat(inv_quat(cur_quat), tgt_quat))
    e_norm = float(np.linalg.norm(e))
    max_ang = EE_MAX_ANG * dt
    if e_norm > max_ang:
        new_quat = transform_quat_by_quat(cur_quat, rotvec_to_quat(e * (max_ang / e_norm)))
    else:
        new_quat = np.asarray(tgt_quat, float)
    return new_pos, new_quat


def build_env(args):
    """Construct the toy controller-teleop env; return (env, cube, markers, cams)."""
    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    # Per-controller markers: a body sphere at the grip + a small tip showing forward.
    markers = {}
    for side, color in (("lctrl", (0.2, 0.4, 1.0)), ("rctrl", (1.0, 0.3, 0.3))):
        markers[f"{side}_body"] = SphereOptions(radius=0.02, fixed=True, default_root_pos=(0.3, 0.0, 1.0), color=color)
        markers[f"{side}_tip"] = SphereOptions(radius=0.01, fixed=True, default_root_pos=(0.3, 0.0, 1.1), color=color)

    scene_options = SceneOptions(
        plane=PlaneOptions(),
        table=BoxOptions(
            size=(0.4, 0.6, 0.7),
            fixed=True,
            default_root_pos=(0.45, 0.0, 0.35),
            color=(0.6, 0.45, 0.3),
        ),
        cube=BoxOptions(
            size=(0.06, 0.06, 0.06),
            fixed=True,  # kinematic: the clutch mapper drives its pose directly
            default_root_pos=CUBE_HOME_POS,
            color=(0.9, 0.7, 0.1),
        ),
        **markers,
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

    markers = {side: (env.entities[f"{side}_body"], env.entities[f"{side}_tip"]) for side in ("lctrl", "rctrl")}
    cams = (env.cameras["cam_left"], env.cameras["cam_right"])
    return env, env.entities["cube"], markers, cams


def update_marker(body, tip, wrist_mat, device):
    """Move a controller's body+tip markers to its pose. No-op if tracking dropped."""
    pq = mat_to_pos_quat(wrist_mat)
    if pq is None:
        return
    pos = pq[0]
    set_world_pos(body, pos, device)
    set_world_pos(tip, pos + ctrl_axes(wrist_mat) * TIP_LEN, device)


def main():
    parser = argparse.ArgumentParser(description="Eden <-> Quest XR controller teleop demo (Eden side)")
    parser.add_argument("--width", type=int, default=1280, help="full stereo frame width (both eyes)")
    parser.add_argument("--height", type=int, default=480, help="frame height")
    parser.add_argument("--fov", type=float, default=90.0, help="vertical fov per eye (deg)")
    parser.add_argument("--fps", type=float, default=30.0, help="streaming fps")
    parser.add_argument("--physics-hz", type=float, default=100.0, help="physics dt = 1/physics_hz")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("--bridge-host", default="127.0.0.1", help="host running the XR bridge")
    parser.add_argument("-v", "--vis", action="store_true", help="also open the local Genesis viewer")
    args = parser.parse_args()

    env, cube, markers, (cam_left, cam_right) = build_env(args)
    device = env.device

    ctx = zmq.Context.instance()
    frame_sock = P.frame_push(ctx, host=args.bridge_host)
    pose_sock = P.pose_sub(ctx, host=args.bridge_host)

    en.logger.info(
        f"streaming {args.width}x{args.height} stereo to {args.bridge_host}; subscribing to controller state. "
        f"Hold a GRIP to clutch the cube; TRIGGER for precision; A to home."
    )

    mapper = ClutchPoseMapper()  # R = identity: poses already in robot world frame
    engaged_side = None  # "left" | "right" | None — only one hand drives at a time
    side_to_marker = {"left": "lctrl", "right": "rctrl"}
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

            # A button on either controller homes the cube and drops any clutch.
            if last_pose.get("left_a_button") or last_pose.get("right_a_button"):
                set_world_pos(cube, np.array(CUBE_HOME_POS), device)
                set_world_quat(cube, np.array(CUBE_HOME_QUAT), device)
                mapper.disengage()
                engaged_side = None

            cube_pos = cube.get_pos()[0].detach().cpu().numpy()[:3].astype(float)
            cube_quat = cube.get_quat()[0].detach().cpu().numpy()[:4].astype(float)

            for side in ("left", "right"):
                update_marker(*markers[side_to_marker[side]], last_pose[f"{side}_wrist"], device)
                pq = mat_to_pos_quat(last_pose[f"{side}_wrist"])
                grip = bool(last_pose[f"{side}_squeeze"]) and pq is not None
                if grip and engaged_side is None:
                    # Rising clutch edge: capture controller + current cube pose.
                    mapper.engage(pq[0], pq[1], cube_pos, cube_quat)
                    engaged_side = side
                elif not grip and engaged_side == side:
                    mapper.disengage()
                    engaged_side = None

            if engaged_side is not None:
                pq = mat_to_pos_quat(last_pose[f"{engaged_side}_wrist"])
                if pq is not None:
                    # Trigger = precision modifier: lower the gains for fine work.
                    fine = bool(last_pose[f"{engaged_side}_trigger"])
                    mapper.scale = PRECISION_SCALE if fine else 1.0
                    mapper.scale_rotation = PRECISION_SCALE if fine else 1.0
                    # Reach-limited target against the cube's CURRENT pose, then
                    # chase it at capped speed so the slipping clutch is visible.
                    out = mapper.target(pq[0], pq[1], cube_pos, cube_quat)
                    if out is not None:
                        new_pos, new_quat = step_toward(cube_pos, cube_quat, out[0], out[1], period)
                        set_world_pos(cube, new_pos, device)
                        set_world_quat(cube, new_quat, device)

        env.scene.step()

        stereo = np.concatenate([render_rgb_np(cam_left), render_rgb_np(cam_right)], axis=1)
        P.send_frame(frame_sock, stereo)
        frames += 1

        now = time.time()
        if now - last_log > 5.0:
            if last_pose and last_pose.get("ready"):
                clutch = engaged_side or "—"
                en.logger.info(
                    f"frames: {frames} | clutch: {clutch} | "
                    f"L grip={last_pose['left_squeeze_value']:.2f} trig={last_pose['left_trigger_value']:.2f} "
                    f"A={int(last_pose['left_a_button'])} | "
                    f"R grip={last_pose['right_squeeze_value']:.2f} trig={last_pose['right_trigger_value']:.2f} "
                    f"A={int(last_pose['right_a_button'])}"
                )
            else:
                en.logger.info(f"frames sent: {frames} | headset tracking: False")
            last_log = now

        dt = time.time() - loop_start
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()
