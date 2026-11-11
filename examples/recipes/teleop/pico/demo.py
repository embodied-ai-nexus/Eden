"""Eden-side PICO teleop demo: controller clutch, thumbstick locomotion, full-body joints.

---
tier: recipe
capabilities: [teleop.pico]
hardware: cpu
optional_deps: [pico]
run: uv run python examples/recipes/teleop/pico/demo.py
---

Consumes the tracking state published by ``eden pico bridge`` (XRoboToolkit ->
PICO headset) and renders a head-tracked stereo view back to the headset. It
exercises every part of a PICO session:

    * **pose** — a marker sphere follows each controller (blue = left, red =
      right) with a tip showing its forward direction,
    * **grip = clutch** — hold a grip and the cube follows that hand's *relative*
      motion (position + orientation) through ``ClutchPoseMapper``; release to
      reposition your hand without moving the cube. Only one hand drives at a time,
    * **trigger = precision** — hold to scale the response down for fine work,
    * **X / A button** — send the cube back to its home pose,
    * **thumbsticks = locomotion (camera-relative)** — the left stick walks the
      *tracking origin* toward where you're *looking* (its heading follows your
      head yaw, so "forward" = "into the view"); the right stick yaws the origin.
      Head, controllers and body joints all ride that origin, so the view and your
      hands move together — which is exactly what you need when the XR origin is
      mounted on a moving robot base,
    * **full-body joints** — with Full-body mode and calibrated PICO motion
      trackers, the 24 tracked body joints show up as green spheres (they stay
      parked under the floor while body tracking is off).

The bridge publishes the same dict schema as ``eden metaquest bridge --tracking
controller``, so ``examples/recipes/teleop/metaquest/controller_demo.py`` and
``controller_teleop.py`` (bimanual YAM arms + IK) run against a PICO unchanged —
this recipe is the one that also uses the PICO-only extras.

Poses arrive already in the Genesis/robot world frame (z up, x forward, y left,
metres, origin at the headset's tracking origin). The clutch reads controller
motion in the operator's **gaze frame** (``CONTROL_FRAME``): its ``R`` is driven
from the live head pose each tick, so turning your head re-aims "forward" — set
``CONTROL_FRAME="world"`` for room-fixed control instead.

Driving from a screen instead of the headset? ``--viewer-camera`` uses the
interactive Genesis viewer camera (orbit it with the mouse) as the head/control
frame, so the gaze-relative control and locomotion follow the on-screen view.
(The controllers still come from the PICO, so the headset must be tracking.)

Run (after starting the bridge):
    eden pico bridge
    uv run python examples/recipes/teleop/pico/demo.py            # GPU
    uv run python examples/recipes/teleop/pico/demo.py --backend cpu
    uv run python examples/recipes/teleop/pico/demo.py --viewer-camera   # drive from the desktop viewer
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

# The PICO bridge speaks the same ZMQ wire protocol and dict schema as the Meta
# Quest bridge, so the Eden side is device-agnostic.
from eden.utils.devices.metaquest import protocol as P
from eden.utils.devices.metaquest.pose_mapping import ClutchPoseMapper
from eden.utils.devices.pico.xr_client import BODY_JOINT_NAMES, N_BODY_JOINTS
from genesis.utils.geom import R_to_quat, inv_quat, quat_to_rotvec, rotvec_to_quat, transform_quat_by_quat

IPD = 0.063  # interpupillary distance (m) -> stereo camera separation
CUBE_HOME_POS = (0.45, 0.0, 0.75)  # cube reset position (X / A button)
CUBE_HOME_QUAT = (1.0, 0.0, 0.0, 0.0)  # cube reset orientation (wxyz)
TIP_LEN = 0.08  # length (m) of the forward-direction tip marker off each controller
PRECISION_SCALE = 0.25  # response gain while the trigger (precision modifier) is held
EE_MAX_LIN = 0.75  # m/s cap on how fast the cube chases the target (simulated arm lag)
EE_MAX_ANG = 3.0  # rad/s cap on the cube's angular tracking
PARKED_Z = -1.0  # body-joint spheres wait below the floor until tracking starts
# Frame the clutch interprets controller motion in:
#   "world"       — 1:1 with the room (turning your head does NOT change the mapping),
#   "camera_yaw"  — controller motion is read in your gaze's yaw frame, so "push
#                   forward" = "into the current view"; turning your head re-aims it,
#   "camera_full" — full head orientation (pitch/roll too; "forward" tilts with your head).
CONTROL_FRAME = "camera_yaw"
STICK_DEADZONE = 0.15  # thumbstick magnitude below which locomotion is ignored
WALK_SPEED = 0.8  # m/s at full left-stick deflection
YAW_RATE = 1.2  # rad/s at full right-stick x deflection
# The service reports the thumbstick as (x, y) with +x right; flip this to -1.0 if
# pushing the stick forward walks you backwards on your headset/firmware.
STICK_FORWARD_SIGN = 1.0


class TrackingOrigin:
    """Where the operator's XR tracking origin sits in the Genesis world.

    Integrated from the thumbsticks and applied to every tracked pose, so
    "walking" carries the head, the controllers and the body joints together —
    the same composition you need when the XR origin rides a robot base instead
    of the world origin.

    Locomotion is **camera-relative**: the left stick walks toward where the
    operator is *looking* (the head's world-yaw), not along a fixed heading, so
    "push forward" always means "toward my gaze". The right stick still yaws the
    whole origin (a snap-turn for when you can't physically turn far enough).
    """

    def __init__(self) -> None:
        self.pos = np.zeros(3)
        self.yaw = 0.0

    def integrate(self, gaze_yaw, left_axis, right_axis, dt: float) -> None:
        """Advance the origin one tick; walk along ``gaze_yaw`` (world heading of the gaze).

        ``gaze_yaw`` is the world yaw (rad) the operator is looking along — from the
        head pose, or the viewer camera in ``--viewer-camera`` mode — so "push
        forward" tracks the gaze. ``None`` (no gaze this tick) walks nowhere.
        """
        lx, ly = _deadzone(left_axis)
        rx, _ = _deadzone(right_axis)
        self.yaw -= YAW_RATE * rx * dt  # stick right -> turn right (robot yaw is CCW)
        if gaze_yaw is None:
            return
        c, s = np.cos(gaze_yaw), np.sin(gaze_yaw)
        gaze = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = np.array([STICK_FORWARD_SIGN * ly, -lx, 0.0]) * WALK_SPEED * dt  # +x fwd, +y left
        self.pos = self.pos + gaze @ local

    def apply_pose(self, mat) -> np.ndarray:
        """Express a tracked 4x4 pose (origin frame) in the world frame."""
        mat = np.asarray(mat, dtype=float)
        if np.abs(mat).sum() < 1e-6:  # tracking dropped: leave the sentinel intact
            return mat
        return self._transform() @ mat

    def apply_points(self, points) -> np.ndarray:
        """Express tracked (N, 3) points (origin frame) in the world frame."""
        pts = np.asarray(points, dtype=float)
        return pts @ self._rotation().T + self.pos

    def _rotation(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _transform(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self._rotation()
        mat[:3, 3] = self.pos
        return mat


def _deadzone(axis) -> tuple[float, float]:
    """Zero a thumbstick reading that is inside the deadzone."""
    x, y = (float(axis[0]), float(axis[1])) if axis is not None else (0.0, 0.0)
    return (0.0, 0.0) if np.hypot(x, y) < STICK_DEADZONE else (x, y)


def head_axes(mat):
    """Return (forward, up, right) world directions from a robot-frame head pose.

    The pose is in the Genesis/robot frame, so the head's neutral local axes are
    +X forward, +Z up, +Y left (hence rightward = -col 1).
    """
    r = np.asarray(mat, dtype=float)[:3, :3]
    return r[:, 0], r[:, 2], -r[:, 1]


def ctrl_forward(mat):
    """Forward world direction of a controller from its robot-frame pose.

    OpenXR grip space points along local -Z; the tip marker is cosmetic, so drop
    the sign here if it points the wrong way on your firmware.
    """
    return -np.asarray(mat, dtype=float)[:3, 2]


def mat_to_pos_quat(mat):
    """4x4 robot-frame pose -> (pos (3,), wxyz quat (4,)). None if tracking dropped."""
    mat = np.asarray(mat, dtype=float)
    if np.abs(mat).sum() < 1e-6:
        return None
    return mat[:3, 3].copy(), R_to_quat(np.ascontiguousarray(mat[:3, :3]))


def viewer_head_world(scene):
    """Robot-convention head pose (col0=forward, col1=left, col2=up) from the live viewer.

    Reads the interactive ``--viewer`` camera's current pose (updated as you orbit
    with the mouse). Genesis' viewer pose is OpenGL convention (x right, y up, -z
    forward), so it is re-expressed into the head convention the rest of the demo
    uses — letting the desktop viewer stand in for the headset when you drive from
    a screen instead of wearing the PICO.
    """
    cam = np.asarray(scene.viewer.camera_pose, dtype=float)  # OpenGL cam-to-world 4x4
    head = np.eye(4)
    head[:3, 0] = -cam[:3, 2]  # forward = -back
    head[:3, 1] = -cam[:3, 0]  # left = -right
    head[:3, 2] = cam[:3, 1]  # up
    head[:3, 3] = cam[:3, 3]
    return head


def head_world_yaw(head_world):
    """World yaw (rad) of a robot-convention head pose, or None if untracked."""
    head_world = np.asarray(head_world, dtype=float)
    if np.abs(head_world).sum() < 1e-6:
        return None
    return float(np.arctan2(head_world[1, 0], head_world[0, 0]))


def control_rotation(head_world):
    """3x3 rotation that maps controller world-motion into the clutch's control frame.

    Fed to ``ClutchPoseMapper.set_R`` each tick. With the camera frames, the cube's
    response rotates with where the operator is looking (``head_world`` is the head
    pose already expressed in the Genesis world), so "push forward" follows the gaze
    rather than a fixed room axis. See :data:`CONTROL_FRAME`.
    """
    if CONTROL_FRAME == "world":
        return np.eye(3)
    r = np.asarray(head_world, dtype=float)[:3, :3]
    if CONTROL_FRAME == "camera_full":
        return r
    # camera_yaw: keep world up; rotate the control plane by the head's yaw only.
    psi = np.arctan2(r[1, 0], r[0, 0])  # world yaw of head-forward (+x column)
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_env(args):
    """Construct the toy PICO teleop env; return (env, cube, markers, joints, cams)."""
    en.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu)

    # Every marker is a teleported visual: collision=False so it neither knocks
    # the cube around nor spends collision pairs against the ground plane.
    def marker(radius, pos, color):
        return SphereOptions(radius=radius, fixed=True, collision=False, default_root_pos=pos, color=color)

    markers = {}
    for side, color in (("lctrl", (0.2, 0.4, 1.0)), ("rctrl", (1.0, 0.3, 0.3))):
        markers[f"{side}_body"] = marker(0.02, (0.3, 0.0, 1.0), color)
        markers[f"{side}_tip"] = marker(0.01, (0.3, 0.0, 1.1), color)

    # Full-body joints: parked below the floor until the service reports them.
    body = {
        f"joint_{i:02d}": marker(0.03, (0.0, 0.0, PARKED_Z - i * 0.001), (0.25, 0.8, 0.35))
        for i in range(N_BODY_JOINTS)
    }

    scene_options = SceneOptions(
        plane=PlaneOptions(),
        table=BoxOptions(size=(0.4, 0.6, 0.7), fixed=True, default_root_pos=(0.45, 0.0, 0.35), color=(0.6, 0.45, 0.3)),
        cube=BoxOptions(
            size=(0.06, 0.06, 0.06),
            fixed=True,  # kinematic: the clutch mapper drives its pose directly
            default_root_pos=CUBE_HOME_POS,
            color=(0.9, 0.7, 0.1),
        ),
        **markers,
        **body,
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

    env = EnvBase.from_config(config, show_viewer=args.viewer)
    env.build()

    marker_entities = {side: (env.entities[f"{side}_body"], env.entities[f"{side}_tip"]) for side in ("lctrl", "rctrl")}
    joints = [env.entities[f"joint_{i:02d}"] for i in range(N_BODY_JOINTS)]
    cams = (env.cameras["cam_left"], env.cameras["cam_right"])
    return env, env.entities["cube"], marker_entities, joints, cams


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


def set_world_quat(entity, q, device):
    """Set a (num_envs=1) entity's world orientation to wxyz quaternion ``q``."""
    # relative=False to match get_quat()'s world-frame readout (round-trip consistent).
    entity.set_quat(torch.as_tensor(q, dtype=torch.float32, device=device).reshape(1, 4), relative=False)


def render_rgb_np(cam):
    """Render one camera to a contiguous (H, W, 3) uint8 numpy array."""
    rgb = cam.render_rgb()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:  # (num_envs, H, W, C) -> single env
        rgb = rgb[0]
    return np.ascontiguousarray(rgb[..., :3].astype(np.uint8))


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


def update_marker(body, tip, ctrl_mat, device):
    """Move a controller's body+tip markers to its pose. No-op if tracking dropped."""
    pq = mat_to_pos_quat(ctrl_mat)
    if pq is None:
        return
    set_world_pos(body, pq[0], device)
    set_world_pos(tip, pq[0] + ctrl_forward(ctrl_mat) * TIP_LEN, device)


def update_body(joints, positions, device):
    """Move the 24 body-joint spheres; no-op while body tracking is off."""
    pts = np.asarray(positions, dtype=float)
    if pts.shape != (N_BODY_JOINTS, 3):
        return
    for sphere, p in zip(joints, pts):
        set_world_pos(sphere, p, device)


def main():
    ap = argparse.ArgumentParser(description="Eden <-> PICO teleop demo (Eden side)")
    ap.add_argument("--width", type=int, default=1280, help="full stereo frame width (both eyes)")
    ap.add_argument("--height", type=int, default=480, help="frame height")
    ap.add_argument("--fov", type=float, default=90.0, help="vertical fov per eye (deg)")
    ap.add_argument("--fps", type=float, default=30.0, help="streaming fps")
    ap.add_argument("--physics-hz", type=float, default=100.0, help="physics dt = 1/physics_hz")
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"], help="Genesis compute backend")
    ap.add_argument("--bridge-host", default="127.0.0.1", help="host running the PICO bridge")
    ap.add_argument("--viewer", action="store_true", help="also open the local Genesis viewer")
    ap.add_argument(
        "--viewer-camera",
        action="store_true",
        help="use the interactive viewer camera (orbit with the mouse) as the head/control "
        "frame instead of the headset — for driving from a screen (implies --viewer)",
    )
    args = ap.parse_args()
    if args.viewer_camera:
        args.viewer = True  # the viewer must be open to read its camera

    env, cube, markers, joints, (cam_left, cam_right) = build_env(args)
    device = env.device

    ctx = zmq.Context.instance()
    frame_sock = P.frame_push(ctx, host=args.bridge_host)
    pose_sock = P.pose_sub(ctx, host=args.bridge_host)

    en.logger.info(
        f"streaming {args.width}x{args.height} stereo to {args.bridge_host}; subscribing to PICO state. "
        f"Hold a GRIP to clutch the cube; TRIGGER for precision; X/A to home; thumbsticks to walk."
    )

    origin = TrackingOrigin()
    # Two clutches so translation and rotation use different frames: TRANSLATION
    # follows the gaze (its R is aimed at CONTROL_FRAME each tick, so "push forward"
    # = "into the view"), while ROTATION stays hand-following (R = identity) so wrist
    # twists map 1:1 to the cube instead of being spun by the view. Set
    # CONTROL_FRAME="world" to make translation room-fixed too.
    mapper_t = ClutchPoseMapper()  # translation
    mapper_r = ClutchPoseMapper()  # rotation (hand-following)
    engaged_side = None  # "left" | "right" | None — only one hand drives at a time
    ctrl_yaw_deg = 0.0  # control-frame yaw, surfaced in the status line for diagnosis
    side_to_marker = {"left": "lctrl", "right": "rctrl"}
    period = 1.0 / max(args.fps, 1.0)
    last_state = None
    frames = 0
    last_log = time.time()
    warned_device = False
    remaining = 5 if en.is_smoke() else -1

    while remaining != 0:
        loop_start = time.time()

        state = P.recv_pose(pose_sock, flags=zmq.NOBLOCK)
        if state is not None:
            if state.get("device") != "pico" and not warned_device:
                en.logger.warning(
                    f"pose stream reports device={state.get('device')!r}; expected `eden pico bridge`. "
                    "The PICO-only extras (full-body joints, menu buttons) will be missing."
                )
                warned_device = True
            last_state = state

        if last_state is not None and last_state.get("ready"):
            # "Where the operator is looking": the viewer camera when driving from a
            # screen, else the headset head pose (carried by the tracking origin).
            head_world = viewer_head_world(env.scene) if args.viewer_camera else origin.apply_pose(last_state["head"])
            origin.integrate(
                head_world_yaw(head_world),
                last_state.get("left_thumbstick_value"),
                last_state.get("right_thumbstick_value"),
                period,
            )
            if not args.viewer_camera:
                head_world = origin.apply_pose(last_state["head"])  # refresh after the origin moved
            aim_cameras(cam_left, cam_right, head_world)
            # Aim the TRANSLATION clutch's frame at where the operator is looking, so
            # controller motion is read relative to the gaze (see CONTROL_FRAME). The
            # rotation clutch stays hand-following (identity).
            mapper_t.set_R(control_rotation(head_world))
            _yaw = head_world_yaw(head_world)
            ctrl_yaw_deg = np.degrees(_yaw) if _yaw is not None else ctrl_yaw_deg

            # X (left) or A (right) homes the cube and drops any clutch.
            if last_state.get("left_a_button") or last_state.get("right_a_button"):
                set_world_pos(cube, np.array(CUBE_HOME_POS), device)
                set_world_quat(cube, np.array(CUBE_HOME_QUAT), device)
                mapper_t.disengage()
                mapper_r.disengage()
                engaged_side = None

            if last_state.get("body_ready"):
                body = last_state["body"]  # {joint_name: {pos, quat, vel, acc}}
                pts = np.array([body[name]["pos"] for name in BODY_JOINT_NAMES])
                update_body(joints, origin.apply_points(pts), device)

            cube_pos = cube.get_pos()[0].detach().cpu().numpy()[:3].astype(float)
            cube_quat = cube.get_quat()[0].detach().cpu().numpy()[:4].astype(float)

            ctrl_world = {side: origin.apply_pose(last_state[f"{side}_wrist"]) for side in ("left", "right")}
            for side, mat in ctrl_world.items():
                update_marker(*markers[side_to_marker[side]], mat, device)
                pq = mat_to_pos_quat(mat)
                grip = bool(last_state[f"{side}_squeeze"]) and pq is not None
                if grip and engaged_side is None:
                    mapper_t.engage(pq[0], pq[1], cube_pos, cube_quat)  # rising clutch edge
                    mapper_r.engage(pq[0], pq[1], cube_pos, cube_quat)
                    engaged_side = side
                elif not grip and engaged_side == side:
                    mapper_t.disengage()
                    mapper_r.disengage()
                    engaged_side = None

            if engaged_side is not None:
                pq = mat_to_pos_quat(ctrl_world[engaged_side])
                if pq is not None:
                    # Trigger = precision modifier: lower the gains for fine work.
                    fine = bool(last_state[f"{engaged_side}_trigger"])
                    mapper_t.scale = PRECISION_SCALE if fine else 1.0
                    mapper_r.scale_rotation = PRECISION_SCALE if fine else 1.0
                    # Reach-limited targets against the cube's CURRENT pose: take the
                    # position from the gaze-framed clutch, the orientation from the
                    # hand-following one, then chase both at capped speed.
                    out_t = mapper_t.target(pq[0], pq[1], cube_pos, cube_quat)
                    out_r = mapper_r.target(pq[0], pq[1], cube_pos, cube_quat)
                    if out_t is not None and out_r is not None:
                        new_pos, new_quat = step_toward(cube_pos, cube_quat, out_t[0], out_r[1], period)
                        set_world_pos(cube, new_pos, device)
                        set_world_quat(cube, new_quat, device)

        env.scene.step()

        stereo = np.concatenate([render_rgb_np(cam_left), render_rgb_np(cam_right)], axis=1)
        P.send_frame(frame_sock, stereo)
        frames += 1
        remaining -= 1 if remaining > 0 else 0

        now = time.time()
        if now - last_log > 5.0:
            if last_state and last_state.get("ready"):
                src = "viewer" if args.viewer_camera else "head"
                en.logger.info(
                    f"frames: {frames} | clutch: {engaged_side or '—'} | "
                    f"ctrl-frame({src}) yaw={ctrl_yaw_deg:+.0f}°  <- should change as you orbit/turn | "
                    f"origin: xy=({origin.pos[0]:+.2f}, {origin.pos[1]:+.2f}) yaw={np.degrees(origin.yaw):+.0f}° | "
                    f"body: {'on' if last_state.get('body_ready') else 'off'} "
                    f"trackers: {len(last_state.get('trackers', {}))} | "
                    f"L grip={last_state['left_squeeze_value']:.2f} trig={last_state['left_trigger_value']:.2f} | "
                    f"R grip={last_state['right_squeeze_value']:.2f} trig={last_state['right_trigger_value']:.2f}"
                )
            else:
                en.logger.info(f"frames sent: {frames} | headset tracking: False")
            last_log = now

        dt = time.time() - loop_start
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()
