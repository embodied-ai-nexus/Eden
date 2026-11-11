# Eden ⇆ PICO XR teleop

Stream PICO 4 / PICO 4 Ultra tracking into Eden (Genesis backend) and render the
**sim scene back to the headset** in stereo, using Eden's built-in
[XRoboToolkit](https://github.com/XR-Robotics) bridge.

Unlike the Meta Quest path (WebXR served by televuer), a PICO headset never talks
to Eden directly. It streams tracking to the **XRoboToolkit PC Service** on this
host, and Eden's bridge reads that service and drives the headset's *Remote
Vision* video link:

```
 PICO ──Wi-Fi──▶ XRoboToolkit PC Service ──▶ eden pico bridge
                                             │  poses  (ZMQ PUB :5557) ──▶ demo.py
   headset decoder ◀──H.264 / TCP────────────┤  frames (ZMQ PULL :5556) ◀──┘
                                             └─ control server TCP :13579
```

Poses are published in the **Genesis world frame** (z up, x forward, y left,
metres, origin = the headset's tracking origin), so the Eden side consumes them
directly.

## Interchangeable with the Meta Quest bridge

`eden pico bridge` publishes the **same dict schema** as `eden metaquest bridge
--tracking controller` (`head`, `{side}_wrist`, `{side}_trigger[_value]`,
`{side}_squeeze[_value]`, `{side}_a_button` / `_b_button`,
`{side}_thumbstick[_value]`) over the same ZMQ channels. So the Quest recipes run
against a PICO with no changes:

```bash
eden pico bridge                                                    # instead of the Quest bridge
uv run python examples/recipes/teleop/metaquest/controller_demo.py   # clutch-mapped marker cube
uv run python examples/recipes/teleop/metaquest/controller_teleop.py # bimanual YAM arms + IK
```

PICO's extras are additive keys: `{side}_menu_button`, `trackers` (up to 5 motion
trackers), `body_ready` / `body` (the 24 SMPL joints of Full-body mode), and
`body_raw` / `body_raw_timestamp_ns` — the untouched service-frame `(24, 7)`
xyz+xyzw poses, for consumers that do their own frame math (the SONIC teleop
operator). `demo.py` here is the recipe that uses them.

### Motion trackers (up to 5) and full body

`state()["trackers"]` carries up to **5** PICO motion trackers (2 legs + waist, or
wrist + leg + waist, …), each as `{serial, role, pos, quat, vel, acc}` in the robot
frame. The service reports trackers only by **serial number**, so label them by
placement with the bridge:

```bash
eden pico bridge --tracker-role PA12345=waist --tracker-role PB67890=left_foot
```

Run the bridge once and it logs each unlabeled tracker's serial (`motion tracker
<serial> is unlabeled — map it with --tracker-role <serial>=<role>`) so you can
build the mapping. Mapped trackers are keyed by role in `state()["trackers"]`
(e.g. `trackers["waist"]`); unmapped ones fall back to their serial.

With **Full-body mode** on (calibrated trackers), `state()["body"]` is a named dict
`{joint: {pos, quat, vel, acc}}` over the 24 SMPL joints — `left_wrist`,
`right_wrist`, `left_ankle`, `right_ankle`, `left_foot`, `right_foot`, `pelvis`
(waist), etc. (the controllers already cover the wrists as `left_wrist` /
`right_wrist` in the top-level state). `vel`/`acc` are `(linear, angular)`
6-vectors in the robot frame.

> The 5-tracker cap needs a **patched `xrobotoolkit_sdk`** — upstream hard-codes 3.
> See `MAX_MOTION_TRACKERS` in the pybind's `bindings/py_bindings.cpp`; rebuild and
> reinstall the module (the setup notes below).

## Setup

**1. Eden extra** — H.264/H.265 encoding plus ZMQ:

```bash
uv sync --inexact --extra pico          # or --all-extras --all-groups
```

**2. XRoboToolkit PC Service** on this host — follow *Install
XRoboToolkit-PC-Service* at [XR-Robotics](https://github.com/XR-Robotics).

**3. The `xrobotoolkit_sdk` Python module.** It is **not** on PyPI: it is a
pybind extension built against the PC Service's `libPXREARobotSDK.so`, so Eden
cannot declare it as a dependency. Build it with Eden's interpreter active:

```bash
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind
cd XRoboToolkit-PC-Service-Pybind
bash setup_ubuntu.sh          # setup_orin.sh on Jetson, setup_windows.bat on Windows
```

`eden pico bridge` fails immediately with this hint if the module is missing.
No `LD_LIBRARY_PATH` export is needed: `libPXREARobotSDK.so` is preloaded from
the PC Service's install directory (`/opt/apps/roboticsservice/SDK/x64`;
override with `EDEN_PXREA_SDK_DIR`) when the dynamic linker can't find it.

**4. PICO app** — install `XRoboToolkit-PICO-*.apk` on the headset (download it in
the headset's browser from the XR-Robotics releases page, then open it from
*Manage Downloads*). For full-body tracking, pair and calibrate two PICO motion
trackers (one per ankle) from the headset's *Motion Tracker* app first.

## Run

**Terminal 1 — the bridge:**

```bash
eden pico bridge
# --no-video for tracking only; --fps / --control-address / --pose-port / --frame-port to tune
```

It prints this host's LAN addresses, then a status line every 5 s (tracking,
tracker count, full-body on/off, trigger/grip values, video state).

**Terminal 2 — the Eden sim:**

```bash
uv run python examples/recipes/teleop/pico/demo.py            # GPU
uv run python examples/recipes/teleop/pico/demo.py --backend cpu
# --bridge-host <ip> if the bridge runs on another machine
```

**On the PICO:**

1. Join the **same LAN as this host** and open **XRoboToolkit** (it is in the
   *Unknown* section of the app library).
2. Confirm the auto-detected PC address, tap **Connect**, and wait for
   **Status: Working**.
3. Under *Tracking* select **Head** + **Controller** (and set the PICO Motion
   Tracker mode to **Full-body** if you have trackers); enable **Send** under
   *Data & Control*.
4. Under *Remote Vision* tap **Listen** — the headset dials Eden's control server
   on `:13579` and the stereo sim view appears. The **B** button toggles the app's
   stereo/single view.

## Controls (`demo.py`)

| Input | Action |
| --- | --- |
| **Grip** | Clutch — the cube follows that hand's *relative* motion; release to reposition your hand without moving it. One hand at a time. |
| **Trigger** | Precision modifier — scales the response down for fine work. |
| **X** (left) / **A** (right) | Home the cube. |
| **Left thumbstick** | Walk the tracking origin **toward where you're looking** (forward/strafe follow your head yaw). |
| **Right thumbstick** (x) | Yaw the tracking origin (snap-turn). |

A marker sphere follows each controller (blue = left, red = right) with a tip
showing its forward direction. With Full-body mode on, 24 green spheres track your
body joints; they stay parked below the floor while body tracking is off.

The cube is driven through
[`ClutchPoseMapper`](../../../../eden/utils/devices/metaquest/pose_mapping.py)
and chases the mapper's target at a capped speed (a stand-in for a real arm's
velocity limit), so the **reach limit / "slipping clutch"** is visible: flick fast
and the target runs ahead, clamps, and the overshoot is absorbed — reverse and the
cube bites immediately instead of retracing it. For a real arm you pass its
measured end-effector pose into `mapper.target(...)` instead of this follower
(that is what `metaquest/controller_teleop.py` does with IK).

The thumbsticks move the **tracking origin**, not the camera: head, controllers,
and body joints are all expressed through it, which is the composition you need
when the XR origin is mounted on a moving robot base rather than the world origin.
Both are **camera-relative**. Locomotion: the left stick walks toward your gaze
(heading follows your head yaw), so "forward" is always "into the view". The cube
clutch reads controller motion in your **gaze frame** too — `demo.py` feeds the
mapper's `R` from the live head pose each tick (`CONTROL_FRAME`, default
`"camera_yaw"`), so turning your head re-aims "forward" and the cube's response
rotates to follow where you look. Set `CONTROL_FRAME = "world"` for room-fixed
control, or `"camera_full"` to include head pitch/roll.

> By default the control frame follows the **headset** head pose, not the
> `--viewer` desktop window. To drive from a screen instead, pass
> **`--viewer-camera`**: the interactive viewer camera (orbit it with the mouse)
> becomes the head/control frame, so the gaze-relative clutch and locomotion
> follow the on-screen view. The controllers still come from the PICO, so the
> headset must be tracking.

## Gotchas

- **Forward/back inverted?** The service's thumbstick y sign varies by firmware —
  flip `STICK_FORWARD_SIGN` at the top of `demo.py`.
- **Frame size.** The app configures its decoder from its own *Remote Vision*
  profile, so the bridge encodes at the size the `OPEN_CAMERA` request names and
  rescales your frames to match (it warns once). Match `--width`/`--height` to the
  app's profile to skip the rescale.
- **Hand tracking.** `PicoXrClient.hand_joints()` exposes PICO's 26-joint OpenXR
  hand layout, but the bridge publishes controller state only — the Quest hand
  recipe (`metaquest/demo.py`) assumes WebXR's 25-joint layout and is *not*
  index-compatible.

## Bimanual robot arms (YAM) instead of the marker cube

`controller_teleop.py` is the "real arm" build: two i2rt **YAM** 6-DOF arms, one
per controller, mounted on a table picking at a few objects. The controller pose
becomes an end-effector *target* tracked through inverse kinematics, so the clutch
reach limit does real work (the arm genuinely lags the hand). The arms use
`gravity_compensation` so the PD controller holds the IK target without sag.

```bash
eden pico bridge
uv run python examples/recipes/teleop/pico/controller_teleop.py            # GPU
uv run python examples/recipes/teleop/pico/controller_teleop.py --backend cpu
uv run python examples/recipes/teleop/pico/controller_teleop.py --viewer-camera  # drive from the desktop viewer
```

- **Grip = clutch** — drive that arm's end-effector relative to your hand; release
  to reposition. Both hands drive their own arm → true bimanual control.
- **Trigger = gripper** — analog: squeeze to close, release to open.
- **A** (right) resets both arms to home; **X** (left) resets the objects.

Unlike `demo.py` (whose cube *translation* follows your gaze), the arm **mirrors
your hand** — `R = identity`, so the end-effector copies your hand's own world
motion, which is what you want when a real arm should track your hand rather than
move relative to the view. Only the camera you see through follows your head (or
the `--viewer-camera` view). The pipeline each tick is `controller pose →
ClutchPoseMapper.target(.., current EE pose) → RigidEntity.inverse_kinematics →
control_dofs_pos`. Swap `en.robots.Yam` and the `EE_LINK` / joint names at the top
of the file to drive a different arm.

## Using a real Eden task instead of the toy scene

`demo.py` builds a throwaway scene in `build_env()`. To teleoperate a real task,
swap that for a registered task — `RLEnvBase.from_config(cfg)` — add a stereo
camera pair via the config's `cameras_options`, and keep the same pose→action /
render→stream loop. The device code lives in
[`eden/utils/devices/pico/`](../../../../eden/utils/devices/pico/): `protocol.py`
(headset wire format), `video.py` (control server + encoder), `xr_client.py`
(service state → robot frame), `bridge.py` (the process).
