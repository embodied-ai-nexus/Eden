# Eden ⇆ Meta Quest XR teleop

Stream Meta Quest tracking into Eden (Genesis backend) and render the **sim
scene back to the headset** in stereo, using Eden's built-in
[`televuer`](https://github.com/Kashu7100/televuer) XR bridge. The bridge runs
in one of two **mutually exclusive** input modes (WebXR exposes hands and
controllers as separate input sources):

| Mode (`--tracking`) | Tracks | Demo |
| --- | --- | --- |
| `hand` (default) | head / wrist poses + 25 hand-joint positions + pinch/squeeze | `demo.py` |
| `controller` | head / wrist (grip) poses + per-hand trigger, grip, A/B buttons, thumbstick (click + 2D axis) | `controller_demo.py` |

Every published pose dict carries a `"mode"` key (`"hand"` / `"controller"`) so
the Eden side knows which fields to expect.

> Using a **PICO** headset? `eden pico bridge` publishes the same controller
> schema over the same ZMQ channels, so `controller_demo.py` and
> `controller_teleop.py` below run unchanged against it — see
> [`../pico/README.md`](../pico/README.md).

Two processes talk over **ZMQ on localhost**, both in Eden's `uv` env:

```
 Meta Quest ──WebXR/Wi-Fi──▶ eden metaquest bridge
                              │  poses  (PUB :5557) ──▶ demo.py
                              ◀── frames (PULL :5556) ──┘  Genesis sim + stereo cameras
```

Poses are published in the **Genesis world frame** (z up, x forward, y left,
metres, origin = XR tracking origin), so the Eden side consumes them directly.

## Setup

Install the `metaquest` extra (pyzmq) and televuer into Eden's env:

```bash
uv sync --inexact --extra metaquest --group metaquest-git   # televuer is not on PyPI: the group installs it
```

WebXR requires HTTPS, so televuer needs a TLS cert/key reachable by the headset
(a self-signed pair with a SAN covering this host's LAN IP works; the Quest
browser will prompt to accept it the first time).

## Run

**Terminal 1 — XR bridge** (serves the Quest, boots a Vuer WebXR server on `:8012`):

```bash
eden metaquest bridge
# default 1280x480 stereo @ 30fps; override with --width/--height/--fps
```

It prints the exact URL(s) to open on the Quest, e.g.
`https://172.26.79.101:8012/?ws=wss://172.26.79.101:8012`.

**Terminal 2 — Eden sim** (renders the scene, drives markers from your hands):

```bash
uv run python examples/recipes/teleop/metaquest/demo.py            # GPU
uv run python examples/recipes/teleop/metaquest/demo.py -c
# --bridge-host <ip> if the bridge runs on another machine
```

**On the Quest:**

1. Join the **same Wi-Fi/subnet as this host**.
2. Open the printed `https://<host-ip>:8012/?ws=wss://<host-ip>:8012`. First time:
   accept the self-signed cert (**Advanced → Proceed**).
3. Tap **Virtual Reality** and allow all prompts — you should see the sim in stereo.
4. 25 spheres track each hand (blue = left, red = right). **Pinch** (thumb+index)
   near the yellow cube to grab it; release to drop it.

## Controllers instead of hands

Run the bridge in controller mode and use the controller demo:

```bash
# Terminal 1
eden metaquest bridge --tracking controller

# Terminal 2
uv run python examples/recipes/teleop/metaquest/controller_demo.py
```

A marker sphere follows each controller (blue = left, red = right) with a small
tip showing its forward direction. The cube is driven through
[`ClutchPoseMapper`](../../../../eden/utils/devices/metaquest/pose_mapping.py)
(clutch-relative mapping ported from
[vr-teleop-kit](https://github.com/Dream-Machines-Robotics/vr-teleop-kit),
Apache-2.0):

- **Grip = clutch** — hold a grip and the cube follows that hand's *relative*
  motion (position + orientation); release to reposition your hand without
  moving the cube, then grab again. Only one hand drives at a time.
- **Trigger = precision** — hold to scale the response down for fine work.
- **A button** — home the cube (either controller).

The cube chases the mapper's target at a capped speed (a stand-in for a real
arm's velocity limit), so the mapper's **reach limit / "slipping clutch"** is
visible: flick fast and the target runs ahead, clamps, and the overshoot is
absorbed — reverse and the cube bites immediately instead of retracing it. For a
real arm you pass its measured end-effector pose into `mapper.target(...)`
instead of the simulated follower.

## Bimanual robot arms (YAM) instead of a marker cube

`controller_teleop.py` is the real-arm build: two i2rt **YAM** 6-DOF arms, one per
controller, mounted on a table picking at a few objects (cube, block, brick,
ball). The controller pose becomes an end-effector *target* tracked through
inverse kinematics, so the clutch mapper's reach limit does real work (the arm
genuinely lags the hand). The arms use `gravity_compensation` so the PD position
controller holds the IK target without gravity sag.

```bash
# Terminal 1
eden metaquest bridge --tracking controller
# Terminal 2
uv run python examples/recipes/teleop/metaquest/controller_teleop.py            # GPU
uv run python examples/recipes/teleop/metaquest/controller_teleop.py -c
```

- **Grip = clutch** — hold to drive that arm's end-effector relative to your hand
  (position + orientation); release to reposition. Both hands drive their own arm
  independently → true bimanual control.
- **Trigger = gripper** — analog: squeeze to close, release to open.
- **A button** (right controller) — reset both arms to the home pose.
- **X button** (left controller) — reset the objects to their start poses.

The pipeline each tick is `controller pose → ClutchPoseMapper.target(.., current EE
pose) → RigidEntity.inverse_kinematics("link_6") → control_dofs_pos(arm joints)`.
To drive a different arm, swap `en.robots.Yam` and the `EE_LINK` / joint names at
the top of the file. (Base placement, grasp force, and gripper jaw tuning are
geometry-specific — adjust `ARM_BASES` / `OBJECT_SPECS` for your workspace.)

## Using a real Eden task instead of the toy scene

`demo.py` builds a throwaway scene in `build_env()`. To teleoperate a real
task, swap that for a registered task env --- `RLEnvBase.from_config(cfg)` --- add a
stereo camera pair via the config's `cameras_options`, and keep the same
pose→action / render→stream loop. The wire protocol lives in
`eden.utils.devices.metaquest.protocol` and is shared by both the bridge and the
Eden side, so there is a single source of truth for ports and framing.
