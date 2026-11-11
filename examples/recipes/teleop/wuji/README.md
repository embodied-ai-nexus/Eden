# Eden ⇆ Wuji Glove

Stream a [Wuji Glove](https://docs.wuji.tech/docs/en/wuji-glove/latest/)'s **hand pose**
and **tactile sheet** into Eden, and drive a dexterous hand in Genesis with them.

Unlike the Meta Quest / PICO headsets, the glove needs **no bridge process**. It talks to
`wuji_sdk` in this process over USB/Ethernet, so `WujiGlove` is a plain Eden device —
construct it, read it once per sim step, close it:

```
 glove ──USB / Ethernet──▶ wuji_sdk ──▶ eden.utils.devices.wuji.WujiGlove ──▶ your loop
                                         (one background callback per stream,
                                          each keeping only its newest frame)
```

## What the glove gives you

| Stream | `WujiGlove` getter | Shape |
| --- | --- | --- |
| `hand_skeleton` | `keypoints()` / `keypoint_poses()` | 21 MediaPipe landmarks, `(21, 3)` m |
| `hand_joint_angles` | `joint_angles()` | the IK solver's `(21,)` human DOFs, rad |
| `tip_poses` | `fingertips()` | 5 fingertip poses (pos + wxyz quat + confidence) |
| `emf_poses` | `emf_poses()` | the raw 5 coil poses behind the skeleton |
| `tactile` | `tactile()` | `(24, 31)` pressure sheet in `[0, 1]` |
| ⤷ node-wise | `tactile_taxels()` | `(index, value)` for just the live taxels, dead ones dropped |
| `tactile_zones` | `tactile_zones()` | the same sheet split into palm + one zone per finger |
| `tactile_binary` / `tactile_residual` | `tactile_binary()` / `tactile_residual()` | `(24, 31)` calibrated contact |
| `tactile_point_cloud` | `tactile_points()` | the 526 real taxels placed on the live skeleton |

Subscribe only to what you need — each stream is another 120 Hz callback into Python:

```python
from eden.utils.devices.wuji import WujiGlove

glove = WujiGlove(side="right", streams=("hand_skeleton", "tactile"), rate_hz=60)
keypoints = glove.get_command()   # (21, 3) torch tensor, Eden hand frame
pressure = glove.tactile()        # (24, 31), masked taxels are nan
glove.close()
```

Every getter returns `None` until that stream's first frame arrives, so a teleop loop can
hold its last pose instead of blocking.

## Frames

Wuji reports hand data in its **wrist frame**, defined anatomically: x radial (toward the
thumb), z proximal (toward the elbow), y by the right-hand rule — which makes **+y the
back of the hand on a left glove and the palm on a right one**. Eden re-expresses it as a
**hand frame** in the Genesis basis:

```
x = distal   (out along the fingers)
y = z x x    (ulnar on a left hand, radial on a right one)
z = dorsal   (out of the back of the hand)
```

so an axis means the same physical direction on either glove. Pass `frame="wuji"` to any
getter for the vendor's raw numbers. What this does **not** buy is left/right equivalence:
a left hand is a mirrored right hand, so y necessarily flips — no right-handed frame can
make the two agree.

> **The wrist frame is local.** A glove reports the hand *relative to the wrist*; it does
> not know where your wrist is in the room. Pair it with an arm, a headset, or a tracker
> for a world-frame wrist pose. Both recipes here park the hand at a fixed point.

## Setup

**1. The extra:**

```bash
uv sync --inexact --extra wuji                    # add --extra dex-retargeter for --retargeter dexpilot
```

`wuji-sdk` ships manylinux wheels only (x86_64 / aarch64), so this is Linux-only.

> **Telemetry.** Importing `wuji_sdk` prints a privacy notice and starts a background
> worker that batches anonymous usage events to a third-party analytics host. Set
> `WUJI_TELEMETRY_DISABLE=1` in your environment to turn it off. Eden does not set it for
> you — it is the vendor's setting to make, and yours to choose.

**2. Plug the glove in** and follow the vendor's
[Getting Started](https://docs.wuji.tech/docs/en/wuji-glove/latest/getting-started/) for
the acquisition board, Ethernet, and host network configuration.

**3. Check it from the terminal** before building any scene:

```bash
eden wuji scan                              # list the gloves on the bus
eden wuji monitor                           # live tip reach + tactile peak, Ctrl-C to stop
eden wuji monitor --side right --seconds 30
```

`monitor` prints one line per tick: whether the glove is solving a pose, each fingertip's
distance from the wrist, and the tactile sheet's active-taxel count and peak. Wiggle your
fingers and press something — if both numbers move, the device layer is good.

## Run the recipes

```bash
uv run python examples/recipes/teleop/wuji/glove_demo.py       # skeleton + tactile, no robot
uv run python examples/recipes/teleop/wuji/glove_teleop.py     # drive a simulated Wuji Hand 2
WUJI_TELEMETRY_DISABLE=1 uv run python examples/recipes/teleop/wuji/glove_real_hand.py  # drive a REAL one
```

**`glove_demo.py`** — the device on its own. It draws the 21 landmarks as a hand skeleton
and swells each fingertip sphere with that finger's tactile pressure: both streams visible
at once, no robot, no retargeting, and no `dex-retargeter` extra. Fork this when you want
raw glove data in a scene.

**`glove_teleop.py`** — input and output in one viewer. The glove's live hand skeleton is
drawn as debug spheres **25 cm to the left**, turned to face the same way as the robot
(fingers up, palm forward) so the mimicry is readable at a glance, and the **Wuji Hand 2
(beta2)** beside it is driven by the retargeted pose. Both fingertip sets carry the same
contact markers, so a grasp shows up on the operator's hand *and* on the robot finger it
maps to. `--no-skeleton` hides the operator's hand if you only want the robot.

```
glove.keypoints()  ->  retarget (--retargeter)  ->  control_dofs_pos
                   \-> draw_hand()  (debug spheres + bones, tactile on the fingertips)
```

**Two retargeters.** `--retargeter wuji` (the default) is the vendor's
`wuji_sdk.RetargetSession.for_hand(HandModel.WujiHand2, side)`: built-in tuning for this
hand, ~0.2 ms a frame, and it follows every joint. `--retargeter dexpilot` is Eden's
`DexRetargeter`, which matches **fingertip vectors only**. On a human-sized hand that leaves
the PIP / DIP joints under-constrained: a half-curled fingertip can also be reached by
bending the MCP alone, and on synthetic curls dexpilot does exactly that (index MCP/PIP/DIP
34/40/29 deg -> 28/45/23, middle -> 54/7/1), where the vendor solver returns 34/48/31 and
43/30/32. Prefer `wuji` for the Wuji Hand 2; dexpilot is the route to *other* hands.

The dexpilot preset's `scaling_factor` is **1.05**, not the 1.5 typical of larger robot
hands: the Wuji Hand 2's straight wrist-to-tip reach is 1.0-1.1x the glove's default hand,
and at 1.5 an open hand outreaches every robot finger (they stay straight) while a scaled
fist is matched by the MCPs alone.

For dexpilot, `keypoints_to_mano` is the piece worth knowing about. `dex-retargeting` normally re-derives
a wrist frame from the keypoints every frame (an SVD over the wrist / index-MCP / middle-MCP
triangle), because a MediaPipe detection arrives in an arbitrary camera frame. The glove's
output is already anchored to the hand, so that estimate is a **constant** —
`eden.utils.devices.wuji.protocol.HAND_TO_MANO`, pinned against the upstream procedure in
`tests/utils/devices/wuji/test_protocol.py`. Swap `--preset` (or the `HAND_PRESETS` entry) to drive a
different dexterous hand (with `--retargeter dexpilot`); `--side left` drives the left one.

**`glove_real_hand.py`** — the same loop, driving a **real** Wuji Hand 2 through
`WujiHandDeployment` (the backend `examples/recipes/deployment/real_wuji_hand2.py` uses).
Each 50 Hz tick retargets the newest glove pose, writes the 20 targets, and `step(None)`
streams 200 Hz sub-targets and reads the hand back. Nothing moves until the glove is
tracking; the hand then eases to your pose over `--ease-seconds` (2 s) and follows. It runs
on the hand's firmware MIT gains (kp 3, kd 0.05) at a 1 A current limit, and prints the
tracking error once a second. Ctrl-C, a hand fault or lost hand feedback disables the motors
(the fingers go limp); a glove pose that stops updating only holds the last target.
`--dry-run` retargets and prints without connecting the hand.

## Gotchas

- **`tactile_binary` / `tactile_residual` read zero until you calibrate.** They come from a
  contact model trained *per SDK user and per glove serial*. Run the vendor's tactile
  calibration ([`7.tactile_calibration.py`](https://github.com/wuji-technology/wuji-sdk/blob/main/examples/python/wuji_glove/7.tactile_calibration.py))
  first. `WujiGlove` logs a warning when you subscribe to either without one. Plain
  `tactile` needs no calibration.
- **`-1.0` is not a pressure.** 210 of the 744 grid cells are dead taxels the hardware pins
  to `-1.0` (a fixed mask, measured stable over 360 consecutive frames on both gloves).
  Eden's getters turn those into `nan`, so `grid.mean()` reports `nan` instead of a plausible
  number dragged toward -1 — reduce with `torch.nanmean` / `np.nanmax`. `state()` is the
  exception: `nan` is not valid JSON, so it ships the raw sentinel.
- **Don't count non-sentinel cells to get the active taxel count.** A live sheet reports
  **534** non-`-1.0` cells, but the vendor's "active" figure — the `tactile_point_cloud` point
  count, and the sum of the zones' active counts — is **526**. Eight cells read real pressure
  yet fall outside the contact model. Both numbers are constants in `protocol.py`
  (`N_UNMASKED_TAXELS`, `N_ACTIVE_TAXELS`).
- **Raw `tactile` is not contact.** An unworn glove still reads well above zero across much of
  the sheet from its own resting deformation (~300 of 534 cells over 0.05, peak ~0.4 with
  nothing touching it). For actual contact detection use `tactile_binary` — which needs the
  calibration above.
- **`hand_joint_angles` is the *human* hand's 21 DOFs**, not a robot's. It is packed
  finger-major (5 thumb DOFs, then 4 each) with the vendor's padding slot dropped. To drive
  a robot hand, retarget from `keypoints()`.
- **Firmware 0.11.1 has no rate control at all.** `rate_hz` warns and leaves every stream at
  the device default (120 Hz) rather than refusing to run — being unable to stream *more
  slowly* is a missing optimisation, not a dead glove. Where the firmware does support it,
  only `emf_poses`, `tactile` and `tactile_zones` are adjustable; everything else is derived
  and raises on `set_rate`, so `rate_hz` retunes the *source* each subscribed stream rides on
  (`STREAM_RATE_SOURCE`) — asking for `tactile_binary` at 30 Hz retunes `tactile` rather than
  silently doing nothing. The rate is shared by every subscriber on the host, Wuji Studio
  included, and resets when the last subscription to that stream closes.
- **Two gloves? Pass `--side`.** The serial encodes it (`WG1J…` = left, `WG1K…` = right), which
  is only ever a hint used to connect to the likely glove first — the device's own
  `hand_side()` decides. With no `--side` and two gloves the device refuses and names both
  serials with their likely sides, rather than guessing.
- **The SDK is multi-client by default**, so Wuji Studio or a recorder can read the glove
  while Eden does. Pass `exclusive=True` to take it alone.
- **SDK users.** The glove's online IK runs against the hand model of the *currently
  selected* `wuji_sdk` user (`~/.wuji`), and tactile calibrations are keyed to it too. The
  default user always uses the built-in URDF; create and calibrate a named user for a
  personal hand model, or pass `hand_model_path=` for a custom URDF.

## Verified on hardware

Checked against two gloves on firmware **0.11.1** (`WG1JA…` left, `WG1KA…` right): the
21-landmark order matches `HAND_LANDMARKS` exactly, the wrist landmark sits at the origin,
`hand_joint_angles` arrives as 5 fingers x 5 slots with only the thumb's fifth slot non-zero,
the zone lengths match `TACTILE_ZONE_SIZES` exactly, and the point cloud is 526 points of
`(x, y, z, pressure)` float32 at stride 16 — every decoder ran clean on live frames.

Glove -> real hand, 2026-10-07: glove `WG1KA…` (firmware 0.12.0, default hand model) and a
right Wuji Hand 2, both on a LimX Oli's onboard switch, PC on the same LAN. Two 60 s runs of
`glove_real_hand.py`: target-vs-measured error, sampled each second, had the worst joint
within 5 deg in ~50/60 and 57/60 samples, peaking at 13 and 8.5 deg on fast moves, with a
1.3-1.5 deg mean over all joints; no hand faults, stale feedback or missed ticks.

## The device code

[`eden/utils/devices/wuji/`](../../../../eden/utils/devices/wuji/):

- `protocol.py` — stream layouts, the MediaPipe landmark table, the wrist → hand basis
  change, and `HAND_TO_MANO`. Pure numpy, imports without `wuji_sdk`, fully unit-tested.
- `glove.py` — `WujiGlove`: connection, device selection, subscriptions, the getters.
- `monitor.py` — the `eden wuji scan` / `eden wuji monitor` commands.
