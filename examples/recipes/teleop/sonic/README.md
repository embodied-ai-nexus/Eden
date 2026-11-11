# SONIC whole-body teleoperation (PICO)

Whole-body G1 teleoperation with the GEAR-SONIC policy, driven by a PICO 4
Ultra (headset + two controllers + two ankle trackers). This is Eden's port of
the NVIDIA GR00T-WholeBodyControl sim-teleop stack: the reference's three
processes (MuJoCo sim + DDS, SONIC deploy binary, PICO streamer) become two
Eden processes talking the same byte-compatible ZMQ protocol — the Genesis env
and the policy live in one process, so the DDS hop disappears.

```
YOU (PICO)
  └─WiFi→ XRoboToolkit PC service (:63901)
            └→ eden pico bridge ──ZMQ :5559 (pose, spawned by the operator)
                └→ eden teleop sonic operator ──ZMQ :5556 (pose/planner/command)──→ eden teleop sonic robot
                                              ←──ZMQ :5557 (g1_debug feedback)────┘  (Genesis G1 + SONIC + planner)
```

## Modes (encoder used on the robot side)

| Mode | PICO input | Body | Encoder |
|---|---|---|---|
| PLANNER | A+B+X+Y engages here; sticks to move | joystick locomotion | g1 |
| POSE | A+X | whole body (arms + legs) from SMPL | smpl |
| VR_3PT | Left-stick click | arms follow hands, legs walk on planner | teleop |
| PLANNER_FROZEN_UPPER | B+Y | planner legs, frozen arms | g1 |

## Prerequisites

- `pip install -e '.[sonic,onnx-gpu]'` (or `onnx` for CPU inference), plus
  `xrobotoolkit_sdk` built against the installed XRoboToolkit PC Service (see
  `examples/recipes/teleop/pico/README.md`). The operator viz needs the
  `sonic-viz` extra (pyvista).
- **Assets** download automatically from the HF mirrors on first run: the G1
  MJCF (`humanoids/unitree_g1/g1_29dof_with_hand_sonic.xml` in
  `Kashu7100/eden_assets`) and the SONIC encoder/decoder/planner
  (`sonic/*.onnx` in `Kashu7100/eden_policies`, md5-pinned). `--planner-path`
  / `--encoder-path` / `--decoder-path` override with local files.
- The XRoboToolkit PC service (`/opt/apps/roboticsservice/runService.sh`) and
  a pose-only `eden pico bridge` — the operator process spawns both
  (`--no-service-start` / `--no-bridge-start` to skip; `--bridge-host` to use a
  bridge on another machine; `--sdk` for the legacy in-process SDK reader). No
  `LD_LIBRARY_PATH` export needed — the service library is preloaded.

Port note: the spawned bridge publishes poses on :5559 with video off, so it
coexists with the SONIC wire. Only a *default-port standalone* `eden pico
bridge` (frame :5556 / pose :5557) collides with the sonic 5556/5557 — run it
with `--no-video --pose-port 5559` (what the operator spawns), or move the
sonic side (`--cmd-port` / `--debug-port` on the robot, `--port` /
`--feedback-port` on the operator). The manual 3-terminal variant:

```bash
eden pico bridge --no-video --no-derived-body --pose-port 5559 --fps 120   # T0 (fps ≥ device body rate)
eden teleop sonic operator --no-bridge-start ...                           # T2
```

(`--no-derived-body` skips the per-joint body dict and trackers the SONIC
operator never reads, keeping the bridge fast enough that the 50 Hz pose
resampler never starves; drop the flag if another consumer shares the bridge.)

## Start it (2 terminals)

```bash
# T1 — robot (Genesis viewer + SONIC policy + planner); --mujoco / --real: see below
eden teleop sonic robot --vis

# T2 — operator (PICO body tracking; add --robot-host <ip> across machines)
eden teleop sonic operator --vis-vr3pt --vis-smpl
```

PICO app: same WiFi as this PC, PC Service = this host's IP → **Status:
WORKING**; Tracking = Head + Controller; Data/Control = Send; Pico Motion
Tracker = Full body; ankle trackers paired and calibrated. Before engaging,
confirm the SMPL avatar in the `--vis-smpl` window mirrors you.

## Engage POSE (full body)

1. Stand in the calibration pose: upright, feet together, upper arms down,
   **forearms bent 90° forward**, palms in (this *is* the G1's zero-joint pose).
2. **A+B+X+Y** → engage + calibrate (enters PLANNER; the robot still hangs
   from the elastic band at the origin).
3. Click the robot terminal → press **9** → drop the robot. The policy
   catches and stands it.
4. Align your body to the robot's standing pose → **A+X** → POSE.
5. Disengage: **A+B+X+Y**. The robot is handed to the planner's IDLE and keeps
   balancing in place; A+B+X+Y again re-engages with a full calibration (stand
   in the neutral pose again, looking ahead). Ctrl-C exits the operator and
   leaves the robot idling.
6. End the session from the robot terminal: `o` (or Ctrl-C; KILL under a
   deployment body).

## Controls

| PICO input | Action |
|---|---|
| A+B+X+Y | engage + calibrate / disengage (robot idles on the planner) |
| A+X | toggle POSE ↔ PLANNER |
| B+Y | toggle PLANNER_FROZEN_UPPER |
| Left-stick click | toggle VR_3PT |
| Left / right stick | move / turn (planner modes) |
| A+B / X+Y | next / previous locomotion mode |
| Trigger (per hand) | close/open that hand (Dex3) |
| Menu (hold) | pause the pose stream (re-align before releasing) |

Robot-terminal keys: `9` toggle elastic band, `7`/`8` band length, `o` e-stop,
`i` re-capture heading, `q`/`e` delta heading ±0.1 rad. A headless host has no
keyboard — `--drop-band-after SECONDS` releases the band on a timer instead.

## VR_3PT: calibration and recalibration

VR_3PT drives the robot's hands from your tracked wrists and neck, relative
to your pelvis. Two moments set the mapping, and a bad one skews everything
after it:

1. **A+B+X+Y (full calibration)** captures your neck orientation (it rotates
   every later target) and your wrists against the robot's zero pose. Stand
   straight, **look ahead — not down at the controllers** — upper arms down,
   forearms forward, palms in. The operator prints the neck angle it captured
   and warns above 20°.
2. **Left-stick click (entering VR_3PT)** re-captures only the wrists: your
   hands at that instant become "where the robot's hands are now". Click with
   your hands held like the robot's (forearms forward, palms in); a large
   wrist mismatch there makes hand rotations carry over about skewed axes.

Then keep your hips facing forward (turn the robot with the right stick), and
move within the robot's reach — targets are not scaled to its shorter arms.
`--vis-vr3pt` on the operator shows the targets the robot receives.

To recalibrate the wrists, click the left stick out of VR_3PT and back in. To
recalibrate the neck too, press A+B+X+Y (disengage — the robot keeps standing)
and A+B+X+Y again in the neutral pose.

Health diagnostics: the robot prints a loop line once per second (achieved Hz,
work time, streamed-motion lag in frames — steady ≈46 in g1/teleop mode, ≈10
in POSE — plus mid-stream catch-up resets and deadline resyncs); the operator
prints `[BridgeReader]`/`[PicoReader]` device rates and `[PoseLoop] FPS` every
5 s, with a warning when the pose emit rate sags below 90% of target (that is
exactly when the robot-side stream starts to stall).

With `eden teleop sonic robot --auto-reset`, any fall while the band is off (base
up-axis 60°+ off the commanded posture for 0.4 s — the reference is the
active motion when engaged, upright for the disengaged standing hold, so
crawling/lying modes don't trigger) resets the session to the launch state:
robot back on the band at the origin, operator dropped to disengaged with a
console prompt to re-engage (A+B+X+Y, full recalibration). The reset is
signalled through a `reset_count` key in `g1_debug`; the reference PICO
streamer ignores it and would stay engaged — re-engage manually there.

## Sim-to-sim and the real robot

The same robot process drives two more bodies, through the deployment backends
`eden deploy --policy sonic` runs a motion clip on. Both are the **body-only
29-DOF G1** at the checkpoint's gains (that deploy's robot), so the Dex3 hands
are not driven and the triggers do nothing; the operator side is unchanged.

```bash
eden teleop sonic robot --mujoco --vis     # MuJoCo sim-to-sim (MuJoCo's own viewer)
eden teleop sonic robot --real             # the real Unitree G1 over DDS (--vis mirrors it in Genesis)
```

The deployment keeps its own operator sequence, so there are two gates: the
robot boots in STOP holding its pose, **INIT** ramps it to the SONIC stand,
**START** hands control to this loop, and only then does the PICO's
**A+B+X+Y** engage the policy. A START always asks for a fresh engage — also
after a STOP — because a session that kept running through the hold tracks a
motion the robot never followed.

| | `--mujoco` | `--real` |
|---|---|---|
| INIT / START / STOP / KILL | `i` / `s` / `x` / `k` + Enter in the robot terminal (a run with no TTY presses INIT, then START, itself) | G1 remote `Y` / `A` / `B` / `L1+R1`, or the same keys |
| Support before engaging | simulated gantry, on by default; `9` (or `--drop-band-after`) releases it, `+` / `-` + Enter move the anchor, `--no-band` starts without it | hang the robot from a real gantry |
| `--auto-reset` | yes — back on the gantry in STOP; INIT and START again | rejected |
| Base position in `g1_debug` | ground truth | zeros (an IMU measures none) |

Flow under `--mujoco`: `i` + Enter, `s` + Enter, A+B+X+Y on the PICO, then `9`
to release the gantry. On the robot: support it, INIT, START, engage, then
lower it onto its feet. The single-key bindings above (`9`, `o`, `i`, `q`,
`e`) are a global listener, so pressed in the robot terminal they are also
typed into the line the deployment reads; the robot process clears that line
right after, so they do not garble the next `x` / `k` + Enter. The reverse
overlap is `i`: INIT also requests a heading re-capture, harmless before the
policy is engaged.

Deployment flags keep their `eden deploy` spelling: `--deploy_kp_kd_scale`
(ramp 0.2 → 1.0 when commissioning), `--target_lpf`, `--disable_joints`,
`--debug_record` / `--debug_record_path`. `--joint-vel-limit` (default 35
rad/s, the reference deploy's runaway trip; `--real` only) ends the session
with the backend's soft damping when any body joint exceeds it, as does `o`.
A pose jump at START is avoided the same way on both: the pre-engage hold
approaches the stand from the measured pose at 0.5 rad/s.

`--real` needs `unitree_sdk2py` and has been exercised only against an
in-memory wire, not on a robot — commission it at a low `--deploy_kp_kd_scale`
with the robot supported.

## Deviations from the reference (documented, not behavioral)

- One robot process instead of MuJoCo-sim + deploy-binary over DDS; the
  operator-facing protocol and all mode/encoder semantics are unchanged, so
  the Eden robot also works against the reference PICO streamer (and vice
  versa) for A/B checks.
- Inference runs on onnxruntime (CUDA/TensorRT EP via the usual env flags)
  instead of TensorRT-with-CUDA-graphs.
- Pose protocol v4 (external token streaming, a VLA path) is rejected with a
  clear log instead of consumed.
- A+B+X+Y while engaged **disengages** instead of stopping: the reference
  sends `stop`, which ends the robot process with a damping-only command (an
  unsupported real robot falls); here the robot is handed to the planner's
  IDLE and keeps balancing, and the session ends from the robot side. The
  robot still honours a reference streamer's `stop`.

The plant itself is reference-parity: PD gains from the deploy binary's
`policy_parameters.hpp` (including the 7520-22 hip-pitch motor of the released
checkpoint), MJCF-authored armature (uniform 0.01 like the reference MuJoCo
sim), the reference sim's per-joint torque clamps, multi-point foot contacts,
and MuJoCo's default contact time constant (0.02 s).
