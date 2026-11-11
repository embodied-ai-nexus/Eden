# Robot setup for deployment

This page is the one-time setup and the robot-specific behaviour of each hardware backend
`eden deploy` drives: the SDK to install, how the robot is reached, its remote, and what its wire
does differently. Read [Deployment](deployment.md) first for the loop, the operator state machine
and the safety mechanisms every backend shares. None of this applies to `eden deploy --mujoco`,
which needs no SDK.

| Robot | Backend | SDK (Python import) | Operator inputs |
|---|---|---|---|
| [Unitree G1](#unitree-g1) | `UnitreeG1Deployment` | `unitree_sdk2py` + CycloneDDS | G1 remote, keyboard |
| [Unitree Go2](#unitree-go2) | `UnitreeGo2Deployment` | `unitree_sdk2py` + CycloneDDS | keyboard |
| [Unitree Go1](#unitree-go1) | `UnitreeGo1Deployment` | `robot_interface` (legacy `unitree_legged_sdk`) | keyboard |
| [LimX Oli](#limx-oli) | `LimxOliDeployment` | `limxsdk` | LimX remote, keyboard |
| [Franka Panda / FR3](#franka-panda-and-fr3) | `FrankaPandaDeployment` | `panda_py` | keyboard |
| [RoboTera XHand](#robotera-xhand) | `RoboTeraXHandDeployment` | `xhand_controller` | keyboard |
| [Wuji Hand 1 / Hand 2](#wuji-hand-1-and-hand-2) | `WujiHandDeployment` | `wujihandpy` / `wuji_sdk` | keyboard |

Every SDK is imported at `connect()` (or guarded at import), so a missing one never breaks
`import eden`; the backend raises with an install hint when it connects.

## Before any robot

### ONNX runtime

On a G1 29-DOF, `eden deploy` preloads the Holosoma locomotion policy as the boot-active safety slot
(`--base_policy auto`), and Holosoma, like `sonic` and `omnixtreme`, is an ONNX policy. Install one
of the two mutually exclusive runtime extras:

=== "pip"

    ```bash
    pip install "eden-world[onnx-gpu]"   # onnxruntime-gpu; needs a matching CUDA/cuDNN
    pip install "eden-world[onnx]"       # CPU wheel
    ```

=== "uv (checkout)"

    ```bash
    uv sync --inexact --extra onnx-gpu
    uv sync --inexact --extra onnx
    ```

Without it, the policy fails at startup, before anything connects, with `onnxruntime is not
installed` and both install commands.

The two wheels share the `onnxruntime/` package directory. To switch, reinstall the one you want
on top of the other: `pip install --force-reinstall --no-deps "onnxruntime-gpu==<its version>"`, or
`uv sync --inexact --extra onnx-gpu --reinstall-package onnxruntime-gpu` in a checkout (the error
Eden prints gives the exact command). **Never uninstall either wheel**: that deletes files the other
needs, and Eden then fails fast on the half-removed install with a message saying so.

The CPU wheel is fast enough for the deployed policies, because Eden bounds every session's thread
pool (see [Control-loop timing](deployment.md#control-loop-timing)).

### SDKs live outside the lockfile

The robot SDKs below (except Wuji Hand 2's, which is the `wuji` extra) are installed into the venv
by hand and are not declared in `pyproject.toml`. A plain `uv sync` removes packages the lockfile
does not list, these included: reinstall them after one, or sync with `--inexact`.

### State-check recipes

Before running a policy on a new robot, run its state-check recipe under
`examples/recipes/deployment/`: each reads the real robot through its backend, mirrors it in sim and
prints a side-by-side comparison of every quantity the observation terms use, flagging mismatches.

| Recipe | Robot |
|---|---|
| `real_g1_states.py` | Unitree G1 |
| `real_oli_states.py`, `real_oli_teleoperation_slider.py` (joint sliders in the viewer) | LimX Oli |
| `real_franka_states.py`, `real_franka_teleoperation_keyboard.py` | Franka Panda |
| `real_xhand.py` (same sinusoid to sim and real) | XHand |
| `real_wuji_hand2.py` (open and close, reports tracking) | Wuji Hand 2 |

The G1 and Franka recipes are run with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`, as
their headers show. The headers do not say why. If `eden deploy` on the same machine fails to import the robot's
SDK with a `GLIBCXX_...` not found error, preload the same library.

## Unitree G1

`UnitreeG1Deployment` drives `G1Dofs23`, `G1Dofs29` and `G1Dofs29NoHands` over the Unitree SDK2
DDS low-level interface (`unitree_hg`): `LowState_` on `rt/lowstate`, a CRC-stamped `LowCmd_` on
`rt/lowcmd`, joints mapped to motor slots by name.

### Install the SDK and CycloneDDS

The backend imports `unitree_sdk2py`, which needs the CycloneDDS native library (`libddsc.so.0`).
When it is missing, `connect()` raises:

```text
RuntimeError: Unitree SDK (unitree_sdk2py) failed to import. Install it and ensure it
(and its native deps, e.g. libddsc.so.0 from CycloneDDS) are on the loader path.
Underlying error: ModuleNotFoundError: No module named 'unitree_sdk2py'
```

1. **Build CycloneDDS** once and point `CYCLONEDDS_HOME` at the install prefix; the `cyclonedds`
   Python wheel the SDK pins (`cyclonedds==0.10.2`) builds against it.

    ```bash
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
    cmake -S cyclonedds -B cyclonedds/build -DCMAKE_INSTALL_PREFIX=$HOME/cyclonedds/install -DBUILD_IDLC=ON
    cmake --build cyclonedds/build --target install

    # persist it (the loader and the cyclonedds build both read this)
    echo 'export CYCLONEDDS_HOME=$HOME/cyclonedds/install' >> ~/.bashrc
    export CYCLONEDDS_HOME=$HOME/cyclonedds/install
    ```

    `$CYCLONEDDS_HOME/lib` must contain `libddsc.so.0`.

2. **Install the SDK** into the Eden venv, editable so it stays linked to the checkout:

    ```bash
    git clone https://github.com/unitreerobotics/unitree_sdk2_python
    CYCLONEDDS_HOME=$HOME/cyclonedds/install uv pip install -e ./unitree_sdk2_python
    ```

3. **Verify** both import:

    ```bash
    uv run python -c "import cyclonedds, unitree_sdk2py; print('OK')"
    ```

If the SDK was installed without its native CRC library (`unitree_sdk2py/utils/lib/crc_*.so`),
Eden falls back to the SDK's pure-Python CRC and logs a warning; it is slower but well within a
50 Hz loop's budget.

### Network interface

The DDS channel binds to the first wired interface (`en*` / `eth*`) that is up, skipping loopback
and virtual ones (`docker*`, `br-*`, `veth*`, `tailscale*`, `tun*`, `tap*`). Override it with
`GS_UNITREE_NETWORK_INTERFACE` (also read: `UNITREE_NETWORK_INTERFACE`, `CYCLONEDDS_NETWORK_INTERFACE`),
or the backend option `network_interface`. With no usable interface, `connect()` raises naming the
variable. The same applies to the Go2.

### Connecting

`connect()` first releases any high-level motion mode the robot is running (through the SDK's
`MotionSwitcherClient`), then waits up to `connect_timeout_s` (`eden deploy` passes 10 s) for the
first `LowState_`. Nothing goes on the wire before the first command. <!-- TODO(material): the G1's physical bring-up procedure (hoist, damping/debug mode on the remote) before connect -->

Remote: **A** START, **B** STOP, **Y** INIT, **L1 + R1** KILL, **Select** next policy slot (L1 +
Select is ignored so it does not collide with other chords). The keyboard keys work alongside it.

### Publish rate on the G1

The G1 has parallel-linkage joints, the ankles always and, on the 3-DOF-waist 29-DOF build, waist
roll and pitch too, driven in PR (pitch/roll) space via `LowCmd_.mode_pr`. The per-packet PD
mechanism described in [Republishing the wire command](deployment.md#republishing-the-wire-command)
applies, so the backend re-sends the held `LowCmd_` between control ticks at `publish_rate`,
**default 500 Hz**. 500 Hz is the rate both vendor references feed the robot at, independent of any
policy rate: Unitree's own `g1_low_level_example` publishes at `control_dt_ = 0.002`, and NVIDIA's
SONIC G1 deploy runs a dedicated 500 Hz command-writer thread beside its 50 Hz control thread.

!!! warning "Inferred on the G1, measured on the Oli"
    The per-packet damping fault was measured by a torque fit on Oli hardware captures. Nothing
    equivalent has been measured on a G1: the mechanism and the 500 Hz default come from the two
    vendor references. `--publish_rate 0` restores one packet per tick, which is the A/B to run
    before concluding anything about a particular robot. The hardware example scripts under
    `examples/policies/` take the same flag, spelled `--publish-rate`. `--debug_record` writes the
    commanded and measured history (including `stamp`), and the re-send loop logs the packet rate
    it achieved when it stops.

On shutdown the backend holds the last pose it read at kp = 0 for `shutdown_damping_steps` ticks,
with the re-send loop still running, before it drops the channel.

## Unitree Go2

`UnitreeGo2Deployment` drives `Go2` over the same SDK2 DDS stack as the G1 (`unitree_go`), so the
[G1 install](#install-the-sdk-and-cyclonedds) and [network](#network-interface) steps apply
unchanged. It has no remote reader: use the keyboard.

- **Publish rate.** The backend re-sends its held `LowCmd_` at `publish_rate`, **default 500 Hz**,
  for parity with the vendor, not to fix a fault: the Go2's twelve joints are all serial (its
  `LowCmd_` has no `mode_pr`), so the parallel-linkage mechanism does not apply. 500 Hz is what
  Unitree's own Go2 low-level example feeds the robot (`go2_stand_example.py`: `dt = 0.002`, written
  from a 2 ms `RecurrentThread`). `--publish_rate 0` sends one packet per tick. The re-send loop's
  health reporting applies unchanged.
- **Go2W** inherits this binding but is refused at construction: its wheel motors have no confirmed
  SDK motor index or velocity-mode mapping, and guessing them on real hardware is unsafe.

## Unitree Go1

The Go1 predates the DDS stack. `UnitreeGo1Deployment` uses the **legacy `unitree_legged_sdk`**,
exposed as the `robot_interface` pybind module (synchronous UDP, not CycloneDDS), so the CycloneDDS
setup above does **not** apply.

### Setup

- Build and install the `robot_interface` module from the `unitree_legged_sdk` Python wrapper
  (`python -m build --wheel`, then install the wheel into the venv), **or** point
  `EDEN_GO1_SDK_PATH` at the directory holding `robot_interface.*.so`:

    ```bash
    export EDEN_GO1_SDK_PATH=/path/to/unitree_go1_sdk
    uv run python -c "import robot_interface; print('OK')"
    ```

    When missing, `connect()` raises `RuntimeError: Unitree Go1 SDK (the legacy 'robot_interface'
    module) failed to import`, with both options.
- **No ONNX runtime is needed**: Holosoma is a G1-only policy, and `--base_policy auto` loads no
  base policy on a Go1, so your `--checkpoint` is slot 1.
- Put the robot in **low-level mode** on the remote: L2 + A → L2 + B → L1 + L2 + start.

### Running

```bash
eden deploy --task move_forward --robot go1 --checkpoint /path/to/model.pt
```

- `connect()` primes the bidirectional UDP link: the robot only streams state to a client it has
  heard from, so the backend sends a limp command (kp = kd = tau = 0) while it waits for the first
  state. A timeout means the robot is not powered, not in low-level mode, or not reachable at
  `robot_ip:robot_port` (default `192.168.123.10:8007`) from local UDP port `local_port` (default
  8080).
- `--power_protect_factor N` sets the SDK's `Safety.PowerProtect` factor, `1`..`10` = 10%..100% of
  rated motor power, **default 10**. Lower factors clamp the torque a gait needs (the backend's
  source records factor 7 freezing the `move_forward` Go1 mid-stance), and the SDK examples' value
  of `1` starves locomotion entirely and trips `Error: Power Protection`. Lower it (for example to
  `5`) only for cautious bench bring-up with the robot supported, then raise it back toward `10`.
- **Stand-up.** INIT is a joint-space ramp that is blind to where the feet are (see
  [Gains and the INIT ramp](deployment.md#gains-and-the-init-ramp)). If the robot starts from an
  awkward stance, some legs fight the ground to reach their target angles; that shows up as a large
  *persistent* tracking error on those joints and drives their motors into thermal derating, so
  the leg goes weak and the robot freezes or sags. Lift the robot and set it down squarely, feet
  roughly under the hips, before INIT; wait for `INIT reached the target pose` and confirm full
  height before START.

### Monitoring

- `--debug_record` additionally records, per step, each motor's temperature and mode and the
  battery (`x_motor_temp`, `x_motor_mode`, `x_battery_v`, `x_battery_soc`).
- The backend warns, once per joint until it recovers, when a motor reports Overheat mode or reaches
  `motor_temp_warn_c` (default 80 °C). A derating motor is the usual cause of a frozen or weak leg:
  let the motors cool between runs.
- If a leg is stuck with its measured `tau` far below the commanded PD demand
  (`kp·(cmd − q) − kd·qd`, reconstructable from the record), the motor is limiting itself
  (temperature, battery), not the controller or the policy.

!!! danger "Go1 safety"
    `connect()` touches the real robot. Keep the e-stop in reach and the robot supported for first
    runs. The default `--power_protect_factor 10` (100%) leaves **no** PowerProtect margin; dial it
    back while commissioning. Do not disable load-bearing leg joints on an unsupported robot.

## LimX Oli

`LimxOliDeployment` drives `Oli` (HU_D04) through LimX's low-level SDK, `limxsdk`.

### Install

The SDK is not on PyPI; install the platform wheel shipped in the
[`limxsdk-lowlevel`](https://github.com/limxdynamics/limxsdk-lowlevel) repo:

```bash
git clone https://github.com/limxdynamics/limxsdk-lowlevel
uv pip install limxsdk-lowlevel/python3/amd64/limxsdk-*.whl   # aarch64/ on the onboard PC
uv run python -c "import limxsdk.robot, limxsdk.datatypes; print('OK')"
```

### Address

The backend connects to the `robot_ip` option, else the `ROBOT_IP` environment variable, else
`10.192.1.2` (the robot's address on its own network). `ROBOT_IP=127.0.0.1` targets the
[LimX MuJoCo simulator](https://github.com/limxdynamics/humanoid-mujoco-sim), which speaks the same
protocol on localhost.

### Hardware preparation

Follow LimX's procedure; **both steps are required**. Hang the robot, power it on, **click the right
joystick button** on the remote to *activate the motors*, then press **R1 + DOWN** to enter
*developer mode* (it persists across reboots; exit with R1 + LEFT).

The failure modes are distinctive. Outside developer mode the onboard motion controller keeps
commanding the motors, so SDK commands *fight* it: the motors buzz and shake. In developer mode
with the motors not activated, SDK commands are silently ignored: the robot stays limp. From the
moment a deploy boots (STOP, holding), the robot should feel **stiff**; limp means the activation
step is missing. The backend prints this at `reset()` too.

### Remote

The remote is PS4-style (□ square, △ triangle, ✕ cross, ○ circle); LimX's docs and SDK comments name
the same buttons with Xbox-style letters (✕ = A, ○ = B, □ = X, △ = Y). The chords follow LimX's own
deploy tooling where it has a counterpart:

| Chord | Command |
|---|---|
| L1 + △ | INIT |
| R1 + □ | START |
| L1 + ✕ | STOP |
| L1 + □ | KILL (soft-damp and exit) |
| L1 + ○ | Next policy slot |

### Wire behaviour

- **Joint order by name.** Motor order is mapped by *name* from the robot's live `getMotorNames()`
  at connect, never by position (LimX's docs and controller files disagree on the head-joint
  order). If a robot reports names that do not match the URDF joint names even after `_joint`-suffix
  normalization, map them with the `motor_name_map` option (Eden DOF name → SDK motor name).
- **PR space.** Commands are sent in PR (serial-joint) space with the SDK's
  `parallel_solve_required` flag set, so the SDK handles the parallel ankle and waist linkages,
  matching Eden's serial `HU_D04_01.urdf` and LimX's own RL deployment stack.
- **Publish rate is load-bearing on this robot.** In PR mode the robot evaluates the PD law of its
  six parallel-linkage joints (ankle pitch/roll, waist roll/pitch) once per received packet, on
  that packet's state. Measured on hardware (recorded in
  `eden/extensions/deployment/utils/republisher.py`): with one packet per tick their torque fits
  `kp·(q_des − q) − kd·dq[previous packet]` at R² 0.92–0.99 while the serial joints fit the
  same-instant velocity, and the waist-roll linkage rang at 25 Hz once a foot strike kicked it; at
  1 kHz the audible buzz is gone and the >10 Hz torque RMS drops from 23.7 to 2.1 N·m. The backend
  re-sends the held command at `publish_rate`, **default 1000 Hz** (LimX's own rate).
  `--publish_rate 0` sends one packet per tick, for A/B only.
- **Torque clamp.** Every commanded target is clamped to `q ± soft_torque_limit · budget / kp`
  around the measured pose, so the *position* term of the implied PD torque never exceeds the
  per-joint budget (`soft_torque_limit` defaults to 0.95, like LimX's walk controller; 0 disables
  it). The budget is the robot's `default_dofs_force_limits` with the `clamp_torque_limits` option
  on top, whose default is LimX's deploy-time `user_torque_limit` where that differs from Eden's
  training limits: 80 N·m on the six parallel ankle and waist joints (each driven by two 42 N·m
  motors), while training, and the action scale derived from it, keeps 42. The clamp runs after the
  target low-pass, on every command path (INIT, hold, policy). Unlike LimX's clamp, the window is
  **not** shifted by `kd·dq/kp`: theirs is re-evaluated every 1 ms with fresh state, Eden's once
  per 20 ms tick on the previous tick's state, and while the waist rang the SDK's reported velocity
  for the parallel joints was 8–11× the derivative of the reported position, so that term pinned
  the command to the window edge on 11–28% of walking ticks and amplified the oscillation.
- **Diagnostics.** Error-level robot diagnostics (motor faults, overheat, …) are logged as
  `[deploy] robot diagnostic '<name>': level=… code=…` whenever they change. Set the
  `verbose_diagnostics` option (`--verbose-diagnostics` on the slider recipe) to also log warning
  and OK transitions.

`--menagerie` under `--mujoco` loads LimX's own MJCF for the Oli, with the linkages as equality
constraints; see [Which model is loaded](deployment.md#which-model-is-loaded).

## Franka Panda and FR3

`FrankaPandaDeployment` drives `FrankaEmikaPanda`, `FrankaResearch3` and `FrankaEmikaPandaNoHand`
through `panda-python` (`panda_py`): the 7-DOF arm on panda-py's real-time `JointPosition`
controller at 1 kHz inside libfranka, the Franka Hand through libfranka's blocking `Gripper` API
from a worker thread, so a step never blocks on the gripper.

### Install and prepare

- `panda-python` wheels ship inside per-libfranka zips on panda-py's GitHub releases, with no
  direct wheel URL; install the one whose libfranka matches your controller's server version (server
  v4 → libfranka 0.8.0). The comment above the `franka` extra in `pyproject.toml` has the
  download-and-install commands for that case. Without it, `connect()` raises `panda-python package
  is not installed`.

- Activate FCI in Desk and release the joint brakes.
- A PREEMPT_RT kernel is strongly recommended for stable 1 kHz control; without it, expect
  occasional `communication_constraints_violation` reflexes.
- `robot_ip` defaults to `192.168.1.1`.

### Behaviour

- **Gains.** The arm runs panda-py's own stiffness (`arm_stiffness` / `arm_damping`, panda-py's
  defaults when unset), not the command's, so `--deploy_kp_kd_scale` and `--disable_joints` do not
  reach it, and each policy slot runs at the same gains.
- **Smoothing.** `arm_filter_coeff` (default 0.05; panda-py's own is 1.0) low-passes the setpoint
  inside the 1 kHz loop: at policy rates the controller sees abrupt setpoint changes, which make
  the arm buzz under PD. Lower it if it still oscillates, raise it toward 1.0 for responsiveness.
- **Velocity clamp.** Each step's commanded joint delta is scaled down, direction preserved, so the
  implied joint velocity stays within `velocity_safety_factor` (default 0.5) of the datasheet
  limits; without it, a differential-IK step near a high-Jacobian configuration can trip
  libfranka's `joint_velocity_violation` reflex.
- **Boot.** The arm boots in STOP holding its measured pose (the first command starts the
  controller at that setpoint); INIT ramps it to `init_target` over `init_steps`, with the
  velocity clamp.
- **The gripper is the policy's alone.** The hold, INIT and the START tick leave it where the policy
  last commanded it, so STOP or INIT never opens a grasp. With `use_grasp_on_close` (default on),
  a target width below `close_width_threshold` (0.04 m) closes with force-controlled `grasp`
  (`gripper_force`, default 20 N), which holds objects during arm motion; wider targets use `move`.
  `target_lpf` smooths the arm only. `skip_gripper_homing` skips homing when an object is already
  held.
- **Position terms only.** Torque-mode action terms (`OperationalSpaceController`) are refused at
  construction; use `DifferentialIKController` or another position-mode term.
- The arm has no IMU: `read_state` reports an identity base orientation.

## RoboTera XHand

`RoboTeraXHandDeployment` drives `XHand1_R` / `XHand1_L` through the vendor's `xhand_controller`
package. It is not on PyPI and no Eden extra installs it: get it from RoboTera with the hand's SDK.

- **Bus.** `protocol` is `"RS485"` (default; `serial_port` `/dev/ttyUSB0`, `baud_rate` 3000000) or
  `"EtherCAT"` (the first EtherCAT port found). `hand_id` 0 selects the first enumerated hand; a
  configured ID that is not present raises, listing the ones that are. CRC failures are retried
  `crc_retries` times (default 3) before raising.
- **The hand runs its own PD on firmware gains**, programmed once at connect from `dofs_kp` /
  `dofs_ki` / `dofs_kd` (defaults 50 / 0 / 25) and `tor_max` (300), all in firmware units with no
  known mapping to N·m/rad; `connect` logs what it programs. Only positions are sent per command. So
  `--deploy_kp_kd_scale` other than 1.0 and `--disable_joints` are refused at construction: lower
  the firmware gains instead. `finger_mode` is 0 powerless, 3 position (default), 5 powerful.
- **Joint velocity is a finite difference** of consecutive positions over wall-clock time (the SDK
  reports none), zero on the first read: noisier and a period later than the simulator's.
- The hand has no IMU: `read_state` reports an identity orientation.

## Wuji Hand 1 and Hand 2

`WujiHandDeployment` (`eden/extensions/deployment/wuji_hand.py`) is bound to `WujiHand1*` and
`WujiHand2Beta2*`, left and right.

### Install

- **Hand 1** needs `wujihandpy` (`pip install wujihandpy==1.5.1`, the version upstream pins).
- **Hand 2** needs `wuji_sdk`, the `wuji` extra: `pip install "eden-world[wuji]"`, or
  `uv sync --inexact --extra wuji` in a checkout. It is Linux-only (manylinux wheels for x86-64 and
  aarch64); elsewhere the extra installs nothing.

Both are imported at `connect`, so neither is needed to construct the backend or to run it against a
simulator.

!!! note "Wuji SDK telemetry"
    Importing `wuji_sdk` starts a background worker that sends anonymous usage events to a
    third-party analytics host. Set `WUJI_TELEMETRY_DISABLE=1` in the environment to turn it off;
    Eden does not set it for you.

### Behaviour

- **Side and joint order come from the joint names.** Both SDKs speak 20 angles, finger-major and
  thumb first, the same for either hand. The backend maps `{right,left}_finger{1-5}_joint{1-4}`
  (Hand 1 and the `wuji-mjlab` task models) and `{r,l}_thumb_cmc_flex` … `{r,l}_pinky_dip`
  (Hand2Beta2) onto that order, so a robot may declare its DOFs in any order. A robot outside those
  classes must name its generation: `WujiHandDeployment.configure(hand="hand2")`.
- **Gains are programmed once, at connect.** Hand 2 takes MIT `mit_kp` / `mit_kd` (one value, or a
  pattern-keyed dict covering all 20) and a current limit `effort_limit_a` (default 1.0 A). Hand 1
  runs its own position loop, so it refuses `deploy_kp_kd_scale` and gains; soften it with
  `effort_limit_a` (default 0.5 A). Both refuse `--disable_joints`.

    !!! warning "Hand 2's default MIT gains are the simulation's"
        Left unset, `mit_kp` / `mit_kd` default to the robot's `default_dofs_kp` / `default_dofs_kd`
        times `deploy_kp_kd_scale`, which for `WujiHand2Beta2` are 50 / 5, simulation gains. The
        hand's firmware ships with kp 3.0, kd 0.05, which is what
        `examples/recipes/deployment/real_wuji_hand2.py` programs. Set them explicitly on real
        hardware, and compare them with what your policy trained at: the deploy warns per slot when
        the two differ.

- **Streaming.** `stream_mode="sync"` (default) interpolates `servo_hz` (default 200) sub-targets
  across each control period and blocks until the period's deadline, chained from the previous one,
  so a tick is one control period and the inference time is absorbed into it (the base rate limiter
  is not stacked on top). `"thread"` ramps from a background thread and the rate limiter paces the
  loop. A tick with nothing to send holds the last target for the period, so a `"sync"` loop never
  spins faster than the control rate.
- **Safety.** Lost joint feedback always stops the run (raising `DeployStop`): on Hand 2 a joint
  feed older than `stale_after_s` (default 0.5 s), on Hand 1 a position read that gets no answer
  within it. A latched hardware fault stops it unless `stop_on_fault=False`. Both are checked
  before every control tick (hold, INIT and policy alike) and before every waypoint of `ease_to`.
  `clip_targets_to_joint_limits` clamps every target to the built robot's range, `ease_to`
  included.
- **Loop order.** The backend reads the hand before the commands publish
  (`sync_before_commands`), and it advances the env's episode clock once per tick, only while the
  policy drives, restarting it at every START (`advances_episode_clock`), so a warm-up gated on it
  runs from the handover however long the hand held and INITed before.
- **No IMU.** The base orientation is the robot's `default_root_quat`, and joint velocity is a finite
  difference of consecutive reads (`vel_lowpass_hz` smooths it).
- **Extending it.** A subclass overrides `sync_scene()` to write anything else it observes into the
  scene after each read (a camera-tracked object) and `check_safety()` to add its own stop
  conditions. `use_transport(transport)` swaps the hardware for a simulator, for sim-to-sim through
  the same code path: any `HandTransport` built with the deployment's `dep.stream`.

## See also

- [Deployment](deployment.md): the deploy loop, state machine, safety, timing and MuJoCo sim-to-sim
- [Installation](../getting-started/installation.md): extras and the `uv` / `pip` install commands
- [Environment variables](../reference/environment-variables.md): every variable Eden reads
- [Teleoperation](teleoperation.md): driving these robots from a human operator
