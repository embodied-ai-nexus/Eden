"""Compare sim vs real sensor readings on a LimX Oli to verify convention consistency.

---
tier: recipe
capabilities: [robot.oli, deploy.real_robot]
hardware: real-robot
optional_deps: []
run: uv run python examples/recipes/deployment/real_oli_states.py
---

Reads real robot state from the LimX Oli (HU_D04) through the LimX low-level
SDK, mirrors the pose in sim, and prints a side-by-side comparison of every
quantity that observation terms use.  Flags any mismatch above a configurable
tolerance.

Read-only: the script subscribes to state / IMU but never publishes a command,
so the robot stays in whatever mode it is in.  Because the deployment operates
in PR (serial-joint) space, the values read back are directly comparable to
the serial ``HU_D04_01.urdf`` sim model — including the parallel-linkage
ankle / waist joints.  On connect the script also prints the robot's live
motor table next to the name-resolved Eden DOF mapping, which is the first
thing to verify on a new unit (LimX's own docs disagree on the head-joint
order, so the mapping is by name, never by position).

The robot IP resolves ``ROBOT_IP`` env var -> ``10.192.1.2`` (real robot);
requires the ``limxsdk`` wheel (see ``docs/guides/deployment.md``).

Usage:
    uv run examples/recipes/deployment/real_oli_states.py
    ROBOT_IP=127.0.0.1 uv run examples/recipes/deployment/real_oli_states.py   # LimX MuJoCo sim
"""

from collections import deque

import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat

import eden as en
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.limx_oli import LimxOliDeployment
from eden.extensions.deployment.utils.state import RobotStateEntity
from eden.options import EnvOptions
from eden.options.robots.limx_oli import Oli
from eden.options.scenes.empty import EmptySceneOptions
from eden.utils.torch import to_numpy

PRINT_EVERY = 50  # print comparison every N steps
WARN_TOL = 1e-2  # flag values whose absolute diff exceeds this
PLOT_EVERY = 2  # refresh live plot every N steps
PLOT_WINDOW = 500  # rolling history length (steps)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _header(title: str) -> str:
    return f"\n  {title}\n  {'-' * len(title)}"


def _compare(
    label: str,
    real: torch.Tensor,
    sim: torch.Tensor,
    names: list[str] | None = None,
) -> str:
    """Return a formatted comparison table string."""
    real_np = real.detach().cpu().numpy().flatten()
    sim_np = sim.detach().cpu().numpy().flatten()
    diff = np.abs(real_np - sim_np)

    lines = [_header(label)]
    if names and len(names) == len(real_np):
        lines.append(f"  {'name':<30s} {'real':>10s} {'sim':>10s} {'diff':>10s}")
        for n, r, s, d in zip(names, real_np, sim_np, diff):
            flag = " <--" if d > WARN_TOL else ""
            lines.append(f"  {n:<30s} {r:>10.4f} {s:>10.4f} {d:>10.6f}{flag}")
    else:
        lines.append(f"  {'idx':<6s} {'real':>10s} {'sim':>10s} {'diff':>10s}")
        for i, (r, s, d) in enumerate(zip(real_np, sim_np, diff)):
            flag = " <--" if d > WARN_TOL else ""
            lines.append(f"  {i:<6d} {r:>10.4f} {s:>10.4f} {d:>10.6f}{flag}")

    lines.append(f"  max |diff| = {diff.max():.6f}")
    return "\n".join(lines)


def _show(label: str, val: torch.Tensor, names: list[str] | None = None) -> str:
    """Return a single-column table (for values with no sim counterpart)."""
    val_np = val.detach().cpu().numpy().flatten()
    lines = [_header(label)]
    if names and len(names) == len(val_np):
        lines.append(f"  {'name':<30s} {'value':>10s}")
        for n, v in zip(names, val_np):
            lines.append(f"  {n:<30s} {v:>10.4f}")
    else:
        for i, v in enumerate(val_np):
            lines.append(f"  {i:<6d} {v:>10.4f}")
    return "\n".join(lines)


def _print_motor_table(dep: LimxOliDeployment, dofs_name: list[str]) -> None:
    """Print the live SDK motor order next to the name-resolved Eden DOF map."""
    motor_names = list(dep._cmd.motor_names)
    slot_to_dof = {slot: dofs_name[k] for k, slot in enumerate(dep._dof_index)}
    print(_header(f"Motor table (robot order, {len(motor_names)} motors)"))
    print(f"  {'slot':<6s} {'SDK motor name':<30s} {'Eden DOF':<30s}")
    for slot, sdk_name in enumerate(motor_names):
        print(f"  {slot:<6d} {sdk_name:<30s} {slot_to_dof.get(slot, '(not driven — held limp)'):<30s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    en.init(backend=gs.cpu, debug=True)

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.02,
            decimation=1,
            show_world_frame=True,
        ),
        scene_options=EmptySceneOptions(
            robot=Oli(),
        ),
        show_viewer=True,
    )
    env.build()

    # Keep a reference to the sim entity *before* the deployment swaps it
    # with a RobotStateEntity (so we can read from both).
    sim_robot = env.entities["robot"]

    dep = LimxOliDeployment(
        env,
        LimxOliDeployment.configure(
            entity_name="robot",
            sync=True,
            connect_timeout_s=10.0,
        ),
    )
    dep.connect()
    env.reset()

    state_entity: RobotStateEntity = dep._state_entity
    device = env.device
    dofs_name = list(sim_robot.dofs_name)
    _print_motor_table(dep, dofs_name)

    # Live plot setup: stream real/sim trends and their mismatch over time.
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.canvas.manager.set_window_title("Oli Real vs Sim Live Streams")
    fig.suptitle("Oli Data Streams (Real vs Sim)")
    ax_rpy, ax_joint, ax_err = axes

    hist_step: deque[int] = deque(maxlen=PLOT_WINDOW)
    hist_rpy_real: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_rpy_sim: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_joint_real: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_joint_sim: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_err: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)

    step = 0
    while True:
        robot_state = dep.read_state()

        # --- Mirror real pose into sim (for viewer) -------------------------
        real_quat_np = robot_state.base_quat
        real_dofs_pos_np = robot_state.dofs_pos
        real_dofs_vel_np = robot_state.dofs_vel

        sim_robot.set_quat(real_quat_np)
        sim_robot.set_dofs_pos(real_dofs_pos_np)
        sim_robot.set_dofs_vel(real_dofs_vel_np)

        # --- Update the deployment wrapper ----------------------------------
        state_entity.update(robot_state)

        # --- Periodic comparison --------------------------------------------
        step += 1
        quat_real = state_entity.get_quat()
        quat_sim = sim_robot.get_quat()
        rpy_real = quat_to_xyz(quat_real, rpy=True)
        rpy_sim = quat_to_xyz(quat_sim, rpy=True)
        dofs_pos_real = state_entity.get_dofs_pos()
        dofs_pos_sim = sim_robot.get_dofs_pos()
        dofs_vel_real = state_entity.get_dofs_vel()
        dofs_vel_sim = sim_robot.get_dofs_vel()

        hist_step.append(step)
        hist_rpy_real.append(to_numpy(rpy_real).flatten())
        hist_rpy_sim.append(to_numpy(rpy_sim).flatten())
        hist_joint_real.append(
            np.array(
                [
                    float(dofs_pos_real.mean().item()),
                    float(dofs_vel_real.mean().item()),
                ]
            )
        )
        hist_joint_sim.append(
            np.array(
                [
                    float(dofs_pos_sim.mean().item()),
                    float(dofs_vel_sim.mean().item()),
                ]
            )
        )
        hist_err.append(
            np.array(
                [
                    float(torch.max(torch.abs(quat_real - quat_sim)).item()),
                    float(torch.max(torch.abs(rpy_real - rpy_sim)).item()),
                    float(torch.max(torch.abs(dofs_pos_real - dofs_pos_sim)).item()),
                    float(torch.max(torch.abs(dofs_vel_real - dofs_vel_sim)).item()),
                ]
            )
        )

        if step % PLOT_EVERY == 0 and len(hist_step) > 1:
            x = np.asarray(hist_step)
            rpy_real_np = np.vstack(hist_rpy_real)
            rpy_sim_np = np.vstack(hist_rpy_sim)
            joint_real_np = np.vstack(hist_joint_real)
            joint_sim_np = np.vstack(hist_joint_sim)
            err_np = np.vstack(hist_err)

            ax_rpy.clear()
            ax_rpy.plot(x, rpy_real_np[:, 0], "r-", label="roll real")
            ax_rpy.plot(x, rpy_sim_np[:, 0], "r--", label="roll sim")
            ax_rpy.plot(x, rpy_real_np[:, 1], "g-", label="pitch real")
            ax_rpy.plot(x, rpy_sim_np[:, 1], "g--", label="pitch sim")
            ax_rpy.plot(x, rpy_real_np[:, 2], "b-", label="yaw real")
            ax_rpy.plot(x, rpy_sim_np[:, 2], "b--", label="yaw sim")
            ax_rpy.set_ylabel("rad")
            ax_rpy.set_title("Base RPY")
            ax_rpy.grid(True, alpha=0.3)
            ax_rpy.legend(loc="upper left", ncol=3, fontsize=8)

            ax_joint.clear()
            ax_joint.plot(x, joint_real_np[:, 0], "c-", label="mean dof_pos real")
            ax_joint.plot(x, joint_sim_np[:, 0], "c--", label="mean dof_pos sim")
            ax_joint.plot(x, joint_real_np[:, 1], "m-", label="mean dof_vel real")
            ax_joint.plot(x, joint_sim_np[:, 1], "m--", label="mean dof_vel sim")
            ax_joint.set_ylabel("rad / rad/s")
            ax_joint.set_title("Joint Stream Overview (mean across DOFs)")
            ax_joint.grid(True, alpha=0.3)
            ax_joint.legend(loc="upper left", ncol=2, fontsize=8)

            ax_err.clear()
            ax_err.plot(x, err_np[:, 0], label="max |quat diff|")
            ax_err.plot(x, err_np[:, 1], label="max |rpy diff|")
            ax_err.plot(x, err_np[:, 2], label="max |dof_pos diff|")
            ax_err.plot(x, err_np[:, 3], label="max |dof_vel diff|")
            ax_err.set_xlabel("step")
            ax_err.set_ylabel("abs error")
            ax_err.set_title("Real vs Sim Error Trend")
            ax_err.grid(True, alpha=0.3)
            ax_err.legend(loc="upper left", ncol=2, fontsize=8)

            fig.tight_layout()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        if step % PRINT_EVERY == 0:
            out: list[str] = []
            out.append(f"\n{'=' * 64}")
            out.append(f"  Convention comparison   (step {step})")
            out.append(f"{'=' * 64}")

            # 1. Quaternion (wxyz) -----------------------------------------
            #    The LimX IMU quaternion is wxyz on the wire, same as Eden.
            out.append(
                _compare(
                    "Base Quaternion (w, x, y, z)",
                    quat_real,
                    quat_sim,
                    names=["w", "x", "y", "z"],
                )
            )

            # 2. Projected gravity -----------------------------------------
            real_proj_grav = transform_by_quat(
                torch.tensor([[0.0, 0.0, -1.0]], device=device),
                inv_quat(state_entity.get_quat()),
            )
            sim_proj_grav = transform_by_quat(
                torch.tensor([[0.0, 0.0, -1.0]], device=device),
                inv_quat(sim_robot.get_quat()),
            )
            out.append(
                _compare(
                    "Projected Gravity",
                    real_proj_grav,
                    sim_proj_grav,
                    names=["gx", "gy", "gz"],
                )
            )

            # 3. Roll / Pitch / Yaw ----------------------------------------
            out.append(
                _compare(
                    "Base RPY (rad)",
                    rpy_real,
                    rpy_sim,
                    names=["roll", "pitch", "yaw"],
                )
            )

            # 4. Angular velocity – body frame -----------------------------
            #    Real: IMU gyroscope is body-frame directly.
            #    Sim: cannot set base ang vel, so show real wrapper value
            #    alongside the raw IMU reading as a sanity check.
            real_ang_body = state_entity.get_ang(frame="body")
            raw_gyro = torch.as_tensor(
                robot_state.base_ang_vel,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            out.append(
                _compare(
                    "Angular Velocity [body frame] — wrapper vs raw IMU gyro",
                    real_ang_body,
                    raw_gyro,
                    names=["wx", "wy", "wz"],
                )
            )

            # 5. Angular velocity – world frame ----------------------------
            #    Show wrapper's world-frame transformation so the user can
            #    check the sign / direction visually.
            real_ang_world = state_entity.get_ang(frame="world")
            manual_ang_world = transform_by_quat(raw_gyro, state_entity.get_quat())
            out.append(
                _compare(
                    "Angular Velocity [world frame] — wrapper vs manual quat_apply",
                    real_ang_world,
                    manual_ang_world,
                    names=["wx", "wy", "wz"],
                )
            )

            # 6. DOF positions ---------------------------------------------
            #    PR-space readback: the parallel ankle/waist joints come back
            #    as serial pitch/roll values, directly comparable to the URDF.
            out.append(
                _compare(
                    "DOF Positions (rad)",
                    dofs_pos_real,
                    dofs_pos_sim,
                    names=dofs_name,
                )
            )

            # 7. DOF velocities --------------------------------------------
            out.append(
                _compare(
                    "DOF Velocities (rad/s)",
                    dofs_vel_real,
                    dofs_vel_sim,
                    names=dofs_name,
                )
            )

            # 8. DOF torques (real only — sim has no matching setter) ------
            out.append(
                _show(
                    "DOF Torques (Nm) — real only (tau)",
                    state_entity.get_dofs_force(),
                    names=dofs_name,
                )
            )

            print("\n".join(out))

        env.step()


if __name__ == "__main__":
    main()
