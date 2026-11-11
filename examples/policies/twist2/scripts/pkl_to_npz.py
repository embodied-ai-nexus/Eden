"""Convert a TWIST2 enriched PKL motion into an Eden ``.npz`` motion file.

---
tier: policy
capabilities: [task.mimic, robot.g1]
hardware: gpu
optional_deps: []
run: uv run python examples/policies/twist2/scripts/pkl_to_npz.py --input-file examples/policies/twist2/resources/eyes_japan_greeting_01_hello_aita.pkl -o examples/policies/twist2/resources/greeting.npz
---

TWIST2 ships BeyondMimic-style enriched PKLs (``root_pos``, ``root_rot`` in
xyzw, ``dof_pos``, ``fps``, ...). Eden's :class:`MotionLoader` (used by the
beyondmimic ``MotionCommand``) expects an ``.npz`` with ``joint_pos`` /
``joint_vel`` / ``body_{pos,quat,lin_vel,ang_vel}_w`` in the robot's Genesis
link order, sampled at the control rate.

This mirrors ``examples/projects/mimic/beyondmimic/scripts/csv_to_npz.py``: interpolate the
clip to ``--output-fps`` (default 50 Hz), replay it through a Genesis G1 env
by forward kinematics, and log the per-frame robot state. Doing FK in the
target engine guarantees the npz body axis matches the robot's link order and
that velocities use the same convention as every other Eden motion.

Usage::

    python examples/policies/twist2/scripts/pkl_to_npz.py \
        --input-file examples/policies/twist2/resources/eyes_japan_greeting_01_hello_aita.pkl \
        -o examples/policies/twist2/resources/greeting.npz
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types

import numpy as np
import torch

from genesis.utils.geom import slerp

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions
from genesis.utils.geom import inv_quat, inv_transform_by_quat

from eden.utils.geom import axis_angle_from_quat, quat_mul, xyzw_to_wxyz

# Some enriched PKLs were pickled with newer numpy; alias the moved module.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = types.ModuleType("numpy._core")
if "numpy._core.multiarray" not in sys.modules:
    _mc = types.ModuleType("numpy._core.multiarray")
    if hasattr(np.core, "multiarray"):
        _mc._reconstruct = np.core.multiarray._reconstruct  # type: ignore[attr-defined]
    sys.modules["numpy._core.multiarray"] = _mc


def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1 - blend) + b * blend


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Central-difference SO3 angular velocity for a (B, 4) wxyz trajectory."""
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, inv_quat(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a TWIST2 enriched PKL to an Eden npz.")
    parser.add_argument("--input-file", required=True, help="Path to the enriched TWIST2 .pkl.")
    parser.add_argument("-o", "--output", help="Output .npz path (defaults next to the input).")
    parser.add_argument("--output-fps", type=int, default=50, help="Output (control) fps.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the viewer.")
    args = parser.parse_args()

    out_path = args.output or args.input_file.rsplit(".", 1)[0] + ".npz"

    en.init()  # before any genesis.utils.geom call (slerp needs gs.tc_float)

    with open(args.input_file, "rb") as f:
        data = pickle.load(f)
    input_fps = float(data["fps"])
    base_pos_in = torch.tensor(np.asarray(data["root_pos"]), dtype=torch.float32)
    base_rot_in = xyzw_to_wxyz(torch.tensor(np.asarray(data["root_rot"]), dtype=torch.float32))
    dof_pos_in = torch.tensor(np.asarray(data["dof_pos"]), dtype=torch.float32)
    n_in = base_pos_in.shape[0]
    duration = (n_in - 1) / input_fps
    print(f"Loaded {args.input_file}: {n_in} frames @ {input_fps:.3f} fps ({duration:.2f}s)")

    # Interpolate to the output (control) fps.
    out_dt = 1.0 / args.output_fps
    times = torch.arange(0, duration, out_dt, dtype=torch.float32)
    phase = times / duration
    idx0 = (phase * (n_in - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(n_in - 1))
    blend = (phase * (n_in - 1) - idx0).unsqueeze(1)
    base_pos = _lerp(base_pos_in[idx0], base_pos_in[idx1], blend)
    base_rot = slerp(base_rot_in[idx0], base_rot_in[idx1], blend.squeeze(1))
    dof_pos = _lerp(dof_pos_in[idx0], dof_pos_in[idx1], blend)
    n_out = times.shape[0]

    # Velocities at the output rate (same conventions as csv_to_npz).
    base_lin_vel = torch.gradient(base_pos, spacing=out_dt, dim=0)[0]
    dof_vel = torch.gradient(dof_pos, spacing=out_dt, dim=0)[0]
    base_ang_vel = _so3_derivative(base_rot, out_dt)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=out_dt),
        scene_options=en.scenes.Empty(robot=en.robots.G1Dofs29()),
        show_viewer=args.vis,
    )
    env.build()
    robot = env.entities["robot"]
    dofs_name, dofs_idx = robot.find_named_dofs_idx_local(env.scene_options.robot.dofs_name, preserve_order=True)
    _, base_idx = robot.find_named_dofs_idx_local("floating_base_joint")

    # Persist column labels so the npz is self-describing: ``joint_*`` are in ``dofs_name`` order
    # (what get_dofs_pos/get_dofs_vel return below) and ``body_*`` in the robot's internal link
    # order (what get_links_* return below). Consumers resolve columns by name and stay correct
    # across genesis versions that reorder the internal kinematic tree (BFS->DFS in genesis 1.2.0).
    body_names = [link.name for link in robot.links]
    log: dict = {"fps": [args.output_fps], "dof_names": dofs_name, "body_names": body_names,
                 "joint_pos": [], "joint_vel": [],
                 "body_pos_w": [], "body_quat_w": [], "body_lin_vel_w": [], "body_ang_vel_w": []}  # fmt: skip
    for i in range(n_out):
        robot.set_qpos(torch.cat([base_pos[i : i + 1], base_rot[i : i + 1], dof_pos[i : i + 1]], dim=1).to(env.device))
        robot.set_dofs_vel(dof_vel[i : i + 1].to(env.device), dofs_idx_local=dofs_idx)
        # Genesis stores a free joint's rotational DOF velocity in the base BODY frame
        # (MuJoCo qvel convention); base_ang_vel is world-frame. Rotate world->body with
        # the same quat written to qpos above. Linear DOFs stay world-frame.
        base_ang_vel_b = inv_transform_by_quat(base_ang_vel[i : i + 1], base_rot[i : i + 1])
        robot.set_dofs_vel(
            torch.cat([base_lin_vel[i : i + 1], base_ang_vel_b], dim=1).to(env.device),
            dofs_idx_local=base_idx,
        )
        env.scene.visualizer.update()
        log["joint_pos"].append(robot.get_dofs_pos()[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.get_dofs_vel()[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.get_links_pos()[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.get_links_quat()[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.get_links_vel()[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.get_links_ang()[0].cpu().numpy().copy())

    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)
    np.savez(out_path, **log)
    print(f"[twist2] Saved {n_out} frames @ {args.output_fps} fps to {out_path}")


if __name__ == "__main__":
    main()
