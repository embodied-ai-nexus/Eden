r"""
Write the BONES-SEED G1 clip list SuperMimic trains on: a name filter, then a feasibility filter.

---
tier: project
capabilities: [task.mimic, robot.g1, dataset.seed]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/mimic/supermimic/bones_seed_clip_list.py --raw-data-path /data/bones_seed/g1/csv -o bones_seed_clips.json
---

The name filter drops clips whose file name holds a keyword of SONIC's BONES-SEED filter or one of ``50cm``,
``fence`` and ``bicycle`` (142,220 -> 129,667 clips). The feasibility filter resamples each kept clip to 50 Hz and
drops it when the retarget is physically implausible:

* elevated support: every body origin above 0.20 m for 0.6 s or more;
* ground penetration: a body origin below -0.12 m, or below -0.08 m in more than 80 % of the frames;
* floating feet: both feet more than 2 cm above the floor for 0.5 s or more while the pelvis is upright
  (z axis within ~26 degrees of vertical), above 0.7 m and moving slower than 0.7 m/s.

On the full G1 split this keeps 129,514 clips (262.3 h). The list feeds ``eden train``::

    eden train --task supermimic --dataset bones_seed --raw_data_path /data/bones_seed/g1/csv \\
        --processed_data_path /data/bones_seed_g1.pt --clip_list bones_seed_clips.json

Usage::

    python examples/projects/mimic/supermimic/bones_seed_clip_list.py --raw-data-path /data/bones_seed/g1/csv \\
        -o bones_seed_clips.json --workers 32
    python examples/projects/mimic/supermimic/bones_seed_clip_list.py --raw-data-path /data/bones_seed/g1/csv \\
        -o names_only.json --name-filter-only
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import mujoco
import numpy as np

INPUT_FPS = 120.0
OUTPUT_FPS = 50.0

#: SONIC's BONES-SEED name filter (case-insensitive substrings of the file name), and the three terms SuperMimic adds.
NAME_FILTER_KEYWORDS = (
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
    "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "acrobatics_", "box_HSPU", "cartwheel",
    "50cm_box_", "on_box", "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m", "230m",
    "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll", "kozak_dance", "stair", "handstand",
    "box_jump", "monkey_jump", "safety_roll", "box_dips", "walking_on_edge", "push_obstacle",
    "50cm", "fence", "bicycle",
)  # fmt: skip

ELEVATED_HEIGHT, ELEVATED_MIN_S = 0.20, 0.6
PENETRATION_HARD, PENETRATION_SOFT, PENETRATION_SOFT_FRACTION = -0.12, -0.08, 0.8
FLOATING_CLEARANCE, FLOATING_MIN_S = 0.02, 0.5
FLOATING_UPRIGHT_COS, FLOATING_MIN_PELVIS_HEIGHT, FLOATING_MAX_SPEED = 0.9, 0.7, 0.7


def passes_name_filter(path: str) -> bool:
    name = os.path.basename(path).lower()
    return not any(keyword.lower() in name for keyword in NAME_FILTER_KEYWORDS)


def read_clip(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Root position (m), root quaternion (wxyz) and joint angles (rad) of a 120 fps G1 CSV.

    Columns: ``Frame, root_translate{X,Y,Z} [cm], root_rotate{X,Y,Z} [deg], 29 joints [deg]``; the root rotation is
    composed intrinsic Z-Y-X.
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=tuple(range(1, 36)), ndmin=2)
    rx, ry, rz = np.deg2rad(raw[:, 3:6]).T
    cx, sx, cy, sy, cz, sz = (
        np.cos(rx / 2),
        np.sin(rx / 2),
        np.cos(ry / 2),
        np.sin(ry / 2),
        np.cos(rz / 2),
        np.sin(rz / 2),
    )
    quat = np.stack(
        [
            cz * cy * cx + sz * sy * sx,
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx,
        ],
        axis=-1,
    )
    return raw[:, 0:3] * 0.01, quat, np.deg2rad(raw[:, 6:35])


def resample(pos: np.ndarray, quat: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, ...]:
    """Linear and spherical interpolation onto the 50 Hz grid."""
    n = pos.shape[0]
    duration = (n - 1) / INPUT_FPS
    times = np.arange(int(np.floor(duration * OUTPUT_FPS + 1e-6)) + 1) / OUTPUT_FPS
    phase = times * INPUT_FPS
    i0 = np.minimum(np.floor(phase).astype(int), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (phase - i0)[:, None]
    q0, q1 = quat[i0], quat[i1]
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), -1.0, 1.0))
    sin = np.sin(theta)
    small = sin < 1e-6
    a = np.where(small, 1 - w, np.sin((1 - w) * theta) / np.where(small, 1.0, sin))
    b = np.where(small, w, np.sin(w * theta) / np.where(small, 1.0, sin))
    q = a * q0 + b * q1
    return (
        pos[i0] * (1 - w) + pos[i1] * w,
        q / np.linalg.norm(q, axis=-1, keepdims=True),
        joints[i0] * (1 - w) + joints[i1] * w,
    )


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return int((edges[1::2] - edges[0::2]).max())


class FeasibilityProbe:
    """MuJoCo kinematics of the G1 over a floor: the lowest body origin and each foot's floor clearance."""

    def __init__(self, mjcf_path: str):
        spec = mujoco.MjSpec.from_file(mjcf_path)
        spec.worldbody.add_geom(name="clip_list_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        geom = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)  # noqa: E731
        self.floor = geom("clip_list_floor")
        self.feet = [[geom(f"{side}_foot{i}_collision") for i in range(1, 8)] for side in ("left", "right")]
        if min(min(ids) for ids in self.feet) < 0:
            raise ValueError(f"{mjcf_path} has no left/right_foot[1-7]_collision geoms.")

    def infeasible(self, pos: np.ndarray, quat: np.ndarray, joints: np.ndarray) -> list[str]:
        """The flags a 50 Hz clip raises: ``elevated_support``, ``ground_penetration``, ``floating_feet``."""
        m, d, fromto = self.model, self.data, np.zeros(6)
        n = pos.shape[0]
        zmin, clearance = np.empty(n), np.empty((n, 2))
        for t in range(n):
            d.qpos[:3], d.qpos[3:7], d.qpos[7:] = pos[t], quat[t], joints[t]
            mujoco.mj_kinematics(m, d)
            zmin[t] = d.xpos[1:, 2].min()
            for side, ids in enumerate(self.feet):
                clearance[t, side] = min(mujoco.mj_geomDistance(m, d, g, self.floor, 1.0, fromto) for g in ids)
        speed = np.linalg.norm(np.gradient(pos, 1.0 / OUTPUT_FPS, axis=0), axis=-1)
        upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        still = (
            (upright > FLOATING_UPRIGHT_COS) & (pos[:, 2] > FLOATING_MIN_PELVIS_HEIGHT) & (speed < FLOATING_MAX_SPEED)
        )
        floating = (clearance > FLOATING_CLEARANCE).all(axis=-1) & still
        flags = {
            "elevated_support": _longest_run(zmin > ELEVATED_HEIGHT) >= round(ELEVATED_MIN_S * OUTPUT_FPS),
            "ground_penetration": bool(
                (zmin < PENETRATION_HARD).any() or (zmin < PENETRATION_SOFT).mean() > PENETRATION_SOFT_FRACTION
            ),
            "floating_feet": _longest_run(floating) >= round(FLOATING_MIN_S * OUTPUT_FPS),
        }
        return [flag for flag, raised in flags.items() if raised]


_PROBE: FeasibilityProbe | None = None


def _init_worker(mjcf_path: str) -> None:
    global _PROBE
    _PROBE = FeasibilityProbe(mjcf_path)


def _check(path: str) -> list[str]:
    return _PROBE.infeasible(*resample(*read_clip(path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n---")[0].strip())
    parser.add_argument("--raw-data-path", type=str, required=True, help="The BONES-SEED G1 CSVs (g1/csv).")
    parser.add_argument("-o", "--output", type=str, default="bones_seed_clips.json", help="Clip list to write.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Processes of the feasibility filter.")
    parser.add_argument("--name-filter-only", action="store_true", help="Skip the feasibility filter.")
    args = parser.parse_args()

    from eden.options.robots.unitree_g1 import G1Dofs29
    from eden.utils.assets import get_asset_path

    root = args.raw_data_path
    clips = sorted(
        os.path.relpath(os.path.join(d, f), root) for d, _, files in os.walk(root) for f in files if f.endswith(".csv")
    )
    if not clips:
        raise FileNotFoundError(f"No .csv under {root!r}; point --raw-data-path at the extracted g1/csv directory.")
    kept = [c for c in clips if passes_name_filter(c)]
    print(f"name filter: {len(clips)} -> {len(kept)} clips")
    flagged: dict[str, list[str]] = {}
    if not args.name_filter_only:
        with Pool(args.workers, initializer=_init_worker, initargs=(get_asset_path(G1Dofs29().file),)) as pool:
            flags = pool.map(_check, [os.path.join(root, c) for c in kept], chunksize=64)
        for clip, raised in zip(kept, flags, strict=True):
            for flag in raised:
                flagged.setdefault(flag, []).append(clip)
        kept = [clip for clip, raised in zip(kept, flags, strict=True) if not raised]
        print(f"feasibility filter: {len(kept)} clips kept; removed { ({k: len(v) for k, v in flagged.items()}) }")
    with open(args.output, "w") as f:
        json.dump({"motions": kept, "flagged": flagged}, f, indent=0)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
