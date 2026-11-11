r"""Retarget human (+ object) motion to a humanoid with Eden's OTRetargeter.

---
tier: recipe
capabilities: [retarget.human, robot.g1, robot.oli, dataset.omomo, dataset.seed]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/retargeting/otretarget/retarget_motion.py --source omomo --data-dir /path/to/eden_omomo --body-model-path /path/to/smplx --clip sub3_largebox_003 --rig g1_wuji -c
---

Two demonstration sources:

* ``--source omomo``: a canonical OMOMO sequence (``sequences_canonical/<clip>/`` of the ``eden_omomo``
  release, SMPL-X with finger articulation) and its object. Hands + body + object are retargeted
  jointly; use a dexterous rig (``g1_wuji``, ``oli_wuji``). ``--body-model-path`` is the folder holding
  ``SMPLX_{NEUTRAL,MALE,FEMALE}.npz``.
* ``--source seed``: a BONES-SEED SOMA-rig ``.bvh`` (``<data-dir>/<clip>.bvh``), skinned with the SOMA
  neutral body. ``--body-model-path`` is ``SOMA_neutral.npz`` from the ``nvidia/soma-x`` release. Body-only
  rigs (``g1``, ``oli``) or the dexterous ones.

The result (rig ``qpos`` per entity, solved object poses, per-frame solver diagnostics) is written to
``<output-dir>/<rig>_<clip>.npz`` with the interaction metrics next to it; ``--record`` renders the
demonstration and the retargeted robot side by side.

Usage:
    uv run python examples/recipes/retargeting/otretarget/retarget_motion.py -c --source omomo \\
        --data-dir /path/to/eden_omomo --body-model-path /path/to/smplx --clip sub3_largebox_003 --rig g1_wuji -r
    uv run python examples/recipes/retargeting/otretarget/retarget_motion.py -c --source seed \\
        --data-dir /path/to/bones_seed_bvh --body-model-path /path/to/SOMA_neutral.npz \\
        --clip high_jump_R_001__A277 --rig oli -r
"""

from __future__ import annotations

import argparse
import json
import os

import genesis as gs
import numpy as np
import trimesh

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.retargeting.otretarget import PRESETS, OTRetargeter, for_body_model
from eden.extensions.retargeting.otretarget.kinematics import RigKinematics
from eden.extensions.retargeting.otretarget.metrics import evaluate_retarget
from eden.extensions.retargeting.otretarget.sources import load_omomo_sequence, load_soma_bvh
from eden.options import CameraOptions, CamerasOptions, EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.options.robots.composite import G1Wuji, OliWuji
from eden.utils.geom import quat_to_R

RIGS = {"g1_wuji": G1Wuji, "oli_wuji": OliWuji, "g1": lambda: en.robots.G1Dofs29(), "oli": lambda: en.robots.Oli()}


def load_demonstration(source: str, data_dir: str, body_model_path: str, clip: str):
    """Return ``(human, objects, body_model)``."""
    if source == "omomo":
        human, obj = load_omomo_sequence(
            os.path.join(data_dir, "sequences_canonical", clip), os.path.join(data_dir, "objects"), body_model_path
        )
        return human, [obj], "smplx"
    return load_soma_bvh(os.path.join(data_dir, f"{clip}.bvh"), body_model_path), [], "soma"


def record(env: EnvBase, kin: RigKinematics, result, human, objects, path: str, offset: float = 1.4) -> None:
    """Render the demonstration (``+offset`` along y) and the retargeted robot side by side."""
    camera = env.cameras["camera"]
    camera.start_recording(path, fps=result.fps)
    shift = np.array([0.0, offset, 0.0])

    def posed(mesh, pose, at, rgba):
        m = mesh.copy()
        T = np.eye(4)
        T[:3, :3] = quat_to_R(pose[3:])
        T[:3, 3] = pose[:3] + at
        m.apply_transform(T)
        m.visual.vertex_colors = np.tile(rgba, (len(m.vertices), 1))
        return m

    for t in range(len(result.qpos)):
        kin.set_configuration(result.qpos[t])
        env.scene.clear_debug_objects()
        body = trimesh.Trimesh(human.vertices[t] + shift, human.faces, process=False)
        body.visual.vertex_colors = np.tile([217, 178, 153, 255], (len(body.vertices), 1))
        env.scene.draw_debug_mesh(body)
        for obj in objects:
            env.scene.draw_debug_mesh(
                posed(obj.mesh, result.object_poses[obj.name][t], np.zeros(3), [90, 140, 210, 255])
            )
            env.scene.draw_debug_mesh(posed(obj.mesh, obj.poses[t], shift, [90, 140, 210, 255]))
        centre = result.qpos[t, :3] + shift / 2
        centre[2] = 0.8
        camera.set_pose(pos=centre + np.array([3.6, 0.0, 0.6]), lookat=centre)
        env.scene.visualizer.update()
        camera.render_rgb()
    camera.stop_recording()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget OMOMO / BONES-SEED motion with OTRetarget.")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run Genesis on the CPU backend")
    parser.add_argument("--source", choices=("omomo", "seed"), default="omomo", help="Demonstration format")
    parser.add_argument("--data-dir", type=str, required=True, help="eden_omomo root, or the folder of SOMA .bvh files")
    parser.add_argument("--body-model-path", type=str, required=True, help="SMPL-X model folder, or SOMA_neutral.npz")
    parser.add_argument("--clip", type=str, default="sub3_largebox_003", help="Sequence name (BVH file stem for seed)")
    parser.add_argument("--rig", choices=sorted(RIGS), default="g1_wuji", help="Target embodiment")
    parser.add_argument("--num-frames", type=int, default=None, help="Retarget only the first N frames")
    parser.add_argument("-o", "--output-dir", type=str, default="results/otretarget", help="Where results are written")
    parser.add_argument("-r", "--record", action="store_true", help="Render a side-by-side video")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, precision="64")
    human, objects, body_model = load_demonstration(args.source, args.data_dir, args.body_model_path, args.clip)
    if args.num_frames is not None:
        human = human.slice(0, args.num_frames)
        for obj in objects:
            obj.poses = obj.poses[: args.num_frames]

    preset = for_body_model(PRESETS[args.rig], body_model)
    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=RIGS[args.rig](), floor=PlaneOptions()),
        cameras_options=CamerasOptions(
            camera=CameraOptions(cam_pos=(3.0, 0.0, 1.2), cam_lookat=(0.0, 0.0, 0.8), cam_fov=45, debug=True)
        )
        if args.record
        else None,
        show_viewer=False,
    )
    env.build()
    env.reset()

    retargeter = OTRetargeter(env, OTRetargeter.configure(**preset.retargeter_options()))
    result = retargeter.retarget_motion(human, objects)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, f"{args.rig}_{args.clip}")
    result.save(f"{stem}.npz")
    metrics = evaluate_retarget(retargeter, human, objects, result)
    metrics |= {f"solver/{k}": float(v) for k, v in result.stats.items()}
    metrics["solver/ms_per_frame_median"] = float(np.median(result.info["solve_ms"]))
    with open(f"{stem}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    en.logger.info(f"Wrote {stem}.npz; metrics: {json.dumps({k: round(v, 3) for k, v in metrics.items()})}")
    if args.record:
        record(env, retargeter.kin, result, human, objects, f"{stem}.mp4")
        en.logger.info(f"Wrote {stem}.mp4")


if __name__ == "__main__":
    main()
