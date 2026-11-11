"""Motion-retargeting commands: ``eden retarget {omniretarget|otretarget|soma|dex|flashdex}``.

Each handler is a self-contained batch driver over one of the retargeting
backends in :mod:`eden.extensions.retargeting`. The backends have
*disjoint* APIs (SMPL-family poses, MANO keypoints, OMOMO joint trajectories,
FlashDexRetarget's dataset pipeline, whose ``flashdex import`` / ``flashdex kinematic``
take its options classes through ``--config`` / ``--override`` like ``eden train``), so
there is deliberately no shared runner or registry — only the ``--src`` / ``--out``
vocabulary (:mod:`eden.utils.cli.paths`) and the small ``_build_single_env`` helper are
shared. The domain payload (hand presets, G1
OmniRetarget config) lives in the eden package alongside the backends, not here.

Every handler parses its own ``extra_args`` *before* importing its (optional)
backend, so ``eden retarget <backend> --help`` and argument errors work even
when the backend extra is not installed. A missing dependency then surfaces as an
actionable install hint: ``dex``, ``omniretarget`` and ``otretarget`` wrap the backend import,
while ``soma`` — whose heavy deps live in a worker subprocess, where an
``except ImportError`` here could never see them — preflights the installed
distributions instead.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

from eden.utils.cli.paths import iter_src_files, resolve_out
from eden.utils.distribution import install_hint

_DEFAULT_FPS = 30.0

# ---------------------------------------------------------------------------
# Shared scene plumbing
# ---------------------------------------------------------------------------


def _build_single_env(robot_options, *, cpu: bool, precision: str = "32"):
    """Build a headless single-env scene holding ``robot_options`` (core imports kept lazy)."""
    import genesis as gs

    import eden as en
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, SceneOptions
    from eden.options.entities import PlaneOptions

    en.init(backend=gs.cpu if cpu else gs.gpu, precision=precision)
    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=robot_options, floor=PlaneOptions()),
        show_viewer=False,
    )
    env.build()
    env.reset()
    return env


# ---------------------------------------------------------------------------
# soma — SMPL / SMPL-H / SMPL-X → robot (SomaRetargeter, library-backed)
# ---------------------------------------------------------------------------

# FPS npz key(s) per body model (SMPL-X AMASS uses the underscored spelling).
_SOMA_FPS_KEYS = {
    "smpl": ("mocap_framerate", "mocap_frame_rate"),
    "smplh": ("mocap_framerate", "mocap_frame_rate"),
    "smplx": ("mocap_frame_rate", "mocap_framerate"),
}


def _load_soma_clip(npz_path: Path, body_model: str, source_fps: float | None) -> dict:
    """Read one SMPL-family AMASS-style ``.npz`` into ``retarget_motion`` kwargs.

    The library selects the body model explicitly (``body_model=``), so the full
    ``poses`` array is forwarded verbatim — no per-component slicing (and thus no
    silent mis-slice of an SMPL-H file fed as SMPL-X).
    """
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as d:
        keys = _SOMA_FPS_KEYS[body_model]
        if source_fps is not None:
            fps = float(source_fps)
        else:
            fps = next((float(d[k]) for k in keys if k in d.files), None)
            if fps is None:
                raise SystemExit(f"{npz_path}: no FPS field {keys} in npz; pass --source-fps.")
        missing = [k for k in ("poses", "trans", "betas") if k not in d.files]
        if missing:
            raise SystemExit(
                f"{npz_path}: SMPL-family clip is missing required field(s) {missing}; got {list(d.files)}."
            )
        clip = {
            "poses": np.asarray(d["poses"], dtype=np.float32),
            "trans": np.asarray(d["trans"], dtype=np.float32),
            "betas": np.asarray(d["betas"], dtype=np.float32),
            "source_fps": fps,
        }
        if "gender" in d.files:
            g = d["gender"]
            g = g.item() if hasattr(g, "item") else g
            clip["gender"] = g.decode() if isinstance(g, bytes) else str(g)
        return clip


def _run_soma(extra_args: list[str]) -> None:
    """``eden retarget soma`` — batch SMPL / SMPL-H / SMPL-X → robot via SomaRetargeter.

    Reads AMASS-style SMPL-family ``.npz`` clips (a single file or a directory
    searched recursively) and writes one Genesis ``qpos`` NPZ per input,
    mirroring the source tree under ``--out``. Native SOMA (arrays / BVH) input
    and mp4 rendering are out of scope — see
    ``examples/recipes/retargeting/soma_retargeter/``.
    """
    parser = argparse.ArgumentParser(
        prog="eden retarget soma",
        description="Batch-retarget AMASS-style SMPL/SMPL-H/SMPL-X motion to a robot (SomaRetargeter).",
        epilog="Native SOMA (arrays/BVH) input and mp4 rendering are out of scope; "
        "see examples/recipes/retargeting/soma_retargeter/.",
    )
    parser.add_argument(
        "--src", type=Path, required=True, help="A single .npz clip, or a directory searched recursively."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (source tree mirrored); may be a file for a single-file --src.",
    )
    parser.add_argument(
        "--body-model", choices=("smpl", "smplh", "smplx"), default="smplx", help="Source body model. Default: smplx."
    )
    parser.add_argument(
        "--body-model-path",
        type=str,
        required=True,
        help="Path to the body-model file (e.g. SMPLX_NEUTRAL.npz) or a per-gender directory.",
    )
    parser.add_argument(
        "--robot", choices=("unitree_g1", "limx_oli"), default="unitree_g1", help="Target robot. Default: unitree_g1."
    )
    parser.add_argument(
        "--gender",
        choices=("neutral", "male", "female"),
        default="neutral",
        help="Default gender; a per-clip npz 'gender' field overrides it. Default: neutral.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=_DEFAULT_FPS,
        help=f"FPS to resample every clip to. Default: {_DEFAULT_FPS:g}.",
    )
    parser.add_argument(
        "--source-fps", type=float, default=None, help="Override source FPS when npz metadata is missing/unreliable."
    )
    parser.add_argument(
        "--num-frames", type=int, default=None, help="Truncate each clip to N frames (default: full clip)."
    )
    parser.add_argument(
        "--snap-to-ground",
        action="store_true",
        help="Shift each clip vertically so its lowest collision point touches the ground.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip a clip when its output NPZ already exists.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU backend (default: GPU).")
    args = parser.parse_args(extra_args)

    import numpy as np

    import eden as en

    # SomaRetargeter runs the library in a Genesis-free worker subprocess (its
    # heavy deps are imported there), so a plain import guard here can't catch a
    # missing or known-broken extra — preflight the installed distributions up
    # front and fail with the actionable hint instead of paying for a scene build
    # and then crashing in the worker.
    from eden.extensions.retargeting.soma_retargeter import SomaRetargeter, check_backend_available

    check_backend_available(args.body_model)

    single, src_root, inputs = iter_src_files(args.src, "*.npz")
    out_paths = [resolve_out(args.out, src_root, s, ".npz", single=single) for s in inputs]

    robots = {"unitree_g1": en.robots.G1Dofs29, "limx_oli": en.robots.Oli}
    # soma runs the body-model forward + IK in its worker, so it defaults to GPU; dex/omni solve on CPU.
    env = _build_single_env(robots[args.robot](), cpu=args.cpu)
    robot = env.entities["robot"]
    en.logger.info(f"Found {len(inputs)} input file(s) under {src_root}")
    retargeter = SomaRetargeter(
        env=env,
        options=SomaRetargeter.configure(
            entity_name="robot",
            body_model=args.body_model,
            body_model_path=args.body_model_path,
            gender=args.gender,
            target_fps=args.target_fps,
            snap_to_ground=args.snap_to_ground,
            device="cpu" if args.cpu else "cuda",
            robot=args.robot,
        ),
    )
    try:
        for i, (src, out_path) in enumerate(zip(inputs, out_paths)):
            if args.skip_existing and out_path.exists():
                en.logger.info(f"[{i + 1}/{len(inputs)}] skip (exists): {out_path}")
                continue
            clip = _load_soma_clip(src, args.body_model, args.source_fps)
            if args.num_frames is not None:
                for key in ("poses", "trans"):  # betas is a per-clip shape vector, not per-frame
                    clip[key] = clip[key][: args.num_frames]
            T = clip["poses"].shape[0]
            en.logger.info(f"[{i + 1}/{len(inputs)}] {src.name}  T={T}  {clip['source_fps']}->{args.target_fps} fps")
            qpos = retargeter.retarget_motion(**clip)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Stamp dof_names so downstream loads columns by name (order-independent).
            np.savez(
                out_path,
                qpos=qpos,
                dof_names=np.array(list(robot.dofs_name)),
                fps=np.float32(args.target_fps),
                source_path=str(src.relative_to(src_root)),
            )
            en.logger.info(f"      -> {out_path}  qpos {qpos.shape}")
    finally:
        retargeter.close()


# ---------------------------------------------------------------------------
# dex — MANO keypoints → dexterous hand (DexRetargeter, position path)
# ---------------------------------------------------------------------------


def _run_dex(extra_args: list[str]) -> None:
    """``eden retarget dex`` — offline MANO-keypoint → dexterous-hand position retargeting.

    Reads ``.npz`` files holding ``keypoints`` (T, 21, 3) MANO-ordered keypoints
    plus a ``fps`` scalar and writes one pickle of per-frame qpos per input. The
    live-video (MediaPipe) and DexYCB variants are out of scope — see
    ``examples/recipes/retargeting/dex_retargeter/``.
    """
    from eden.extensions.retargeting.dex_retargeter.presets import POSITION_PRESET_NAMES

    parser = argparse.ArgumentParser(
        prog="eden retarget dex",
        description="Offline position retargeting of MANO keypoints to a dexterous hand (DexRetargeter).",
        epilog="Video/MediaPipe and DexYCB retargeting are out of scope; see examples/recipes/retargeting/dex_retargeter/.",
    )
    parser.add_argument(
        "--src", type=Path, required=True, help="A single .npz with keypoints (T, 21, 3) + fps, or a directory of them."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .pkl file (single-file --src) or directory (source tree mirrored).",
    )
    parser.add_argument(
        "--preset",
        default="xhand1_right_position",
        choices=POSITION_PRESET_NAMES,
        help="Position-retargeting hand preset. Default: xhand1_right_position.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip a clip when its output pickle already exists."
    )
    args = parser.parse_args(extra_args)

    import numpy as np

    import eden as en

    try:
        from eden.extensions.retargeting.dex_retargeter import DexRetargeter
    except ImportError as e:
        raise ImportError(
            f"The DexRetargeter backend is not installed. Install it with {install_hint('dex-retargeter')}."
        ) from e
    import pickle

    import tqdm

    from eden.extensions.retargeting.dex_retargeter.presets import HAND_PRESETS

    preset = HAND_PRESETS[args.preset]
    single, src_root, inputs = iter_src_files(args.src, "*.npz")
    out_paths = [resolve_out(args.out, src_root, s, ".pkl", single=single) for s in inputs]
    en.logger.info(f"Found {len(inputs)} input file(s) under {src_root}")

    env = _build_single_env(preset.robot_options_factory(), cpu=True)
    retargeter = DexRetargeter(env=env, options=preset.retargeter_options())
    # Guards future non-position presets that would fail only after the env build.
    if retargeter.retargeting_type != "POSITION":
        raise SystemExit(f"Preset {args.preset!r} is type {retargeter.retargeting_type!r}; this path needs POSITION.")
    indices = retargeter.optimizer.target_link_human_indices

    for i, (src, out_path) in enumerate(zip(inputs, out_paths)):
        if args.skip_existing and out_path.exists():
            en.logger.info(f"[{i + 1}/{len(inputs)}] skip (exists): {out_path}")
            continue
        npz = np.load(src)
        keypoints = np.asarray(npz["keypoints"], dtype=np.float32)
        fps = float(npz["fps"]) if "fps" in npz.files else _DEFAULT_FPS
        if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 3):
            raise SystemExit(f"{src}: expected keypoints shape (T, 21, 3); got {keypoints.shape}.")

        en.logger.info(f"[{i + 1}/{len(inputs)}] {src.name}  T={keypoints.shape[0]}")
        data = [
            retargeter.retarget(keypoints[t, indices, :]) for t in tqdm.tqdm(range(keypoints.shape[0]), desc=src.name)
        ]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(
                dict(
                    data=data,
                    meta_data=dict(
                        preset=args.preset, dof=len(retargeter.joint_names), joint_names=retargeter.joint_names, fps=fps
                    ),
                ),
                f,
            )
        en.logger.info(f"      -> {out_path}  ({len(data)} frames)")


# ---------------------------------------------------------------------------
# omniretarget — OMOMO human motion → G1 (OmniRetarget, robot-only path)
# ---------------------------------------------------------------------------


def _run_omniretarget(extra_args: list[str]) -> None:
    """``eden retarget omniretarget`` — OMOMO human motion → G1 (robot-only).

    Loads one OMOMO sequence (by ``--task-name``), ignores the interaction
    object, and retargets the human motion to the G1 29-DOF robot via
    OmniRetarget, writing an ``.npy`` of per-frame qpos. The object-interaction
    variant is out of scope — see
    ``examples/recipes/retargeting/omniretarget/retarget_object_interaction.py``.
    """
    parser = argparse.ArgumentParser(
        prog="eden retarget omniretarget",
        description="Robot-only OMOMO → G1 retargeting (OmniRetarget), ignoring the interaction object.",
        epilog="Object-interaction retargeting is out of scope; see examples/recipes/retargeting/omniretarget/retarget_object_interaction.py.",
    )
    parser.add_argument(
        "--src", type=Path, required=True, help="OMOMO dataset directory (containing the .pt sequence files)."
    )
    parser.add_argument(
        "--task-name",
        default="sub3_largebox_003",
        help="Sequence to retarget, with or without the .pt suffix (matched by basename stem).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory; qpos is written as <task-name>.npy.")
    parser.add_argument(
        "--num-frames", type=int, default=None, help="Truncate the sequence to N frames (default: full sequence)."
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip when <out>/<task-name>.npy already exists.")
    args = parser.parse_args(extra_args)

    import genesis as gs
    import numpy as np

    import eden as en

    try:
        from eden.extensions.retargeting.omniretarget import OmniRetargeter
        from eden.extensions.retargeting.omniretarget.g1_presets import (
            FOOT_STICKING_LINKS,
            FOOT_STICKING_MAP,
            LINK_OFFSETS,
            MANUAL_COST,
            MANUAL_LB,
            MANUAL_UB,
            NOMINAL_TRACKING_DOFS,
            extract_foot_sticking_velocity,
            preprocess_human_motion,
        )
    except ImportError as e:
        raise ImportError(
            "OmniRetarget needs cvxpy, which has no dedicated extra. "
            f"Install it alone with `pip install cvxpy`, or with the `all` bundle (visualization and teleop "
            f"packages, cvxpy among them): {install_hint('all')}."
        ) from e
    from eden.datasets.constants import SMPLH_KEYPOINTS_NAME
    from eden.datasets.omomo_dataset import OMOMOHolosomaDataset
    from eden.extensions.retargeting.constants import JOINTS_MAP_UNITREE_G1, TOE_NAMES_BY_FORMAT

    if not args.src.is_dir():
        raise SystemExit(f"--src OMOMO dataset directory does not exist: {args.src}")

    # Tolerate a trailing ".pt" so passing the raw filename (e.g. from tab-completion) works.
    task_name = args.task_name.removesuffix(".pt")
    out_path = args.out / f"{task_name}.npy"
    if args.skip_existing and out_path.exists():
        en.logger.info(f"skip (exists): {out_path}")
        return

    # Build the env first: _build_single_env runs en.init(), so gs.device is set
    # before the dataset reads it (the dataset does not depend on the env).
    env = _build_single_env(en.robots.G1Dofs29(), cpu=True)

    dataset = OMOMOHolosomaDataset(
        options=OMOMOHolosomaDataset.configure(raw_data_path=str(args.src), target_framerate=_DEFAULT_FPS),
        device=gs.device,
    )
    en.logger.info(f"Loaded {len(dataset)} sequences from {args.src}")

    # all_motion_files stores paths relative to raw_data_path (with dir
    # components), so match on the basename stem rather than exact equality.
    all_files = dataset.metadata["all_motion_files"]
    seq_idx = next((i for i, f in enumerate(all_files) if Path(f).stem == task_name), None)
    if seq_idx is None:
        raise SystemExit(f"Task '{task_name}' not found in dataset. Available (first 5): {all_files[:5]}...")
    dataset.select_trajectories([seq_idx])

    human_joints = dataset.motion_sequences["human"].links_pos_w[0].cpu().numpy()  # (T, 52, 3)
    en.logger.info(f"Loaded {human_joints.shape[0]} frames, {human_joints.shape[1]} joints")

    toe_names = TOE_NAMES_BY_FORMAT["smplh"]
    retargeter = OmniRetargeter(
        env=env,
        options=OmniRetargeter.configure(
            entity_name="robot",
            foot_links=FOOT_STICKING_LINKS,
            demo_joints=list(SMPLH_KEYPOINTS_NAME),
            laplacian_match_links=dict(JOINTS_MAP_UNITREE_G1["smplh"]),
            link_offsets=LINK_OFFSETS,
            manual_lb=MANUAL_LB,
            manual_ub=MANUAL_UB,
            manual_cost=MANUAL_COST,
            nominal_tracking_dofs=NOMINAL_TRACKING_DOFS,
            foot_sticking_map=FOOT_STICKING_MAP,
            activate_joint_limits=True,
            activate_obj_non_penetration=False,
            activate_foot_sticking=True,
            activate_foot_ground_constraint=True,
            foot_ground_z=0.0,
            foot_ground_margin=0.005,
            activate_foot_z_sticking=True,
            foot_z_sticking_tolerance=5e-3,
            step_size=0.2,
            w_nominal_tracking_init=5.0,
        ),
    )

    if args.num_frames is not None and args.num_frames < human_joints.shape[0]:
        human_joints = human_joints[: args.num_frames]
        en.logger.info(f"Truncated to {args.num_frames} frames")

    sub_name = task_name.split("_")[0]
    human_heights = dataset.metadata["human_height"]
    if sub_name not in human_heights:
        raise SystemExit(
            f"No human height for subject {sub_name!r} (from task '{task_name}'). Known subjects: {sorted(human_heights)}."
        )
    smpl_scale = en.robots.G1Dofs29.metadata["robot_height"] / human_heights[sub_name]
    en.logger.info(f"Scale factor: {smpl_scale:.4f}")

    human_joints = preprocess_human_motion(human_joints, list(SMPLH_KEYPOINTS_NAME), toe_names, smpl_scale)
    foot_sticking_sequences = extract_foot_sticking_velocity(human_joints, retargeter.demo_joints, toe_names)
    foot_sticking_sequences[0][toe_names[0]] = False
    foot_sticking_sequences[0][toe_names[1]] = False

    en.logger.info(f"Starting retargeting ({human_joints.shape[0]} frames) ...")
    retargeted = retargeter.retarget_motion(
        human_joint_motions=human_joints, foot_sticking_sequences=foot_sticking_sequences
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, retargeted)
    en.logger.info(f"-> {out_path}  ({len(retargeted)} frames)")


# ---------------------------------------------------------------------------
# otretarget — OMOMO (hands + body + object) / BONES-SEED BVH → humanoid (OTRetarget)
# ---------------------------------------------------------------------------

_OTR_RIGS = ("g1_wuji", "oli_wuji", "g1", "oli")


#: The errnos a loaded NFS server answers a valid open or rename with (11 of 9780 clips of one OMOMO run), and only
#: those: a full disk or a permission error fails the same way every time, so retrying it only delays the error.
_TRANSIENT_IO_ERRNOS = frozenset({errno.EINVAL, errno.EXDEV, errno.EIO, errno.ESTALE, errno.EAGAIN})


def _with_io_retries(write, attempts: int = 5, wait_s: float = 2.0) -> None:
    """Run ``write()``, retrying a transient ``OSError`` (:data:`_TRANSIENT_IO_ERRNOS`) a few times.

    Any other ``OSError``, and the last transient one, is raised unchanged.
    """
    import time

    for attempt in range(attempts):
        try:
            write()
            return
        except OSError as e:
            if e.errno not in _TRANSIENT_IO_ERRNOS or attempt == attempts - 1:
                raise
            time.sleep(wait_s * (attempt + 1))


def _replace(src: Path, dst: Path) -> None:
    """``os.replace`` that accepts an NFS rename reported failed but done (``src`` gone, ``dst`` present)."""
    try:
        os.replace(src, dst)
    except OSError:
        if src.exists() or not dst.exists():
            raise


def _otr_rig_options(name: str):
    import eden as en
    from eden.options.robots.composite import G1Wuji, OliWuji

    return {"g1_wuji": G1Wuji, "oli_wuji": OliWuji, "g1": en.robots.G1Dofs29, "oli": en.robots.Oli}[name]()


def _run_otretarget(extra_args: list[str]) -> None:
    """``eden retarget otretarget`` — surface-interaction retargeting via OTRetarget.

    ``--source omomo`` reads canonical OMOMO sequences (``sequences_canonical/<clip>/human.npz`` +
    ``object.npz``; ``--src`` is one sequence directory or the folder holding them) and retargets hands,
    body and object jointly. ``--source seed`` reads SOMA-rig ``.bvh`` clips (a file or a directory searched
    recursively). One ``<rig>_<clip>.npz`` (:meth:`RetargetResult.save`) is written per clip. Rendering
    and metrics are out of scope — see ``examples/recipes/retargeting/otretarget/``.
    """
    parser = argparse.ArgumentParser(
        prog="eden retarget otretarget",
        description="Joint robot + object retargeting from surface interactions (OTRetarget).",
        epilog="Videos and interaction metrics: see examples/recipes/retargeting/otretarget/.",
    )
    parser.add_argument("--source", choices=("omomo", "seed"), required=True, help="Demonstration format.")
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="omomo: a sequence directory or the folder of them; seed: a .bvh file or a directory of them.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory; one <rig>_<clip>.npz per clip.")
    parser.add_argument(
        "--body-model-path",
        type=str,
        required=True,
        help="omomo: the folder with SMPLX_{NEUTRAL,MALE,FEMALE}.npz; seed: SOMA_neutral.npz.",
    )
    parser.add_argument("--objects-dir", type=Path, default=None, help="omomo: the objects/ folder of the release.")
    parser.add_argument("--rig", choices=_OTR_RIGS, default="g1_wuji", help="Target embodiment. Default: g1_wuji.")
    parser.add_argument("--num-frames", type=int, default=None, help="Truncate each clip to N frames.")
    parser.add_argument("--cpu", action="store_true", help="Run Genesis on the CPU backend.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a clip whose output already exists.")
    parser.add_argument(
        "--shard",
        default="0/1",
        help="K/N: retarget only every N-th clip of the sorted list, starting at the K-th (0 <= K < N). Default: 0/1.",
    )
    parser.add_argument(
        "--clip-list",
        type=Path,
        default=None,
        help="A text file of clip names (one per line): retarget only those (before --shard).",
    )
    parser.add_argument(
        "--supports",
        action="store_true",
        help="omomo: add the untracked tables the object rests on as fixed plates "
        "(otretarget.supports.find_untracked_supports); writes <rig>_<clip>.supports.json.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="On a clip that raises, write <rig>_<clip>.error.txt, go on to the next clip and exit non-zero at the end.",
    )
    args = parser.parse_args(extra_args)
    if args.source == "omomo" and args.objects_dir is None:
        parser.error("--source omomo needs --objects-dir (the release's objects/ folder).")
    if args.supports and args.source != "omomo":
        parser.error(f"--supports reads supports off an object track; --source {args.source} has no objects.")
    try:
        shard, n_shards = (int(v) for v in args.shard.split("/"))
        if not 0 <= shard < n_shards:
            raise ValueError
    except ValueError:
        parser.error(f"--shard {args.shard!r} must be K/N with integers 0 <= K < N, e.g. --shard 3/16.")

    import numpy as np

    import eden as en

    if args.source == "omomo":
        if (args.src / "human.npz").is_file():
            clips = [args.src]
        elif args.src.is_dir():
            clips = sorted(p.parent for p in args.src.glob("*/human.npz"))
        else:
            raise SystemExit(f"--src {args.src} is neither an OMOMO sequence directory nor a folder of them.")
        if not clips:
            raise SystemExit(f"--src {args.src} holds no <clip>/human.npz sequence.")
        names = [c.name for c in clips]
    else:
        _, _, clips = iter_src_files(args.src, "*.bvh")
        names = [c.stem for c in clips]
        # The search is recursive but the outputs are flat (<rig>_<clip>.npz): a name found in two folders would have
        # the second clip overwrite the first, or --skip-existing skip it as done.
        clashes = sorted(n for n, count in Counter(names).items() if count > 1)
        if clashes:
            paths = [str(c) for c, n in zip(clips, names, strict=True) if n == clashes[0]]
            raise SystemExit(
                f"--src {args.src}: {len(clashes)} clip name(s) occur in more than one folder, and each clip is "
                f"written to --out as <rig>_<clip>.npz: {', '.join(clashes[:10])}{' ...' if len(clashes) > 10 else ''} "
                f"(e.g. {clashes[0]!r} at {', '.join(paths)}). Retarget each folder as its own --src with its own "
                f"--out."
            )

    if args.clip_list is not None:
        wanted = [line.strip() for line in args.clip_list.read_text().splitlines() if line.strip()]
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise SystemExit(
                f"--clip-list {args.clip_list}: {len(missing)} clip(s) not in --src {args.src}: "
                f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''} (e.g. {names[0]!r} is one)."
            )
        keep = set(wanted)
        clips, names = zip(*[(c, n) for c, n in zip(clips, names, strict=True) if n in keep], strict=True)
    clips, names = list(clips)[shard::n_shards], list(names)[shard::n_shards]

    # The inputs are checked above without the extra, so a wrong --src or --clip-list says so wherever it is run.
    try:
        import proxsuite  # noqa: F401
        import threadpoolctl  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"OTRetarget needs ProxQP and threadpoolctl. Install them with {install_hint('ot-retargeter')}."
        ) from e
    from eden.extensions.retargeting.otretarget import PRESETS, OTRetargeter, for_body_model
    from eden.extensions.retargeting.otretarget.sources import SMPLX_HAND_JOINTS, load_omomo_sequence, load_soma_bvh
    from eden.extensions.retargeting.otretarget.supports import find_untracked_supports

    preset = for_body_model(PRESETS[args.rig], "smplx" if args.source == "omomo" else "soma")
    # The retargeter's kinematics read the rig's link poses from Genesis: float64, as it was tuned and validated.
    env = _build_single_env(_otr_rig_options(args.rig), cpu=args.cpu, precision="64")
    retargeter = OTRetargeter(env, OTRetargeter.configure(**preset.retargeter_options()))
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, (clip, name) in enumerate(zip(clips, names)):
        out_path = args.out / f"{args.rig}_{name}.npz"
        if args.skip_existing and out_path.exists():
            en.logger.info(f"[{i + 1}/{len(clips)}] skip (exists): {out_path}")
            continue
        try:
            if args.source == "omomo":
                human, obj = load_omomo_sequence(str(clip), str(args.objects_dir), args.body_model_path)
                objects = [obj]
            else:
                human, objects = load_soma_bvh(str(clip), args.body_model_path), []
            if args.num_frames is not None:
                human = human.slice(0, args.num_frames)
                for obj in objects:
                    obj.poses = obj.poses[: args.num_frames]
            if args.supports:
                hand = np.isin(human.vertex_parts, [human.joint_index(j) for j in SMPLX_HAND_JOINTS])
                supports = find_untracked_supports(objects[0], human.vertices[:, hand])
                objects += [s.as_object(human.num_frames) for s in supports]
                support_path = out_path.with_name(f"{out_path.stem}.supports.json")
                text = json.dumps([s.to_dict() for s in supports], indent=1)
                _with_io_retries(lambda: support_path.write_text(text))
                en.logger.info(f"      supports: {[(s.name, round(s.height, 3)) for s in supports]}")
            en.logger.info(f"[{i + 1}/{len(clips)}] {name}  T={human.num_frames}")
            result = retargeter.retarget_motion(human, objects, progress=sys.stderr.isatty())
            # Written under a temporary name and renamed: a job killed mid-write leaves no .npz that
            # --skip-existing would take for a finished clip.
            tmp_path = out_path.with_name(f"{out_path.stem}.partial.npz")
            _with_io_retries(lambda: result.save(str(tmp_path)))
            _with_io_retries(lambda: _replace(tmp_path, out_path))
        except Exception:
            if not args.keep_going:
                raise
            error_path = out_path.with_name(f"{out_path.stem}.error.txt")
            error_path.write_text(traceback.format_exc())
            en.logger.error(f"[{i + 1}/{len(clips)}] {name} failed; traceback in {error_path}")
            failed.append(name)
            continue
        en.logger.info(f"      -> {out_path}")
    if failed:
        raise SystemExit(
            f"eden retarget otretarget: {len(failed)} of {len(clips)} clips failed (tracebacks in "
            f"{args.out}/<rig>_<clip>.error.txt): {', '.join(failed)}"
        )


# ---------------------------------------------------------------------------
# flashdex — human hand-object demos → bimanual robot-hand motion.pt (FlashDexRetarget)
# ---------------------------------------------------------------------------

_FLASHDEX_CONFIG_HELP = (
    "A .yaml/.json file of option values (the options classes in eden.options.extensions.flashdexretarget); "
    "flags and --override apply on top."
)
#: ``eden retarget flashdex import --source`` choices: ``importer.SOURCE_DATASETS``'s keys, listed here so ``--help``
#: answers without importing the loaders (a test pins the two together).
_FLASHDEX_SOURCES = ("dexycb", "hot3d", "oakink2", "taco")
_FLASHDEX_OVERRIDE_HELP = (
    'Inline YAML mapping merged over the options last, e.g. "retarget: {attempts: 3}"; repeatable.'
)


def _merge_named(base: dict, override: dict) -> dict:
    """``override`` merged into ``base`` recursively, except that a mapping naming another ``name`` (``convex:
    {name: cuacd}`` over a CoACD config, ``--source taco`` over an OakInk2 one) replaces the old one: its keys belong
    to another options class, which would reject the old one's fields.
    """
    out = dict(base)
    for k, v in override.items():
        old = out.get(k)
        renamed = (
            isinstance(old, dict) and isinstance(v, dict) and "name" in v and old.get("name", v["name"]) != v["name"]
        )
        out[k] = _merge_named(old, v) if isinstance(old, dict) and isinstance(v, dict) and not renamed else v
    return out


def _with_group_names(cls, data: dict) -> dict:
    """``data`` with each options group of ``cls`` that it gives without a ``name`` named after the group's default
    (``convex: {seed: 2}`` is CoACD's), so a later ``{name: cuacd}`` replaces it instead of merging into it.
    """
    out = dict(data)
    for key, field in cls.model_fields.items():
        if isinstance(out.get(key), dict) and "name" not in out[key] and field.default_factory is not None:
            name = getattr(field.get_default(call_default_factory=True), "name", None)
            if isinstance(name, str):
                out[key] = {"name": name, **out[key]}
    return out


def _flashdex_options(cls, args: argparse.Namespace, flags: dict):
    """``cls`` built from ``--config``, then the flags that were passed, then every ``--override``."""
    from eden.utils.configs import overrides_from_str, read_config_file

    data = read_config_file(str(args.config)) if args.config is not None else {}
    for override in ({k: v for k, v in flags.items() if v is not None}, *map(overrides_from_str, args.override or ())):
        data = _merge_named(_with_group_names(cls, data), override)
    return cls.model_validate(data)


def _add_flashdex_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help=_FLASHDEX_CONFIG_HELP)
    parser.add_argument("--override", action="append", default=None, help=_FLASHDEX_OVERRIDE_HELP)


def _run_flashdex_import(extra_args: list[str]) -> None:
    """``eden retarget flashdex import`` — a released dataset → ``<dataset>/human_demo/`` + ``objects/``."""
    parser = argparse.ArgumentParser(
        prog="eden retarget flashdex import",
        description="Import a released hand-object dataset (MANO hands + object poses) into a FlashDexRetarget "
        "dataset: <dataset>/human_demo/<clip>.npz and <dataset>/objects/<id>/visual.obj.",
        epilog="Then retarget it with `eden retarget flashdex kinematic --dataset <dataset> --robot xhand`.",
    )
    parser.add_argument(
        "--source",
        choices=_FLASHDEX_SOURCES,
        default=None,
        help="The release (source.name); read by its eden.datasets loader.",
    )
    parser.add_argument("--raw", type=Path, default=None, help="The downloaded release's root directory.")
    parser.add_argument("--dataset", type=Path, default=None, help="The dataset directory to write.")
    parser.add_argument(
        "--mano-dir",
        type=Path,
        default=None,
        help="Directory with MANO_RIGHT.pkl and MANO_LEFT.pkl (mano.is.tue.mpg.de, 'Models & Code').",
    )
    parser.add_argument(
        "--sides", choices=("both", "right", "left"), default=None, help="Hands to import. Default: both."
    )
    parser.add_argument(
        "--textures",
        action="store_true",
        default=None,
        help="Export the objects with the release's materials and textures (for rendering); the retargeting reads "
        "only their geometry. Default: off.",
    )
    _add_flashdex_config_args(parser)
    args = parser.parse_args(extra_args)

    from eden.options.extensions.flashdexretarget import FDRImportOptions

    flags = {
        "source": {"name": args.source} if args.source else None,
        "raw": str(args.raw) if args.raw else None,
        "dataset": str(args.dataset) if args.dataset else None,
        "mano_dir": str(args.mano_dir) if args.mano_dir else None,
        "sides": args.sides,
        "textures": args.textures,
    }
    options = _flashdex_options(FDRImportOptions, args, flags)

    from eden.extensions.retargeting.flashdexretarget import import_demos

    import_demos(options)


def _run_flashdex_kinematic(extra_args: list[str]) -> None:
    """``eden retarget flashdex kinematic`` — imported clips → ``<dataset>/<robot>/motion.pt`` for one robot."""
    parser = argparse.ArgumentParser(
        prog="eden retarget flashdex kinematic",
        description="FlashDexRetarget kinematic retargeting of an imported dataset: support disks and reject rules "
        "(<robot>/report.csv), convex object parts, MuJoCo physics IK onto the robot hands (<robot>/retargeted/), "
        "<robot>/motion.pt and object SDF grids.",
        epilog='Steps can be skipped, e.g. --override "steps: {retarget: false, pack: false}" to review '
        "report.csv first.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=None, help="The dataset directory (human_demo/ and objects/ from the import)."
    )
    parser.add_argument("--robot", choices=("xhand", "sharpa"), default=None, help="Target robot. Default: xhand.")
    parser.add_argument("--workers", type=int, default=None, help="Processes. Default: half the CPU cores.")
    _add_flashdex_config_args(parser)
    args = parser.parse_args(extra_args)

    from eden.options.extensions.flashdexretarget import FDRKinematicRetargetingOptions

    flags = {"dataset": str(args.dataset) if args.dataset else None, "robot": args.robot, "workers": args.workers}
    options = _flashdex_options(FDRKinematicRetargetingOptions, args, flags)

    import eden as en
    from eden.extensions.retargeting.flashdexretarget import run_kinematic_retargeting

    motion = run_kinematic_retargeting(options)
    if motion is not None:
        en.logger.info(f"Done: {motion}")
