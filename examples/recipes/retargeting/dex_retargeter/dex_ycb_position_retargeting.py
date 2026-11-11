r"""Offline position retargeting from the DexYCB hand-object dataset.

---
tier: recipe
capabilities: [retarget.hand, robot.xhand, dataset.dex_ycb]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py --data-path /path/to/dex-ycb/extracted/root --calibration-dir /path/to/dex-ycb/calibration
---

Eden port of ``dex-retargeting``'s ``example/position_retargeting`` workflow,
using :class:`eden.datasets.dex_ycb_dataset.DexYCBDataset` instead of MANO+DexYCB-toolkit
plumbing. Each clip's 21 MANO keypoints per frame (in
:data:`eden.datasets.constants.MANO_KEYPOINTS_NAME` order) are matched onto
the robot fingertips named by ``XHand1_R.metadata.target_hand_links``, indexed
by ``XHand1_R.metadata.mano_hand_idx``.

The Eden port targets a **fixed-base** hand entity, so each frame's MANO
keypoints are re-anchored from DexYCB's AprilTag world frame to the robot's
wrist-mount position before being fed to the optimizer:

.. math::
    p_\\text{target} = p_\\text{MANO} - p_\\text{MANO,wrist} + p_\\text{robot,wrist}

The retargeting frame and the saved orientation **must be the same**, or
the fingertip and wrist alignment diverges catastrophically. Approach:

1. For each frame, pre-rotate the MANO keypoints into the MANO wrist's
   local frame: ``rel_local = R(mano_quat).T @ (mano_kp - mano_wrist)``.
2. Anchor at the robot wrist (still fixed-base at identity orientation):
   ``target = robot_wrist + rel_local``. The retargeter solves
   ``dofs_pos`` such that the robot's identity-orientation fingertips
   reach ``target``, i.e. ``finger_local ≈ rel_local``.
3. Save ``base_quat_w = mano_wrist_quat``. At viz time, the robot is
   placed at ``(mano_wrist, mano_quat)`` and applies ``dofs_pos``,
   reconstructing world-frame fingertips as
   ``mano_wrist + R(mano_quat) @ finger_local = mano_kp`` ✓.

This works because the MANO wrist-local convention (fingers along +Z,
thumb along +Y for the right hand) is close enough to the XHand1 URDF's
wrist-local convention that the optimizer's reachable workspace covers
MANO's wrist-relative finger positions. The residual error is dominated
by SLSQP non-convexity (~1-3 cm typical), not 10+ cm rigid offsets from
a "fingers up" URDF being asked to reach "fingers down" targets.

Per-clip ``fingertip_err_{mean,max}_m`` is logged at retarget time and
saved into the NPZ for filtering. ``--view --debug-vis`` overlays the
MANO keypoints (blue/green) and the robot fingertips (red) so the
alignment is visible side-by-side.

Selecting a subset of clips
---------------------------

Filters are AND-combined; each accepts one or more substring patterns
matched case-insensitively against the per-clip metadata:

- ``--side {right,left}``: hand side (required match, defaults to ``right``).
- ``--objects PAT [PAT ...]``: matches against the YCB folder name
  (e.g. ``mustard`` matches ``006_mustard_bottle``).
- ``--subjects PAT [PAT ...]``: matches against the sequence id prefix
  (e.g. ``subject-01`` matches ``20200709-subject-01/...``).
- ``--max-clips N``: cap on the number of matching clips.
- ``--list``: print the filtered catalog and exit (no retargeting).

Example::

    uv run python examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py \\
        --data-path /path/to/dex-ycb/extracted/root \\
        --calibration-dir /path/to/dex-ycb/calibration \\
        --side right \\
        --objects mustard tomato_soup \\
        --subjects subject-01

Output
------

For each retargeted clip a NPZ is written **next to** that clip's
``pose.npz`` as ``retargeted_<label>.npz`` (default label is the preset
name with the optimizer suffix stripped, e.g. ``xhand1_right``). The
schema is what :class:`DexYCBDataset` expects for its third motion key:

- ``dofs_pos``: ``(T, n_dofs)`` — retargeted joint angles.
- ``dofs_name``: ``(n_dofs,)`` string array — the robot's
  ``entity.dofs_name`` ordering.
- ``base_pos_w``: ``(T, 3)`` — the **MANO wrist keypoint**
  (``hand.links_pos_w[..., 0]``) in the AprilTag world frame. This is
  what the retargeting was anchored at, so placing the robot here at
  viz time produces a round-trip-consistent overlay.
- ``base_quat_w``: ``(T, 4)`` wxyz — identity per frame (see above).
- ``framerate``: scalar.
- ``label`` / ``preset``: string metadata.

Reload the whole dataset (hand + object + robot) via::

    DexYCBDataset(options=DexYCBDataset.configure(
        raw_data_path=..., calibration_dir=...,
        retargeted_label="xhand1_right",
    ))
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import genesis as gs
import numpy as np
import torch

import eden as en
from eden.datasets.constants import MANO_KEYPOINTS_NAME
from eden.datasets.dex_ycb_dataset import DexYCBDataset
from eden.envs.base import EnvBase
from eden.extensions.retargeting.dex_retargeter import DexRetargeter
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions

from hand_presets import HAND_PRESETS

_MANO_WRIST_IDX = MANO_KEYPOINTS_NAME.index("wrist")

# (urdf↔mano rotation is derived at runtime from the robot's `up`/`front`
# declarations via `dex_retargeter.calibrate.urdf_to_mano_rot`; no
# metadata field needed.)


# Skeleton edges over ``MANO_KEYPOINTS_NAME`` for the debug overlay: five
# chains of wrist → 3 knuckles → 1 tip, mirroring `examples/recipes/datasets/dex_ycb.py`.
_KP = {n: i for i, n in enumerate(MANO_KEYPOINTS_NAME)}
_FINGER_CHAINS = (
    ("thumb1", "thumb2", "thumb3", "thumb_tip"),
    ("index1", "index2", "index3", "index_tip"),
    ("middle1", "middle2", "middle3", "middle_tip"),
    ("ring1", "ring2", "ring3", "ring_tip"),
    ("pinky1", "pinky2", "pinky3", "pinky_tip"),
)
_HAND_BONES: tuple[tuple[int, int], ...] = tuple(
    (_KP["wrist" if i == 0 else chain[i - 1]], _KP[chain[i]]) for chain in _FINGER_CHAINS for i in range(4)
)

# Filename pattern for the retargeted NPZ written next to each clip's pose.npz.
# Matches DexYCBDataset's `retargeted_label` loader: it looks for
# ``<seq_dir>/retargeted_<label>.npz`` when configured.
_RETARGETED_NPZ_TEMPLATE = "retargeted_{label}.npz"


def _ycb_class_for_folder(folder_name: str):
    """Resolve a DexYCB ``ycb_id`` folder name to its ``YCBOptions`` subclass.

    Same lookup as ``examples/recipes/datasets/dex_ycb.py``: matches on the
    declared ``file`` default ``models/<folder>/textured.obj``.
    """
    import inspect

    from eden.options.objects import ycb as ycb_module
    from eden.options.objects.ycb import YCBOptions

    needle = f"models/{folder_name}/"
    for _, cls in inspect.getmembers(ycb_module, inspect.isclass):
        if cls is YCBOptions or not issubclass(cls, YCBOptions):
            continue
        field = cls.model_fields.get("file")
        if field is not None and needle in (field.default or ""):
            return cls
    raise KeyError(f"No YCBOptions subclass found for DexYCB object {folder_name!r}.")


def _matches_any(value: str, patterns: list[str] | None) -> bool:
    """``True`` when ``patterns`` is empty/None or any substring matches (case-insensitive)."""
    if not patterns:
        return True
    v = value.lower()
    return any(p.lower() in v for p in patterns)


def _select_clip_indices(
    dataset: DexYCBDataset,
    side: str,
    objects_patterns: list[str] | None,
    subjects_patterns: list[str] | None,
    max_clips: int | None,
) -> list[int]:
    """Return the indices of clips passing every active filter, in dataset order."""
    sides = list(dataset.metadata["sides"])
    objects = list(dataset.metadata["objects"])
    sequence_ids = list(dataset.metadata["sequence_ids"])

    matching: list[int] = []
    for i, s in enumerate(sides):
        if s != side:
            continue
        if not _matches_any(objects[i], objects_patterns):
            continue
        if not _matches_any(sequence_ids[i], subjects_patterns):
            continue
        matching.append(i)

    if not matching:
        available = sorted(set(objects))
        raise SystemExit(
            f"No clips passed the filters (side={side!r}, objects={objects_patterns}, "
            f"subjects={subjects_patterns}). Available object names: {available}"
        )
    if max_clips is not None:
        matching = matching[:max_clips]
    return matching


def _print_catalog(dataset: DexYCBDataset, indices: list[int]) -> None:
    """Print the filtered clip catalog to stdout."""
    objects = list(dataset.metadata["objects"])
    sides = list(dataset.metadata["sides"])
    sequence_ids = list(dataset.metadata["sequence_ids"])
    hand_seq = dataset.motion_sequences["hand"]
    print(f"{'#':>4}  {'side':<5}  {'object':<24}  {'frames':>6}  sequence_id")
    for n, i in enumerate(indices):
        T = int(hand_seq.buffer_length[i].item())
        print(f"{n:>4}  {sides[i]:<5}  {objects[i]:<24}  {T:>6}  {sequence_ids[i]}")
    print(f"\n{len(indices)} clips match.")


def _default_retargeted_label(preset: str) -> str:
    """Derive a default label from the preset name (strip trailing optimizer tag)."""
    for suffix in ("_position", "_vector", "_dexpilot"):
        if preset.endswith(suffix):
            return preset[: -len(suffix)]
    return preset


def _retarget_clip(
    retargeter: DexRetargeter,
    keypoints: np.ndarray,  # (T, 21, 3) — MANO_KEYPOINTS_NAME order, AprilTag world frame
    mano_wrist_quat: np.ndarray,  # (T, 4) wxyz — per-frame MANO wrist orientation
    robot_wrist_pos: np.ndarray,  # (3,) — robot wrist mount position
    R_mano_to_world: np.ndarray,  # (3, 3) — derived via `calibrate.urdf_to_mano_rot`
) -> np.ndarray:
    """Retarget one clip and return ``(T, n_dofs)`` qpos.

    Transforms each frame's MANO keypoints into the **canonical world
    frame** the robot is loaded into (via its ``up`` / ``front``
    declarations), then anchors at the robot wrist:

        rel_canonical = R_mano_to_world @ R(mano_quat).T @ (mano_kp - mano_wrist)
        target        = robot_wrist + rel_canonical

    ``R_mano_to_world`` is ``align_up_and_front(MANO_UP, MANO_FRONT)``
    — the same machinery Eden uses to map any asset's canonical frame
    to a z-up, x-front world. With the hand-axis convention applied via
    ``up``/``front`` on both the robot and (implicitly, as a constant)
    MANO, the URDF↔MANO conversion is automatic.

    Without this transformation, the optimizer's targets land outside
    the robot's reachable fingertip workspace (e.g. fingers in MANO -X
    pointing direction vs robot's URDF +Z direction) and the retargeted
    poses are off by 10+ cm.
    """
    from genesis.utils.geom import quat_to_R

    retargeter.reset()
    wrist_per_frame = keypoints[:, _MANO_WRIST_IDX : _MANO_WRIST_IDX + 1, :]  # (T, 1, 3)
    rel_world = keypoints - wrist_per_frame  # (T, 21, 3)

    # Per-frame: world → MANO local (undo wrist quat), then MANO local
    # → canonical world (= the robot's post-load world frame).
    # Row-vector identity: `v @ R == (R.T @ v.T).T`.
    rel_canonical = np.empty_like(rel_world)
    for t in range(rel_world.shape[0]):
        R_mano = quat_to_R(mano_wrist_quat[t].astype(np.float32))
        rel_local_mano = rel_world[t] @ R_mano  # world → MANO local
        rel_canonical[t] = rel_local_mano @ R_mano_to_world.T  # MANO local → canonical

    anchored = rel_canonical + robot_wrist_pos[None, None, :]  # (T, 21, 3)
    indices = retargeter.optimizer.target_link_human_indices
    ref_seq = anchored[:, indices, :]  # (T, n_target_links, 3)
    return retargeter.retarget_motion(ref_seq)


def _verify_fingertip_alignment(
    retargeter: DexRetargeter,
    robot,
    qpos: np.ndarray,  # (T, n_dofs)
    keypoints: np.ndarray,  # (T, 21, 3) — world frame
    mano_wrist_quat: np.ndarray,  # (T, 4) wxyz
    robot_wrist_pos: np.ndarray,  # (3,) — robot wrist mount
    R_mano_to_world: np.ndarray,  # (3, 3)
) -> tuple[float, float]:
    """FK check that the retargeted fingertips match MANO in canonical world.

    Drives the fixed-base retargeting env with each frame's qpos, reads
    world-frame fingertip positions, and compares against the (MANO →
    canonical world, then anchored at the robot wrist) MANO fingertip
    targets. Returns ``(mean_err, max_err)`` in metres.
    """
    from genesis.utils.geom import quat_to_R

    indices = retargeter.optimizer.target_link_human_indices
    target_links = retargeter.optimizer._target_links

    wrist_per_frame = keypoints[:, _MANO_WRIST_IDX : _MANO_WRIST_IDX + 1, :]
    rel_world = keypoints - wrist_per_frame
    rel_canonical = np.empty_like(rel_world)
    for t in range(rel_world.shape[0]):
        R_mano = quat_to_R(mano_wrist_quat[t].astype(np.float32))
        rel_local_mano = rel_world[t] @ R_mano
        rel_canonical[t] = rel_local_mano @ R_mano_to_world.T
    mano_fingertips_robot = (rel_canonical + robot_wrist_pos[None, None, :])[:, indices, :]

    T = qpos.shape[0]
    robot_fingertips = np.empty_like(mano_fingertips_robot)
    for t in range(T):
        qpos_t = torch.as_tensor(qpos[t], dtype=torch.float32, device=robot.device).unsqueeze(0)
        robot.set_qpos(qpos_t)
        robot_fingertips[t] = np.stack([link.get_pos().squeeze(0).cpu().numpy() for link in target_links])

    err = np.linalg.norm(robot_fingertips - mano_fingertips_robot, axis=-1)
    return float(err.mean()), float(err.max())


def view_clip(
    dataset: DexYCBDataset,
    clip_idx: int,
    spec,
    show_viewer: bool = True,
    output_video: Path | None = None,
    debug_vis: bool = False,
) -> None:
    """Replay one retargeted clip in the Genesis viewer: YCB object + robot hand.

    Requires ``dataset`` to have been loaded with ``retargeted_label=<label>``
    so the third motion key (``"robot"``) is populated. The robot is built
    floating-base so its full pose ``[base_pos(3), base_quat(4), dofs(N)]``
    can be driven directly via ``set_qpos``; the YCB object pose is driven
    via ``set_pos``/``set_quat`` per frame.

    Parameters
    ----------
    dataset : DexYCBDataset
        Already prepared with ``retargeted_label`` set.
    clip_idx : int
        Index into ``dataset.metadata`` / motion sequences.
    spec : HandPreset
        The preset whose ``robot_options_factory`` provides the robot URDF;
        we override ``is_fixed_base=False`` to expose a free joint at the
        root for visualization-time relocation.
    show_viewer : bool
        Open the interactive Genesis viewer.
    output_video : Path or None
        When set, also write an MP4 of the playback.
    debug_vis : bool
        Overlay the source MANO keypoints (21 spheres) and skeleton edges
        from ``dataset.motion_sequences['hand'].links_pos_w`` on top of
        the robot. Useful for eyeballing how well the retargeting tracks
        the human hand.
    """
    if "robot" not in dataset.motion_sequences:
        raise RuntimeError(
            "view_clip requires the dataset to be loaded with "
            "`retargeted_label=...` so motion_sequences['robot'] is populated."
        )

    object_name = dataset.metadata["objects"][clip_idx]
    sequence_id = dataset.metadata["sequence_ids"][clip_idx]
    object_cls = _ycb_class_for_folder(object_name)

    # Floating-base hand: lets us set the wrist pose every frame as part of qpos.
    robot_options = spec.robot_options_factory()
    robot_options.is_fixed_base = False

    cameras_options = None
    if output_video is not None:
        from eden.options.camera import CameraOptions, CamerasOptions

        cameras_options = CamerasOptions(
            camera=CameraOptions(
                # Frame fits the action cluster in AprilTag world coords
                # (matches `examples/recipes/datasets/dex_ycb.py`).
                cam_pos=(0.9, -0.05, 0.45),
                cam_lookat=(0.55, 0.35, 0.15),
                cam_fov=45,
                debug=True,
            )
        )

    framerate = float(dataset.motion_sequences["robot"].framerate[clip_idx].item())
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / framerate,
            decimation=1,
            show_world_frame=True,
        ),
        scene_options=SceneOptions(object=object_cls(), robot=robot_options, plane=PlaneOptions()),
        cameras_options=cameras_options,
        show_viewer=show_viewer,
    )
    env.build()
    env.reset()

    obj = env.entities["object"]
    robot = env.entities["robot"]
    obj_seq = dataset.motion_sequences["object"]
    robot_seq = dataset.motion_sequences["robot"]
    # MANO keypoints for the optional overlay (skipped when debug_vis=False).
    hand_seq = dataset.motion_sequences["hand"] if debug_vis else None
    side = dataset.metadata["sides"][clip_idx]
    overlay_kp_color = (0.0, 0.0, 1.0, 0.8) if side == "right" else (0.0, 1.0, 0.0, 0.8)
    overlay_bone_color = (1.0, 0.4, 0.4, 0.8)
    # In debug_vis mode also draw the *robot's* fingertips (red) on top of
    # the MANO ones (blue/green) so any residual alignment gap is visible.
    fingertip_link_names: list[str] = []
    fingertip_links: list = []
    fingertip_mano_idx = None
    if debug_vis:
        # Read fingertip link names from the robot's metadata.
        md = getattr(robot.options, "metadata", None)
        if md is not None and getattr(md, "target_hand_links", None) is not None:
            fingertip_link_names = list(md.target_hand_links)
            fingertip_links = [robot.get_link(n) for n in fingertip_link_names]
            mano_idx_attr = getattr(md, "mano_hand_idx", None)
            if mano_idx_attr is not None:
                fingertip_mano_idx = np.asarray(mano_idx_attr, dtype=int)

    n_frames = int(obj_seq.buffer_length[clip_idx].item())
    cam = env.cameras.get("camera") if output_video is not None else None
    if cam is not None:
        cam.start_recording(video_path=str(output_video), fps=framerate)

    en.logger.info(f"Replaying {sequence_id} ({object_name}, T={n_frames})")
    for t in range(n_frames):
        # YCB object: free-base, drive directly.
        obj.set_pos(obj_seq.base_pos_w[clip_idx, t])
        obj.set_quat(obj_seq.base_quat_w[clip_idx, t])

        # Robot: floating-base qpos layout = [pos(3), quat_wxyz(4), joints(N)].
        # `base_pos_w` / `base_quat_w` are the MANO wrist pose; `dofs_pos`
        # is the retargeter output (wrist-relative finger angles).
        qpos = torch.cat(
            [
                robot_seq.base_pos_w[clip_idx, t],
                robot_seq.base_quat_w[clip_idx, t],
                robot_seq.dofs_pos[clip_idx, t],
            ]
        ).unsqueeze(0)
        robot.set_qpos(qpos)

        # Source MANO hand overlay (21 keypoints + 20 bones) and the
        # robot's actual fingertip world positions (RED) for visual
        # alignment debugging.
        if hand_seq is not None:
            kp = hand_seq.links_pos_w[clip_idx, t]  # (21, 3) in MANO_KEYPOINTS_NAME order
            env.draw_debug_spheres(kp, radius=0.006, color=overlay_kp_color)
            for a, b in _HAND_BONES:
                env.draw_debug_line(kp[a], kp[b], color=overlay_bone_color)

            if fingertip_links:
                robot_tips = np.stack([link.get_pos().squeeze(0).cpu().numpy() for link in fingertip_links])
                # Red = robot fingertips. Slightly larger so they stand out
                # even when overlapping the MANO keypoints.
                env.draw_debug_spheres(robot_tips, radius=0.008, color=(1.0, 0.1, 0.1, 0.9))

                # Log world-frame alignment to MANO fingertip keypoints for
                # the first frame of each clip — surfaces any frame-mismatch
                # bug numerically without spamming the log.
                if t == 0 and fingertip_mano_idx is not None:
                    kp_np = kp.cpu().numpy() if hasattr(kp, "cpu") else np.asarray(kp)
                    mano_tips = kp_np[fingertip_mano_idx, :]
                    diff = robot_tips - mano_tips
                    err_mm = np.linalg.norm(diff, axis=-1) * 1000.0
                    en.logger.info(
                        f"  frame 0 fingertip alignment (robot vs MANO, mm): "
                        f"{np.round(err_mm, 1).tolist()}  "
                        f"(mean offset xyz mm: {np.round(diff.mean(0) * 1000, 1).tolist()})"
                    )

        env.scene.visualizer.update()
        if cam is not None:
            cam.render_rgb()

        if hand_seq is not None:
            env.clear_debug_objects()

    if cam is not None:
        cam.stop_recording()
        en.logger.info(f"Wrote {output_video}")


def main(
    data_path: Path,
    calibration_dir: Path | None,
    preset: str,
    side: str,
    objects_patterns: list[str] | None,
    subjects_patterns: list[str] | None,
    max_clips: int | None,
    list_only: bool,
    record: bool,
    retargeted_label: str | None,
    target_framerate: float,
    view: bool,
    debug_vis: bool,
) -> None:
    if preset not in HAND_PRESETS:
        raise SystemExit(f"Unknown preset {preset!r}. Available: {list(HAND_PRESETS)}")

    en.init(backend=gs.cpu)
    spec = HAND_PRESETS[preset]
    label = retargeted_label or _default_retargeted_label(preset)

    # ------------------------------------------------------------------
    # Dataset (load first so --list can skip env construction).
    # In --view mode we also ask the dataset to pull the retargeted
    # NPZs as the third motion key.
    # ------------------------------------------------------------------
    if calibration_dir is None:
        # Convention from examples/recipes/datasets/dex_ycb.py: `<data-path>/calibration`.
        default_calib = data_path / "calibration"
        calibration_dir = default_calib if default_calib.is_dir() else None
    dataset_kwargs = dict(
        raw_data_path=str(data_path),
        calibration_dir=str(calibration_dir) if calibration_dir else None,
        target_framerate=target_framerate,
    )
    if view:
        dataset_kwargs["retargeted_label"] = label
    dataset = DexYCBDataset(options=DexYCBDataset.configure(**dataset_kwargs), device=gs.device)
    en.logger.info(f"Loaded {len(dataset)} DexYCB clips from {data_path}")

    matching = _select_clip_indices(dataset, side, objects_patterns, subjects_patterns, max_clips)

    if list_only:
        _print_catalog(dataset, matching)
        return

    if view:
        en.logger.info(f"Viewing {len(matching)} clip(s) with retargeted label {label!r}")
        for n, clip_idx in enumerate(matching):
            video_path = None
            if record:
                seq_dir = Path(os.path.dirname(dataset.all_motion_files[clip_idx]))
                video_path = seq_dir / f"retargeted_{label}.mp4"
            # Genesis allows only one interactive viewer at a time, so each
            # clip needs a fresh `gs.destroy()` + `en.init()` before
            # `view_clip` builds its scene.
            if n > 0:
                gs.destroy()
                en.init(backend=gs.cpu)
            view_clip(
                dataset,
                clip_idx,
                spec,
                show_viewer=True,
                output_video=video_path,
                debug_vis=debug_vis,
            )
        return

    hand_seq = dataset.motion_sequences["hand"]
    objects = list(dataset.metadata["objects"])
    sequence_ids = list(dataset.metadata["sequence_ids"])
    # `all_motion_files` is aligned 1:1 with the metadata lists post-prepare
    # (DexYCBDataset filters in lock-step in `prepare`).
    pose_paths = list(dataset.all_motion_files)
    en.logger.info(f"Retargeting {len(matching)} {side}-hand clips")

    if record:
        en.logger.warning(
            "--record is only honored in --view mode (the retargeting env has no "
            "YCB object). Re-run with `--view --record` after retargeting to "
            "produce hand+object MP4s."
        )

    # ------------------------------------------------------------------
    # Env + retargeter
    # ------------------------------------------------------------------
    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=spec.robot_options_factory(), floor=PlaneOptions()),
        show_viewer=False,
    )
    env.build()
    env.reset()
    robot = env.entities["robot"]

    retargeter = DexRetargeter(env=env, options=spec.retargeter_options())
    if retargeter.retargeting_type != "POSITION":
        raise SystemExit(f"Preset {preset!r} is type {retargeter.retargeting_type!r}; expected POSITION.")

    # For hand URDFs the root link is the wrist mount.
    wrist_link = robot.links[0]
    robot_wrist_pos = wrist_link.get_pos().squeeze(0).cpu().numpy()
    en.logger.info(f"Robot wrist link: {wrist_link.name!r} at {np.round(robot_wrist_pos, 3).tolist()}")

    # Robot↔MANO frame alignment, derived from the robot's `up`/`front`
    # declarations via the same `align_up_and_front` machinery Eden uses
    # to align any asset's canonical frame with a z-up, x-front world.
    # MANO's canonical (up=fingers, front=palm-normal) is a constant in
    # the calibrate module. Stays constant per clip; compute once.
    from eden.extensions.retargeting.dex_retargeter.calibrate import urdf_to_mano_rot

    R_mano_to_world, q_urdf_to_mano = urdf_to_mano_rot(robot)

    # ------------------------------------------------------------------
    # Retarget each clip — write an NPZ next to the source `pose.npz` so
    # `DexYCBDataset(retargeted_label=<label>)` can reload it as a third
    # motion key. The base pose tracks the MANO wrist (matches
    # `hand_seq.base_pos_w`/`base_quat_w`), and `dofs_pos` is the
    # retargeter output. Together they encode a world-frame-coherent
    # robot trajectory if a consumer places the robot at
    # ``(base_pos_w, base_quat_w)`` and applies ``dofs_pos``.
    # ------------------------------------------------------------------
    en.logger.info(
        f"Writing retargeted NPZs with label={label!r} (file: {_RETARGETED_NPZ_TEMPLATE.format(label=label)})"
    )

    total_tic = time.perf_counter()
    for n, clip_idx in enumerate(matching):
        T = int(hand_seq.buffer_length[clip_idx].item())
        kp = hand_seq.links_pos_w[clip_idx, :T].cpu().numpy()  # (T, 21, 3)
        mano_quat = hand_seq.base_quat_w[clip_idx, :T].cpu().numpy()  # (T, 4) wxyz
        clip_id = sequence_ids[clip_idx]
        seq_dir = Path(os.path.dirname(pose_paths[clip_idx]))

        tic = time.perf_counter()
        qpos = _retarget_clip(retargeter, kp, mano_quat, robot_wrist_pos, R_mano_to_world)
        elapsed = time.perf_counter() - tic

        # FK check: do the world-frame fingertips line up with the
        # (canonical-world-anchored) MANO targets? Should be within
        # SLSQP residual (~1-3 cm).
        mean_err, max_err = _verify_fingertip_alignment(
            retargeter, robot, qpos, kp, mano_quat, robot_wrist_pos, R_mano_to_world
        )
        en.logger.info(
            f"[{n + 1}/{len(matching)}] {clip_id} ({objects[clip_idx]}, T={T}) — "
            f"{elapsed:.1f}s; fingertip err mean={mean_err * 1000:.2f}mm "
            f"max={max_err * 1000:.2f}mm"
        )

        out_path = seq_dir / _RETARGETED_NPZ_TEMPLATE.format(label=label)
        # Compose `base_quat_w = mano_quat ⊗ q_urdf_to_mano` so viz
        # reconstructs world-frame fingertips matching MANO. The
        # `urdf_to_mano_quat` comes from the robot's metadata, so this
        # generalizes across hand URDFs without per-script hacks.
        # `transform_quat_by_quat(v, u) = u * v`, so passing
        # (q_urdf_to_mano, mano_quat) gives `mano_quat * q_urdf_to_mano`.
        from genesis.utils.geom import transform_quat_by_quat

        q_urdf_to_mano_b = np.broadcast_to(q_urdf_to_mano, (T, 4)).copy()
        base_quat = transform_quat_by_quat(q_urdf_to_mano_b, mano_quat.astype(np.float32)).astype(np.float32)
        wrist_kp = kp[:, _MANO_WRIST_IDX, :].astype(np.float32)
        np.savez(
            out_path,
            dofs_pos=qpos.astype(np.float32),
            dofs_name=np.array(retargeter.joint_names),
            base_pos_w=wrist_kp,
            base_quat_w=base_quat,
            framerate=np.float32(dataset.target_framerate),
            label=label,
            preset=preset,
            fingertip_err_mean_m=np.float32(mean_err),
            fingertip_err_max_m=np.float32(max_err),
        )

    en.logger.info(
        f"Done. Total {time.perf_counter() - total_tic:.1f}s. "
        f"Reload via DexYCBDataset(..., retargeted_label={label!r}) or "
        f"replay with `--view --retargeted-label {label}`."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Extracted DexYCB root (holding <subject>/<sequence>/pose.npz).",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="DexYCB calibration/ directory. Defaults to <data-path>/calibration.",
    )
    parser.add_argument(
        "--retargeted-label",
        type=str,
        default=None,
        help="Label used in the inline NPZ filename (`retargeted_<label>.npz` "
        "written next to each clip's pose.npz). Defaults to the preset name "
        "with the optimizer-type suffix stripped (e.g. xhand1_right_position "
        "→ xhand1_right). Pass the same label to "
        "`DexYCBDataset(retargeted_label=...)` to reload.",
    )
    parser.add_argument(
        "--target-framerate",
        type=float,
        default=30.0,
        help="Resample DexYCB clips to this framerate before retargeting "
        "(also recorded in the output NPZ so reload is a no-op). DexYCB's "
        "raw capture rate is 30 Hz.",
    )
    parser.add_argument("--preset", default="xhand1_right_position", help="Retargeting preset to apply")
    parser.add_argument("--side", choices=["right", "left"], default="right", help="Which hand to drive")
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        metavar="PAT",
        help="Substring patterns to match against the YCB object folder name "
        "(e.g. 'mustard tomato_soup'). Case-insensitive. Default: match all.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        metavar="PAT",
        help="Substring patterns to match against the sequence id prefix "
        "(e.g. 'subject-01'). Case-insensitive. Default: match all.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Cap on clips to retarget (default: all matching clips).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="Print the filtered clip catalog and exit (no retargeting).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Skip retargeting; reload previously-retargeted clips via "
        "DexYCBDataset(retargeted_label=...) and replay each in the Genesis "
        "viewer (YCB object + robot hand). Requires `retargeted_<label>.npz` "
        "to already exist next to each matched clip's pose.npz.",
    )
    parser.add_argument(
        "-r",
        "--record",
        action="store_true",
        help="In retargeting mode: render an MP4 per clip. In --view mode: "
        "also save an MP4 per viewed clip as `retargeted_<label>.mp4` "
        "next to its pose.npz.",
    )
    parser.add_argument(
        "-d",
        "--debug-vis",
        action="store_true",
        help="In --view mode: overlay the source MANO hand (21 keypoint "
        "spheres + 20 skeleton edges) on top of the robot, so you can "
        "eyeball the retargeting fidelity. Blue spheres for right hand, "
        "green for left; pink bone lines.",
    )
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        calibration_dir=args.calibration_dir,
        preset=args.preset,
        side=args.side,
        objects_patterns=args.objects,
        subjects_patterns=args.subjects,
        max_clips=args.max_clips,
        list_only=args.list_only,
        record=args.record,
        retargeted_label=args.retargeted_label,
        target_framerate=args.target_framerate,
        view=args.view,
        debug_vis=args.debug_vis,
    )
