r"""Realtime hand teleoperation in the Genesis viewer.

---
tier: recipe
capabilities: [retarget.hand, robot.xhand]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/retargeting/dex_retargeter/show_realtime_retargeting.py --preset xhand1_right_dexpilot --hand-type right
---

Eden port of ``dex-retargeting``'s ``example/vector_retargeting/show_realtime_retargeting.py``.
Captures a webcam stream, runs MediaPipe + the Eden ``DexRetargeter`` per
frame, and drives a Genesis ``RigidEntity`` so the user can see the robot
hand mimic their motion in realtime.

Run::

    uv run python examples/recipes/retargeting/dex_retargeter/show_realtime_retargeting.py \\
        --preset xhand1_right_dexpilot --hand-type right

Camera selection: ``--camera-path`` accepts ``None`` (default = index 0),
an integer index, or ``/dev/videoN``. On multi-stream cameras like
RealSense, several ``/dev/video*`` nodes exist (RGB / depth / IR /
metadata) and only the RGB one produces a usable stream — try
``--camera-path 1``, ``2``, ``3``, etc. until you hit RGB. The
producer prints a clear FATAL message naming all available devices if
the chosen index doesn't open or yields no frames.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import time
from queue import Empty, Full
from typing import Optional

import cv2
import torch
import tyro

import eden as en
import genesis as gs
from eden.envs.base import EnvBase
from eden.extensions.retargeting.dex_retargeter import DexRetargeter
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions

from hand_presets import HAND_PRESETS
from single_hand_detector import SingleHandDetector


def _resolve_camera_source(camera_path: Optional[str]) -> int | str:
    """Resolve a `--camera-path` argument to something `cv2.VideoCapture` can open.

    Accepts ``None`` (default = index 0), an integer index, ``/dev/videoN``,
    or any other string (e.g. RTSP URL) passed through verbatim.

    No auto-discovery: probing UVC devices is dangerous on RealSense and
    similar multi-stream cameras because ``cv2.VideoCapture(idx)`` can
    block in the constructor (holding the GIL — Ctrl+C cannot interrupt
    it) and there is no portable way to time-bound a C++ call from
    Python without forking a subprocess. The producer process surfaces a
    clear FATAL message listing alternatives if the chosen index doesn't
    open, so the user can iterate.
    """
    if camera_path is None:
        return 0
    if camera_path.isdigit():
        return int(camera_path)
    if camera_path.startswith("/dev/video") and camera_path[len("/dev/video") :].isdigit():
        # Prefer integer index — string paths route through OpenCV's FFmpeg
        # backend and hang on non-capture UVC nodes.
        return int(camera_path[len("/dev/video") :])
    return camera_path


def _produce_frames(queue: mp.Queue, src: int | str) -> None:
    cap = cv2.VideoCapture(src)
    try:
        if not cap.isOpened():
            existing = sorted(glob.glob("/dev/video*"))
            print(
                f"[producer] FATAL: cv2.VideoCapture({src!r}) failed to open. "
                f"Existing video devices: {existing or '(none)'}. "
                f"Check `fuser /dev/video*` for stuck holders, "
                f"or pass a different --camera-path.",
                flush=True,
            )
            return
        print(f"[producer] capturing from {src}", flush=True)
        consecutive_failures = 0
        first_frame_seen = False
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                consecutive_failures += 1
                # If we never got a frame, the chosen device is almost
                # certainly a depth/IR/metadata node — exit fast with a
                # clear message instead of looping forever.
                if not first_frame_seen and consecutive_failures >= 5:
                    existing = sorted(glob.glob("/dev/video*"))
                    print(
                        f"[producer] FATAL: no frames from {src} after 5 read attempts. "
                        f"This is usually a depth/IR/metadata node on a multi-stream camera "
                        f"(e.g. RealSense). Existing devices: {existing or '(none)'}. "
                        f"Pass --camera-path N with a different index.",
                        flush=True,
                    )
                    return
                if consecutive_failures % 30 == 0:
                    print(
                        f"[producer] {consecutive_failures} consecutive read failures from {src}",
                        flush=True,
                    )
                continue
            first_frame_seen = True
            consecutive_failures = 0
            # Drop-oldest: queue holds at most 1 frame so the consumer always
            # processes the latest. Without this, frames pile up whenever the
            # consumer (MediaPipe + SLSQP + viewer) runs slower than 30fps and
            # the user sees seconds of accumulated lag.
            try:
                queue.put_nowait(image)
            except Full:
                try:
                    queue.get_nowait()
                except Empty:
                    pass
                try:
                    queue.put_nowait(image)
                except Full:
                    pass
    finally:
        # Release the V4L2 handle so the next run isn't blocked by a stuck holder.
        cap.release()


def _start_retargeting(queue: mp.Queue, preset: str, hand_type: str) -> None:
    spec = HAND_PRESETS[preset]
    en.init(backend=gs.cpu)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=spec.robot_options_factory(),
            floor=PlaneOptions(),
        ),
        show_viewer=True,
    )
    env.build()
    env.reset()

    robot = env.entities["robot"]
    retargeter = DexRetargeter(env=env, options=spec.retargeter_options())

    detector = SingleHandDetector(hand_type=hand_type.capitalize(), selfie=False)

    target_link_human_indices = retargeter.optimizer.target_link_human_indices
    retargeting_type = retargeter.retargeting_type

    while True:
        try:
            bgr = queue.get(timeout=15)
        except Empty:
            en.logger.error("No webcam frame received in 15s — producer may have died.")
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        num_box, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is None:
            en.logger.warning(f"{hand_type} hand not detected.")
        else:
            ref_value = DexRetargeter.extract_ref_value(joint_pos, target_link_human_indices, retargeting_type)
            qpos = retargeter.retarget(ref_value)
            qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device=robot.device).unsqueeze(0)
            robot.set_qpos(qpos_t)
            env.scene.visualizer.update()


def main(
    preset: str = "xhand1_right_dexpilot",
    hand_type: str = "right",
    camera_path: Optional[str] = None,
) -> None:
    """Realtime webcam hand teleoperation.

    Args:
        preset: which preset in :mod:`examples.dex_retargeter.hand_presets`.
        hand_type: ``"left"`` or ``"right"``.
        camera_path: OpenCV camera path. ``None`` opens the default webcam.
    """
    if preset not in HAND_PRESETS:
        raise SystemExit(f"Unknown preset {preset!r}. Available: {list(HAND_PRESETS)}")

    src = _resolve_camera_source(camera_path)
    print(f"[main] resolved camera source: {src}", flush=True)

    # Single-slot queue + drop-oldest in the producer keeps the consumer on
    # the freshest frame instead of slogging through stale ones.
    queue: mp.Queue = mp.Queue(maxsize=1)
    producer = mp.Process(target=_produce_frames, args=(queue, src))
    consumer = mp.Process(target=_start_retargeting, args=(queue, preset, hand_type))

    producer.start()
    consumer.start()

    try:
        # When one side exits, terminate the other so the camera handle is released
        # and we don't leave orphaned holders that block the next run. The
        # signal.pause / time.sleep wakes on SIGINT (Ctrl+C), giving us a clean
        # shutdown path instead of hanging in some uninterruptible C call.
        while producer.is_alive() and consumer.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received — terminating children", flush=True)
    finally:
        for p in (producer, consumer):
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
                if p.is_alive():
                    # SIGKILL the last few stubborn ones (e.g. blocked in a C++ call
                    # holding the GIL) so the camera handle gets released.
                    p.kill()
                    p.join(timeout=1)


if __name__ == "__main__":
    tyro.cli(main)
