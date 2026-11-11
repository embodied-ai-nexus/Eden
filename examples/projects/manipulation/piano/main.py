"""Train two Wuji Hand2 hands to play the piano, then watch and hear what they learned.

---
tier: project
capabilities: [task.piano, robot.wuji_hand2beta2, sensor.piano_audio]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/manipulation/piano/main.py train --song twinkle_twinkle
---

The ``piano`` task (``eden/tasks/piano/config.py``) is Eden's take on RoboPianist: a policy sees which keys are due
now and over the next half second, and is paid for having exactly those keys down. ``train`` is the ordinary PPO
run. ``play`` is where the first half of the piano work pays off: the task itself is silent, so ``play`` attaches
the ``PianoAudio`` engine sensor to the keyboard, rolls the policy out, and streams or saves what the strings
sounded like. Nothing about the policy or the task changes; the sensor reads only the key joints.

    # train (about an hour for 1000 iterations of a short tune on an RTX 3080)
    uv run python examples/projects/manipulation/piano/main.py train --song twinkle_twinkle

    # or on real repertoire: any MIDI file, by filename alone once it is in `eden.tasks.piano.songs.SCORES_DIR`
    uv run python examples/projects/manipulation/piano/main.py train --song chpn-p7.mid

    # watch and listen, live
    uv run python examples/projects/manipulation/piano/main.py play --song twinkle_twinkle --vis --play

    # or write the performance to a WAV file, or film it with its own sound as the soundtrack
    uv run python examples/projects/manipulation/piano/main.py play --song twinkle_twinkle --wav /tmp/twinkle.wav
    uv run python examples/projects/manipulation/piano/main.py play --song twinkle_twinkle --record -o /tmp/twinkle.mp4

``play`` takes the newest checkpoint under ``logs/piano`` unless ``--model`` names one, and reports the
performance the way RoboPianist does: precision, recall and F1 over the keys, averaged over the song.
"""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import wave

import eden as en  # isort: skip  (must be imported before genesis and torch: it isolates GPUs per DDP rank)
import genesis as gs
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from rsl_rl.runners import OnPolicyRunner

from eden.engine.sensors import PianoAudio
from eden.engine.sensors.playback import LiveAudioPlayer
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, load_runner_checkpoint
from eden.options.camera import CameraOptions
from eden.options.objects.piano import NUM_KEYS, is_black_key
from eden.options.sensors import SensorOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.piano import custom_terms as ct
from eden.tasks.piano.config import KEYBOARD_Z

LOG_ROOT = "logs/piano"
# 16 kHz at the task's 5 ms simulation step: PianoAudio's default, enough for every fundamental on the keyboard.
AUDIO_SUBSTEPS = 80
# Live playback cannot peak-normalize, so it scales by a fixed gain. A full-scale note peaks near 1.5.
PLAY_GAIN = 0.3


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("WANDB_MODE", "offline")
    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path=LOG_ROOT, performance_mode=True)

    config = TASK_REGISTRY.build("piano", song=args.song)
    config.env_options.num_envs = args.num_envs
    if args.iterations is not None:
        config.runner_options.max_iterations = args.iterations
    if args.cpu:
        config.runner_options.device = "cpu"
    config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config)
    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.learn(num_learning_iterations=config.runner_options.max_iterations, init_at_random_ep_len=True)


def latest_checkpoint() -> str:
    """The newest ``model_*.pt`` under :data:`LOG_ROOT`, across runs.

    Returns
    -------
    str
        Path to the checkpoint.
    """
    checkpoints = glob.glob(os.path.join(LOG_ROOT, "**", "model_*.pt"), recursive=True)
    if not checkpoints:
        raise SystemExit(f"No checkpoint under {LOG_ROOT}/. Train first, or pass --model.")
    return max(checkpoints, key=os.path.getmtime)


def write_wav(path: str, samples: np.ndarray, rate: int) -> None:
    """Write a mono 16-bit WAV, normalized to just under full scale.

    Parameters
    ----------
    path : str
        Output file.
    samples : np.ndarray
        1-D float waveform in arbitrary units.
    rate : int
        Samples per second.
    """
    peak = float(np.abs(samples).max())
    scaled = samples / peak * 0.95 if peak > 0.0 else samples
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((scaled * 32767.0).astype(np.int16).tobytes())


# Score overlay colours: what should be down and is, should be and is not, should not be and is.
_PLAYED, _MISSED, _WRONG = (40, 200, 80), (255, 170, 30), (230, 50, 50)


def draw_score(frame: np.ndarray, due: torch.Tensor, down: torch.Tensor) -> np.ndarray:
    """Draw the keyboard along the bottom of a frame, coloured by what is due and what is down.

    Green is a due key that is down, amber a due key that is not, red a key that is down and should not be. It is
    the step's precision and recall made visible, and it is drawn on the picture because Genesis' offscreen cameras
    do not render debug markers placed in the scene.

    Parameters
    ----------
    frame : np.ndarray
        ``(H, W, 3)`` uint8 image.
    due : torch.Tensor
        ``(88,)`` bool, keys that should be down this step.
    down : torch.Tensor
        ``(88,)`` bool, keys that are down.

    Returns
    -------
    np.ndarray
        The frame with the strip drawn on it.
    """
    image = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    due, down = due.cpu().numpy(), down.cpu().numpy()
    whites = [key for key in range(NUM_KEYS) if not is_black_key(key)]
    width, height = image.size
    key_w = width / len(whites)
    top, black_bottom = height - 70, height - 28

    def colour(key: int, idle: tuple[int, int, int]) -> tuple[int, int, int]:
        if due[key]:
            return _PLAYED if down[key] else _MISSED
        return _WRONG if down[key] else idle

    for i, key in enumerate(whites):
        draw.rectangle(
            (i * key_w, top, (i + 1) * key_w - 1, height - 1), fill=colour(key, (235, 235, 235)), outline=(90, 90, 90)
        )
    for i, key in enumerate(whites[:-1]):
        if is_black_key(key + 1):
            centre = (i + 1) * key_w
            draw.rectangle(
                (centre - 0.3 * key_w, top, centre + 0.3 * key_w, black_bottom), fill=colour(key + 1, (25, 25, 25))
            )
    return np.asarray(image)


def add_soundtrack(video: str, samples: np.ndarray, rate: int, output: str, codec: str) -> None:
    """Mux what the strings played into the film of them being played.

    The track is resampled to 48 kHz stereo whatever the simulation's audio rate: a 16 kHz mono track is legal, and
    enough players refuse it that the video may as well have none.

    Parameters
    ----------
    video : str
        The silent video.
    samples : np.ndarray
        1-D float waveform, as long as the video.
    rate : int
        Samples per second of ``samples``.
    output : str
        The video with sound.
    codec : str
        ``"aac"``, which is what an mp4 normally carries, or ``"mp3"`` for players built without an AAC decoder. VS
        Code's video preview is one: it shows an AAC mp4 in silence.
    """
    if shutil.which("ffmpeg") is None:
        shutil.copyfile(video, output)
        en.logger.warning("ffmpeg is not on PATH, so the video has no sound. Use --wav for the audio.")
        return
    encoder = {"aac": "aac", "mp3": "libmp3lame"}[codec]
    with tempfile.NamedTemporaryFile(suffix=".wav") as track:
        write_wav(track.name, samples, rate)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", track.name]
        command += ["-c:v", "copy", "-c:a", encoder, "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest", output]
        subprocess.run(command, check=True)


def play(args: argparse.Namespace) -> None:
    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path=LOG_ROOT, performance_mode=True)

    config = TASK_REGISTRY.build("piano", song=args.song)
    config.env_options.num_envs = 1
    config.runner_options.logger = "tensorboard"
    if args.cpu:
        config.runner_options.device = "cpu"
    # The only change to the task: give the keyboard its voice.
    config.sensors_options.strings = SensorOptions(
        sensor=PianoAudio(audio_substeps=AUDIO_SUBSTEPS), attach_entity_name="piano"
    )

    if args.record:
        config.cameras_options.rec = CameraOptions(
            cam_pos=(0.62, -0.42, KEYBOARD_Z + 0.42),
            cam_lookat=(0.03, -0.06, KEYBOARD_Z + 0.02),
            cam_fov=38.0,
            cam_resolution=(1280, 720),
        )

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=args.vis, eval_mode=True)
    if args.vis:
        env.unwrapped.scene.viewer.set_camera_pose(
            pos=(0.9, -0.55, KEYBOARD_Z + 0.65), lookat=(0.05, -0.08, KEYBOARD_Z)
        )

    model = args.model or latest_checkpoint()
    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    load_runner_checkpoint(runner, model)
    policy = runner.get_inference_policy(device=env.device)
    en.logger.info(f"🎹 playing '{args.song}' with {model}")

    base = env.unwrapped
    strings = base.sensors["strings"].sensor
    song = base.command_manager.get_term("song")
    rate = round(AUDIO_SUBSTEPS / base.env_options.sim_dt)
    n_steps = 3 if en.is_smoke() else song.n_steps - 1
    # A control step is ten simulation steps and `read()` would return only the last one's block, so record.
    strings.start_recording()
    # A chunk is a whole control step (50 ms), so the cushion has to be deeper than the 60 ms default.
    speaker = LiveAudioPlayer(rate, gain=PLAY_GAIN, lead=0.15) if args.play and not en.is_smoke() else None

    frames = []

    heard, scores = [], {"precision": [], "recall": [], "f1": []}
    obs = env.get_observations()
    for _ in range(n_steps):
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))
        chunk = strings.take_recording()[0].cpu().numpy()
        heard.append(chunk)
        if speaker is not None:
            speaker.write(chunk)
        if args.record:
            frame = base.cameras["rec"].render_rgb()
            frame = frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
            frames.append(draw_score(frame.reshape(-1, *frame.shape[-2:]), song.goal[0], song.key_down[0]))
        scores["precision"].append(float(ct.key_press_precision(base)[0]))
        scores["recall"].append(float(ct.key_press_recall(base)[0]))
        scores["f1"].append(float(ct.key_press_f1(base)[0]))
    if speaker is not None:
        speaker.close()

    summary = ", ".join(f"{name} {np.mean(values):.3f}" for name, values in scores.items())
    en.logger.info(f"🎼 over {n_steps} control steps: {summary}")
    samples = np.concatenate(heard)
    if args.wav:
        write_wav(args.wav, samples, rate)
        en.logger.info(f"💾 wrote {samples.size / rate:.2f}s of audio to {args.wav}")
    if args.record:
        with tempfile.TemporaryDirectory() as scratch:
            silent_video = os.path.join(scratch, "silent.mp4")
            # One frame per control step, so the film runs at the control rate and stays in step with the sound.
            imageio.mimwrite(silent_video, frames, fps=round(1.0 / base.dt), quality=8)
            add_soundtrack(silent_video, samples, rate, args.output, args.audio_codec)
        en.logger.info(f"🎬 wrote {args.output}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--song", default="twinkle_twinkle", help="a built-in song, or a .mid file (see eden/tasks/piano/songs.py)"
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="use the CPU backend")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a piano-playing policy, or watch and hear one.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the piano task with PPO.")
    _add_common(train_parser)
    train_parser.add_argument("-b", "--num-envs", type=int, default=1024, help="number of environments")
    train_parser.add_argument("--iterations", type=int, default=None, help="PPO iterations (default: the task's)")

    play_parser = subparsers.add_parser("play", help="Roll out a checkpoint with the keyboard's voice attached.")
    _add_common(play_parser)
    play_parser.add_argument("--model", type=str, default=None, help="checkpoint .pt (default: the newest)")
    play_parser.add_argument("-v", "--vis", action="store_true", help="open the interactive viewer")
    play_parser.add_argument("--play", action="store_true", help="play the performance live, in real time")
    play_parser.add_argument("--wav", type=str, default=None, help="write the performance to this WAV file")
    play_parser.add_argument("-r", "--record", action="store_true", help="film the performance, with its sound")
    play_parser.add_argument("-o", "--output", type=str, default="piano.mp4", help="where --record writes the video")
    play_parser.add_argument(
        "--audio-codec",
        choices=("aac", "mp3"),
        default="aac",
        help="soundtrack codec for --record; use mp3 if your player shows the video in silence (VS Code does)",
    )

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "play":
        play(args)
    else:
        parser.print_help()
