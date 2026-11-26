---
name: eden-iterate
description: Staged, measured iteration on code that makes a simulated (or real) robot act in Eden. Split the task into stages, run one script per stage with a timeout, record video or a Viser replay, review frames with an unbiased subagent when stuck, and declare success only from state measured after the scene settles. Use when writing or debugging a scripted controller, a demonstrator, a reward or a task that moves a robot. Don't use for non-robotics code or pure config edits.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent
---

<!--
Adapted from google-deepmind/robotics-antigravity-plugin, plugins/robotics/skills/iteration/SKILL.md
(Copyright 2026 Google LLC, Apache License 2.0). Modified for Eden: Eden's recording APIs, frame extraction for
agents that read images rather than video, and Eden's frame and reset pitfalls were added.
-->

# Iterating on robot code in Eden

## Work in stages

- Split a long task into stages, one per sub-goal: putting five objects in a box is five stages. A behavior that
  already works reliably, such as a scripted pick-and-place, counts as one.
- Write each stage as one script under the user's workspace (or `examples/` when it is meant to ship) that builds
  the scene, acts, measures the result and prints it. Run it with a timeout:
  `timeout 300 uv run python stage_1.py`. Never put multi-line code in `python -c`.
- For a registered task, stage 0 is `eden smoke --task <name>`: it runs every term's `build()` and catches a bad
  name in seconds.
- After each stage, decide from the measured result whether to move on. On a failure, find in the trajectory where
  it went wrong, fix the code, re-measure the scene, and go back to the first stage whose goal no longer holds.

## Record what happened

- Video: `camera.start_recording(path, fps=1 / env.dt)`, one `camera.render_rgb()` per control step,
  `camera.stop_recording()` (`docs/guides/sensors-and-rendering.md`, "Recording video"). Never use Genesis'
  `gs.Camera.start_recording`. `eden inference --rec` records a policy rollout without code.
- A 3D replay anyone can scrub in a browser (the `vis` extra):
  `from eden.extensions.visualization.viser import ViserViewer`, `ViserViewer(env, enable_gui=False)`,
  `viewer.start_recording("run.html")`, `viewer.update()` per step, `viewer.stop_recording()`.
- To inspect a video, extract frames, for example `ffmpeg -i run.mp4 -vf fps=4 frames/%03d.png`, and read a handful
  of them. An `.mp4` cannot be read directly.

## Get a second opinion when stuck

- When a stage fails twice, give a subagent the frames of the last attempt and only the frames: no code, no plan,
  no expectation. Ask it to describe what happens and what goes wrong. Its description is then not anchored on
  yours.
- If the stage still fails, give a subagent the frames, the code and the measurements, and ask what to change.

## Done means measured

- Decide success from state read after the robot has let go and the scene has settled for a second of sim time.
  A commanded pose and a printed "SUCCESS" are not evidence. Check every goal condition before you finish.
- Read object poses with `get_pos(relative=True)` / `get_quat(relative=True)`. The default reads the solver's
  center-of-mass frame, which sits 2.9 to 14.8 cm from the authored origin on the DexToolBench tools.
  Quaternions are wxyz.
- In an `RLEnvBase`, the observation `step()` returns for a finished env is already the reset state. Measure the
  episode's outcome from the reward or metric terms, or before the step that ends it.
- Contact-rich rollouts can differ between runs at a fixed seed on the GPU. Judge a behavior over several seeds or
  envs, not one rollout.
