# ConTrack — constrained hand-object motion tracking

Port of [ConTrack](https://github.com/EmptyBlueBox/ConTrack) (arXiv:2606.03177):
a fixed-base xArm7 + XHand tracks a reference clip of robot joints, manipulated
object poses, and per-finger-link contacts, trained with a constrained-MDP PPO
whose task / style / penalty reward channels (`rg` / `rs` / `rp`) are balanced by
a dual controller.

## Install the training backend (separate from Eden)

The CMDP-PPO algorithm lives in a standalone fork of `rsl_rl`, not in Eden. Eden
ships the integration (`eden/learning/wrappers/rsl_rl_contrack/`: the runner
options the task selects, the VecEnv wrapper, and the runner `eden train` builds).
Install the backend before training:

```bash
pip install "git+https://github.com/Kashu7100/rsl_rl_contrack@7b6c043f3ceea45e8afc8175ba74ba3876510e25"
```

It is a small, dependency-light fork of `rsl_rl` (`torch` + `tensordict`); it does
**not** pull in Isaac Lab. `eden train --task contrack` raises with this command if
the package is missing. Everything else — building the env, playing back a clip,
the task tests — works without it.

## Robot assets

Nothing to fetch. The rig is the `XArm7XHand1_L` / `_R` **composite**
(`eden/options/robots/composite/xarm_xhand.py`), which welds Eden's own xArm7 and XHand1
unit robots — both already on the Hugging Face asset dataset — rather than loading
ConTrack's pre-merged URDF. `tests/options/robots/composite/test_xarm_xhand.py` pins the mount transform
against ConTrack's combined file: the arm chain is bit-identical, and the hand chain is the
same geometry (292 nm agreement when mounted at ConTrack's own weld rotation). See the
composite's module docstring for the known deltas — ConTrack's `rpy="3.14 0 ±1.57"` truncates
π/π-2, and the rig mounts at the exact rotation instead, leaving a deliberate 0.37 mm
fingertip offset; Eden's XHand1 is also a later asset revision (per-link masses differ by
≤ 8 g, `thumb_rota_joint1/2` cap at 1.570 rad instead of 1.745, and every hand link *frame*
is rotated 180°, so hand link quaternions are not frame-comparable with the source).

The composite expands into two entities — `robot` (the 7-DOF arm) and `hand` (12 finger
DOFs) — so each DOF-addressed term comes in an arm/hand pair, and the clip's flat 19-column
`qpos` is split at the arm/hand boundary by `DexterHandDataset`.

## Train

One clip per run (per-clip specialists, matching the source):

```bash
eden train --task contrack \
  --data examples/projects/mimic/contrack/sample/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5 \
  --num_envs 8000 --logger wandb
```

The task's `runner_options` (`RslRlContrackRunnerOptions`) name
`ContrackOnPolicyRunner`, which wraps the env so the fork's PPO receives the
`rg` / `rs` / `rp` reward-group extras. `--data` selects the reference clip; handedness (`is_rhand`) picks the L/R robot,
the object mesh is exported from the clip, and the episode length matches the clip.
The same four sample clips are in this checkout under
`examples/projects/mimic/contrack/sample/` and on the Hub under `contrack/` in
[`Kashu7100/eden_example`](https://huggingface.co/datasets/Kashu7100/eden_example) —
omit `--data` and the task fetches `dexterhand-Cylinder_00-…` from there on first use, so the
default resolves the same way on a pip-installed Eden as in a checkout. Pass a
distinct `--run_name` per clip when training several in parallel — the log dir is
timestamped to the second, so same-second launches on one node would otherwise
share a dir and clobber each other's checkpoints.

`eden smoke --task contrack` builds and steps the env (no backend needed).
Defaults match the ConTrack release (`num_steps_per_env=16`, 5000 iterations,
`cmdp_alpha=0.9`). Resume with `eden train --task contrack --resume latest`.

## Render a trained checkpoint

`eden inference --task contrack --checkpoint logs/contrack/<run>/model_<it>.pt`
replays a checkpoint through the same runner. `render.py` adds start-phase
selection and records a single-env rollout to mp4:

```bash
python examples/projects/mimic/contrack/render.py \
  --data examples/projects/mimic/contrack/sample/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5 \
  --model logs/contrack/<run>/model_<it>.pt --auto-start
```

`--auto-start` probes several start phases and renders from the one with the
longest termination-free run (the policy often tracks a mid-clip phase cleanly
even when a kinematic reset from frame 0 can't establish the grasp — see the
cached-state reset follow-up). Use `--start-frame N` to force a phase, or the
default (frame 0).
