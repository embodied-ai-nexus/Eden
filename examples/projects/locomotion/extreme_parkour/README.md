# Extreme Parkour (Unitree Go2)

Port of [*Extreme Parkour with Legged Robots*](https://extreme-parkour.github.io)
(Cheng, Shi, Agarwal, Pathak, 2023) to Eden — both the privileged **teacher** phase and the
vision **student** (depth-distillation) phase.

Two halves, both in the library — so `eden train` trains either phase:

- **The tasks** (`eden/tasks/extreme_parkour/`): `extreme_parkour_teacher` — parkour terrain grid
  (gaps, hurdles, steps, stepping stones) with per-cell waypoint **goals** consumed by a
  `ParkourCommand` term (goal heading + terrain-level curriculum), the 753-d RMA observation
  layout, and the paper's reward set; `extreme_parkour_student` — the same environment plus a
  gs-madrona batched forward **depth camera**, a one-control-step action delay and camera-mount
  randomization.
- **The training stack** (`eden/learning/wrappers/rsl_rl/rma/`, rsl-rl-lib 5): the RMA actor model
  (estimator inside), `RmaPPO`, and the depth-distillation runner. The task configs select them in
  their `runner_options`; this example only adds a viewer playback script.

## Architecture (teacher)

Flat 753-d observation `[prop 53 | scandots 132 | priv_explicit 9 | priv_latent 29 | history 10×53]`:

| Module | Shape | Role |
|---|---|---|
| scan encoder | 132 → [128, 64] → 32 (Tanh) | scandots latent (replaced by a depth latent in the student phase) |
| priv encoder | 29 → [64, 20] | privileged latent (mass/CoM/friction/gains) |
| history encoder | 10×53 → conv1d → 20 | adaptation latent, DAgger-regressed onto the priv latent |
| estimator | 53 → [128, 64] → 9 | predicts base velocity; **overwrites** `priv_explicit` in the actor's input at act time |
| actor | 53+32+9+20 → [512, 256, 128] → 12 | joint-position offsets (scale 0.25) |
| critic | 753 → [512, 256, 128] → 1 | sees ground-truth privileged obs |

Training (`RmaPPO`, driven by rsl_rl's stock `OnPolicyRunner`) = PPO + estimator MSE (own Adam,
1e-4) + privileged regularization (‖priv − hist latent‖, coefficient ramped 0→0.1 from counter step
2000 over 3000) + a DAgger iteration every 20 (rollout with the history latent, then regress the
history encoder) + the mirror loss below. The estimator lives inside the actor model, whose default
forward is the deployed composition (estimator-injected input, history latent) — so
`eden inference` and `eden deploy` run a teacher checkpoint as deployed.

## Architecture (student)

The student runs on `extreme_parkour_student`: the teacher env plus a 106×60 depth camera on the
`base` link (87° horizontal FOV, forward, 20° down, ±5° per-env mount pitch at startup), a
one-control-step action delay, and the source's student terrain mix (per row: 4 gap | 4 hurdle |
1 flat | 4 step | 4 stones | 3 demo, horizontal scale 0.1 m, corridors narrowed to ±0.1 m).
The env exposes two extra observation groups: `depth_camera` — the processed 58×87 frame
(crop → clip [0, 2] m → bicubic resize → `d/2 − 0.5`, refreshed every 5 control steps and
returned with one-interval latency) — and `delta_yaw_ok` (the yaw-substitution gate,
`delta_yaw < 0.6`).

| Module | Shape | Role |
|---|---|---|
| depth CNN | 58×87 → conv32(k5)/pool → conv64(k3) → 128 → 32 | per-frame depth features |
| recurrent encoder | (32 + 53 prop) → 128 → 32 → GRU(512) → 34 (Tanh) | 32-d depth latent (replaces the scandots latent) + 2-d yaw prediction |
| depth actor | copy of the teacher actor | acts on `[prop | depth latent | priv_explicit | hist latent]` |

Distillation (`RmaDepthDistillationRunner`): each iteration collects 24 × 5 control
steps. The frozen teacher (history-latent path, true scandots + true delta-yaw) labels every
step; the depth encoder runs on the 5-step camera cadence with the proprio delta-yaw slots
zeroed; the student sees its own yaw prediction (×1.5) wherever the gate allows and **drives
the environment**. One Adam step (1e-3) per iteration on
`‖a_teacher − a_student‖ + ‖Δyaw_true − Δyaw_pred‖`, gradient clip on the actor, GRU hidden
detached between updates (and carried across env resets — a faithful-port quirk).

## Usage

```bash
# Train the teacher (source setup used 6144 envs; the task default is 4096)
eden train --task extreme_parkour_teacher --logger wandb
eden train --task extreme_parkour_teacher --resume latest

# Distill the depth student from a teacher checkpoint (source setup: 192 envs)
eden train --task extreme_parkour_student --checkpoint logs/extreme_parkour_teacher/<run>/model_49999.pt
eden train --task extreme_parkour_student --resume latest      # a student checkpoint resumes

# Replay
eden inference --task extreme_parkour_teacher --checkpoint logs/extreme_parkour_teacher/<run>/model_49999.pt
eden inference --task extreme_parkour_student --checkpoint logs/extreme_parkour_student/<run>/model_4999.pt

# Play with the viewer and the physics randomization off (the source's play setup)
uv run python examples/projects/locomotion/extreme_parkour/main.py play --model logs/extreme_parkour_teacher/<run>/model_49999.pt
uv run python examples/projects/locomotion/extreme_parkour/main.py play --student --model logs/extreme_parkour_student/<run>/model_4999.pt
```

> The first student run compiles the gs-madrona render kernels for the scene (~10–15 min);
> subsequent runs hit the kernel cache.

> Stock PPO does not learn this task — a plain MLP over the raw 753-d observation (no scan
> encoder, proprio history fed in flat) plateaus at negative reward while the RMA stack climbs —
> which is why the teacher's runner options select the RMA actor and `RmaPPO`.

## Checkpoint format

Teacher `model_<it>.pt` is rsl_rl 5's PPO checkpoint — `actor_state_dict` (the `RmaActorModel`:
scan/priv/history encoders, backbone, **and the estimator**), `critic_state_dict`,
`optimizer_state_dict` — plus `estimator_optimizer_state_dict`,
`hist_encoder_optimizer_state_dict` and `rma_counter` (the regularization ramp), so a resume
continues the DAgger cadence and the ramp where it stopped.

Student checkpoints carry the frozen teacher through (`actor_state_dict`, so one file holds
everything a replay needs) and add `depth_actor_state_dict`, `depth_encoder_state_dict`,
`depth_optimizer_state_dict`. `--checkpoint` accepts either format for training: a teacher
checkpoint starts a fresh distillation (the student actor is re-copied from the teacher), a student
checkpoint resumes one. Replaying the student (`eden inference --task extreme_parkour_student`,
`main.py play --student`) needs a student checkpoint; a teacher checkpoint is refused there, since
its depth encoder is untrained.

The student is an in-sim policy. Like the source, it reads the base linear velocity as the env
provides it (the ground truth) in distillation and in replay, and `eden deploy` refuses its
runner; a hardware port would substitute the teacher's estimator for that chunk, in distillation
too. The teacher has no such gap: its replay and `eden deploy` run the estimator.

Checkpoints from the earlier rsl-rl-lib 3 version of this stack (`model_state_dict` +
`estimator_state_dict`) are refused with an error, by `eden inference`, `eden deploy` and as a
training `--checkpoint` alike; retrain the teacher.

## Attribution / licensing

This is a clean-room reimplementation written against the paper and Eden's installed
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) (BSD-3), whose model/algorithm/runner classes
the stack subclasses. The upstream reference implementations —
[chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour) (CC BY-NC 4.0) and
[CAI23sbP/Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour) (GPL-3.0) — were used
to verify behavior (hyperparameters, update rules, observation layout) only; no code was copied
from either.

## Known deviations from the source setup

- Genesis clamps joint torques to the configured force limits; the source's velocity-dependent
  DC-motor saturation envelope is not reproduced.
- Default 4096 envs / terrain 10×20 cells (source: 6144 envs / 10×40) for the teacher. The
  student phase matches its source exactly (192 envs, 10×20 — the source student also halves
  the teacher's grid).
- The estimator/priv-latent chunks use Eden's DR readback conventions (shift/ratio-nominal
  values rather than absolute mass/friction) — affine re-parameterizations of the same signals.
- Student depth is metric **plane depth** from the gs-madrona batch renderer (rasterizer mode) —
  the original Isaac Gym convention; the IsaacLab port's ray-cast camera returns euclidean ray
  length instead (difference bounded by the 2 m far clip, largest in the image corners).
- The depth resize is `torch.nn.functional.interpolate(mode="bicubic", antialias=False)` — the
  original torchvision transform of the source era; modern torchvision defaults antialias on.
- The student task keeps the teacher's reward set for logging (the source zeroes it to save
  compute); distillation never reads rewards either way.

### Gait / posture tuning beyond the faithful port

The faithful reward set reproduced the source but converged to a Genesis-specific posture/gait
that the source (Isaac Gym dynamics + DC-motor envelope) did not exhibit: a persistent forward
lean, then — once that was penalized — an asymmetric "limping" gait with one front leg parked.
Both were diagnosed as learned local optima (the orientation math and per-slot obs/action layout
were verified correct), not bugs. The following three changes were added on top of the faithful
port to correct them; all are opt-out (set the weight / `symmetry_coef` to `0.0` to recover the
source behavior):

- **Orientation (uprightness) reward is stronger and no longer flat-only.** The source applies
  `_reward_orientation` only on flat cells at weight −1.0; here it is weight −2.0 with
  `nonflat_scale=0.3`, so the ~80% of obstacle terrain (which otherwise had zero pitch pressure)
  keeps a reduced uprightness penalty while still allowing the pitch needed to climb.
- **Two gait-quality rewards not in the paper's teacher set:** `feet_air_time` (legged-gym-style
  air-time credit on first contact, +1.0, target 0.35 s) and `feet_contact_balance` (variance of
  per-foot contact duty, −0.5) — they penalize a permanently planted/airborne leg and front/rear
  imbalance.
- **Mirror-equivariance (symmetry) loss** (`symmetry_coef`, default 1.0), which the source teacher
  does not use. It adds a `π(mirror(obs)) == mirror(π(obs))` MSE term via a sagittal left↔right
  mirror of the full 753-d observation and 12-d action (see `eden/tasks/extreme_parkour/symmetry.py`), forcing
  an L/R-symmetric policy. Measured effect: front-left/right thigh-angle difference dropped from
  ~1.2 rad to ~0.03 rad and all four feet reached ~50% contact duty.
