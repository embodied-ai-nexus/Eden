# SuperMimic

[SuperMimic](https://supermimic.github.io) trains one Unitree G1 whole-body motion tracker through a learned,
differentiable world model instead of model-free RL. This is the Eden port of the official implementation
([hybridrobotics/supermimic](https://github.com/hybridrobotics/supermimic)):

> Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath.
> SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model. CoRL 2026.
> [arXiv 2610.28480](https://arxiv.org/abs/2610.28480)

The task is `supermimic` (`eden/tasks/supermimic`) and the learner is `SuperMimicRunnerOptions`
(`eden/learning/wrappers/rsl_rl/supermimic`). The learner's defaults are SuperMimic's method with the learner-side
recipe of [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (F. Chen, 2026; `PRESET[main]` in its
`scripts/presets.sh`), which trains the same tracker in fewer updates. [Options](#options) lists what the recipe
changes and the override that restores the paper's values, and [Measured](#measured) compares the two.

## Method

Each runner iteration steps 4096 simulated envs 10 times with the deterministic policy and stores the transitions
in a replay buffer of 1000 steps per env. On average 0.4 times per env step:

1. The world model fits 5 batches of 2048 replay windows of 8 steps, half of each batch replaced by its left-right
   mirror image. It predicts the change of the root twist and the joint velocities from the state, the commanded
   joint targets and the env's randomized parameters (foot friction, torso centre of mass, encoder bias). Poses
   integrate analytically.
2. The policy improves on 2048 rollouts of 32 steps that the frozen world model imagines from replay states. The
   loss is the negative of the task's tracking reward, computed analytically on the predicted state, discounted
   and weighted by the probability of not having fallen. A critic bootstraps the return beyond the horizon. The
   actor's output is added to the action that commands the next reference pose, so the network learns a correction
   to it.

Resets concentrate on the 0.2 s bins of the motion library where the policy earns a low return, and 30% of them
start up to 1 s before the frames of recent falls. Replay and evaluation run a Polyak average of the actor.

## Train

LAFAN1 (40 clips, 2.45 h) downloads on first use:

```bash
eden train --task supermimic --processed_data_path /data/lafan1_g1.pt
```

BONES-SEED needs the G1 split of [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) (gated).
Write the clip list of the paper's filter once (129,514 clips, 262.3 h), then train on it:

```bash
python examples/projects/mimic/supermimic/bones_seed_clip_list.py --raw-data-path /data/bones_seed/g1/csv \
    -o bones_seed_clips.json
eden train --task supermimic --dataset bones_seed --raw_data_path /data/bones_seed/g1/csv \
    --processed_data_path /data/bones_seed_g1.pt --clip_list bones_seed_clips.json
```

The first run prepares the library into `--processed_data_path` (13.5 GB for BONES-SEED, root and joints only;
link states are computed by forward kinematics when an env reads them); later runs load it. Training takes about
13 GB of GPU memory on top of the library. One iteration takes about 0.6 s on an L40S (0.24 s collection, 0.37 s
learning), which is 147 ms per update round. A run starts in about 1 minute when `torch.compile`'s caches are warm
and in about 5 minutes when they are empty. The caches live under `/tmp` unless `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` name another directory, so on a cluster point both at a shared filesystem.

## Measured

LAFAN1 with 4096 envs on one L40S and one seed. An update round takes 147 ms with the defaults and 300 ms with
the paper's values ([Options](#options)). `evaluate.py` on checkpoints of both (40 clips, randomization on):

| Options | Iterations | Training time | Failed clips | MPJPE |
|---|---|---|---|---|
| paper values | 500 | 10 min | 8 | 44.7 mm |
| paper values | 1000 | 20 min | 5 | 28.5 mm |
| defaults | 300 | 3 min | 8 | 34.0 mm |
| defaults | 500 | 5 min | 8 | 29.2 mm |
| defaults | 1000 | 10 min | 3 | 24.8 mm |

On BONES-SEED (129,513 clips), same GPU and env count, one seed, on 2,000 clips drawn from the training list:

| Options | Iterations | Training time | Failed clips | MPJPE |
|---|---|---|---|---|
| paper values | 2000 | 49 min | 5 | 24.2 mm |
| defaults | 500 | 5 min | 7 | 26.0 mm |
| defaults | 1000 | 10 min | 4 | 24.4 mm |
| defaults | 2000 | 20 min | 2 | 22.9 mm |

Training time is that many iterations at 0.6 s for the defaults and 1.2 s for the paper's values, without
startup. The BONES-SEED checkpoint of the paper's values was trained before the speed changes, at 1.48 s per
iteration. The failed-clip count moves by a few clips from one checkpoint to the next and, on BONES-SEED, by about
two between evaluations of one checkpoint. Time per round varies by a few percent between nodes of the same GPU
model.

## Evaluate

Play every clip once from its first frame, without reset noise, and count the falls:

```bash
python examples/projects/mimic/supermimic/evaluate.py --checkpoint logs/supermimic/<run>/model_20000.pt
```

Replay (`eden inference` and this script) drives every environment through the motion encoder, whatever `slot_mix`
the checkpoint trained on, and reads the proprioception without the training sensor noise. Both rebuild the learner
from the run's `config.json`, so a checkpoint trained with `--override` replays as it trained, and both take
`--override` themselves. The script's library flags (`--dataset`, `--raw-data-path`, `--processed-data-path`,
`--clip-list`) replace the run's library as a set; pass none to play the clips the run trained on.

A run trained before the recipe became the default has a `config.json` without the recipe's options, which then take
today's defaults. Loading its checkpoint stops with the options that differ and the `--override` that sets them.

## Options

`--override` reaches every learner option (`eden/learning/wrappers/rsl_rl/supermimic/options.py`). These defaults
are BlinkMimic's recipe and differ from the SuperMimic paper's values:

| Option (under `runner_options`) | Default | Paper |
|---|---|---|
| `world_model.hidden_dims` | 3 × 1024 | 3 × 2048 |
| `world_model.num_batches` | 5 | 10 |
| `world_model.lr`, `policy.lr` | 4e-4 | 2e-4 |
| `world_model.mirror_share` (share of each batch mirrored left-right) | 0.5 | 0 |
| `replay_ratio` (0.4 update rounds per env step with either pair) | 8 | 16 |
| `policy.batch_size` | 2048 | 4096 |
| `policy.lr_multiplier` (the actor's rate over `policy.lr`) | 4 | 1 |
| `policy.lr_decay_return`, `lr_decay_deadline`, `lr_decay_iterations` | 0.75, 350, 75 | not used |
| `policy.ema_decay` (Polyak average of the actor, which replay runs) | 0.995 | 0 |
| `policy.reference_action` (the actor corrects the next reference pose) | `true` | `false` |
| `return_sampler.failure` (share of resets before recent falls) | 0.3 | 0 |
| `return_sampler.explore`, `keep` | 0.35, 0.07 | 0.5, 0.1 |

The multiplier ramps to 1 over `lr_decay_iterations` iterations once `Train/mean_normalized_return` reaches
`lr_decay_return`, or at iteration `lr_decay_deadline`. That return is the mean, over the episodes that ended in an
iteration, of the episode's reward per step over the largest reward of one step (BlinkMimic's `ep_ret_norm`).
BlinkMimic counts update rounds, 4 per iteration here, so its 3500-iteration run is 875 iterations. Its contact
reward, its evaluation-guided resets and its world model's position correction are not ported.

To train with the paper's values:

```bash
eden train --task supermimic --override "runner_options: {replay_ratio: 16, world_model: {hidden_dims: [2048, 2048, 2048], num_batches: 10, lr: 0.0002, mirror_share: 0}, policy: {lr: 0.0002, lr_multiplier: 1, batch_size: 4096, ema_decay: 0, reference_action: false}, return_sampler: {failure: 0, explore: 0.5, keep: 0.1}}"
```

The options the official implementation exposes as variants:

| Option | Default | Variant |
|---|---|---|
| `runner_options.policy.algorithm` | `actor_critic` | `bptt` (no critic) |
| `runner_options.policy.observation` | `state` (the model state) | `proprio` (deployable proprioception with sensor noise) |
| `runner_options.policy.history_length` | 1 | more frames of proprioception |
| `runner_options.policy.slot_mix` | `{motion: 1.0, velocity: 0.0}` | a share for the velocity-command encoder |
| `runner_options.return_sampler.enabled` | `true` | `false`: uniform resets |

The official maximal-coordinate world-model state and the ONNX export are not ported: `eden export` and `eden deploy`
exit for this task before building anything. Replay a checkpoint with `eden inference --task supermimic`.
