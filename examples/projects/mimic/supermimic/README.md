# SuperMimic

[SuperMimic](https://supermimic.github.io) trains one Unitree G1 whole-body motion tracker through a learned,
differentiable world model instead of model-free RL. This is the Eden port of the official implementation
([hybridrobotics/supermimic](https://github.com/hybridrobotics/supermimic)):

> Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath.
> SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model. CoRL 2026.
> [arXiv 2610.28480](https://arxiv.org/abs/2610.28480)

The task is `supermimic` (`eden/tasks/supermimic`) and the learner is `SuperMimicRunnerOptions`
(`eden/learning/wrappers/rsl_rl/supermimic`).

## Method

Each runner iteration steps 4096 simulated envs 10 times with the deterministic policy and stores the transitions
in a replay buffer of 1000 steps per env. On average 0.4 times per env step:

1. The world model fits 10 batches of 2048 replay windows of 8 steps. It predicts the change of the root twist and
   the joint velocities from the state, the commanded joint targets and the env's randomized parameters (foot
   friction, torso centre of mass, encoder bias). Poses integrate analytically.
2. The policy improves on 4096 rollouts of 32 steps that the frozen world model imagines from replay states. The
   loss is the negative of the task's tracking reward, computed analytically on the predicted state, discounted
   and weighted by the probability of not having fallen. A critic bootstraps the return beyond the horizon.

Resets concentrate on the 0.2 s bins of the motion library where the policy earns a low return.

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
13 GB of GPU memory on top of the library. One iteration takes about 2 s on an A100.

## Evaluate

Play every clip once from its first frame, without reset noise, and count the falls:

```bash
python examples/projects/mimic/supermimic/evaluate.py --checkpoint logs/supermimic/<run>/model_20000.pt \
    --processed-data-path /data/lafan1_g1.pt
```

## Options

`--override` reaches every learner option (`eden/learning/wrappers/rsl_rl/supermimic/options.py`). The ones the
official implementation exposes as variants:

| Option | Default | Variant |
|---|---|---|
| `runner_options.policy.algorithm` | `actor_critic` | `bptt` (no critic) |
| `runner_options.policy.observation` | `state` (the model state) | `proprio` (deployable proprioception with sensor noise) |
| `runner_options.policy.history_length` | 1 | more frames of proprioception |
| `runner_options.policy.slot_mix` | `{motion: 1.0, velocity: 0.0}` | a share for the velocity-command encoder |
| `runner_options.return_sampler.enabled` | `true` | `false`: uniform resets |

The official maximal-coordinate world-model state and the ONNX export are not ported.
