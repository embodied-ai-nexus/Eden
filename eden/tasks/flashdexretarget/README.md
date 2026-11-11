# flashdexretarget — FlashDexRetarget on Genesis

Port of the RL half of [FlashDexRetarget](https://github.com/Holiday-Robot/FlashDexRetarget) (Lee et al., arXiv
2610.01849; Isaac Sim): one policy, trained across every clip of a retargeted dataset, drives one or two robot hands so
the objects follow their recorded trajectories while the hands follow the human (MANO) wrists and fingertips. The
clips the policy completes are physically valid robot hand-object trajectories — the retargeting's real output.

```bash
# 1. a dataset: a released hand-object dataset -> <dataset>/<robot>/motion.pt (+ object URDFs and SDF grids),
#    Isaac-free; each robot retargeted from one import keeps its own
eden retarget flashdex import --source taco --raw /data/taco --mano-dir /data/mano_v1_2/models --dataset /data/taco_fdr
eden retarget flashdex kinematic --dataset /data/taco_fdr --robot xhand
# 2. train; the robot is the one motion.pt was retargeted to
eden train --task flashdexretarget --motion /data/taco_fdr/xhand/motion.pt                     # PPO
eden train --task flashdexretarget --motion /data/taco_fdr/xhand/motion.pt --learner flashsac  # FlashSAC
eden smoke --task flashdexretarget --motion /data/taco_fdr/xhand/motion.pt --num_envs 4
# 3. roll the policy out (either learner), the assist off; prints the per-episode metrics below
eden inference --task flashdexretarget --checkpoint <run>/model_<iter>.pt  # replays the run's saved config
# on held-out clips: --motion is a task flag, which a run's saved config refuses, so roll out a copy of the checkpoint
cp <run>/model_<iter>.pt /tmp/policy.pt
eden inference --task flashdexretarget --motion /data/held_out/xhand/motion.pt --learner <learner> \
  --checkpoint /tmp/policy.pt --num_envs 512 --episodes 2048
```

`--episodes 2048` counts each env's first 4 episodes (`ceil(N / num_envs)`). Each object holds an equal share of the
envs, so the rates weight every object equally, and every clip of an object equally.

## Pieces

| Piece | Where |
|---|---|
| Robot: the fixed-base hand(s) — 6 actuated forearm joints + fingers each — from FlashDexRetarget's `*_spec` URDF, with the MJCF's drive gains, armature, damping and ranges | `assets.py` (`robot_spec`); assets on `Kashu7100/eden_assets` under `hands/flashdexretarget/` |
| Objects: one URDF per object (convex parts collide; density 800, mass clamped to [0.05, 0.5] kg); one entity per held object, one variant per object slot, each env playing only its object's clips | `eden retarget flashdex kinematic` (`urdf` step), `config.py` (`GroupedEntityOptions` + `env_motion_groups`) |
| Dataset: `motion.pt` -> `robot` (joints), `mano_<side>` (wrist + 20 joints), `object_<side>` sequences | `dataset.py` (`FlashDexRetargetDataset`) |
| Residual action: forearm targets = reference + `a` x (0.1 slides, 1.0 hinges), fingers = `a` mapped onto their range; 5 Hz low-pass on the forearm | `custom_terms/actions.py` (`FDRHandAction`) |
| Object assist: a PD wrench toward the reference object pose, every physics step; decayed by the adaptive curriculum once the task, imitation and completion gates hold | `custom_terms/actions.py` (`FDRObjectAssist`), `custom_terms/curricula.py` (`FDRAssistCurriculum`) |
| Metrics, per episode: `completed` (a time-out ended it — the clip ran out or the episode time limit — rather than a failure term: `Metrics/Success`), mean object position / rotation and fingertip errors. An eval build (`eden inference`) runs with the assist at zero and starts every episode at its clip's first frame (training: a uniform frame in [0, 0.99 T)) | `custom_terms/metrics.py` (`FDREpisodeCompleted`, `FDRObjectPosError`, `FDRObjectRotError`, `FDRTipPosError`), `custom_terms/curricula.py` (`FDRAssistCurriculum`), `custom_terms/motions.py` (`FDRMotionTerm`) |
| Learners: rsl_rl PPO, or FlashSAC (`--learner flashsac`, `uv sync --inexact --extra flashsac --group flashsac-git`) with FlashDexRetarget's hyperparameters | `config.py` (`flashsac_runner_options`), `eden/learning/wrappers/rsl_rl/flashsac/` |
| Observations (palm frame), rewards (object point-cloud match, fingertip-surface bound, wrist/fingertip tracking, joint BC, action penalty, termination penalty), terminations (clip end, object drop, cumulative task reward, fingertip divergence, blow-ups) | `custom_terms/` (`observations.py`, `rewards.py`, `terminations.py`; the hand / object state they read: `common.py`), `config.py` |

Palm and fingertips are FlashDexRetarget's MJCF sites (`<side>_palm`, `track_hand_<side>_<finger>_tip`), placed on the
URDF links: the URDF's own fingertip frames sit up to 10 mm off them.

## Differences from FlashDexRetarget

- **Learner**: PPO (rsl_rl) by default. `--learner flashsac` trains FlashSAC with FlashDexRetarget's hyperparameters
  (gamma 0.98, 5-step returns, 2 updates of 4096 per env step, a 1024-wide distributional critic, a float16 50M-row
  replay buffer), but one actor and one critic on the summed reward: FlashDexRetarget's per-hand actor-critic heads,
  its future-trajectory encoder and its compact replay rows are not ported. A row here is the whole observation (1544
  float16 values with one XHand, 3.1 KB), so 50M rows need about 154 GB of GPU memory, where FlashDexRetarget's
  compact rows need 88 GB: set `runner_options.algorithm.replay_buffer_size` for the GPU (10M rows, 31 GB, fit a
  48 GB L40S). The run is 500M env steps whatever
  `--num_envs` is (its iterations are sized when the env is built).
- **Assist dial-back**: a completion rate under 0.5 for over 500 epochs returns the assist to 0.98 x its gains before the
  last decay, also once the assist has decayed to zero, and holds it there while the rate stays low. FlashDexRetarget
  checks this only once its decay gates pass, which need a completion rate of at least 0.65, so its dial-back never
  fires; here it does.
- **Contacts**: Genesis, not PhysX TGS. Friction 2.0 on every body (Genesis combines a pair as `max`; PhysX averaged two
  2.0s, so the same), no contact offset. The object reset caps the velocity Genesis adds to separate overlapping
  bodies at 1 m/s, as PhysX's `max_depenetration_velocity` did.
- **Action timing**: the residual is built on the reference of the step it acts in (Eden advances the motion before
  the action); FlashDexRetarget's on the previous one.
- **Fingertip distance-bound reward**: the MANO fingertip's distance to the object is the SDF's (surface distance);
  FlashDexRetarget precomputes the distance to the nearest mesh vertex.
- **Object variants**: each gets an equal block of envs; FlashDexRetarget sizes the blocks by clip count.
- **Both hands on one object and one object per hand cannot mix** in one motion file (FlashDexRetarget parks a ghost
  object for the shared clips); split such a dataset by kind.
- Robot gravity stays on and uncompensated, as in FlashDexRetarget's Isaac setup: the forearm sags a few cm on its
  PD and the wrist residual corrects it.
