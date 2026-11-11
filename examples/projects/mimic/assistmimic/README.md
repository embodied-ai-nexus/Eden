# AssistMimic in Eden

[AssistMimic](https://github.com/YutoShibata07/AssistMimic) (CVPR 2026) on Genesis: a caregiver humanoid helps a
weakened recipient up while both track a paired interaction clip. Both humanoids live in one env, each with its own
observation group, reward group and action term. It is a simplified port — PHC's SMPL humanoid (69 DOFs, no
fingers) instead of SMPL-X, wrist contacts instead of finger contacts, no AMP reward; see *Port notes*.

| `--dataset` | Eden task | AssistMimic config | Scene | Recipient |
|---|---|---|---|---|
| `inter-x` (default) | `assistmimic` | `env_im_interx_helpup.yaml` | floor | lower body at half gain, 80 N m |
| `hhi-assist` | `assistmimic_bed` | `env_im_hhi-assist.yaml` | floor + bed | also hips (20 N m) and trunk (40 N m); caregiver paid for unloading them |

The tasks live in `eden/tasks/assistmimic/`, the humanoid in `eden/options/robots/smpl_humanoid.py`. `main.py` trains
them with rl_games' PPO through Eden's rl_games wrapper, the caregiver and recipient being two agents of each env
(`RlGamesPpoRunnerOptions.agents`); `assist_training/` adds the PHC-initialized network (one actor per role, picked
by the `role` observation) and an agent with its own critic learning rate. PSI — restarting half the resets from
states of good past rollouts — is part of the env's `motion` command term (`--no-psi` turns it off). The configs
declare no `runner_options`, so `eden train --task assistmimic` refuses them. Each `--dataset` logs to its own
`logs/assistmimic_<dataset>/<stage>/`.

## Train

```bash
# PHC's phc_3 comes from Kashu7100/eden_policies on first use (--phc / $PHC_CHECKPOINT for another)
D=inter-x; L=logs/assistmimic_${D/-/_}                    # or D=hhi-assist
uv run python main.py adapt --dataset $D -b 1024 -s 700   # stage 1: phc_3 -> Genesis, humanoids apart
uv run python main.py train --dataset $D \                # stage 2: AssistMimic, 8192 envs, 2000 epochs
    --phc-caregiver $L/adapt/<run>/phc_genesis_caregiver.pth \
    --phc-recipient $L/adapt/<run>/phc_genesis_recipient.pth
```

Checkpoints are `nn/last_assistmimic_ep_<epoch>_rew_<r>.pth`, saved every `--save-every` epochs (default 500, ~344 MB
each); only the newest stays on disk unless `--keep-last N` (0 keeps every one). `--resume <checkpoint> -s 2000`
continues a run to epoch 2000; PSI's stored trajectories are not in the checkpoint, so its buffer refills.

The task steps with Genesis' CG constraint solver (50 iterations), which halves the physics step against Newton
(`env_options.solver="newton"`): 2000 Inter-X epochs take about 11.5 h on one L40S, against 21.3 h. Help-up success
over the 6 help-up clips varies from run to run — GPU simulation is not bit-reproducible, so even the same seed
diverges. Code that steps as this does trained to 1.000, 0.998, 0.833, 0.822 and 0.681 with CG (five runs, three
seeds) and to 0.698 with Newton (one run); each run below 1 misses one or two of the hardest clips (G029 R000, R001,
R003).

## Evaluate

```bash
C=$L/train/<run>/nn/last_assistmimic_ep_2000_rew_<r>.pth
uv run python main.py play --dataset $D --checkpoint $C                  # per-clip length / completion / help-up success
uv run python main.py play --dataset $D --checkpoint $C --clip 0 -r      # record one clip to $L/play/<run>/
uv run python main.py play --dataset $D --checkpoint $C --no-caregiver   # ablation: the recipient alone
```

## Data

The MJCFs come from `Kashu7100/eden_assets` (`humanoids/smpl_humanoid/`): `smpl_humanoid_phc.xml` for the caregiver
and `smpl_humanoid_phc_70pct_mass.xml` (every geom density x0.7) for the recipient. The reference clips come from
`Kashu7100/eden_example` (`assistmimic/`) on first use; `--motion-file <npz>` plays a locally converted one. The
scripts run in a PHC / AssistMimic environment (`smpl_sim`, `poselib`, SMPL models), not in Eden's:
`generate_smpl_xml.py` regenerates the two MJCFs, and `convert_pair_motion.py` converts AssistMimic's pkls —
`--dataset` picks the reference height of the task that reads it (see *Port notes*):

```bash
cd <AssistMimic> && CONV=<Eden>/examples/projects/mimic/assistmimic
python $CONV/scripts/convert_pair_motion.py --dataset inter-x --smpl-dir data/smpl \
    --pkl sample_data/interx_processed_fixed_v9_cluster_ids_1_n_clusters_10.pkl -o $CONV/data/interx_cluster1.npz
python $CONV/scripts/convert_pair_motion.py --dataset hhi-assist --smpl-dir data/smpl \
    --pkl sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist.pkl -o $CONV/data/hhi_assist_aa_rm.npz
```

<details>
<summary>Port notes — read before changing anything</summary>

| AssistMimic (Isaac Gym) | Here (Genesis) | Why |
|---|---|---|
| SMPL-X humanoid, 153 DOFs, fingers | PHC's SMPL humanoid, 69 DOFs | Genesis' cost grows with DOFs |
| PHC `phc_interx_recipient_finetune2` (SMPL-X) | PHC `phc_3` (SMPL), adapted to Genesis (`main.py adapt`) | only an SMPL checkpoint fits the body |
| finger contact / force observations and rewards | the same on the wrists (wrist + hand link) | no fingers |
| one humanoid per env, envs paired | both humanoids in one env | Genesis collides entities of one env |
| AMP discriminator reward (0.5) | not used | task reward only |
| future-trajectory features | not built | AssistMimic's network drops them |

The PHC weights work only if what the policy sees and acts through matches Isaac Gym; against an Isaac Gym rollout
of `phc_3`, the observations match to 1e-6 and the actions to 1e-7. What it takes:

* **Velocities**: Isaac Gym's rigid-body velocity is the center-of-mass velocity (`get_links_vel(ref="link_com")`).
* **Joints**: Isaac Gym makes each SMPL body one exp-map spherical joint; Genesis keeps three hinges (intrinsic
  X-Y-Z Euler). References are written in Euler angles, PD targets go rotvec -> quaternion -> Euler
  (`custom_terms/actions.py`).
* **Motion data**: `phc_3`'s SMPL robot is upright-start and AssistMimic's SMPL-X is not: global rotations are
  multiplied by `R([0.5, 0.5, 0.5, 0.5])^-1`.
* **Reference height**: by two accidents of AssistMimic's code, Inter-X clips play at `root_trans_offset` with
  `full_fix` and HHI-Assist clips at `trans` with `pene_fix` — about a bed's height apart. The converter's
  `--dataset` reproduces each; the recipient then starts 7–19 cm above the bed in all ten clips, which
  `AssistMimicPairMotion` checks against the built box.
* **Bed**: AssistMimic's box pose is reproduced (top face at 0.4386 m), at `(0.3, 0.19, 0.2286)` in the world frame
  the pair shares.
* **Dynamic reference retargeting** (`hhi-assist`): within 1.3 m, each caregiver hand's *observation* reference
  moves by the recipient's tracking error at the nearest recipient body (wrist and palm, not 16 finger links). The
  reward still grades against the clip.
* **MJCF**: Isaac Gym reads joint `stiffness`/`damping` as PD gains and motor `gear` as an effort limit; Genesis would
  read springs and motors, so `PhcSmplHumanoid` moves them to `default_dofs_kp/kd`.
* **Drive gains**: PHC's stiff drives, under-converged by PhysX TGS, ring in Genesis at nominal gains:
  `PHC_DRIVE_KP_SCALE = 0.4`, `PHC_DRIVE_KD_SCALE = 0.7`.

</details>

## Citation

```bibtex
@inproceedings{shibata2026assistmimic,
  title={Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning},
  author={Shibata, Yuto and Yamazaki, Kashu and Jayanti, Lalit and Aoki, Yoshimitsu and Isogawa, Mariko and Fragkiadaki, Katerina},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
