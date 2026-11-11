# weave_hoi — Weave G1-Inspire-HOI in Eden

Port of [Weave](https://github.com/xiaohu-art/Weave)'s `G1-Inspire-HOI-v0` (Isaac Lab): a Unitree G1 with two
Inspire RH56DFQ hands tracks OMOMO human–object clips, with the object simulated and tracked.

```bash
eden train --task weave_hoi                                   # all 9 objects, test split (Weave's train.yaml)
eden train --task weave_hoi --objects smallbox                # single-object specialist
eden train --task weave_hoi --objects smallbox,largebox --split train
eden smoke --task weave_hoi --objects smallbox --num_envs 4   # build + step
```

A run on all nine objects logs under `weave_hoi`; any other set logs apart, under `weave_hoi_<obj>-<obj>…`.

## Pieces

| Piece | Where |
|---|---|
| Robot: G1 body (`G1Dofs29NoHands` on the bare-wrist URDF) + Inspire RH56DFQ hands welded on the wrists at Weave's mount (three scene entities) | `eden/options/robots/composite/g1_inspire.py` (`G1Inspire`), `inspire_rh56.py` (`InspireRH56DFQ_L/R`) |
| Assets, split from Weave's merged URDF (link poses match Weave's recorded bodies to < 10 µm, total mass 33.3847 kg) | `Kashu7100/eden_assets`: `humanoids/unitree_g1/g1_29dof_rev_1_0_bare_wrist.urdf`, `hands/inspire_hand/inspire_hand_dfq_{left,right}.urdf` |
| Dataset: `appolyn/Weave` npz → `robot` / `left_hand` / `right_hand` / `object` sequences, contact labels, per-clip object | `eden/datasets/weave_dataset.py` (`WeaveG1InspireHOIDataset`) |
| One object mesh per env, clips restricted to the env's object | `GroupedEntityOptions` + `MotionTermOptions.env_motion_groups` |
| Coupled fingers driven in software (41-D action) | `ImplicitPDController(mimic_dofs=...)` |
| Object geometry obs (BPS-SDF, fingertip→surface), contact / opposition rewards, lost-contact termination, feet terms | `custom_terms.py` |
| SimBaV2 actor-critic + Muon PPO, at Weave's hyperparameters | `eden/learning/wrappers/rsl_rl/{simba,muon_ppo}/`; Weave's preset in `runner.py` |

## Deviations from Weave (engine-forced or deliberate)

- **Friction** combines as `max(a, b)` in Genesis (PhysX averages), so the randomized ranges are Weave's static
  ranges averaged against its 1.0 ground plane: robot and hands (0.65, 1.3), object (0.7, 1.1). The ground is authored
  at 0.01 so those values decide every ground contact; at 1.0 it would win the max against most of them. Hand-object
  contacts take the larger coefficient where Weave averages the two. Static and dynamic friction are one coefficient
  here.
- **Per-object density**: each object variant carries its own `rho` (`largebox` 25 kg/m³, the rest 200), which
  Eden applies per env after the build, inertia included.
- **Finger gain randomization** draws uniformly in (0.5, 2.0) (Weave: log-uniform); finger armature is not randomized.
- **Contact force** is `Σ‖f_i‖` over the link's contacts with the object (Weave: the net force vector's norm).
- **Observation layout** differs slightly: the hands' base links are their own bodies here (Weave merges them into
  the wrists), so the contact-label/flag blocks are 56 wide instead of 54 (policy 1011-D vs 999-D). Trained from
  scratch, so this does not matter for parity of behaviour.
- **Depenetration cap**: PhysX's `max_depenetration_velocity=1.0` bounds the velocity its solver adds to push
  bodies apart. Genesis does not report that correction on its own, so for 0.1 s after each reset or resync the
  robot's and the object's base linear velocity gain beyond free fall is scaled, each sim step, so the base stays
  within 1.0 m/s of its seeded velocity (gravity passes untouched); a lasting overlap cannot stack its pushes past
  the cap. Always on in PhysX; here only after a seed, which is where the overlap comes from.
- **rsl-rl 5**: actor and critic gradients are clipped separately, and the adaptive-LR floor is 1e-5.

## Expected results (Weave paper)

Joint 9-object policy on the test split: success 95.3% at 27k iterations; single-object specialists ~91% at 3k.
