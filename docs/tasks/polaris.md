# PolaRiS

Eden ports the six DROID scenes of [PolaRiS](https://arxiv.org/abs/2512.16881), a real-to-sim evaluation suite in
which a Franka with a Robotiq 2F-85 works in front of a Gaussian-splat capture of a real scene. Read this page to
run the scenes, score a policy with PolaRiS's rubrics, or generate demonstrations. The code is in
`eden/tasks/polaris/`.

## The tasks

Each scene is registered twice: a single-env task for teleoperation and policy inference, and a `*_batched`
variant that renders many envs at once.

| Task (`--task`) | Batched variant | Instruction | Rubric (success) |
|-----------------|-----------------|-------------|------------------|
| `polaris_block_stack_kitchen` | `polaris_block_stack_kitchen_batched` | "Place and stack the blocks on top of the green tray" | both cubes put in the tray, then the green cube over the wood cube |
| `polaris_food_bussing` | `polaris_food_bussing_batched` | "Put all the foods in the bowl" | the ice cream and the grapes put in the bowl |
| `polaris_pan_clean` | `polaris_pan_clean_batched` | "Use the yellow sponge to scrub the blue handle frying pan" | the sponge put in the pan |
| `polaris_move_latte_cup` | `polaris_move_latte_cup_batched` | "put the latte art cup on top of the cutting board" | the cup put on the cutting board |
| `polaris_organize_tools` | `polaris_organize_tools_batched` | "put the scissor into the large container" | the scissor put in the large container |
| `polaris_tape_into_container` | `polaris_tape_into_container_batched` | "put the tape into the container" | the tape put in the container |

- **Single-env tasks** are `EdenSplatConfig`s (built as a `SplatEnvBase`): one env, no rewards or terminations
  beyond the episode length, for teleoperation and policy inference.
- **Batched tasks** are `EdenSplatRLConfig`s (built as a `SplatRLEnvBase`): four envs by default, the gs-madrona
  batch renderer with a soft two-light rig, and every env's splat rendered and composited under it. They reuse the
  single-env task's objects, splat, instruction and rubric verbatim. Their reward and termination managers are
  empty for now: they are RL-typed scaffolding that renders batched but has nothing to train against yet.
  `env_options.num_envs` and `splat_options.minibatch` trade VRAM for throughput.

## Scoring: PolaRiS's rubrics

PolaRiS scores each task with a rubric: criteria built from three checkers, each counted once it has ever passed,
some only after others have. Success is every criterion reached; progress is the fraction reached. Every Polaris task,
batched or not, carries one success metric, `rubric` (a `RubricMetricTerm`), built in `eden/tasks/polaris/rubric.py`:

| PolaRiS checker | Eden criterion |
|-----------------|----------------|
| `reach(obj, 0.2)` | `EeNearEntity`: the gripper's `base` link (PolaRiS's `ee_frame`, the Robotiq `base_link`) within 20 cm of the object |
| `lift(obj, h)` | `object_bottom_above_height`: the object's lowest point `h` above the table |
| `is_within_xy(a, b, fraction)` | the gripper open (`left_driver_joint` < 0.1 rad, PolaRiS's `finger_joint` test) and `object_a_within_b_xy` |

Most tasks are one `put_in(object, container)` chain: reach, lift, then let go over the container (80 % of the
object's footprint over the container's).

!!! note "Lift is measured from the table, not from the spawn height"
    PolaRiS compares the object's origin with its spawn height. The scanned assets put their origins anywhere, and
    Eden drops its objects onto the table at reset, so the height is measured from the table top (`z = 0` in every
    scene) to the object's lowest point instead.

## The robot and the scene

- **Robot.** `PolarisDroid`: a gravity-compensated Franka arm and a Robotiq 2F-85, welded at the arm's `attachment`
  link. It differs from Eden's `Droid` in its home pose (calibrated to the splat cameras) and in a 2 cm torsional
  friction patch on the pads, without which a pair of scissors pinched across its middle pivots and slips out once
  lifted. The gripper is the pad-only DROID gripper: a full-collision gripper penetrates mesh objects and ejects them
  on contact.
- **Physics.** `sim_dt=0.004` with 4 substeps, so the substep is 1 ms and the 2F-85's closed loop stays stiff, and
  torsional friction on. Graspable objects are stiffened per object (`default_geoms_sol_params`), never through a
  global constraint time constant, which ejects mesh objects on contact.
- **Control.** `decimation=5`: **50 Hz**, a `DifferentialIKController` on `attachment` (`ee_controller`) and a
  binary gripper on the two driver joints (`gripper_controller`). The episode is 30 s.
- **Cameras.** `external_camera` and `wrist_camera`, with intrinsics written for a 1280 × 720 image, each
  compositing the splat behind the rendered scene (`SplatComposite`, the ground plane drawn by the splat).
- **Splats and objects.** Each scene's splat (`<scene>/assets/.../splat.ply`) and its objects
  (`eden.options.objects.polaris`) download on first use from the `eden_polaris` dataset into `polaris/` under the
  asset directory.

## Requirements

Splat rendering goes through gsplat, which JIT-compiles a CUDA extension on first import, so the CUDA toolkit
(`nvcc`) must be installed.

=== "uv"

    ```bash
    uv sync --inexact --extra vis           # single-env tasks
    uv sync --inexact --extra batch-splat   # batched tasks
    ```

=== "pip"

    ```bash
    pip install 'eden-world[vis]'           # single-env tasks
    pip install 'eden-world[batch-splat]'   # batched tasks
    ```

The batched tasks also need [gs-madrona](https://github.com/Genesis-Embodied-AI/gs-madrona), the CUDA-only batch
renderer, installed separately. See [Installation](../getting-started/installation.md) and
[Sensors and rendering](../guides/sensors-and-rendering.md).

## Evaluating a DROID policy

The DROID-family remote-policy clients (`PiPolicy` with `policy_setup="droid"` or `"droid_jointpos"`,
`Gr00tPolicy` with `"oxe_droid_relative_eef_relative_joint"`, `Cosmos3Policy`, `MolmoAct2Policy` and
`GalaxeaVlaPolicy` with `"droid_franka"`) target this rig. Replace the config's observation and action options with
the client's, as on the [SIMPLER page](simpler.md#evaluating-a-generalist-policy).

!!! warning "Set the env to DROID's 15 Hz first"
    Every DROID checkpoint's action row is one 1/15 s step (DROID's `RobotEnv.control_hz`), and the DROID clients
    check the env's control rate when they are constructed. Eden's PolaRiS tasks step every 20 ms (50 Hz), which would replay each chunk 3.3× too fast. Set the
    timing PolaRiS itself uses (`droid_cfg.py`: `sim.dt = 1/120`, `decimation = 8`), with `sim_substeps=9` to keep
    the Robotiq's substep under 1 ms:

    ```python
    cfg.env_options.sim_dt = 1 / 120
    cfg.env_options.decimation = 8
    cfg.env_options.sim_substeps = 9
    ```

See [Remote policies](../guides/remote-policies.md) for the DROID setups and their checks.

## Scripted demonstrations

Every `*_batched` task has an `eden datagen` demonstrator (`eden/tasks/polaris/scripted_demos.py`): the put-on
tool routine on the DROID rig, one put-on per object the rubric scores, reading object footprints off the collision
meshes (privileged). They are registered on the batched tasks because `eden datagen` needs episode ends and many
envs, which the single-env tasks do not have.

```bash
eden datagen --task polaris_move_latte_cup_batched --num_envs 16 --num_episodes 64
```

What each demonstrator does and its success rates are in
[Scripted demonstrations](../guides/scripted-demos.md).

## See also

- [SIMPLER](simpler.md), the WidowX counterpart
- [Scenes and terrains](../guides/scenes-and-terrains.md) for Gaussian-splat scenes
- [Remote policies](../guides/remote-policies.md)
- [Task suites overview](index.md)
