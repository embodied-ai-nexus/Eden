# Grasp sampling

A grasp sampler turns the point cloud of one entity into ranked 6-DoF grasp poses that you can hand straight to
inverse kinematics. Eden wraps two learned backends behind one interface, [`GraspSamplerBase`](../api/extensions/grasp-sampling.md)
(`eden/extensions/grasp_sampling/base.py`). This page covers installing them, sampling, the frame each pose
places (the part that is easy to get wrong), where the object's points come from, checking grasps against the
scene, and adding a backend.

| Backend | Model | Grippers | Collision filter |
| --- | --- | --- | --- |
| `GraspGenXSampler` (`eden.extensions.grasp_sampling.graspgenx`) | [GraspGenX](https://github.com/NVlabs/GraspGenX): one model, conditioned on a gripper description | every gripper GraspGenX has a description of, picked by name (`GraspGenXGripper`) — parallel jaws and dexterous hands | no (see [below](#checking-a-grasp-against-the-scene)) |
| `GraspGenSampler` (`eden.extensions.grasp_sampling.graspgen`) | [GraspGen](https://github.com/NVlabs/GraspGen): one model per gripper | `franka_panda`, `robotiq_2f_140`, `suction_cup_30mm` | yes |

## Sample grasps

```python
from eden.extensions.grasp_sampling.graspgenx import GraspGenXSampler

sampler = GraspGenXSampler(env, GraspGenXSampler.configure(end_effector="FrankaEmikaPanda"))
result = sampler.sample_grasps("mustard")        # GraspResult, best first
robot = env.entities["robot"]
qpos = robot.inverse_kinematics(sampler.pose_link_name, result.pos[:1], result.quat[:1])
```

`env` is a built env holding the target entity and, for the default `object_cloud="camera"`, a camera named by
`camera_name` (default `"camera"`). The model loads at construction and runs on CUDA only; the Eden env itself may
stay on `gs.cpu`.

A `GraspResult` holds world-frame `pos` `(n, 3)`, wxyz `quat` `(n, 4)` and `confidences` `(n,)` on the env's device,
in descending confidence, so they go straight into `inverse_kinematics` or a setter. `result.success` is false when
no grasp cleared `grasp_threshold` (or the collision filter). Each pose places the link
`sampler.pose_link_name` — see [Which link a pose places](#which-link-a-pose-places).

Runnable versions:

- `examples/recipes/planning/grasp_sampling_graspgenx.py` — the Franka hand or a UR5e with a Robotiq 2F-85, IK
  onto the grasps, and a collision check in simulation.
- `examples/recipes/planning/grasp_sampling_graspgenx_hands.py` — every GraspGenX end-effector Eden pairs (the
  parallel jaws and the dexterous hands) placed at its grasps on YCB objects, the fingers closed, and the contacts
  reported.
- `examples/recipes/planning/grasp_sampling_graspgen.py` — GraspGen with its collision filter on a Franka.

### Batched envs

`sample_grasps(name, env_idx=0)` samples one env. `sample_grasps_batch(name, envs_idx=None)` samples several (all
of them by default) with one render and one model call, where a call per env would render the whole batch each
time; a scripted demonstrator running one routine per env samples every env there first.

With `num_envs > 1`, a camera the sampler reads must render each env on its own — `EnvOptions(env_separate_rigid=True)`
or the batch renderer — because a shared view holds every env's copy of the target under one segmentation id. The
sampler checks this at construction. `object_cloud="surface"` reads no camera.

## Install

Each backend is an extra; they install side by side:

```bash
pip install 'eden-world[graspgenx]'    # or, in a checkout: uv sync --inexact --extra graspgenx
pip install 'eden-world[graspgen]'     # then run `grasp_gen download` and export GRASPGEN_CHECKPOINT_DIR as it prints
```

- **`graspgenx`** installs [`eden-graspgenx`](https://pypi.org/project/eden-graspgenx/) (import name
  `graspgenx`), an inference-only packaging of GraspGenX. Its checkpoints and gripper descriptions download from
  Hugging Face on first use. To run offline, point `GRASPGENX_CHECKPOINT_DIR` (a directory holding `gen/` and
  `dis/`) and `GRASPGENX_GRIPPER_CFG_DIR` at local copies, or pass `checkpoint_dir=`.
- **`graspgen`** installs [`eden-grasp-gen`](https://pypi.org/project/eden-grasp-gen/) (import name `grasp_gen`),
  under NVIDIA's non-commercial license. `gripper_config=` is a path, or a path relative to
  `GRASPGEN_CHECKPOINT_DIR`.
- Both are pure-Python wheels with floor-only pins, so neither downgrades Eden's dependencies.

A missing package raises an `ImportError` carrying the install command.

## Which link a pose places

Both models return *grasp frames*: z is the approach direction and x the direction the fingers close along,
whatever the gripper. A grasp frame is generally no link of the real gripper, and driving a link to it anyway
fails quietly — a parallel jaw closes 90° off its intended axis, a hand lands upside down — while the poses look
plausible until the fingers close. So each pose places a named link, `sampler.pose_link_name`:

- **With `end_effector=`** — the Eden robot that executes the grasps, by its options class name. It picks the
  backend's gripper (`gripper_name` for GraspGenX, `gripper_config` for GraspGen) and makes each pose a pose of that
  robot's link. Only pairings checked against the Eden robot are offered: each backend's `END_EFFECTORS`, typed as
  a `Literal` (`GraspGenXEndEffector`, `GraspGenEndEffector`). Passing `end_effector=` together with a different
  gripper option raises.
- **Without it**, GraspGenX's poses place the description's base link (its URDF welds that link to the grasp frame
  through a fixed `world_joint`, which the sampler applies), and GraspGen's are its bare grasp frame
  (`pose_link_name` is `None`). Mapping either onto an Eden link is then yours, and needs the check below first.

| `end_effector` | Backend gripper | Eden link the pose places |
| --- | --- | --- |
| `FrankaEmikaPanda`, `FrankaHand` | GraspGenX `franka_panda` | `hand` |
| `FrankaEmikaPanda`, `FrankaHand` | GraspGen `graspgen_franka_panda.yml` | `hand`: the grasp frame **turned 90° about z** |
| `Robotiq2f85` | GraspGenX `robotiq_2f_85` | the root, `world` (not the gripper's `base` link) |
| `SharpaWave_R` | GraspGenX `sharpa_wave` | the root, `right_hand_C_MC` |
| `WujiHand1_R` | GraspGenX `wuji_hand` | the root, `right_palm_link` (`WujiHand2Beta2_R` is a different hand) |
| `InspireRH56_R` | GraspGenX `inspire_hand` | the root, `base`: the description's base **after a half turn** about (0, 1, −1)/√2 |

`tests/extensions/grasp_sampling/test_end_effectors.py` checks every row on the built Eden robot: a parallel jaw's
pads, open, must straddle the TCP along the closing axis (within 1 cm), and a dexterous hand's links must sit
where its GraspGenX description puts them. For GraspGen on YCB objects, the bare grasp frame put the Franka hand
into the cracker box and the mustard bottle on all ten best grasps; turned the quarter turn, 7 and 9 of them
cleared and the best pinched (recorded next to `GraspGenSampler.END_EFFECTORS`).

A new pairing is an `EndEffectorPairing(gripper, link_name, link_in_frame)` in the backend's `END_EFFECTORS`,
where `link_in_frame` is the Eden link's pose in the frame the backend's poses place without a pairing. Add it to
that test as well.

Besides the pose, the sampler exposes what executing a grasp needs, all in the posed link's frame:
`sampler.tcp_offset` (the point the grasp puts on the object), `sampler.closing_axis`, and `sampler.approach_axis`
(to back off a pregrasp along). GraspGenX also gives the description's `open_dofs_pos` / `close_dofs_pos`, by
joint name. Eden's dexterous hands share those joint names (`sharpa_wave` with `SharpaWave_R`, `inspire_hand` with
`InspireRH56_R`), so the configuration a grasp assumes applies to them as is.

`gripper_name` is typed as `GraspGenXGripper`, a `Literal` of the descriptions GraspGenX ships, grouped by kind
(`parallel_2f`, `revolute_2f`, `revolute_3f`), so an editor offers them. The descriptions found on disk at
construction are what is checked, so a custom description under `GRASPGENX_GRIPPER_CFG_DIR` works too.

## Options

Shared options (`GraspSamplerOptions`, `eden/options/extensions/grasp_sampling.py`), passed to
`<Backend>.configure(...)`:

| Option | Default | Meaning |
| --- | --- | --- |
| `camera_name` | `"camera"` | Camera whose point cloud is read (for `object_cloud="camera"` and the collision filter). It must use the legacy camera backend — no `sensor_options` — which renders point clouds and segmentation. |
| `object_cloud` | `"camera"` | `"camera"` or `"surface"`; see below. |
| `surface_points` | `2048` | Points sampled over the surface when `object_cloud="surface"`. |
| `grasp_threshold` | `0.8` (GraspGenX: `0.7`) | Confidence a grasp needs to be returned. GraspGenX's confidences run lower than GraspGen's. |
| `num_grasps` | `200` | Grasps proposed per inference pass, before thresholding. |
| `max_grasps` | `None` | Keep at most this many of the best grasps. |
| `filter_collisions` | `False` | Drop grasps whose gripper collision mesh comes within `collision_threshold` (`0.02` m) of the obstacle cloud. |
| `collision_ignore_entities` | `None` | Entities left out of the obstacle cloud. `None` leaves out every actuated entity (the arm about to move); `[]` keeps all. Names are checked at construction. |
| `max_scene_points` | `8192` | The obstacle cloud is downsampled to this many points. |
| `end_effector` | `None` | See [above](#which-link-a-pose-places). |

GraspGenX adds `gripper_name`, `checkpoint_dir`, `planner` and `graspmoe`. `planner="graspmoe"` (the default)
scores the diffusion model's grasps together with grasps swept over the object's oriented bounding box, which
helps on boxy objects; `"diffusion"` returns the diffusion model's grasps alone. `graspmoe` is a `GraspMoEOptions`
holding the bounding-box sweep's knobs. GraspGen adds `gripper_config`.

## Where the object's points come from

`object_cloud` picks the cloud the model sees:

- `"camera"` (default) — the pixels of the entity that `camera_name` sees, through its segmentation. That is what a
  real robot has. The model centres its grasps on the visible part, so on a round object seen from one side the
  grasp centre is offset toward the camera — an offset a parallel jaw with little spare stroke may not absorb.
- `"surface"` — points sampled over the entity's whole surface (no camera needed). Grasps are centred on the true
  object, but the hidden underside is included, so the model also proposes grasps from below whatever the object
  rests on; the reachability and collision checks below reject those.

A camera that sees none of the entity raises a `ValueError` instead of returning an empty result: "nothing to
sample from" and "the model found no grasp" (`result.success is False`) are different failures.

## Checking a grasp against the scene

`filter_collisions=True` drops grasps whose gripper collision mesh comes within `collision_threshold` of the
camera's obstacle cloud — every entity but the target and the ignored ones. Only GraspGen supports it.
**GraspGenX refuses it at construction**: none of its descriptions ships a collision mesh (`coll_mesh.obj`), and
GraspGenX substitutes a 1 cm box that every grasp clears.

Check a GraspGenX grasp in simulation instead: IK the arm onto it, set the fingers open, and call
`detect_collision()` on the arm and gripper entities. That runs the narrow phase on the posed geometry without
stepping, against the real collision geometry of everything in the scene. Subtract the pairs already in contact at
the home pose (a gripper welded onto its flange) and ignore pairs with the target.
`examples/recipes/planning/grasp_sampling_graspgenx.py` does exactly this.

## Adding a backend

1. Subclass `GraspSamplerBase` and implement `_infer(object_points) -> (grasp frames (k, 4, 4), confidences (k,))`
   in world frame. Override `_infer_batch(clouds)` too if the model batches. The base class reads the clouds,
   filters, places the posed link, ranks, applies `max_grasps` and packs the result.
2. At the end of `__init__`, call `_set_pose_frame(link_name, grasp_T_frame, tcp_in_grasp)` to name the frame the
   poses place without a pairing.
3. List the verified pairings in `END_EFFECTORS` and name the option that selects the gripper in `_gripper_option`.
4. If the backend's grippers carry collision geometry, implement `_collision_free(grasps, scene_points) -> mask`
   on grasp frames. Otherwise leave it: the base class then refuses `filter_collisions=True` at construction,
   quoting `_no_collision_filter_reason`, which you can override to say why.
5. Register it with `@GRASP_SAMPLER_REGISTRY.register()` and add its module to the registry's `populated_by`.

## See also

- [Planning and system identification](planning-and-sysid.md) — collision-free paths to a grasp (cuRobo, OMPL).
- [Scripted demonstrations](scripted-demos.md) — routines that execute grasps.
- [Entities](entities.md) — `inverse_kinematics`, `detect_collision`, pose frames.
- [Sensors and rendering](sensors-and-rendering.md) — cameras, segmentation and the batch renderer.
