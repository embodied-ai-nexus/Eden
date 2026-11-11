# Sensors and rendering

This page covers everything an Eden environment can observe besides its own state: cameras and the renderers behind
them, Gaussian-splat backgrounds, camera overlays, video recording, and the Genesis sensors (IMU, contact, lidar,
depth, tactile and the rest). Read it when you give a policy pixels, record a rollout, or attach a sensor to a robot.

Cameras are declared in a config's `cameras_options`, sensors in its `sensors_options`, and the renderer in its
`renderer_options`. All three are built with the env, in `env.build()`; afterwards they are reachable as
`env.cameras[name]`, `env.sensors[name]` and `env.scene.visualizer`.

## Cameras

### Declaring a camera

A camera is a named [`CameraOptions`](../api/options/sensors.md) entry of `CamerasOptions`. From the vision
lift task (`eden/tasks/manipulation/lift_box_rgb/config.py`), a 128x72 wrist camera mounted on the robot's tool
flange:

```python
from eden.options.camera import CameraOptions, CamerasOptions

cameras_options = CamerasOptions(
    wrist=CameraOptions(
        cam_resolution=(128, 72),
        cam_fov=80,
        attach_entity_name="robot",
        attach_link_name="attachment",
        attach_offset=(0.070, -0.026, 0.013),
        attach_quat=(0.103966, 0.692961, 0.703320, 0.119734),  # wxyz
    ),
)
```

| Field | Default | Meaning |
|---|---|---|
| `cam_pos`, `cam_lookat`, `cam_up` | `(1.5, 0.0, 1)`, `(0.0, 0.0, 0.5)`, `(0.0, 0.0, 1.0)` | World pose of a free camera. |
| `cam_resolution` | `(720, 480)` | Image size as `(width, height)`. |
| `cam_fov` | `80.0` | Vertical field of view, in degrees. |
| `cam_intrinsic` | `None` | A 3x3 intrinsic matrix. When set, it overrides `cam_fov` and `cam_resolution` (with a warning). |
| `cam_extrinsic` | `None` | A 4x4 world-to-camera matrix. When set, it overrides `cam_pos`, `cam_lookat` and `cam_up`. |
| `cam_near`, `cam_far` | `None` | Clipping planes in meters; `None` keeps the renderer's default. |
| `model`, `cam_aperture` | `"pinhole"`, `2.0` | Camera model (`"pinhole"` or `"thinlens"`) and aperture. |
| `follow_entity_name` | `""` | Follow an entity, keeping the camera's orientation. |
| `attach_entity_name`, `attach_link_name`, `attach_offset`, `attach_quat` | `""`, `""`, `None`, `None` | Mount the camera on a link. All four go together; the quaternion is wxyz. |
| `debug` | `False` | A debug camera rasterizes even in a scene that uses the batch renderer. |
| `overlays` | `[]` | What is painted onto every RGB frame (see [Camera overlays](#camera-overlays)). |
| `sensor_options` | `None` | Build the camera as a Genesis camera *sensor* instead (see [Camera backends](#camera-backends)). |

`follow_entity_name` and the `attach_*` fields exclude each other. An entity name that does not exist raises at build
and lists the scene's entities.

### Reading images

On a built env, a camera renders on demand:

| Method | Returns |
|---|---|
| `render_rgb()` | The RGB frame, with the camera's overlays painted on. Recorded when the camera is recording. |
| `render_depth()` | Depth. |
| `render_segm()` | Segmentation ids. |
| `render_normal()` | Surface normals. |
| `render_pointcloud(world_frame=True)` | `(points, valid_mask)`, one 3D point per pixel. |
| `render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)` | The raw multi-output pass, `(rgb, depth, segmentation, normal)`. No overlays. |
| `snapshot(image_path=None)` | Writes `render_rgb()` of the first image to a PNG (default `{log_dir}/snapshot_<time>.png`). |

A camera that renders several environments at once (the batch renderer, the sensor backend, or a scene with
`EnvOptions.env_separate_rigid=True`) returns one image per environment, with a leading batch dimension. A plain
rasterizer camera returns the image of the one environment it is bound to.

The pose can be read and moved after the build: `get_pos()`, `get_quat()`, `get_lookat()`, `get_up()`,
`get_transform()`, `set_pose(transform=None, pos=None, lookat=None, up=None)`, `set_pos(pos)`, `set_quat(quat)`,
each with an optional `envs_idx=`.

### Cameras as observations

Five observation terms read a camera by name: `rgb_image`, `depth_image`, `segmentation_image`, `normal_image` and
`pointcloud_image` (which zeroes the pixels that hit nothing, so infinite depth never reaches the policy).
`rgb_image` returns the renderer's native output, `(N, H, W, C)` `uint8`, unless you ask for a CNN-ready tensor:
`normalize=True` converts to float in `[0, 1]`, `channels_first=True` permutes to `(N, C, H, W)`, and either one drops
an alpha channel (the batch renderer returns RGBA). The vision lift task feeds the wrist camera to its CNN this way:

```python
from eden import observations
from eden.options import ObservationGroupOptions
from eden.terms import ObsTerm

wrist_rgb = ObsTerm.configure(
    func=observations.rgb_image,
    params={"camera_name": "wrist", "normalize": True, "channels_first": True},
)
camera_group = ObservationGroupOptions(wrist_rgb=wrist_rgb, enable_corruption=False)
```

A group that holds a single image term keeps its `(N, C, H, W)` shape, which is how rsl_rl's CNN actor-critic
recognizes an image group. See [Observations](observations.md) for groups, noise and history.

Rendering one image per environment per step is the expensive part of a vision task: use the batch renderer when
you train from pixels with many environments.

### Segmentation

`EnvOptions.segmentation_level` (`"entity"` by default, or `"link"`, `"geom"`) sets what one segmentation id stands
for. `env.segmentation_idx_dict` maps an id to the scene element (an `entity_idx`, or a tuple at link and geom level);
`env.entity_segmentation_ids` maps the other way, `entity_idx -> [ids]`. Use the latter to mask an entity out of a
segmentation image: at link or geom level an entity owns several ids.

### Camera backends

A camera is built one of two ways:

- **The default backend** (`sensor_options=None`) calls Genesis' `scene.add_camera()`. It renders RGB, depth,
  segmentation, normals and point clouds, through the scene's renderer.
- **The sensor backend** (`sensor_options=<Genesis camera sensor options>`) calls `scene.add_sensor()`. Use it for
  Nyx (`NyxCameraOptions`) or for a camera with its own lights (`RasterizerCameraOptions(lights=...)`, as the LIBERO
  port does). It renders **RGB only**: `render_depth()`, `render_segm()`, `render_normal()`,
  `render_pointcloud()` and `render()` raise. Read it with `render_rgb()` or `read()` (Genesis' `CameraData`, whose
  `.rgb` is the image), and use `pick_pixel(camera_index, x, y)` to cast a ray through a pixel. The sensor's `pos`,
  `lookat`, `up`, `fov` and `res` are filled from the `cam_*` fields unless you set them on the sensor options.
  `cam_near` / `cam_far` are passed on too; setting a different plane on both raises rather than letting one win.

An overlay that reads segmentation or a point cloud cannot run on the sensor backend and is refused when the env
is constructed.

## Renderers

`renderer_options` picks the renderer the scene's cameras use. The interactive viewer always uses the rasterizer.

| Renderer | Options | Install | Use it for |
|---|---|---|---|
| Rasterizer | `gs.renderers.Rasterizer()` (the default) | included | Everything by default: debugging, recordings, small vision tasks. |
| Batch renderer (Madrona) | `BatchRendererOptions` (`eden.options`) | `gs-madrona`, Linux x86-64 only | Training from pixels: one batched render of every environment. |
| Ray tracer | `RayTracerOptions` (`eden.options`) | Genesis' ray tracer | Offline, high-quality stills. |
| Nyx path tracer | per camera: `CameraOptions(sensor_options=NyxCameraOptions(...))` | `pip install 'eden-world[nyx]'` / `uv sync --inexact --extra nyx`; no macOS build; CUDA only | Photoreal renders and Nyx splat scenes. |

### Rasterizer

The rasterizer draws shadows when `EnvOptions.rasterizer_shadows` is on (the default); Eden applies the switch to
rasterizer camera sensors too. `EnvOptions.background_color` (default `(0.04, 0.08, 0.12)`) is the empty-space
colour. Shadows need a GPU-backed OpenGL context: a software renderer (llvmpipe) draws none.

### Batch renderer

`BatchRendererOptions` subclasses Genesis' `gs.renderers.BatchRenderer` and adds a declarative `lights` list.

!!! warning "Madrona renders ambient-only until you add lights"
    Without lights the cameras come out near-black, and Madrona bakes the light count when the scene is built, so
    every light has to exist before `env.build()`. Declare them on `BatchRendererOptions.lights` (one
    `scene.add_light(**kwargs)` dict each) and Eden adds them right after it creates the scene. Genesis' plain
    `gs.renderers.BatchRenderer` has no such field.

From `eden/tasks/manipulation/lift_box_rgb/config.py`:

```python
from eden.options import BatchRendererOptions

renderer_options = BatchRendererOptions(
    lights=[
        {"pos": (0.3, -0.6, 1.5), "dir": (0.0, 0.4, -1.0), "directional": True, "castshadow": False, "intensity": 3.0},
        {"pos": (0.3, 0.6, 1.5), "dir": (0.0, -0.4, -1.0), "directional": True, "castshadow": False, "intensity": 3.0},
    ]
)
```

`use_rasterizer=True` (Genesis' field) makes Madrona rasterize instead of ray trace; the extreme-parkour student
uses it for its batched depth camera. The first run of a scene compiles Madrona's render kernels, which takes a while
(the extreme-parkour README reports 10–15 minutes for its scene).

### Nyx

Nyx is a CUDA path tracer that plugs into Genesis as a camera sensor (`gs_nyx_plugin`). Import its options inside your
script, after `eden.init()`; importing them registers the plugin with Genesis. A Nyx scene needs at least one light or
environment map. From `examples/recipes/rendering/nyx_multi_camera_multi_env.py`, two Nyx cameras over four
environments:

```python
import eden as en
import genesis as gs

from eden.options.camera import CameraOptions, CamerasOptions

en.init(backend=gs.gpu)
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions  # registers the plugin with Genesis

lights = [
    {"type": "point", "pos": (2.5, -2.5, 4.0), "color": (1.0, 0.97, 0.92), "intensity": 80.0},
    {"type": "directional", "dir": (0.2, 0.3, -1.0), "color": (1.0, 1.0, 1.0), "intensity": 2.0},
]
cameras_options = CamerasOptions(
    wide=CameraOptions(
        cam_pos=(0.5, -2.5, 4.0),
        cam_lookat=(0.0, 0.0, 0.3),
        cam_fov=42.0,
        sensor_options=NyxCameraOptions(spp=32, denoise=True, lights=lights),
    ),
)
# after env.build() and a step:
rgb = env.cameras["wide"].read().rgb   # (num_envs, H, W, 3)
```

The lights of every Nyx camera are merged scene-wide. More recipes live in `examples/recipes/rendering/`
(`minimal_nyx.py`, `nyx_light_types.py`, `nyx_materials.py`, `nyx_object_picking.py`). The Nyx plugin knows only
Genesis' own solvers: a Mochi scene rendered through Nyx needs the bridge in `examples/recipes/mochi/_nyx.py`, or it
renders a still frame of the rest pose without any error.

### Ray tracer

`RayTracerOptions` subclasses Genesis' `gs.renderers.RayTracer` with an indoor environment map (decoded on first
use) and one overhead light. Several Eden features refuse it at build: the Mochi solver, `PlanePaintOverlay`, and a
background composite with a non-zero `shadow_strength`.

### Surfaces

What a body looks like is its `surface`, an `eden.options.surfaces` class: `DefaultSurfaceOptions`,
`PlasticSurfaceOptions`, `MetalSurfaceOptions`, `BSDFSurfaceOptions`, `GlassSurfaceOptions` or
`EmissionSurfaceOptions`. A Genesis `gs.surfaces.*` object is accepted too.

## Gaussian splats

A Gaussian splat gives a scene a photoreal background (a captured kitchen, a table top) while the physics runs on
simple collision geometry. Eden has two backends, each with its own config leaf and env class:

| Backend | Config | Env | Options | Install |
|---|---|---|---|---|
| gsplat | `EdenSplatConfig` / `EdenSplatRLConfig` | `SplatEnvBase` / `SplatRLEnvBase` | `splat_options=SplatOptions(...)` | `pip install 'eden-world[batch-splat]'` / `uv sync --inexact --extra batch-splat` (also in `vis`); CUDA |
| Nyx | `EdenNyxSplatConfig` | `NyxSplatEnvBase` | `splat_options=NyxSplatOptions(...)` | the `nyx` extra |

`env_class_for_config(cfg)` picks the env from the config class. Each splat is a [`PCDOptions`](../api/options/splat.md)
entry given as a named extra of the splat options. `path` is a PLY file, fetched from the Hugging Face dataset named by
`registry` / `dataset` when it is not a local file; `scale`, `pos` and `quat` (wxyz) place it.

### gsplat

From the PolaRiS port (`eden/tasks/polaris/base_config.py`), simplified:

```python
from eden.entities.camera_overlays import SplatComposite
from eden.options.camera import CameraOptions
from eden.options.splat import PCDOptions, SplatOptions

splat_options = SplatOptions(
    background=PCDOptions(path=path, dataset="eden_polaris", assets_subdir="polaris", scale=scale, pos=pos, quat=quat),
)
external_camera = CameraOptions(..., overlays=[SplatComposite.configure(entities=["plane"])])
```

The splat is composited over the sim pass by segmentation: wherever a camera's frame shows nothing, or one of the
`entities` of its `SplatComposite` (here the ground plane standing in for the real table), the splat replaces it.
Robot and object pixels stay from the sim pass.

!!! note "A camera shows the splat only if it declares a `SplatComposite`"
    A splat env's camera without one shows the plain sim pass. `EdenSplatConfig.recording_camera()` (what
    `eden inference --rec` adds) declares one with no `entities`, so the splat shows behind empty space; a task whose
    splat stands in for a sim surface overrides it to name that entity.

`SplatOptions` fields:

- `per_env_gaussians` (default `False`). `False` stores one Gaussian set and renders every environment's camera from
  it: memory-light, and the splat looks the same in every environment. `True` gives each environment its own
  Gaussians, so a splat bound to an entity (`PCDOptions(attach_entity=..., attach_link=...)`) follows that
  environment's simulated pose. Memory then grows with `num_envs`, which fits tens to hundreds of environments.
- `camera_mode` (default `"auto"`). A `SplatCamera` renders one view and uses it for every environment's image (a
  fixed camera over a shared background); a `BatchSplatCamera` renders every environment's view. `"auto"` picks the
  batched camera for an attached or following camera or under `per_env_gaussians=True`, the single view otherwise.
  `"single"` and `"batched"` force one class for every camera; `"single"` with `per_env_gaussians=True` is refused.
- `minibatch` (default `None`). Chunk size over the views, which bounds peak GPU memory: gsplat evaluates
  view-dependent colour per rendered view.

### Nyx splats

`NyxSplatOptions` renders the splats inside the path tracer as Nyx light fields. Every camera of a
`NyxSplatEnvBase` becomes a Nyx camera: one declared without `sensor_options` gets a `NyxCameraOptions` built from
the splat options (`spp=32`, `denoise=True`, `near`, `far`, `lights`, `env_map_paths`), and the env's light fields and
environment maps are merged into one that brings its own. `tone_mapper` (default `"ACES_1_3"`; also `"Neutral"`,
`"Off"`, `"Reinhard"`) and `exposure_compensation` (in EV, default `0`) are written into each camera's profile.
`PCDOptions.multiplier` and `tint` scale a splat's radiance on this backend only. `max_pt_depth` must stay `None`: the
plugin has no path-depth control, so a value is refused rather than ignored.

## Camera overlays

Some task state is visible in the real world but has no geometry in the scene: ink on a whiteboard, strokes of a
brush. A camera overlay paints that state onto every RGB frame the camera produces, after the renderer made it. A
background composite is an overlay too. Overlays apply in `render_rgb()`, so they reach the returned frame, the
recording, `snapshot()` and any observation that reads the camera; `render()` stays the raw pass.

An overlay is declared by registry key on `CameraOptions.overlays`, with the overlay class's `configure()`, so it
survives a saved `config.json`. From the whiteboard task (`eden/tasks/whiteboard/config.py`):

```python
from eden.entities.camera_overlays import PlanePaintOverlay

CameraOptions(
    cam_pos=(-0.30, -1.05, 1.68),
    cam_lookat=(0.45, 0.02, 1.18),
    cam_fov=42.0,
    cam_resolution=(960, 540),
    overlays=[PlanePaintOverlay.configure(metric_name="drawing", color=ink)],
)
```

Eden ships three overlays (`eden.entities.camera_overlays`):

| Overlay | Paints |
|---|---|
| `PlanePaintOverlay(metric_name, color=None, max_gap=0.002, supersample=4)` | The marks a metric term returns (a `PaintedPlaneSource`, such as the whiteboard's drawing metric) where the camera sees the plane they lie on. Shaded like the surface, so marks keep its shadows. |
| `BackgroundImageComposite(image_path, entities=[], shadow_strength=0.0)` | An image in place of the background: a real-image backdrop, or a green screen. |
| `SplatComposite(entities=[], shadow_strength=0.0)` | The env's Gaussian splat in place of the background. Only a gsplat env's camera can carry it. |

Rules:

- **One background composite per camera, declared first.** The overlays after it leave its pixels alone, since
  they no longer show the sim surface an overlay locates its marks on.
- **Parameters are checked in the config.** An unknown parameter or a bad value raises when the
  `CameraOverlayOptions` is constructed, naming the overlay and its parameters.
- **A point-cloud overlay costs a second render.** `PlanePaintOverlay` needs each pixel's world point, so the camera
  renders the view's depth once more per frame (once for all its overlays).
- **`PlanePaintOverlay` needs a plain-coloured surface** on the rasterizer or the batch renderer; an image-textured
  surface and the ray tracer are refused at build.
- **Frames rendered before the managers exist are not painted**, for instance the one the observation manager
  renders while it loads to size its terms.

### Shadows on a background

By default a composited background receives no shadows: the objects seem to float. Set `shadow_strength` (0 to 1) to
darken the background wherever the sim pass shows a shadow on the composite's `entities`; it needs at least one
entity, since the shadows are read from them.

- **Rasterizer:** the camera renders the view a second time with shadows off and compares each pixel with itself, so
  any surface works. `EnvOptions.rasterizer_shadows` must stay on, and the OpenGL context must be GPU-backed.
- **Batch renderer:** shadows cannot be switched off per render, so the unshadowed brightness of each background
  entity is estimated from its brightest pixels in view. This is exact for a flat, untextured surface under
  directional lights. An image-textured background entity is refused, and Genesis' plane is one by default (a
  checkerboard), so give it a plain colour, `PlaneOptions(color=(0.5, 0.5, 0.5))`. At least one light needs
  `castshadow=True`. A point light's falloff is read as shadow, and a view where almost all of the entity is in
  shadow loses the shadow.

Shading holds about `20 * N * H * W` bytes of temporaries per frame (about 1 GB at 1024 envs of 224x224).

### Writing an overlay

Subclass `CameraOverlay`, declare its parameters as class attributes, register it with
`@CAMERA_OVERLAY_REGISTRY.register()`, and implement `paint(rgb, buffers, envs_idx)`. `rgb` is `(N, H, W, 3)` on the
env's device in the renderer's dtype; return the same shape and dtype, and never write into `rgb` (it may be the
renderer's own buffer). Set `_needs_segmentation = True` or `_needs_pointcloud = True` to receive those in `buffers`.
Resolve whatever the overlay reads in `build()`, which runs after the managers are built, and check parameter values in
the classmethod `check_parameters(params)`.

## Recording video

Recording belongs to Eden's `Camera`:

```python
camera = env.cameras["rec"]
camera.start_recording("rollout.mp4", fps=1.0 / env.dt)   # real-time playback
for _ in range(steps):
    env.step(actions)
    camera.render_rgb()                                   # one frame per call
camera.stop_recording()
```

The contract:

- **One frame per `render()` or `render_rgb()` call**, not per `scene.step()`. A kinematic replay that never steps
  the scene records exactly as a stepped rollout does.
- `start_recording(video_path=None, fps=60.0)` fixes the path and the frame rate for the whole video. The default
  path is `{log_dir}/video_<time>.mp4`. Playback runs at `fps` render calls per second, whatever the simulation time
  step, so pass `fps=1 / env.dt` for real time when you render once per control step. (`eden inference --rec`
  uses the default 60 fps and renders once per control step, so a task with a 50 Hz policy plays back 1.2x fast.)
- Frames are encoded on a background thread and streamed to the file, so a long recording does not fill memory.
- A camera that renders several environments writes one video per environment, `{stem}_{i}{ext}`.
- Starting a second recording raises, stopping one that was never started raises, and a recording that captured no
  frame writes no file and warns. A camera destroyed while recording finalizes its file.

!!! warning "Never use Genesis' own `start_recording`"
    Genesis' `gs.Camera.start_recording` captures inside `scene.step()`, so it records nothing for a replay script.
    Use `Camera.start_recording` / `stop_recording`. Contributors: do not inline `Camera._render()` into `render()` —
    the public methods record exactly once per call, and the inlined call would capture twice.

`eden inference --rec` records a run without any code: it calls `config.add_recording_camera()`, which adds
`config.recording_camera()` as a camera named `rec`, and records it every step to the run's log directory. The default
recording camera follows the `robot` entity from 1.5 m out; a task whose subject is elsewhere, or that paints state
through an overlay, overrides `recording_camera()`. A `rec` camera the config already declares is kept; if it lacks
the overlays of the task's own recording camera, Eden warns and names them. The recording camera is a method rather
than a `cameras_options` entry so that training runs never pay for it.

## Sensors

A sensor is a named [`SensorOptions`](../api/options/sensors.md) entry of `SensorsOptions`, wrapping any Genesis
sensor options. From `examples/recipes/sensors/sensors_imu_contact_go1.py`:

```python
import genesis as gs

from eden.options.sensors import SensorOptions, SensorsOptions

sensors_options = SensorsOptions(
    imu=SensorOptions(
        sensor=gs.sensors.IMU(delay=0.01, acc_noise=(0.01, 0.01, 0.01)),
        attach_entity_name="robot",
    ),
    contact_fl=SensorOptions(
        sensor=gs.sensors.Contact(),
        attach_entity_name="robot",
        attach_link_name="FL_foot",
    ),
)
```

After the build, `env.sensors["imu"].read()` returns the measured value (with noise, delay and jitter applied) and
`read_ground_truth()` the exact one. `get_pos()` / `get_quat()` give the sensor's world pose (wxyz), and
`set_pos_offset()` / `set_quat_offset()` move an attached sensor relative to its link.

| `SensorOptions` field | Meaning |
|---|---|
| `sensor` | The Genesis sensor options (`gs.sensors.*`, or an Eden engine sensor). |
| `attach_entity_name`, `attach_link_name` | The entity and link the sensor rides on. Empty for a static sensor. Eden resolves them into the sensor's `entity_idx` / `link_idx_local` at build. |
| `track_link_names` | For sensors with a `track_link_idx` field: links to track, each `"entity"` (all its links) or `"entity/link"`. |
| `filter_link_names` | For contact sensors (`Contact`, `ContactForce`, and the taxels): counterpart links to ignore, same grammar. Refused on a sensor without `filter_link_idx`. |

### Sensor types

Every Genesis sensor options class works. The ones Eden's tasks and recipes use, and the rest that Genesis ships:

| Sensor (`gs.sensors.*`) | Measures |
|---|---|
| `IMU` | Linear acceleration and angular velocity of a link. |
| `Contact` | Whether a link is in contact. |
| `ContactForce` | Contact force on a link. |
| `JointTorque` | Joint torques of an entity. |
| `Lidar` (alias of `Raycaster`) with `GridPattern` / `SphericalPattern` | Ray distances and, optionally, hit points. |
| `DepthCamera` with `DepthCameraPattern` | A ray-cast depth image. |
| `KinematicTaxel`, `ElastomerTaxel`, `ProximityTaxel`, `ContactProbe`, `ContactDepthProbe` | Tactile readings at probe points on a link. |
| `SurfaceDistanceProbe`, `TemperatureGrid` | Distance to surfaces; a temperature grid. |

The ray-casting sensors return a `(points, distances)` named tuple. `return_points=False` measures distances only
(`points` is then `None`), which is cheaper; see `examples/recipes/sensors/sensors_lidar_go2.py`. A non-colliding
background mesh is invisible to ray casts unless its material sets `use_visual_raycasting=True` before the build.

Camera *sensors* (`RasterizerCameraOptions`, `BatchRendererCameraOptions`, `NyxCameraOptions`) go on
`CameraOptions.sensor_options`, not in `SensorsOptions`, so they get the camera API. Eden's own audio sensors
(motor noise, contact sounds, a piano, microphones) are described in [Engine add-ons](engine-addons.md).

### Sensors as observations

`SensorRead` concatenates sensor readings into one observation: `sensor_names` (sensor names, regular expressions or
glob patterns; `None`, the default, for every sensor), `read_ground_truth` (default `False`) and an optional
`post_process_func` applied to each reading. A reading with more than two dimensions is flattened per environment,
and a named tuple contributes each of its tensors.

```python
from eden import observations

contacts = observations.SensorRead.configure(sensor_names=["contact_*"])
```

### When sensors update

By default (`EnvOptions.update_sensors_every_substep=False`) Genesis' sensor update runs only on the **last** scene
step of a control step: the intermediate readings would be overwritten before the observation manager reads them at
the end of `env.step()`, so updating them is wasted work. For ray-casting sensors it is a large saving; the code
comment in `EnvBase._scene_step_with_sensor_gating` records about 2.3x on a depth-camera motion-tracking task at
`decimation=4`.

The skip turns itself off when any sensor sets Genesis' `delay` or `history_length`, whose buffers need a fresh sample
every simulation step. Set `update_sensors_every_substep=True` to update on every step unconditionally. Eden's audio
sensors advance their per-step state outside this loop, so they keep time either way (see
[Engine add-ons](engine-addons.md)).

!!! warning "`scene.read_sensors()` raises in a scene with an Eden sensor"
    Genesis' `scene.read_sensors()` / `entity.read_sensors()` label results through `gs.sensors.types`, which cannot
    name a sensor defined outside Genesis. Read sensors through `env.sensors[name].read()` or `SensorRead` instead.

## The viewer and browser recordings

`show_viewer=True` (on `EnvBase` or `from_config`) opens Genesis' interactive viewer, which always rasterizes. These
`EnvOptions` fields change what it draws: `show_link_frame`, `show_world_frame`, `show_camera_frustums`, `show_FPS`
(all `False`), `enable_default_keybinds` (`True`) and `background_color`.

`eden.extensions.visualization.viser.ViserViewer` shows a running scene in the browser through
[Viser](https://viser.studio) (the `vis` extra: `pip install 'eden-world[vis]'` / `uv sync --inexact --extra vis`;
recording needs Viser 1.1.0 or newer). It can also record a trajectory into a file anyone can replay in a browser:

```python
from eden.extensions.visualization.viser import ViserViewer

viewer = ViserViewer(env, enable_gui=False)           # host="localhost", port=8080 by default
viewer.server.initial_camera.position = (2.5, -2.5, 1.5)
viewer.server.initial_camera.look_at = (0.0, 0.0, 0.3)
try:
    viewer.start_recording("trajectory.html")
    for _ in range(200):
        env.step()
        viewer.update()
    viewer.stop_recording()
finally:
    viewer.close()
```

- `start_recording(output_path, fps=None)` captures the scene at time zero. The path must end in `.html` (a
  standalone page with the player, which you can open directly or embed in an `<iframe>`) or `.viser` (the binary
  recording, for a Viser player you host). Each `update()` advances playback by `env.dt`, or by `1 / fps` when `fps`
  is given, independent of wall-clock speed. Call the recording methods and `update()` from the simulation thread.
- `stop_recording()` saves the file atomically and returns its path; a failed save ends the recording, since Viser's
  serializer is single-use. `close()` saves an active recording and stops the server; `is_recording` reports whether
  one is open.
- A recording keeps interactive camera control and trajectory playback, not GUI callbacks or a live simulation. It is
  held in memory until saved, and its size is dominated by the meshes, not by the number of frames.

`examples/recipes/rendering/viser_recording.py -c -s 200 -o go1.html` records a Go1 headless on the CPU;
`viser_visualization.py` in the same folder shows the live viewer.

## See also

- [Physics](physics.md): time stepping and solver settings, including the Mochi solver's renderer limits.
- [Observations](observations.md): image and sensor observation groups, noise and history.
- [Scenes and terrains](scenes-and-terrains.md): splat scenes and real-to-sim assets.
- [Engine add-ons](engine-addons.md): Eden's audio sensors and how out-of-tree sensors are hosted.
- API: [camera and sensor options](../api/options/sensors.md), [splat options](../api/options/splat.md),
  [`Camera` and overlays](../api/entities.md), [engine sensors](../api/engine.md),
  [visualization](../api/extensions/visualization.md).
