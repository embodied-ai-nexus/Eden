# Entities

Entities are Eden's wrappers over Genesis entities: robots, rigid objects, deformables, terrain, cameras and
sensors. An env creates one per entry in its scene options and exposes them as `env.entities["<name>"]`. This page
is the class reference; the [entities guide](../guides/entities.md) covers pose frames, the two DOF index spaces,
armature, contacts and the other behaviour you need before writing a term against an entity.

```python
from eden.entities.rigid import RigidEntity
```

The class an entry gets is chosen from its options (rigid, particle, FEM, terrain, Mochi). Setters take
`envs_idx=` to touch a subset of environments, and quaternions are **wxyz** everywhere.

## Base and rigid entities

::: eden.entities.base.Entity
    options:
      show_bases: true

::: eden.entities.rigid.RigidEntity
    options:
      show_bases: true
      inherited_members: false

::: eden.entities.rigid
    options:
      show_root_heading: false
      members:
        - StagedStateWrite
        - resolve_link_ref_frame
        - LINK_REF_FRAMES

### Named DOF access

`find_named_dofs(names)` returns a `DofSelection`: hand its `.columns` to tensors indexed in the entity's
configured DOF order and its `.dofs_idx_local` to Genesis calls.

::: eden.entities.dofs_default_mixin
    options:
      show_root_heading: false
      members:
        - DofSelection
        - DofDefaultsMixin

### Support surfaces

The support-surface queries (`current_support` and the kernels behind it) that `RigidEntity.place_on_to` and the
placement events use. `RigidEntity` inherits them.

::: eden.entities.support_mixin.SupportSurfaceMixin
    options:
      show_bases: true

## Terrain

::: eden.entities.terrain.Terrain
    options:
      show_bases: true
      inherited_members: false

## Deformables

`set_pos()` and `set_quat()` raise on a `ParticleEntity` (MPM, SPH, PBD), and `set_quat()` raises on a
`FEMEntity`: bake the pose into the morph when you create the entity.

::: eden.entities.particle.ParticleEntity
    options:
      show_bases: true
      inherited_members: false

::: eden.entities.fem.FEMEntity
    options:
      show_bases: true
      inherited_members: false

## Mochi entities

Entities simulated by the Eden-hosted MochiSolver. A Mochi scene is all-Mochi; see [Engine](engine.md) and the
[physics guide](../guides/physics.md).

::: eden.entities.mochi.MochiRigidEntity
    options:
      show_bases: true
      inherited_members: false

::: eden.entities.mochi.MochiSoftEntity
    options:
      show_bases: true
      inherited_members: false

## Cameras

Record video with `Camera.start_recording(video_path, fps)` and `stop_recording()`, not with Genesis' own camera
recorder.

::: eden.entities.camera.Camera
    options:
      show_bases: true
      inherited_members: false

::: eden.entities.camera.SplatCamera
    options:
      show_bases: true
      inherited_members: false

::: eden.entities.camera.BatchSplatCamera
    options:
      show_bases: true
      inherited_members: false

### Camera overlays

Overlays paint state the scene does not render (ink, marks) or composite a background onto every RGB frame a camera
produces. Declare them in `CameraOptions.overlays`.

::: eden.entities.camera_overlays
    options:
      show_root_heading: false
      members:
        - CameraOverlay
        - OverlayBuffers
        - PlanePaintOverlay
        - PaintedPlane
        - PaintedPlaneSource
        - BackgroundComposite
        - BackgroundImageComposite
        - SplatComposite
        - build_camera_overlay
        - CAMERA_OVERLAY_REGISTRY

## Sensors

::: eden.entities.sensor.Sensor
    options:
      show_bases: true
      inherited_members: false

## See also

- [Entities guide](../guides/entities.md)
- [Entity options](options/entities.md), [robot presets](options/robots.md), [sensor and camera options](options/sensors.md)
- [Sensors and rendering guide](../guides/sensors-and-rendering.md)
