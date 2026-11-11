# Engine add-ons

`eden/engine/` holds Genesis components that stock Genesis does not ship: the audio sensors and the MochiSolver.
They are Genesis code (they subclass Genesis engine classes and run inside `scene.step()`), but they live in Eden so
that Eden runs on the released Genesis pin. This page is for two readers: users of the audio sensors, and
contributors adding a sensor or a solver. It also says which components cannot live here and where they go instead.

`eden/engine/` is a plain package, not a framework: there is no registry, and `import eden` never imports it. A
sensor is installed by importing its module; a solver is installed on one scene at a time.

## Why not a Genesis fork

A forked Genesis is a second pin to keep in sync and a rebase every time upstream moves. Worse, it makes configs
unreadable without the fork: Eden saves an options class by its module path (`_option_module_` /
`_option_class_`), so an options class that lives in a Genesis fork makes every saved config that uses it unopenable
on stock Genesis. An add-on in `eden/engine/` is importable by anyone who has Eden, and a saved config re-imports its
module by itself through that same path.

Genesis is pinned to an exact release in `pyproject.toml`, so even an add-on that reaches into Genesis internals
reaches into a surface that only moves when someone bumps the pin deliberately, and the checks below catch it then.

## What can live here

Whether a component fits is decided by whether Genesis already has a seam where it belongs:

| Component | Genesis change needed | Where it goes |
|---|---|---|
| Sensor | none: `SensorManager.SENSOR_TYPES_MAP` is an open registry | `eden/engine/sensors/` |
| Recorder, viewer plugin | none: `@register_recording`, `Viewer.add_plugin` | `eden/engine/`, once the first one arrives |
| A new solver with its own materials, alone in its scene | `Simulator.__init__` builds a fixed list of solvers | `eden/engine/solvers/`, installed per scene |
| A constraint-solver flavor | `RigidSolver` hardcodes its constraint solver | a Genesis fork pin |
| A coupler | `Simulator.__init__` picks it from a closed `isinstance` chain | a Genesis fork pin |

Genesis has no entry-point or plugin discovery; the open registries above are its whole extension surface, and they
are what Genesis' own out-of-tree Nyx plugin (`gs_nyx_plugin`) uses.

**What stays out** is a component that must change a solver that other scenes use, or run beside one. A
constraint-solver flavor or a coupler replaces a Genesis attribute for every scene in the process. For those, pin a
Genesis fork that carries the component and keep only the Genesis-side code there: the options classes, the entity
wrappers and the validation stay in `eden/options` and `eden/entities`, so a saved config still names Eden modules.
Check what such a component gives up before adopting it: a faster constraint solver without static friction, for
example, silently breaks every grasp.

## Hosting a solver: the MochiSolver

A solver can be hosted one scene at a time. The simulator steps, resets and snapshots whatever is in its solver list,
so a solver appended to *one scene's* list after `gs.Scene()` runs. What the list does not carry is bound on that
scene's instances, never on a class, so a scene without the solver runs exactly the upstream code. The MochiSolver
(`eden/engine/solvers/mochi/hosting.py`) is the worked example. Users configure it as described in
[Physics](physics.md#mochi-the-eden-hosted-implicit-solver), and `EnvBase` installs it on the scene's first Mochi
entity.

`install_mochi_solver(scene, options)` must run before the scene is built and before any Mochi entity is added. Its
seams are the ones any hosted solver meets:

| Upstream assumes | Hosted as |
|---|---|
| `SceneOptions` declares every solver's options and fills their unset `dt` / `gravity` from `SimOptions` | the options class (`MochiOptions`) lives in Eden; `install_mochi_solver` copies the unset values from `scene.options.sim` |
| `Scene.add_entity` picks a surface's defaults from a closed `isinstance` chain on the material, and `Simulator._add_entity` gives a material to the solver declaring its most derived class | the scene's `add_entity` is bound to route Mochi materials to `add_mochi_entity`, which sets the defaults and calls the solver's `add_entity` |
| the rasterizer draws the solvers it names, and the visualizer refreshes their visual geoms | those methods are bound on the scene's rasterizer context and visualizer |
| `Simulator.destroy` frees the arrays of the solvers it can list | the scene's `sim.destroy` is bound to free the solver's arrays after it |
| `Simulator.reset` restores a solver only while it holds kinematic entities | the scene's `sim.reset` is bound to restore a solver of deformable bodies alone |
| the scene serializer names a material class by its bare `__name__` | a material extending an upstream one takes a name of its own (`MochiRigid`, not `Rigid`), or every rigid scene of the process would load back with it; `scene.export` is bound to raise, since `gs.Scene.load` rebuilds a scene without the hosted solver |

The ray tracer and the batch renderer have no such seam (their per-solver code is inline), so
`install_mochi_solver` refuses a scene that renders with either. Each binding is a seam upstream could open for any
third-party solver: a solver registry, material dispatch by method resolution order, surface defaults owned by the
material, renderers that iterate the active solvers, and a serializer that tells namesakes apart.

The Genesis internals a hosted solver subclasses or binds are declared with `eden.engine._surface.require_members`,
and their behaviour is pinned by `tests/engine/solvers/mochi/test_hosting.py`, so a Genesis pin bump fails there.

## Writing a sensor add-on

`Sensor.__init_subclass__` writes every subclass into `SensorManager.SENSOR_TYPES_MAP`, keyed on the options class
it reads off the generic parameterization, with no check on which package defined it. `scene.add_sensor(options)`
resolves through that map, so an Eden sensor class is reachable from Eden's own `SensorsOptions` as soon as its
module is imported:

```python
from genesis.engine.sensors.sensor_manager import SensorManager

from eden.engine.sensors import SpatialAudio, SpatialAudioSensor

assert SensorManager.SENSOR_TYPES_MAP[SpatialAudio] is SpatialAudioSensor  # no Genesis edit
```

!!! warning "`scene.read_sensors()` raises `AttributeError` in a scene with an out-of-tree sensor"
    The `gs.sensors.types.<Name>` tag namespace resolves names out of `genesis.options.sensors`, so it cannot see an
    Eden-side options class, and `scene.read_sensors()` / `entity.read_sensors()` label their results through it.
    They therefore raise `AttributeError` in any scene that holds an out-of-tree sensor. Nothing in Eden uses that
    path: `env.sensors[name].read()` and the `SensorRead` observation term go through `sensor.read()`, which works.
    `tests/engine/sensors/test_spatial_audio.py` pins the limitation, so a Genesis fix is noticed at the pin bump.

A sensor subclasses `SimpleSensor`, declaring its options, shared-context and metadata classes as type parameters.
Genesis reads them off `__orig_bases__`, so the parameterization is the registration:

```python
class MySensor(SimpleSensor[MyOptions, MyContext, MyMetadata]):
    def build(self): ...                      # resolve handles, grow per-class metadata
    def _get_return_format(self): ...         # read() shape; runs at construction
    @classmethod
    def _update_raw_data(cls, ctx, meta, raw_data_T): ...   # per step, batched over all instances
```

Five things to know before you write one:

- **`_get_return_format()` runs during `__init__`, not `build()`.** The read cache is sized from it, so anything the
  shape depends on has to be resolvable at construction. Eden creates entities before sensors, so resolving names
  against the scene is fine (see `ActuationAudioSensor._resolve_dofs`).
- **The per-step hooks are classmethods.** One call updates every instance of the class, which is why per-instance
  state goes in the shared *metadata* (grown one column per sensor in `build()`) and not on `self`.
- **`entity_idx` is not final at construction.** Eden injects it in `eden.entities.sensor.Sensor.pre_build`, from
  `SensorOptions.attach_entity_name`. A `model_post_init` validator that requires `entity_idx >= 0` would reject every
  declarative config. Validate it in the options' `validate_scene(scene)`, which `scene.add_sensor` calls after
  Eden's injection and before constructing the sensor, and name `attach_entity_name` in the error: that is the knob
  the caller set. Genesis' `RigidEntitySensorOptionsMixin` already does this for a sensor bound to a whole entity.
- **A link-attached sensor must use Genesis' attachment mixins.** Subclass `KinematicSensorMixin` (or
  `RigidSensorMixin`) and mix `KinematicSensorMetadataMixin` into the metadata, as `SpatialAudioSensor` does. Eden's
  `Sensor.get_pos()` / `get_quat()` / `set_pos_offset()` read the `offsets_pos` / `offsets_quat` those mixins
  maintain, so a sensor that resolves its own link breaks all three, and gets `pos_offset` wrong unless it rotates it
  into the link's frame. Conversely, an options base brings fields with it: if the sensor has no use for an inherited
  one (`ActuationAudio` ignores `link_idx_local`), refuse a non-default value in `validate_scene` rather than drop it.
- **Use a `SharedSensorContext` for anything scene-level**, as the next section explains.

### `SharedSensorContext`: the seam that avoids a fork

A `SharedSensorContext` is a resource shared across sensor *types*, and `SensorManager` gives it a full lifecycle:
constructed on first use, `update()`d once per simulation step **before** the per-type update loop, `reset()` before
the per-type reset, and `destroy()`ed on teardown.

That is the lifecycle a scene-level subsystem needs. The fork Eden's audio sensors come from keeps its audio
registry as a scene-level `AudioManager`, which takes three Genesis edits; hosted on a context, the same subsystem
takes none:

| The fork needs | A context gives you |
|---|---|
| `Simulator.__init__` constructs the manager | the sensor manager constructs the context on first use |
| `Scene.step` pumps it | `update()` once per step, before any sensor reads |
| `Scene.reset` clears it | `reset(envs_idx)` before the per-type reset |
| `Scene.add_audio_source` fills it | a producing sensor registers itself in `build()` |

The ordering is the valuable part: sources synthesize before microphones render, whatever order the sensors were
declared in. Synthesizing inside a sensor's own update hook would make correctness depend on declaration order.

**Anything that must advance every simulation step belongs on the context, not on a sensor.** Eden skips the per-type
update loop on the intermediate scene steps of a control step (`EnvOptions.update_sensors_every_substep=False`, the
default) but still calls `context.update()`. State advanced from a sensor hook would see one step in `decimation`;
for the audio delay line that would have silently shortened every propagation delay by the decimation factor.
`tests/engine/sensors/test_audio.py` pins this.

### Provenance, and what happens when the pin moves

State provenance in the module docstring: the upstream source and commit the component was ported from, every
deviation, and the limitation that kept it out of Genesis. A component with no traceable origin cannot be re-synced
when upstream moves, and one ported from a *closed* upstream pull request may have been closed for cause; that reason
has to reach whoever decides whether to use it.

Each module then declares the Genesis class members it overrides, calls or reads beyond what its imports already
guarantee:

```python
require_members(__name__, {SimpleSensor: ("_update_raw_data", "_get_return_format"), RigidJoint: ("dof_start",)})
```

`eden.engine._surface.require_members` raises at import, naming the module, every missing member and the installed
Genesis version. It checks *structure*, which catches a rename or a removal. It cannot check *behaviour*, such as
`SensorManager.step` still calling `context.update()` before the per-type loop; tests in
`tests/engine/sensors/test_audio.py` pin those contracts so that a pin bump that breaks one fails loudly instead of
silencing a scene. When you add a component that leans on a behavioural guarantee, add the test that would notice it
change.

## The audio sensors

Three stages, installed by `import eden.engine.sensors`. All but `PianoAudio`, which is Eden's own, are ported from
the [`neuroagents-lab/tactile-genesis`](https://github.com/neuroagents-lab/tactile-genesis) fork (commit `3023bcd`):

| Stage | Component | What it does |
|---|---|---|
| Synthesis | `ActuationAudio` | Per-DOF motor noise from joint torque and speed. |
| Synthesis | `ContactAudio` | Taps, ring-down and scraping of one link's contacts, voiced by the material of what it touches. |
| Synthesis | `PianoAudio` | The struck strings of a keyboard entity, voiced from the motion of its key joints. |
| Propagation | `AudioContext` | The source registry and the delay line carrying their sound. |
| Rendering | `SpatialAudio` | A microphone: distance attenuation, propagation delay, Doppler. |

The sensors are declared like any other (see [Sensors and rendering](sensors-and-rendering.md#sensors)), and every
microphone hears every registered source:

```python
from eden.engine.sensors import ActuationAudio, ActuationAudioProperties, SpatialAudio
from eden.options.sensors import SensorOptions, SensorsOptions

AUDIO_SUBSTEPS = 20  # samples per sim step: with sim_dt=0.005 that is a 4 kHz audio rate

sensors_options = SensorsOptions(
    motors=SensorOptions(
        sensor=ActuationAudio(
            audio_substeps=AUDIO_SUBSTEPS,
            default_properties=ActuationAudioProperties(pitch_slope=180.0, power_gain=0.05),
            properties={"joint1": ActuationAudioProperties(reversal_click_gain=0.3)},
        ),
        attach_entity_name="robot",
    ),
    mic_onboard=SensorOptions(
        sensor=SpatialAudio(audio_substeps=AUDIO_SUBSTEPS),
        attach_entity_name="robot",
        attach_link_name="hand",
    ),
    mic_room=SensorOptions(sensor=SpatialAudio(audio_substeps=AUDIO_SUBSTEPS, pos_offset=(2.0, 0.0, 0.5))),
)
```

`ActuationAudio.read()` returns `(B, n_dofs, audio_substeps)`, the per-joint signal, usable as an observation.
`SpatialAudio.read()` returns `(B, audio_substeps)`, the mixed signal at that microphone.

### Contact sounds

`ContactAudio` sits on the link that does the touching and takes its timbre from what is touched. Materials are keyed
by Eden scene name, `"bowl"` for every link of that entity or `"robot/hand"` for one, and merge across every
`ContactAudio` sensor of the scene:

```python
from eden.engine.sensors import ContactAudio, ContactAudioProperties
from eden.options.sensors import SensorOptions

steel = ContactAudioProperties(modal_freqs=(1180.0, 2050.0), modal_decays=(5.0, 8.0), modal_gains=(1.0, 0.7))
wood = ContactAudioProperties(
    modal_freqs=(310.0, 640.0),
    modal_decays=(90.0, 140.0),
    modal_gains=(1.0, 0.5),
    roughness_gain=0.3,
    roughness_spatial_freq=400.0,
)

fingertip = SensorOptions(
    sensor=ContactAudio(audio_substeps=80, properties={"bowl": steel, "table": wood}),
    attach_entity_name="robot",
    attach_link_name="left_finger",
)
```

`ContactAudio.read()` returns `(B, audio_substeps)`, the structure-borne vibration at that link.

### The piano

`PianoAudio` voices a keyboard entity, by default `eden.options.objects.piano.Piano`: 88 spring-loaded hinged keys
with the dimensions of [RoboPianist](https://github.com/google-research/robopianist)'s piano. The sensor reads only the
key joints. A key that travels past `strike_fraction` (default `0.87`) of its stroke throws a hammer, as loud as the
key was fast; the note rings while the key stays down and is damped once it comes back up past `release_fraction`
(default `0.5`), unless the sustain pedal is down. Whatever presses the keys, a robot hand or a joint torque, sounds
the same:

```python
from eden.engine.sensors import PianoAudio
from eden.options import SceneOptions
from eden.options.entities import PlaneOptions
from eden.options.objects.piano import Piano
from eden.options.sensors import SensorOptions

scene_options = SceneOptions(floor=PlaneOptions(), piano=Piano(default_root_pos=(0.0, 0.0, 0.7)))
strings = SensorOptions(sensor=PianoAudio(), attach_entity_name="piano")

# after env.build():
env.sensors["strings"].sensor.set_sustain(True)   # pedal down; pass envs_idx= for some environments
env.sensors["strings"].sensor.keys_down           # (B, 88) bool copy, the same test that throws the hammers
```

To voice only some keys, name each with the note it plays; a subset is never renumbered:

```python
from eden.options.objects.piano import LOWEST_MIDI_NOTE, key_joint_name

PianoAudio(key_notes={key_joint_name(key): LOWEST_MIDI_NOTE + key for key in (39, 43, 46)})  # C4 E4 G4
```

`PianoAudio.read()` returns `(B, audio_substeps)`, the whole instrument mixed, which also radiates from the
keyboard's base link. A pressed key dips 10 mm below the keyboard's underside, so the keyboard cannot rest on a floor:
`Piano` defaults to 2 cm of clearance, and any other placement must leave that gap too.

The `piano` task (`eden/tasks/piano/config.py`, `eden train --task piano --song twinkle_twinkle`) reads the key
joints directly and does not attach `PianoAudio`, since a reward needs only which keys are down. To hear a policy,
attach the sensor at evaluation time and record it. A control step spans `decimation` simulation steps and `read()`
returns only the last one's block, so `sensor.start_recording()` keeps every block and `sensor.take_recording()` hands
over what has accumulated; call it once per `env.step()`. `examples/projects/manipulation/piano/main.py play` does
exactly that.

!!! warning "Drain a piano recording every step"
    Unlike `Camera.start_recording`, which streams to a file, `PianoAudio` holds its blocks in memory for every
    environment until `take_recording()` empties them, so an undrained recording grows without bound. Starting a
    second recording while one is open raises rather than dropping what it holds, and so does stopping one that was
    never started.

### Things to get right

- **One sample rate per scene.** Every source and microphone must share `audio_substeps`; a disagreement raises at
  build, naming both values.
- **`max_delay` sizes the delay line.** A source farther than `max_delay * speed_of_sound` is heard at the clamped
  delay rather than its true one. The defaults (`max_delay=0.03` s, `speed_of_sound=343.0` m/s) reach about 10 m.
- **Continuous audio needs one render per block.** At Eden's default `decimation=4`, reading a microphone once per
  `env.step()` samples one block in four: fine as an observation, gapped as a recording. Use `decimation=1` (or
  `update_sensors_every_substep=True` plus a reader at the simulation rate) when writing to a file; the sensors warn
  once when they notice. Propagation delay, attenuation and Doppler are unaffected either way.
- **`ActuationAudio` covers a whole entity.** Each DOF radiates from its own joint's child link, so
  `attach_link_name` (and the inherited offsets) raise rather than being ignored; use `joints` to voice a subset.
- **Every partial stays below Nyquist.** A whine or hum fundamental saturates where its highest partial with a
  non-zero gain meets the audio rate's ceiling (0.45 x Nyquist: 900 Hz at the default `sim_dt=0.005` and
  `audio_substeps=20`), so a bank with fewer sounding partials reaches a higher pitch. A `friction_freq`,
  `click_freq` or `idle_freq` the rate cannot carry is clamped, with a build-time warning when its gain makes it
  audible. The defaults fit the default rate; raise `audio_substeps` for more headroom.
- **Register sources at build.** The delay line is sized on the first registry update that finds a radiating source,
  and `env.build()` already runs one, so a source declared through `SensorsOptions` is always in time. A
  hand-registered `AudioSource` is in time only when no declared source exists yet; a late `AudioContext.register`
  raises rather than being dropped.
- **A still joint is silent.** The whine fades out below 20 Hz: otherwise a joint holding a load has torque but no
  pitch, and `load_gain * |tau|` times its frozen oscillator is a DC offset. Use `idle_gain` for the hum of an
  energized motor at rest.
- **Contact materials need audio headroom.** Real objects ring at kilohertz, and a mode above the ceiling is silenced
  (with a warning) rather than aliased: at the default 4 kHz rate nothing above 900 Hz sounds. The contact recipe runs
  `audio_substeps=80` for a 16 kHz rate.
- **A tap lasts one simulation step**, which is why `ContactAudio` synthesizes from the registry's per-step update
  and not from its own sensor hook: at `decimation=4` the hook would miss three taps in four.
- **A scene where no `ContactAudio` declares a material raises at build**, since nothing could ever sound. A link with
  no material (and no `default_properties`) is silent by design.
- **A piano needs a 16 kHz rate**, which is why `PianoAudio` defaults to `audio_substeps=80` where the other sensors
  default to 20: the keyboard's fundamentals run to 4186 Hz. Its strings are exact sinusoids, so they run up to
  0.9 x Nyquist rather than the filters' 0.45; a partial above that is dropped, and the keys whose fundamental is
  dropped are reported at build. Put it in a scene with other audio sensors and they must use `audio_substeps=80`
  too.
- **How hard a key is hit is how fast it was moving** over the simulation step that carried it past the strike
  point, or the step before if that was faster, since a fast key meets its stop inside that step. A key eased down
  makes no sound. `velocity_ref` (default `6.0`) is the speed of a full-scale note.
- **Synthesis is block-vectorized.** `ActuationVoice.emit` renders a whole block per call, with the two-pole filters
  in closed form (`two_pole_block_kernels`), so the kernel count does not grow with `audio_substeps`. A piano block
  is one matrix product: the block is linear in the strings' phasors, so `PianoVoice.emit` multiplies them into
  precomputed envelope-times-sine tables. On CUDA the product is split so that TF32, which `eden.init` enables by
  default, costs no precision; unsplit it would put a -70 dB error on every sample.
- **The piano is not a sampled instrument.** It has the right pitch, dynamics and damping, and is recognizably a
  struck string rather than a grand piano; amplitudes are in arbitrary units. RoboPianist itself synthesizes nothing:
  it sends MIDI to FluidSynth.
- **Not ported from the fork's contact sensor:** `from_mesh` modal analysis with strike-location timbre, and the
  active-acoustic excitation emitter.
- **Free-field only.** No occlusion, reflections or room response: a microphone behind a wall hears the source as if
  the wall were not there.

### Hearing it live

`eden.engine.sensors.playback.LiveAudioPlayer` streams a sensor's blocks to the sound card while the simulation runs.
It needs no Python audio package: it pipes raw PCM into `aplay` or `ffplay`, whichever the machine has, and paces the
loop to real time so the sound stays with the viewer's picture:

```python
from eden.engine.sensors.playback import LiveAudioPlayer

with LiveAudioPlayer(sample_rate=int(AUDIO_SUBSTEPS / SIM_DT), gain=0.1) as speaker:
    for _ in range(steps):
        env.step()
        speaker.write(env.sensors["mic"].read()[0])
```

The sound trails the simulation by the `lead` cushion (default `0.06` s) plus the sound server's own latency;
`close()` reports how far it trailed. The players are told not to buffer ahead (a stock `aplay` would add about
0.7 s and `ffplay` 2 s), so pass a custom `command` only if it is equally well behaved.

A live stream cannot be peak-normalized, so `gain` is fixed up front and the signal is soft-limited. A simulation
slower than real time cannot play in sync: each time the sound card runs dry the player counts a dropout and rebuilds
its cushion, and it reports the count on close. Eden's viewer already paces `env.step()` to `sim_dt`, so a scene that
holds that rate plays cleanly with the viewer open. The player needs one `read()` per simulation step
(`decimation=1`).

Every audio recipe takes `--play` (live, best with `--vis`) and `--wav <file>`:

- `examples/recipes/sensors/engine_addon_spatial_audio.py` runs a Franka with three microphones.
- `examples/recipes/sensors/engine_addon_contact_audio.py` drops a ball on steel and another on wood, and slides a box
  down a ramp.
- `examples/recipes/sensors/engine_addon_piano.py` plays *Ode to Joy* by pushing the keys of a simulated piano.

## See also

- [Physics](physics.md): using the Mochi solver.
- [Sensors and rendering](sensors-and-rendering.md): declaring sensors, reading them, and when they update.
- [Engine API](../api/engine.md): every field of the audio sensors and the Mochi solver.
