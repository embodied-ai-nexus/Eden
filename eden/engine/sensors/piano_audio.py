"""The sound of a piano keyboard: struck strings, voiced from the motion of the keys.

A keyboard entity (:class:`~eden.options.objects.piano.Piano`) is only hinges and springs. This sensor turns the
motion of those hinges into notes and publishes them to the scene's audio registry, so any
:class:`~eden.engine.sensors.spatial_audio.SpatialAudio` microphone hears the instrument, and ``read()`` returns
the same signal directly.

The model is a piano action reduced to its three audible events:

- **Strike.** When a key travels past ``strike_fraction`` of its stroke the hammer is thrown, and how loud the note
  is depends on how fast the key was moving at that moment, nothing else. A key pressed slowly enough makes no
  sound, as on the real instrument.
- **Ring.** The string is a bank of partials at ``n * f0 * sqrt(1 + inharmonicity * n**2)``, each an exponentially
  decaying sinusoid; higher partials and higher notes die faster. While the key stays down the damper is off the
  string and the note rings on.
- **Damp.** When the key comes back up past ``release_fraction`` the damper falls and the note is gone in a few
  tens of milliseconds, unless the sustain pedal (:meth:`PianoAudioSensor.set_sustain`) is down.

What a key is touched *by* plays no part, which is the difference from
:class:`~eden.engine.sensors.contact_audio.ContactAudio`: there the contact force is the excitation and staying
in contact damps the ring, the opposite of both on a piano.

Each partial is held as a rotating phasor rather than as a two-pole recurrence, so a block is linear in the
phasors: one matrix product against precomputed tables of decay envelope times sine and cosine, exact to rounding
at any pitch. The recurrence would do badly here: at 27.5 Hz and a one-second decay its two state samples differ
in the fifth digit, and float32 would turn that into noise. It also makes the damper free: lifting or dropping it
only swaps the envelope table.

RoboPianist, where the keyboard's dimensions come from, does not synthesize sound: it sends note-on/off events to
a MIDI synthesizer with every strike at full velocity. Its "key is down" test, within half a degree of the joint
limit, is about 87% of a white key's stroke, which is where ``strike_fraction`` defaults.

Limitation: parametric synthesis, not a sampled instrument. It is recognizably a struck string with the right
pitch, dynamics and damping, not a grand piano; amplitudes are arbitrary units.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

import genesis as gs
import numpy as np
import torch
from genesis.engine.entities.rigid_entity.rigid_joint import RigidJoint
from genesis.engine.sensors.base_sensor import SimpleSensor, SimpleSensorMetadata
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.options.sensors.options import RigidEntitySensorOptionsMixin, SimpleSensorOptions
from genesis.typing import PositiveFloat, PositiveInt, UnitInterval
from pydantic import Field

import eden as en
from eden.engine._surface import require_members
from eden.engine.sensors.audio import AudioContext, AudioSource

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.sensors.sensor_manager import SensorManager

require_members(
    __name__,
    {
        SimpleSensor: ("_update_raw_data", "_get_return_format", "_get_cache_dtype", "build"),
        RigidSolver: ("get_dofs_position", "get_dofs_limit"),
        RigidJoint: ("n_dofs", "dof_start", "link"),
    },
)


def _split_tf32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a float32 tensor into ``hi + lo``, where ``hi`` survives conversion to TF32 exactly.

    ``hi`` keeps the top 10 of float32's 23 mantissa bits and ``lo`` is the exact remainder, at most ``2**-10`` of
    ``x``. A TF32 product then loses nothing of ``hi`` and only rounding of ``lo``, so ``hi @ hi' + lo @ hi' +
    hi @ lo'`` is ``x @ x'`` to about float32 precision; the dropped ``lo @ lo'`` is ``2**-20`` of it.

    Parameters
    ----------
    x : torch.Tensor
        A float32 tensor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``hi`` and ``lo``.
    """
    hi = (x.view(torch.int32) & -(1 << 13)).view(torch.float32)
    return hi, x - hi


class PianoAudio(RigidEntitySensorOptionsMixin["PianoAudioSensor"], SimpleSensorOptions["PianoAudioSensor"]):
    """The sound of a keyboard entity, from the motion of its key joints.

    ``read()`` returns ``(B, audio_substeps)``: this step's block of the whole instrument. The same block is
    registered with the scene's audio registry and radiates from the keyboard's base link.

    Parameters
    ----------
    entity_idx : int
        Global entity index of the keyboard. Required.
    link_idx_local, pos_offset, euler_offset
        Inherited from Genesis' link-attached sensor options. ``pos_offset`` moves the point the sound radiates from,
        in the base link's frame; the other two are meaningless here and anything but the default raises at build.
    key_notes : dict[str, int] | None
        The keys to voice and what each plays: key joint name to MIDI note number. Its order is the column order of
        :attr:`PianoAudioSensor.keys_down`. ``None`` voices every single-DOF joint of the entity, in order, as a
        chromatic run from ``lowest_midi_note``, which is right for :class:`~eden.options.objects.piano.Piano`. To
        voice only some of its keys, name each with its own note, e.g.
        ``{key_joint_name(key): LOWEST_MIDI_NOTE + key for key in keys}``. An unknown name raises at build.
    lowest_midi_note : int
        MIDI note number of the entity's first key joint when ``key_notes`` is ``None``; ``21`` is the A0 of an
        88-key piano. Key ``i`` plays note ``lowest_midi_note + i``, in equal temperament like every note. Changing
        it alongside ``key_notes`` raises, since the notes are then given there.
    tuning : float
        Frequency of A4 (MIDI note 69), in Hz.
    strike_fraction : float
        Fraction of a key's stroke at which the hammer is thrown.
    release_fraction : float
        Fraction of the stroke below which a returning key drops its damper. Must be below ``strike_fraction``; the
        gap between them is what stops a key hovering at the strike point from re-striking.
    velocity_ref : float
        Key speed at the strike, in rad/s, that gives a full-scale note. Loudness is proportional to speed up to
        here and saturates above. A forte stroke takes a white key through its 10 mm in about 20 ms, 3 rad/s.
    n_partials : int
        Partials per string.
    inharmonicity : float
        String stiffness coefficient ``B``: partial ``n`` sits at ``n * f0 * sqrt(1 + B * n**2)``, slightly sharp
        of harmonic. ``0`` is an ideal string.
    partial_rolloff : float
        Partial ``n`` starts at amplitude ``n ** -partial_rolloff``. Larger is mellower.
    ring_decay : float
        Amplitude decay rate, in 1/s, of the fundamental of A4 while the damper is up.
    decay_pitch_exponent : float
        Decay rates scale with ``(f0 / tuning) ** decay_pitch_exponent``: a treble string dies sooner than a bass one.
    partial_decay_growth : float
        Partial ``n`` decays ``1 + partial_decay_growth * (n - 1)`` times faster than the fundamental, which is what
        makes a note mellow as it rings.
    damped_decay : float
        Decay rate, in 1/s, once the damper is down. Partials that already decay faster keep their own rate.
    audio_substeps : int
        Samples emitted per physics step. Must match every other audio participant in the scene. The default of 80
        is a 16 kHz rate at Eden's default ``sim_dt=0.005``, which carries every fundamental on the keyboard; the
        20 of the other audio sensors' defaults would leave everything above A5 mute.

    Examples
    --------
    >>> from eden.options import SensorOptions
    >>> from eden.engine.sensors.piano_audio import PianoAudio
    >>> voice = SensorOptions(sensor=PianoAudio(), attach_entity_name="piano")
    """

    key_notes: dict[str, Annotated[int, Field(ge=0, le=127)]] | None = None
    lowest_midi_note: int = Field(default=21, ge=0, le=127)
    tuning: PositiveFloat = 440.0
    strike_fraction: UnitInterval = 0.87
    release_fraction: UnitInterval = 0.5
    velocity_ref: PositiveFloat = 6.0
    n_partials: PositiveInt = 6
    inharmonicity: float = Field(default=4e-4, ge=0.0)
    partial_rolloff: float = Field(default=1.3, ge=0.0)
    ring_decay: PositiveFloat = 1.2
    decay_pitch_exponent: float = 0.6
    partial_decay_growth: float = Field(default=0.6, ge=0.0)
    damped_decay: PositiveFloat = 40.0
    audio_substeps: PositiveInt = 80

    def model_post_init(self, context: Any) -> None:
        """Reject thresholds that could never work, before anything is built.

        ``entity_idx`` is not checked here: Eden injects it during env build, so :meth:`validate_scene` does that.

        Parameters
        ----------
        context : Any
            Pydantic validation context, forwarded to the base implementation.
        """
        super().model_post_init(context)
        if not 0.0 < self.release_fraction < self.strike_fraction:
            gs.raise_exception(
                f"PianoAudio needs 0 < release_fraction < strike_fraction, got release_fraction="
                f"{self.release_fraction} and strike_fraction={self.strike_fraction}. A key re-arms only once it has "
                "come back up past the release point."
            )
        if self.key_notes is not None:
            if not self.key_notes:
                gs.raise_exception("PianoAudio was given an empty `key_notes`; pass None to voice every key joint.")
            # Compared against the default rather than checked for being set: a saved config writes every field back.
            if self.lowest_midi_note != type(self).model_fields["lowest_midi_note"].default:
                gs.raise_exception(
                    f"PianoAudio got both `key_notes` and lowest_midi_note={self.lowest_midi_note}, which would be "
                    "ignored: `key_notes` gives each key its own note. Drop lowest_midi_note."
                )

    def validate_scene(self, scene: "Scene"):
        """Require a real rigid entity, and refuse the link attachment this sensor would ignore.

        Parameters
        ----------
        scene : Scene
            The scene the sensor is being added to.
        """
        if self.entity_idx < 0:
            gs.raise_exception(
                "PianoAudio needs the keyboard entity it voices, but entity_idx is still -1. Set "
                "`attach_entity_name` on the enclosing eden SensorOptions (or entity_idx directly when adding the "
                "sensor through Genesis)."
            )
        super().validate_scene(scene)
        ignored = {name: getattr(self, name) for name in ("link_idx_local", "euler_offset")}
        set_anyway = {name: value for name, value in ignored.items() if any(v != 0 for v in np.atleast_1d(value))}
        if set_anyway:
            gs.raise_exception(
                f"PianoAudio voices a whole keyboard and radiates from its base link, but got {set_anyway}. Drop "
                "`attach_link_name` from the enclosing eden SensorOptions; use `pos_offset` to move the point the "
                "sound comes from."
            )


class PianoVoice(AudioSource):
    """Synthesis for one :class:`PianoAudioSensor`: every string of one keyboard, mixed to one emission point.

    Owns the state of each partial, a phasor ``(c, s)`` whose signal over the next block is
    ``envelope[k] * (c * sin((k + 1) * w) + s * cos((k + 1) * w))``, plus what it needs to tell a strike from a
    key that is merely down.
    """

    def __init__(
        self, options: PianoAudio, solver: "RigidSolver", dof_idx: list[int], notes: list[int], emit_link: int
    ):
        """Allocate the strings of the keys resolved by the owning sensor.

        Parameters
        ----------
        options : PianoAudio
            The owning sensor's options.
        solver : RigidSolver
            Solver to read the key positions from.
        dof_idx : list[int]
            Solver-global DOF index of each key.
        notes : list[int]
            MIDI note number each key plays, parallel to ``dof_idx``.
        emit_link : int
            Solver-global index of the link the instrument radiates from.
        """
        self._solver = solver
        sim = solver._sim
        B = sim._B
        K = options.audio_substeps
        n_keys = len(dof_idx)
        n_partials = options.n_partials
        self._K = K
        self._dt = sim.dt
        dt_sub = sim.dt / K
        # A sampled sine is exact at any pitch below Nyquist, so the strings run up to 90% of it. The other sources
        # stop at 45%, which their two-pole filters need and these oscillators do not.
        band_edge = 0.9 * 0.5 / dt_sub

        self._dof_idx = torch.tensor(dof_idx, dtype=gs.tc_int, device=gs.device)
        self._emit_links_t = torch.tensor([emit_link], dtype=gs.tc_int, device=gs.device)
        offset = torch.tensor(options.pos_offset, dtype=gs.tc_float, device=gs.device)
        self._emit_offset_t = offset.expand(B, 1, 3).contiguous()
        self._velocity_ref = options.velocity_ref

        lower, upper = solver.get_dofs_limit(self._dof_idx)
        lower, upper = lower.reshape(-1, n_keys)[0], upper.reshape(-1, n_keys)[0]
        unbounded = torch.nonzero(~torch.isfinite(upper - lower) | (upper <= lower)).flatten()
        if unbounded.numel() > 0:
            gs.raise_exception(
                f"PianoAudio: key {int(unbounded[0])} (DOF {dof_idx[int(unbounded[0])]}) has no finite joint range, so "
                "there is no stroke to place the strike point on. Every key joint needs a lower and an upper limit."
            )
        self._strike_at = lower + options.strike_fraction * (upper - lower)  # (n_keys,)
        self._release_at = lower + options.release_fraction * (upper - lower)

        # Everything below is built in float64 and only the finished tables are cast down.
        f64 = torch.float64
        notes_t = torch.tensor(notes, dtype=f64, device=gs.device)
        f0 = options.tuning * 2.0 ** ((notes_t - 69.0) / 12.0)  # (n_keys,)
        n = torch.arange(1, n_partials + 1, dtype=f64, device=gs.device)
        freq = f0[:, None] * n * torch.sqrt(1.0 + options.inharmonicity * n * n)  # (n_keys, n_partials)
        ring = (
            options.ring_decay
            * (f0[:, None] / options.tuning) ** options.decay_pitch_exponent
            * (1.0 + options.partial_decay_growth * (n - 1.0))
        )
        damped = ring.clamp(min=options.damped_decay)

        carried = freq < band_edge
        mute = ~carried[:, 0]
        if mute.any():
            en.logger.warning(
                f"PianoAudio: the {1.0 / dt_sub:.0f} Hz audio rate carries nothing above {band_edge:.0f} Hz, so "
                f"{int(mute.sum())} of {n_keys} keys are mute, from MIDI note {int(notes_t[mute].min())} "
                f"({float(f0[mute].min()):.0f} Hz) up. Raise `audio_substeps` to hear them."
            )
        # A partial the rate cannot carry is dropped rather than folded down to a wrong pitch.
        self._gain = (n**-options.partial_rolloff * carried).to(gs.tc_float)  # (n_keys, n_partials)

        steps = torch.arange(1, K + 1, dtype=f64, device=gs.device)
        phase = 2.0 * torch.pi * freq[..., None] * dt_sub * steps  # (n_keys, n_partials, K)
        oscillation = torch.stack((torch.sin(phase), torch.cos(phase)), dim=2)  # (n_keys, n_partials, 2, K)
        ring_env = torch.exp(-ring[..., None] * dt_sub * steps)
        damped_env = torch.exp(-damped[..., None] * dt_sub * steps)
        # Row (damper, key, partial, sin | cos) is the block one unit of that phasor component rings out under that
        # damper, so a block is the phasors times these tables; see `emit`.
        tables = torch.stack((ring_env[:, :, None] * oscillation, damped_env[:, :, None] * oscillation))
        tables = tables.reshape(-1, K).to(gs.tc_float)  # (2 * n_keys * n_partials * 2, K)
        # A float32 matmul on CUDA may run in TF32, which Eden enables by default (`eden.init(allow_tf32=True)`): 10
        # mantissa bits, a -70 dB error on every sample. Splitting both sides into a TF32-exact part and a remainder
        # and multiplying out three of the four products recovers float32 accuracy, whatever the global flag says.
        # With `_tables_lo` set, `_tables` holds only the TF32-exact part.
        self._tables, self._tables_lo = tables, None
        if gs.tc_float == torch.float32 and gs.device.type == "cuda":
            self._tables, self._tables_lo = _split_tf32(tables)
        # Carrying a phasor across a block: decay it by the envelope's last sample, turn it by the block's phase.
        self._ring_end = ring_env[..., -1].to(gs.tc_float)  # (n_keys, n_partials)
        self._damped_end = damped_env[..., -1].to(gs.tc_float)
        self._end_sin = torch.sin(phase[..., -1]).to(gs.tc_float)
        self._end_cos = torch.cos(phase[..., -1]).to(gs.tc_float)

        self._c = torch.zeros((B, n_keys, n_partials), dtype=gs.tc_float, device=gs.device)
        self._s = torch.zeros((B, n_keys, n_partials), dtype=gs.tc_float, device=gs.device)
        self._prev_q = torch.zeros((B, n_keys), dtype=gs.tc_float, device=gs.device)
        self._prev_speed = torch.zeros((B, n_keys), dtype=gs.tc_float, device=gs.device)
        self._down = torch.zeros((B, n_keys), dtype=torch.bool, device=gs.device)
        # An environment's first step after a reset only takes stock: a key already down was not just struck.
        self._fresh = torch.ones((B,), dtype=torch.bool, device=gs.device)
        self._sustain = torch.zeros((B,), dtype=torch.bool, device=gs.device)
        self._block = torch.zeros((B, 1, K), dtype=gs.tc_float, device=gs.device)
        # Every block since `start_recording`, or None while not recording. `_has_recorded` separates the two
        # ways of being None — never started, and started then stopped — so neither is reported as the other.
        self._recorded: list[torch.Tensor] | None = None
        self._has_recorded = False

    @property
    def block(self) -> torch.Tensor:
        return self._block

    @property
    def emit_links(self) -> torch.Tensor:
        return self._emit_links_t

    @property
    def emit_offset(self) -> torch.Tensor:
        return self._emit_offset_t

    @property
    def audio_substeps(self) -> int:
        return self._K

    @property
    def keys_down(self) -> torch.Tensor:
        """``(B, n_keys)`` bool: keys past their strike point and not yet back up past their release point.

        A copy, so it can be kept as the previous step's state: the voice overwrites its own every step.
        """
        return self._down.clone()

    def set_sustain(self, down, envs_idx=None) -> None:
        """Press or lift the sustain pedal.

        Parameters
        ----------
        down : bool | torch.Tensor
            Whether the pedal is down: one value, or one per selected environment.
        envs_idx : array-like | None
            Environments to set. ``None`` sets all of them.
        """
        idx = slice(None) if envs_idx is None else envs_idx
        self._sustain[idx] = torch.as_tensor(down, dtype=torch.bool, device=gs.device)

    def start_recording(self) -> None:
        """Start keeping every synthesized block.

        Refuses a second start while one is open, as :meth:`~eden.entities.camera.Camera.start_recording` does:
        the only thing it could do instead is throw away what the first one kept, and there is no getting that
        back.
        """
        if self._recorded is not None:
            gs.raise_exception(
                f"PianoAudio: already recording, with {len(self._recorded)} block(s) kept. Starting again would "
                f"drop them. Call take_recording() to collect them and go on recording, or stop_recording() to "
                f"throw them away on purpose."
            )
        self._recorded = []
        self._has_recorded = True

    def take_recording(self) -> torch.Tensor:
        """Return what was synthesized since the last call (or since recording started), and keep recording.

        Returns
        -------
        torch.Tensor
            ``(B, n_blocks * audio_substeps)`` samples, empty along the last axis if no step has run since.
        """
        if self._recorded is None:
            # Two different mistakes, and the fix differs: one call is missing, or one came too early.
            gs.raise_exception(
                "PianoAudio: take_recording() after stop_recording(), which dropped what had been kept. Take the "
                "samples before stopping, or leave the recording open."
                if self._has_recorded
                else "PianoAudio: take_recording() was called without start_recording()."
            )
        blocks, self._recorded = self._recorded, []
        if not blocks:
            return torch.zeros((self._block.shape[0], 0), dtype=gs.tc_float, device=gs.device)
        return torch.cat(blocks, dim=-1)

    def stop_recording(self) -> None:
        """Stop keeping blocks and drop what was kept.

        Refuses when no recording is open, as :meth:`~eden.entities.camera.Camera.stop_recording` does: a stop
        with nothing to stop is a mistake in the calling loop far more often than an intent.
        """
        if self._recorded is None:
            gs.raise_exception(
                "PianoAudio: not recording, so there is nothing to stop. start_recording() opens one; "
                "take_recording() collects what it holds without closing it."
            )
        self._recorded = None

    def reset(self, envs_idx) -> None:
        """Silence every string, lift the pedal and forget which keys were down, for ``envs_idx``.

        Parameters
        ----------
        envs_idx : array-like | None
            Environments to reset. ``None`` resets all of them.
        """
        idx = slice(None) if envs_idx is None else envs_idx
        for tensor in (self._c, self._s, self._prev_speed, self._block):
            tensor[idx] = 0.0
        self._down[idx] = False
        self._sustain[idx] = False
        self._fresh[idx] = True

    def emit(self) -> None:
        """Throw the hammers of the keys that just passed their strike point, then ring every string one block."""
        q = self._solver.get_dofs_position(self._dof_idx)
        if self._solver.n_envs == 0:
            q = q[None]  # (B, n_keys)

        fresh = self._fresh[:, None]
        # The key's speed as it passed the strike point, from finite differences rather than the joint velocity,
        # which reads zero once the key has met its stop. A fast key meets it within the very step that carried it
        # past the strike point, cutting that step's travel short, so the step before counts too.
        step_speed = torch.where(fresh, 0.0, (q - self._prev_q) / self._dt)
        speed = torch.maximum(step_speed, self._prev_speed)
        self._prev_speed.copy_(step_speed)
        down = (q >= self._strike_at) | (self._down & (q > self._release_at))
        struck = down & ~self._down & ~fresh
        level = (speed / self._velocity_ref).clamp(min=0.0, max=1.0) * struck  # (B, n_keys)
        self._down.copy_(down)
        self._prev_q.copy_(q)
        self._fresh.fill_(False)

        # A strike starts each partial at a zero crossing, so the note begins without a click.
        self._c += level[..., None] * self._gain

        damped = ~(down | self._sustain[:, None])  # (B, n_keys)
        # One matrix product with the tables: each phasor goes into the rows of the envelope it rings under this
        # block, and zeros into the other's. Nothing the size of (B, n_keys, P, K) is ever built.
        phasor = torch.stack((self._c, self._s), dim=-1)  # (B, n_keys, P, 2)
        ringing = torch.where(damped[..., None, None], 0.0, phasor)
        coefficients = torch.stack((ringing, phasor - ringing), dim=1).flatten(1)  # (B, 2 * n_keys * P * 2)
        if self._tables_lo is None:
            self._block[:, 0] = coefficients @ self._tables
        else:
            hi, lo = _split_tf32(coefficients)
            self._block[:, 0] = (hi @ self._tables).addmm_(lo, self._tables).addmm_(hi, self._tables_lo)

        # Carry each phasor to the end of the block: decay it, and turn it by the phase the block covered.
        end_env = torch.where(damped[..., None], self._damped_end, self._ring_end)  # (B, n_keys, P)
        c, s = self._c, self._s
        self._c = end_env * (c * self._end_cos - s * self._end_sin)
        self._s = end_env * (c * self._end_sin + s * self._end_cos)
        if self._recorded is not None:
            self._recorded.append(self._block[:, 0].clone())


@dataclass
class PianoAudioMetadata(SimpleSensorMetadata):
    """Per-class state of every :class:`PianoAudioSensor`: each one's voice and its offset in the read cache."""

    voices: list[PianoVoice] = field(default_factory=list)
    cache_offsets: list[int] = field(default_factory=list)
    last_render: int = 0  # AudioContext.n_updates at the last read-cache refresh

    def destroy(self):
        """Drop the voices, releasing their synthesis buffers."""
        self.voices.clear()
        self.cache_offsets.clear()


class PianoAudioSensor(SimpleSensor[PianoAudio, AudioContext, PianoAudioMetadata]):
    """Registers a :class:`PianoVoice` with the audio registry and reads back its block."""

    def __init__(
        self,
        options: PianoAudio,
        idx: int,
        shared_context: AudioContext,
        shared_metadata: PianoAudioMetadata,
        manager: "SensorManager",
    ):
        self._voice: PianoVoice | None = None
        super().__init__(options, idx, shared_context, shared_metadata, manager)

    def _resolve_keys(self) -> tuple[list[int], list[int], int]:
        """Map the key joints to solver-global DOF indices and MIDI notes, or fail naming the gap.

        Returns
        -------
        tuple[list[int], list[int], int]
            The DOF index of each key, the MIDI note it plays, and the index of the entity's base link.
        """
        # `PianoAudio.validate_scene` has already established that entity_idx names a rigid entity.
        entity = self._manager._sim.entities[self._options.entity_idx]
        hinges = [joint for joint in entity.joints if joint.n_dofs == 1]
        if not hinges:
            gs.raise_exception(
                f"PianoAudio on entity {self._options.entity_idx} found no single-DOF joints to use as keys."
            )

        key_notes = self._options.key_notes
        if key_notes is None:
            notes = [self._options.lowest_midi_note + i for i in range(len(hinges))]
            if notes[-1] > 127:
                gs.raise_exception(
                    f"PianoAudio: {len(hinges)} keys from lowest_midi_note={self._options.lowest_midi_note} run to "
                    f"MIDI note {notes[-1]}, past the top of the MIDI range (127)."
                )
        else:
            by_name = {joint.name: joint for joint in hinges}
            unknown = [name for name in key_notes if name not in by_name]
            if unknown:
                gs.raise_exception(
                    f"PianoAudio references key joints {unknown} that entity {self._options.entity_idx} does not "
                    f"have as single-DOF joints. Available: {sorted(by_name)}."
                )
            hinges = [by_name[name] for name in key_notes]
            notes = list(key_notes.values())
        return [joint.dof_start for joint in hinges], notes, entity.base_link.idx

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._options.audio_substeps,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    def build(self):
        """Build this sensor's voice and register it with the scene's audio registry."""
        super().build()

        dof_idx, notes, base_link = self._resolve_keys()
        self._voice = PianoVoice(self._options, self._manager._sim.rigid_solver, dof_idx, notes, base_link)
        self._shared_context.register(self._voice, type(self._options).__name__)

        sm = self._shared_metadata
        sm.voices.append(self._voice)
        sm.cache_offsets.append(self._cache_offset)

    @property
    def keys_down(self) -> torch.Tensor:
        """``(B, n_keys)`` bool: which keys are down, by the same test that throws the hammers.

        A copy, safe to keep across steps. Columns follow ``key_notes``, or the entity's joint order without it.
        """
        return self._voice.keys_down

    def set_sustain(self, down, envs_idx=None) -> None:
        """Press or lift the sustain pedal. While it is down, released keys keep ringing.

        Parameters
        ----------
        down : bool | torch.Tensor
            Whether the pedal is down: one value, or one per selected environment.
        envs_idx : array-like | None
            Environments to set. ``None`` sets all of them.
        """
        self._voice.set_sustain(down, envs_idx)

    def start_recording(self) -> None:
        """Start keeping every block the strings synthesize, one per *simulation* step.

        ``read()`` returns the block of the latest simulation step only, so a loop that steps the environment once
        per control step (``decimation`` simulation steps) reads one block in ``decimation``. A recording has them
        all: it is how to listen to a policy that acts at 20 Hz. The signal is the instrument's own, before any
        microphone's distance and delay.

        Unlike :meth:`~eden.entities.camera.Camera.start_recording`, which streams to a file, this holds what it
        keeps in memory for every environment at once — ``num_envs * audio_substeps`` floats a simulation step,
        65 MB/s at 1024 environments and the default 80 substeps. :meth:`take_recording` empties it, so a loop
        that drains each step holds one step's worth; one that never drains grows without bound. Starting a
        second recording while one is open raises rather than dropping what it holds, and a reset does not cut
        the recording: episodes are concatenated, at a different sample in each row.
        """
        self._voice.start_recording()

    def take_recording(self) -> torch.Tensor:
        """Return the samples synthesized since the last call, and keep recording.

        Returns
        -------
        torch.Tensor
            ``(B, n_samples)``; calling it once per ``env.step()`` yields ``decimation * audio_substeps`` samples
            each time, ready for :class:`~eden.engine.sensors.playback.LiveAudioPlayer`.
        """
        return self._voice.take_recording()

    def stop_recording(self) -> None:
        """Stop keeping blocks and drop what was kept, raising if no recording is open."""
        self._voice.stop_recording()

    @classmethod
    def _update_raw_data(
        cls, shared_context: AudioContext, shared_metadata: PianoAudioMetadata, raw_data_T: torch.Tensor
    ):
        """Copy each voice's already-synthesized block into this class's read cache.

        Parameters
        ----------
        shared_context : AudioContext
            The registry that drove synthesis earlier this step; told about this refresh so a gapped ``read()``
            is reported.
        shared_metadata : PianoAudioMetadata
            Holds the voices and their row offsets in the class cache.
        raw_data_T : torch.Tensor
            Transposed write target, ``(class_cache_size, B)``.
        """
        shared_context.blocks_since(shared_metadata.last_render, PianoAudio.__name__)
        shared_metadata.last_render = shared_context.n_updates
        for voice, offset in zip(shared_metadata.voices, shared_metadata.cache_offsets):
            block = voice.block[:, 0]  # (B, K)
            raw_data_T[offset : offset + block.shape[1]] = block.T
