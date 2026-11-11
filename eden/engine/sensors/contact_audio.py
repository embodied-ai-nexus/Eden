# Portions Copyright 2026 the Tactile Genesis authors. Derived from Tactile Genesis
# (https://github.com/neuroagents-lab/tactile-genesis, commit 3023bcd), licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified by the Eden authors.
"""Contact microphone: the sound of a link touching, tapping and scraping the things around it.

A source-filter model. The *filter* is the struck object — a bank of damped modal oscillators whose frequencies
and decay rates are its material signature (metal rings long and high, wood thuds). The *source* is the contact:
a tap is an impulse into the bank, sliding is force- and speed-scaled noise shaped by the surface roughness, and a
hard tap adds a short high click. The timbre is keyed by the material of the link being *struck*, not the link
carrying the sensor: a fingertip sounds like whatever it touches.

Like :mod:`~eden.engine.sensors.actuation_audio` the synthesis lives in an
:class:`~eden.engine.sensors.audio.AudioSource`, so :class:`~eden.engine.sensors.audio.AudioContext` drives it on
every simulation step and any :class:`~eden.engine.sensors.spatial_audio.SpatialAudio` microphone hears it from
the sensor's link. ``read()`` exposes the same block as a structure-borne vibration signal.

Provenance
----------
Ported from ``neuroagents-lab/tactile-genesis`` @ ``3023bcd`` (``Genesis/genesis/engine/sensors/contact_audio.py``
and ``Genesis/genesis/options/sensors/audio.py``). The fused block kernel and the contact-driver math are upstream's.

Deviations from upstream:

- Synthesis runs from ``AudioContext.update`` rather than the sensor's own update hook. Eden skips sensor updates
  on intermediate decimation sub-steps, and a tap lasts one simulation step — synthesized from the sensor hook,
  three taps in four would never sound at ``decimation=4``.
- Materials are keyed by Eden scene names (``"table"``, ``"robot/hand"``) instead of solver link indices, which a
  declarative config cannot know; an unknown name raises at build. Two sensors giving one name different
  materials raise instead of the later one winning.
- Not ported: ``ContactAudioProperties.from_mesh`` and the strike-location mode-shape tables it fills (they need
  upstream's ``genesis.utils.modal_analysis``, which stock Genesis does not have), and the active-acoustic
  ``ExcitationSignal`` emitter.
- ``accel_noise_freq`` defaults to 800 Hz, not 5000 Hz, which the default 4 kHz audio rate cannot carry; a mode or
  click the rate cannot carry warns at build when it would have sounded.

Limitations: one dominant contact per sensor link drives the timbre; slip speed is taken between link origins,
ignoring the lever arm of a rotating body; radiation is a point source at the sensor link with no directivity.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import genesis as gs
import quadrants as qd
import torch
from genesis.engine.sensors.base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    SimpleSensor,
    SimpleSensorMetadata,
)
from genesis.engine.solvers.rigid.collider.collider import Collider
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.options.sensors.options import RigidSensorOptionsMixin, SimpleSensorOptions
from genesis.typing import PositiveInt
from genesis.utils.misc import concat_with_tensor, tensor_to_array
from pydantic import Field

import eden as en
from eden.engine._surface import require_members
from eden.engine.sensors.audio import AudioContext, AudioSource

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.sensors.sensor_manager import SensorManager
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.vis.rasterizer_context import RasterizerContext

require_members(
    __name__,
    {
        SimpleSensor: ("_update_raw_data", "_get_return_format", "_get_cache_dtype", "build", "_draw_debug"),
        RigidSensorMixin: ("_register_link", "set_pos_offset"),
        RigidSolver: ("get_links_vel", "get_links_ang"),
        Collider: ("get_contacts",),
    },
)

# Share of the sliding-texture drive injected into the modal bank rather than the broadband scrape resonator. The
# modal formants carry the material identity of a scrape, but the broadband resonator has to dominate for the slide
# to read as noise and not a tone, so this stays below 1.
_MODAL_NOISE_FRAC = 0.7

# Cutoff of the output DC-blocking high-pass, below the lowest audible mode. The texture resonator's center
# frequency collapses toward 0 Hz at low slip, so its noise drive integrates into a subsonic random walk that would
# swamp the audible signal (and a peak-normalized WAV); this removes that wander and nothing one can hear.
_DC_BLOCK_HZ = 30.0

# Fraction of Nyquist the audio rate is trusted to carry, the same margin the other audio sensors use.
_BAND_EDGE = 0.45


@qd.kernel
def _kernel_synthesize_block(
    # Per-(env, sensor, mode) drivers, constant over the block.
    a1: qd.types.ndarray(),  # (B, S, M) modal resonator feedback
    a2: qd.types.ndarray(),  # (B, S, M)
    gains: qd.types.ndarray(),  # (B, S, M) modal readout weight
    modal_drive: qd.types.ndarray(),  # (B, S) sliding noise injected into every mode
    modal_noise: qd.types.ndarray(),  # (B, S, M, K) white noise coloring the modal bank
    tex_noise: qd.types.ndarray(),  # (B, S, K) white noise driving the texture resonator
    impulse: qd.types.ndarray(),  # (B, S) tap, injected into every mode at block start
    # Per-(env, sensor) drivers: the texture (at*) and click (aa*) resonator feedback, and their drives.
    at1: qd.types.ndarray(),  # (B, S)
    at2: qd.types.ndarray(),  # (B, S)
    aa1: qd.types.ndarray(),  # (B, S)
    aa2: qd.types.ndarray(),  # (B, S)
    tex_amp: qd.types.ndarray(),  # (B, S)
    accel_kick: qd.types.ndarray(),  # (B, S)
    # Persistent state, updated in place. Two-pole filters keep (y1, y2) in the trailing dim; hp keeps (x1, y1).
    modal_y: qd.types.ndarray(),  # (B, S, M, 2)
    tex_y: qd.types.ndarray(),  # (B, S, 2)
    acc_y: qd.types.ndarray(),  # (B, S, 2)
    hp_state: qd.types.ndarray(),  # (B, S, 2)
    out: qd.types.ndarray(),  # (B, S, K)
    hp_r: float,
):
    """One lane per (env, sensor) runs the whole block: modal bank + texture + click, then the DC block.

    The coefficients change every step with the contact state, so unlike ``ActuationVoice`` there is no fixed
    block kernel to precompute; a fused kernel replaces what would be ~12 small tensor ops per sample. Noise is
    drawn by torch and passed in, where a per-sample Box-Muller inside the kernel would dominate.
    """
    B = out.shape[0]
    S = out.shape[1]
    K = out.shape[2]
    M = modal_y.shape[2]
    for i_s, i_b in qd.ndrange(S, B):
        drive = modal_drive[i_b, i_s]
        tap = impulse[i_b, i_s]
        for m in range(M):
            y1 = modal_y[i_b, i_s, m, 0]
            y2 = modal_y[i_b, i_s, m, 1]
            a1m = a1[i_b, i_s, m]
            a2m = a2[i_b, i_s, m]
            gm = gains[i_b, i_s, m]
            for k in range(K):
                y = a1m * y1 - a2m * y2 + drive * modal_noise[i_b, i_s, m, k]
                if k == 0:
                    y = y + tap
                y2 = y1
                y1 = y
                if m == 0:
                    out[i_b, i_s, k] = gm * y
                else:
                    out[i_b, i_s, k] = out[i_b, i_s, k] + gm * y
            modal_y[i_b, i_s, m, 0] = y1
            modal_y[i_b, i_s, m, 1] = y2

        ty1 = tex_y[i_b, i_s, 0]
        ty2 = tex_y[i_b, i_s, 1]
        ay1 = acc_y[i_b, i_s, 0]
        ay2 = acc_y[i_b, i_s, 1]
        hpx = hp_state[i_b, i_s, 0]
        hpy = hp_state[i_b, i_s, 1]
        for k in range(K):
            ty = at1[i_b, i_s] * ty1 - at2[i_b, i_s] * ty2 + tex_amp[i_b, i_s] * tex_noise[i_b, i_s, k]
            ty2 = ty1
            ty1 = ty
            ay = aa1[i_b, i_s] * ay1 - aa2[i_b, i_s] * ay2
            if k == 0:
                ay = ay + accel_kick[i_b, i_s]
            ay2 = ay1
            ay1 = ay
            raw = out[i_b, i_s, k] + ty + ay
            hpy = hp_r * hpy + raw - hpx
            hpx = raw
            out[i_b, i_s, k] = hpy
        tex_y[i_b, i_s, 0] = ty1
        tex_y[i_b, i_s, 1] = ty2
        acc_y[i_b, i_s, 0] = ay1
        acc_y[i_b, i_s, 1] = ay2
        hp_state[i_b, i_s, 0] = hpx
        hp_state[i_b, i_s, 1] = hpy


class ContactAudioProperties(NamedTuple):
    """How an object sounds when something touches it: its resonances, and the roughness of its surface.

    Metal-like materials have high modes that decay slowly (a long ring); wood-like ones have lower modes that
    decay fast, and a stronger roughness source.

    Parameters
    ----------
    modal_freqs : tuple[float, ...]
        Resonant mode frequencies in Hz. Fewer than the sensor's ``n_modes`` is zero-padded; more raises. A mode
        the audio rate cannot carry is silenced, with a warning at build.
    modal_decays : tuple[float, ...]
        Amplitude decay rate of each mode in 1/s. Large is a thud, small a long ring. Same length as
        ``modal_freqs``.
    modal_gains : tuple[float, ...]
        Output weight of each mode. Same length as ``modal_freqs``.
    roughness_gain : float
        Amplitude of the sliding-texture noise, scaled at runtime by normal force and slip speed. ``0`` is a
        smooth, silent slide.
    roughness_spatial_freq : float
        Surface bumps per meter. The scrape's pitch is ``roughness_spatial_freq * slip_speed`` Hz, so sliding
        faster raises it.
    roughness_bandwidth : float
        Spectral width in Hz of the scrape around that pitch. Several hundred Hz gives a noisy "shhh"; a narrow
        band an unrealistic tone.
    impact_gain : float
        Scale of the impulse a tap injects into the modal bank.
    impact_threshold : float
        Force jump in N per simulation step that counts as a tap. A tap fires only when the force crosses this
        from below, so the force ripple of a sustained slide does not re-ping the modes into a buzz.
    contact_damping : float
        Extra modal decay in 1/s *while in contact*: a finger pressing on an object deadens it, so the free
        ring-down is only heard after release.
    contact_damping_per_force : float
        Further in-contact decay per newton of normal force, in 1/(s·N): a firm press deadens the ring more than
        a light touch (Zheng & James 2011).
    accel_noise_gain : float
        Amplitude of the short high click a tap adds on top of the ring-down: the Hertzian contact transient of
        small hard objects, which the modal bank misses. ``0`` disables it.
    accel_noise_freq : float
        Center frequency in Hz of the click. Clamped to what the audio rate can carry, with a warning at build
        if the click is audible.
    accel_noise_decay : float
        Decay rate in 1/s of the click. Large is short and snappy.
    """

    modal_freqs: tuple[float, ...] = (250.0,)
    modal_decays: tuple[float, ...] = (40.0,)
    modal_gains: tuple[float, ...] = (1.0,)
    roughness_gain: float = 0.0
    roughness_spatial_freq: float = 0.0
    roughness_bandwidth: float = 600.0
    impact_gain: float = 1.0
    impact_threshold: float = 0.5
    contact_damping: float = 80.0
    contact_damping_per_force: float = 0.0
    accel_noise_gain: float = 0.0
    accel_noise_freq: float = 800.0
    accel_noise_decay: float = 800.0


def _material_links(scene_entities, key: str) -> list[int]:
    """Solver-global indices of the links a material key names, or raise naming what exists.

    Parameters
    ----------
    scene_entities : list
        The Genesis entities of the scene; Eden names each after its scene key.
    key : str
        ``"<entity>"`` for every link of that entity, or ``"<entity>/<link>"`` for one.
    """
    entity_name, _, link_name = key.partition("/")
    by_name = {entity.name: entity for entity in scene_entities if hasattr(entity, "links")}
    if entity_name not in by_name:
        gs.raise_exception(
            f"ContactAudio material '{key}' names entity '{entity_name}', which is not in the scene. Entities with "
            f"links: {sorted(by_name)}."
        )
    links = list(by_name[entity_name].links)
    if not link_name:
        return [link.idx for link in links]
    matching = [link.idx for link in links if link.name == link_name]
    if not matching:
        gs.raise_exception(
            f"ContactAudio material '{key}' names link '{link_name}', which entity '{entity_name}' does not have. "
            f"Its links: {[link.name for link in links]}."
        )
    return matching


class ContactAudio(RigidSensorOptionsMixin["ContactAudioSensor"], SimpleSensorOptions["ContactAudioSensor"]):
    """The sound of one link's contacts, keyed by the material of whatever it touches.

    ``read()`` returns ``(B, audio_substeps)``: this step's block of the structure-borne vibration at the link.
    The same block is registered with the scene's audio registry, radiating from the link (plus ``pos_offset``),
    so any :class:`~eden.engine.sensors.spatial_audio.SpatialAudio` microphone hears it.

    Parameters
    ----------
    entity_idx : int
        Global entity index of the body carrying the sensor. Required; Eden fills it from ``attach_entity_name``.
    link_idx_local : int
        Local index of the link whose contacts are voiced. Eden fills it from ``attach_link_name``.
    pos_offset : array-like[float, float, float]
        Where on the link the sound radiates from, in the link's frame.
    euler_offset : array-like[float, float, float]
        Accepted for parity with other link-attached sensors; a point source has no direction.
    properties : dict[str, ContactAudioProperties]
        Material of the things this link may touch, by Eden scene name: ``"table"`` covers every link of that
        entity, ``"robot/hand"`` one link. Merged across every ``ContactAudio`` sensor in the scene; an unknown
        name raises at build, and so does one name given two different materials.
    default_properties : ContactAudioProperties | None
        Material of every link ``properties`` does not name. ``None`` leaves contacts with them silent.
    audio_substeps : int
        Samples emitted per physics step. Must match every other audio participant in the scene; the sample rate
        is ``audio_substeps / sim_dt``. Real materials ring at kHz, so expect to raise this well above the default.
    n_modes : int
        Size of the modal bank. Must be the same on every ``ContactAudio`` sensor in the scene.
    velocity_gate_ref : float
        If positive, scale the output by ``motion / (motion + velocity_gate_ref)``, where ``motion`` is the speed
        of the sensor's own link in m/s: a body at rest goes quiet, which suppresses the clicks of a resting
        contact re-settling. ``0`` disables the gate.
    velocity_gate_ang_weight : float
        Meters per radian converting the link's angular speed into ``motion``, roughly the body's radius.
    velocity_gate_smooth : float
        One-pole smoothing in ``(0, 1]`` of the gate across simulation steps, so it cannot click by itself. ``1``
        is no smoothing.

    Examples
    --------
    A fingertip that hears what it touches:

    >>> from eden.options import SensorOptions
    >>> from eden.engine.sensors.contact_audio import ContactAudio, ContactAudioProperties
    >>> steel = ContactAudioProperties(modal_freqs=(1200.0, 2900.0), modal_decays=(6.0, 9.0), modal_gains=(1.0, 0.6))
    >>> finger = SensorOptions(
    ...     sensor=ContactAudio(audio_substeps=80, properties={"bowl": steel}),
    ...     attach_entity_name="robot",
    ...     attach_link_name="left_finger",
    ... )
    """

    properties: dict[str, ContactAudioProperties] = Field(default_factory=dict)
    default_properties: ContactAudioProperties | None = None
    audio_substeps: PositiveInt = 20
    n_modes: PositiveInt = 8
    velocity_gate_ref: float = Field(default=0.0, ge=0.0)
    velocity_gate_ang_weight: float = Field(default=0.0, ge=0.0)
    velocity_gate_smooth: float = Field(default=1.0, gt=0.0, le=1.0)

    def materials(self) -> dict[str, ContactAudioProperties]:
        """Every material this sensor declares, with the default under the key ``""``."""
        named = dict(self.properties)
        if self.default_properties is not None:
            named[""] = self.default_properties
        return named

    def model_post_init(self, context: Any) -> None:
        """Reject a material whose modal tuples disagree, or that has more modes than the bank.

        Parameters
        ----------
        context : Any
            Pydantic validation context, forwarded to the base implementation.
        """
        super().model_post_init(context)
        for key, props in self.materials().items():
            name = f"material '{key}'" if key else "default_properties"
            n_freqs, n_decays, n_gains = len(props.modal_freqs), len(props.modal_decays), len(props.modal_gains)
            if not n_freqs == n_decays == n_gains:
                gs.raise_exception(
                    f"ContactAudio {name}: modal_freqs, modal_decays and modal_gains must have equal length, got "
                    f"{n_freqs}, {n_decays}, {n_gains}."
                )
            if n_freqs > self.n_modes:
                gs.raise_exception(
                    f"ContactAudio {name} has {n_freqs} modes but n_modes={self.n_modes}, so the last "
                    f"{n_freqs - self.n_modes} would never sound. Raise `n_modes` or drop them."
                )

    def validate_scene(self, scene: "Scene"):
        """Require a link to listen on, and that every named material exists in the scene.

        Runs in ``scene.add_sensor``, after Eden has injected ``entity_idx`` / ``link_idx_local``. Entities are
        all added before any sensor, so a name that does not resolve here never will.

        Parameters
        ----------
        scene : Scene
            The scene the sensor is being added to.
        """
        if self.entity_idx < 0:
            gs.raise_exception(
                "ContactAudio needs the link whose contacts it voices, but entity_idx is still -1. Set "
                "`attach_entity_name` (and usually `attach_link_name`) on the enclosing eden SensorOptions."
            )
        super().validate_scene(scene)
        for key in self.properties:
            _material_links(scene.entities, key)


class ContactVoice(AudioSource):
    """Synthesis for every :class:`ContactAudioSensor` in the scene: one emission point per sensor.

    One voice for the whole class rather than one per sensor, so the contact list and link velocities are read
    once per step however many sensors exist. Sensors add themselves at build; the material table is rebuilt as
    their ``properties`` merge in.
    """

    # Per-(env, sensor) tensors: grown one sensor column at a time, and zeroed together on reset.
    _STATE = (
        "_modal_y",
        "_cur_a1",
        "_cur_a2",
        "_cur_gains",
        "_tex_y",
        "_acc_y",
        "_hp_state",
        "_prev_force",
        "_gate_state",
        "_block",
    )

    def __init__(
        self, solver: "RigidSolver", metadata: "ContactAudioSensorMetadata", n_modes: int, audio_substeps: int
    ):
        """Create an empty voice; sensors join through :meth:`add_sensor`.

        Parameters
        ----------
        solver : RigidSolver
            Solver to read contacts and link velocities from.
        metadata : ContactAudioSensorMetadata
            The sensor class's shared metadata, read for each sensor's link and radiation offset.
        n_modes : int
            Size of the modal bank.
        audio_substeps : int
            Samples per simulation step.
        """
        self._solver = solver
        self._metadata = metadata
        self._n_modes = n_modes
        self._K = audio_substeps
        self._dt_sub = solver._sim.dt / audio_substeps
        self._band_edge = _BAND_EDGE * 0.5 / self._dt_sub
        # One-pole DC-blocking high-pass: R = 1 - 2*pi*fc/fs.
        self._hp_r = max(0.0, 1.0 - 2.0 * torch.pi * _DC_BLOCK_HZ * self._dt_sub)

        self._materials: dict[str, ContactAudioProperties] = {}
        self._link_material = torch.full((solver.n_links,), -1, dtype=torch.long, device=gs.device)
        self._table: dict[str, torch.Tensor] = {}

        def state(*shape):
            return torch.zeros((solver._sim._B, 0, *shape), dtype=gs.tc_float, device=gs.device)

        # Persistent synthesis state, one column per sensor. `cur_*` latch the *free* coefficients of the last
        # material touched, so a released mode keeps ringing down with that material's own decay.
        self._modal_y = state(n_modes, 2)
        self._cur_a1, self._cur_a2, self._cur_gains = state(n_modes), state(n_modes), state(n_modes)
        self._tex_y, self._acc_y, self._hp_state = state(2), state(2), state(2)
        self._prev_force, self._gate_state = state(), state()
        self._block = state(audio_substeps)
        # Per-sensor velocity gate, (n_sensors,).
        self._gate_ref = torch.zeros((0,), dtype=gs.tc_float, device=gs.device)
        self._gate_ang_w = torch.zeros((0,), dtype=gs.tc_float, device=gs.device)
        self._gate_smooth = torch.zeros((0,), dtype=gs.tc_float, device=gs.device)
        self._has_gate = False

    def add_sensor(self, options: ContactAudio, scene_entities) -> None:
        """Grow every state tensor by one sensor column and merge in the sensor's materials.

        Parameters
        ----------
        options : ContactAudio
            The joining sensor's options.
        scene_entities : list
            The scene's Genesis entities, for resolving material names to links.
        """
        for key, props in options.materials().items():
            if self._materials.setdefault(key, props) != props:
                name = f"material '{key}'" if key else "default_properties"
                gs.raise_exception(
                    f"ContactAudio {name} is declared twice with different values. Materials are shared by every "
                    "ContactAudio sensor in the scene; declare each one once, or identically."
                )
        self._rebuild_material_table(scene_entities)

        for attr in self._STATE:
            tensor = getattr(self, attr)
            setattr(self, attr, torch.cat((tensor, tensor.new_zeros((tensor.shape[0], 1, *tensor.shape[2:]))), dim=1))
        self._gate_ref = concat_with_tensor(self._gate_ref, options.velocity_gate_ref)
        self._gate_ang_w = concat_with_tensor(self._gate_ang_w, options.velocity_gate_ang_weight)
        self._gate_smooth = concat_with_tensor(self._gate_smooth, options.velocity_gate_smooth)
        self._has_gate = self._has_gate or options.velocity_gate_ref > 0.0

    def _rebuild_material_table(self, scene_entities) -> None:
        """Pack ``self._materials`` into per-material rows and map each struck link to its row.

        Parameters
        ----------
        scene_entities : list
            The scene's Genesis entities, for resolving material names to links.
        """
        # The default (key "") sorts first, so it is row 0 whenever it exists.
        keys = sorted(self._materials)
        n_mat, n_modes = len(keys), self._n_modes

        def rows(*shape):
            return torch.zeros((n_mat, *shape), dtype=gs.tc_float, device=gs.device)

        table = {name: rows(n_modes) for name in ("freqs", "decays", "gains")}
        scalar_fields = {
            "rough_gain": "roughness_gain",
            "rough_sf": "roughness_spatial_freq",
            "rough_bw": "roughness_bandwidth",
            "impact_gain": "impact_gain",
            "impact_thresh": "impact_threshold",
            "contact_damping": "contact_damping",
            "damp_per_force": "contact_damping_per_force",
            "accel_gain": "accel_noise_gain",
            "accel_freq": "accel_noise_freq",
            "accel_decay": "accel_noise_decay",
        }
        table.update({name: rows() for name in scalar_fields})

        self._link_material[:] = 0 if "" in self._materials else -1
        for row, key in enumerate(keys):
            props = self._materials[key]
            n = len(props.modal_freqs)
            table["freqs"][row, :n] = torch.tensor(props.modal_freqs, dtype=gs.tc_float, device=gs.device)
            table["decays"][row, :n] = torch.tensor(props.modal_decays, dtype=gs.tc_float, device=gs.device)
            table["gains"][row, :n] = torch.tensor(props.modal_gains, dtype=gs.tc_float, device=gs.device)
            for name, attr in scalar_fields.items():
                table[name][row] = getattr(props, attr)
            if key:
                self._link_material[_material_links(scene_entities, key)] = row

            # A mode above the band edge would alias down to a phantom low tone, so it is silenced; a click is
            # clamped instead, since it is a burst rather than a pitch. Either way, say so if it would have sounded.
            label = f"material '{key}'" if key else "default_properties"
            lost = [f for f, g in zip(props.modal_freqs, props.modal_gains) if f >= self._band_edge and g != 0]
            if lost:
                en.logger.warning(
                    f"ContactAudio {label}: modes at {[round(f) for f in lost]} Hz are above the {self._band_edge:.0f} "
                    f"Hz the {1.0 / self._dt_sub:.0f} Hz audio rate can carry and are silenced. Raise "
                    "`audio_substeps` to hear them."
                )
            if props.accel_noise_freq > self._band_edge and props.accel_noise_gain != 0:
                en.logger.warning(
                    f"ContactAudio {label}: `accel_noise_freq` of {props.accel_noise_freq:.0f} Hz exceeds what the "
                    f"{1.0 / self._dt_sub:.0f} Hz audio rate can carry and is clamped to {self._band_edge:.0f} Hz. "
                    "Raise `audio_substeps` to keep it."
                )

        table["gains"][table["freqs"] >= self._band_edge] = 0.0
        table["accel_freq"].clamp_(max=self._band_edge)
        table["rough_bw"].clamp_(min=gs.EPS)
        table["accel_decay"].clamp_(min=gs.EPS)
        self._table = table

    @property
    def block(self) -> torch.Tensor:
        return self._block

    @property
    def emit_links(self) -> torch.Tensor:
        return self._metadata.links_idx

    @property
    def emit_offset(self) -> torch.Tensor:
        return self._metadata.offsets_pos

    @property
    def audio_substeps(self) -> int:
        return self._K

    def reset(self, envs_idx) -> None:
        """Silence the resonators and forget the last contact for ``envs_idx``.

        Parameters
        ----------
        envs_idx : array-like | None
            Environments to reset. ``None`` resets all of them.
        """
        idx = slice(None) if envs_idx is None else envs_idx
        for attr in self._STATE:
            getattr(self, attr)[idx] = 0.0

    def emit(self) -> None:
        """Synthesize this step's block from the contacts on each sensor link."""
        solver, table = self._solver, self._table
        B, n_sensors, n_modes = self._modal_y.shape[:3]
        if n_sensors == 0:
            return
        if not self._materials:
            # Every sensor has joined by the first emit, which runs inside env.build(), so this still fails at build.
            gs.raise_exception(
                "No ContactAudio sensor in the scene declares a material, so none of them could ever sound. Set "
                "`properties` (by Eden scene name) or `default_properties` on at least one."
            )
        K, dt_sub = self._K, self._dt_sub
        sensor_links = self._metadata.links_idx.long()  # (S,)

        # --- 1) Contact drivers, constant over the block, aggregated per sensor link.
        contacts = solver.collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
        force, normal = contacts["force"], contacts["normal"]  # (B, C, 3)
        link_a, link_b = contacts["link_a"].long(), contacts["link_b"].long()  # (B, C), -1 where padded
        n_contacts = link_a.shape[-1]

        zeros = torch.zeros((B, n_sensors), dtype=gs.tc_float, device=gs.device)
        if n_contacts == 0:
            # Nothing touching: no drive, and the modal bank rings down with its latched free coefficients.
            f_normal = zeros
            a1, a2, gains = self._cur_a1, self._cur_a2, self._cur_gains
            # Zero feedback lets the texture and click memory die at once; both are sub-millisecond anyway.
            impulse = modal_drive = tex_amp = accel_kick = at1 = at2 = aa1 = aa2 = zeros
        else:
            is_a = link_a[:, None, :] == sensor_links[None, :, None]  # (B, S, C)
            is_b = link_b[:, None, :] == sensor_links[None, :, None]
            involved = is_a | is_b
            any_contact = involved.any(dim=2)  # (B, S)

            # Total contact force on the sensor link (Newton's third law: +force when the sensor is link_b).
            orient = is_b.to(gs.tc_float) - is_a.to(gs.tc_float)
            f_normal = (orient[..., None] * force[:, None]).sum(dim=2).norm(dim=-1)  # (B, S)

            # The contact with the largest force speaks for the link: its counterpart is the struck material.
            primary = torch.where(involved, force.norm(dim=-1)[:, None], 0.0).argmax(dim=2)  # (B, S), into C
            other = torch.where(is_a, link_b[:, None, :], link_a[:, None, :])  # (B, S, C)
            struck = other.gather(2, primary[..., None]).squeeze(-1).clamp(min=0)  # (B, S)
            primary_normal = normal[:, None].expand(B, n_sensors, n_contacts, 3)
            primary_normal = primary_normal.gather(2, primary[..., None, None].expand(B, n_sensors, 1, 3)).squeeze(2)

            # Slip: the tangential part of the two link origins' relative velocity.
            links_vel = solver.get_links_vel()
            if solver.n_envs == 0:
                links_vel = links_vel[None]  # (B, n_links, 3)
            v_rel = links_vel[:, sensor_links] - links_vel.gather(1, struck[..., None].expand(B, n_sensors, 3))
            v_tan = v_rel - (v_rel * primary_normal).sum(dim=-1, keepdim=True) * primary_normal
            slip = v_tan.norm(dim=-1) * any_contact

            material = self._link_material[struck]  # (B, S), -1 where the struck link has no material
            valid = any_contact & (material >= 0)
            validf = valid.to(gs.tc_float)
            material = material.clamp(min=0)

            # Two coefficient sets per mode. While pressed, the finger damps the object (a floor plus a term per
            # newton), so a slide does not ring into a tone; the free decay alone is latched for after release.
            theta = 2.0 * torch.pi * table["freqs"][material] * dt_sub  # (B, S, M)
            decays = table["decays"][material]
            pressed = table["contact_damping"][material] + table["damp_per_force"][material] * f_normal
            free_r = torch.exp(-decays * dt_sub)
            pressed_r = torch.exp(-(decays + pressed[..., None]) * dt_sub)

            touching = valid[..., None]
            a1 = torch.where(touching, 2.0 * pressed_r * torch.cos(theta), self._cur_a1)
            a2 = torch.where(touching, pressed_r * pressed_r, self._cur_a2)
            gains = torch.where(touching, table["gains"][material], self._cur_gains)
            self._cur_a1.copy_(torch.where(touching, 2.0 * free_r * torch.cos(theta), self._cur_a1))
            self._cur_a2.copy_(torch.where(touching, free_r * free_r, self._cur_a2))
            self._cur_gains.copy_(gains)

            # A tap fires only when the force crosses the threshold from below. The solver's contact force ripples
            # during a slide; gating on the jump alone would inject an impulse every step, a buzz at the step rate.
            d_force = (f_normal - self._prev_force).clamp(min=0.0)
            threshold = table["impact_thresh"][material]
            tap = d_force * ((d_force > threshold) & (self._prev_force <= threshold)) * validf
            impulse = table["impact_gain"][material] * tap
            accel_kick = table["accel_gain"][material] * tap
            click_r = torch.exp(-table["accel_decay"][material] * dt_sub)
            aa1 = 2.0 * click_r * torch.cos(2.0 * torch.pi * table["accel_freq"][material] * dt_sub)
            aa2 = click_r * click_r

            # Sliding texture: a broadband scrape at the slip pitch, plus the same noise into the (damped) modal
            # bank, so the object's resonances color the scrape as formants instead of ringing as a tone.
            tex_amp = table["rough_gain"][material] * f_normal * slip * validf
            modal_drive = _MODAL_NOISE_FRAC * tex_amp
            tex_freq = (table["rough_sf"][material] * slip).clamp(max=self._band_edge)
            tex_r = torch.exp(-torch.pi * table["rough_bw"][material] * dt_sub)
            at1 = 2.0 * tex_r * torch.cos(2.0 * torch.pi * tex_freq * dt_sub)
            at2 = tex_r * tex_r

        # --- 2) The block itself, in one fused kernel over (env, sensor).
        tex_noise = torch.randn((B, n_sensors, K), dtype=gs.tc_float, device=gs.device)
        modal_noise = torch.randn((B, n_sensors, n_modes, K), dtype=gs.tc_float, device=gs.device)
        out = torch.empty((B, n_sensors, K), dtype=gs.tc_float, device=gs.device)
        _kernel_synthesize_block(
            a1.contiguous(),
            a2.contiguous(),
            gains.contiguous(),
            modal_drive.contiguous(),
            modal_noise,
            tex_noise,
            impulse.contiguous(),
            at1.contiguous(),
            at2.contiguous(),
            aa1.contiguous(),
            aa2.contiguous(),
            tex_amp.contiguous(),
            accel_kick.contiguous(),
            self._modal_y,
            self._tex_y,
            self._acc_y,
            self._hp_state,
            out,
            float(self._hp_r),
        )
        self._prev_force.copy_(f_normal)

        # --- 3) Optional gate on the sensor link's own motion, so a body at rest goes quiet.
        if self._has_gate:
            links_idx = self._metadata.links_idx
            links_vel, links_ang = solver.get_links_vel(links_idx), solver.get_links_ang(links_idx)
            if solver.n_envs == 0:
                links_vel, links_ang = links_vel[None], links_ang[None]
            motion = links_vel.norm(dim=-1) + self._gate_ang_w * links_ang.norm(dim=-1)  # (B, S)
            target = motion / (motion + self._gate_ref).clamp(min=gs.EPS)
            self._gate_state += self._gate_smooth * (target - self._gate_state)
            out = out * torch.where(self._gate_ref > 0.0, self._gate_state, 1.0)[..., None]

        self._block.copy_(out)


@dataclass
class ContactAudioSensorMetadata(RigidSensorMetadataMixin, SimpleSensorMetadata):
    """Per-class state of every :class:`ContactAudioSensor`: the link buckets, and the one voice they share."""

    voice: ContactVoice | None = None
    last_render: int = 0  # AudioContext.n_updates at the last read-cache refresh

    def destroy(self):
        """Drop the voice, releasing its synthesis buffers."""
        self.voice = None


class ContactAudioSensor(
    RigidSensorMixin[ContactAudioSensorMetadata], SimpleSensor[ContactAudio, AudioContext, ContactAudioSensorMetadata]
):
    """Joins the scene's :class:`ContactVoice` and reads back its own column of the block.

    The voice does the synthesis, driven once per step by
    :meth:`~eden.engine.sensors.audio.AudioContext.update`. This sensor only copies the result into its read cache.
    """

    def __init__(
        self,
        options: ContactAudio,
        idx: int,
        shared_context: AudioContext,
        shared_metadata: ContactAudioSensorMetadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self.debug_object: "Mesh | None" = None

    def build(self):
        """Create the class's voice if this is its first sensor, and add this sensor's column to it."""
        super().build()

        sm, context, options = self._shared_metadata, self._shared_context, self._options
        if sm.voice is None:
            sm.voice = ContactVoice(sm.solver, sm, options.n_modes, options.audio_substeps)
            context.register(sm.voice, type(options).__name__)
        else:
            context.require_audio_substeps(options.audio_substeps, type(options).__name__)
            if options.n_modes != sm.voice._n_modes:
                gs.raise_exception(
                    f"ContactAudio declares n_modes={options.n_modes} but the scene's other ContactAudio sensors use "
                    f"n_modes={sm.voice._n_modes}. They share one modal bank, so n_modes must agree."
                )
        sm.voice.add_sensor(options, self._manager._sim.entities)

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._options.audio_substeps,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_raw_data(
        cls, shared_context: AudioContext, shared_metadata: ContactAudioSensorMetadata, raw_data_T: torch.Tensor
    ):
        """Copy the voice's already-synthesized block into this class's read cache.

        Parameters
        ----------
        shared_context : AudioContext
            The registry that drove synthesis earlier this step; told about this refresh so a gapped ``read()``
            is reported.
        shared_metadata : ContactAudioSensorMetadata
            Holds the voice.
        raw_data_T : torch.Tensor
            Transposed write target, ``(n_sensors * audio_substeps, B)``.
        """
        shared_context.blocks_since(shared_metadata.last_render, ContactAudio.__name__)
        shared_metadata.last_render = shared_context.n_updates
        block = shared_metadata.voice.block  # (B, n_sensors, K)
        raw_data_T[:] = block.permute(1, 2, 0).reshape(-1, block.shape[0])

    def _draw_debug(self, context: "RasterizerContext"):
        """Draw a sphere at the link whose radius grows with the loudness of the last block.

        Parameters
        ----------
        context : RasterizerContext
            The viewer context to draw into.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        pos = tensor_to_array(self._link.get_pos(env_idx).reshape((3,)))
        amp = float(self.read(env_idx).reshape(-1).abs().max())

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None
        self.debug_object = context.draw_debug_sphere(
            pos=pos, radius=0.01 + min(amp, 1.0) * 0.05, color=(1.0, 0.3, 0.0, 0.6)
        )
