# Portions Copyright 2026 the Tactile Genesis authors. Derived from Tactile Genesis
# (https://github.com/neuroagents-lab/tactile-genesis, commit 3023bcd), licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified by the Eden authors.
"""Motor and gearbox noise: the synthesis stage of the audio pipeline.

Ported from the ``neuroagents-lab/tactile-genesis`` fork, where it is a standalone ``AudioSource``
added through a Genesis-side ``scene.add_audio_source``. Eden has no such API and does not want to
add one, so the source is wrapped in a sensor: the sensor declares itself the usual way through
``SensorsOptions``, hands its voice to :class:`~eden.engine.sensors.audio.AudioContext` at build, and
exposes the synthesized block through ``read()`` as a bonus — per-joint motor audio is a usable
observation in its own right, which the upstream source cannot provide.

The sound model is unchanged: each actuated DOF radiates from its joint's child link, voiced by
a velocity-pitched whine (gear mesh), an idle electrical hum, velocity-scaled bearing hiss, and a
click on direction reversal, with loudness driven by joint torque and mechanical power.

Provenance
----------
Ported from ``neuroagents-lab/tactile-genesis`` @ ``3023bcd``
(``Genesis/genesis/engine/audio/actuation_source.py``).

Deviations from upstream: a sensor rather than a standalone source (above); ``entity_idx`` is validated against
the scene rather than at options construction, since Eden injects it at build; every oscillator partial is kept
below Nyquist, where upstream bounds only the whine fundamental; a gain tuple longer than ``n_partials``
raises instead of being truncated; the whine fades out below 20 Hz, where upstream's frozen oscillators turn
the holding torque of a still joint into a DC offset; a block is synthesized in closed form rather than sample by
sample (same signal, to float rounding); and the ``friction_freq`` / ``friction_bandwidth`` / ``click_freq``
defaults are lowered from upstream's 1200 / 800 / 3000 Hz, which the default 4 kHz audio rate cannot carry.

Limitation: parametric synthesis, not a measured motor model. The timbre is plausible rather than calibrated to
any real actuator, so treat absolute amplitudes as arbitrary units.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import genesis as gs
import numpy as np
import torch
from genesis.engine.entities.rigid_entity.rigid_joint import RigidJoint
from genesis.engine.sensors.base_sensor import SimpleSensor, SimpleSensorMetadata
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.options.sensors.options import RigidEntitySensorOptionsMixin, SimpleSensorOptions
from genesis.typing import PositiveInt
from pydantic import Field

import eden as en
from eden.engine._surface import require_members
from eden.engine.sensors._dsp import bandpass_coeffs, resonator_coeffs, two_pole_block_kernels
from eden.engine.sensors.audio import AudioContext, AudioSource

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.sensors.sensor_manager import SensorManager

# Below this pitch a tone is not heard as one. The whine fades out under it; see `ActuationVoice.emit`.
_AUDIBLE_FLOOR_HZ = 20.0

require_members(
    __name__,
    {
        SimpleSensor: ("_update_raw_data", "_get_return_format", "_get_cache_dtype", "build"),
        RigidSolver: ("get_dofs_control_force", "get_dofs_velocity"),
        RigidJoint: ("n_dofs", "dof_start", "dof_end", "link"),
    },
)


class ActuationAudioProperties(NamedTuple):
    """Timbre of one joint's actuator: how a motor and gearbox sound as a function of torque and speed.

    Parameters
    ----------
    pitch_slope : float
        Whine fundamental in Hz per rad/s of joint speed, bundling gear ratio and mesh teeth. The whine
        pitch is ``pitch_slope * |omega|``.
    harmonic_gains : tuple[float, ...]
        Relative gains of the whine partials (fundamental, 2x, 3x, ...). Zero-padded to ``n_partials``; more
        gains than ``n_partials`` raises at build. The whine pitch saturates where the highest partial with a
        non-zero gain reaches the audio rate's ceiling, so fewer partials buy a higher top pitch.
    idle_freq : float
        Frequency in Hz of the idle electrical hum (PWM, magnetostriction).
    idle_gain : float
        Hum amplitude while the motor is energized, i.e. whenever ``|tau| > 0``. This is the *static*
        component: with torque roughly constant while a joint holds position it is a constant tone. Use
        ``idle_velocity_gain`` instead to keep a held joint quiet. ``0`` disables it.
    idle_velocity_gain : float
        Adds ``idle_velocity_gain * |omega|`` to the hum amplitude, so the hum's fixed-pitch lines are
        driven by motion rather than holding torque: quiet when still, louder when moving.
    idle_harmonic_gains : tuple[float, ...]
        Relative gains of the hum's partials at ``idle_freq``. A real servo hum is harmonic-rich, so
        something like ``(1.0, 0.6, 0.4)`` sounds buzzy rather than like a pure sine. Same length rule as
        ``harmonic_gains``.
    friction_gain : float
        Amplitude of the velocity-scaled bearing and friction hiss. ``0`` disables it.
    friction_freq : float
        Center frequency in Hz of the friction noise band. Like ``click_freq`` and ``idle_freq``, the default
        fits the default audio rate (``audio_substeps=20`` at Eden's default ``sim_dt=0.005``, a 900 Hz
        ceiling); a value the scene's rate cannot carry is clamped, with a warning at build if it is audible.
    friction_bandwidth : float
        Spectral width in Hz of the friction noise band.
    load_gain : float
        Whine loudness per unit ``|torque|`` (1/N·m): a loaded joint whines louder than a free-running one at the
        same speed. Like all of the whine it needs motion, fading out as the pitch drops below 20 Hz; a joint
        straining in place is voiced by ``idle_gain``.
    power_gain : float
        Loudness per unit mechanical power ``|tau*omega|`` (1/W): the working-whine component.
    reversal_click_gain : float
        Amplitude of the click emitted when joint velocity changes sign (backlash takeup), scaled by
        ``|omega|``. ``0`` disables it.
    click_freq : float
        Center frequency in Hz of the reversal click resonator.
    click_decay : float
        Decay rate in 1/s of the reversal click resonator.
    slew_coeff : float
        One-pole smoothing in ``(0, 1]`` applied to torque and velocity before synthesis, to suppress
        the controller's per-step ripple. ``1`` is no smoothing; smaller is heavier.
    """

    pitch_slope: float = 120.0
    harmonic_gains: tuple[float, ...] = (1.0, 0.5, 0.25)
    idle_freq: float = 120.0
    idle_gain: float = 0.0
    idle_velocity_gain: float = 0.0
    idle_harmonic_gains: tuple[float, ...] = (1.0,)
    friction_gain: float = 0.0
    friction_freq: float = 600.0
    friction_bandwidth: float = 400.0
    load_gain: float = 0.2
    power_gain: float = 0.1
    reversal_click_gain: float = 0.0
    click_freq: float = 800.0
    click_decay: float = 900.0
    slew_coeff: float = 0.3


class ActuationAudio(
    RigidEntitySensorOptionsMixin["ActuationAudioSensor"], SimpleSensorOptions["ActuationAudioSensor"]
):
    """Motor noise radiated by a robot's actuated joints, and readable as a per-joint signal.

    ``read()`` returns ``(B, n_dofs, audio_substeps)``: this step's block per covered DOF, in the order
    the joints appear on the entity. The same blocks are registered with the scene's audio registry, so
    any :class:`~eden.engine.sensors.spatial_audio.SpatialAudio` microphone hears them from the right
    place — each DOF radiates from its joint's child link.

    Parameters
    ----------
    entity_idx : int
        Global entity index of the actuated robot. Required; motor noise needs an actuator.
    link_idx_local, pos_offset, euler_offset
        Inherited from Genesis' link-attached sensor options and meaningless here: the sensor covers the whole
        entity and each DOF radiates from its own joint's child link. Anything but the default raises at build.
    joints : tuple[str, ...] | None
        Names of the joints to voice. ``None`` covers every joint of the entity that has DOFs. An
        unknown name raises at build rather than being ignored.
    default_properties : ActuationAudioProperties
        Timbre applied to every covered joint that ``properties`` does not override.
    properties : dict[str, ActuationAudioProperties]
        Per-joint-name timbre overrides.
    n_partials : int
        Number of partials in the whine and hum oscillator banks.
    audio_substeps : int
        Samples emitted per physics step. Must match every other audio participant in the scene.

    Examples
    --------
    Voice an arm's motors, with a louder whine on the shoulder:

    >>> from eden.options import SensorOptions
    >>> from eden.engine.sensors.actuation_audio import ActuationAudio, ActuationAudioProperties
    >>> motors = SensorOptions(
    ...     sensor=ActuationAudio(
    ...         audio_substeps=20,
    ...         properties={"shoulder_pan_joint": ActuationAudioProperties(power_gain=0.4)},
    ...     ),
    ...     attach_entity_name="robot",
    ... )
    """

    joints: tuple[str, ...] | None = None
    default_properties: ActuationAudioProperties = ActuationAudioProperties()
    properties: dict[str, ActuationAudioProperties] = Field(default_factory=dict)
    n_partials: PositiveInt = 3
    audio_substeps: PositiveInt = 20

    def model_post_init(self, context: Any) -> None:
        """Reject an empty joint selection, which would silently voice nothing.

        ``entity_idx`` is deliberately *not* checked here. Eden injects it during env build (see
        ``eden.entities.sensor.Sensor.pre_build``), so at construction time the declarative path
        legitimately still has the ``-1`` default; :meth:`validate_scene` enforces it once the value is real.

        Parameters
        ----------
        context : Any
            Pydantic validation context, forwarded to the base implementation.
        """
        super().model_post_init(context)
        if self.joints is not None and not self.joints:
            gs.raise_exception("ActuationAudio was given an empty `joints` tuple; pass None to cover every joint.")

    def validate_scene(self, scene: "Scene"):
        """Require a real rigid entity, and refuse the link attachment this sensor would ignore.

        Runs in ``scene.add_sensor`` before the sensor is constructed, which is after Eden has injected
        ``entity_idx`` / ``link_idx_local`` from ``attach_entity_name`` / ``attach_link_name``.

        Parameters
        ----------
        scene : Scene
            The scene the sensor is being added to.
        """
        if self.entity_idx < 0:
            gs.raise_exception(
                "ActuationAudio needs the entity whose motors it voices, but entity_idx is still -1. Set "
                "`attach_entity_name` on the enclosing eden SensorOptions (or entity_idx directly when adding the "
                "sensor through Genesis)."
            )
        super().validate_scene(scene)
        ignored = {name: getattr(self, name) for name in ("link_idx_local", "pos_offset", "euler_offset")}
        set_anyway = {name: value for name, value in ignored.items() if any(v != 0 for v in np.atleast_1d(value))}
        if set_anyway:
            gs.raise_exception(
                f"ActuationAudio covers a whole entity and ignores its link attachment, but got {set_anyway}. Each "
                "DOF radiates from its own joint's child link; drop `attach_link_name` from the enclosing eden "
                "SensorOptions and leave the offsets at their defaults. Use `joints` to voice a subset."
            )


class ActuationVoice(AudioSource):
    """Synthesis for one :class:`ActuationAudioSensor`: one emission point per covered DOF.

    Owns the oscillator phases and filter memory, which persist across steps, so it is built once and
    then driven by :meth:`emit` from the audio registry each step.
    """

    def __init__(self, options: ActuationAudio, solver: "RigidSolver", dof_idx, emit_links, properties):
        """Allocate the synthesis state for the DOFs resolved by the owning sensor.

        Parameters
        ----------
        options : ActuationAudio
            The owning sensor's options.
        solver : RigidSolver
            Solver to read per-DOF torque and velocity from.
        dof_idx : list[int]
            Solver-global DOF indices to voice, in read order.
        emit_links : list[int]
            Solver-global index of the link each DOF radiates from, parallel to ``dof_idx``.
        properties : list[ActuationAudioProperties]
            Timbre of each DOF, parallel to ``dof_idx``.
        """
        self._options = options
        self._solver = solver
        self._sim = solver._sim

        n_emit = len(dof_idx)
        n_partials = options.n_partials
        B = self._sim._B
        K = options.audio_substeps
        self._K = K
        self._dt_sub = self._sim.dt / K
        # Keep every synthesized partial below Nyquist for the audio rate, with margin, so a fast joint
        # saturates into a scream instead of silently folding down into a low rumble.
        self._nyquist = 0.45 * 0.5 / self._dt_sub
        self._harm_mult = torch.arange(1, n_partials + 1, dtype=gs.tc_float, device=gs.device)

        self._dof_idx = torch.tensor(dof_idx, dtype=gs.tc_int, device=gs.device)
        self._emit_links_t = torch.tensor(emit_links, dtype=gs.tc_int, device=gs.device)
        self._emit_offset_t = torch.zeros((B, n_emit, 3), dtype=gs.tc_float, device=gs.device)

        def col(attr):
            return torch.tensor([getattr(p, attr) for p in properties], dtype=gs.tc_float, device=gs.device)

        def bank(attr):
            gains_t = torch.zeros((n_emit, n_partials), dtype=gs.tc_float, device=gs.device)
            for i, props in enumerate(properties):
                gains = getattr(props, attr)
                if len(gains) > n_partials:
                    gs.raise_exception(
                        f"ActuationAudio: `{attr}` has {len(gains)} gains but n_partials={n_partials}, so the "
                        f"last {len(gains) - n_partials} would never sound. Raise `n_partials` or drop them."
                    )
                gains_t[i, : len(gains)] = torch.tensor(gains, dtype=gs.tc_float, device=gs.device)
            return gains_t

        def max_fundamental(gains_t):
            # Partial p sits at p times its fundamental, so a bank's fundamental is bounded by the ceiling over
            # the multiple of its highest partial that actually sounds.
            top_multiple = ((gains_t != 0) * self._harm_mult).amax(dim=-1)
            return self._nyquist / top_multiple.clamp(min=1.0)

        def col_below(attr, ceiling, audible):
            values = col(attr)
            ceiling = torch.as_tensor(ceiling, dtype=gs.tc_float, device=gs.device).expand_as(values)
            # A band whose gain is zero never sounds, so clamping it changes nothing worth a warning.
            clamped_audibly = torch.nonzero((values > ceiling) & audible).flatten()
            if clamped_audibly.numel() > 0:
                i = int(clamped_audibly[0])
                en.logger.warning(
                    f"ActuationAudio: `{attr}` of {float(values[i]):.0f} Hz exceeds what the "
                    f"{1.0 / self._dt_sub:.0f} Hz audio rate can carry and is clamped to {float(ceiling[i]):.0f} Hz. "
                    "Raise `audio_substeps` to keep it."
                )
            return torch.minimum(values, ceiling)

        self._harm = bank("harmonic_gains")  # (n_emit, n_partials)
        self._idle_harm = bank("idle_harmonic_gains")  # (n_emit, n_partials)
        self._max_whine_f0 = max_fundamental(self._harm)  # (n_emit,)

        self._pitch_slope = col("pitch_slope")
        self._idle_gain = col("idle_gain")
        self._idle_velocity_gain = col("idle_velocity_gain")
        self._idle_freq = col_below(
            "idle_freq", max_fundamental(self._idle_harm), (self._idle_gain != 0) | (self._idle_velocity_gain != 0)
        )
        self._friction_gain = col("friction_gain")
        self._friction_freq = col_below("friction_freq", self._nyquist, self._friction_gain != 0)
        self._friction_bw = col("friction_bandwidth").clamp(min=gs.EPS)
        self._load_gain = col("load_gain")
        self._power_gain = col("power_gain")
        self._reversal_click_gain = col("reversal_click_gain")
        self._click_freq = col_below("click_freq", self._nyquist, self._reversal_click_gain != 0)
        self._click_decay = col("click_decay").clamp(min=gs.EPS)
        self._slew = col("slew_coeff")

        # The friction band-pass and the click resonator are two-pole recurrences with constant coefficients,
        # so each block is filtered in closed form rather than sample by sample.
        self._friction_free, self._friction_forced = two_pole_block_kernels(
            *bandpass_coeffs(self._friction_freq, self._friction_bw, self._dt_sub), K
        )
        self._click_free, self._click_forced = two_pole_block_kernels(
            *resonator_coeffs(self._click_freq, self._click_decay, self._dt_sub), K
        )
        # Sample k of a block is k + 1 oscillator steps after the phase the previous block ended on.
        self._osc_steps = torch.arange(1, K + 1, dtype=gs.tc_float, device=gs.device)
        self._idle_dphase = 2.0 * torch.pi * self._idle_freq.unsqueeze(-1) * self._harm_mult * self._dt_sub

        # Persistent synthesis state: oscillator phases, band-pass and resonator memory, slew filters.
        self._phase = torch.zeros((B, n_emit, n_partials), dtype=gs.tc_float, device=gs.device)
        self._idle_phase = torch.zeros((B, n_emit, n_partials), dtype=gs.tc_float, device=gs.device)
        self._ty1 = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._ty2 = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._ay1 = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._ay2 = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._prev_omega = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._slew_tau = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._slew_omega = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._block = torch.zeros((B, n_emit, K), dtype=gs.tc_float, device=gs.device)

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

    def reset(self, envs_idx) -> None:
        """Zero the oscillator phases, filter memory, and last block for ``envs_idx``.

        Parameters
        ----------
        envs_idx : array-like | None
            Environments to reset. ``None`` resets all of them.
        """
        idx = slice(None) if envs_idx is None else envs_idx
        for tensor in (
            self._phase,
            self._idle_phase,
            self._ty1,
            self._ty2,
            self._ay1,
            self._ay2,
            self._prev_omega,
            self._slew_tau,
            self._slew_omega,
            self._block,
        ):
            tensor[idx] = 0.0

    def emit(self) -> None:
        """Synthesize this step's block from the current per-DOF torque and velocity."""
        solver = self._solver
        if self._block.numel() == 0:
            return
        B, n_emit, K = self._block.shape
        dt_sub = self._dt_sub

        tau = solver.get_dofs_control_force(self._dof_idx)
        omega = solver.get_dofs_velocity(self._dof_idx)
        if solver.n_envs == 0:
            tau, omega = tau[None], omega[None]  # (B, n_emit)

        # One-pole slew limit denoises the controller's per-step torque and velocity ripple, which would
        # otherwise be audible as a buzz at the control rate rather than as motor noise.
        self._slew_tau += self._slew * (tau - self._slew_tau)
        self._slew_omega += self._slew * (omega - self._slew_omega)
        tau_f, omega_f = self._slew_tau, self._slew_omega

        load_amp = self._load_gain * tau_f.abs() + self._power_gain * (tau_f * omega_f).abs()  # (B, n_emit)
        f0 = torch.minimum(self._pitch_slope * omega_f.abs(), self._max_whine_f0)
        # A joint holding still under load has torque but no pitch: its oscillators freeze mid-cycle, and
        # `load_gain * |tau|` times a frozen sine is a DC offset that steps whenever the torque does — a thump, and
        # headroom lost to something inaudible. Gear mesh only sounds while the gears turn, so fade the whine out
        # as its pitch drops below hearing.
        load_amp = load_amp * (f0 / _AUDIBLE_FLOOR_HZ).clamp(max=1.0)
        # Static (torque-on) hum floor plus a motion-driven term, so the hum tracks joint speed instead
        # of droning constantly while the joint merely holds position.
        idle_amp = self._idle_gain * (tau_f.abs() > 1e-3).to(gs.tc_float) + self._idle_velocity_gain * omega_f.abs()
        friction_amp = self._friction_gain * omega_f.abs()
        # Reversal click: velocity changed sign since last step, i.e. the gear backlash took up.
        sign_flip = (torch.sign(omega_f) * torch.sign(self._prev_omega)) < 0.0
        click_kick = self._reversal_click_gain * omega_f.abs() * sign_flip.to(gs.tc_float)

        # Oscillator banks: every sample's phase at once, (B, n_emit, n_partials, K), summed over partials.
        dphase = 2.0 * torch.pi * f0.unsqueeze(-1) * self._harm_mult * dt_sub  # (B, n_emit, n_partials)
        whine_phase = self._phase.unsqueeze(-1) + dphase.unsqueeze(-1) * self._osc_steps
        idle_phase = self._idle_phase.unsqueeze(-1) + self._idle_dphase.unsqueeze(-1) * self._osc_steps
        whine = load_amp.unsqueeze(-1) * (self._harm.unsqueeze(-1) * torch.sin(whine_phase)).sum(dim=2)
        idle = idle_amp.unsqueeze(-1) * (self._idle_harm.unsqueeze(-1) * torch.sin(idle_phase)).sum(dim=2)

        # Two-pole filters in closed form (see `two_pole_block_kernels`): the memory ringing on, plus the
        # response to this block's drive. Friction is driven by noise on every sample, the click by one kick
        # on the first.
        noise = torch.randn((B, n_emit, K), dtype=gs.tc_float, device=gs.device)
        friction_memory = torch.stack((self._ty1, self._ty2), dim=-1)  # (B, n_emit, 2)
        friction = torch.einsum("ekm,bem->bek", self._friction_free, friction_memory) + torch.einsum(
            "ekj,bej->bek", self._friction_forced, friction_amp.unsqueeze(-1) * noise
        )
        click_memory = torch.stack((self._ay1, self._ay2), dim=-1)
        click = torch.einsum("ekm,bem->bek", self._click_free, click_memory) + (
            self._click_forced[:, :, 0] * click_kick.unsqueeze(-1)
        )

        self._phase.copy_(whine_phase[..., -1] % (2.0 * torch.pi))
        self._idle_phase.copy_(idle_phase[..., -1] % (2.0 * torch.pi))
        # The filters' memory is the block's last two outputs; a one-sample block keeps one sample of the old.
        friction_tail = torch.cat((self._ty1.unsqueeze(-1), friction), dim=-1)
        click_tail = torch.cat((self._ay1.unsqueeze(-1), click), dim=-1)
        self._ty1.copy_(friction_tail[..., -1])
        self._ty2.copy_(friction_tail[..., -2])
        self._ay1.copy_(click_tail[..., -1])
        self._ay2.copy_(click_tail[..., -2])
        self._prev_omega.copy_(omega_f)
        self._block.copy_(whine + idle + friction + click)


@dataclass
class ActuationAudioMetadata(SimpleSensorMetadata):
    """Per-class state of every :class:`ActuationAudioSensor`.

    Each sensor contributes its voice and the offset of its rows in the class read cache, so the
    per-step classmethod hook can copy every voice's block without reaching back to instances.
    """

    voices: list[ActuationVoice] = field(default_factory=list)
    cache_offsets: list[int] = field(default_factory=list)
    last_render: int = 0  # AudioContext.n_updates at the last read-cache refresh

    def destroy(self):
        """Drop the voices, releasing their synthesis buffers."""
        self.voices.clear()
        self.cache_offsets.clear()


class ActuationAudioSensor(SimpleSensor[ActuationAudio, AudioContext, ActuationAudioMetadata]):
    """Registers an :class:`ActuationVoice` with the audio registry and reads back its block.

    The voice does the synthesis, driven once per step by
    :meth:`~eden.engine.sensors.audio.AudioContext.update`. This sensor only copies the result into its
    read cache, which is what makes the block available as an observation without synthesizing twice.
    """

    def __init__(
        self,
        options: ActuationAudio,
        idx: int,
        shared_context: AudioContext,
        shared_metadata: ActuationAudioMetadata,
        manager: "SensorManager",
    ):
        # Resolved here rather than in `build` because `_get_return_format` runs during construction and
        # the read-cache size depends on how many DOFs the options actually cover.
        self._dof_idx: list[int] = []
        self._emit_links: list[int] = []
        self._properties: list[ActuationAudioProperties] = []
        self._voice: ActuationVoice | None = None
        super().__init__(options, idx, shared_context, shared_metadata, manager)

    def _resolve_dofs(self) -> None:
        """Map the requested joint names to solver-global DOF and link indices, or fail naming the gap."""
        # `ActuationAudio.validate_scene` has already established that entity_idx names a rigid entity.
        entity = self._manager._sim.entities[self._options.entity_idx]
        joints = list(entity.joints)

        wanted = self._options.joints
        if wanted is not None:
            by_name = {joint.name: joint for joint in joints}
            unknown = [name for name in wanted if name not in by_name]
            if unknown:
                gs.raise_exception(
                    f"ActuationAudio references joints {unknown} that entity {self._options.entity_idx} does not "
                    f"have. Available joints: {sorted(by_name)}."
                )
            joints = [by_name[name] for name in wanted]

        overrides = self._options.properties
        unknown_overrides = [name for name in overrides if name not in {joint.name for joint in joints}]
        if unknown_overrides:
            gs.raise_exception(
                f"ActuationAudio has `properties` overrides for joints {unknown_overrides}, which are not among the "
                f"voiced joints {[joint.name for joint in joints]}. Add them to `joints` or drop the override."
            )

        for joint in joints:
            if joint.n_dofs == 0:
                continue
            props = overrides.get(joint.name, self._options.default_properties)
            # One emission point per DOF, radiating from the joint's child link.
            for dof in range(joint.dof_start, joint.dof_end):
                self._dof_idx.append(dof)
                self._emit_links.append(joint.link.idx)
                self._properties.append(props)

        if not self._dof_idx:
            gs.raise_exception(
                f"ActuationAudio on entity {self._options.entity_idx} resolved no actuated DOFs "
                f"(joints={self._options.joints}); it would emit nothing."
            )

    def _get_return_format(self) -> tuple[int, ...]:
        if not self._dof_idx:
            self._resolve_dofs()
        return (len(self._dof_idx), self._options.audio_substeps)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    def build(self):
        """Build this sensor's voice and register it with the scene's audio registry."""
        super().build()

        context = self._shared_context
        self._voice = ActuationVoice(
            self._options,
            self._manager._sim.rigid_solver,
            self._dof_idx,
            self._emit_links,
            self._properties,
        )
        context.register(self._voice, type(self._options).__name__)

        sm = self._shared_metadata
        sm.voices.append(self._voice)
        sm.cache_offsets.append(self._cache_offset)

    @classmethod
    def _update_raw_data(
        cls, shared_context: AudioContext, shared_metadata: ActuationAudioMetadata, raw_data_T: torch.Tensor
    ):
        """Copy each voice's already-synthesized block into this class's read cache.

        Parameters
        ----------
        shared_context : AudioContext
            The registry that drove synthesis earlier this step; told about this refresh so a gapped ``read()``
            is reported.
        shared_metadata : ActuationAudioMetadata
            Holds the voices and their row offsets in the class cache.
        raw_data_T : torch.Tensor
            Transposed write target, ``(class_cache_size, B)``.
        """
        shared_context.blocks_since(shared_metadata.last_render, ActuationAudio.__name__)
        shared_metadata.last_render = shared_context.n_updates
        for voice, offset in zip(shared_metadata.voices, shared_metadata.cache_offsets):
            block = voice.block  # (B, n_emit, K)
            span = block.shape[1] * block.shape[2]
            raw_data_T[offset : offset + span] = block.reshape(block.shape[0], span).T
