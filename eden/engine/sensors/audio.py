"""The audio source registry and delay line every Eden audio sensor shares.

Eden's audio addon runs the usual three-stage pipeline: *synthesis* (a body turns physics into a block
of samples), *propagation* (the samples travel, attenuating and arriving late), *rendering* (a
microphone sums what reaches it). Sources publish into the registry defined here; microphones render
what they find in it, with no knowledge of concrete source types.

Where this lives is the whole reason the addon needs no Genesis fork. Upstream owns the registry as a
scene-level ``AudioManager``, which takes three Genesis edits to exist: ``Simulator`` to construct it,
``Scene.step`` to pump it, and ``Scene.add_audio_source`` to fill it. Genesis already ships a
mechanism with exactly those lifecycle hooks — ``SharedSensorContext``, a resource shared across
sensor *types* that ``SensorManager`` constructs on first use, refreshes once per step, resets, and
destroys. Hosting the registry there gets the same behaviour as a pure addition.

The ordering that matters comes for free. ``SensorManager.step`` calls ``context.update()`` *before*
the per-sensor-type update loop, so every source has synthesized the current step's block by the time
any microphone renders — whatever order the sensors were declared in. A source that synthesized inside
its own sensor's update hook would instead be at the mercy of declaration order, and a microphone
declared first would render a block one step stale.

The delay line lives here for a related but sharper reason: it must advance on *every* simulation
step, and a microphone does not. Eden skips the per-sensor-type update loop on intermediate decimation
sub-steps (``EnvOptions.update_sensors_every_substep``) while ``context.update()`` still runs, so a
history advanced from a microphone's render hook would drop every block but the last of each control
step — silently shortening every propagation delay. Advancing it here keeps the recorded sound
continuous no matter how often anything renders it.
"""

from abc import ABC, abstractmethod

import genesis as gs
import torch
from genesis.engine.sensors.base_sensor import SharedSensorContext
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.utils.geom import transform_by_quat

import eden as en
from eden.engine._surface import require_members

require_members(
    __name__,
    {
        SharedSensorContext: ("activate", "update", "reset", "destroy"),
        RigidSolver: ("get_links_pos", "get_links_quat"),
    },
)


class AudioSource(ABC):
    """One radiating body in the scene: the synthesis stage of the audio pipeline.

    A source *generates* sound from physics and is never read by the user — a microphone
    (:class:`~eden.engine.sensors.spatial_audio.SpatialAudioSensor`) renders it. Subclasses own their
    sample buffers and expose the radiation interface below so receivers stay decoupled from concrete
    source types.

    An emission point is one place sound leaves the body: a motor source has one per actuated DOF, a
    contact source one per contact patch. All three tensors are indexed by that axis.
    """

    @abstractmethod
    def emit(self) -> None:
        """Synthesize this step's block into the source's own persistent state.

        Called once per simulation step by :meth:`AudioContext.update`, before anything renders it.
        """

    @abstractmethod
    def reset(self, envs_idx) -> None:
        """Clear persistent synthesis state (filter memory, oscillator phase) for ``envs_idx``.

        Parameters
        ----------
        envs_idx : array-like | None
            Environments to reset. ``None`` resets all of them.
        """

    @property
    @abstractmethod
    def block(self) -> torch.Tensor:
        """``(B, n_emit, audio_substeps)`` most recently synthesized samples per emission point."""

    @property
    @abstractmethod
    def emit_links(self) -> torch.Tensor:
        """``(n_emit,)`` solver-global rigid link index each emission point rides. ``-1`` is static.

        Solver-global indices (``RigidLink.idx``), the ``links_idx`` of ``RigidSolver.get_links_pos()`` — not
        entity-local ``idx_local``. Read once, when the source registers.
        """

    @property
    @abstractmethod
    def emit_offset(self) -> torch.Tensor:
        """``(B, n_emit, 3)`` offset of each emission point in its link's frame, or world position if static.

        Read every step, so a source whose emission points move on the body (a contact patch) may update it.
        """

    @property
    @abstractmethod
    def audio_substeps(self) -> int:
        """Samples this source emits per simulation step. Must match every other participant."""


class AudioContext(SharedSensorContext):
    """Registry of :class:`AudioSource` objects plus the delay line carrying their sound.

    Held by ``SensorManager`` as the shared context of the audio sensor classes, so exactly one
    instance exists per scene however many audio sensors are declared. See the module docstring for
    why both the registry and the history live here rather than on the scene or on a microphone.

    The delay line is a ring of whole blocks: :meth:`update` overwrites the oldest block in place, and a
    reader maps a sample's age to its slot through :attr:`hist_newest`. Nothing is allocated per step.
    """

    def __init__(self, sim):
        super().__init__(sim)
        self._sources: list[AudioSource] = []
        # The scene-wide sample rate multiplier, adopted from the first participant to declare one.
        # 0 means "not yet claimed"; `require_audio_substeps` fixes it and rejects disagreement.
        self._audio_substeps = 0
        # Delay-line depth in samples requested by the microphones' propagation reach at build.
        self._hist_len = 0
        # (B, n_src, H) ring of samples. The source count is only final once every source has registered, so
        # this and the per-emission-point tensors below are allocated on the first update rather than at build.
        self._src_hist = torch.empty(0)
        self._hist_newest = -1
        self._src_offset = torch.empty(0)
        self._src_is_static = torch.empty(0, dtype=torch.bool)
        self._attached_cols = torch.empty(0, dtype=torch.long)
        self._attached_links = torch.empty(0, dtype=gs.tc_int)
        # (first emission-point column, source) for every source that radiates.
        self._src_slices: list[tuple[int, AudioSource]] = []
        # Blocks synthesized so far. A consumer remembers the value it last rendered at, which tells it how
        # many blocks it skipped; see `blocks_since`.
        self._n_updates = 0
        self._warned_gapped = False

    def activate(self) -> None:
        """Mark the registry live. Idempotent; the delay line is allocated on first use."""
        self._active = True

    def register(self, source: AudioSource, owner: str) -> None:
        """Add ``source`` to the registry and activate it. Call from a producing sensor's ``build``.

        Parameters
        ----------
        source : AudioSource
            The source to register. Its ``audio_substeps`` must agree with every other participant.
        owner : str
            Name of the options class that declared the source, for the error message.
        """
        if self._src_hist.numel() > 0:
            gs.raise_exception(
                f"{owner} registered an audio source after the delay line was sized for {self._src_hist.shape[1]} "
                "emission points. Register every source at build, before the registry's first update."
            )
        self.require_audio_substeps(source.audio_substeps, owner)
        self._sources.append(source)
        self._active = True

    def require_audio_substeps(self, audio_substeps: int, who: str) -> None:
        """Claim the scene-wide ``audio_substeps``, or fail if it disagrees with the claim on record.

        Sources and microphones exchange fixed-length sample blocks, so one rate has to hold across
        the scene. Checking it here — at build, naming both values — beats discovering a ragged block
        inside the render loop.

        Parameters
        ----------
        audio_substeps : int
            The caller's samples-per-step.
        who : str
            Name of the calling participant, for the error message.
        """
        if self._audio_substeps == 0:
            self._audio_substeps = audio_substeps
        elif self._audio_substeps != audio_substeps:
            gs.raise_exception(
                f"{who} declares audio_substeps={audio_substeps} but the scene's audio pipeline is already "
                f"running at audio_substeps={self._audio_substeps}. Every audio source and microphone in a "
                "scene must share one sample rate."
            )

    def require_history(self, samples: int) -> None:
        """Ensure the delay line is at least ``samples`` deep. Call from a consuming sensor's ``build``.

        Parameters
        ----------
        samples : int
            Depth this consumer needs: its longest propagation delay plus one block.
        """
        self._hist_len = max(self._hist_len, samples)

    @property
    def audio_substeps(self) -> int:
        """The scene-wide samples-per-step every audio participant agrees on."""
        self._assert_active("audio_substeps")
        return self._audio_substeps

    @property
    def sources(self) -> tuple[AudioSource, ...]:
        """Every registered source. Empty when microphones exist but nothing radiates."""
        self._assert_active("sources")
        return tuple(self._sources)

    @property
    def src_hist(self) -> torch.Tensor:
        """``(B, n_src, H)`` ring of samples of every emission point; see :attr:`hist_newest` for the order.

        Empty until the first :meth:`update` with at least one radiating source.
        """
        self._assert_active("src_hist")
        return self._src_hist

    @property
    def hist_newest(self) -> int:
        """Slot of the newest sample in :attr:`src_hist`. A sample ``a`` samples old is at ``(hist_newest - a) % H``."""
        self._assert_active("hist_newest")
        return self._hist_newest

    @property
    def n_updates(self) -> int:
        """Blocks synthesized so far, for :meth:`blocks_since`."""
        return self._n_updates

    def update(self) -> None:
        """Synthesize every source's block and write it over the oldest block of the delay line.

        Manager-driven, once per simulation step, before anything renders.
        """
        if not self._active:
            return
        for source in self._sources:
            source.emit()
        if self._src_hist.numel() == 0 and not self._allocate():
            return

        K = self._audio_substeps
        start = (self._hist_newest + 1) % self._src_hist.shape[2]
        for col, source in self._src_slices:
            n_emit = source.emit_links.shape[0]
            self._src_hist[:, col : col + n_emit, start : start + K] = source.block
            self._src_offset[:, col : col + n_emit] = source.emit_offset
        self._hist_newest = start + K - 1
        self._n_updates += 1

    def _allocate(self) -> bool:
        """Size the delay line and the per-emission-point tensors. False when nothing radiates."""
        links, col = [], 0
        for source in self._sources:
            n_emit = source.emit_links.shape[0]
            if n_emit == 0:
                continue
            self._src_slices.append((col, source))
            links.append(source.emit_links)
            col += n_emit
        if not links:
            return False

        K = self._audio_substeps
        # Whole blocks only, so a block never wraps around the end of the ring.
        n_blocks = -(-max(self._hist_len, K) // K)
        B = self._sim._B
        self._src_hist = torch.zeros((B, col, n_blocks * K), dtype=gs.tc_float, device=gs.device)
        self._src_offset = torch.zeros((B, col, 3), dtype=gs.tc_float, device=gs.device)
        src_link = torch.cat(links, dim=0).to(gs.tc_int)
        self._src_is_static = src_link < 0
        self._attached_cols = torch.nonzero(~self._src_is_static).flatten()
        self._attached_links = src_link[self._attached_cols]
        return True

    def source_positions(self) -> torch.Tensor:
        """``(B, n_src, 3)`` world position of every emission point, parallel to :attr:`src_hist`.

        An emission point rides its link with the offset rotated into the link's frame; a static one sits at its
        offset. Poses are the solver-frame ones (``relative=False``), matching ``eden.entities.sensor.Sensor``.
        """
        self._assert_active("source_positions")
        pos = self._src_offset.clone()
        if self._attached_cols.numel() > 0:
            solver = self._sim.rigid_solver
            links_pos = solver.get_links_pos(links_idx=self._attached_links)
            links_quat = solver.get_links_quat(links_idx=self._attached_links)
            if solver.n_envs == 0:
                links_pos, links_quat = links_pos[None], links_quat[None]
            pos[:, self._attached_cols] = links_pos + transform_by_quat(pos[:, self._attached_cols], links_quat)
        return pos

    def blocks_since(self, last_render: int, who: str) -> int:
        """Blocks synthesized since a consumer last rendered at ``n_updates == last_render``; warn once if over 1.

        A consumer reads the newest block, so one render per :meth:`update` yields a continuous signal. Eden's
        default ``EnvOptions.update_sensors_every_substep=False`` skips sensor updates on intermediate decimation
        sub-steps, so with ``decimation=4`` a sensor renders every fourth block and its ``read()`` is a gapped
        stream — fine as an observation, wrong as a recording. That distinction is invisible in the data, and
        the decimation is Eden's, which a Genesis sensor cannot see at build, so say so on the first gapped render.

        Parameters
        ----------
        last_render : int
            :attr:`n_updates` as it stood at the consumer's previous render.
        who : str
            Name of the consuming sensor's options class, for the warning.
        """
        n_blocks = self._n_updates - last_render
        if n_blocks > 1 and not self._warned_gapped:
            self._warned_gapped = True
            en.logger.warning(
                f"{who} is rendered once per {n_blocks} synthesized audio blocks, so read() is a gapped stream "
                "rather than continuous sound. Propagation delay, attenuation and Doppler stay correct (the delay "
                "line advances every step), and this is harmless when the signal is an observation. Set "
                "EnvOptions.update_sensors_every_substep=True to render every block, e.g. when writing the signal "
                "to an audio file."
            )
        return n_blocks

    def reset(self, envs_idx) -> None:
        """Clear every source's synthesis state and the delay line for ``envs_idx``.

        Parameters
        ----------
        envs_idx : array-like | None
            Environments being reset. ``None`` resets all of them.
        """
        if not self._active:
            return
        for source in self._sources:
            source.reset(envs_idx)
        if self._src_hist.numel() > 0:
            self._src_hist[slice(None) if envs_idx is None else envs_idx] = 0.0

    def destroy(self) -> None:
        """Drop every registered source and the delay line, releasing their buffers."""
        self._sources.clear()
        self._src_slices.clear()
        self._src_hist = torch.empty(0)
        self._src_offset = torch.empty(0)
        self._hist_newest = -1
        self._audio_substeps = 0
        self._hist_len = 0
        self._active = False

    def _assert_active(self, what: str) -> None:
        """Refuse reads of an inactive registry, per the ``SharedSensorContext`` contract.

        Parameters
        ----------
        what : str
            Name of the member being read, for the error message.
        """
        if not self._active:
            gs.raise_exception(
                f"AudioContext.{what} read while the context is inactive: no audio sensor has activated it. "
                "Only a sensor that called activate() (or register()) in its build() may read the registry."
            )
