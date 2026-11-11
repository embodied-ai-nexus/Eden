"""Options for the FlashDexRetarget kinematic-retargeting pipeline (human demos -> ``motion.pt``).

Two stages, each driven by one options object:

- :class:`FDRImportOptions` — a released dataset (TACO, OakInk2, HOT3D, DexYCB, each read by its
  :mod:`eden.datasets` loader) -> ``<dataset>/human_demo/<clip>.npz`` and ``<dataset>/objects/<id>/visual.obj``.
- :class:`FDRKinematicRetargetingOptions` — those clips -> support disks and reject rules (``report.csv``), convex
  object parts, MuJoCo physics IK onto the robot hand, and ``<dataset>/<robot>/motion.pt``.

Defaults are FlashDexRetarget's (Holiday-Robot/FlashDexRetarget ``config/retarget/*.yaml``); the Isaac-only ``usd``
step is gone. ``eden retarget flashdex`` builds these from ``--config`` / ``--override`` the way ``eden train`` does.
"""

from __future__ import annotations
from typing import Annotated, Any, Literal

from pydantic import Discriminator, Field, Tag, model_validator

from eden.options.options import ConfigurableOptions


def _by_name(group: str, names: tuple[str, ...], default: str | None) -> Discriminator:
    """The ``name`` discriminator of an options group; a mapping without ``name`` is ``default``'s (so an override
    can set one field, ``convex: {force: true}``), or an error listing the names when there is no default.
    """

    def tag(value: Any) -> str | None:
        return value.get("name", default) if isinstance(value, dict) else getattr(value, "name", None)

    listed = ", ".join(repr(n) for n in names)
    hint = f" (default {default!r})" if default else f", e.g. {group}: {{name: {names[0]}, ...}}"
    return Discriminator(
        tag,
        custom_error_type="unknown_option_name",
        custom_error_message=f"{group}.name must be one of {listed}{hint}",
    )


# --------------------------------------------------------------------------------------------------------------------
# Import: one options class per released dataset, picked by ``name``
# --------------------------------------------------------------------------------------------------------------------


class FDRTacoSourceOptions(ConfigurableOptions):
    """TACO, read by :class:`~eden.datasets.TacoDataset` (30 Hz, one clip per recording).

    Parameters
    ----------
    table_z : float | None
        Height to put the surface the objects rest on at; ``None`` keeps the release's heights.
    dataset : dict[str, Any]
        Options of :class:`~eden.datasets.TacoDataset`; checked when it is built. ``raw_data_path``,
        ``body_model_dir``, ``body_model_params``, ``target_framerate`` and ``sides`` come from the import options.
    """

    name: Literal["taco"] = "taco"
    table_z: float | None = None
    dataset: dict[str, Any] = Field(default_factory=dict)


class FDROakInk2SourceOptions(ConfigurableOptions):
    """OakInk2, read by :class:`~eden.datasets.OakInk2Dataset` (120 Hz, one clip per program stage).

    Parameters
    ----------
    table_z : float | None
        Height to put the surface the objects rest on at; ``None`` keeps the release's heights.
    desk_over_recording : bool
        Fit the support disks on every object of the whole recording (CHORD), not only the clip's own.
    scales : dict[str, float]
        Mesh scale per object id (1.0 when absent): OakInk2's scans of these are off in size.
    dataset : dict[str, Any]
        Options of :class:`~eden.datasets.OakInk2Dataset` (e.g. ``contact``, the distance at which a MANO vertex
        touches an object); checked when it is built. ``raw_data_path``, ``body_model_dir``,
        ``body_model_params``, ``target_framerate`` and ``sides`` come from the import options.
    """

    name: Literal["oakink2"] = "oakink2"
    table_z: float | None = None
    desk_over_recording: bool = True
    scales: dict[str, float] = Field(
        default_factory=lambda: {
            "O02@0015@00001": 1.1,
            "O02@0015@00004": 1.08,
            "O02@0015@00007": 1.2,
            "O02@0015@00010": 1.2,
            "O02@0015@00013": 1.2,
            "O02@0015@00019": 1.02,
            "O02@0015@00020": 1.05,
            "O02@0015@00022": 1.2,
            "O02@0015@00025": 1.2,
            "O02@0018@00001": 1.05,
            "O02@0029@00002": 1.3,
            "O02@0029@00009": 1.3,
            "O02@0029@00011": 0.9,
            "O02@0029@00012": 1.3,
            "O02@0206@00001": 1.15,
            "O02@0206@00002": 0.95,
        }
    )
    dataset: dict[str, Any] = Field(default_factory=dict)


class FDRHot3dSourceOptions(ConfigurableOptions):
    """HOT3D, read by :class:`~eden.datasets.Hot3dDataset` (30 Hz, cut into hand-object segments).

    Parameters
    ----------
    table_z : float | None
        Height to put the surface the objects rest on at; ``None`` keeps the release's heights.
    desk_over_recording : bool
        Fit the support disks on every object of the whole recording, not only the segment's own.
    dataset : dict[str, Any]
        Options of :class:`~eden.datasets.Hot3dDataset`, e.g. its segmenting (``tip_contact``, ``link_contact``,
        ``gap_frames``, ``min_frames``, ``pad_frames``, ``modes``); checked when it is built. ``raw_data_path``,
        ``body_model_dir``, ``body_model_params``, ``target_framerate`` and ``sides`` come from the import options.
    """

    name: Literal["hot3d"] = "hot3d"
    table_z: float | None = 1.0
    desk_over_recording: bool = False
    dataset: dict[str, Any] = Field(default_factory=dict)


class FDRDexYCBSourceOptions(ConfigurableOptions):
    """DexYCB, read by :class:`~eden.datasets.DexYCBDataset` (30 Hz, one hand per clip).

    Not one of FlashDexRetarget's releases: a single-hand grasp-and-lift source, so import it with ``sides`` set to
    the hand the subject used (every subject's ``mano_sides`` is ``right`` in the public release). The release's
    root holds ``<subject>/<sequence>/``, ``calibration/`` and ``models/``; poses stay in the AprilTag frame, on the
    release's meshes.

    Parameters
    ----------
    table_z : float | None
        Height to put the surface the objects rest on at; ``None`` keeps the AprilTag frame's (the table top is
        at about 0).
    dataset : dict[str, Any]
        Options of :class:`~eden.datasets.DexYCBDataset`; checked when it is built. ``raw_data_path``,
        ``body_model_dir``, ``body_model_params``, ``target_framerate``, ``calibration_dir``, ``models_dir``,
        ``ground_initial_pose`` and ``apply_mesh_correction`` come from the import.
    """

    name: Literal["dexycb"] = "dexycb"
    table_z: float | None = None
    dataset: dict[str, Any] = Field(default_factory=dict)


FDRSourceOptions = Annotated[
    Annotated[FDRTacoSourceOptions, Tag("taco")]
    | Annotated[FDROakInk2SourceOptions, Tag("oakink2")]
    | Annotated[FDRHot3dSourceOptions, Tag("hot3d")]
    | Annotated[FDRDexYCBSourceOptions, Tag("dexycb")],
    _by_name("source", ("taco", "oakink2", "hot3d", "dexycb"), None),
]


class FDRImportOptions(ConfigurableOptions):
    """A released dataset -> ``<dataset>/human_demo/<clip>.npz`` + ``<dataset>/objects/<id>/visual.obj``.

    Parameters
    ----------
    source : FDRSourceOptions
        The release and its loader settings, picked by ``name`` (``taco`` / ``oakink2`` / ``hot3d`` / ``dexycb``).
    raw : str
        The downloaded release's root directory.
    dataset : str
        The dataset directory to write.
    mano_dir : str
        Directory holding ``MANO_RIGHT.pkl`` and ``MANO_LEFT.pkl`` (mano.is.tue.mpg.de; not redistributable).
    sides : Literal["both", "right", "left"]
        Hands to import; a clip without an object in each of them is skipped. DexYCB has one hand per clip, so it
        needs ``right`` or ``left``.
    fps : float
        Rate every clip is resampled to.
    max_obj_rot_step_deg : float
        A clip whose object turns more than this in one source frame is skipped (a pose flip).
    textures : bool
        Export each object with the release's materials and textures, for rendering ``visual.obj``. Off, it is
        exported from its geometry alone (a mesh with UVs gets trimesh's grey placeholder texture), which is all the
        retargeting reads; an object whose geometry is unchanged keeps the materials an earlier import wrote. On,
        every import loads each object's textures once more (a 4096^2 one decodes to ~50 MB, one object at a time)
        to compare them with what it wrote last time, and re-encodes them where they changed.
    """

    source: FDRSourceOptions
    raw: str
    dataset: str
    mano_dir: str
    sides: Literal["both", "right", "left"] = "both"
    fps: float = Field(60.0, gt=0)
    max_obj_rot_step_deg: float = Field(60.0, gt=0)
    textures: bool = False

    @model_validator(mode="after")
    def _one_hand_per_dexycb_clip(self) -> FDRImportOptions:
        if self.source.name == "dexycb" and self.sides == "both":
            raise ValueError(
                "source.name='dexycb' has one hand per clip, so sides='both' would skip every clip. Set sides to the "
                "hand the subjects used: `--sides right` (every subject of the public release is right-handed)."
            )
        return self


# --------------------------------------------------------------------------------------------------------------------
# Kinematic retargeting
# --------------------------------------------------------------------------------------------------------------------


class FDRStepsOptions(ConfigurableOptions):
    """Which steps of the kinematic pipeline run (preprocess always does, when the dataset has clips).

    Parameters
    ----------
    convex : bool
        Convex decomposition of every object (``<object>/convex/<i>.obj``).
    retarget : bool
        Physics IK of every kept clip (``<robot>/retargeted/<clip>.npz``).
    pack : bool
        Kept clips -> ``<robot>/motion.pt``.
    urdf : bool
        Object URDFs for simulation (``<object>/object.urdf``); needs ``<robot>/motion.pt``.
    sdf : bool
        Object SDF grids (``<object>/.sdf_cache/``); needs ``<robot>/motion.pt``.
    """

    convex: bool = True
    retarget: bool = True
    pack: bool = True
    urdf: bool = True
    sdf: bool = True


class FDRDeskOptions(ConfigurableOptions):
    """Support disks (tables, shelves) fitted where the objects rest still; ported from CHORD.

    Parameters
    ----------
    mode : Literal["fit", "keep", "none"]
        ``fit`` the disks, ``keep`` a clip's own ``support_disks`` (fit when absent), ``none``: the floor only.
    still_speed : float
        Linear speed (m/s) under which an object rests.
    still_ang_speed : float
        Angular speed (rad/s) under which an object rests.
    min_still_sec : float
        Shortest rest that fits a disk.
    disk_height : float
        Height of a fitted disk.
    phantom_in_disk_frac : float
        A disk that another simulated object occupies this fraction of the time is that object, dropped.
    phantom_z : float
        Vertical band (m) of that occupancy test.
    consolidate_z : float
        Overlapping disks within this height difference merge.
    rest_band : tuple[float, float]
        An object passes over a disk within ``[lo, hi]`` of its top: the disk is used.
    contact_z : float
        With the recording's context: an object stands on a disk within this distance of its top.
    """

    mode: Literal["fit", "keep", "none"] = "fit"
    still_speed: float = 0.03
    still_ang_speed: float = 0.3
    min_still_sec: float = Field(0.1667, ge=0)
    disk_height: float = 0.01
    phantom_in_disk_frac: float = 0.9
    phantom_z: float = 0.10
    consolidate_z: float = 0.05
    rest_band: tuple[float, float] = (-0.02, 0.25)
    contact_z: float = 0.02


class FDROffDiskUntouchedOptions(ConfigurableOptions):
    """Reject: an untouched object floats where nothing supports it for ``frames`` frames below ``max_height``."""

    frames: int = Field(5, ge=1)
    max_height: float = 0.15


class FDRWristAccSpikeOptions(ConfigurableOptions):
    """Reject: the ``smooth``-frame averaged MANO wrist accelerates faster than ``max_acc`` (m/s^2)."""

    max_acc: float = 60.0
    smooth: int = Field(3, ge=1)


class FDRObjBelowDeskOptions(ConfigurableOptions):
    """Reject: an untouched object starts or ends more than ``-min_z`` below the desk."""

    min_z: float = -0.05


class FDRTooShortOptions(ConfigurableOptions):
    """Reject: the clip is shorter than ``min_sec`` seconds."""

    min_sec: float = 2.5


class FDRFirstUnsupportedOptions(ConfigurableOptions):
    """Reject: an object is not over its own base (``base_band`` of its lowest point, + ``slack``) at the start."""

    base_band: float = 0.01
    slack: float = 0.005


class FDRFilterOptions(ConfigurableOptions):
    """Reject rules over the imported clips (``report.csv``).

    Parameters
    ----------
    enable : bool
        ``False`` keeps every clip.
    contact : float
        Distance (m) at which a fingertip or another object touches an object.
    edge_frames : int
        Frames at either end the start/end rules look at.
    obj_offdisk_untouched : FDROffDiskUntouchedOptions
        An untouched object floats where nothing supports it.
    wrist_acc_spike : FDRWristAccSpikeOptions
        The MANO wrist accelerates too fast.
    obj_below_desk : FDRObjBelowDeskOptions
        An untouched object starts or ends below the desk.
    too_short : FDRTooShortOptions
        The clip is too short.
    first_unsupported : FDRFirstUnsupportedOptions
        An object topples in the first frames.
    """

    enable: bool = True
    contact: float = Field(0.015, gt=0)
    edge_frames: int = Field(5, ge=1)
    obj_offdisk_untouched: FDROffDiskUntouchedOptions = Field(default_factory=FDROffDiskUntouchedOptions)
    wrist_acc_spike: FDRWristAccSpikeOptions = Field(default_factory=FDRWristAccSpikeOptions)
    obj_below_desk: FDRObjBelowDeskOptions = Field(default_factory=FDRObjBelowDeskOptions)
    too_short: FDRTooShortOptions = Field(default_factory=FDRTooShortOptions)
    first_unsupported: FDRFirstUnsupportedOptions = Field(default_factory=FDRFirstUnsupportedOptions)


class FDRCoacdParams(ConfigurableOptions):
    """Keyword arguments of ``coacd.run_coacd`` (see CoACD's documentation)."""

    threshold: float = Field(0.07, gt=0)
    max_convex_hull: int = Field(32, ge=-1)
    preprocess_mode: Literal["auto", "on", "off"] = "auto"
    preprocess_resolution: int = 50
    resolution: int = 2000
    mcts_nodes: int = 20
    mcts_iterations: int = 2000
    mcts_max_depth: int = 5
    pca: bool = False
    merge: bool = True
    decimate: bool = False
    max_ch_vertex: int = 256
    extrude: bool = False
    apx_mode: Literal["ch", "box"] = "ch"


class FDRCoacdConvexOptions(ConfigurableOptions):
    """CoACD convex decomposition on CPU workers.

    Parameters
    ----------
    force : bool
        Redo objects whose convex parts were made from their current mesh and these settings (parts made from any
        other are redone anyway).
    decimate_faces : int
        Decimate a mesh with more faces first (dense scans make CoACD very slow); 0 never.
    seed : int
        CoACD's random seed.
    coacd : FDRCoacdParams
        ``coacd.run_coacd`` keyword arguments.
    """

    name: Literal["coacd"] = "coacd"
    force: bool = False
    decimate_faces: int = Field(0, ge=0)
    seed: int = 1
    coacd: FDRCoacdParams = Field(default_factory=FDRCoacdParams)


class FDRCuacdConvexOptions(ConfigurableOptions):
    """CuACD convex decomposition on GPUs (one process per GPU, about 20 GB each), then CoACD's merge.

    Parameters
    ----------
    force : bool
        Redo objects whose convex parts were made from their current mesh and these settings (parts made from any
        other are redone anyway).
    decimate_faces : int
        Decimate a mesh with more faces first; 0 never.
    gpus : list[int] | None
        Indices into the visible GPUs; ``None`` uses every one.
    threshold : float
        Concavity threshold.
    depth : int
        Lookahead depth.
    preprocess_resolution : int
        Remeshing resolution for a mesh CuACD cannot take as is.
    max_convex_hull : int
        Parts kept after the merge.
    """

    name: Literal["cuacd"] = "cuacd"
    force: bool = False
    decimate_faces: int = Field(0, ge=0)
    gpus: list[Annotated[int, Field(ge=0)]] | None = Field(None, min_length=1)
    threshold: float = Field(0.05, gt=0)
    depth: int = Field(3, ge=1)
    preprocess_resolution: int = 128
    max_convex_hull: int = Field(32, ge=1)


FDRConvexOptions = Annotated[
    Annotated[FDRCoacdConvexOptions, Tag("coacd")] | Annotated[FDRCuacdConvexOptions, Tag("cuacd")],
    _by_name("convex", ("coacd", "cuacd"), "coacd"),
]


class FDRApproachOptions(ConfigurableOptions):
    """The first IK attempt starts from outside the object instead of from random restarts.

    Parameters
    ----------
    enable : bool
        Use the approach for attempt 0.
    backoff : float
        Distance (m) behind the wrist target (the back of the hand) the hand starts at.
    slide_frames : int
        Frames over which the hand slides onto the wrist targets.
    close_frames : int
        Frames over which the fingertip targets move onto frame 0's.
    preshape : float
        Fraction of the contact-free start search's finger shape to begin with; 0 opens the hand.
    """

    enable: bool = True
    backoff: float = Field(0.06, ge=0)
    slide_frames: int = Field(30, ge=1)
    close_frames: int = Field(30, ge=1)
    preshape: float = Field(1.0, ge=0, le=1)


class FDREqualityOptions(ConfigurableOptions):
    """A MuJoCo equality constraint pulling a robot site onto its mocap target."""

    torque: float
    solref: tuple[float, float]
    solimp: tuple[float, float, float, float, float]


class FDRConstraintsOptions(ConfigurableOptions):
    """The IK's equality constraints: palm weld, fingertip connects and object weld.

    Parameters
    ----------
    palm : FDREqualityOptions
        Weld of the palm site onto the MANO wrist.
    tip : FDREqualityOptions
        Connect of each fingertip site onto its MANO fingertip.
    tip_width_ring_pinky : float
        Factor on the tip ``solimp`` width for the ring and pinky (they track MANO less closely).
    object : FDREqualityOptions
        Stiff weld of each object onto its reference pose.
    """

    palm: FDREqualityOptions = Field(
        default_factory=lambda: FDREqualityOptions(torque=10.0, solref=(0.02, 1.0), solimp=(0.0, 0.95, 10.0, 0.5, 2.0))
    )
    tip: FDREqualityOptions = Field(
        default_factory=lambda: FDREqualityOptions(torque=1.0, solref=(0.01, 1.0), solimp=(0.0, 0.95, 0.01, 0.5, 2.0))
    )
    tip_width_ring_pinky: float = 3.0
    object: FDREqualityOptions = Field(
        default_factory=lambda: FDREqualityOptions(
            torque=10.0, solref=(0.002, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0)
        )
    )


class FDRContactOptions(ConfigurableOptions):
    """Contact settings of the IK model. ``mjENBL_OVERRIDE`` is on (as in FlashDexRetarget), so every contact, the
    explicit hand/object/support pairs included, takes ``o_solref``, ``o_solimp`` and ``o_friction``.

    Parameters
    ----------
    impratio : float
        Ratio of frictional to normal constraint impedance.
    o_solref : tuple[float, float]
        ``solref`` of every contact.
    o_solimp : tuple[float, float, float, float, float]
        ``solimp`` of every contact.
    o_friction : tuple[float, float, float, float, float]
        Friction of every contact (sliding, sliding, torsional, rolling, rolling); MuJoCo's default.
    """

    impratio: float = 1.0
    o_solref: tuple[float, float] = (0.02, 1.0)
    o_solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.002, 0.5, 2.0)
    o_friction: tuple[float, float, float, float, float] = (1.0, 1.0, 0.005, 1.0e-4, 1.0e-4)


class FDRIKSceneOptions(ConfigurableOptions):
    """The IK model built around the robot MJCF.

    Parameters
    ----------
    armature : float
        Armature of every robot joint (heavy joints keep the IK stable).
    kp_wrist : float
        Position gain of the forearm actuators (critically damped at ``qpos0``).
    kp_finger : float
        Position gain of the finger actuators (critically damped at ``qpos0``).
    desk_depth : float
        Minimum thickness of a support-disk cylinder.
    object_density : float
        Density of the object bodies.
    object_armature : float
        Free-joint armature of the objects.
    object_frictionloss : float
        Free-joint friction loss of the objects.

    The contact pairs' ``solref`` and friction are ``retarget.contact``'s (``mjENBL_OVERRIDE``).
    """

    armature: float = 1.0
    kp_wrist: float = 1000.0
    kp_finger: float = 300.0
    desk_depth: float = 0.3
    object_density: float = 1000.0
    object_armature: float = 1.0e-4
    object_frictionloss: float = 1.0e-4


class FDRMaxAccOptions(ConfigurableOptions):
    """Flag: the palm accelerates faster than ``max_acc`` (m/s^2)."""

    max_acc: float = 60.0


class FDRMaxDepthOptions(ConfigurableOptions):
    """Flag: a depth beyond ``max_depth`` (m)."""

    max_depth: float


class FDRTipTrackingOptions(ConfigurableOptions):
    """Flag: more than ``max_frac`` of the fingertip samples are ``max_dist`` (m) from MANO."""

    max_dist: float = 0.03
    max_frac: float = 0.35


class FDRHandDeskPenetrationOptions(ConfigurableOptions):
    """Flag: the hand is ``max_depth`` (m) inside a disk for ``frames`` frames after frame ``after``."""

    max_depth: float = 0.02
    frames: int = 5
    after: int = 30


class FDRQualityOptions(ConfigurableOptions):
    """Flags (``report.csv``) on a retargeted clip; a flagged clip is still packed."""

    palm_acc_spike: FDRMaxAccOptions = Field(default_factory=FDRMaxAccOptions)
    hand_far_below_desk: FDRMaxDepthOptions = Field(default_factory=lambda: FDRMaxDepthOptions(max_depth=0.08))
    tip_tracking: FDRTipTrackingOptions = Field(default_factory=FDRTipTrackingOptions)
    hand_obj_penetration: FDRMaxDepthOptions = Field(default_factory=lambda: FDRMaxDepthOptions(max_depth=0.02))
    hand_desk_penetration: FDRHandDeskPenetrationOptions = Field(default_factory=FDRHandDeskPenetrationOptions)


class FDRRetargetOptions(ConfigurableOptions):
    """MuJoCo physics IK: the robot's palm and fingertip sites follow the MANO wrist and fingertips while the hands
    collide with the objects and support disks.

    Parameters
    ----------
    force : bool
        Redo clips whose ``<robot>/retargeted/<clip>.npz`` was made from this run's inputs and settings (a clip
        made from any other is redone anyway).
    sim_dt : float
        MuJoCo timestep.
    solver_iterations : int
        MuJoCo solver iterations.
    solver_ls_iterations : int
        MuJoCo line-search iterations.
    n_guess : int
        Random restarts of a start search.
    guess_steps : int
        Steps each restart runs.
    guess_jitter : float
        Noise (m) on the fingertip targets of a restart.
    start_settle : int
        Frame-0 steps before the clip runs.
    approach : FDRApproachOptions
        Attempt 0's start from outside the object.
    settle_steps : int
        Steps per frame with the fingertip constraints off and the joints PD-held (contacts push out).
    recover_tip_err : float
        A restart is searched when the recent mean fingertip error exceeds this (m).
    recover_window : int
        Frames that error averages over.
    recover_cooldown : int
        Frames between restarts.
    attempts : int
        IK runs per clip; the best is kept.
    pen_weight : float
        Weight of a run's max hand-object penetration against its mean fingertip error.
    smooth : int
        Moving-average window, centred on each frame, over the kept run (the first ``smooth`` frames are cropped).
    constraints : FDRConstraintsOptions
        Palm, fingertip and object equality constraints.
    contact : FDRContactOptions
        Global contact settings.
    scene : FDRIKSceneOptions
        The IK model around the robot MJCF.
    quality : FDRQualityOptions
        Quality flags of the kept run.
    """

    force: bool = False
    sim_dt: float = Field(0.002083, gt=0)
    solver_iterations: int = Field(20, ge=1)
    solver_ls_iterations: int = Field(50, ge=1)
    n_guess: int = Field(48, ge=1)
    guess_steps: int = Field(30, ge=1)
    guess_jitter: float = Field(0.002, ge=0)
    start_settle: int = Field(60, ge=0)
    approach: FDRApproachOptions = Field(default_factory=FDRApproachOptions)
    settle_steps: int = Field(80, ge=0)
    recover_tip_err: float = Field(0.045, gt=0)
    recover_window: int = Field(15, ge=1)
    recover_cooldown: int = Field(30, ge=0)
    attempts: int = Field(6, ge=1)
    pen_weight: float = Field(0.5, ge=0)
    smooth: int = Field(3, ge=1)
    constraints: FDRConstraintsOptions = Field(default_factory=FDRConstraintsOptions)
    contact: FDRContactOptions = Field(default_factory=FDRContactOptions)
    scene: FDRIKSceneOptions = Field(default_factory=FDRIKSceneOptions)
    quality: FDRQualityOptions = Field(default_factory=FDRQualityOptions)


class FDRPackOptions(ConfigurableOptions):
    """``motion.pt`` packing.

    Parameters
    ----------
    contact_threshold : float
        A fingertip within this distance (m) of the object mesh is in contact (dilated by one frame).
    vel_sigma : float
        Gaussian smoothing (frames) of the finite-difference velocities.
    """

    contact_threshold: float = Field(0.015, gt=0)
    vel_sigma: float = Field(2.0, ge=0)


class FDRSdfOptions(ConfigurableOptions):
    """Object SDF grids over ``[-extent, extent]^3`` in the object frame (signed distance + gradient).

    Parameters
    ----------
    grid_n : int
        Grid points per axis; must match the training task's SDF settings.
    extent : float
        Half-size (m) of the grid; must match the training task's SDF settings.
    device : Literal["cuda", "cpu"]
        Where warp bakes.
    gpus : list[int] | None
        Indices into the visible GPUs; ``None`` uses every one.
    workers_per_gpu : int
        Processes per GPU.
    """

    grid_n: int = Field(128, ge=2)
    extent: float = Field(0.30, gt=0)
    device: Literal["cuda", "cpu"] = "cuda"
    gpus: list[Annotated[int, Field(ge=0)]] | None = Field(default_factory=lambda: [0], min_length=1)
    workers_per_gpu: int = Field(8, ge=1)


class FDRKinematicRetargetingOptions(ConfigurableOptions):
    """Imported clips -> ``<dataset>/<robot>/motion.pt`` for one robot.

    Parameters
    ----------
    dataset : str
        The dataset directory (``human_demo/`` and ``objects/`` from the import).
    robot : Literal["xhand", "sharpa"]
        The bimanual robot the clips are retargeted to.
    workers : int | None
        Processes; ``None`` uses half the CPU cores.
    steps : FDRStepsOptions
        Which steps run.
    desk : FDRDeskOptions
        Support-disk fitting.
    filters : FDRFilterOptions
        Reject rules.
    convex : FDRConvexOptions
        Convex decomposition backend (``name``: ``coacd`` or ``cuacd``).
    retarget : FDRRetargetOptions
        Physics IK.
    pack : FDRPackOptions
        ``motion.pt`` packing.
    sdf : FDRSdfOptions
        Object SDF grids.
    """

    dataset: str
    robot: Literal["xhand", "sharpa"] = "xhand"
    workers: int | None = Field(None, ge=1)
    steps: FDRStepsOptions = Field(default_factory=FDRStepsOptions)
    desk: FDRDeskOptions = Field(default_factory=FDRDeskOptions)
    filters: FDRFilterOptions = Field(default_factory=FDRFilterOptions)
    convex: FDRConvexOptions = Field(default_factory=FDRCoacdConvexOptions)
    retarget: FDRRetargetOptions = Field(default_factory=FDRRetargetOptions)
    pack: FDRPackOptions = Field(default_factory=FDRPackOptions)
    sdf: FDRSdfOptions = Field(default_factory=FDRSdfOptions)
