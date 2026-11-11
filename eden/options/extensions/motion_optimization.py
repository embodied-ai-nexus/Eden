"""Configuration options for the offline motion-optimization extensions.

Holds the config surface of the TCRS terrain-adaptive motion optimizer
(:mod:`eden.extensions.motion_optimization.tcrs`).

Upstream TCRS hardcodes Unitree G1 geometry (body/joint names, foot contact
offsets, leg reach) across several modules. The port lifts all of it into
:class:`TCRSRobotOptions` so another biped can be targeted without touching
the algorithms; the no-argument default reproduces the upstream G1 values with
the name lists taken from Eden's canonical G1 definitions.

:class:`TCRSPlannerOptions` collects the pipeline knobs that upstream exposed
as CLI flags (defaults preserved). Both are :class:`ConfigurableOptions`
(``extra="forbid"``), matching Eden's library-wide config pattern; the TCRS
engine consumes them as read-only attribute bags and stays env-free.
"""

from __future__ import annotations

import copy
import json

from pydantic import Field, field_validator, model_validator

from eden.options.options import ConfigurableOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME

# Verbatim port of upstream ``ik_config.json`` (G1 lower body). ``pos_weight``/
# ``rot_weight`` scale the error and Jacobian rows; keep in 0-10 to avoid
# ill-conditioning, 0 disables. Virtual toe/heel points carry an ``offset`` in
# the ankle-roll-link frame and must keep ``rot_weight`` 0 (points have no
# orientation). Pelvis position itself is handled by root_xy/z_weight.
DEFAULT_IK_CONFIG: dict = {
    "link_weights": {
        "pelvis": {"pos_weight": 1, "rot_weight": 1.0},
        "left_hip_pitch_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "left_hip_roll_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "left_hip_yaw_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "left_knee_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "left_ankle_pitch_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "right_hip_pitch_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "right_hip_roll_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "right_hip_yaw_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "right_knee_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "right_ankle_pitch_link": {"pos_weight": 0.0, "rot_weight": 0.0},
        "left_ankle_roll_link": {"pos_weight": 10.0, "rot_weight": 6.0},
        "right_ankle_roll_link": {"pos_weight": 10.0, "rot_weight": 6.0},
        "left_toe": {"pos_weight": 6.0, "rot_weight": 0.0, "offset": [0.12, 0.0, -0.03]},
        "left_heel": {"pos_weight": 6.0, "rot_weight": 0.0, "offset": [-0.05, 0.0, -0.03]},
        "right_toe": {"pos_weight": 6.0, "rot_weight": 0.0, "offset": [0.12, 0.0, -0.03]},
        "right_heel": {"pos_weight": 6.0, "rot_weight": 0.0, "offset": [-0.05, 0.0, -0.03]},
    },
    "solver": {
        "max_iters": 60,
        "tol": 1e-3,
        "damping": 0.05,
        "step_size": 0.5,
        "orientation_weight": 0.2,
        "posture_weight": 0.01,
        "max_joint_step": 0.20,
        "penetration_weight": 20.0,
        "root_xy_weight": 3.0,
        "root_z_weight": 20,
        # Penalizes the knee falling too far behind the pelvis along the
        # forward direction; 0 turns it off.
        "knee_forward_weight": 15.0,
        "multiseed_enabled": True,
        "multiseed_num_seeds": 3,
        "multiseed_trigger_pos_err": 0.20,
        "multiseed_trigger_joint_step": 0.80,
        "multiseed_seed_span": 0.35,
        "multiseed_continuity_weight": 0.25,
        "multiseed_posture_weight": 0.02,
    },
}


def merged_ik_solver_config(solver_overrides: dict) -> dict:
    """Merge a (possibly partial) ``ik_config["solver"]`` section over the defaults.

    ``DEFAULT_IK_CONFIG["solver"]`` is the complete set of solver knobs; every consumer
    (:meth:`~eden.extensions.motion_optimization.tcrs.ik.JacobianIK.from_config`, the pipeline's
    root-weight lookup) reads the merged dict, so a partial
    override changes exactly the named keys and never falls back to a second,
    disagreeing set of hardcoded defaults. Unknown keys raise.
    """
    valid = set(DEFAULT_IK_CONFIG["solver"])
    unknown = set(solver_overrides) - valid
    if unknown:
        raise ValueError(f"Unknown ik_config solver key(s) {sorted(unknown)}. Valid solver keys: {sorted(valid)}.")
    return {**copy.deepcopy(DEFAULT_IK_CONFIG["solver"]), **solver_overrides}


def merged_ik_config(overrides: dict) -> dict:
    """Deep-merge a partial ``ik_config`` dict onto :data:`DEFAULT_IK_CONFIG`.

    Merge semantics:

    - ``solver``: key-wise merge over the default solver section; unknown
      solver keys raise (see :func:`merged_ik_solver_config`).
    - ``link_weights``: per-link-name merge — each override entry is merged
      over the default entry of the same name (so ``{"pelvis": {"pos_weight": 2}}``
      keeps the default ``rot_weight``); names absent from the defaults are
      added as new constraints.

    Sections other than ``solver``/``link_weights`` raise.
    """
    if not isinstance(overrides, dict):
        raise ValueError(f"ik_config overrides must be a dict; got {type(overrides).__name__}.")
    if "body_weights" in overrides:
        raise ValueError(
            "ik_config has no 'body_weights' section: Eden names it 'link_weights' (its dofs_*/links_* naming "
            "convention); rename the section to 'link_weights'."
        )
    unknown_sections = set(overrides) - {"link_weights", "solver"}
    if unknown_sections:
        raise ValueError(
            f"Unknown ik_config section(s) {sorted(unknown_sections)}. Valid sections: ['link_weights', 'solver']."
        )
    merged = copy.deepcopy(DEFAULT_IK_CONFIG)
    merged["solver"] = merged_ik_solver_config(overrides.get("solver", {}))
    for link_name, weights in overrides.get("link_weights", {}).items():
        merged["link_weights"][link_name] = {**merged["link_weights"].get(link_name, {}), **weights}
    return merged


class TCRSRobotOptions(ConfigurableOptions):
    """Robot geometry and naming the TCRS pipeline needs.

    The no-argument default is the upstream Unitree G1 spec with Eden's
    canonical name lists as clip orderings. There is no separate override
    API: construction is always overrides-on-the-G1-defaults — any
    unspecified field keeps its G1 value, and targeting another biped just
    means overriding more of them. Eden clips store joints in
    ``G1_DOFS29_NAME`` (chain) order; upstream PMT clips use an interleaved
    "policy" order. Both are supported — pass the clip's actual ordering;
    every consumer resolves columns by name.
    """

    dofs_name: list[str] = Field(default_factory=lambda: list(G1_DOFS29_NAME))
    """DOF order of the motion clip's ``dofs_pos`` columns. All MuJoCo qpos addressing
    is resolved by these names — never by column index."""

    links_name: list[str] = Field(default_factory=lambda: list(G1_DOFS29_TRACKED_LINKS_NAME))
    """Link order of the motion clip's ``links_pos_w``/``links_quat_w`` rows (pelvis first)."""

    pelvis_link: str = "pelvis"
    left_foot_link: str = "left_ankle_roll_link"
    right_foot_link: str = "right_ankle_roll_link"
    left_hip_link: str = "left_hip_pitch_link"
    right_hip_link: str = "right_hip_pitch_link"
    left_knee_link: str = "left_knee_link"
    right_knee_link: str = "right_knee_link"

    left_leg_dofs_name: tuple[str, ...] = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    right_leg_dofs_name: tuple[str, ...] = (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )
    waist_dofs_name: tuple[str, ...] = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

    toe_offset: tuple[float, float, float] = (0.12, 0.0, -0.03)
    """Representative toe contact point in the ankle-roll-link frame (from the G1 MJCF contact spheres)."""

    heel_offset: tuple[float, float, float] = (-0.05, 0.0, -0.03)
    """Representative heel contact point in the ankle-roll-link frame."""

    toe_probe_offset_x: float = 0.14
    """Forward probe distance used by footstep resolution (upstream ``TOE_OFFSET_X``)."""

    foot_nominal_z: float = 0.06
    """Ankle height above the support surface when the foot is flat on the ground."""

    hip_offset_z: float = -0.251
    """Pelvis origin to hip joint, z component (m)."""

    leg_max_length: float = 0.494
    """Hip-to-ankle chain length at full extension (m)."""

    pelvis_ankle_reach_slack: float = 0.06
    """Extra reach beyond ``|hip_offset_z| + leg_max_length`` from the hip chain's
    lateral/forward offsets; the pipeline works in pelvis-link ↔ ankle-link coordinates."""

    ik_config: dict = Field(default_factory=dict)
    """Per-link IK weights and solver settings (schema of ``ik_config.json``). Partial dicts —
    including this empty default — deep-merge onto :data:`DEFAULT_IK_CONFIG` at construction
    (see :func:`merged_ik_config`), so the stored value is always the full config."""

    @field_validator("ik_config", mode="before")
    @classmethod
    def _merge_ik_config(cls, value):
        # Runs on the field default too (validate_default=True inherited from the genesis
        # Options base), expanding the empty default into the full DEFAULT_IK_CONFIG; the
        # cache-key parity test pins this expansion.
        merged = merged_ik_config(value)
        try:
            json.dumps(merged)
        except TypeError as exc:
            raise ValueError(
                "TCRSRobotOptions.ik_config values must be JSON-serializable (plain lists/numbers/strings) "
                f"so they can enter the on-disk cache key ({exc})."
            ) from None
        return merged

    @model_validator(mode="after")
    def _validate_names(self):
        if len(set(self.dofs_name)) != len(self.dofs_name):
            raise ValueError("TCRSRobotOptions.dofs_name contains duplicates.")
        if len(set(self.links_name)) != len(self.links_name):
            raise ValueError("TCRSRobotOptions.links_name contains duplicates.")
        for role in (
            "pelvis_link",
            "left_foot_link",
            "right_foot_link",
            "left_hip_link",
            "right_hip_link",
            "left_knee_link",
            "right_knee_link",
        ):
            name = getattr(self, role)
            if name not in self.links_name:
                raise ValueError(f"TCRSRobotOptions.{role}='{name}' is not in links_name {self.links_name}.")
        for role, names in (
            ("left_leg_dofs_name", self.left_leg_dofs_name),
            ("right_leg_dofs_name", self.right_leg_dofs_name),
            ("waist_dofs_name", self.waist_dofs_name),
        ):
            missing = [n for n in names if n not in self.dofs_name]
            if missing:
                raise ValueError(f"TCRSRobotOptions.{role} entries {missing} are not in dofs_name.")
        return self

    # Derived clip-column indices (links resolved by name, never hardcoded).

    @property
    def pelvis_index(self) -> int:
        return self.links_name.index(self.pelvis_link)

    @property
    def left_foot_index(self) -> int:
        return self.links_name.index(self.left_foot_link)

    @property
    def right_foot_index(self) -> int:
        return self.links_name.index(self.right_foot_link)

    @property
    def left_hip_index(self) -> int:
        return self.links_name.index(self.left_hip_link)

    @property
    def right_hip_index(self) -> int:
        return self.links_name.index(self.right_hip_link)

    @property
    def left_knee_index(self) -> int:
        return self.links_name.index(self.left_knee_link)

    @property
    def right_knee_index(self) -> int:
        return self.links_name.index(self.right_knee_link)

    @property
    def mid_foot_offset(self) -> tuple[float, float, float]:
        """Midpoint of toe and heel contact points (upstream ``MID_FOOT_OFFSET``)."""
        return tuple(0.5 * (t + h) for t, h in zip(self.toe_offset, self.heel_offset))

    @property
    def pelvis_ankle_reach(self) -> float:
        return abs(self.hip_offset_z) + self.leg_max_length + self.pelvis_ankle_reach_slack

    @property
    def max_reach(self) -> float:
        """Reach cap used when shaping pelvis height (upstream ``_MAX_REACH``)."""
        return self.pelvis_ankle_reach - 0.01


class TCRSPlannerOptions(ConfigurableOptions):
    """TCRS pipeline knobs, mirroring the upstream CLI defaults.

    Not ported (dropped features): ``planner="plum"`` (imports a module that
    does not exist upstream), ``ik_backend`` curobo/drake, ``ik_mode="window"``
    trajectory IK, and ``rolling_interval`` viewer-time replanning.

    Range constraints live on the fields (``gt``/``ge``/``le``/``allow_inf_nan``),
    mirroring the upstream valid ranges.
    """

    # Contact detection (MotionClip).
    contact_z_threshold: float = Field(0.06, ge=0)
    contact_speed_threshold: float = Field(0.5, gt=0)

    # Terrain reference / footstep resolution.
    lookahead: float = Field(0.0, ge=0)
    smoothing_alpha: float = Field(0.08, gt=0, le=1)
    heel_offset_x: float = -0.06
    mid_offset_x: float = 0.15
    swing_floor_margin: float = 0.003

    # MPPI swing optimization.
    w_track: float = Field(10.0, allow_inf_nan=False)
    w_terrain: float = Field(1000.0, allow_inf_nan=False)
    h_clearance: float = Field(0.03, ge=0)
    mppi_n_knots: int = Field(5, gt=0)
    mppi_n_samples: int = Field(128, gt=0)
    mppi_n_iterations: int = Field(10, gt=0)
    mppi_temperature: float = Field(0.1, gt=0)

    # Post-MPPI leg/foot collision resolution and pelvis side clearance.
    leg_collision_iters: int = Field(4, ge=0)
    leg_collision_margin: float = 0.025
    shin_collision_radius: float = 0.06
    foot_collision_radius: float = 0.04
    collision_push_gain: float = 0.9
    collision_max_push: float = 0.06
    pelvis_side_margin: float = 0.04
    pelvis_side_gain: float = 0.5

    # Stance latching.
    stance_latch: bool = True
    stance_latch_ground_tol: float = 0.05

    # Per-frame Jacobian IK solver settings (max_iters, tol, damping,
    # step_size, posture/penetration weights, root_xy/z_weight, multiseed)
    # live in ``TCRSRobotOptions.ik_config["solver"]`` — override them there,
    # not here. Only the stabilizer post-pass is parameterized on this class:

    # Lower-body stance stabilizer post-pass.
    ik_lower_body_stabilizer: bool = True
    ik_stabilizer_stance_scale: float = 2.2
    ik_stabilizer_swing_scale: float = 0.60
    ik_stabilizer_root_xy_scale: float = 0.60
    ik_stabilizer_root_z_scale: float = 0.75
    ik_stabilizer_root_continuity_weight: float = 80.0
    ik_stabilizer_trigger_mm: float = 60.0
    ik_stabilizer_cooldown_frames: int = 6

    # Zero-phase Butterworth lowpass applied to the optimized output.
    lowpass_cutoff_hz: float = Field(5.0, ge=0)
    lowpass_order: int = Field(4, gt=0)

    # Quality threshold: frames whose per-frame IK max error exceeds
    # ``reject_ik_mm`` are counted in ``AdaptResult.ik_bad_frame_count``.
    reject_ik_mm: float = 200.0

    seed: int = 42
