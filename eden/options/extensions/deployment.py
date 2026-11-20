"""Base configuration options for real-robot deployment backends."""

from __future__ import annotations

import math

from pydantic import model_validator

from eden.options.options import ConfigurableOptions, DynamicConfigurableOptions


class DeploymentOptions(DynamicConfigurableOptions):
    """
    Options for deployment extensions.

    Parameters
    ----------
    entity_name: str
        The name of the entity to deploy
    control_dt: float | None
        The control time step in seconds.
        Defaults to None, in which case it is determined by the control frequency, fallback to sim dt * decimation.
    control_freq: float | None
        The control frequency in Hz. Defaults to None, in which case it is determined by 1/control_dt.
    decimation: int | None
        The decimation factor. Defaults to None, in which case sim decimation is used.
    sync: bool
        Whether to synchronize the control loop. Defaults to True.
    connect_timeout_s: float
        The timeout for connecting to the robot. Defaults to 5.0 seconds.
    auto_yaw_align: bool
        Capture the robot's IMU yaw at START (the MuJoCo backend also captures
        it on ``reset``'s read) and remove it from subsequent world-frame
        observations so the deployment world frame matches the (yaw-aligned)
        world frame the policy was trained in. Defaults to True. Disable when an external estimator already provides
        a yaw-consistent base orientation (e.g. mocap).
    """

    entity_name: str = "robot"
    control_dt: float | None = None
    control_freq: float | None = None
    decimation: int | None = None
    sync: bool = True
    connect_timeout_s: float = 5.0
    auto_yaw_align: bool = True


class LinkBoxOptions(ConfigurableOptions):
    """An axis-aligned box that links of the deployed robot must stay inside, in metres.

    The positions are those of the links on the robot's scene entity, which the deployment syncs from each
    reading, so they are forward kinematics of the measured joints in the frame the task was trained in. For a
    fixed-base arm that is the training scene's world frame. A floating base without a ``base_pos`` reading keeps
    its spawn position on that entity and turns with the measured orientation. The yaw that ``auto_yaw_align``
    removes from that orientation is captured at START, so before START (the hold and the INIT ramp) x and y turn
    with the IMU's own heading, which on hardware is arbitrary. On a floating base, bound z alone unless the robot
    boots facing the trained heading.

    Attributes
    ----------
    entity_name : str | None
        The scene entity the links belong to: one the deployment syncs from its readings. In a composite rig, a
        part's name (``"robot"``, ``"hand"``). ``None`` (default): the deployed robot (the deployment's
        ``entity_name``; a rig's primary part, or its first part with joints when the primary has none).
    links : tuple[str, ...]
        Link names, globs or regexes of that entity. Every link they match is checked.
    lower : tuple[float | None, float | None, float | None]
        The box's lower corner (x, y, z). ``None`` leaves that axis open below.
    upper : tuple[float | None, float | None, float | None]
        The box's upper corner (x, y, z). ``None`` leaves that axis open above.
    """

    entity_name: str | None = None
    links: tuple[str, ...]
    lower: tuple[float | None, float | None, float | None] = (None, None, None)
    upper: tuple[float | None, float | None, float | None] = (None, None, None)

    @model_validator(mode="after")
    def _check_box(self) -> LinkBoxOptions:
        if not self.links:
            raise ValueError("LinkBoxOptions.links is empty, so the box would check nothing. Name at least one link.")
        for axis, low, high in zip("xyz", self.lower, self.upper):
            if any(v is not None and math.isnan(v) for v in (low, high)):
                raise ValueError(f"LinkBoxOptions: {axis} bound is NaN (lower {low}, upper {high}); use None for open.")
            if low is not None and high is not None and low > high:
                raise ValueError(
                    f"LinkBoxOptions: lower {axis} = {low} is above upper {axis} = {high}, so no position is inside."
                )
        return self


class DeploySafetyOptions(ConfigurableOptions):
    """The limits a deployment's safety layer applies on every control tick (``eden deploy --safety FILE``).

    The layer (:mod:`eden.extensions.deployment.utils.safety`) clamps the policy's joint targets before each send
    and checks each reading against the limits below. Every limit is off until it is set, except the clamp into the
    built robot's joint range, which always applies. Joint values are in radians (metres on a prismatic joint).

    The filter applies the send cap, then the measured cap, then the clamp into the range, each to the result of the
    one before, so a later bound wins where two disagree: the filtered target is always inside the (shrunk) range,
    and is inside the measured band unless that band reaches past the range. The backend's own filters run after
    it, in ``send_payload`` (``target_lpf``, the Franka's velocity clamp, the Oli's torque clamp). The send cap and
    the tracking error measure from the target the backend leaves in the payload, which on the Go1 is before its
    low-pass and on the Franka before its velocity clamp.

    A per-joint field takes a float, which applies to every joint, or a dict from joint names, globs or regexes to
    values. A joint that no key matches is not limited by that field, and a key that matches no joint is an error.

    Attributes
    ----------
    dofs_pos_margin : float
        The policy's targets are clamped into the robot's joint range shrunk by this much at each end. 0 (default)
        clamps to the range itself.
    max_dofs_step_from_measured : float | dict[str, float] | None
        The most a policy target may differ from the joint's measured position on that tick. It bounds the PD
        error, and so the force, that one target can ask for, except on a joint read within ``dofs_pos_margin`` of a
        limit or past it: there the clamp into the shrunk range runs after the cap and may pull the target further,
        by up to the margin plus how far past it the joint was read.
    max_dofs_step_from_sent : float | dict[str, float] | None
        The most a policy target may move from the target sent on the previous tick.
    limit_excess_tolerance : float
        How far past a joint limit the policy's raw target may ask before the tick counts toward
        ``limit_excess_steps`` (default 0.02).
    limit_excess_steps : int | None
        Stop the run when the policy's raw target is more than ``limit_excess_tolerance`` past a joint limit on this
        many consecutive ticks. The clamp keeps such a target in range either way; a policy that keeps asking is
        outside what it was trained on. ``None`` (default) never stops for it.
    max_dofs_tracking_error : float | dict[str, float] | None
        Stop the run when a joint's measured position is further than this from the target sent on the same tick,
        one control period before the reading.
    max_read_interval_s : float | None
        Stop the run when two consecutive readings are more than this many seconds apart.
    link_boxes : dict[str, LinkBoxOptions]
        Stop the run when a link leaves its box. The key names the box in messages and in the record.
    """

    dofs_pos_margin: float = 0.0
    max_dofs_step_from_measured: float | dict[str, float] | None = None
    max_dofs_step_from_sent: float | dict[str, float] | None = None
    limit_excess_tolerance: float = 0.02
    limit_excess_steps: int | None = None
    max_dofs_tracking_error: float | dict[str, float] | None = None
    max_read_interval_s: float | None = None
    link_boxes: dict[str, LinkBoxOptions] = {}

    @model_validator(mode="after")
    def _check_limits(self) -> DeploySafetyOptions:
        if not self.dofs_pos_margin >= 0.0:  # NaN fails this too
            raise ValueError(f"DeploySafetyOptions.dofs_pos_margin must be >= 0, got {self.dofs_pos_margin}.")
        if not self.limit_excess_tolerance >= 0.0:
            raise ValueError(
                f"DeploySafetyOptions.limit_excess_tolerance must be >= 0, got {self.limit_excess_tolerance}."
            )
        if self.limit_excess_steps is not None and self.limit_excess_steps < 1:
            raise ValueError(
                f"DeploySafetyOptions.limit_excess_steps must be >= 1, or None to turn it off; got "
                f"{self.limit_excess_steps}."
            )
        if self.max_read_interval_s is not None and not self.max_read_interval_s > 0.0:
            raise ValueError(
                f"DeploySafetyOptions.max_read_interval_s must be > 0, or None to turn it off; got "
                f"{self.max_read_interval_s}."
            )
        for field in ("max_dofs_step_from_measured", "max_dofs_step_from_sent", "max_dofs_tracking_error"):
            value = getattr(self, field)
            values = value.values() if isinstance(value, dict) else () if value is None else (value,)
            if bad := [v for v in values if not v > 0.0]:
                raise ValueError(
                    f"DeploySafetyOptions.{field} must be > 0 on every joint it names (leave a joint out to not limit "
                    f"it, or set the field to None to turn it off); got {bad} in {value!r}."
                )
        return self
