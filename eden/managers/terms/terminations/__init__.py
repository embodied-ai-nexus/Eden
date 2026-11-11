"""Built-in termination terms."""

from eden.managers.terms.terminations.common import (
    time_out,
    on_success,
    on_failure,
    OnMetricSuccess,
    MetricAbove,
    illegal_contact,
    IllegalLinkContact,
    cumulative_reward_below_threshold,
)
from eden.managers.terms.terminations.locomotion import (
    base_orientation_roll_greater_than,
    base_orientation_pitch_greater_than,
    root_height_below_minimum,
)
from eden.managers.terms.terminations.manipulation import (
    object_out_of_reach,
)
from eden.managers.terms.terminations.tracking import (
    BadMotionLinkPosZOnly,
    bad_anchor_ori,
    bad_anchor_pos_z_only,
    motion_base_orientation_tracking_error_above,
    motion_base_position_tracking_error_above,
    motion_dofs_pos_tracking_error_above,
    motion_ends,
    motion_link_orientation_tracking_error_above,
    motion_link_position_tracking_error_above,
)
from eden.managers.terms.terminations.velocity_tracking import (
    fell_over,
)


__all__ = [
    "time_out",
    "on_success",
    "on_failure",
    "OnMetricSuccess",
    "MetricAbove",
    "illegal_contact",
    "IllegalLinkContact",
    "cumulative_reward_below_threshold",
    "base_orientation_roll_greater_than",
    "base_orientation_pitch_greater_than",
    "root_height_below_minimum",
    "object_out_of_reach",
    "motion_base_position_tracking_error_above",
    "motion_base_orientation_tracking_error_above",
    "motion_link_position_tracking_error_above",
    "motion_link_orientation_tracking_error_above",
    "motion_dofs_pos_tracking_error_above",
    "BadMotionLinkPosZOnly",
    "bad_anchor_ori",
    "bad_anchor_pos_z_only",
    "motion_ends",
    "fell_over",
]
