"""Robot feedback for the operator station (``g1_debug`` channel, polled on demand).

The operator polls the robot's state only at mode transitions: entering
PLANNER_FROZEN_UPPER_BODY grabs the measured upper-body joints as the frozen
targets, and entering PLANNER_VR_3PT grabs the full measured configuration
for wrist recalibration. :class:`RobotResetWatcher` is a separate per-tick
subscriber for the robot's fall auto-reset counter — separate because
:meth:`FeedbackReader.poll_feedback` drains the conflated socket, and the
transition effects rely on finding a message there.
"""

from __future__ import annotations

import numpy as np

from eden.extensions.policies.sonic.wire import DEFAULT_DEBUG_PORT, G1DebugPoller

# The 17 upper-body joints in IsaacLab order, gathered from the MuJoCo-order
# ``body_q_measured`` (`upper_body_joint_isaaclab_order_in_mujoco_index`).
UPPER_BODY_MJ_GATHER = (12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)


class FeedbackReader:
    """Conflate-latest reader of the robot's ``g1_debug`` state."""

    def __init__(self, host: str = "localhost", port: int = DEFAULT_DEBUG_PORT) -> None:
        self._poller = G1DebugPoller(host=host, port=port)
        self.upper_body_position_target: np.ndarray | None = None
        self.left_hand_position_target: np.ndarray | None = None
        self.right_hand_position_target: np.ndarray | None = None
        self.full_body_q_measured: np.ndarray | None = None

    def poll_feedback(self, quiet: bool = False) -> bool:
        """Read the newest robot state (if any) into the target fields.

        Returns whether the measured body configuration arrived. ``quiet`` skips the
        miss message, for callers that retry every tick and report once themselves.
        """
        data = self._poller.get_latest()
        if data is None:
            if not quiet:
                print("[FeedbackReader] no feedback data received")
            self.upper_body_position_target = None
            self.left_hand_position_target = None
            self.right_hand_position_target = None
            self.full_body_q_measured = None
            return False

        body_q = data.get("body_q_measured")
        if body_q is not None:
            full = np.asarray(body_q, dtype=np.float64)
            self.full_body_q_measured = full
            self.upper_body_position_target = full[list(UPPER_BODY_MJ_GATHER)]
        else:
            if not quiet:
                print("[FeedbackReader] body_q_measured not in feedback data")
            self.full_body_q_measured = None
            self.upper_body_position_target = None

        left = data.get("left_hand_q_measured")
        right = data.get("right_hand_q_measured")
        self.left_hand_position_target = None if left is None else np.asarray(left, dtype=np.float64)
        self.right_hand_position_target = None if right is None else np.asarray(right, dtype=np.float64)
        return self.full_body_q_measured is not None

    def close(self) -> None:
        self._poller.close()


class RobotResetWatcher:
    """Watches ``g1_debug``'s ``reset_count`` for robot-side fall auto-resets.

    Owns its own conflate subscriber (PUB fans out to every subscriber). The
    counter is monotonic, so conflation can never hide an increment; the first
    observed value is the baseline, which also covers an operator (re)started
    after the robot already reset.
    """

    def __init__(self, host: str = "localhost", port: int = DEFAULT_DEBUG_PORT) -> None:
        self._poller = G1DebugPoller(host=host, port=port)
        self._last_count: int | None = None

    def poll(self) -> bool:
        """True exactly when a previously-seen ``reset_count`` has increased."""
        data = self._poller.get_latest()
        count = None if data is None else data.get("reset_count")
        if count is None:
            return False
        count = int(count)
        increased = self._last_count is not None and count > self._last_count
        self._last_count = count
        return increased

    def close(self) -> None:
        self._poller.close()
