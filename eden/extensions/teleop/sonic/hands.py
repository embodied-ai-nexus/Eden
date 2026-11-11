"""Trigger -> Dex3 hand joint targets (the reference's gripper gesture solver).

There is no Cartesian hand IK: the controller trigger fabricates synthetic
fingertip positions (thumb tip fixed, the middle tip snapped onto it when the
trigger passes 0.5), and the solver turns thumb-to-finger distances into a
gesture choice plus a linear interpolation from the open pose to a preset
closed pose. With trigger input only, the result is a binary open/close of
the middle-finger power grasp.
"""

from __future__ import annotations

import numpy as np

_DIST_THRESHOLD = 0.05  # grip dead zone


def _close_pose(side: str, thumb_rot: float, amps: tuple[float, float, float, float]) -> np.ndarray:
    """Build one gesture's closed pose the reference way.

    The thumb rotation (joint 0) gets the opposite sign on the right hand
    BEFORE the final full negation of the right hand's joint convention — so
    it ends up with the same sign on both hands, while joints 1..6 mirror.
    """
    q = np.zeros(7)
    q[0] = -thumb_rot if side == "L" else thumb_rot
    q[1] = q[2] = 0.7
    q[3:7] = -np.asarray(amps)
    return q if side == "L" else -q


# Per-gesture closed poses, per side. Order: thumb_0/1/2, [A1, B1, A2, B2].
_CLOSE_POSES = {
    side: {
        "index": _close_pose(side, 0.5, (1.5, 1.5, 0.6, 1.5)),
        "middle": _close_pose(side, 0.0, (1.0, 1.5, 1.0, 1.5)),
        "ring": _close_pose(side, -0.5, (0.6, 1.5, 1.5, 1.5)),
        "pinky": np.zeros(7),
    }
    for side in ("L", "R")
}

# Fingertip slots in the synthetic 25-point hand: 5 joints per finger, tips at
# base + 4 (thumb 4, index 9, middle 14, ring 19, pinky 24).
_TIP = {"thumb": 4, "index": 9, "middle": 14, "ring": 19, "pinky": 24}


def generate_finger_data(trigger: float, grip: float) -> np.ndarray:
    """Synthesize ``(25, 4, 4)`` fingertip transforms from controller inputs.

    The thumb tip sits at x=1; the middle tip joins it when the trigger passes
    0.5 (closing the middle-gesture distance to zero). ``grip`` is accepted
    for interface parity but unused, exactly like the reference.
    """
    del grip
    fingertips = np.zeros((25, 4, 4))
    fingertips[_TIP["thumb"], 0, 3] = 1.0
    if trigger > 0.5:
        fingertips[_TIP["middle"], 0, 3] = 1.0
    return fingertips


class G1GripperGestureSolver:
    """Distance-based gesture selection + lerp to a preset Dex3 closed pose."""

    def __init__(self, side: str) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def __call__(self, fingertips: np.ndarray) -> np.ndarray:
        """Fingertip transforms ``(25, 4, 4)`` -> 7-DOF Dex3 joint targets."""
        positions = np.asarray(fingertips)[:, :3, 3]
        thumb = positions[_TIP["thumb"]]
        grips = {}
        for finger in ("index", "middle", "ring", "pinky"):
            grip = float(np.clip(1.0 - np.linalg.norm(thumb - positions[_TIP[finger]]), 0.0, 1.0))
            grips[finger] = 0.0 if grip < _DIST_THRESHOLD else grip

        max_grip = max(grips.values())
        if max_grip == 0.0:
            return np.zeros(7)
        poses = _CLOSE_POSES[self.side]  # side convention already folded in
        for finger in ("index", "middle", "ring"):
            if grips[finger] == max_grip:
                return max_grip * poses[finger]
        return max_grip * poses["pinky"]


def hand_targets_from_triggers(left_trigger: float, right_trigger: float) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: both hands' Dex3 targets straight from the trigger values."""
    left = G1GripperGestureSolver("left")(generate_finger_data(left_trigger, 0.0))
    right = G1GripperGestureSolver("right")(generate_finger_data(right_trigger, 0.0))
    return left, right
