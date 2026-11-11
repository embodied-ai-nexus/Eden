"""PolaRiS's task rubrics as Eden success metrics.

PolaRiS (``polaris/environments/__init__.py``) scores each task with a ``Rubric``: criteria built from three
checkers, each counted once it has ever passed, some only after others have. Success is every criterion
reached; progress is the fraction reached. :class:`~eden.managers.terms.metrics.sequence.RubricMetricTerm`
keeps that bookkeeping, and the checkers map onto Eden metric terms:

=====================================  ==========================================================================
PolaRiS checker                        Eden criterion
=====================================  ==========================================================================
``reach(obj, 0.2)``                    :class:`EeNearEntity`: the gripper's ``base`` link (PolaRiS's ``ee_frame``,
                                       the Robotiq ``base_link``) within 20 cm of the object
``lift(obj, threshold)``               :func:`object_bottom_above_height`: the object's lowest point ``threshold``
                                       above the table. PolaRiS compares the object's origin with its spawn
                                       height; the scanned assets put their origins anywhere and Eden's
                                       objects are dropped onto the table at reset, so the height is measured
                                       from the table (``z = 0``) instead
``is_within_xy(a, b, fraction)``       the gripper open (``left_driver_joint`` < 0.1 rad, PolaRiS's
                                       ``finger_joint`` test) and :func:`object_a_within_b_xy`
=====================================  ==========================================================================
"""

from __future__ import annotations

from eden.terms import MetricTerm
from eden.managers.terms.metrics.articulation import joint_angle_at_target
from eden.managers.terms.metrics.boolean import MetricTermAND
from eden.managers.terms.metrics.containment import object_bottom_above_height
from eden.managers.terms.metrics.proximity import EeNearEntity, object_a_within_b_xy
from eden.managers.terms.metrics.sequence import RubricMetricTerm
from eden.options.managers.metrics import CriterionOptions, MetricManagerOptions

#: PolaRiS's ``reach`` threshold [m], the same in every task.
REACH_DISTANCE = 0.2
#: PolaRiS's open-gripper test: the Robotiq driver joint below this [rad] (it closes to 0.8).
OPEN_GRIPPER_ANGLE = 0.1


def reach(entity: str) -> CriterionOptions:
    """The gripper has come within :data:`REACH_DISTANCE` of ``entity``."""
    return CriterionOptions(
        name=f"reach_{entity}",
        term=EeNearEntity.configure(
            robot_name="gripper", ee_link_names=["base"], entity_name=entity, threshold=REACH_DISTANCE
        ),
    )


def lift(entity: str, height: float) -> CriterionOptions:
    """``entity`` has been lifted ``height`` [m] off the table, after the gripper reached it."""
    return CriterionOptions(
        name=f"lift_{entity}",
        term=MetricTerm.configure(func=object_bottom_above_height, params={"entity_name": entity, "height": height}),
        after=[f"reach_{entity}"],
    )


def within(entity: str, container: str, fraction: float, *, after: list[str]) -> CriterionOptions:
    """``fraction`` of ``entity``'s footprint lies over ``container``'s, with the gripper open."""
    return CriterionOptions(
        name=f"{entity}_in_{container}",
        term=MetricTermAND.configure(
            terms=[
                MetricTerm.configure(
                    func=joint_angle_at_target,
                    params={
                        "entity_name": "gripper",
                        "joint_name": "left_driver_joint",
                        "target_angle": 0.0,
                        "threshold": OPEN_GRIPPER_ANGLE,
                    },
                ),
                MetricTerm.configure(
                    func=object_a_within_b_xy,
                    params={"entity_a_name": entity, "entity_b_name": container, "min_fraction": fraction},
                ),
            ]
        ),
        after=after,
    )


def put_in(entity: str, container: str, *, lift_height: float, fraction: float = 0.8) -> list[CriterionOptions]:
    """PolaRiS's three-criterion chain: reach ``entity``, lift it, let it go over ``container``."""
    return [
        reach(entity),
        lift(entity, lift_height),
        within(entity, container, fraction, after=[f"lift_{entity}"]),
    ]


def rubric(criteria: list[CriterionOptions]) -> MetricManagerOptions:
    """A task's metric options: one success metric, ``rubric``, whose value is PolaRiS's progress score."""
    return MetricManagerOptions(rubric=RubricMetricTerm.configure(criteria=criteria))
