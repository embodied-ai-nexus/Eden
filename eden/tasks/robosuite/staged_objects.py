"""The staged per-object MDP of the robosuite PickPlace port, shared with NutAssembly.

The two tasks are the same five-stage shape — reach, grasp, lift, hover, placed — over a different
object set and a different terminal predicate (PickPlace's ``in_bin``, NutAssembly's ``on_peg``), so
the stages are built here once and parameterized rather than written out per task.

robosuite's ``PickPlace.staged_rewards`` / ``NutAssembly.staged_rewards`` shape the carry phase as
``r_hover = r_lift + 0.2 (1 - tanh(10 d_xy))`` towards a *fixed* target point — the centre of the
object's target quadrant in bin2, or the matching peg:
:class:`~eden.tasks.robosuite.custom_terms.HoverOverPoint`.
"""

from __future__ import annotations

from eden import metrics, rewards
from eden.managers.terms.metrics.boolean import MetricTermAND, MetricTermNOT
from eden.options.managers.metrics import MetricTermOptions
from eden.options.managers.rewards import RewardTermOptions
from eden.options.robots import ManipulatorOptions
from eden.tasks.robosuite import custom_terms
from eden.terms import MetricTerm

#: robosuite's ``staged_rewards`` multipliers, in stage order: reach, grasp, lift, hover, placed.
#: robosuite ``max``-es the stages over the objects still to place; the ports sum one set per object,
#: which keeps the same ordering of states with a dense signal throughout.
STAGE_WEIGHTS = {"reach": 0.1, "grasp": 0.35, "lift": 0.5, "hover": 0.7, "placed": 1.0}


def staged_object_metrics(
    rig: ManipulatorOptions,
    name: str,
    *,
    lift_height: float,
    placed_key: str,
    placed_predicate: MetricTermOptions,
) -> dict[str, MetricTermOptions]:
    """The three metrics one object contributes: ``grasping_``, ``lifted_`` and the placed predicate.

    ``placed_predicate`` is the task's own geometric test (PickPlace's ``object_in_region`` over the
    bin2 quadrant, NutAssembly's ``ObjectNearPointXY`` around the peg); it is ANDed here with "the
    object is released", which is how robosuite reads a completed place. Only the placed metric
    contributes to task success.
    """
    return {
        f"grasping_{name}": custom_terms.PadsGrasping.configure(
            rig=rig.arm_name, object_name=name, contributes_to_success=False
        ),
        f"lifted_{name}": MetricTerm.configure(
            func=metrics.object_above_height,
            params={"entity_name": name, "height": lift_height, "relative": True},
            contributes_to_success=False,
        ),
        placed_key: MetricTermAND.configure(
            terms=[
                placed_predicate,
                MetricTermNOT.configure(term=custom_terms.PadsGrasping.configure(rig=rig.arm_name, object_name=name)),
            ]
        ),
    }


def staged_object_rewards(
    rig: ManipulatorOptions,
    name: str,
    *,
    target_xy: tuple[float, float],
    lift_height: float,
    placed_key: str,
) -> dict[str, RewardTermOptions]:
    """The ``staged_rewards`` robosuite computes for one object, weighted by :data:`STAGE_WEIGHTS`.

    The grasp, lift and placed stages read the metrics :func:`staged_object_metrics` declares for the
    same object, so the two must be built with the same ``name`` and ``placed_key``.
    """

    def metric(stage: str, metric_name: str) -> RewardTermOptions:
        return rewards.MetricReward.configure(metric_name=metric_name, weight=STAGE_WEIGHTS[stage])

    return {
        f"reach_{name}": custom_terms.ReachTanh.configure(
            rig=rig.arm_name, object_name=name, coef=10.0, weight=STAGE_WEIGHTS["reach"]
        ),
        f"grasp_{name}": metric("grasp", f"grasping_{name}"),
        f"lift_{name}": metric("lift", f"lifted_{name}"),
        f"hover_{name}": custom_terms.HoverOverPoint.configure(
            object_name=name,
            point_xy=target_xy,
            lift_height=lift_height,
            coef=10.0,
            weight=STAGE_WEIGHTS["hover"],
        ),
        placed_key: metric("placed", placed_key),
    }
