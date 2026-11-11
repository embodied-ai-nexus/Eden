"""TCRS robot and planner option validation, and the dump digests that key cached adaptations."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from eden.options.extensions.motion_optimization import TCRSPlannerOptions, TCRSRobotOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME

_DOFS, _LINKS = list(G1_DOFS29_NAME), list(G1_DOFS29_TRACKED_LINKS_NAME)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dofs_name": _DOFS, "links_name": _LINKS, "left_foot_link": "no_such_link"}, "left_foot_link"),
        ({"dofs_name": [n for n in _DOFS if n != "left_knee_joint"], "links_name": _LINKS}, "left_leg_dofs_name"),
        ({"ik_config": {"solver": {"max_itters": 10}}}, "max_itters"),
        ({"ik_config": {"not_a_section": {}}}, "not_a_section"),
        # The pre-rename section must fail loudly, pointing at its new name.
        ({"ik_config": {"body_weights": {"pelvis": {"pos_weight": 2.0}}}}, "link_weights"),
        ({"ik_config": {"link_weights": {"pelvis": {"offset": np.array([0.1, 0.0, 0.0])}}}}, "JSON-serializable"),
    ],
)
def test_robot_options_reject_a_bad_value_naming_it(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TCRSRobotOptions(**kwargs)


def test_a_partial_ik_override_deep_merges_over_the_defaults():
    default = TCRSRobotOptions().ik_config
    spec = TCRSRobotOptions(
        ik_config={
            "solver": {"tol": 1e-4},
            "link_weights": {
                "pelvis": {"pos_weight": 2.0},  # merges over the default entry
                "torso_link": {"pos_weight": 1.0, "rot_weight": 0.5},  # a new body name is allowed
            },
        }
    ).ik_config
    assert spec["solver"] == {**default["solver"], "tol": 1e-4}
    assert spec["link_weights"]["pelvis"] == {"pos_weight": 2.0, "rot_weight": 1.0}
    assert spec["link_weights"]["torso_link"] == {"pos_weight": 1.0, "rot_weight": 0.5}
    assert spec["link_weights"]["left_toe"] == default["link_weights"]["left_toe"]


def test_numpy_field_values_coerce_to_json_clean_floats():
    spec = TCRSRobotOptions(toe_offset=np.array([0.1, 0.0, 0.0]))
    assert spec.toe_offset == (0.1, 0.0, 0.0)
    json.dumps(spec.model_dump())  # the cache-key path


def test_a_dump_reconstructs_an_equal_instance():
    """The yaml save/load path: re-merging an already-merged ``ik_config`` must be idempotent."""
    spec = TCRSRobotOptions(ik_config={"solver": {"tol": 1e-4}})
    assert TCRSRobotOptions(**spec.model_dump()) == spec
    params = TCRSPlannerOptions(mppi_n_samples=64)
    assert TCRSPlannerOptions(**params.model_dump()) == params


def test_default_dump_digests_are_pinned():
    """These bytes key every cached adaptation: a change here needs a ``_TCRS_CACHE_VERSION`` bump."""

    def digest(options) -> str:
        return hashlib.sha256(json.dumps(options.model_dump(), sort_keys=True).encode()).hexdigest()

    assert digest(TCRSRobotOptions()) == "08872417da5c29a8500bd1791d763a0be7f05de91228564fca8879b70398e145"
    assert digest(TCRSPlannerOptions()) == "422aea97c8bb59c84cc51c8aa4baf125fe2373b3115508a5441cfe36667ea35a"
