"""The LIBERO task catalog parsed from the upstream BDDL files: task lookup and suite ordering."""

from __future__ import annotations

import pytest

from eden.tasks.libero import catalog


def test_catalog_holds_the_130_uniquely_named_tasks():
    task_names = catalog.get_task_names()

    assert len(set(task_names)) == len(task_names) == 130
    for task_name in task_names:
        task_spec = catalog.get_task_spec(task_name)
        assert task_spec.name == task_name
        assert task_spec.suite_name in catalog.get_suite_names()
        assert task_spec.language_instruction and task_spec.entity_categories and task_spec.goal_state


def test_unknown_task_is_rejected():
    with pytest.raises(KeyError, match="Unknown LIBERO task"):
        catalog.get_task_spec("not_a_libero_task")


def test_suite_ordering_uses_the_canonical_permutation():
    suite_tasks = catalog.get_suite_task_names("libero_object")
    ordered = catalog.get_suite_task_names("libero_object", task_order_index=7)

    assert tuple(suite_tasks.index(task_name) for task_name in ordered) == (4, 2, 9, 7, 6, 8, 5, 1, 3, 0)


def test_libero_100_is_the_union_of_libero_90_and_libero_10():
    assert "libero_100" in catalog.get_suite_names()
    assert catalog.get_suite_task_names("libero_100") == (
        *catalog.get_suite_task_names("libero_90"),
        *catalog.get_suite_task_names("libero_10"),
    )


@pytest.mark.parametrize(
    ("suite_name", "task_order_index"),
    [("libero_object", -1), ("libero_object", 21), ("libero_90", 1)],
)
def test_suite_ordering_rejects_unsupported_indices(suite_name, task_order_index):
    with pytest.raises(IndexError):
        catalog.get_suite_task_names(suite_name, task_order_index=task_order_index)


def test_unknown_suite_is_rejected():
    with pytest.raises(KeyError, match="Unknown LIBERO suite"):
        catalog.get_suite_task_names("unknown")
