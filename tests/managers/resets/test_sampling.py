"""Tests for the reset-configuration sampler seam: the uniform default and the bind contract."""

from __future__ import annotations

import pytest
import torch

from eden.managers.resets.sampling import CategoryWeightedSampler, ResetConfigSampler, UniformResetConfigSampler

NUM_ENVS = 8


def _bound(num_configs: int = 5) -> UniformResetConfigSampler:
    sampler = UniformResetConfigSampler(NUM_ENVS, "cpu")
    sampler.bind(num_configs, consumer="reset_from_buffer")
    return sampler


class _RecordingSampler(ResetConfigSampler):
    """A minimal subclass: proposes ``num_configs - 1`` everywhere and records what the base hands it."""

    num_envs = NUM_ENVS
    device = "cpu"

    def __init__(self) -> None:
        self.on_bind_calls: list[int] = []
        self.draw_masks: list[torch.Tensor] = []
        self.assignment_at_draw: list[torch.Tensor] = []
        self.first_spawn_at_draw: list[torch.Tensor] = []

    def _on_bind(self, num_configs: int) -> None:
        self.on_bind_calls.append(num_configs)

    def _draw(self, selected):
        self.draw_masks.append(selected.clone())
        self.assignment_at_draw.append(self.assignment.clone())
        self.first_spawn_at_draw.append(self.first_spawn.clone())
        return torch.full((NUM_ENVS,), self._num_configs - 1)


def test_bind_runs_on_bind_once_after_validation():
    sampler = _RecordingSampler()
    with pytest.raises(ValueError):
        sampler.bind(0, consumer="spawn")
    assert sampler.on_bind_calls == []

    sampler.bind(7, consumer="spawn")
    assert sampler.on_bind_calls == [7]
    assert sampler.num_configs == 7


def test_num_configs_before_bind_raises():
    with pytest.raises(RuntimeError, match="_RecordingSampler reached num_configs unbound"):
        _ = _RecordingSampler().num_configs


@pytest.mark.parametrize(
    "envs_idx",
    [
        torch.tensor([1, 4, 6]),
        torch.tensor([False, True, False, False, True, False, True, False]),
    ],
    ids=["index", "mask"],
)
def test_sample_changes_only_the_selected_rows(envs_idx):
    sampler = _bound(num_configs=1000)
    sampler._assignment.fill_(-1)  # a sentinel no draw can produce

    assignment = sampler.sample(envs_idx)

    selected = torch.zeros(NUM_ENVS, dtype=torch.bool)
    selected[envs_idx] = True
    assert assignment.shape == (NUM_ENVS,)
    assert (assignment[~selected] == -1).all()
    assert ((assignment[selected] >= 0) & (assignment[selected] < 1000)).all()


def test_sample_slice_covers_every_env_within_range():
    assignment = _bound(num_configs=3).sample(slice(None))
    assert ((assignment >= 0) & (assignment < 3)).all()


def test_sample_draws_every_config():
    sampler = _bound(num_configs=4)
    seen = torch.cat([sampler.sample(slice(None)).clone() for _ in range(50)])
    assert set(seen.tolist()) == {0, 1, 2, 3}


def test_sample_before_bind_raises_naming_the_fix():
    with pytest.raises(RuntimeError, match=r"bind\(num_configs"):
        UniformResetConfigSampler(NUM_ENVS, "cpu").sample(slice(None))


def test_second_bind_raises_naming_both_consumers():
    sampler = _bound()
    with pytest.raises(RuntimeError, match="'reset_from_buffer'.*'spawn'"):
        sampler.bind(5, consumer="spawn")


@pytest.mark.parametrize("num_configs", [0, -3, 2.0, True])
def test_bind_rejects_a_non_positive_int(num_configs):
    with pytest.raises(ValueError, match=f"num_configs={num_configs!r}"):
        UniformResetConfigSampler(NUM_ENVS, "cpu").bind(num_configs, consumer="spawn")


@pytest.mark.parametrize(
    "envs_idx",
    [slice(2, 5), torch.tensor([2, 3, 4]), torch.arange(NUM_ENVS).ge(2) & torch.arange(NUM_ENVS).lt(5)],
    ids=["slice", "index", "mask"],
)
def test_draw_sees_one_mask_for_every_selector_and_the_finished_assignment(envs_idx):
    sampler = _RecordingSampler()
    sampler.bind(3, consumer="spawn")

    sampler.sample(envs_idx)
    sampler.sample(envs_idx)

    expected_mask = torch.tensor([False, False, True, True, True, False, False, False])
    assert all(torch.equal(mask, expected_mask) for mask in sampler.draw_masks)
    # The second draw sees the configurations the first one assigned: what a grader reads as "just finished".
    assert sampler.assignment_at_draw[1].tolist() == [0, 0, 2, 2, 2, 0, 0, 0]
    assert sampler.assignment.tolist() == [0, 0, 2, 2, 2, 0, 0, 0]
    # ``first_spawn`` tells a grader which of those rows hold no finished episode: every row at the first spawn.
    assert sampler.first_spawn_at_draw[0].all()
    assert torch.equal(sampler.first_spawn_at_draw[1], ~expected_mask)


class _StatefulSampler(_RecordingSampler):
    """Adds a checkpoint entry of its own through the base's hooks."""

    def _sampler_state(self) -> dict:
        return {"draws": len(self.draw_masks)}

    def _load_sampler_state(self, state: dict) -> None:
        self.loaded = state["draws"]


def test_the_checkpoint_carries_the_set_size_and_the_subclass_state():
    sampler = _StatefulSampler()
    sampler.bind(3, consumer="spawn")
    sampler.sample(slice(None))
    assert sampler.state_dict() == {"num_configs": 3, "draws": 1}


def test_a_load_makes_every_next_reset_a_first_spawn():
    """The episodes running at a load started from the pre-load state: no subclass can grade them."""
    sampler = _StatefulSampler()
    sampler.bind(3, consumer="spawn")
    sampler.sample(slice(None))
    assert not sampler.first_spawn.any()

    sampler.load_state_dict({"num_configs": 3, "draws": 5})
    assert sampler.loaded == 5
    sampler.sample(slice(0, 2))
    assert sampler.first_spawn_at_draw[-1].all()
    assert sampler.first_spawn.tolist() == [False, False] + [True] * (NUM_ENVS - 2)


def test_a_checkpoint_for_another_set_size_is_refused_before_the_subclass_reads_it():
    sampler = _StatefulSampler()
    sampler.bind(3, consumer="spawn")
    with pytest.raises(ValueError, match="saved for a set of 4 reset configurations, but spawn bound this one to 3"):
        sampler.load_state_dict({"num_configs": 4, "draws": 5})
    assert not hasattr(sampler, "loaded")


def test_a_base_that_shadows_the_checkpoint_fails_the_sampler_class():
    """A default ``state_dict`` on a curriculum-term base would drop the sampler's checkpoint without a word."""

    class _BaseWithACheckpoint:
        def state_dict(self) -> dict:
            return {}

    with pytest.raises(TypeError, match=r"state_dict\(\) resolves to .*_BaseWithACheckpoint\.state_dict.*re-point"):

        class _Shadowed(_BaseWithACheckpoint, _RecordingSampler):
            pass

    class _RePointed(_BaseWithACheckpoint, _RecordingSampler):
        state_dict = ResetConfigSampler.state_dict  # the fix the message names

    assert _RePointed.state_dict is ResetConfigSampler.state_dict


def test_a_sampler_that_overrides_the_checkpoint_is_refused():
    """The template is final: an override would skip the size check and the first-spawn reset."""
    with pytest.raises(
        TypeError, match=r"load_state_dict\(\) resolves to .*_Overriding\.load_state_dict.*_load_sampler_state"
    ):

        class _Overriding(_RecordingSampler):
            def load_state_dict(self, state: dict) -> None:
                pass


def test_the_registered_sampler_terms_checkpoint_through_the_base():
    import eden.tasks.extreme_parkour.custom_terms.curricula  # noqa: F401  (registers TerrainLevelCurriculum)
    from eden.managers.curriculum_manager import CURRICULUM_TERM_REGISTRY

    samplers = [
        term for _, term in CURRICULUM_TERM_REGISTRY if isinstance(term, type) and issubclass(term, ResetConfigSampler)
    ]
    assert {term.__name__ for term in samplers} >= {"SuccessGuidedSampling", "TerrainLevelCurriculum"}
    for term in samplers:
        assert term.state_dict is ResetConfigSampler.state_dict, term.__name__
        assert term.load_state_dict is ResetConfigSampler.load_state_dict, term.__name__


def test_assignment_before_bind_raises():
    with pytest.raises(RuntimeError, match="reached assignment unbound"):
        _ = _RecordingSampler().assignment


def test_category_weighted_draws_by_category_share_not_by_size():
    """Category 0 holds 90 of 100 configurations but carries a quarter of the weight; category 2 none."""
    category = torch.tensor([0] * 90 + [1] * 8 + [2] * 2)
    sampler = CategoryWeightedSampler(4096, "cpu", category, torch.tensor([1.0, 3.0, 0.0]))
    sampler.bind(100, consumer="test")
    torch.manual_seed(0)
    drawn = category[sampler.sample(slice(None))]
    share = torch.bincount(drawn, minlength=3).float() / 4096
    torch.testing.assert_close(share, torch.tensor([0.25, 0.75, 0.0]), atol=0.03, rtol=0)
    assert set(sampler.assignment[drawn == 1].tolist()) == set(range(90, 98)), "not every category-1 row was drawn"


def test_category_weighted_relabel_draws_by_the_new_labels():
    """A harvester's ``write_rows`` relabels configurations in place; the grouping made at bind must follow."""
    category = torch.tensor([0, 0, 1, 1])
    sampler = CategoryWeightedSampler(64, "cpu", category, torch.tensor([0.0, 1.0]))
    sampler.bind(4, consumer="test")
    category[:] = torch.tensor([1, 0, 0, 0])
    sampler.relabel(category)
    assert (sampler.sample(slice(None)) == 0).all(), "only configuration 0 is now in the weighted category"


@pytest.mark.parametrize(
    "category, weights, match",
    [
        (torch.tensor([0, 0, 1]), torch.tensor([0.0, 0.0]), r"at least one positive"),
        (torch.tensor([0, 0, 0]), torch.tensor([1.0, 1.0]), r"positive weight but no configuration: \[1\]"),
        (torch.tensor([0, 1, 2]), torch.tensor([1.0, 1.0]), r"do not match weights for 2 categories"),
        (torch.tensor([0, 1]), torch.tensor([float("nan"), 1.0]), r"must be finite"),
        (torch.tensor([0, 1]), torch.tensor([float("inf"), 1.0]), r"must be finite"),
    ],
    ids=["all-zero", "empty-category", "unlabelled-category", "nan", "inf"],
)
def test_category_weighted_rejects_weights_that_do_not_fit_the_labels(category, weights, match):
    with pytest.raises(ValueError, match=match):
        CategoryWeightedSampler(4, "cpu", category, weights)
