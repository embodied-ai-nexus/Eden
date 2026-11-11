"""Tests for ``eden.utils.contacts``: grouped-contact aggregation and local link-index resolution, on plain tensors."""

from __future__ import annotations

import pytest
import torch

from eden.utils.contacts import aggregate_grouped_contacts, resolve_local_link_idx

SELF, OTHER = torch.tensor([10, 11]), torch.tensor([20, 21, 22])  # global link ids of the two groups


def _contacts(num_envs=1, max_contacts=1):
    return {
        "link_a": torch.full((num_envs, max_contacts), -1, dtype=torch.long),
        "link_b": torch.full((num_envs, max_contacts), -1, dtype=torch.long),
        "position": torch.zeros(num_envs, max_contacts, 3),
        "force": torch.zeros(num_envs, max_contacts, 3),
        "valid_mask": torch.zeros(num_envs, max_contacts, dtype=torch.bool),
    }


def _put(inp, env, slot, a, b, pos, force, valid=True):
    inp["link_a"][env, slot], inp["link_b"][env, slot] = a, b
    inp["position"][env, slot] = torch.tensor(pos)
    inp["force"][env, slot] = torch.tensor(force)
    inp["valid_mask"][env, slot] = valid


def test_no_valid_contact_aggregates_to_zeros():
    out = aggregate_grouped_contacts(**_contacts(2, 4), a_global=SELF, b_global=OTHER)
    assert out["position"].shape == (2, 2, 3, 3) and not out["valid"].any()
    assert (out["position"] == 0).all() and (out["force_norm_sum"] == 0).all()


@pytest.mark.parametrize(("a", "b"), [(11, 22), (22, 11)], ids=["self-as-a", "self-as-b"])
def test_a_contact_lands_on_its_self_other_pair_whichever_side_genesis_labels_self(a, b):
    inp = _contacts()
    _put(inp, 0, 0, a, b, [1.0, 2.0, 3.0], [3.0, 4.0, 0.0])
    out = aggregate_grouped_contacts(**inp, a_global=SELF, b_global=OTHER)
    assert out["valid"].sum() == 1 and out["valid"][0, 1, 2]
    torch.testing.assert_close(out["position"][0, 1, 2], torch.tensor([1.0, 2.0, 3.0]))
    assert out["force_norm_sum"][0, 1, 2].item() == pytest.approx(5.0)


def test_contacts_on_one_pair_average_their_positions_by_force():
    inp = _contacts(max_contacts=2)
    _put(inp, 0, 0, 10, 20, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    _put(inp, 0, 1, 10, 20, [10.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    out = aggregate_grouped_contacts(**inp, a_global=SELF[:1], b_global=OTHER[:1])
    torch.testing.assert_close(out["position"][0, 0, 0], torch.tensor([7.5, 0.0, 0.0]))
    assert out["force_norm_sum"][0, 0, 0].item() == pytest.approx(4.0)


def test_padding_and_unrelated_contacts_are_dropped():
    inp = _contacts(max_contacts=4)
    _put(inp, 0, 0, 10, 20, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    _put(inp, 0, 1, 10, 20, [1000.0, 0.0, 0.0], [5.0, 0.0, 0.0], valid=False)  # stale padding with matching ids
    _put(inp, 0, 2, 99, 88, [1.0, 1.0, 1.0], [1.0, 0.0, 0.0])  # neither group
    out = aggregate_grouped_contacts(**inp, a_global=SELF[:1], b_global=OTHER[:1])
    torch.testing.assert_close(out["position"][0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert out["force_norm_sum"][0, 0, 0].item() == pytest.approx(1.0) and out["valid"].sum() == 1


def test_non_finite_padding_does_not_poison_the_pair():
    """Padding slots hold whatever the solver buffer did; a NaN there once turned the weave_hoi contact reward NaN."""
    inp = _contacts(max_contacts=3)
    _put(inp, 0, 0, 10, 20, [1.0, 0.0, 0.0], [0.0, 2.0, 0.0])
    _put(inp, 0, 1, 10, 20, [float("nan")] * 3, [float("nan")] * 3, valid=False)  # stale NaN padding, matching ids
    _put(inp, 0, 2, 11, 21, [float("inf")] * 3, [float("inf")] * 3, valid=False)
    out = aggregate_grouped_contacts(**inp, a_global=SELF, b_global=OTHER)
    assert torch.isfinite(out["force_norm_sum"]).all() and torch.isfinite(out["position"]).all()
    assert out["force_norm_sum"][0, 0, 0].item() == pytest.approx(2.0) and out["valid"].sum() == 1
    torch.testing.assert_close(out["position"][0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_pairs_and_envs_aggregate_independently():
    """The bimanual case: two (self, other) pairs in one env, and nothing leaking into the next env."""
    inp = _contacts(num_envs=2, max_contacts=3)
    _put(inp, 0, 0, 10, 20, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    _put(inp, 0, 1, 21, 11, [0.0, 1.0, 0.0], [0.0, 2.0, 0.0])
    _put(inp, 0, 2, 11, 21, [2.0, 1.0, 0.0], [0.0, 0.0, 2.0])
    out = aggregate_grouped_contacts(**inp, a_global=SELF, b_global=OTHER[:2])
    assert out["valid"][0].tolist() == [[True, False], [False, True]] and not out["valid"][1].any()
    torch.testing.assert_close(out["position"][0, 1, 1], torch.tensor([1.0, 1.0, 0.0]))
    assert out["force_norm_sum"][0, 1, 1].item() == pytest.approx(4.0)


def test_gradients_are_finite_and_zero_for_an_env_without_contacts():
    inp = _contacts(num_envs=2, max_contacts=2)
    _put(inp, 0, 0, 10, 20, [1.0, 2.0, 3.0], [0.0, 0.0, 4.0])
    pos, force = inp["position"].requires_grad_(True), inp["force"].requires_grad_(True)
    aggregate_grouped_contacts(**inp, a_global=SELF[:1], b_global=OTHER[:1])["position"].sum().backward()
    assert torch.isfinite(pos.grad).all() and torch.isfinite(force.grad).all()
    torch.testing.assert_close(pos.grad[1], torch.zeros(2, 3))


def test_overlapping_groups_are_rejected():
    """Disjointness is what the orientation folding relies on; an overlap would double-count."""
    with pytest.raises(ValueError, match="disjoint"):
        aggregate_grouped_contacts(**_contacts(), a_global=torch.tensor([3, 7]), b_global=torch.tensor([7, 9]))


# ---------------------------------------------------------------------------
# resolve_local_link_idx
# ---------------------------------------------------------------------------


def test_resolve_local_link_idx_normalises_its_input():
    cpu = torch.device("cpu")
    assert torch.equal(resolve_local_link_idx(None, num_links=5, device=cpu), torch.arange(5))
    assert resolve_local_link_idx([2, 4, 1], num_links=5, device=cpu).tolist() == [2, 4, 1]
    ready = torch.tensor([0, 2, 4], dtype=torch.long)
    assert resolve_local_link_idx(ready, num_links=5, device=cpu) is ready  # hoisted to build(): no per-call copy


@pytest.mark.parametrize(
    ("idx", "error", "match"),
    [
        ([], ValueError, "cannot be empty"),
        ([5], IndexError, r"\[0, 5\)"),
        ([-1], IndexError, None),
        (torch.tensor([[0, 1]]), ValueError, "must be 1-D"),
        (torch.tensor([0, 5], dtype=torch.long), IndexError, r"\[0, 5\)"),  # the already-resolved form too
    ],
)
def test_resolve_local_link_idx_rejects_a_bad_selection(idx, error, match):
    with pytest.raises(error, match=match):
        resolve_local_link_idx(idx, num_links=5, device=torch.device("cpu"))


def test_link_bounds_are_checked_once_per_tensor(count_cuda_syncs):
    idx = torch.tensor([1, 2], dtype=torch.long, device="cuda")
    assert resolve_local_link_idx(idx, 4, idx.device) is idx
    assert count_cuda_syncs(lambda: resolve_local_link_idx(idx, 4, idx.device)) == 0
    with pytest.raises(IndexError, match="out of range"):
        resolve_local_link_idx(torch.tensor([1, 9], dtype=torch.long, device="cuda"), 4, idx.device)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_link_bounds_are_checked_again_after_an_in_place_write(device):
    idx = torch.tensor([1, 2], dtype=torch.long, device=device)
    assert resolve_local_link_idx(idx, 4, idx.device) is idx
    idx[1] = 9
    with pytest.raises(IndexError, match="out of range"):
        resolve_local_link_idx(idx, 4, idx.device)
    with torch.inference_mode():
        frozen = torch.tensor([1, 9], dtype=torch.long, device=device)
    with pytest.raises(IndexError, match="out of range"):  # no version counter: checked on every call
        resolve_local_link_idx(frozen, 4, frozen.device)
