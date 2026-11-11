"""Tests for ``ObservationManager``: groups, reset fan-out, history, scale, the cross-group raw cache,
``nan_policy``, group-level override logging, what the build refuses, and the stateful-term (``post_physics_refresh``)
contract.

Every manager is constructed standalone against the real built ``box_env``.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.modifiers.base import (
    NOISE_MODEL_REGISTRY,
    OBSERVATION_MODIFIER_REGISTRY,
    NoiseModel,
    ObservationModifier,
)
from eden.managers.modifiers.observations import ExponentialMovingAverage, ObservationDelay
from eden.managers.observation_manager import (
    OBSERVATION_TERM_REGISTRY,
    ObservationManager,
    ObservationTerm,
)
from eden.options.managers.observations import (
    ObservationGroupOptions,
    ObservationManagerOptions,
    ObservationTermOptions,
)
from eden.terms import ObsTerm

NUM_ENVS = 4
ODD_ONES = [0.0, 1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Test terms
# ---------------------------------------------------------------------------


@OBSERVATION_TERM_REGISTRY.register()
class ScriptedObs(ObservationTerm):
    """Delivers ``frame``, which a test writes (ones until then); counts its computes for the cache tests."""

    dim: int = 3
    tag: object = None  # an opaque compute parameter: two terms differing only in it must not share a cache

    def build(self) -> None:
        self.frame = torch.ones(self.num_envs, self.dim, device=self.device)
        self.computes = 0

    def compute(self) -> torch.Tensor:
        self.computes += 1
        return self.frame.clone()


@OBSERVATION_TERM_REGISTRY.register()
class ClockObs(ObservationTerm):
    """The smallest stateful term: ``k * dt`` after ``k`` refreshes, zero on reset."""

    def build(self) -> None:
        self._clock = torch.zeros(self.num_envs, 1, device=self.device)
        self.computes = 0

    def post_physics_refresh(self) -> None:
        self._clock.add_(self._env.dt)

    def reset(self, envs_idx=None) -> None:
        self._clock[slice(None) if envs_idx is None else envs_idx] = 0.0

    def compute(self) -> torch.Tensor:
        self.computes += 1
        return self._clock


@OBSERVATION_TERM_REGISTRY.register()
class NaNBuffer(ObservationTerm):
    """Returns its own buffer, NaN in env 0, the way a stateful term does: an in-place guard would corrupt it."""

    def build(self) -> None:
        self._buffer = torch.ones(self.num_envs, 2, device=self.device)
        self._buffer[0] = torch.nan

    def compute(self) -> torch.Tensor:
        return self._buffer


@OBSERVATION_TERM_REGISTRY.register()
class ImageObs(ObservationTerm):
    """A ``(num_envs, 2, 3)`` frame: more than one feature axis, so its group takes the general path."""

    def compute(self) -> torch.Tensor:
        return torch.arange(6.0, device=self.device).reshape(2, 3).expand(self.num_envs, 2, 3).contiguous()


@OBSERVATION_TERM_REGISTRY.register()
def box_height(env) -> torch.Tensor:
    """The box's world z, ``(num_envs, 1)``: a function term whose value a test sets through the real setter."""
    return env.entities["box"].get_pos()[:, 2:3]


@OBSERVATION_TERM_REGISTRY.register()
def shifted_box_height(env, shift: float) -> torch.Tensor:
    return env.entities["box"].get_pos()[:, 2:3] + shift


@OBSERVATION_TERM_REGISTRY.register()
def ramp_pair(env) -> torch.Tensor:
    """``[1, 2]`` in every env, so a per-component scale is visible."""
    return torch.tensor([1.0, 2.0], device=env.device).expand(env.num_envs, 2)


@OBSERVATION_TERM_REGISTRY.register()
def per_env_ones(env) -> torch.Tensor:
    """``(num_envs,)``: a batch dim and no feature axis."""
    return torch.ones(env.num_envs, device=env.device)


@OBSERVATION_TERM_REGISTRY.register()
def risky_ratio(env) -> torch.Tensor:
    """``x / x`` with ``x = env index % 2``: NaN in the even envs, 1.0 in the odd ones."""
    x = (torch.arange(env.num_envs, device=env.device) % 2).to(torch.float32).unsqueeze(-1)
    return x / x


@OBSERVATION_TERM_REGISTRY.register()
def signed_infinities(env) -> torch.Tensor:
    """Row 0: ``[+Inf, 2]``; row 1: ``[-Inf, 2]``; every other row ``[2, 2]``."""
    out = torch.full((env.num_envs, 2), 2.0, device=env.device)
    out[0, 0] = torch.inf
    out[1, 0] = -torch.inf
    return out


@OBSERVATION_TERM_REGISTRY.register()
def no_hit_depth(env) -> torch.Tensor:
    """A depth reading whose ``+inf`` means "nothing hit": env 0 sees nothing, the others 2 m."""
    out = torch.full((env.num_envs, 1), 2.0, device=env.device)
    out[0] = torch.inf
    return out


@OBSERVATION_TERM_REGISTRY.register()
def finite_pair(env) -> torch.Tensor:
    return torch.ones(env.num_envs, 2, device=env.device)


@OBSERVATION_TERM_REGISTRY.register()
def contact_counts(env) -> torch.Tensor:
    """An integer frame: nothing in it can be non-finite."""
    return torch.ones(env.num_envs, 3, dtype=torch.int64, device=env.device)


@NOISE_MODEL_REGISTRY.register()
class DriftNoise(NoiseModel):
    """Adds a per-env offset that grows by one per call and restarts at zero on ``reset``."""

    def __init__(self, env, options):
        super().__init__(env=env, options=options)
        self._drift = torch.zeros(env.num_envs, 1, device=env.device)

    def compute(self, data):
        self._drift += 1.0
        return data + self._drift

    def reset(self, envs_idx=None):
        self._drift[slice(None) if envs_idx is None else envs_idx] = 0.0


@NOISE_MODEL_REGISTRY.register()
class PlusHundredNoise(NoiseModel):
    def compute(self, data):
        return data + 100.0


@OBSERVATION_MODIFIER_REGISTRY.register()
class TimesTen(ObservationModifier):
    def compute(self, frame):
        return frame * 10.0


@OBSERVATION_MODIFIER_REGISTRY.register()
class PlusOne(ObservationModifier):
    def compute(self, frame):
        return frame + 1.0


@OBSERVATION_MODIFIER_REGISTRY.register()
class DropsAComponent(ObservationModifier):
    def compute(self, frame):
        return frame[:, :1]


@OBSERVATION_MODIFIER_REGISTRY.register()
class CastsToDouble(ObservationModifier):
    def compute(self, frame):
        return frame.double()


@OBSERVATION_MODIFIER_REGISTRY.register()
class RecordsItsInput(ObservationModifier):
    """Keeps the last frame it was handed, so a test can see where in the pipeline modifiers sit."""

    def compute(self, frame):
        self.seen = frame.clone()
        return frame


@OBSERVATION_MODIFIER_REGISTRY.register()
class RaisesAtBuild(ObservationModifier):
    def build(self, frame):
        raise RuntimeError("no")

    def compute(self, frame):
        return frame


def _manager(env, **group) -> ObservationManager:
    """A manager with one group, ``policy``, built from ``group`` (its fields and its terms alike)."""
    return ObservationManager(env=env, options=ObservationManagerOptions(policy=ObservationGroupOptions(**group)))


def _feed(mgr: ObservationManager, value: float) -> None:
    """Write ``value`` into every ``ScriptedObs`` frame of ``mgr``."""
    for _, _, term in mgr.iter_terms():
        if isinstance(term, ScriptedObs):
            term.frame.fill_(value)


def _computes(mgr: ObservationManager) -> int:
    return sum(getattr(term, "computes", 0) for _, _, term in mgr.iter_terms())


# ---------------------------------------------------------------------------
# Groups and reset fan-out
# ---------------------------------------------------------------------------


def test_a_group_concatenates_class_and_function_terms_in_declaration_order(box_env, set_box_heights):
    set_box_heights(box_env, [1.0, 2.0, 3.0, 4.0])
    mgr = _manager(box_env, cls=ScriptedObs.configure(), func=ObsTerm.configure(func=box_height))

    expected = torch.cat([torch.ones(NUM_ENVS, 3), torch.tensor([[1.0], [2.0], [3.0], [4.0]])], dim=1)
    torch.testing.assert_close(mgr.compute()["policy"], expected)


def test_reset_zeroes_a_class_terms_state_on_the_given_rows_only(box_env):
    mgr = _manager(box_env, clock=ClockObs.configure())
    for _ in range(3):
        mgr.post_physics_refresh()

    mgr.reset(envs_idx=torch.tensor([1, 3]))

    expected = torch.full((NUM_ENVS, 1), 3 * box_env.dt)
    expected[[1, 3]] = 0.0
    torch.testing.assert_close(mgr.compute()["policy"], expected)


def test_reset_reaches_the_noise_model_of_a_function_term(box_env, set_box_heights):
    set_box_heights(box_env, [0.0] * NUM_ENVS)
    mgr = _manager(box_env, h=ObsTerm.configure(func=box_height, noise=DriftNoise.configure()))
    mgr.compute()
    mgr.compute()

    mgr.reset(envs_idx=torch.tensor([0, 2]))

    assert mgr.compute()["policy"][:, 0].tolist() == [1.0, 3.0, 1.0, 3.0]


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["class", "function"])
def test_a_term_history_window_backfills_and_a_partial_reset_restarts_only_those_rows(box_env, set_box_heights, kind):
    """Flattened by default, like the group: ``(num_envs, history_length * dim)``."""
    if kind == "class":
        mgr = _manager(box_env, x=ScriptedObs.configure(dim=1, history_length=3))

        def feed(value):
            _feed(mgr, value)

    else:
        mgr = _manager(box_env, x=ObsTerm.configure(func=box_height, history_length=3))

        def feed(value):
            set_box_heights(box_env, [value] * NUM_ENVS)

    for value in (1.0, 2.0):
        feed(value)
        mgr.compute(update_history=True)

    mgr.reset(envs_idx=slice(0, 2))
    feed(3.0)

    assert mgr.compute(update_history=True)["policy"].tolist() == [[3.0] * 3] * 2 + [[1.0, 2.0, 3.0]] * 2


def test_a_group_history_window_can_be_reset_before_its_first_compute(box_env):
    """Group-level history is one buffer over the concatenated group, not a dict of per-term buffers."""
    mgr = _manager(box_env, history_length=2, a=ScriptedObs.configure(dim=1), b=ScriptedObs.configure(dim=2))

    mgr.reset()

    torch.testing.assert_close(mgr.compute(update_history=True)["policy"], torch.ones(NUM_ENVS, 6))


@pytest.mark.parametrize("level", ["term", "group"])
def test_a_snapshot_shows_the_fresh_frame_without_advancing_the_window(box_env, level):
    """``update_history=False`` is the phase-12 snapshot: the new frame, most recent, and the window unchanged."""
    window = dict(history_length=3, flatten_history_dim=False)
    if level == "term":
        mgr = _manager(box_env, x=ScriptedObs.configure(dim=1, **window))
    else:
        mgr = _manager(box_env, x=ScriptedObs.configure(dim=1), **window)
    for value in (1.0, 2.0):
        _feed(mgr, value)
        mgr.compute(update_history=True)

    _feed(mgr, 3.0)
    snapshot = mgr.compute(update_history=False)["policy"]
    _feed(mgr, 4.0)
    stepped = mgr.compute(update_history=True)["policy"]

    assert snapshot[:, :, 0].tolist() == [[1.0, 2.0, 3.0]] * NUM_ENVS
    assert stepped[:, :, 0].tolist() == [[1.0, 2.0, 4.0]] * NUM_ENVS  # the snapshot frame was never pushed


def test_a_flattened_snapshot_is_chronological(box_env):
    mgr = _manager(box_env, x=ScriptedObs.configure(dim=2, history_length=4))
    for value in (1.0, 2.0):
        _feed(mgr, value)
        mgr.compute(update_history=True)

    _feed(mgr, 3.0)
    snapshot = mgr.compute(update_history=False)["policy"]

    assert snapshot.tolist() == [[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]] * NUM_ENVS


def test_a_snapshot_as_the_first_compute_backfills_the_window(box_env):
    mgr = _manager(box_env, x=ScriptedObs.configure(dim=2, history_length=3, flatten_history_dim=False))
    _feed(mgr, 5.0)

    torch.testing.assert_close(mgr.compute(update_history=False)["policy"], torch.full((NUM_ENVS, 3, 2), 5.0))


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale, frame, effective",
    [
        (2.0, [2.0, 2.0, 2.0], 2.0),
        ([2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]),
        (1.0, [1.0, 1.0, 1.0], None),
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], None),  # identity short-circuits like the scalar 1.0
    ],
    ids=["scalar", "vector", "scalar-identity", "vector-identity"],
)
def test_scale_multiplies_per_component_and_reports_what_it_applies(box_env, scale, frame, effective):
    mgr = _manager(box_env, scaled=ScriptedObs.configure(scale=scale), plain=ScriptedObs.configure())
    term = mgr.get_term("scaled")

    expected = torch.cat([torch.tensor(frame).expand(NUM_ENVS, 3), torch.ones(NUM_ENVS, 3)], dim=1)
    torch.testing.assert_close(mgr.compute()["policy"], expected)
    if effective is None:
        assert term.effective_scale is None
    else:
        torch.testing.assert_close(torch.as_tensor(term.effective_scale), torch.tensor(effective))
    assert not hasattr(term, "scale")  # options-only: compute() cannot read it, or be deduped on it
    # A vector scale must not evict the group from the preallocated concat fast path, which has no
    # observable output of its own.
    assert "policy" in mgr._group_concat_buffer


def test_a_vector_scale_lands_on_every_frame_of_a_history_window(box_env):
    mgr = _manager(box_env, x=ScriptedObs.configure(scale=[2.0, 3.0, 4.0], history_length=2))

    assert mgr.compute(update_history=True)["policy"].tolist() == [[2.0, 3.0, 4.0, 2.0, 3.0, 4.0]] * NUM_ENVS


def test_a_vector_scale_through_the_function_term_configure(box_env):
    """``ObsTerm.configure(func=..., scale=[...])`` routes through a different ``_build_config_dict``."""
    mgr = _manager(box_env, ramp=ObsTerm.configure(func=ramp_pair, scale=[2.0, 3.0]))

    assert mgr.compute()["policy"].tolist() == [[2.0, 6.0]] * NUM_ENVS


@pytest.mark.parametrize("scale", [[2.0, 1.0], [1.0, 1.0]], ids=["plain", "all-ones"])
def test_a_wrong_length_vector_scale_fails_at_construction(box_env, scale):
    """The identity short-circuit must not swallow a length mismatch."""
    with pytest.raises(ValueError, match=r"'x' in group 'policy'.*feature dim 3.*contain 3 values; got 2"):
        _manager(box_env, x=ScriptedObs.configure(scale=scale))


def test_a_term_with_no_feature_axis_takes_a_scalar_scale_but_not_a_vector(box_env):
    """Length ``num_envs`` is the dangerous case: without the shape guard it would scale per env."""
    with pytest.raises(ValueError, match=r"'x' in group 'policy'.*no feature dim"):
        _manager(box_env, x=ObsTerm.configure(func=per_env_ones, scale=[1.0] * NUM_ENVS), concatenate_terms=False)

    mgr = _manager(box_env, x=ObsTerm.configure(func=per_env_ones, scale=2.0), concatenate_terms=False)
    assert mgr.compute()["policy"]["x"].tolist() == [2.0] * NUM_ENVS


def test_scale_is_resolved_at_construction_and_retuned_only_through_the_manager(box_env):
    mgr = _manager(box_env, x=ScriptedObs.configure(scale=2.0))

    mgr.get_term("x")._options.scale = 5.0  # the misuse the contract documents: inert
    torch.testing.assert_close(mgr.compute()["policy"], torch.full((NUM_ENVS, 3), 2.0))

    mgr.set_term_scale("policy", "x", 5.0)
    torch.testing.assert_close(mgr.compute()["policy"], torch.full((NUM_ENVS, 3), 5.0))


def test_set_term_scale_writes_the_applied_scale_and_the_options_together(box_env):
    mgr = _manager(box_env, x=ScriptedObs.configure(scale=2.0))
    term = mgr.get_term("x")

    mgr.set_term_scale("policy", "x", [2.0, 3.0, 4.0])
    assert term._options.scale == [2.0, 3.0, 4.0]
    torch.testing.assert_close(term.effective_scale, torch.tensor([2.0, 3.0, 4.0]))
    assert mgr.compute()["policy"].tolist() == [[2.0, 3.0, 4.0]] * NUM_ENVS

    mgr.set_term_scale("policy", "x", 1.0)
    assert (term._options.scale, term.effective_scale) == (1.0, None)
    torch.testing.assert_close(mgr.compute()["policy"], torch.ones(NUM_ENVS, 3))


def test_set_term_scale_rejects_a_bad_value_and_leaves_the_term_untouched(box_env):
    mgr = _manager(box_env, x=ScriptedObs.configure(scale=2.0))
    term = mgr.get_term("x")

    with pytest.raises(ValueError, match=r"'x' in group 'policy'.*feature dim 3.*contain 3 values; got 2"):
        mgr.set_term_scale("policy", "x", [2.0, 1.0])
    with pytest.raises(TypeError, match=r"'x' in group 'policy'.*must be a scalar.*got NoneType"):
        mgr.set_term_scale("policy", "x", None)

    assert (term._options.scale, term.effective_scale) == (2.0, 2.0)
    torch.testing.assert_close(mgr.compute()["policy"], torch.full((NUM_ENVS, 3), 2.0))


def test_set_term_scale_on_a_term_with_no_feature_axis_reports_a_bad_type_as_a_type_error(box_env):
    mgr = _manager(box_env, x=ObsTerm.configure(func=per_env_ones), concatenate_terms=False)

    with pytest.raises(TypeError, match=r"'x' in group 'policy'.*must be a scalar.*got bool"):
        mgr.set_term_scale("policy", "x", True)
    with pytest.raises(ValueError, match=r"'x' in group 'policy'.*no feature dim"):
        mgr.set_term_scale("policy", "x", [1.0] * NUM_ENVS)


def test_set_term_scale_names_what_is_available_for_a_bad_address(box_env):
    mgr = _manager(box_env, x=ScriptedObs.configure())

    with pytest.raises(ValueError, match=r"group 'critic' not found.*Available options: policy"):
        mgr.set_term_scale("critic", "x", 2.0)
    with pytest.raises(ValueError, match=r"'nope' not found in observation group 'policy'.*options: x"):
        mgr.set_term_scale("policy", "nope", 2.0)


def test_set_term_scale_retunes_one_group_and_reaches_function_terms(box_env):
    """The same term name in two groups is two instances; nothing lives on the term class."""
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(x=ScriptedObs.configure(), ramp=ObsTerm.configure(func=ramp_pair)),
            critic=ObservationGroupOptions(x=ScriptedObs.configure(), enable_corruption=False),
        ),
    )

    mgr.set_term_scale("policy", "x", [2.0, 3.0, 4.0])
    mgr.set_term_scale("policy", "ramp", [2.0, 3.0])

    obs = mgr.compute()
    assert obs["policy"].tolist() == [[2.0, 3.0, 4.0, 2.0, 6.0]] * NUM_ENVS
    torch.testing.assert_close(obs["critic"], torch.ones(NUM_ENVS, 3))


@pytest.mark.parametrize(
    "group, retuned, first_compute, message",
    [
        (dict(x=dict(history_length=3)), "x", True, r"'x' in group 'policy'.*its history buffer has already"),
        (dict(x={}, history_length=2), "x", True, r"'x' in group 'policy'.*its group's history buffer"),
        (dict(x=dict(history_length=3)), "x", False, r"already accumulated frames"),  # the snapshot path arms it
        (dict(x={}, second={}, history_length=2), "second", True, r"'second' in group 'policy'.*its group's history"),
    ],
    ids=["term-window", "group-window", "snapshot-path", "group-window-other-term"],
)
def test_set_term_scale_refuses_once_a_window_holds_frames(box_env, group, retuned, first_compute, message):
    """Retuning after frames are buffered would splice two scales into one window: 1.0 -> 5.0 gave [1,1,1, 5,5,5]."""
    fields = {key: value for key, value in group.items() if not isinstance(value, dict)}
    terms = {key: ScriptedObs.configure(**value) for key, value in group.items() if isinstance(value, dict)}
    mgr = _manager(box_env, **fields, **terms)
    mgr.compute(update_history=first_compute)

    with pytest.raises(RuntimeError, match=message):
        mgr.set_term_scale("policy", retuned, 5.0)


def test_set_term_scale_is_open_while_the_window_is_empty(box_env):
    """Before the first compute, after a full reset (where the training wrappers leave you), and with no history."""
    mgr = _manager(box_env, x=ScriptedObs.configure(history_length=2), plain=ScriptedObs.configure(dim=1))
    mgr.set_term_scale("policy", "x", [2.0, 3.0, 4.0])
    assert mgr.compute(update_history=True)["policy"][0].tolist() == [2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 1.0]

    mgr.reset()
    mgr.set_term_scale("policy", "x", 3.0)
    mgr.set_term_scale("policy", "plain", 5.0)  # no history: retunable after a compute
    assert mgr.compute(update_history=True)["policy"][0].tolist() == [3.0] * 6 + [5.0]

    # A partial reset leaves the other envs holding frames at the old scale, so it stays refused.
    mgr.reset(envs_idx=torch.tensor([0]))
    with pytest.raises(RuntimeError, match=r"already accumulated frames"):
        mgr.set_term_scale("policy", "x", 9.0)


# ---------------------------------------------------------------------------
# Group-level overrides: what gets logged, at which level
# ---------------------------------------------------------------------------


def _group_logs(env, caplog, group: ObservationGroupOptions) -> tuple[list[str], list[str]]:
    ObservationManager(env=env, options=ObservationManagerOptions(policy=group))
    info = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    return info, warnings


def test_group_history_over_default_terms_is_one_info_line(box_env, eden_caplog):
    group = ObservationGroupOptions(a=ScriptedObs.configure(), b=ObsTerm.configure(func=box_height), history_length=5)

    info, warnings = _group_logs(box_env, eden_caplog, group)

    assert warnings == []
    assert info == [
        "observation_options.policy: history_length=5, flatten_history_dim=True, backfill=True applied to 2 terms."
    ]


@pytest.mark.parametrize(
    "own, reported",
    [
        (dict(history_length=3), "own (history_length=3)"),
        (dict(history_length=5, flatten_history_dim=False), "own (flatten_history_dim=False)"),
        (dict(flatten_history_dim=False, backfill=False), "own (flatten_history_dim=False, backfill=False)"),
    ],
    ids=["length", "flatten", "flags-without-a-length"],
)
def test_group_history_replacing_a_terms_own_setting_warns_naming_the_term(box_env, eden_caplog, own, reported):
    group = ObservationGroupOptions(own=ScriptedObs.configure(**own), plain=ScriptedObs.configure(), history_length=5)

    _, warnings = _group_logs(box_env, eden_caplog, group)

    assert len(warnings) == 1
    assert warnings[0].startswith("observation_options.policy: the group's history_length=5")
    assert reported in warnings[0] and "plain" not in warnings[0]
    if "history_length" not in own:
        assert "history_length = None" in warnings[0], "the message must name a fix the author can type"


@pytest.mark.parametrize(
    "own",
    [dict(history_length=5, flatten_history_dim=True, backfill=True), dict(history_length=5)],
    ids=["all-three", "length-only"],
)
def test_group_history_matching_the_terms_own_does_not_warn(box_env, eden_caplog, own):
    _, warnings = _group_logs(
        box_env, eden_caplog, ObservationGroupOptions(same=ScriptedObs.configure(**own), history_length=5)
    )
    assert warnings == []


def test_group_history_still_overrides_the_term(box_env):
    group = ObservationGroupOptions(
        own=ScriptedObs.configure(history_length=3, flatten_history_dim=False), history_length=5
    )
    mgr = ObservationManager(env=box_env, options=ObservationManagerOptions(policy=group))

    options = mgr.get_term("own")._options
    assert (options.history_length, options.flatten_history_dim) == (5, True)


@pytest.mark.parametrize("given_by", ["construction", "assignment"])
def test_a_dict_spelled_term_reaches_both_group_overrides(box_env, eden_caplog, given_by):
    """An assigned term is a term of the group like any other, so the group's switches apply to it."""
    raw = {"name": "box_height", "noise": {"name": "plus_hundred_noise"}, "backfill": False}
    if given_by == "construction":
        group = ObservationGroupOptions(raw=raw, enable_corruption=False, history_length=4)
    else:
        group = ObservationGroupOptions(enable_corruption=False, history_length=4)
        group.raw = raw

    info, warnings = _group_logs(box_env, eden_caplog, group)

    assert any("the noise set on raw is not applied" in line for line in info)
    assert len(warnings) == 1 and "raw (backfill=False)" in warnings[0]


def test_no_group_override_logs_nothing(box_env, eden_caplog):
    assert _group_logs(box_env, eden_caplog, ObservationGroupOptions(a=ScriptedObs.configure())) == ([], [])


@pytest.mark.parametrize("uncorrupt_at", ["construction", "after"], ids=["constructor", "task-mod"])
def test_noise_an_uncorrupted_group_ignores_is_one_info_line(box_env, eden_caplog, uncorrupt_at):
    """Info, not a warning: switching corruption off is the documented way to build a clean critic."""
    noise = PlusHundredNoise.configure()
    terms = dict(
        noisy_a=ScriptedObs.configure(noise=noise),
        clean=ScriptedObs.configure(),
        noisy_b=ObsTerm.configure(func=box_height, noise=noise),
    )
    if uncorrupt_at == "construction":
        group = ObservationGroupOptions(**terms, enable_corruption=False)
    else:
        group = ObservationGroupOptions(**terms)
        group.enable_corruption = False

    info, warnings = _group_logs(box_env, eden_caplog, group)

    assert info == [
        "observation_options.policy: enable_corruption is False, so the noise set on noisy_a, noisy_b is not "
        "applied. Set observation_options.policy.enable_corruption = True if it was meant to be."
    ]
    assert warnings == []


def test_an_uncorrupted_group_drops_the_noise_at_build_and_keeps_it_in_the_config(box_env):
    """Following the logged fix — ``enable_corruption = True`` — then applies it."""
    noise = PlusHundredNoise.configure()
    group = ObservationGroupOptions(noisy=ScriptedObs.configure(noise=noise), enable_corruption=False)

    mgr = ObservationManager(env=box_env, options=ObservationManagerOptions(policy=group))
    torch.testing.assert_close(mgr.compute()["policy"], torch.ones(NUM_ENVS, 3))
    assert group.noisy.noise == noise

    group.enable_corruption = True
    mgr = ObservationManager(env=box_env, options=ObservationManagerOptions(policy=group))
    torch.testing.assert_close(mgr.compute()["policy"], torch.full((NUM_ENVS, 3), 101.0))


def test_the_terms_named_do_not_depend_on_how_the_edits_were_split(box_env, eden_caplog):
    """One ``updated()`` call or two must give the same group and the same line."""
    noise = PlusHundredNoise.configure()
    base = ObservationGroupOptions(edited=ScriptedObs.configure(), other=ScriptedObs.configure(noise=noise))

    one_call = base.updated(edited=base.edited.updated(noise=noise), enable_corruption=False)
    two_calls = base.updated(edited=base.edited.updated(noise=noise)).updated(enable_corruption=False)

    for group in (one_call, two_calls):
        eden_caplog.clear()
        info, _ = _group_logs(box_env, eden_caplog, group)
        assert len(info) == 1 and "the noise set on edited, other is not applied" in info[0]


# ---------------------------------------------------------------------------
# Cross-group raw cache
# ---------------------------------------------------------------------------


def _two_groups(env, policy, critic, **critic_group) -> ObservationManager:
    return ObservationManager(
        env=env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(obs=policy),
            critic=ObservationGroupOptions(obs=critic, **critic_group),
        ),
    )


@pytest.mark.parametrize(
    "policy, critic, critic_group",
    [
        ({}, {}, {}),
        ({}, {}, dict(enable_corruption=False)),
        (
            dict(noise=PlusHundredNoise.configure(), scale=2.0, clip=(-1.0, 1.0), history_length=4),
            dict(scale=5.0),
            {},
        ),
        (dict(scale=[2.0, 1.0, 3.0]), dict(scale=5.0), {}),
    ],
    ids=["identical", "group-fields", "every-post-compute-field", "vector-scale"],
)
def test_a_term_shared_by_two_groups_computes_once_per_step(box_env, policy, critic, critic_group):
    """Post-compute fields are applied per group after the one raw compute, so they cannot split the cache."""
    mgr = _two_groups(box_env, ScriptedObs.configure(**policy), ScriptedObs.configure(**critic), **critic_group)
    before = _computes(mgr)

    mgr.compute()

    assert _computes(mgr) - before == 1


@pytest.mark.parametrize(
    "policy, critic",
    [(dict(dim=3), dict(dim=4)), (dict(tag={1, 2}), dict(tag={1, 2}))],
    ids=["compute-param", "unhashable-param"],
)
def test_terms_whose_compute_config_differs_compute_separately(box_env, policy, critic):
    """An unhashable parameter cannot be keyed, so its term is simply not cached rather than raising."""
    mgr = _two_groups(box_env, ScriptedObs.configure(**policy), ScriptedObs.configure(**critic))
    before = _computes(mgr)

    mgr.compute()

    assert _computes(mgr) - before == 2


def test_a_function_terms_params_are_part_of_its_cache_key(box_env, set_box_heights):
    """``params`` is a *declared* field, not an extra: a signature built from extras alone would merge these."""
    set_box_heights(box_env, [1.0] * NUM_ENVS)
    mgr = _two_groups(
        box_env,
        ObsTerm.configure(func=shifted_box_height, params={"shift": 1.0}),
        ObsTerm.configure(func=shifted_box_height, params={"shift": 2.0}),
    )

    obs = mgr.compute()

    assert obs["policy"][:, 0].tolist() == [2.0] * NUM_ENVS
    assert obs["critic"][:, 0].tolist() == [3.0] * NUM_ENVS


def test_the_shared_compute_keeps_each_groups_noise_its_own(box_env):
    mgr = _two_groups(
        box_env,
        ScriptedObs.configure(noise=PlusHundredNoise.configure()),
        ScriptedObs.configure(),
        enable_corruption=False,
    )

    obs = mgr.compute()

    torch.testing.assert_close(obs["policy"], torch.full((NUM_ENVS, 3), 101.0))
    torch.testing.assert_close(obs["critic"], torch.ones(NUM_ENVS, 3))


def test_the_cache_serves_this_steps_state_every_step(box_env, set_box_heights):
    mgr = _two_groups(box_env, ObsTerm.configure(func=box_height), ObsTerm.configure(func=box_height))

    for height in (1.0, 5.0, 9.0):
        set_box_heights(box_env, [height] * NUM_ENVS)
        obs = mgr.compute()
        assert obs["policy"][:, 0].tolist() == obs["critic"][:, 0].tolist() == [height] * NUM_ENVS


def test_each_group_keeps_its_own_history_window_over_a_shared_term(box_env, set_box_heights):
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(obs=ObsTerm.configure(func=box_height), history_length=3),
            critic=ObservationGroupOptions(obs=ObsTerm.configure(func=box_height), history_length=3),
        ),
    )

    for height in (0.0, 1.0, 2.0, 3.0):
        set_box_heights(box_env, [height] * NUM_ENVS)
        obs = mgr.compute(update_history=True)

    assert obs["policy"].tolist() == obs["critic"].tolist() == [[1.0, 2.0, 3.0]] * NUM_ENVS


def test_without_a_per_group_modifier_the_groups_receive_one_shared_tensor(box_env, set_box_heights):
    """Pinned so a future defensive clone is a decision: the non-concatenated path hands both groups
    the one raw tensor a shared function term computed.
    """
    set_box_heights(box_env, [6.0] * NUM_ENVS)
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(obs=ObsTerm.configure(func=box_height), concatenate_terms=False),
            critic=ObservationGroupOptions(obs=ObsTerm.configure(func=box_height), concatenate_terms=False),
        ),
    )

    obs = mgr.compute()

    assert obs["policy"]["obs"] is obs["critic"]["obs"]
    assert obs["policy"]["obs"][:, 0].tolist() == [6.0] * NUM_ENVS


@pytest.mark.parametrize("term", [ScriptedObs.configure(dim=2), ImageObs.configure()], ids=["fast-path", "general"])
def test_each_step_returns_a_fresh_buffer(box_env, term):
    """A caller holding the previous step's observation across the next ``compute()`` must not see it rewritten."""
    mgr = _manager(box_env, x=term)
    first = mgr.compute()["policy"]
    held = first.clone()
    _feed(mgr, 7.0)

    second = mgr.compute()["policy"]

    assert second is not first
    torch.testing.assert_close(first, held)


# ---------------------------------------------------------------------------
# The modifier stage: between the raw frame and noise
# ---------------------------------------------------------------------------


PIPELINE_PATHS = pytest.mark.parametrize("history_length", [0, 2], ids=["concat-fast-path", "general-path"])


@PIPELINE_PATHS
def test_modifiers_run_in_list_order(box_env, history_length):
    mgr = _manager(
        box_env,
        x=ScriptedObs.configure(
            dim=1, modifiers=[TimesTen.configure(), PlusOne.configure()], history_length=history_length
        ),
    )
    _feed(mgr, 2.0)

    assert float(mgr.compute(update_history=True)["policy"][0, -1]) == 21.0  # 2 * 10 + 1, not (2 + 1) * 10


@PIPELINE_PATHS
def test_modifiers_take_the_raw_frame_and_noise_clip_and_scale_follow_in_that_order(box_env, history_length):
    mgr = _manager(
        box_env,
        x=ScriptedObs.configure(
            dim=1,
            modifiers=[RecordsItsInput.configure(), TimesTen.configure()],
            noise=PlusHundredNoise.configure(),
            clip=(0.0, 50.0),
            scale=0.5,
            history_length=history_length,
        ),
    )
    _feed(mgr, 2.0)

    obs = mgr.compute(update_history=True)["policy"]

    assert float(mgr.get_term("x")._modifiers[0].seen[0, 0]) == 2.0  # the raw frame, before noise
    # 2 * 10 = 20 -> + 100 = 120 -> clip to 50 -> * 0.5. Clip before noise gives 60, scale before clip 50.
    assert float(obs[0, -1]) == 25.0


def test_modifiers_run_at_eval_where_noise_does_not(eval_box_env):
    mgr = _manager(
        eval_box_env,
        x=ScriptedObs.configure(dim=1, modifiers=[TimesTen.configure()], noise=PlusHundredNoise.configure()),
    )
    _feed(mgr, 2.0)

    assert float(mgr.compute(update_history=True)["policy"][0, 0]) == 20.0


def test_a_modifier_on_one_group_leaves_the_shared_raw_frame_to_the_other(box_env):
    mgr = _two_groups(
        box_env, ScriptedObs.configure(dim=1, modifiers=[TimesTen.configure()]), ScriptedObs.configure(dim=1)
    )
    _feed(mgr, 3.0)
    before = _computes(mgr)

    obs = mgr.compute(update_history=True)

    assert _computes(mgr) - before == 1  # still one raw compute
    assert (float(obs["policy"][0, 0]), float(obs["critic"][0, 0])) == (30.0, 3.0)


def test_the_same_stateful_modifier_on_two_groups_keeps_one_state_per_group(box_env):
    """A shared state would advance twice per step: the groups would then disagree with the recurrence."""
    ema = ExponentialMovingAverage.configure(alpha=0.5)
    mgr = _two_groups(
        box_env, ScriptedObs.configure(dim=1, modifiers=[ema]), ScriptedObs.configure(dim=1, modifiers=[ema])
    )

    for value in (2.0, 6.0):
        _feed(mgr, value)
        obs = mgr.compute(update_history=True)

    assert obs["policy"][:, 0].tolist() == obs["critic"][:, 0].tolist() == [4.0] * NUM_ENVS


@pytest.mark.parametrize(
    "modifier, returned",
    [
        (DropsAComponent.configure(), r"a \(4, 1\) torch.float32 tensor"),
        (CastsToDouble.configure(), r"a \(4, 2\) torch.float64 tensor"),
    ],
    ids=["shape", "dtype"],
)
def test_a_modifier_that_changes_the_frame_is_refused_at_construction(box_env, modifier, returned):
    with pytest.raises(ValueError, match=rf"Modifier 1 \('\w+'\) of observation term 'policy.x' returned {returned}"):
        _manager(box_env, x=ScriptedObs.configure(dim=2, modifiers=[PlusOne.configure(), modifier]))


def test_a_modifier_failure_is_noted_with_the_modifier_and_the_term(box_env):
    with pytest.raises(ValueError, match="alpha is required") as constructing:
        _manager(
            box_env, x=ScriptedObs.configure(modifiers=[PlusOne.configure(), ExponentialMovingAverage.configure()])
        )
    with pytest.raises(RuntimeError, match="no") as building:
        _manager(box_env, x=ScriptedObs.configure(modifiers=[RaisesAtBuild.configure()]))

    assert "while constructing modifier 1 ('exponential_moving_average')" in constructing.value.__notes__
    assert any("ObservationManager term 'policy.x'" in note for note in constructing.value.__notes__)
    assert "while building modifier 0 ('raises_at_build') of observation term 'policy.x'" in building.value.__notes__


def test_a_noise_model_is_not_accepted_as_a_modifier():
    with pytest.raises(ValueError, match="ObservationModifierOptions"):
        ScriptedObs.configure(modifiers=[PlusHundredNoise.configure()])


def test_modifiers_under_a_group_history_window(box_env):
    """The delay and the window run on one clock, so the window holds the delayed frames of steps 2 and 3."""
    mgr = _manager(
        box_env,
        x=ScriptedObs.configure(dim=2, modifiers=[ObservationDelay.configure(control_steps=1)]),
        history_length=2,
    )

    for value in (1.0, 2.0, 3.0):
        _feed(mgr, value)
        obs = mgr.compute(update_history=True)["policy"]

    assert obs[0].reshape(2, 2)[:, 0].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# nan_policy
# ---------------------------------------------------------------------------


def test_nan_passes_through_a_group_without_a_policy(box_env):
    mgr = _manager(box_env, r=ObsTerm.configure(func=risky_ratio))

    assert torch.isnan(mgr.compute()["policy"][::2]).all()


def test_sanitize_delivers_zero_for_nan_and_both_infinities_silently(box_env, eden_caplog):
    mgr = _manager(
        box_env,
        nan_policy="sanitize",
        r=ObsTerm.configure(func=risky_ratio),
        s=ObsTerm.configure(func=signed_infinities),
    )

    obs = mgr.compute(update_history=True)["policy"]

    assert obs.tolist() == [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 2.0, 2.0], [1.0, 2.0, 2.0]]
    assert not eden_caplog.records


def test_warn_sanitizes_and_warns_once_per_term_in_each_group(box_env, eden_caplog):
    terms = dict(r=ObsTerm.configure(func=risky_ratio), s=ObsTerm.configure(func=signed_infinities))
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(**terms, nan_policy="warn"),
            critic=ObservationGroupOptions(**terms, nan_policy="warn"),
        ),
    )

    for _ in range(3):
        assert mgr.compute(update_history=True)["policy"][:, 0].tolist() == ODD_ONES

    messages = [record.getMessage() for record in eden_caplog.records]
    for path in ("policy.r", "policy.s", "critic.r", "critic.s"):
        assert sum(f"'{path}'" in message for message in messages) == 1, path
    assert any("'policy.r' produced NaN in 2 of 4 envs (env ids: 0, 2)" in message for message in messages)


def test_error_raises_naming_the_term_the_kinds_and_the_envs(box_env):
    mgr = _manager(box_env, nan_policy="error", s=ObsTerm.configure(func=signed_infinities))

    with pytest.raises(FloatingPointError, match=r"'policy\.s' produced \+Inf/-Inf in 2 of 4 envs \(env ids: 0, 1\)"):
        mgr.compute()


def test_a_policy_assigned_past_validation_is_refused_when_used(box_env):
    group = ObservationGroupOptions(r=ObsTerm.configure(func=risky_ratio), nan_policy="sanitize")
    mgr = ObservationManager(env=box_env, options=ObservationManagerOptions(policy=group))
    group.nan_policy = "sanitise"  # assignment on options is not validated

    with pytest.raises(ValueError, match=r"nan_policy='sanitise'; expected one of"):
        mgr.compute()


def test_the_policy_applies_at_eval_where_noise_does_not(eval_box_env):
    """A guard written as noise vanished whenever ``env.training`` was False."""
    mgr = _manager(
        eval_box_env, nan_policy="sanitize", r=ObsTerm.configure(func=risky_ratio, noise=PlusHundredNoise.configure())
    )

    assert mgr.compute(update_history=True)["policy"][:, 0].tolist() == ODD_ONES


@PIPELINE_PATHS
def test_the_policy_applies_to_what_the_term_delivers_after_clip_and_scale(box_env, history_length):
    """NaN survives clip and scale; the guard, last, replaces it."""
    mgr = _manager(
        box_env,
        nan_policy="sanitize",
        r=ObsTerm.configure(func=risky_ratio, clip=(0.5, 2.0), scale=4.0, history_length=history_length),
    )

    assert mgr.compute(update_history=True)["policy"][:, -1].tolist() == [0.0, 4.0, 0.0, 4.0]


@PIPELINE_PATHS
@pytest.mark.parametrize("nan_policy", ["sanitize", "warn", "error"])
def test_an_infinity_the_clip_bounds_is_not_a_finding(box_env, eden_caplog, nan_policy, history_length):
    """A depth term's no-hit ``+inf`` clipped to its range stays "far", with no report."""
    mgr = _manager(
        box_env,
        nan_policy=nan_policy,
        d=ObsTerm.configure(func=no_hit_depth, clip=(0.0, 10.0), history_length=history_length),
    )

    assert mgr.compute(update_history=True)["policy"][:, -1].tolist() == [10.0, 2.0, 2.0, 2.0]
    assert not eden_caplog.records


@pytest.mark.parametrize("nan_policy", ["disabled", "sanitize", "warn", "error"])
def test_an_infinity_the_clip_bounds_stays_far_behind_a_moving_average(box_env, eden_caplog, nan_policy):
    """A modifier that turned ``+inf`` into NaN would hand the clip a value it cannot bound."""
    mgr = _manager(
        box_env,
        nan_policy=nan_policy,
        d=ObsTerm.configure(
            func=no_hit_depth, clip=(0.0, 10.0), modifiers=[ExponentialMovingAverage.configure(alpha=0.5)]
        ),
    )

    for _ in range(3):
        assert mgr.compute(update_history=True)["policy"][:, 0].tolist() == [10.0, 2.0, 2.0, 2.0]
    assert not eden_caplog.records


def test_an_infinity_no_clip_bounds_is_replaced(box_env):
    mgr = _manager(box_env, nan_policy="sanitize", d=ObsTerm.configure(func=no_hit_depth))

    assert mgr.compute()["policy"][:, 0].tolist() == [0.0, 2.0, 2.0, 2.0]


def test_a_nan_inside_a_moving_average_is_delivered_as_zero_until_the_env_resets(box_env):
    """The guard cleans what the policy receives, not the average upstream of it."""
    mgr = _manager(
        box_env,
        nan_policy="sanitize",
        x=ScriptedObs.configure(dim=1, modifiers=[ExponentialMovingAverage.configure(alpha=0.5, backfill=False)]),
    )
    term = mgr.get_term("x")
    term.frame[0] = torch.nan
    mgr.compute(update_history=True)
    _feed(mgr, 2.0)

    # From zero: 0.5, then 0.5 * 2 + 0.5 * 0.5; env 0's average is still NaN inside.
    assert mgr.compute(update_history=True)["policy"][:, 0].tolist() == [0.0, 1.25, 1.25, 1.25]

    mgr.reset(torch.tensor([0]))
    assert mgr.compute(update_history=True)["policy"][:, 0].tolist() == [1.0, 1.625, 1.625, 1.625]


def test_the_guard_writes_neither_the_term_nor_the_other_groups_copy(box_env):
    """Sanitizing is out of place: the raw frame is the term's own buffer and is shared across groups."""
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(b=NaNBuffer.configure(), nan_policy="sanitize"),
            critic=ObservationGroupOptions(b=NaNBuffer.configure()),
        ),
    )

    obs = mgr.compute(update_history=True)

    assert obs["policy"][0].tolist() == [0.0, 0.0]
    assert torch.isnan(obs["critic"][0]).all()
    assert all(torch.isnan(term._buffer[0]).all() for _, _, term in mgr.iter_terms())


@pytest.mark.parametrize("nan_policy", ["sanitize", "error"])
def test_an_integer_frame_keeps_its_type_and_values(box_env, nan_policy):
    mgr = _manager(box_env, c=ObsTerm.configure(func=contact_counts), nan_policy=nan_policy, concatenate_terms=False)

    out = mgr.compute()["policy"]["c"]

    assert out.dtype == torch.int64 and torch.equal(out, torch.ones(NUM_ENVS, 3, dtype=torch.int64))


@pytest.mark.parametrize("nan_policy, syncs", [("sanitize", 0), ("warn", 1), ("error", 1)])
def test_the_gpu_sync_budget_is_per_group_not_per_term(cuda_stand_in_env, count_cuda_syncs, nan_policy, syncs):
    """Three clean terms: "sanitize" reads nothing back, "warn" and "error" one flag for the whole group."""
    clean = ObsTerm.configure(func=finite_pair)
    mgr = _manager(cuda_stand_in_env, nan_policy=nan_policy, a=clean, b=clean, c=clean)
    mgr.compute()

    assert count_cuda_syncs(lambda: mgr.compute(update_history=True)) == syncs


# ---------------------------------------------------------------------------
# Stateful terms: post_physics_refresh ticks, compute() reads
# ---------------------------------------------------------------------------


def test_the_refresh_ticks_every_class_term_instance_and_no_function_term(box_env):
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(clock=ClockObs.configure(), func=ObsTerm.configure(func=box_height)),
            critic=ObservationGroupOptions(clock=ClockObs.configure()),
        ),
    )

    mgr.post_physics_refresh()

    obs = mgr.compute()
    torch.testing.assert_close(obs["policy"][:, :1], torch.full((NUM_ENVS, 1), box_env.dt))
    torch.testing.assert_close(obs["critic"], torch.full((NUM_ENVS, 1), box_env.dt))
    # Stateless by construction: the function wrapper has no refresh hook to be reached by.
    assert not hasattr(mgr.get_term("func"), "post_physics_refresh")


def test_the_ticked_terms_are_listed_by_group_and_name_in_tick_order(box_env):
    """What a graded-phase CUDA graph watches beside the graded managers' terms: exactly what the refresh ticks."""
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(clock=ClockObs.configure(), func=ObsTerm.configure(func=box_height)),
            critic=ObservationGroupOptions(clock=ClockObs.configure()),
        ),
    )

    ticked = list(mgr.iter_post_physics_terms())

    assert [(group, name) for group, name, _ in ticked] == [("policy", "clock"), ("critic", "clock")]
    assert all(isinstance(term, ClockObs) for _, _, term in ticked)


def test_compute_is_a_read_however_often_it_runs(box_env):
    """Including the build-time shape probe, which reads zero."""
    mgr = _manager(box_env, clock=ClockObs.configure())
    torch.testing.assert_close(mgr.compute()["policy"], torch.zeros(NUM_ENVS, 1))

    mgr.post_physics_refresh()
    for _ in range(5):
        mgr.compute()
        mgr.compute(update_history=True)

    torch.testing.assert_close(mgr.compute()["policy"], torch.full((NUM_ENVS, 1), box_env.dt))


def test_the_cross_group_cache_stays_sound_for_a_stateful_term(box_env):
    """Both instances are ticked and one computes, so either stands for both."""
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(clock=ClockObs.configure()),
            critic=ObservationGroupOptions(clock=ClockObs.configure()),
        ),
    )
    for _ in range(4):
        mgr.post_physics_refresh()
    before = _computes(mgr)

    obs = mgr.compute(update_history=True)

    torch.testing.assert_close(obs["policy"], torch.full((NUM_ENVS, 1), 4 * box_env.dt))
    torch.testing.assert_close(obs["critic"], obs["policy"])
    assert _computes(mgr) - before == 1


# ---------------------------------------------------------------------------
# What the build refuses: a group field written after construction, a nameless term
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "terms, group_fields, writes, message",
    [
        (("a", "b"), {}, dict(terms_order=["a"]), r"terms_order = \['a', 'b'\] puts it last"),
        ((), {}, dict(terms_order=["a"]), r"Term 'a' is named in this group's terms_order"),
        (("a",), {}, dict(terms_order=["a", "a"]), r"terms_order names 'a' \(2 times\)"),
        (("a",), dict(history_length=3), dict(concatenate_terms=False), "Concatenate terms must be True"),
    ],
    ids=["a-term-left-out", "no-terms-at-all", "a-term-named-twice", "history-without-concatenation"],
)
def test_a_group_field_written_after_construction_is_checked_at_build(box_env, terms, group_fields, writes, message):
    """A declared field is written unchecked, so the build asks the group for its whole construction rule."""
    group = ObservationGroupOptions(**{name: ScriptedObs.configure() for name in terms}, **group_fields)
    for field, value in writes.items():
        setattr(group, field, value)

    with pytest.raises(ValueError, match=message):
        ObservationManager(env=box_env, options=ObservationManagerOptions(policy=group))


def test_a_nameless_term_is_refused_at_build_naming_the_group_and_the_entry(box_env):
    """``ObservationTermOptions()`` is a legal placeholder; the registry used to answer it as ``KeyError: None``."""
    group = ObservationGroupOptions(a=ScriptedObs.configure(), placeholder=ObservationTermOptions())

    with pytest.raises(ValueError) as excinfo:
        ObservationManager(env=box_env, options=ObservationManagerOptions(policy=group))
    message = str(excinfo.value)
    assert "ObservationManager term 'policy.placeholder' names no registered term (name=None)" in message, message
    assert "<TermClass>.configure(...)" in message, message


# ---------------------------------------------------------------------------
# Group views and lookup
# ---------------------------------------------------------------------------


def test_active_terms_is_the_first_seen_union_across_groups_and_group_term_names_a_copy(box_env):
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                a=ScriptedObs.configure(), b=ScriptedObs.configure(), c=ScriptedObs.configure()
            ),
            critic=ObservationGroupOptions(
                b=ScriptedObs.configure(), c=ScriptedObs.configure(), d=ScriptedObs.configure()
            ),
        ),
    )

    assert mgr.active_terms == ["a", "b", "c", "d"]
    view = mgr.group_term_names
    view.clear()
    assert mgr.group_term_names == {"policy": ["a", "b", "c"], "critic": ["b", "c", "d"]}


def test_get_term_miss_lists_the_terms_by_group(box_env):
    mgr = ObservationManager(
        env=box_env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(a=ScriptedObs.configure()),
            critic=ObservationGroupOptions(b=ScriptedObs.configure()),
        ),
    )

    with pytest.raises(ValueError, match=r"'not_a_term' not found.*Available options: policy: a, critic: b"):
        mgr.get_term("not_a_term")
