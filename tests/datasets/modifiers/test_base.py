"""Tests for ``eden.datasets.modifiers.base``: registry keys and which motion keys a modifier applies to."""

import pytest
import torch

from eden.datasets.modifiers import (
    DATASET_MODIFIER_REGISTRY,
    Compose,
    Jitter,
    MirrorLR,
    Rotate,
    SpeedScale,
    Translate,
    TrimFrames,
)
from eden.datasets.modifiers.base import build_modifier


def test_each_modifier_is_registered_under_the_name_a_config_uses():
    expected = {
        "compose": Compose,
        "translate": Translate,
        "rotate": Rotate,
        "mirror_lr": MirrorLR,
        "speed_scale": SpeedScale,
        "trim_frames": TrimFrames,
        "jitter": Jitter,
    }
    assert {name: DATASET_MODIFIER_REGISTRY.get(name) for name in expected} == expected


@pytest.mark.parametrize("as_dict", [False, True], ids=["options", "dict"])
def test_build_modifier_takes_options_or_their_yaml_dict_form(as_dict):
    options = Rotate.configure(angle_deg=45.0)
    modifier = build_modifier(options.model_dump() if as_dict else options)
    assert type(modifier) is Rotate and modifier.angle_deg == 45.0


def test_an_empty_key_tuple_applies_to_no_key(make_sequence):
    """Explicitly empty is "none", not silently "all"."""
    seq = make_sequence()
    before = seq.base_pos_w.concatenated_tensor.clone()
    out = Translate(options=Translate.configure(offset=(1.0, 2.0, 3.0), motion_sequence_keys=())).apply({"robot": seq})
    assert torch.equal(out["robot"].base_pos_w.concatenated_tensor, before)


@pytest.mark.parametrize("wrapped", [False, True], ids=["modifier", "compose"])
def test_a_key_the_sequences_lack_is_a_key_error_naming_it(make_sequence, wrapped):
    translate = Translate.configure(offset=(1.0, 0.0, 0.0))
    if wrapped:
        modifier = Compose(options=Compose.configure(modifiers=[translate], motion_sequence_keys=("missing",)))
    else:
        modifier = Translate(options=Translate.configure(offset=(1.0, 0.0, 0.0), motion_sequence_keys=("missing",)))
    with pytest.raises(KeyError, match="missing"):
        modifier.apply({"robot": make_sequence()})


def test_a_missing_key_is_refused_before_any_key_is_modified(make_sequence):
    """Translate adds in place, so checking each key only when its turn came left the earlier ones shifted."""
    seq = make_sequence()
    before = seq.base_pos_w.concatenated_tensor.clone()
    translate = Translate(
        options=Translate.configure(offset=(1.0, 0.0, 0.0), motion_sequence_keys=("robot", "missing"))
    )
    with pytest.raises(KeyError, match=r"\['missing'\] not in motion_sequences \['robot'\]"):
        translate.apply({"robot": seq})
    assert torch.equal(seq.base_pos_w.concatenated_tensor, before)
