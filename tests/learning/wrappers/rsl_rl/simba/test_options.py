"""SimBa actor / critic options: a plain-MLP ``hidden_dims`` is refused with the fields that size SimBa instead."""

from __future__ import annotations

import pytest

from eden.learning.wrappers.rsl_rl.simba import RslRlSimBaActorOptions, RslRlSimBaCriticOptions


def test_simba_options_refuse_mlp_hidden_dims():
    with pytest.raises(ValueError, match="hidden_dims=\\[256\\].*hidden_dim / num_blocks"):
        RslRlSimBaActorOptions(hidden_dims=[256])
    with pytest.raises(ValueError, match="SimBa has no plain MLP trunk"):
        RslRlSimBaCriticOptions(hidden_dims=[64, 64])
