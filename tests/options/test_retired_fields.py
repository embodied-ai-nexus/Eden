"""``eden.options._retired_fields``: the notes on fields Genesis options retired."""

import genesis as gs
import pytest

from eden.options._retired_fields import GENESIS_RETIRED_FIELDS, retired_fields
from eden.utils.configs import get_options_class_from_qualname


@pytest.mark.parametrize("qualname", sorted(GENESIS_RETIRED_FIELDS))
def test_every_noted_genesis_field_is_one_its_class_no_longer_has(qualname):
    """A note naming a field its class still has, or a class that moved, would word a wrong error, or none."""
    cls = get_options_class_from_qualname(qualname)
    # Its own module, not a re-export: the lookup walks each class's own __module__ and __qualname__
    assert f"{cls.__module__}.{cls.__qualname__}" == qualname
    assert GENESIS_RETIRED_FIELDS[qualname] and not set(GENESIS_RETIRED_FIELDS[qualname]) & set(cls.model_fields)


def test_a_genesis_subclass_inherits_the_notes_of_its_base():
    class Tuned(gs.options.SAPCouplerOptions):
        pass

    assert retired_fields(Tuned) == GENESIS_RETIRED_FIELDS["genesis.options.solvers.SAPCouplerOptions"]


def test_an_eden_subclass_of_a_noted_genesis_class_keeps_its_notes_under_its_own():
    from eden.options.materials import SPHLiquidMaterialOptions

    assert retired_fields(SPHLiquidMaterialOptions) == SPHLiquidMaterialOptions._retired_fields
