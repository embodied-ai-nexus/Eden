"""``eden.options._retired_fields``: the notes on fields Genesis options retired, and Eden options' own declarations."""

import genesis as gs
import pytest
from pydantic import ValidationError

from eden.options._retired_fields import GENESIS_RETIRED_FIELDS, retired_fields
from eden.options.materials import SPHLiquidMaterialOptions
from eden.options.options import ConfigurableOptions, unknown_fields_message
from eden.utils.configs import get_options_class_from_qualname


@pytest.mark.parametrize("qualname", sorted(GENESIS_RETIRED_FIELDS))
def test_every_noted_genesis_field_is_one_its_class_no_longer_has(qualname):
    """A note naming a field its class still has, or a class that moved, would word a wrong error, or none."""
    cls = get_options_class_from_qualname(qualname)
    assert GENESIS_RETIRED_FIELDS[qualname] and not set(GENESIS_RETIRED_FIELDS[qualname]) & set(cls.model_fields)


def test_a_genesis_subclass_inherits_the_notes_of_its_base():
    class Tuned(gs.options.SAPCouplerOptions):
        pass

    assert retired_fields(Tuned) == GENESIS_RETIRED_FIELDS["genesis.options.solvers.SAPCouplerOptions"]


def test_an_eden_option_cannot_retire_a_field_it_still_has():
    with pytest.raises(TypeError, match=r"declares \['mu'\] in _retired_fields but still has them as fields"):

        class Stale(SPHLiquidMaterialOptions):
            _retired_fields = {"mu": "renamed"}


class _Retiring(ConfigurableOptions):
    rate: float = 1.0
    _retired_fields = {"speed": "renamed 'rate'; set that instead", "gain": "gone; remove from the config"}


def test_unknown_fields_are_told_in_one_error_with_each_retired_note_once():
    with pytest.raises(ValidationError) as error:
        _Retiring(speed=1.0, gain=2.0, rat=3.0)
    (only,) = error.value.errors()
    assert only["loc"] == () and only["type"] == "extra_forbidden"
    message = only["msg"]
    assert message.startswith("'rat' is not a field of _Retiring. Did you mean 'rate'?")
    assert "Retired ['speed'] of _Retiring: renamed 'rate'; set that instead." in message
    assert "Retired ['gain'] of _Retiring: gone; remove from the config." in message
    assert message.count("Valid fields") == 1 and message.endswith("Valid fields: rate: float.")


def test_a_lone_unknown_field_keeps_its_location_and_context_precedes_the_field_list():
    with pytest.raises(ValidationError) as error:
        _Retiring(speed=1.0)
    assert error.value.errors()[0]["loc"] == ("speed",)
    assert unknown_fields_message(_Retiring, ["x"], context="Read as _Retiring.").endswith(
        "Read as _Retiring. Valid fields: rate: float."
    )
