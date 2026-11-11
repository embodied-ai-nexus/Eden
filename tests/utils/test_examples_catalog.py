"""``eden.utils.examples_catalog``: every example's metadata header, and the catalog's query API over them."""

import pytest

from eden.utils import examples_catalog
from eden.utils.examples_catalog import (
    CAP_NAMESPACES,
    CATALOG,
    OPTIONAL_DEPS,
    REPO_ROOT,
    ExampleCatalog,
    HeaderError,
    entry_points,
    parse_header,
    validate,
)

_VALID = {"tier": "recipe", "capabilities": ["solver.rigid"], "hardware": "cpu", "run": "go", "purpose": "Fine."}


@pytest.mark.parametrize("path", entry_points(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_example_has_valid_header(path):
    meta = parse_header(path)  # raises HeaderError (naming the file) if the block is missing/malformed
    errs = validate(meta)
    assert not errs, f"{path.relative_to(REPO_ROOT)}: " + "; ".join(errs)


def test_catalog_indexes_every_entry_point():
    # also guards the glob against silently matching nothing
    assert len(CATALOG.specs()) == len(entry_points()) > 100


@pytest.mark.parametrize(
    "override, error",
    [
        ({"tier": "demo"}, "tier 'demo' not in"),
        ({"hardware": "tpu"}, "hardware 'tpu' not in"),
        ({"capabilities": []}, "capabilities must be a non-empty list"),
        ({"capabilities": ["rigid"]}, "capability 'rigid' is not 'namespace.leaf'"),
        ({"capabilities": ["physics.rigid"]}, "capability 'physics.rigid' has unknown namespace"),
        ({"optional_deps": ["not_a_dep"]}, "optional_dep 'not_a_dep' not in"),
        ({"run": "  "}, "run command is empty"),
    ],
)
def test_validate_names_each_violation(override, error):
    assert validate(_VALID) == []
    errs = validate({**_VALID, **override})
    assert len(errs) == 1 and errs[0].startswith(error), errs


def test_every_optional_dep_tag_is_declared_by_an_example():
    """No dead vocabulary in either direction.

    A wrong probe module only shows up on a machine that has the package, so deny it a place to
    hide: ``madrona`` probed a module that does not exist for as long as no example declared it.
    """
    declared = {d for s in CATALOG.specs() for d in s.optional_deps}
    assert set(OPTIONAL_DEPS) == declared, (
        f"tags with no example: {sorted(set(OPTIONAL_DEPS) - declared)}; "
        f"undeclared in OPTIONAL_DEPS: {sorted(declared - set(OPTIONAL_DEPS))}"
    )


@pytest.mark.parametrize("dep", sorted(OPTIONAL_DEPS))
def test_optional_dep_probes_a_third_party_module_and_says_how_to_install_it(dep):
    """A first-party probe (``eden.extensions.policies.onnx`` for ``onnx``) always resolves, so the dep never
    skips; the install hint is what the skip message offers.
    """
    info = OPTIONAL_DEPS[dep]
    assert not info.module.startswith("eden"), f"{dep}: {info.module!r} is a first-party module"
    assert info.install.strip(), f"{dep}: empty install hint"


def test_unconditional_batch_renderer_examples_are_not_tagged_cpu():
    """The gs-madrona BatchRenderer is CUDA-only, so an example that always builds one cannot claim
    ``hardware: cpu``. Textual tripwire for the literal unconditional construction only.
    """
    for s in CATALOG.specs():
        if "renderer_options=BatchRenderer" in (REPO_ROOT / s.path).read_text():
            assert s.hardware in ("gpu", "gpu-fp64"), (
                f"{s.key}: unconditionally builds the CUDA-only BatchRenderer but is tagged hardware: {s.hardware}"
            )


def test_facets_list_the_values_present_on_each_filter_axis():
    facets = CATALOG.facets()
    assert set(facets) == {"tier", "hardware", "capability_kind", "capability", "optional_dep"}
    assert facets["tier"] == ["policy", "project", "recipe"]
    assert set(facets["capability_kind"]) <= CAP_NAMESPACES


def test_tier_filter_partitions_catalog():
    fresh = ExampleCatalog()
    assert sum(len(fresh.names(tier=t)) for t in ("project", "policy", "recipe")) == len(fresh.specs())


def test_filters_select_matching_examples():
    sensors = CATALOG.names(capability_kind="sensor")
    assert sensors and all(any(c.startswith("sensor.") for c in CATALOG.get(k).capabilities) for k in sensors)
    assert any("lidar" in k for k in CATALOG.names(capability="sensor.lidar"))
    assert CATALOG.names(hardware="real-robot"), "deployment examples should exist"
    assert CATALOG.names(tier="recipe", capability_kind="solver"), "physics recipes should exist"


def test_search():
    assert any("lidar" in k for k in CATALOG.search("lidar"))
    with pytest.raises(ValueError, match="empty search query"):
        CATALOG.search("   ")


def test_get_and_unknown_key():
    key = CATALOG.specs()[0].key
    assert CATALOG.get(key).key == key
    with pytest.raises(KeyError, match="unknown example 'recipes/nope/does_not_exist'"):
        CATALOG.get("recipes/nope/does_not_exist")


@pytest.mark.parametrize(
    "source, reason",
    [
        ("x = 1\n", "no module docstring"),
        ('"""Just prose, no metadata."""\n', "no '---'-fenced YAML block"),
        ('"""\n---\ntier: recipe\n---\n"""\n', "no one-line summary"),
        ('"""Summary.\n\n---\ntier: [unclosed\n---\n"""\n', "YAML block does not parse"),
        ('"""Summary.\n\n---\njust-a-string\n---\n"""\n', "YAML block is not a mapping"),
    ],
)
def test_parse_header_names_the_offending_file(tmp_path, source, reason):
    """A bad header must say *where* it is — else you hand-loop entry_points() to find it."""
    path = tmp_path / "broken_example.py"
    path.write_text(source)
    with pytest.raises(HeaderError) as excinfo:
        parse_header(path)
    assert "broken_example.py" in str(excinfo.value)
    assert excinfo.value.reason.startswith(reason)  # bare message, for callers that print the path


def test_catalog_reports_every_bad_header_at_once(tmp_path, monkeypatch):
    """One omission must not mask the rest, and must not surface as a bare KeyError."""
    (tmp_path / "good.py").write_text(
        '"""Fine.\n\n---\ntier: recipe\ncapabilities: [solver.rigid]\nhardware: cpu\nrun: go\n---\n"""\n'
    )
    (tmp_path / "no_block.py").write_text('"""Prose only."""\n')
    (tmp_path / "no_tier.py").write_text('"""Fine.\n\n---\ncapabilities: [solver.rigid]\nhardware: cpu\n---\n"""\n')
    # The catalog's filesystem roots: point them at a throwaway examples tree.
    monkeypatch.setattr(examples_catalog, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(examples_catalog, "EXAMPLES", tmp_path)

    with pytest.raises(ValueError) as excinfo:
        ExampleCatalog().specs()
    message = str(excinfo.value)
    assert "2 of 3 examples" in message
    assert "no_block.py: no '---'-fenced YAML block in docstring" in message
    assert "no_tier.py: missing fields: ['run', 'tier']" in message
    assert "good.py" not in message
