"""Tests for ``eden.managers.catalog``: it covers every registry, reports each term's parameters as
``configure()`` accepts them, and renders and searches them. The checked-in pages are pinned in
``tests/resources/test_term_catalog.py``, the ``eden terms`` command in ``tests/utils/cli/test_terms.py``.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import textwrap
from functools import partial

import pytest

from eden.managers import catalog as catalog_module
from eden.managers.catalog import (
    KIND_NAMES,
    KINDS,
    builtin_catalog,
    catalog,
    kind_shared_specs,
    render_markdown,
    render_text,
    search,
)
from eden.utils.registry import TermRegistry

# The shipped terms only: another test module in this session may have registered its own.
ENTRIES = builtin_catalog()
BY_NAME = {(entry.kind, entry.name): entry for entry in ENTRIES}


class TestCoverage:
    """A term missing from the catalog is invisible again."""

    def test_kinds_cover_every_term_registry_in_the_tree(self):
        """Identity, not name: two registries could share a ``_name``, and it is the object the catalog reads."""
        listed = {id(kind.registry()) for kind in KINDS}
        found = {}
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith("eden.") or module is None:
                continue
            for attr, value in vars(module).items():
                if isinstance(value, TermRegistry):
                    found.setdefault(id(value), f"{module_name}.{attr}")
        missing = [where for registry_id, where in found.items() if registry_id not in listed]
        assert not missing, f"term registries missing from catalog.KINDS: {sorted(missing)}"

    def test_every_kind_has_terms(self):
        """An empty kind means its terms' module was never imported — the catalog would omit them."""
        counts = {kind: 0 for kind in KIND_NAMES}
        for entry in ENTRIES:
            counts[entry.kind] += 1
        assert all(counts.values()), f"kinds with no terms: {[k for k, n in counts.items() if not n]}"

    def test_populated_by_covers_every_term_the_registry_holds(self):
        """``populated_by`` is what a cold registry imports before a lookup; a term defined elsewhere reads as missing.

        ``TERMINATION_TERM_REGISTRY`` names ``terms.metrics`` as well as ``terms.terminations``: seven of
        its terms are dual-registered from there.
        """
        uncovered = []
        for kind in KINDS:
            declared = kind.registry()._populated_by or (kind.module,)
            for entry in builtin_catalog(kind.name):
                if not any(entry.module == d or entry.module.startswith(f"{d}.") for d in declared):
                    uncovered.append(f"{kind.name}/{entry.name} defined in {entry.module}, declared {declared}")
        assert not uncovered, "terms outside their registry's populated_by:\n  " + "\n  ".join(sorted(uncovered))

    def test_every_term_module_in_the_tree_is_reachable_from_a_populated_by(self):
        """A term module no ``__init__`` imports registers nothing (e.g. ``observations/manipulation.py``).

        In a subprocess: "something else imported it first" is the bug's own camouflage in-process.
        """
        script = textwrap.dedent(
            """
            import json, sys
            from pathlib import Path
            from eden.managers.catalog import KINDS, REPO_ROOT

            for kind in KINDS:
                kind.registry().keys()  # force this registry's populated_by imports

            unreachable = []
            for path in sorted((REPO_ROOT / "eden" / "managers" / "terms").rglob("*.py")):
                parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                if ".".join(parts) not in sys.modules:
                    unreachable.append(f"{'.'.join(parts)} ({path.relative_to(REPO_ROOT)})")
            print("UNREACHABLE=" + json.dumps(unreachable))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
        )
        assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
        marker = [line for line in result.stdout.splitlines() if line.startswith("UNREACHABLE=")]
        assert marker, f"probe printed no verdict:\n{result.stdout}\n{result.stderr}"
        unreachable = json.loads(marker[-1].removeprefix("UNREACHABLE="))

        assert not unreachable, (
            "term modules no registry's populated_by reaches, so nothing in them is registered:\n  "
            + "\n  ".join(unreachable)
            + "\nImport each from its package __init__ (and regenerate eden/resources/term_catalog.md), "
            "or delete it."
        )

    def test_kind_filter_partitions_the_catalog(self):
        assert sum(len(builtin_catalog(kind)) for kind in KIND_NAMES) == len(ENTRIES)

    def test_unknown_kind_names_the_known_ones(self):
        with pytest.raises(ValueError) as exc_info:
            catalog("rewards")
        msg = str(exc_info.value)
        assert "rewards" in msg
        assert "reward" in msg and "observation" in msg


class TestEntries:
    """What each row says about a term has to match what ``configure()`` will accept."""

    def test_function_term_reports_required_params(self):
        """The worked example: two required params, neither of them documented anywhere."""
        entry = BY_NAME[("reward", "dofs_vel_limits")]
        assert entry.form == "function"
        assert [param.name for param in entry.params] == ["entity_name", "max_vel"]
        assert all(param.required for param in entry.params)
        assert entry.params[1].annotation == "float | torch.Tensor"
        assert entry.source.startswith("eden/managers/terms/rewards/common.py:")

    def test_class_term_reports_fields_with_defaults(self):
        entry = BY_NAME[("reward", "undesired_contacts")]
        assert entry.form == "class"
        assert entry.qualname == "UndesiredContacts"
        params = {param.name: param for param in entry.params}
        assert params["threshold"].default == "1.0"
        assert not any(param.required for param in entry.params), "a class param always has a class-attr default"

    def test_class_term_omits_the_fields_shared_by_its_kind(self):
        """``weight`` is on every reward term, so it belongs to the kind's header, not to 43 rows."""
        entry = BY_NAME[("reward", "undesired_contacts")]
        assert "weight" not in {param.name for param in entry.params}
        assert "weight" in {spec.name for spec in kind_shared_specs(KINDS[2])}

    def test_optional_params_carry_their_default(self):
        entry = BY_NAME[("reward", "action_rate_l2")]
        (param,) = entry.params
        assert (param.name, param.required, param.default) == ("action_term_name", False, "None")

    def test_manager_injected_params_are_not_listed(self):
        """``env`` and the event manager's ``envs_idx`` are supplied by the framework, not the author."""
        for entry in ENTRIES:
            names = {param.name for param in entry.params}
            assert not names & {"env", "envs_idx"}, f"{entry.kind}/{entry.name} lists a manager-injected param"

    def test_function_era_params_are_parameters_of_their_class(self):
        """A declared key that is not a field would be hoisted into an extra the class never reads."""
        declared = {}
        for kind in KINDS:
            for key, registered in kind.registry():
                if isinstance(registered, partial):
                    continue
                function_era = getattr(registered, "_function_era_params", frozenset())
                if function_era:
                    declared[key] = function_era
                    assert function_era <= set(registered.get_parameter_names()), key
        assert set(declared) >= {"metric_reward", "metric_observation", "on_metric_success"}

    def test_params_match_what_the_term_actually_accepts(self):
        """Cross-check every row against the signature/class it describes — no drift, no invention."""
        for entry in ENTRIES:
            registry = next(kind for kind in KINDS if kind.name == entry.kind).registry()
            registered = registry.get(entry.name)
            if entry.form == "function":
                accepted = set(inspect.signature(registered.keywords["func"]).parameters)
            else:
                accepted = set(registered.get_parameter_names()) | set(registered._extra_option_params)
            listed = {param.name for param in entry.params}
            assert listed <= accepted, f"{entry.kind}/{entry.name} lists {listed - accepted}"

    def test_summary_is_the_term_s_own_docstring_not_an_inherited_one(self):
        """An undocumented term reports nothing rather than describing its base class."""
        entry = BY_NAME[("noise_model", "uniform_noise")]
        assert entry.summary == "", "UniformNoise has no docstring of its own"
        assert BY_NAME[("reward", "undesired_contacts")].summary.startswith("Penalty count of links")

    def test_summary_does_not_leak_a_documented_intermediate_base(self):
        """Both share the documented ``_MotionGlobalBodyVelocityErrorExp``, which ``stop_at`` misses."""
        for name in (
            "motion_global_body_linear_velocity_error_exp",
            "motion_global_body_angular_velocity_error_exp",
        ):
            assert BY_NAME[("reward", name)].summary == "", name


class TestRendering:
    def test_json_round_trips_and_names_required_params(self):
        payload = json.loads(json.dumps([entry.to_dict() for entry in ENTRIES]))
        assert len(payload) == len(ENTRIES)
        record = next(r for r in payload if r["kind"] == "reward" and r["name"] == "dofs_vel_limits")
        assert record["required"] == ["entity_name", "max_vel"]
        assert record["form"] == "function"

    def test_text_rendering_names_every_term(self):
        text = render_text(ENTRIES)
        for entry in ENTRIES:
            assert entry.name in text

    def test_markdown_escapes_union_pipes(self):
        """An unescaped ``str | None`` would split the row; the width comes from each table's own header."""
        markdown = render_markdown(builtin_catalog("reward"))
        assert "`action_term_name: str \\| None = None`" in markdown
        expected = 0
        for line in markdown.splitlines():
            if line.startswith("| Term |"):
                expected = line.count("|")
            elif line.startswith("| `"):
                assert line.count("|") - line.count("\\|") == expected, f"row has the wrong column count: {line}"

    def test_markdown_marks_required_params_bold(self):
        markdown = render_markdown(builtin_catalog("reward"))
        assert "**`entity_name: str`**, **`max_vel: float \\| torch.Tensor`**" in markdown

    def test_full_catalog_header_claims_completeness(self):
        """The checked-in page spans every kind, so its intro must keep the 'complete list' claim."""
        markdown = render_markdown(builtin_catalog())
        assert "Every term registered in Eden" in markdown
        assert "This is the complete list" in markdown

    def test_single_kind_markdown_header_does_not_claim_completeness(self):
        """``eden terms --kind reward --markdown`` must not announce itself as every term in Eden."""
        markdown = render_markdown(builtin_catalog("reward"))
        assert "Every reward term registered in Eden" in markdown
        assert "the reward section only" in markdown
        # The exact full-catalog sentence must be absent — this view is not the complete list.
        assert "This is the complete list" not in markdown


class TestRequiresColumn:
    """A term's ``EnvOptions`` preconditions must be readable before the first build."""

    def test_entry_carries_the_declared_flags(self):
        entry = next(e for e in ENTRIES if e.name == "randomize_kp_kd_gains")
        assert entry.requires_env_options == ("batch_dofs_info",)

    def test_terms_without_preconditions_declare_none(self):
        entry = next(e for e in ENTRIES if e.name == "push_by_velocity")
        assert entry.requires_env_options == ()

    def test_every_declared_flag_is_a_real_env_options_field(self):
        """Whatever reaches a row is what a reader copies into ``env_options``."""
        from eden.options.envs import EnvOptions

        for entry in ENTRIES:
            unknown = [flag for flag in entry.requires_env_options if flag not in EnvOptions.model_fields]
            assert not unknown, f"{entry.name} declares non-EnvOptions field(s) {unknown}"

    def test_json_carries_the_flags(self):
        payload = json.loads(json.dumps([entry.to_dict() for entry in ENTRIES]))
        record = next(r for r in payload if r["name"] == "randomize_com_shift")
        assert record["requires_env_options"] == ["batch_links_info"]

    def test_text_rendering_names_the_requirement(self):
        text = render_text(builtin_catalog("event"))
        assert "requires:  EnvOptions(batch_dofs_info=True)" in text

    def test_markdown_adds_the_column_only_where_it_is_used(self):
        """A fifth column of em-dashes on every table would cost width to say nothing."""
        markdown = render_markdown(builtin_catalog("event"))
        assert "| Term | Form | Parameters | Requires | Summary |" in markdown
        assert "`EnvOptions(batch_links_info=True)`" in markdown
        # A kind with no preconditions anywhere keeps the original four columns.
        assert "| Term | Form | Parameters | Summary |" in render_markdown(builtin_catalog("reward"))

    def test_markdown_row_column_count_is_consistent(self):
        """Every row of a table that gained the column must gain a cell, or the table breaks."""
        markdown = render_markdown(builtin_catalog("event"))
        in_requires_table = False
        for line in markdown.splitlines():
            if line.startswith("| Term |"):
                in_requires_table = "| Requires |" in line
            elif line.startswith("| `"):
                expected = 6 if in_requires_table else 5
                assert line.count("|") - line.count("\\|") == expected, f"wrong column count: {line}"


def test_builtin_catalog_drops_out_of_tree_terms():
    """The registries are global, so anything imported can add to them — including these tests."""
    from eden.managers import REWARD_TERM_REGISTRY

    def out_of_tree_reward(env, *, gain: float):
        return gain

    REWARD_TERM_REGISTRY.register(out_of_tree_reward)
    try:
        names = {entry.name for entry in catalog("reward")}
        builtin_names = {entry.name for entry in builtin_catalog("reward")}
        assert "out_of_tree_reward" in names, "eden terms should list a user's own term"
        assert "out_of_tree_reward" not in builtin_names, "the shipped page should not"
    finally:
        REWARD_TERM_REGISTRY._obj_map.pop("out_of_tree_reward", None)


def test_builtin_catalog_drops_in_tree_task_and_extension_terms():
    """An ``eden.tasks.*`` term would leak into the checked-in page by import order."""
    from eden.managers import REWARD_TERM_REGISTRY

    def a_task_reward(env, *, gain: float):
        return gain

    a_task_reward.__module__ = "eden.tasks.some_task.config"
    REWARD_TERM_REGISTRY.register(a_task_reward)
    try:
        assert "a_task_reward" in {entry.name for entry in catalog("reward")}, "eden terms lists it"
        assert "a_task_reward" not in {entry.name for entry in builtin_catalog("reward")}, (
            "an eden.tasks.* term must not leak into the shipped page"
        )
    finally:
        REWARD_TERM_REGISTRY._obj_map.pop("a_task_reward", None)


class TestSearch:
    """``eden terms --grep``: the one-lookup form of the catalog."""

    def test_matches_a_term_name(self):
        assert "base_height" in {entry.name for entry in search(ENTRIES, "base_height")}

    def test_matches_a_parameter_name(self):
        """The question term names cannot answer."""
        matched = {entry.name for entry in search(catalog("reward"), "max_vel")}

        assert "dofs_vel_limits" in matched
        assert all(
            "max_vel" in {param.name for param in entry.params} for entry in search(catalog("reward"), "max_vel")
        )

    def test_matches_a_summary(self):
        """A term whose name shares nothing with the concept still has to be findable."""
        matched = search(ENTRIES, "watertight|quaternion|friction")

        assert matched, "expected at least one term to mention one of these in its summary"

    def test_is_case_insensitive(self):
        assert {entry.name for entry in search(ENTRIES, "BASE_HEIGHT")} == {
            entry.name for entry in search(ENTRIES, "base_height")
        }

    def test_is_a_regex_so_alternation_works(self):
        """Is there a term for X, under whatever name Eden gave it -- one call, not three."""
        either = {entry.name for entry in search(ENTRIES, "height|altitude")}

        assert either >= {entry.name for entry in search(ENTRIES, "height")}
        assert either >= {entry.name for entry in search(ENTRIES, "altitude")}

    def test_no_match_is_an_empty_result_not_an_error(self):
        assert search(ENTRIES, "definitely_not_a_term_anywhere") == ()

    def test_an_invalid_pattern_names_the_pattern_and_the_fix(self):
        """A bare ``re.error`` several frames in tells the reader nothing about ``--grep``."""
        with pytest.raises(ValueError, match=r"not a valid regular expression"):
            search(ENTRIES, "dofs[0")

    def test_preserves_catalog_order(self):
        matched = search(ENTRIES, ".")

        assert list(matched) == list(ENTRIES)


class TestFilteredRendering:
    """A filtered view must not claim to be the catalog."""

    def test_text_says_the_count_is_a_match_count(self):
        """``reward terms (3)`` after a ``--grep`` reads as "Eden ships 3 reward terms"."""
        entries = search(catalog("reward"), "height")

        assert "matching 'height'" in render_text(entries, grep="height")
        assert "matching" not in render_text(entries).split("\n")[0]

    def test_markdown_drops_the_complete_list_claim(self):
        entries = search(catalog("reward"), "height")

        page = render_markdown(entries, grep="height")

        assert "This is the complete list" not in page
        assert "match `height`" in page


class TestRegenerate:
    """``python -m eden.managers.catalog`` writes both checked-in pages, and only into a repo checkout."""

    @pytest.fixture
    def checkout(self, tmp_path, monkeypatch):
        """A minimal checkout: the two markers ``main()`` requires, and both pages redirected under it."""
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "docs").mkdir()
        monkeypatch.setattr(catalog_module, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(catalog_module, "CATALOG_PAGE", tmp_path / "eden" / "resources" / "term_catalog.md")
        monkeypatch.setattr(catalog_module, "DOCS_CATALOG_PAGE", tmp_path / "docs" / "api" / "terms" / "catalog.md")
        return tmp_path

    def test_the_checked_in_pages_pass_check(self):
        assert catalog_module.main(["--check"]) == 0

    def test_writes_both_pages_then_check_passes_until_one_goes_stale(self, checkout):
        assert catalog_module.main(["--check"]) == 1  # neither page exists yet

        assert catalog_module.main([]) == 0
        assert catalog_module.CATALOG_PAGE.read_text().startswith("<!-- Generated by")
        assert catalog_module.DOCS_CATALOG_PAGE.read_text().startswith("<!-- Generated by")
        assert catalog_module.main(["--check"]) == 0

        catalog_module.DOCS_CATALOG_PAGE.write_text("hand-edited\n")
        assert catalog_module.main(["--check"]) == 1

    @pytest.mark.parametrize("missing", ["pyproject.toml", "docs"])
    def test_refuses_outside_a_checkout_and_writes_nothing(self, checkout, capsys, missing):
        """An installed eden's ``REPO_ROOT`` is the site-packages parent; writing there would succeed silently."""
        target = checkout / missing
        target.unlink() if target.is_file() else target.rmdir()
        before = sorted(checkout.rglob("*"))

        assert catalog_module.main([]) == 2

        err = capsys.readouterr().err
        assert f"'{checkout}' lacks pyproject.toml or docs/" in err
        assert "git clone" in err
        assert sorted(checkout.rglob("*")) == before
