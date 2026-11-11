"""The static CLI-convention checks in ``eden.utils.example_flags``, on the real ``examples/`` tree and on
crafted scripts (see examples/AGENTS.md).
"""

import pytest

from eden.utils import example_flags
from eden.utils.example_flags import (
    EXAMPLES,
    REPO_ROOT,
    check_arg_handoffs,
    check_declarations,
    check_documented_commands,
    check_headless_safe,
    check_path_defaults,
    check_task_builds,
)

_SOURCES = sorted(EXAMPLES.rglob("*.py"))
_DOCS = _SOURCES + sorted(EXAMPLES.rglob("*.md"))
_SOURCE_CHECKS = (check_declarations, check_headless_safe, check_path_defaults, check_arg_handoffs, check_task_builds)


def _id(path):
    return path.relative_to(REPO_ROOT).as_posix()


def test_the_examples_glob_matches_the_tree():
    # an empty glob (a path typo) would turn every parametrized test below into a skip
    assert len(_SOURCES) > 100


@pytest.mark.parametrize("path", _SOURCES, ids=_id)
def test_example_source_follows_the_convention(path):
    problems = []
    for check in _SOURCE_CHECKS:
        check(path, problems)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("doc", _DOCS, ids=_id)
def test_documented_commands_use_flags_that_exist(doc):
    problems = []
    check_documented_commands(doc, problems)
    assert not problems, "\n".join(problems)


@pytest.fixture
def demo_script(tmp_path, monkeypatch):
    """Write ``examples/demo.py`` under a throwaway repo root and return its path.

    ``REPO_ROOT`` is the checkers' only handle on the filesystem (it anchors every path they
    report and resolve), so pointing it at ``tmp_path`` is the filesystem boundary.
    """
    monkeypatch.setattr(example_flags, "REPO_ROOT", tmp_path)
    (tmp_path / "examples").mkdir()
    script = tmp_path / "examples" / "demo.py"

    def write(source: str):
        script.write_text(source)
        return script

    return write


def _problems(check, script) -> list[str]:
    problems = []
    check(script, problems)
    return problems


def test_arg_handoffs_report_names_no_parser_produces(demo_script):
    """The flag spellings (``rec``/``out``) are not the dests argparse derives."""
    script = demo_script(
        """import argparse
parser.add_argument("-r", "--record", action="store_true", help="record")
parser.add_argument("-o", "--output", default=None, help="output path")
chained = argparse.Namespace(rec=True, out=None)
parser.parse_args(["inference", "--rec"])
"""
    )
    assert _problems(check_arg_handoffs, script) == [
        "examples/demo.py:4: Namespace(rec=...) is no parser's dest (did you mean record?)",
        "examples/demo.py:4: Namespace(out=...) is no parser's dest (did you mean output?)",
        "examples/demo.py:5: parse_args passes --rec, which no parser accepts (did you mean --record?)",
    ]


def test_arg_handoffs_accept_a_parsed_namespace_copy(demo_script):
    script = demo_script(
        """import argparse
parser.add_argument("-r", "--record", action="store_true", help="record")
parser.add_argument("--motion-file", default=None, help="motion clip")
copied = argparse.Namespace(**vars(args))
parser.parse_args(["inference", "--record", "--motion-file", args.motion_file])
"""
    )
    assert _problems(check_arg_handoffs, script) == []


def test_path_defaults_report_missing_bundled_files(demo_script):
    script = demo_script(
        """import os
parser.add_argument(
    "--motion-file",
    default=os.path.join(os.path.dirname(__file__), "missing.npz"),
)
parser.add_argument("--config", default="examples/missing.yaml")
"""
    )
    assert _problems(check_path_defaults, script) == [
        "examples/demo.py:2: --motion-file defaults to examples/missing.npz, which does not exist",
        "examples/demo.py:6: --config defaults to examples/missing.yaml, which does not exist",
    ]


def test_path_defaults_skip_user_paths_and_outputs(demo_script):
    script = demo_script(
        """parser.add_argument("--data-path", default="eden/assets/amass")
parser.add_argument("-o", "--output", default="examples/result.mp4")
"""
    )
    assert _problems(check_path_defaults, script) == []


def test_task_builds_flag_get_then_call_only(demo_script):
    script = demo_script(
        """from eden.tasks import TASK_REGISTRY
cls = TASK_REGISTRY.get(args.task)  # the class itself is a legitimate ask
config = TASK_REGISTRY.get(args.task)()
config = TASK_REGISTRY.build(args.task, robot="oli")
"""
    )
    assert _problems(check_task_builds, script) == [
        "examples/demo.py:3: TASK_REGISTRY.get(args.task)() instantiates the task around its TaskMods; "
        "use TASK_REGISTRY.build(args.task) instead"
    ]


def test_task_builds_flag_only_a_call_of_a_registered_task_class(demo_script):
    """A base class, an ``isinstance``, an annotation, or a same-named class from outside ``eden.tasks`` is fine."""
    script = demo_script(
        """from eden.tasks.benchmark.reacher.config import ReacherConfig
from my_project.config import ReacherConfig as LocalReacher

class HarderReacher(ReacherConfig):
    pass

def make() -> ReacherConfig:
    return ReacherConfig()

assert isinstance(make(), ReacherConfig)
local = LocalReacher()
"""
    )
    assert _problems(check_task_builds, script) == [
        "examples/demo.py:8: ReacherConfig() instantiates task 'reacher' around its TaskMods; "
        "use TASK_REGISTRY.build('reacher') instead"
    ]


def test_task_builds_report_an_import_the_registry_cannot_answer_for(demo_script, task_package, isolated_task_registry):
    """A task name that clashes is a finding of its own, not a crash of the whole lint."""
    task_package(
        "clashing_tasks",
        {
            "reach.py": "from eden.tasks import TASK_REGISTRY\n"
            "@TASK_REGISTRY.register(name='reacher')\nclass MyReacher:\n    pass\n"
        },
    )
    isolated_task_registry.add_search_path("clashing_tasks")
    script = demo_script(
        """from eden.tasks.benchmark.move_forward.config import MoveForwardConfig
from eden.tasks.benchmark.reacher.config import ReacherConfig
config = MoveForwardConfig()
"""
    )
    with pytest.warns(UserWarning, match="'reacher' is registered in 2 locations"):
        problems = _problems(check_task_builds, script)
    assert problems == [
        "examples/demo.py:2: importing eden.tasks.benchmark.reacher.config.ReacherConfig: "
        "Task name 'reacher' is registered in 2 locations:",
        "examples/demo.py:3: MoveForwardConfig() instantiates task 'move_forward' around its TaskMods; "
        "use TASK_REGISTRY.build('move_forward') instead",
    ]
