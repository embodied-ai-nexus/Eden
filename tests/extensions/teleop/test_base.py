"""``TeleopOperator.run``: the setup/step/teardown template, including a setup that never finishes."""

from __future__ import annotations

import pytest

from eden.extensions.teleop.base import TeleopOperator


class _Recorder(TeleopOperator):
    """The smallest concrete operator: records the lifecycle calls ``run`` makes."""

    fail_setup_with: type[BaseException] | None = None

    def __init__(self, options) -> None:
        super().__init__(options)
        self.calls: list[str] = []
        self._steps = 0

    def setup(self) -> None:
        self.calls.append("setup")
        if self.fail_setup_with is not None:
            raise self.fail_setup_with()

    def step(self) -> None:
        self.calls.append("step")
        self._steps += 1

    def should_stop(self) -> bool:
        return self._steps >= 2

    def on_interrupt(self) -> None:
        self.calls.append("on_interrupt")

    def teardown(self) -> None:
        self.calls.append("teardown")


def _operator(fail_setup_with=None) -> _Recorder:
    cls = type("Recorder", (_Recorder,), {"fail_setup_with": fail_setup_with})
    return cls(cls.configure())


def test_run_steps_until_should_stop_then_tears_down():
    op = _operator()
    op.run()
    assert op.calls == ["setup", "step", "step", "teardown"]


def test_ctrl_c_during_setup_still_tears_down():
    """Setup can block for minutes (headset wait); a Ctrl-C there skipped teardown and leaked its sockets."""
    op = _operator(fail_setup_with=KeyboardInterrupt)
    op.run()
    assert op.calls == ["setup", "on_interrupt", "teardown"]


def test_a_setup_that_exits_tears_down_and_propagates():
    op = _operator(fail_setup_with=SystemExit)
    with pytest.raises(SystemExit):
        op.run()
    assert op.calls == ["setup", "teardown"]
