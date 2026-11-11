"""Add a custom termination term (illegal-contact) to an RL environment.

---
tier: recipe
capabilities: [term.termination]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/terms/termination.py -v
---
"""

from argparse import ArgumentParser
import genesis as gs
import eden as en
from eden.envs.base import RLEnvBase
from eden.options import EnvOptions

from eden.managers.termination_manager import TerminationManagerOptions
from eden.managers.termination_manager import (
    TerminationTermFuncWrapper as TerminationTerm,
)


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            env_spacing=(1.5, 1.5),
        ),
        scene_options=en.scenes.Empty(
            robot=en.robots.Yam(),
        ),
        termination_options=TerminationManagerOptions(
            illegal_contact=TerminationTerm.configure(
                func=en.terminations.illegal_contact,
                params={
                    "entity_name": "robot",
                    "illegal_contact_entity_names": ["plane"],
                },
            ),
        ),
        show_viewer=args.vis,
    )
    env.build()
    env.reset()

    step = 0
    while True:
        env.step()
        step += 1
        if en.is_smoke() and step >= 5:
            break


if __name__ == "__main__":
    main()
