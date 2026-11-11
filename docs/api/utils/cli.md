# CLI Internals

The `eden` command is an argparse tree declared in one table. `eden/__main__.py` holds `COMMANDS`, one `Command`
per subcommand, and imports each handler from its area module under `eden/utils/cli/`; `eden.utils.cli.registry`
holds the `Command` shape and the build and dispatch helpers, and knows nothing about which commands exist. To add a
subcommand, write its handler in the area module it belongs to and add a `Command` entry to `COMMANDS`; the module
docstrings of `eden/__main__.py` and `eden/utils/cli/registry.py` describe the contract. For what each command
does, see the [CLI reference](../../reference/cli.md).

## Command table

::: eden.__main__
    options:
      show_root_heading: false
      members:
        - COMMANDS
        - main

## Command primitives

::: eden.utils.cli.registry
    options:
      show_root_heading: false

## Task commands

The handlers of `eden train`, `eden inference`, `eden export`, `eden deploy`, `eden preprocess` and `eden smoke`.

::: eden.utils.cli.tasks
    options:
      show_root_heading: false
      members:
        - train_task
        - inference_task
        - export_task
        - deploy_task
        - preprocess_task
        - smoke_task

## See also

- [CLI reference](../../reference/cli.md)
- [Contributing](../../contributing/index.md)
