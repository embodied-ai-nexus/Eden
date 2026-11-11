"""The ``eden terms`` command: list every registered term and the parameters it accepts.

A thin CLI over :mod:`eden.managers.catalog`, which owns the introspection and both renderings —
so what this prints, what the generated catalog pages contain (``docs/api/terms/catalog.md`` in
full, and its shipped index ``eden/resources/term_catalog.md``), and what a ``configure()`` error
quotes back at you cannot disagree.
"""

from __future__ import annotations

import argparse
import json


def _configure_terms(parser: argparse.ArgumentParser) -> None:
    # Kind choices are resolved at parse time, not import time: naming them here would pull the
    # registries (and torch, and genesis) into every `eden --help`.
    parser.add_argument("-k", "--kind", help="Only this kind of term (e.g. reward, observation, event)")
    parser.add_argument(
        "-g",
        "--grep",
        metavar="PATTERN",
        help=(
            "Only terms whose name, parameters or summary match PATTERN (case-insensitive regex, "
            "so 'height|altitude' works). This is the one-lookup form of the catalog"
        ),
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Emit the catalog as JSON, for tooling")
    output.add_argument(
        "--markdown",
        action="store_true",
        help="Emit the catalog as the full docs page, parameters included (docs/api/terms/catalog.md)",
    )


def _run_terms(args) -> None:
    import sys

    from eden.managers.catalog import catalog, render_markdown, render_text, search

    # Diagnostics go to stderr throughout, so stdout carries only the catalog: `--json` is
    # advertised "for tooling", and a prose sentence on stdout is one a pipe cannot parse.
    try:
        entries = catalog(args.kind)
    except ValueError as e:
        # A mistyped --kind is a user error, and the message already lists the valid ones.
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if args.grep is not None:
        scope = f"{args.kind} " if args.kind else ""
        try:
            matched = search(entries, args.grep)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        if not matched:
            # Exit 1 with a message, the way grep does: a caller can branch on it, and a reader
            # is told what was actually searched rather than left with empty output.
            print(
                f"No {scope}term matches '{args.grep}' (searched {len(entries)} {scope}terms by name, "
                f"parameter and summary). Try a shorter pattern or an alternation "
                f"('height|altitude'), or `eden terms{f' --kind {args.kind}' if args.kind else ''}` "
                f"to list them all.",
                file=sys.stderr,
            )
            # Still a well-formed empty document on stdout under --json: "no matches" is a
            # result, and `eden terms --grep X --json | jq` should read it as one rather than
            # fail to parse. The exit code is what a script branches on either way.
            if args.json:
                print("[]")
            sys.exit(1)
        entries = matched

    if args.json:
        print(json.dumps([entry.to_dict() for entry in entries], indent=2))
    elif args.markdown:
        print(render_markdown(entries, grep=args.grep), end="")
    else:
        print(render_text(entries, grep=args.grep), end="")
