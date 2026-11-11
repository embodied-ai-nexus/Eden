"""Examples catalog commands: list/filter the runnable examples under ``examples/``, and smoke-run one."""

from __future__ import annotations

import argparse
import shutil


def _configure_examples_smoke(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("key", help="example key, e.g. 'recipes/physics/mpm_tutorial' (see 'eden examples list')")
    parser.add_argument("--timeout", type=int, default=900, help="per-example timeout in seconds (default: 900)")


def _run_examples_smoke(args) -> None:
    import sys

    from eden.utils.examples_catalog import CATALOG
    from eden.utils.example_smoke import has_gpu, skip_reason, smoke_run

    try:
        spec = CATALOG.get(args.key)
    except KeyError as e:
        print(str(e).strip("'"))
        sys.exit(2)

    reason = skip_reason(spec, has_gpu=has_gpu())
    if reason:
        print(f"cannot smoke {spec.key}: {reason}")
        sys.exit(2)

    print(f"smoking {spec.key}  [{spec.hardware}]  (EDEN_SMOKE=1, timeout {args.timeout}s) ...")
    result = smoke_run(spec, timeout=args.timeout, capture=False)
    print("PASS" if result.returncode == 0 else f"FAIL (exit {result.returncode})")
    sys.exit(result.returncode)


def _configure_examples_list(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tier", choices=["project", "policy", "recipe"], help="filter by tier")
    parser.add_argument(
        "--capability",
        metavar="TAG",
        help="filter by capability — a full 'namespace.leaf' tag (e.g. sensor.lidar) or just a namespace (e.g. sensor)",
    )
    parser.add_argument(
        "--hardware", choices=["cpu", "gpu", "gpu-fp64", "real-robot"], help="filter by hardware requirement"
    )
    parser.add_argument("--search", metavar="QUERY", help="substring search over key/purpose/capabilities")
    parser.add_argument("--json", action="store_true", help="emit JSON records instead of a table")


def _run_examples_list(args) -> None:
    from eden.utils.examples_catalog import CATALOG

    if args.search:
        keys = CATALOG.search(args.search)
    else:
        cap = cap_kind = None
        if args.capability:
            cap, cap_kind = (args.capability, None) if "." in args.capability else (None, args.capability)
        keys = CATALOG.names(tier=args.tier, capability=cap, capability_kind=cap_kind, hardware=args.hardware)
    specs = [CATALOG.get(k) for k in keys]

    if args.json:
        import json

        from dataclasses import asdict

        print(json.dumps([asdict(s) for s in specs], indent=2))
        return

    if not specs:
        print("No examples match the given filters.")
        return

    # One capability per line so wrapping breaks at tag boundaries, not mid-token.
    rows = [(i, s.key, s.tier, s.hardware, "\n".join(s.capabilities), s.purpose) for i, s in enumerate(specs, 1)]

    try:
        from eden.utils.format import format_table

        # Wrap only Purpose to fit the terminal; capabilities already wrap per-tag.
        term_cols = shutil.get_terminal_size(fallback=(100, 24)).columns
        key_w = max(len("Example"), max(len(r[1]) for r in rows))
        cap_w = max(len("Capabilities"), max((len(c) for s in specs for c in s.capabilities), default=12))
        fixed = len("No.") + key_w + len("recipe") + len("real-robot") + cap_w + 18  # borders/padding
        table = format_table(
            f"Eden examples ({len(rows)})",
            ["No.", "Example", "Tier", "HW", "Capabilities", "Purpose"],
            rows,
            align={col: "l" for col in ("Example", "Capabilities", "Purpose")},
            max_width={"Purpose": max(20, term_cols - fixed)},
        )
        print(table)
    except ImportError:
        width = max(len(r[1]) for r in rows)
        for i, s in enumerate(specs, 1):
            print(f"  {i:>3}. {s.key:<{width}}  [{s.tier}/{s.hardware}]  {s.purpose}")
