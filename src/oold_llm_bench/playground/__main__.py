"""Launching the playground.

``python -m oold_llm_bench.playground [--schemas DIR] [--port N]``
"""

from __future__ import annotations

import argparse
import sys

from oold_llm_bench.playground.corpora import SCHEMAS_ENV, MissingSchemas
from oold_llm_bench.playground.session import ORCHESTRATIONS, Options

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oold-llm-bench playground", description=__doc__)
    parser.add_argument("--schemas", default=None, help=f"the generated schema.org module; defaults to ${SCHEMAS_ENV}")
    parser.add_argument("--port", type=int, default=5011)
    parser.add_argument("--orchestration", choices=list(ORCHESTRATIONS), default="segmented")
    parser.add_argument("--arm", default="schema-dump-catalog-flat-enforced")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--catalogue-size", type=int, default=25)
    parser.add_argument("--no-show", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)

    from oold_llm_bench.playground.app import serve

    try:
        serve(
            schemas=args.schemas,
            port=args.port,
            show=not args.no_show,
            options=Options(
                orchestration=args.orchestration,
                arm=args.arm,
                model=args.model,
                catalogue_size=args.catalogue_size or None,
            ),
        )
    except MissingSchemas as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
