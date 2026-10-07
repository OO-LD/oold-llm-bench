"""The command line: fetch a corpus, run a grid, read it back.

A clone plus this is meant to be enough to reproduce any published figure.
Credentials come from the environment and are named when missing, the schema
module is fetched and digest-checked, and the grid is one of the declared
ones rather than a script somebody kept on their own machine.

    oold-bench grids
    oold-bench fetch-corpus quantities --dest ./schemas
    oold-bench run context-ladder --models unsloth/Qwen3.5-9B --tasks 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main"]

_MODULES = {"quantities": "QUANTITIES", "schemaorg": "SCHEMAORG"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oold-bench", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("grids", help="list the declared grids and their rungs")

    fetch = sub.add_parser("fetch-corpus", help="download a schema module and check its digest")
    fetch.add_argument("module", choices=sorted(_MODULES))
    fetch.add_argument("--dest", type=Path, default=None, help="where to put it, default the Hub cache")

    run = sub.add_parser("run", help="run a declared grid")
    run.add_argument("grid", help="a name from `oold-bench grids`")
    run.add_argument("--models", required=True, help="comma-separated ids from the model catalogue")
    run.add_argument("--schemas", type=Path, default=None, help="an already-obtained schema module")
    run.add_argument("--tasks", type=int, default=None, help="tasks per class, default the grid's own")
    run.add_argument("--rungs", default="", help="comma-separated published names, default every rung")
    run.add_argument("--out", type=Path, default=Path("results"), help="where records are written")
    run.add_argument("--workers", type=int, default=None, help="concurrent calls, default the grid's own")
    run.add_argument(
        "--reasoning",
        default=None,
        choices=["off", "low", "medium", "high"],
        help="thinking, where the provider understands it; vLLM does and Azure rejects the parameter",
    )
    run.add_argument("--env", type=Path, default=None, help="a dotenv file, default one in the working directory")

    show = sub.add_parser("report", help="print the table for a finished run")
    show.add_argument("published", type=Path, help="a *.published.jsonl")
    show.add_argument("--dimensions", default="class,value,unit")

    chain = sub.add_parser("chains", help="record one task through one pipeline, as a page")
    chain.add_argument("--model", required=True, help="an id from the model catalogue")
    chain.add_argument("--arm", required=True, help="an arm, or a published condition name")
    chain.add_argument("--orchestration", default="single_shot")
    chain.add_argument("--shortlist-k", type=int, default=3, help="classes the select step may keep")
    chain.add_argument(
        "--plan-retry",
        action="store_true",
        help="ask again when a planning step answers with the schema instead of an answer",
    )
    chain.add_argument(
        "--catalogue-size",
        type=int,
        default=100,
        help="classes on offer, default the grids' own; lower it for a page a slow model can finish",
    )
    chain.add_argument("--schemas", type=Path, default=None)
    chain.add_argument("--out", type=Path, default=None, help="where to write the page, default stdout")
    chain.add_argument("--env", type=Path, default=None, help="a dotenv file, default one in the working directory")

    args = parser.parse_args(argv)
    return {
        "grids": _grids,
        "fetch-corpus": _fetch,
        "run": _run,
        "report": _report,
        "chains": _chains,
    }[args.command](args)


def _grids(_: argparse.Namespace) -> int:
    from oold_llm_bench.experiments import GRIDS

    for name, grid in GRIDS.items():
        print(f"{name}\n  {grid.summary}")
        for rung in grid.rungs():
            print(f"    {rung}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    from oold_llm_bench.corpus import provenance

    module = getattr(provenance, _MODULES[args.module])
    directory = provenance.fetch_module(module, args.dest)
    print(f"{module.name}: {directory}")
    print(f"digest {module.digest} over {module.files} files, as pinned")
    return 0


def _run(args: argparse.Namespace) -> int:
    from oold_llm_bench.experiments import GRIDS, report, run_grid

    _credentials(args.env)
    if args.grid not in GRIDS:
        raise SystemExit(f"no grid {args.grid!r}; `oold-bench grids` lists {', '.join(GRIDS)}")

    grid = GRIDS[args.grid]
    rungs = tuple(name.strip() for name in args.rungs.split(",") if name.strip())
    result = run_grid(
        grid,
        [name.strip() for name in args.models.split(",") if name.strip()],
        schemas=args.schemas,
        per_class=args.tasks,
        out=args.out,
        workers=args.workers,
        rungs=rungs,
        reasoning=args.reasoning,
        on_progress=_progress,
    )
    print(f"\n{result.run_id}: {result.cells} cells -> {result.records}")
    print(report(result.published, grid.dimensions))
    return 0


def _report(args: argparse.Namespace) -> int:
    from oold_llm_bench.experiments import report

    print(report(args.published, tuple(d.strip() for d in args.dimensions.split(","))))
    return 0


def _chains(args: argparse.Namespace) -> int:
    """One real task through one pipeline, written out as a page.

    One task and not a grid: the page exists so a reader can see what an arm
    sends, and a second task of the same shape adds length without adding an
    answer.
    """
    from oold_llm_bench.chains import Recorder, chain_of, render
    from oold_llm_bench.clients import Credentials, build_client
    from oold_llm_bench.config import entry_for, spec_for
    from oold_llm_bench.corpus import QUANTITIES, resolve_module
    from oold_llm_bench.experiments.corpora import quantity_tasks
    from oold_llm_bench.grading.score import score_task
    from oold_llm_bench.report.axes import LEGACY_ARM_NAMES, label_of
    from oold_llm_bench.runner import Cell, Condition, build_agent, build_request, read_answer
    from oold_llm_bench.runner.arms import register_union_arms

    _credentials(args.env)
    register_union_arms()

    arm = LEGACY_ARM_NAMES.get(args.arm, args.arm)
    entry = entry_for(args.model)
    task = quantity_tasks(resolve_module(QUANTITIES, args.schemas), 1, ("Altitude",))[0]
    condition = Condition(
        arm=arm,
        catalogue_size=args.catalogue_size,
        signal="named",
        orchestration=args.orchestration,
        # Only where the orchestration has a select step. Set on a single-shot
        # condition it would appear in the key and fork the row.
        shortlist_k=args.shortlist_k if args.orchestration != "single_shot" else None,
        plan_retry=args.plan_retry,
        reasoning="off",
        describe_catalogue=False,
    )
    cell = Cell(condition=condition, model=spec_for(entry), task=task, repetition=1)

    credentials = Credentials.from_env(transports=[entry.transport])
    recorder = Recorder(build_client(entry, credentials))
    result = build_agent(cell, recorder).run(build_request(cell))

    chain = chain_of(
        result,
        recorder,
        task,
        pipeline=args.orchestration,
        arm=arm,
        label=label_of(arm, condition.describe()),
        catalogue_size=args.catalogue_size,
    )
    # Read through the same extractor the runner uses, so the page's score is
    # the one a grid would have recorded for this cell.
    chain.scores = {"primary": score_task(task, read_answer(cell, result)).primary}
    page = render(chain)
    if args.out:
        args.out.write_text(page, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(page)
    return 0


def _credentials(given: Path | None) -> None:
    """Read a dotenv, named or sitting in the working directory.

    Nothing already exported is overwritten, so a variable set for one run
    beats a file written for another.
    """
    from oold_llm_bench.finetune.cli import load_env_file

    path = given if given is not None else Path(".env")
    if path.is_file():
        load_env_file(path)


def _progress(done: int, total: int) -> None:
    """Every fortieth cell, not every cell.

    A grid is thousands of calls and its output is read in a scrollback, so a
    line per cell buries the preflight result and the table.
    """
    if done % 40 == 0 or done == total:
        print(f"  {done}/{total}", flush=True, file=sys.stderr)
