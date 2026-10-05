"""Running a declared grid, and reporting it under the published names.

This is the part every scratch script repeated: resolve the schemas, build the
tasks, build the models, preflight, run, store, print. Repeating it is how a
run id came to belong to three different scripts and how a published figure
came to have no reproducible origin.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oold_llm_bench.experiments.grids import Grid

__all__ = ["RunResult", "report", "run_grid"]


@dataclass
class RunResult:
    """What a finished grid leaves behind."""

    run_id: str
    cells: int
    records: Path
    published: Path


def run_grid(
    grid: Grid,
    models: list[str],
    *,
    schemas: Path | None = None,
    per_class: int | None = None,
    out: Path = Path("results"),
    workers: int | None = None,
    rungs: tuple[str, ...] = (),
    on_progress: Callable[[int, int], None] | None = None,
    agents: Callable[[Any], Any] | None = None,
) -> RunResult:
    """Run one grid and return where its records went.

    The run id is derived from the grid name, so a result file says which grid
    produced it. Deriving it from a copied string is how three scripts came to
    write under one prefix and a fourth under a name belonging to none of them.
    """
    from oold_llm_bench.clients import Credentials, client_factory
    from oold_llm_bench.config import entry_for, spec_for
    from oold_llm_bench.corpus import QUANTITIES, resolve_module
    from oold_llm_bench.results import ResultStore
    from oold_llm_bench.results.record import Environment
    from oold_llm_bench.runner import ExperimentConfig, agent_factory, build_request, run_experiment
    from oold_llm_bench.runner.arms import register_union_arms
    from oold_llm_bench.runner.preflight import preflight

    register_union_arms()
    grid = grid.select(rungs)
    directory = resolve_module(QUANTITIES, schemas) if grid.needs_schemas else None
    tasks = grid.tasks(per_class if per_class is not None else grid.per_class, directory)
    entries = [entry_for(name) for name in models]
    specs = [spec_for(entry) for entry in entries]

    run_id = f"{grid.name}-{time.strftime('%Y%m%d-%H%M')}"
    config = ExperimentConfig(
        name=run_id,
        conditions=list(grid.conditions),
        models=specs,
        tasks=tasks,
        runs_per_cell=1,
        notes=grid.notes or None,
    )

    environment = Environment(benchmark_version=_version(), benchmark_sha=_revision())
    checked = preflight(config, environment)
    if not checked.clear:
        raise RuntimeError(f"preflight refused this grid: {'; '.join(checked.blocked)}")

    store = ResultStore(out, run_id)
    done = 0

    def on_outcome(outcome: Any) -> None:
        nonlocal done
        done += 1
        if outcome.record is not None:
            store.append(outcome.record)
        if on_progress is not None:
            on_progress(done, config.size)

    if agents is None:
        # Built here rather than taken as an argument by default, so the
        # normal path reads credentials once and names what is missing before
        # a cell is run and billed.
        credentials = Credentials.from_env(transports=sorted({entry.transport for entry in entries}))
        agents = agent_factory(client_factory(credentials))
    finished = run_experiment(
        config,
        agents,
        environment,
        make_request=build_request,
        on_outcome=on_outcome,
        workers=workers if workers is not None else grid.workers,
    )
    store.write_summary(finished.describe())
    return RunResult(run_id=run_id, cells=config.size, records=store.local_path, published=store.published_path)


def report(published: Path, dimensions: tuple[str, ...]) -> str:
    """The grid's own table, then the same cells under their published names.

    Two tables rather than one. The first is keyed on the condition as the
    runner declared it, which is what a rerun has to match; the second is
    keyed on what the prompt carried and what the decoder accepted, which is
    what the result is about.
    """
    import statistics
    from collections import defaultdict

    from oold_llm_bench.report import tabulate
    from oold_llm_bench.report.axes import ENFORCED, label_of
    from oold_llm_bench.results import read_records

    records = list(read_records(published))
    lines = [tabulate(records).render(dimensions=dimensions), ""]

    pooled: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for record in records:
        label = label_of(record["arm"], record.get("enforcement") or {})
        calls = (record.get("calls") or {}).get("calls", [])
        tokens = [call.get("input_tokens") for call in calls if call.get("input_tokens")]
        for score in record["scores"]:
            pooled[(label, record["model"]["model"])].append((score["primary_f1"], tokens[0] if tokens else 0))

    width = max((len(label) for label, _ in pooled), default=4)
    lines.append(f"{'name':{width}s} {'model':22s} {'ctx tok':>8s} {'F1':>7s}  Enforced")
    for (label, model), got in sorted(pooled.items()):
        # Longest suffix wins: the labels contain hyphens, so splitting on the
        # last one would read "enforced" out of "not-enforced".
        key = next(k for k in sorted(ENFORCED, key=len, reverse=True) if f"-{k}" in f"-{label}")
        tokens = [t for _, t in got if t]
        lines.append(
            f"{label:{width}s} {model[-22:]:22s} "
            f"{int(statistics.median(tokens)) if tokens else 0:8d} "
            f"{statistics.mean(f1 for f1, _ in got):7.3f}  {ENFORCED[key][:44]}"
        )
    return "\n".join(lines)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("oold-llm-bench")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "0.0.0"


def _revision() -> str:
    """The commit a record was produced at, or ``unknown``.

    Recorded because a result that does not name the code that made it cannot
    be rerun: the grader, the catalogue rendering and the arm registry all
    move, and each of them changes what a number means.
    """
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:  # pragma: no cover - only where git is absent
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603 - a resolved executable and fixed arguments
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git, or not a clone
        return "unknown"
    return out.stdout.strip()
