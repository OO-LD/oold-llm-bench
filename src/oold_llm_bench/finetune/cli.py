"""Driving the pipeline one step at a time.

Split, build, upload, create, status. Separate subcommands rather than one
run, because the expensive step is the fourth and everything before it is
worth looking at first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from oold_llm_bench.corpus import schemaorg as so
from oold_llm_bench.finetune.azure import DataPlane, FineTuneError
from oold_llm_bench.finetune.dataset import (
    CATALOGUE_SIZE,
    TRAIN_SEED_BASE,
    VALIDATION_SEED_BASE,
    build_corpus,
    lean_condition,
    write_jsonl,
)
from oold_llm_bench.finetune.job import DEVELOPER_TIER, JobRequest, poll, summarise
from oold_llm_bench.finetune.split import SPLIT_PATH, SPLIT_SEED, load_split, partition

__all__ = ["main"]

_ENV_ALIASES = {
    "API_KEY": "OOLD_BENCH_API_KEY",
    "API_ENDPOINT": "OOLD_BENCH_ENDPOINT",
    "API_VERSION": "OOLD_BENCH_API_VERSION",
}
"""How an existing credentials file names the same three things.

The file is shared with other projects and is not this repo's to rename, so
the mapping lives here instead of in the file.
"""


def load_env_file(path: Path) -> list[str]:
    """Read a dotenv-style file without overwriting anything already set.

    The first spelling of a name wins, because these files list several
    endpoints under one name and the first is the one the other scripts in
    this repo use. Returns the names it set, never the values.
    """
    applied: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        name = _ENV_ALIASES.get(key.strip(), key.strip())
        value = value.strip().strip('"').strip("'")
        if value and not os.environ.get(name):
            os.environ[name] = value
            applied.append(name)
    return applied


def _pool(directory: Path, min_own_slots: int) -> list[so.SchemaClass]:
    classes = so.load_classes(directory)
    if not classes:
        raise FineTuneError(f"no schemas under {directory}")
    return so.describable_classes(classes, min_own_slots=min_own_slots)


def _notation(args: argparse.Namespace) -> Any:
    from oold_llm_bench.corpus.quantities import Notation

    return Notation(getattr(args, "notation", "canonical"))


def _quantities(directory: Path, notation: Any) -> tuple[list[Any], Any, dict[str, Any], list[Any]]:
    """The QUDT corpus, and the kinds a task can be answered with."""
    from oold_llm_bench.corpus import load_kinds, load_signals
    from oold_llm_bench.corpus.catalogue import load_entries
    from oold_llm_bench.finetune.sources import quantities_pool

    kinds = load_kinds(directory)
    if not kinds:
        raise FineTuneError(f"no quantity kinds under {directory}")
    data = load_signals()
    return kinds, data, load_entries(directory, kinds), quantities_pool(kinds, data, notation)


def _source(args: argparse.Namespace, split: Any, answers: set[str]) -> Any:
    """The draw this corpus and this half of the split imply.

    There is no wiki draw, and there should not be. Wiki-Measurements is the
    evaluation corpus: H4 trains on generated quantity documents and measures
    on sentences a person wrote, which is the whole point of having both.
    Training on the evaluation set would answer a different question and
    `wiki_tasks` lives in the evaluate path for that reason.

    Refused rather than fallen through. Without this, `--corpus wiki` reached
    the schema.org branch, loaded a class pool out of a quantity schema
    directory, found none, and every draw raised: a build for 400 examples
    walked 16,001 seeds, yielded nothing, and said so only after the seed
    budget ran out.
    """
    from oold_llm_bench.finetune.sources import quantities_source, schemaorg_source

    if args.corpus == "wiki":
        raise ValueError(
            "there is no wiki training draw: Wiki-Measurements is what a tuned model is "
            "measured on. Build with --corpus quantities and evaluate with --corpus wiki."
        )
    if args.corpus == "quantities":
        notation = _notation(args)
        kinds, data, entries, pool = _quantities(args.schemas, notation)
        chosen = [k for k in pool if k.name in answers]
        return quantities_source(
            kinds, chosen, split, data, entries, catalogue_size=args.catalogue_size, notation=notation
        )
    pool = _pool(args.schemas, args.min_own_slots)
    chosen = [c for c in pool if c.name in answers]
    return schemaorg_source(pool, chosen, split, catalogue_size=args.catalogue_size, notation=_notation(args))


def _default_catalogue_size(args: argparse.Namespace) -> int:
    from oold_llm_bench.finetune.sources import QUDT_CATALOGUE_SIZE

    return QUDT_CATALOGUE_SIZE if args.corpus == "quantities" else CATALOGUE_SIZE


def _write_split(args: argparse.Namespace) -> int:
    # Wiki-Measurements is the same QUDT kinds read off Wikipedia sentences
    # rather than generated ones, so its pool is the kind list and not the
    # schema.org classes. Without this the split loads a class pool from a
    # quantity schema directory, finds none, and refuses to halve zero.
    if args.corpus in ("quantities", "wiki"):
        names = [kind.name for kind in _quantities(args.schemas, _notation(args))[3]]
    else:
        names = [cls.name for cls in _pool(args.schemas, args.min_own_slots)]
    split = partition(names, seed=args.seed, min_own_slots=args.min_own_slots, corpus=args.corpus)
    path = split.write(args.out)
    print(f"pool {len(names)} | train {len(split.train)} | heldout {len(split.heldout)} | {path}")
    return 0


def _build(args: argparse.Namespace) -> int:
    split = load_split(args.split)
    # Wiki-Measurements answers with QUDT kinds, so it reads the quantities
    # split, exactly as _evaluate already did. The mapping lived in one of the
    # two and not the other, so a training set could not be built for the
    # corpus an evaluation was already running against.
    wanted = "quantities" if args.corpus == "wiki" else args.corpus
    if split.corpus != wanted:
        raise FineTuneError(f"the split was drawn on {split.corpus} and the run asks for {wanted}")
    args.catalogue_size = args.catalogue_size or _default_catalogue_size(args)
    draw = _source(args, split, set(split.train))
    condition = lean_condition(describe_catalogue=not args.bare_catalogue)
    seen: set[str] = set()

    train = build_corpus(
        split,
        draw,
        name="train",
        count=args.n_train,
        seed_base=TRAIN_SEED_BASE,
        condition=condition,
        seen=seen,
    )
    validation = build_corpus(
        split,
        draw,
        name="valid",
        count=args.n_validation,
        seed_base=VALIDATION_SEED_BASE,
        condition=condition,
        seen=seen,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": write_jsonl(args.out / "train.jsonl", train.examples),
        "validation": write_jsonl(args.out / "validation.jsonl", validation.examples),
    }
    report = {
        "split": {
            "corpus": split.corpus,
            "train": len(split.train),
            "heldout": len(split.heldout),
            "version": split.version,
        },
        "condition": condition.describe(),
        "notation": getattr(args, "notation", "canonical"),
        "catalogue_size": args.catalogue_size,
        "train": train.describe(),
        "validation": validation.describe(),
        "files": {name: str(path) for name, path in paths.items()},
        "chars": {name: path.stat().st_size for name, path in paths.items()},
    }
    (args.out / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _show(args: argparse.Namespace) -> int:
    line = args.file.read_text(encoding="utf-8-sig").splitlines()[args.index]
    for turn in json.loads(line)["messages"]:
        print(f"===== {turn['role']} =====")
        print(turn["content"])
    return 0


def _plane(args: argparse.Namespace) -> DataPlane:
    if args.env:
        load_env_file(args.env)
    plane = DataPlane.from_env()
    if args.api_version:
        plane.api_version = args.api_version
    print(f"endpoint {json.dumps(plane.describe())}", file=sys.stderr)
    return plane


def _upload(args: argparse.Namespace) -> int:
    plane = _plane(args)
    uploaded: dict[str, Any] = {}
    for name in ("train", "validation"):
        path = args.out / f"{name}.jsonl"
        found = plane.upload(path)
        uploaded[name] = {"id": found.get("id"), "bytes": found.get("bytes"), "status": found.get("status")}
        print(f"{name}: {uploaded[name]}")
    (args.out / "files.json").write_text(json.dumps(uploaded, indent=2) + "\n", encoding="utf-8")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """The ten cells, scored by the runner every other arm is scored by.

    One config per cell rather than one grid, because a grid is a product of
    conditions and models and only the diagonal of that product is an H4
    measurement. The other twenty combinations would be paid for and thrown
    away.
    """
    from oold_llm_bench.clients import Credentials, client_factory
    from oold_llm_bench.config.models import spec_for
    from oold_llm_bench.finetune.evaluate import (
        EVAL_SEED_BASE,
        QUDT_EVAL_SEED_BASE,
        Half,
        answer_names,
        ceiling,
        evaluation_tasks,
        pairings,
        tuned_entry,
    )
    from oold_llm_bench.results import ResultStore
    from oold_llm_bench.results.record import Environment
    from oold_llm_bench.runner import ExperimentConfig, agent_factory, build_request, preflight, run_experiment

    if args.env:
        load_env_file(args.env)
    split = load_split(args.split)
    args.catalogue_size = args.catalogue_size or _default_catalogue_size(args)
    seed_base = args.seed_base or (EVAL_SEED_BASE if args.corpus == "schemaorg" else QUDT_EVAL_SEED_BASE)
    # Wiki-Measurements answers with QUDT kinds, so it reads the quantities
    # split. That is the point of running it: the same partition, a register
    # nothing in this benchmark wrote.
    wanted = "quantities" if args.corpus == "wiki" else args.corpus
    if split.corpus != wanted:
        raise FineTuneError(f"the split was drawn on {split.corpus} and the run asks for {wanted}")
    environment = Environment(benchmark_version="0.1.0", benchmark_sha="local")
    store = ResultStore(args.out, args.run_id)

    if args.corpus == "wiki":
        from oold_llm_bench.finetune.sources import wiki_tasks

        kinds, _, entries, _ = _quantities(args.schemas, _notation(args))
        from oold_llm_bench.finetune.sources import WIKI_LIMIT

        limit = WIKI_LIMIT if args.limit is None else (args.limit or None)
        tasks = wiki_tasks(kinds, split, entries, catalogue_size=args.catalogue_size, limit=limit)
    else:
        tasks = {
            half.value: evaluation_tasks(
                _source(args, split, answer_names(split, half)),
                half,
                count=args.n_tasks,
                seed_base=seed_base,
            )
            for half in Half
        }
    tasks = {name: rows for name, rows in tasks.items() if rows}
    ceilings = {name: ceiling(rows) for name, rows in tasks.items()}
    for name, value in ceilings.items():
        print(f"{name}: {len(tasks[name])} tasks, grader ceiling {value:.4f}", flush=True)

    entries = {
        pairing.deployment: tuned_entry(pairing.deployment)
        for pairing in pairings(args.base, args.tuned_lean, args.tuned_bare, pin_units=args.pin_units)
    }
    client_for = client_factory(Credentials.from_env())
    _patch_entries(entries)

    summary: list[dict[str, Any]] = []
    for pairing in pairings(args.base, args.tuned_lean, args.tuned_bare, pin_units=args.pin_units):
        for group, rows in tasks.items():
            config = ExperimentConfig(
                name=f"{pairing.label}/{group}",
                conditions=[pairing.condition],
                models=[spec_for(entries[pairing.deployment])],
                tasks=rows,
                runs_per_cell=1,
            )
            checked = preflight(config, environment)
            if not checked.clear:
                print(f"blocked: {checked.blocked}", file=sys.stderr)
                return 1
            run = run_experiment(
                config,
                agent_factory(client_for),
                environment,
                make_request=build_request,
                workers=args.workers,
            )
            for outcome in run.outcomes:
                if outcome.record is not None:
                    store.append(outcome.record)
            summary.append(_cell_summary(pairing, group, run, checked.controls, ceilings[group]))
            print(json.dumps(summary[-1]), flush=True)

    store.write_summary({
        "cells": summary,
        "ceilings": dict(ceilings),
        "corpus": args.corpus,
        "seed_base": seed_base,
        "notation": getattr(args, "notation", "canonical"),
    })
    print(_render(summary))
    return 0


def _patch_entries(entries: dict[str, Any]) -> None:
    """Make the deployments resolvable by the shared client factory.

    ``client_factory`` looks a cell's model up in the committed ladder, and a
    deployment name cannot be committed. Registering the entries for the
    length of the run is the narrowest way to bridge that.
    """
    from oold_llm_bench.config import models

    models.LADDER = models.LADDER + tuple(entries.values())


def _cell_summary(pairing: Any, group: str, run: Any, controls: dict[str, float], ceiling: float) -> dict[str, Any]:
    """One cell: what it scored, what it cost, and what it could have scored."""
    ok = [o for o in run.outcomes if o.ok]
    totals = [o.record.calls.get("totals", {}) for o in ok if o.record is not None]

    def per_example(key: str) -> int:
        return round(sum(t.get(key, 0) for t in totals) / len(totals)) if totals else 0

    # Both figures. Net is what the cell was billed, and it collapses once a
    # deployment starts serving a cached prefix, which makes it useless for
    # comparing one prompt against another. Raw is the size of the prompt.
    return {
        "cell": pairing.label,
        "prompt": pairing.prompt,
        "tuned": pairing.tuned,
        "half": group,
        "n": len(ok),
        "failures": len(run.failures()),
        "f1": round(sum(o.primary for o in ok) / len(ok), 4) if ok else 0.0,
        # The primary metric is the value triple, and on a quantity corpus a
        # value triple is a magnitude and a unit. It does not read the class at
        # all, so per-class knowledge can only show up in the class dimension
        # and a cell reported on f1 alone cannot see the hypothesis.
        "class_f1": _dimension(ok, "class"),
        "unit_f1": _dimension(ok, "unit"),
        "ceiling": round(ceiling, 4),
        "errors": run.error_kinds(),
        "prompt_tokens": per_example("input_tokens"),
        "net_prompt_tokens": per_example("input_tokens") - per_example("cached_input_tokens"),
        "output_tokens": per_example("output_tokens"),
        "controls": {name: round(value, 4) for name, value in controls.items()},
    }


def _dimension(outcomes: list[Any], name: str) -> float:
    """Mean F1 over one scored dimension, task by task."""
    from oold_llm_bench.grading.triples import Dimension

    wanted = Dimension(name)
    scores = [o.score.dimensions[wanted].f1 for o in outcomes if o.score and wanted in o.score.dimensions]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _render(summary: list[dict[str, Any]]) -> str:
    """Every group side by side, never averaged into one number.

    Value and class are shown together because on a quantity corpus the value
    triple is a magnitude and a unit and never reads the class, so the two
    dimensions answer different questions and only one of them can carry a
    held-out claim.

    A third group appears where the corpus has kinds on neither side of the
    split. Folding them into a half would report a number about kinds that
    half does not contain.
    """
    by_cell: dict[str, dict[str, Any]] = {}
    for row in summary:
        by_cell.setdefault(row["cell"], {})[row["half"]] = row
    groups = [g for g in ("train", "heldout", "unsplit") if any(g in cell for cell in by_cell.values())]

    head = f"{'cell':18}" + "".join(f"{g[:5] + ' val':>11}{g[:5] + ' cls':>11}" for g in groups)
    head += f"{'drop val':>10}{'drop cls':>10}{'prompt':>9}"
    lines = [head, "-" * len(head)]
    for cell, found in by_cell.items():
        row = f"{cell:18}"
        for group in groups:
            here = found.get(group, {})
            row += f"{here.get('f1', 0):>11.4f}{here.get('class_f1', 0):>11.4f}"
        train, held = found.get("train", {}), found.get("heldout", {})
        row += (
            f"{(train.get('f1', 0) or 0) - (held.get('f1', 0) or 0):>10.4f}"
            f"{(train.get('class_f1', 0) or 0) - (held.get('class_f1', 0) or 0):>10.4f}"
            f"{train.get('prompt_tokens', 0):>9}"
        )
        lines.append(row)
    return "\n".join(lines)


def _files(args: argparse.Namespace) -> int:
    """Read back what the service made of the uploads.

    An uploaded file is ``pending`` until the service has validated it, and a
    job created against a file that later fails validation fails with it.
    """
    plane = _plane(args)
    uploaded = json.loads((args.out / "files.json").read_text(encoding="utf-8"))
    ready = True
    for name, found in uploaded.items():
        state = plane.file(found["id"])
        print(f"{name}: {state.get('status')} {state.get('status_details') or ''}".strip())
        ready = ready and state.get("status") == "processed"
    return 0 if ready else 1


def _create(args: argparse.Namespace) -> int:
    plane = _plane(args)
    uploaded = json.loads((args.out / "files.json").read_text(encoding="utf-8"))
    request = JobRequest(
        model=args.model,
        training_file=uploaded["train"]["id"],
        validation_file=uploaded["validation"]["id"],
        suffix=args.suffix,
        n_epochs=args.n_epochs,
        seed=args.job_seed,
        tier=None if args.tier == "none" else args.tier,
    )
    print("body:", json.dumps(request.payload(), indent=2))
    if args.dry_run:
        return 0
    job = plane.create_job(request.payload())
    (args.out / "job.json").write_text(json.dumps(summarise(job), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summarise(job), indent=2))
    return 0


def _cancel(args: argparse.Namespace) -> int:
    """Stop a job, named explicitly.

    The job id is required rather than read from the output directory. Every
    other subcommand works on whatever is in ``--out``, and that default is
    the wrong one to have when the mistake is irreversible.
    """
    plane = _plane(args)
    print(json.dumps(summarise(plane.cancel(args.job_id)), indent=2))
    return 0


def _status(args: argparse.Namespace) -> int:
    plane = _plane(args)
    job_id = args.job_id or json.loads((args.out / "job.json").read_text(encoding="utf-8"))["id"]
    while True:
        found, done = poll(plane, job_id)
        print(f"{time.strftime('%H:%M:%S')} {json.dumps(found)}", flush=True)
        if done or not args.watch:
            break
        time.sleep(args.interval)
    if args.events:
        for event in plane.events(job_id, limit=args.events):
            print(f"  {event.get('created_at')} {event.get('level')} {event.get('message')}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oold-llm-bench-finetune", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results_local/finetune"), help="where the artefacts go")
    parser.add_argument("--env", type=Path, default=None, help="dotenv file to read credentials from")
    parser.add_argument("--api-version", default=None, help="data-plane api-version")
    parser.add_argument(
        "--notation",
        choices=("canonical", "written"),
        default="canonical",
        help="whether a document spells its values as the schema does or as a note does",
    )
    parser.add_argument(
        "--corpus",
        choices=("schemaorg", "quantities", "wiki"),
        default="quantities",
        help="which corpus a task comes from",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="write the train/held-out class partition")
    split.add_argument("--schemas", type=Path, required=True)
    split.add_argument("--min-own-slots", type=int, default=3)
    split.add_argument("--seed", type=int, default=SPLIT_SEED)
    split.add_argument("--out", dest="out", type=Path, default=SPLIT_PATH)
    split.set_defaults(handler=_write_split)

    build = sub.add_parser("build", help="write the training and validation JSONL")
    build.add_argument("--schemas", type=Path, required=True)
    build.add_argument("--min-own-slots", type=int, default=3)
    build.add_argument("--split", type=Path, default=SPLIT_PATH)
    build.add_argument("--n-train", type=int, default=1000)
    build.add_argument("--n-validation", type=int, default=200)
    build.add_argument("--catalogue-size", type=int, default=0, help="0 takes the corpus default")
    build.add_argument(
        "--bare-catalogue",
        action="store_true",
        help="list class identifiers only, so what a class means comes from the weights",
    )
    build.set_defaults(handler=_build)

    evaluate = sub.add_parser("evaluate", help="score the tuned models against the base one, on both halves")
    evaluate.add_argument("--schemas", type=Path, required=True)
    evaluate.add_argument("--min-own-slots", type=int, default=3)
    evaluate.add_argument("--split", type=Path, default=SPLIT_PATH)
    evaluate.add_argument("--base", default=None, help="deployment serving the untuned model")
    evaluate.add_argument("--tuned-lean", default=None, help="deployment serving the described-catalogue model")
    evaluate.add_argument("--tuned-bare", default=None, help="deployment serving the bare-catalogue model")
    evaluate.add_argument(
        "--pin-units",
        action="store_true",
        help=(
            "close the unit slot in the grammar. Off by default, and off in "
            "every H4 number before 2026-10-04, which made those runs "
            "half-enforced without saying so"
        ),
    )
    evaluate.add_argument("--n-tasks", type=int, default=120)
    evaluate.add_argument("--catalogue-size", type=int, default=0, help="0 takes the corpus default")
    evaluate.add_argument("--seed-base", type=int, default=None)
    evaluate.add_argument("--limit", type=int, default=None, help="tasks per kind, for the real-text corpus")
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--run-id", default="h4")
    evaluate.set_defaults(handler=_evaluate)

    show = sub.add_parser("show", help="print one example in full")
    show.add_argument("file", type=Path)
    show.add_argument("--index", type=int, default=0)
    show.set_defaults(handler=_show)

    upload = sub.add_parser("upload", help="send both files to the data plane")
    upload.set_defaults(handler=_upload)

    files = sub.add_parser("files", help="read the uploaded files back")
    files.set_defaults(handler=_files)

    create = sub.add_parser("create", help="create the tuning job")
    create.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    create.add_argument("--suffix", default=None)
    create.add_argument("--n-epochs", type=int, default=None)
    create.add_argument("--job-seed", type=int, default=None)
    create.add_argument("--tier", default=DEVELOPER_TIER)
    create.add_argument("--dry-run", action="store_true")
    create.set_defaults(handler=_create)

    cancel = sub.add_parser("cancel", help="stop a job")
    cancel.add_argument("job_id")
    cancel.set_defaults(handler=_cancel)

    status = sub.add_parser("status", help="read the job, optionally until it settles")
    status.add_argument("--job-id", default=None)
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=int, default=300)
    status.add_argument("--events", type=int, default=0)
    status.set_defaults(handler=_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except FineTuneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
