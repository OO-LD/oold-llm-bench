"""JSONSchemaBench's OpenAI arm, run with and without strict.

Their engine sends `response_format={"type":"json_schema","json_schema":
{"schema":..., "name":...}}` with no `strict`, so its OpenAI column measures
an unenforced hint beside guidance, outlines and xgrammar, which are grammars.
This runs the same schemas, the same adapt_schema and the same prompt, once
their way and once with strict=True, and reports their three metrics.

One difference, deliberate: they validate the answer against the *adapted*
schema, the one their own transform produced. This reports both, because
validating against the transform's output cannot see what the transform gave
away.
"""

import io
import json
import os
import pathlib
import random
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import jsonschema

from oold_llm_bench.clients import Credentials
from oold_llm_bench.config import entry_for
from oold_llm_bench.finetune.cli import load_env_file

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA = pathlib.Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "jsb" / "data"
PER_SPLIT = int(os.environ.get("JSB_PER_SPLIT", "30"))
MODEL = "gpt-5-nano"


def recursively_set_additional_properties_false(schema):
    if not isinstance(schema, dict):
        return
    if ("additionalProperties" not in schema or schema["additionalProperties"]) and "properties" in schema:
        schema["additionalProperties"] = False
    for value in schema.values():
        if isinstance(value, dict):
            recursively_set_additional_properties_false(value)
        elif isinstance(value, list):
            for item in value:
                recursively_set_additional_properties_false(item)


def set_all_properties_required(schema):
    if not isinstance(schema, dict):
        return schema
    if "properties" in schema and isinstance(schema["properties"], dict):
        schema["required"] = list(schema["properties"].keys())
    for value in schema.values():
        if isinstance(value, dict):
            set_all_properties_required(value)
        elif isinstance(value, list):
            for item in value:
                set_all_properties_required(item)
    return schema


def add_root_type_if_missing(schema):
    if isinstance(schema, dict) and "type" not in schema and "properties" in schema:
        schema["type"] = "object"


def adapt(schema):
    """Their three transforms, in their order."""
    out = json.loads(json.dumps(schema))
    recursively_set_additional_properties_false(out)
    add_root_type_if_missing(out)
    return set_all_properties_required(out)


def messages_for(schema):
    """Their formatter at num_shots=0: one system line, then the schema."""
    return [
        {"role": "system", "content": "You need to generate a JSON object that matches the schema below."},
        {"role": "user", "content": json.dumps(schema)},
    ]


def validates(instance, schema):
    try:
        jsonschema.validate(instance, schema)
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        # An unusable schema is not a failed answer. Counted as invalid, so a
        # schema the validator cannot read never passes by default.
        return False
    return True


def sample():
    rng = random.Random(11)  # noqa: S311 - a reproducible sample is the requirement
    picked = []
    for split in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = sorted(split.glob("*.json"))
        for path in rng.sample(files, min(PER_SPLIT, len(files))):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # A schema the corpus cannot parse is one no engine could have
                # been asked, so it is skipped rather than counted as a refusal.
                continue
            if isinstance(raw, dict):
                picked.append((split.name, path.name, raw))
    return picked


def build_client():
    """The bench's own builder, so the endpoint and transport are not guessed."""
    from oold_llm_bench.clients.azure import build_client as bench_client

    return bench_client(entry_for(MODEL), Credentials.from_env()).llm


CLIENT = build_client()


def run_one(item, strict):
    split, name, original = item
    adapted = adapt(original)
    fmt = {"type": "json_schema", "json_schema": {"schema": adapted, "name": "json_schema"}}
    if strict:
        fmt["json_schema"]["strict"] = True
    row = {
        "split": split,
        "file": name,
        "strict": strict,
        "declared": 0,
        "empirical": 0,
        "empirical_original": 0,
        "error": None,
    }
    try:
        reply = CLIENT.invoke(messages_for(adapted), response_format=fmt)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return row
    row["declared"] = 1
    text = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
    try:
        instance = json.loads(text)
    except Exception:
        return row
    row["empirical"] = int(validates(instance, adapted))
    row["empirical_original"] = int(validates(instance, original))
    return row


def _credentials() -> None:
    """Read a dotenv sitting in the working directory, if there is one.

    Nothing already exported is overwritten, and no account's file has to be
    present for this script to be readable.
    """
    path = pathlib.Path(".env")
    if path.is_file():
        load_env_file(path)


def main():
    _credentials()
    items = sample()
    jobs = [(item, strict) for strict in (False, True) for item in items]
    out = pathlib.Path(os.environ.get("JSB_OUT", "jsonschemabench_rows.jsonl"))
    started = time.time()
    done = 0
    with out.open("w", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(lambda a: run_one(*a), jobs):
            handle.write(json.dumps(row) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}, {time.time() - started:.0f}s", flush=True)
    report(out)


def report(path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    print(f"\n{'arm':12s} {'n':>5s} {'declared':>9s} {'empirical':>10s} {'compliance':>11s} {'vs original':>12s}")
    for strict in (False, True):
        group = [r for r in rows if r["strict"] is strict]
        n = len(group)
        dec = sum(r["declared"] for r in group)
        emp = sum(r["empirical"] for r in group)
        emp_o = sum(r["empirical_original"] for r in group)
        label = "strict=True" if strict else "their way"
        comp = emp / dec if dec else 0.0
        print(f"{label:12s} {n:>5d} {dec / n:>9.3f} {emp / n:>10.3f} {comp:>11.3f} {emp_o / n:>12.3f}")
    print("\nby split, empirical coverage:")
    print(f"{'split':18s} {'their way':>10s} {'strict':>8s} {'declared(strict)':>17s}")
    splits = sorted({r["split"] for r in rows})
    for split in splits:
        a = [r for r in rows if r["split"] == split and not r["strict"]]
        b = [r for r in rows if r["split"] == split and r["strict"]]
        if not a or not b:
            continue
        print(
            f"{split:18s} {sum(r['empirical'] for r in a) / len(a):>10.2f} "
            f"{sum(r['empirical'] for r in b) / len(b):>8.2f} {sum(r['declared'] for r in b) / len(b):>17.2f}"
        )
    errs = defaultdict(int)
    for r in rows:
        if r["error"]:
            errs[(r["strict"], r["error"].split(":")[0])] += 1
    print("\nerrors:", dict(errs))


if __name__ == "__main__":
    main()
