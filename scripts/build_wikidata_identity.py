"""Turn harvested Wikidata pairs into the identity corpus.

Three harvests go in, one per class, and each is a uniform draw under a stated
seed from a population that was counted rather than estimated. ``same`` comes
from the namespace-0 redirect table of a dated dump: a draw of page ids, each
resolved to its Q-id, each probed for the merge edit and for the two states
that preceded it. ``different`` and ``unclear`` come from the full truthy
harvests of ``P1889`` and ``P460``, deduplicated to undirected pairs, with the
current state of both items.

The output is committed, so reading the corpus needs no network and no dump.
``--cap`` is what keeps the committed file under the large-file hook, and 90
per class is where it fits: a pair carries two entity states and runs about
1.3 KB against the 290 bytes of a Wiki-Measurements sentence, so 270 pairs is
470 KiB against the hook's 500. The full resolved set is written beside it in
the user cache, because the repository is not a data store and a run reads
neither.

Nothing is dropped silently. Every candidate that does not become a pair is
counted under a stated reason within its own class, and the three arithmetics
have to close before the file is written.

Run this when the redirect dump or the property harvests are refreshed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, OrderedDict
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from oold_llm_bench.corpus.wikidata_identity import (
    DUMP_URL,
    LICENCE,
    MIN_STATEMENTS,
    NOT_AN_ENTITY,
    TRUTH_PROPERTIES,
    IdentityClass,
    label_baseline,
    property_baseline,
    read_pairs,
    score_baseline,
)

REDIRECT_ROWS_NS0 = 4951219
"""Namespace-0 redirect rows with a ``Q``-shaped target in the dump below.

Counted by parsing the file, not read off a page. Confirmed independently
against the query service, where ``SELECT (COUNT(*) AS ?c) WHERE { ?a
owl:sameAs ?b }`` returned 4,986,833 live three weeks later: the gap is new
merges plus the 20,679 Lexeme redirects, which are out of scope here.
"""

DUMP_DATE = "2026-09-05"
MERGE_SEED = 20260929
EARLIEST_MERGE_YEAR = 2017
"""Merges before this are excluded, with their count.

Both wrong merges of 105 hand-checked pairs and 13 of the 18 undecidable ones
fall before 2017. The cut runs after the substance filter, and there it is
nearly free: the pre-2017 merges are overwhelmingly the thin ones the
statement floor has already taken out. The survey put the cost at a tenth of
the stock, which is the cost on the raw draw; the build reports what it
actually removes.
"""

LABEL_STRINGS = 8
ALIAS_STRINGS = 8
SITELINKS = 8
CLAIM_VALUES = 4
CLAIM_PROPERTIES = 40
"""How much of one item's state the committed record carries.

An identifier decides more of this corpus than a name does, so properties are
kept broadly and values narrowly. Labels are kept one language per distinct
string: a person with forty labels has one name in thirty-eight of them, and
the thirty-eight add bytes rather than evidence.

The two tier flags are computed from the trimmed state and not from the state
it was cut out of, so that a reader can recompute them from the record. The
cost is a pair whose only shared string sat past the eighth distinct one,
which would be counted into the hard tier here and out of it upstream. The
measured share of positives sharing no string is the kill test's, so the cut
is not moving it.
"""

ERAS = ((2021, "2021-2026"), (2017, "2017-2020"), (0, "pre-2017"))

STATEMENT_POPULATION = (
    "distinct undirected pairs of items, from the property's full truthy harvest. A statement "
    "whose object is somevalue comes back from the query service as a blank node and is not an "
    "endpoint; the harvest excludes 24 such pairs of P1889 and 16 of P460, and neither draw "
    "contained one."
)


def bulk_root() -> Path:
    """Where the full resolved set goes. Not the repository."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "oold-llm-bench" / "wikidata-identity"


def era_of(timestamp: str) -> str:
    year = int(timestamp[:4])
    return next(name for floor, name in ERAS if year >= floor)


def distinct(values: Iterable[str], limit: int) -> list[str]:
    """The first ``limit`` values that are new under case folding."""
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        kept.append(value)
        if len(kept) >= limit:
            break
    return kept


def all_strings(written: dict[str, Any]) -> set[str]:
    """Every name a written record answers to, case folded."""
    return {value.casefold() for value in written["labels"].values()} | {
        value.casefold() for value in written["aliases"]
    }


def claims_of(state: dict[str, Any]) -> dict[str, list[str]]:
    """The state's claims with the three answer-bearing properties removed."""
    return {
        prop: [str(value) for value in values]
        for prop, values in (state.get("claims") or {}).items()
        if prop not in TRUTH_PROPERTIES and values
    }


def types_of(state: dict[str, Any]) -> list[str]:
    return [str(value) for value in (state.get("claims") or {}).get("P31", [])]


def written_state(qid: str, state: dict[str, Any], revision: int | None, as_of: str) -> dict[str, Any]:
    labels = state.get("labels") or {}
    order = ["en", *sorted(k for k in labels if k != "en")]
    kept_labels: dict[str, str] = {}
    seen: set[str] = set()
    for language in order:
        value = labels.get(language)
        if value is None or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        kept_labels[language] = value
        if len(kept_labels) >= LABEL_STRINGS:
            break
    aliases = [value for language in order for value in (state.get("aliases") or {}).get(language, [])]
    claims = claims_of(state)
    ordered = sorted(claims, key=lambda prop: (prop != "P31", int(prop[1:])))
    sitelinks = state.get("sitelinks") or {}
    return {
        "qid": qid,
        "labels": kept_labels,
        "aliases": [value for value in distinct(aliases, ALIAS_STRINGS) if value.casefold() not in seen],
        "description": (state.get("descriptions") or {}).get("en"),
        "sitelinks": {site: sitelinks[site] for site in sorted(sitelinks)[:SITELINKS]},
        "claims": {prop: claims[prop][:CLAIM_VALUES] for prop in ordered[:CLAIM_PROPERTIES]},
        "statements": int(state.get("nstmts") or sum(len(v) for v in (state.get("claims") or {}).values())),
        "revision": revision,
        "as_of": as_of,
    }


NOT_A_THING = "one side is a Wikimedia page or a name string rather than an entity"


def usable(state: dict[str, Any] | None) -> str | None:
    """Why one side cannot be a pair half, or ``None``."""
    if not state:
        return "one side's state could not be read"
    if set(types_of(state)) & set(NOT_AN_ENTITY):
        return NOT_A_THING
    if int(state.get("nstmts") or 0) < MIN_STATEMENTS:
        return f"one side carries fewer than {MIN_STATEMENTS} statements"
    if not (state.get("labels") or {}).get("en"):
        return "one side carries no English label"
    return None


def read_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def judgements(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    """The hand-checked verdicts, keyed on the pair they were made about."""
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["src"], row["tgt"]): row for row in csv.DictReader(handle, delimiter="\t")}


UNRECOVERED = "the API did not return one of the two pre-merge revisions"
"""One reason for every transport failure, rather than one per url.

The probe records the request it gave up on, which is the right thing for a
rerun and the wrong thing for an exclusion table: fifty timeouts would read as
fifty reasons. They are one reason, and a rerun of the probe retries them.
"""


def why_not(row: dict[str, Any]) -> str:
    stated = row.get("fail") or row.get("fail_tgt") or UNRECOVERED
    return UNRECOVERED if "giving up on" in stated or stated[:1].isupper() else stated


def build_same(
    paths: list[Path], judged: dict[tuple[str, str], dict[str, str]]
) -> tuple[list[dict], Counter, int, Counter]:
    """Merge pairs, one record per source drawn, deduplicated on the source.

    The substance filter runs before the era cut, so the era count reads as
    "usable pairs the cut costs" rather than as a number the thin items have
    already inflated.
    """
    excluded: Counter[str] = Counter()
    checked: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    # A source probed twice has a row per attempt, and the attempt that
    # recovered both states is the one to keep whichever file it landed in.
    rows = sorted(read_jsonl(paths), key=lambda r: (r.get("draw_index", 1 << 30), r["src"], not r.get("tgt_pre")))
    for row in rows:
        if row["src"] in seen:
            continue
        seen.add(row["src"])
        if row.get("fail") or not row.get("tgt_pre"):
            excluded[why_not(row)] += 1
            continue
        stop = usable(row["src_pre"]) or usable(row["tgt_pre"])
        if stop:
            excluded[stop] += 1
            continue
        if era_of(row["merge_ts"]) == "pre-2017":
            excluded[f"the merge predates {EARLIEST_MERGE_YEAR}, where both observed wrong merges fall"] += 1
            continue
        verdict = judged.get((row["src"], row["tgt"]), {}).get("verdict")
        if verdict:
            checked[verdict] += 1
        left = written_state(row["src"], row["src_pre"], row["src_pre_revid"], row["src_pre_ts"])
        right = written_state(row["tgt"], row["tgt_pre"], row["tgt_pre_revid"], row["tgt_pre_ts"])
        # Absent where the probe could not recover it, which the identity
        # corpus already tolerates for other fields: a merge still has a
        # same/different/unclear truth without it, and a pair with none is
        # simply not usable for the patch question, not unusable outright.
        expected_patch = (
            written_state(row["tgt"], row["tgt_post"], row["tgt_post_revid"], row["tgt_post_ts"])
            if row.get("tgt_post")
            else None
        )
        kept.append({
            "class": IdentityClass.SAME.value,
            "order": row.get("draw_index", 0),
            "left": left,
            "right": right,
            "truth": {
                "kind": "merge",
                "detail": f"{row['src']} was merged into {row['tgt']}",
                "recorded_at": row["merge_ts"],
                "recorded_by": row.get("merge_user"),
                "final_redirect_target": row["tgt_dump"] if row.get("tgt_dump") != row["tgt"] else None,
            },
            "era": era_of(row["merge_ts"]),
            "shares_string": bool(all_strings(left) & all_strings(right)),
            "shares_property": bool(set(left["claims"]) & set(right["claims"])),
            "scored": verdict in (None, "correct"),
            "note": None if verdict in (None, "correct") else f"a hand check of this merge returned {verdict}",
            "stratum": types_of(row["tgt_pre"])[:1],
            "expected_patch": expected_patch,
        })
    return kept, excluded, len(seen), checked


def build_statement(
    draw: dict[str, Any], states: dict[str, Any], identity: IdentityClass
) -> tuple[list[dict], Counter, int]:
    """``P1889`` or ``P460`` pairs, both sides read as they stand now."""
    excluded: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    prop = draw["property"]
    types = draw.get("types") or {}
    for order, (left, right) in enumerate(draw["drawn"]):
        if {*types.get(left, ()), *types.get(right, ())} & set(NOT_AN_ENTITY):
            excluded[NOT_A_THING] += 1
            continue
        a, b = states.get(left), states.get(right)
        if a is None or b is None:
            excluded["one side could not be read under its own id, so it was merged or deleted since"] += 1
            continue
        if a == b:
            excluded["both ids resolve to one item, so the statement no longer names two"] += 1
            continue
        stop = usable(a) or usable(b)
        if stop:
            excluded[stop] += 1
            continue
        written_left = written_state(left, a, None, draw["retrieved_at"])
        written_right = written_state(right, b, None, draw["retrieved_at"])
        kept.append({
            "class": identity.value,
            "order": order,
            "left": written_left,
            "right": written_right,
            "truth": {
                "kind": prop,
                "detail": f"{left} {prop} {right}",
                "recorded_at": None,
                "recorded_by": None,
                "final_redirect_target": None,
            },
            "era": None,
            "shares_string": bool(all_strings(written_left) & all_strings(written_right)),
            "shares_property": bool(set(written_left["claims"]) & set(written_right["claims"])),
            "scored": True,
            "note": None,
            "stratum": types_of(a)[:1],
        })
    return kept, excluded, len(draw["drawn"])


def state_shape(pairs: list[dict[str, Any]]) -> dict[str, float]:
    """How rich the states of one class are, so a report can control for it.

    The positives are pre-merge revisions and the other two classes are items
    as they stand now, so the classes differ in shape as well as in truth.
    Carried as a number rather than argued away.
    """
    sides = [side for pair in pairs for side in (pair["left"], pair["right"])]
    if not sides:
        return {}
    return {
        "statements": median(side["statements"] for side in sides),
        "sitelinks": median(len(side["sitelinks"]) for side in sides),
        "label_languages": median(len(side["labels"]) for side in sides),
        "identical_english_label": round(
            sum(
                1
                for pair in pairs
                if (pair["left"]["labels"].get("en") or "").casefold()
                == (pair["right"]["labels"].get("en") or "").casefold()
            )
            / len(pairs),
            3,
        ),
        "shares_any_name": round(sum(1 for pair in pairs if pair["shares_string"]) / len(pairs), 3),
    }


def stratified(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder so a prefix spreads over types instead of one bot batch.

    ``P1889`` carries 179 Indonesian villages and 152 supercomputers in a
    4,000-pair sample, both of them one creation run each. Taking the first
    ``n`` in draw order would hand a cap most of one of them.
    """
    buckets: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for pair in sorted(pairs, key=lambda p: p["order"]):
        buckets.setdefault((pair["stratum"] or ["none"])[0], []).append(pair)
    out: list[dict[str, Any]] = []
    while buckets:
        for key in list(buckets):
            out.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    return out


def dump(payload: dict[str, Any]) -> str:
    """One line per pair under an indented header, for a readable diff."""
    placeholder = "<<pairs>>"
    header = json.dumps({**payload, "pairs": placeholder}, indent=1, ensure_ascii=False)
    body = ",\n  ".join(json.dumps(p, ensure_ascii=False, separators=(",", ":")) for p in payload["pairs"])
    return header.replace(f'"{placeholder}"', "[\n  " + body + "\n ]") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-states", type=Path, nargs="+", required=True, help="probe output, jsonl")
    parser.add_argument("--merge-draw", type=Path, required=True, help="the drawn redirect pairs, json")
    parser.add_argument("--p1889-draw", type=Path, required=True)
    parser.add_argument("--p1889-states", type=Path, required=True)
    parser.add_argument("--p460-draw", type=Path, required=True)
    parser.add_argument("--p460-states", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, default=None, help="the hand-checked kill-test verdicts, tsv")
    parser.add_argument("--retrieved-at", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--cap", type=int, default=90, help="pairs kept per class, 0 for all")
    parser.add_argument("--out", type=Path, default=Path("src/oold_llm_bench/data/wikidata_identity.json"))
    args = parser.parse_args()

    judged = judgements(args.judgements)
    resolved: dict[str, list[dict[str, Any]]] = {}
    excluded: dict[str, dict[str, int]] = {}
    candidates: dict[str, int] = {}

    kept, reasons, went_in, checked = build_same(args.merge_states, judged)
    resolved[IdentityClass.SAME.value] = sorted(kept, key=lambda p: p["order"])
    excluded[IdentityClass.SAME.value] = dict(reasons.most_common())
    candidates[IdentityClass.SAME.value] = went_in

    draws: dict[str, dict[str, Any]] = {}
    for identity, draw_path, states_path in (
        (IdentityClass.DIFFERENT, args.p1889_draw, args.p1889_states),
        (IdentityClass.UNCLEAR, args.p460_draw, args.p460_states),
    ):
        draw = json.loads(draw_path.read_text(encoding="utf-8"))
        draw["retrieved_at"] = args.retrieved_at
        draws[identity.value] = {k: v for k, v in draw.items() if k not in ("drawn", "types", "retrieved_at")}
        states = json.loads(states_path.read_text(encoding="utf-8"))
        kept, reasons, went_in = build_statement(draw, states, identity)
        resolved[identity.value] = stratified(kept)
        excluded[identity.value] = dict(reasons.most_common())
        candidates[identity.value] = went_in

    cap = None if args.cap == 0 else args.cap
    pairs: list[dict[str, Any]] = []
    for name in (IdentityClass.SAME.value, IdentityClass.DIFFERENT.value, IdentityClass.UNCLEAR.value):
        chosen = resolved[name][:cap] if cap else resolved[name]
        for index, pair in enumerate(chosen, 1):
            pairs.append({
                "id": f"wd-{name}-{index:04d}",
                **{k: v for k, v in pair.items() if k not in ("order", "stratum")},
            })

    payload = {
        "schema_version": "1",
        "name": "Wikidata identity pairs",
        "source": {
            "same": {
                "population": "namespace-0 redirect rows with a Q-shaped target",
                "count": REDIRECT_ROWS_NS0,
                "dump": DUMP_URL,
                "dump_date": DUMP_DATE,
                "seed": MERGE_SEED,
                "drawn": len(json.loads(args.merge_draw.read_text(encoding="utf-8"))),
                "probed": candidates[IdentityClass.SAME.value],
                "earliest_year": EARLIEST_MERGE_YEAR,
                "note": (
                    "candidates_in counts the drawn pairs that were probed for both pre-merge "
                    "states. The rest of the draw is unprobed, not excluded: probing is one "
                    "history read and two revision reads per pair and the draw is resumable."
                ),
            },
            "different": {"population": STATEMENT_POPULATION, **draws[IdentityClass.DIFFERENT.value]},
            "unclear": {"population": STATEMENT_POPULATION, **draws[IdentityClass.UNCLEAR.value]},
            "api": "https://www.wikidata.org/w/api.php",
            "retrieved_at": args.retrieved_at,
        },
        "licence": {
            "name": LICENCE,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "note": (
                "Every field here is main-namespace Wikidata structured data, which Wikidata "
                "publishes under CC0 1.0: the entity JSON of two revisions, two Q-ids, revision "
                "ids and timestamps. One licence covers the whole record and attribution is "
                "courtesy rather than obligation. This repository is Apache-2.0; these terms are "
                "the file's own and are not changed by it."
            ),
        },
        "wikidata_base": "http://www.wikidata.org/entity/",
        "built_at": datetime.now(UTC).date().isoformat(),
        "min_statements": MIN_STATEMENTS,
        "earliest_merge_year": EARLIEST_MERGE_YEAR,
        "stripped_properties": list(TRUTH_PROPERTIES),
        "hand_checked": {
            "verdicts": dict(checked.most_common()),
            "of": len(resolved[IdentityClass.SAME.value]),
            "source": "docs/dedup_kill_test_judgements.tsv",
            "note": (
                "Merge pairs that also appear in the 105 hand-checked pairs of the kill test. "
                "The reasoning behind each verdict is in that file and is deliberately not "
                "carried on the record, because it names the answer. A verdict other than "
                "'correct' sets scored=false on the pair."
            ),
        },
        "candidates_in": candidates,
        "excluded": excluded,
        "resolved": {name: len(rows) for name, rows in resolved.items()},
        "state_shape": {name: state_shape(rows[:cap] if cap else rows) for name, rows in resolved.items()},
        "cap": cap,
        "pairs": pairs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(dump(payload))

    bulk = bulk_root()
    bulk.mkdir(parents=True, exist_ok=True)
    with (bulk / "pairs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for name, rows in resolved.items():
            for index, pair in enumerate(rows, 1):
                record = {"id": f"wd-{name}-{index:04d}", **{k: v for k, v in pair.items() if k != "order"}}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (bulk / "header.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=1, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")
    print(f"       {bulk / 'pairs.jsonl'} (every resolved pair, uncapped)")
    for name in candidates:
        print(f"  {name}")
        print(f"    candidates in            : {candidates[name]}")
        for reason, count in excluded[name].items():
            print(f"    excluded {count:6}          : {reason}")
        print(f"    resolved                 : {len(resolved[name])}")
        print(f"    written out              : {min(len(resolved[name]), cap or len(resolved[name]))} (cap {cap})")
    report(args.out)


def report(path: Path) -> None:
    """Read the file back and print what a rule with no judge in it scores.

    Read back rather than computed from memory, so the numbers quoted are the
    committed file's and the loader's refusals have already run.
    """
    corpus = read_pairs(path)
    positives = corpus.of(IdentityClass.SAME)
    print(f"  hard tier                  : {len(corpus.hard_tier)} of {len(positives)} positives share no name")
    print(f"  carried unscored           : {len(corpus.pairs) - len(corpus.scored)}")
    for name, rule in (("exact name match", label_baseline), ("property agreement", property_baseline)):
        baseline = score_baseline(corpus.pairs, rule, name)
        print(f"  {name}")
        print(f"    over all three classes   : {baseline.correct}/{baseline.total} ({baseline.accuracy:.1%})")
        print(
            f"    over same and different  : {baseline.decidable_correct}/{baseline.decidable} "
            f"({baseline.decidable_accuracy:.1%}, a coin scores 50.0%)"
        )
        for identity, (hits, total) in baseline.per_class.items():
            print(f"    {identity:24} : {hits}/{total} ({hits / total:.1%})")


if __name__ == "__main__":
    main()
