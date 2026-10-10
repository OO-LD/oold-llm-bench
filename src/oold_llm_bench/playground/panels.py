"""What the benchmark measured, as rows a table can render.

Four read-outs, each answering a question the answer itself does not. The
score says whether the answer was right, the cost says what each step spent,
the schema read-out says what the model was actually sent, and the validation
read-out says why an answer that looks fine still fails its schema.

The third is the one most easily forgotten. A provider subset drops keywords
before the request leaves, so an arm can be reported as schema-constrained
while a quarter of the schema never reached the model. That is invisible
unless something prints it, so this prints it.

Rows are plain dicts and every function is pure, which keeps the read-outs
testable without a provider and without Panel installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from oold_llm_bench.grading.score import TaskScore

__all__ = [
    "cost_rows",
    "cost_total",
    "degradation_rows",
    "document_yaml",
    "schema_rows",
    "score_rows",
    "session_call_rows",
    "validation_rows",
]


def score_rows(score: TaskScore) -> list[dict[str, Any]]:
    """One row per dimension, primary first.

    Value F1 leads because it is the headline number, and the rest follow in a
    fixed order so two runs read the same way. Class, unit and shortlist are
    beside it rather than folded into it, because an arm can get every value
    right while assigning the wrong class and that is a different finding.

    A ``_near`` dimension follows the strict one it widens rather than sitting
    apart from it: it is the same question asked with a vocabulary- or
    class-lineage-aware reading of what counts as a hit, and a reader checking
    whether a miss was outright or merely under-specified wants the two
    adjacent. Absent from ``score`` wherever the task carries no lineage or the
    vocabulary carries no edge for the property, in which case the row is
    skipped here exactly as every other dimension the task did not produce is.
    """
    order = [
        "value",
        "value_near_property",
        "entity",
        "class",
        "class_near",
        "property",
        "property_near",
        "unit",
        "unit_physical",
        "shortlist",
        "duplicate",
        "provenance",
    ]
    described = score.describe()["dimensions"]
    rows = []
    for name in order:
        found = described.get(name)
        if found is None:
            continue
        rows.append({
            "dimension": name,
            "f1": found["f1"],
            "precision": found["precision"],
            "recall": found["recall"],
            "tp": found["tp"],
            "fp": found["fp"],
            "fn": found["fn"],
            "support": found["support"],
        })
    return rows


def cost_rows(calls: Any) -> list[dict[str, Any]]:
    """Calls and tokens per step, which is the attribution a total hides.

    A segmented run spends its plan call once and its fill calls once per
    distinct shortlist, so the two lines are the shape of the orchestration.
    Pooling them into one number makes a two-call arm and a five-call arm look
    like the same arm at a different price.
    """
    if calls is None:
        return []
    per_step: dict[str, dict[str, int]] = {}
    for call in calls:
        row = per_step.setdefault(
            call.step,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "errors": 0},
        )
        row["calls"] += 1
        row["input_tokens"] += call.usage.input_tokens
        row["output_tokens"] += call.usage.output_tokens
        row["cached_input_tokens"] += call.usage.cached_input_tokens
        row["errors"] += 1 if call.error else 0
    return [
        {"step": step, **values, "total_tokens": values["input_tokens"] + values["output_tokens"]}
        for step, values in sorted(per_step.items())
    ]


def cost_total(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """The same figures summed, reported beside the breakdown and not instead."""
    total = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "errors": 0}
    for row in rows:
        for key in total:
            total[key] += int(row.get(key, 0))
    return total


def session_call_rows(calls: Any, *, turn: int) -> list[dict[str, Any]]:
    """One row per call this turn made, tagged with the turn it belongs to.

    :func:`cost_rows` pools a turn's calls into one line per step, which
    answers what the turn spent; this keeps every call of its own, because a
    session log is a history of what was asked and in what order, not a bill.
    A caller accumulates these across turns, since a run only ever hands over
    one turn's calls at a time.
    """
    if calls is None:
        return []
    return [{"turn": turn, **call.describe()} for call in calls]


def document_yaml(entities: Iterable[Mapping[str, Any]]) -> str:
    """The graph's entities as YAML, one document per entity.

    Each entity is its node's own ``data``, the same dict the hover tooltip
    already reads off, so this is a second rendering of what the graph already
    holds and not a second extraction of it. ``---``-separated by
    :func:`yaml.safe_dump_all`, so one entity can be read, copied or diffed
    against the next without the brackets and quoting a JSON array would add.
    """
    import yaml

    documents = [dict(entity) for entity in entities]
    if not documents:
        return ""
    return yaml.safe_dump_all(documents, sort_keys=True, allow_unicode=True)


def degradation_rows(degradation: Any) -> list[dict[str, Any]]:
    """Which keyword the provider subset removed, and how often.

    Empty when nothing was dropped, which is a result and not a missing panel.
    """
    if degradation is None:
        return []
    return [{"keyword": name, "dropped": count} for name, count in sorted((degradation.dropped or {}).items())]


def schema_rows(
    degradation: Any,
    *,
    declared: dict[str, Any] | None = None,
    sent: dict[str, Any] | None = None,
    profile: Any = None,
) -> list[dict[str, Any]]:
    """What reached the model, against what the corpus declared.

    Three separate losses, reported apart because they have different causes
    and one number covering them would let any of them hide behind another.

    The catalogue trim removes the properties the offered classes do not
    define, which the condition decided. The provider subset removes keywords
    the endpoint will not accept, which the endpoint decided. The provider's
    optional-property cap then removes properties outright, which is the
    sharpest of the three and the least visible: a schema.org entity offered
    under the Anthropic subset arrives at twenty-five properties whatever the
    catalogue said, and nothing in the answer says so.
    """
    rows: list[dict[str, Any]] = []
    declared_properties = _entity_properties(declared)
    sent_properties = _entity_properties(sent)
    if declared_properties is not None and sent_properties is not None:
        rows.append({
            "measure": "entity properties after catalogue trim",
            "value": f"{sent_properties} of {declared_properties}",
        })
    cap = getattr(profile, "max_optional_properties", None)
    if cap is not None:
        rows.append({
            "measure": f"optional properties {getattr(profile, 'name', 'provider')} accepts",
            "value": cap,
        })
    if degradation is not None:
        described = degradation.describe()
        rows += [
            {"measure": "schema fidelity", "value": f"{described['fidelity']:.3f}"},
            {
                "measure": "schema keywords",
                "value": f"{described['keywords_after']} of {described['keywords_before']}",
            },
            {"measure": "refs inlined", "value": described["refs_inlined"]},
            {"measure": "recursion cut", "value": described["recursion_cut"]},
            {"measure": "made required", "value": described["made_required"]},
            {"measure": "constraints described", "value": described["constraints_described"]},
            {"measure": "semantics dropped", "value": described["semantics_dropped"]},
        ]
    return rows


def _entity_properties(schema: dict[str, Any] | None) -> int | None:
    """How many properties one entity may carry under this answer schema.

    The count is taken at the entity and not at the envelope, because the
    envelope is one array whatever the corpus is and says nothing about how
    much of a class reached the model.
    """
    if not isinstance(schema, dict):
        return None
    items = ((schema.get("properties") or {}).get("entities") or {}).get("items")
    if not isinstance(items, dict):
        return len(schema.get("properties") or {}) or None
    return len(items.get("properties") or {}) or None


def validation_rows(result: Any) -> list[dict[str, Any]]:
    """Why an answer still fails its schema, and what was tried.

    ``repairs`` is a count of attempts and not of successes. An answer that
    was repaired and still does not validate keeps its errors here, because
    reporting the last valid answer instead would claim a conformance rate the
    arm did not reach.
    """
    if result is None:
        return []
    rows: list[dict[str, Any]] = [
        {"check": "parsed", "detail": "yes" if getattr(result, "payload", None) is not None else "no"},
        {"check": "repair attempts", "detail": int(getattr(result, "repairs", 0) or 0)},
    ]
    invalid = list(getattr(result, "invalid", ()) or ())
    rows.append({"check": "schema errors", "detail": len(invalid)})
    rows += [{"check": "invalid", "detail": message} for message in invalid]

    dropped = list(getattr(result, "dropped", ()) or ())
    rows += [{"check": "gate dropped", "detail": name} for name in dropped]

    unpinned = list(getattr(result, "unpinned", ()) or ())
    rows += [{"check": "reference left open", "detail": name} for name in unpinned]
    return rows
