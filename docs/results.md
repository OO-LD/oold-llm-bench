# Results

Every graded cell behind a published figure is on the Hub:
[OO-LD/oold-llm-bench-results](https://huggingface.co/datasets/OO-LD/oold-llm-bench-results).
One JSON object per line, 9,837 of them over 18 runs.

```python
from datasets import load_dataset

rows = load_dataset("OO-LD/oold-llm-bench-results", split="train")
```

Or read a local run through the same path the tables use:

```bash
oold-bench report results/context-ladder-20261004-1902.published.jsonl
```

## What a record does not carry

The reply itself. A record keeps the verbatim answer locally so a later
grader fix can be applied without re-running a grid, and that text is page
text whose licence is not ours to extend. The published variant is produced
by an allowlist rather than by removing fields, so a field added later is
withheld until somebody decides otherwise.

## Placing a record

Three fields decide whether two records may be compared.

**`arm`** carries the id the run was written with. Records predate the
current names, and a result file is evidence rather than a document to be
tidied, so the reader translates instead of the file being rewritten:

| in the record | published name |
|---|---|
| `A0-prose` | `no-catalog-not-enforced-prose` |
| `A0-json` | `no-catalog-not-enforced` |
| `A1` | `schema-dump-catalog-not-enforced-gated` |
| `A1-lean` | `catalog-not-enforced-gated` |
| `A2` | `schema-dump-catalog-flat-enforced` |
| `A2-strict` | `schema-dump-catalog-flat-enforced-strict` |
| `A2-enforced-only` | `catalog-flat-enforced` |
| `A3` | `schema-dump-catalog-flat-enforced-grounded` |
| `A4` | `schema-dump-catalog-enforced` |
| `A4-strict` | `schema-dump-catalog-enforced-strict` |
| `A4-enforced-only` | `catalog-enforced` |
| `A4-blind` | `no-catalog-enforced` |

`catalog` resolves to `schema-prose-catalog` or `schema-prose-full-catalog`
depending on the record's own `describe_catalogue`, because how much the
catalogue says about each class is a property of the run and not of the arm.
`oold_llm_bench.report.axes.label_of` does both steps.

**`corpus_hash` and `catalogue_hash`** place a record against the generation
it was measured on. Quantity records are not comparable across a change in
either. The catalogue rendering decides whether the prompt states the unit
enumeration the grammar enforces, and two generations of the quantity schema
module differ in the case of 234 kinds' unit identifiers. A date would not be
checkable by a third party; a hash is.

**`benchmark_sha`** reads `local` in records written before the runner was
committed, so those do not pin a code revision. Later ones do.

## Fidelity is part of the key, not an aside

A provider that cannot accept `anyOf` is sent a flattened schema instead, and
the record says so. Two cells of differing fidelity denote structurally
different treatments and the report refuses to average across them, so a
union arm on a provider that rejects unions never silently pools with one
that accepts them.

## Cells that are absent rather than zero

A call that times out or loses its connection leaves no record. It is not a
zero, and the `n` beside a number says how many cells survived. Two places
where that matters enough to state:

- The Qwen3.5-27B floor rows lost 52 of 240 calls to timeouts.
- The schema.org union arm lost 176 of 400 to the server, where the flat arm
  lost none, which is why that comparison is reported as unresolved.
