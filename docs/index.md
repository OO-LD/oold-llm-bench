# oold-llm-bench

Where does structure have to be enforced for a model to produce data that
validates: in the prompt, at decode time, at commit time, or nowhere at all.

The task is not slot filling against one known schema. A model is given an
ontology-grounded catalogue, has to choose a class from it, and then fill what
that class declares. No judge model is involved anywhere: answers reduce to
triples and are compared programmatically.

## The two axes

A condition is named by what the prompt carried and what the decoder would
accept, so a table can be read without a legend.

| what the prompt carries | |
|---|---|
| `no-catalog` | the document alone |
| `schema-prose-catalog` | the classes, each with the units it admits |
| `schema-prose-full-catalog` | the same plus descriptions |
| `schema-dump-catalog` | the JSON Schema printed out, plus the document |

| what the decoder accepts | |
|---|---|
| `not-enforced` | anything |
| `flat-enforced` | each slot against its own enumeration, independently |
| `enforced` | `anyOf` over one branch per class, each with that class's units |

Four qualifiers name the one further thing an arm fixes: `-prose` asks for
free text, `-gated` validates and repairs without constraining generation,
`-strict` sends the schema as a grammar rather than as a request-body field,
and `-grounded` requires each value to be traceable to the document.

## What it found

**A correct per-class grammar carries the task with no catalogue at all**, and
a tune more than doubles that, and neither replaces the catalogue. Across five
models in three families, parameter count orders neither the base score nor
the gain. The numbers are in [findings](findings.md).

## Running it

```bash
uv sync --extra agent --extra corpora
oold-bench fetch-corpus quantities --dest ./schemas
oold-bench grids
oold-bench run context-ladder --models unsloth/Qwen3.5-9B --schemas ./schemas
```

The schema corpus is a generated artefact held outside this repository and
pinned by a digest, not a version number. See [corpora](corpora.md) for why
that distinction is load-bearing.

## Reading the rest

| | |
|---|---|
| [findings](findings.md) | every measured number, and what each one does not say |
| [method](method.md) | the grader, the four controls and their ceilings, the split discipline |
| [pipelines](pipelines/index.md) | what each orchestration sends, step by step, on a real task |
| [arms](arms.md) | two arms on one document, shown rather than described |
| [corpora](corpora.md) | where the documents come from and what may be redistributed |
| [results](results.md) | the published records and how to place them |
| [local stack](local-stack.md) | training and serving on your own GPUs |
| [HF endpoints](hf-endpoints.md) | training and serving on rented ones |

## Playground

One extraction at a time, in a browser, with the score, the per-step cost, the
schema that was actually sent and the validation errors beside the graph.

```bash
uv sync --extra playground
OOLD_BENCH_SCHEMAS=./schemas uv run python -m oold_llm_bench.playground
```
