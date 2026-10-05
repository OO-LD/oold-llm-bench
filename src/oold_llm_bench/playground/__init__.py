"""A human-evaluation playground for this benchmark.

A document goes in, a graph comes out, and beside the graph sit the four
things the benchmark measures that an answer does not show: the score against
ground truth, what each step of the orchestration cost, how much of the schema
actually reached the model, and why an answer that parsed still fails to
validate.

The orchestration, the arm and the model are choices rather than constants,
which is the whole point. The predecessor this is ported from was wired to one
agent, so it could show that an extraction had happened and nothing about
whether a different way of running it would have gone better.

Panel and panelini are an optional extra. Everything except
:mod:`~oold_llm_bench.playground.app` imports without them, so the graph
conversion, the read-outs and the corpus loading are testable with no
interface installed:

.. code-block:: bash

    uv sync --extra playground
    OOLD_BENCH_SCHEMAS=/path/to/schemaorg/generated uv run python -m oold_llm_bench.playground
"""

from oold_llm_bench.playground.corpora import (
    ALL_CLASSES,
    CORPORA,
    SCHEMAS_ENV,
    MissingSchemas,
    SchemaCorpus,
    catalogue_sets,
    load_schemaorg,
    paste_task,
    schemaorg_tasks,
    schemas_directory,
    wiki_tasks,
)
from oold_llm_bench.playground.graph import (
    VIS_OPTIONS,
    Diff,
    Edge,
    Graph,
    GraphState,
    Node,
    build_graph,
    links_of,
)
from oold_llm_bench.playground.identity import (
    CLOSE_MATCH,
    DIFFERENT,
    EXACT_MATCH,
    Comparable,
    Decision,
    Judge,
    Ledger,
    ModelJudge,
    NoJudge,
    coverage,
    decide,
)
from oold_llm_bench.playground.panels import (
    cost_rows,
    cost_total,
    degradation_rows,
    schema_rows,
    score_rows,
    validation_rows,
)
from oold_llm_bench.playground.replay import ReplayClient
from oold_llm_bench.playground.session import (
    ARM_ORDER,
    ORCHESTRATIONS,
    Options,
    Outcome,
    arm_names,
    build_cell,
    model_names,
    run_once,
)

__all__ = [
    "ALL_CLASSES",
    "ARM_ORDER",
    "CLOSE_MATCH",
    "CORPORA",
    "DIFFERENT",
    "EXACT_MATCH",
    "ORCHESTRATIONS",
    "SCHEMAS_ENV",
    "VIS_OPTIONS",
    "Comparable",
    "Decision",
    "Diff",
    "Edge",
    "Graph",
    "GraphState",
    "Judge",
    "Ledger",
    "MissingSchemas",
    "ModelJudge",
    "NoJudge",
    "Node",
    "Options",
    "Outcome",
    "ReplayClient",
    "SchemaCorpus",
    "arm_names",
    "build_cell",
    "build_graph",
    "catalogue_sets",
    "cost_rows",
    "cost_total",
    "coverage",
    "decide",
    "degradation_rows",
    "links_of",
    "load_schemaorg",
    "model_names",
    "paste_task",
    "run_once",
    "schema_rows",
    "schemaorg_tasks",
    "schemas_directory",
    "score_rows",
    "validation_rows",
    "wiki_tasks",
]
