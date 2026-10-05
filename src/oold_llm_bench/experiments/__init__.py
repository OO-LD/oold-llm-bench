"""The grids the published figures rest on, declared rather than scripted.

Every number this benchmark reports came from a grid: a set of conditions, a
set of models and a set of tasks, run as a cross product. A grid that lives in
a scratch script is a grid nobody else can run, and the figures it produced
are assertions rather than results.

So each one is a :class:`Grid` here, and ``oold-bench run <name>`` runs it.
The conditions are the part that must not move, because they are what the
figure is about; the models, the task count and the endpoint are arguments,
because they are what a reader supplies.

    >>> from oold_llm_bench.experiments import GRIDS
    >>> sorted(GRIDS)                                        # doctest: +SKIP
    ['adapter-ladder', 'context-ladder', 'enforcement-ablation', 'schemaorg-union']
"""

from __future__ import annotations

from oold_llm_bench.experiments.grids import GRIDS, Grid
from oold_llm_bench.experiments.run import RunResult, report, run_grid

__all__ = ["GRIDS", "Grid", "RunResult", "report", "run_grid"]
