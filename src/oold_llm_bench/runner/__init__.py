"""Declaring a study and running it.

A cell is one condition, one model, one task, one repetition. The runner owns
no judgement: it builds what the config declares, uses the shared extractor
and the shared grader, and records every choice it was given.
"""

from oold_llm_bench.runner.adapter import (
    ANSWER_SCHEMA,
    agent_factory,
    build_agent,
    build_enforcement,
    build_request,
)
from oold_llm_bench.runner.config import Cell, Condition, ExperimentConfig
from oold_llm_bench.runner.controls import (
    EmptyClient,
    RandomClient,
    WrongDocumentClient,
)
from oold_llm_bench.runner.execute import (
    CellOutcome,
    ExperimentRun,
    read_answer,
    run_cell,
    run_experiment,
)
from oold_llm_bench.runner.preflight import Preflight, control_scores, preflight

__all__ = [
    "ANSWER_SCHEMA",
    "Cell",
    "CellOutcome",
    "Condition",
    "EmptyClient",
    "ExperimentConfig",
    "ExperimentRun",
    "Preflight",
    "RandomClient",
    "WrongDocumentClient",
    "agent_factory",
    "build_agent",
    "build_enforcement",
    "build_request",
    "control_scores",
    "preflight",
    "read_answer",
    "run_cell",
    "run_experiment",
]
