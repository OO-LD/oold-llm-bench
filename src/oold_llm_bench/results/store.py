"""Writing results down while a run is still going.

A grid takes long enough that a process which only prints has lost everything
when it fails at cell four hundred. Records are appended as they arrive, one
JSON object per line, so a run that dies leaves what it already measured.

Two files are written per run. ``<run>.jsonl`` is local and carries the
deployment it used. ``<run>.published.jsonl`` carries only the fields that let
a third party repeat it. Writing both at once stops the published file
being produced later by a step somebody forgets to take.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oold_llm_bench.results.record import RunRecord

__all__ = ["ResultStore", "read_records"]


@dataclass
class ResultStore:
    """An open pair of files for one run."""

    directory: Path
    run_id: str
    publish_only: bool = False
    """Skip the local file. For a run whose records are meant to be shared as
    they are produced, where keeping a second copy with the deployment in it
    would be the only place that information exists."""

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._local = self.directory / f"{self.run_id}.jsonl"
        self._published = self.directory / f"{self.run_id}.published.jsonl"
        self.written = 0

    @property
    def local_path(self) -> Path:
        return self._local

    @property
    def published_path(self) -> Path:
        return self._published

    def append(self, record: RunRecord) -> None:
        """Write one record to both files, flushing each time.

        Flushing per record costs a little and buys the property the whole
        file exists for: a killed run has its results up to the cell it died
        on.
        """
        if not self.publish_only:
            _append_line(self._local, record.describe())
        _append_line(self._published, record.publish())
        self.written += 1

    def append_many(self, records: list[RunRecord]) -> None:
        for record in records:
            self.append(record)

    def write_summary(self, summary: dict[str, Any]) -> Path:
        """The run's own description, beside its records."""
        path = self.directory / f"{self.run_id}.summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return path


def _append_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read a result file back, skipping nothing silently.

    A truncated last line is the normal state of a file from a killed run, so
    it raises rather than being dropped: a partial record read as a whole one
    would put a wrong number in a table.
    """
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}: line {number} is not a whole record") from error
