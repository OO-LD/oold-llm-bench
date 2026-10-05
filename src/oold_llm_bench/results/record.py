"""What a run records, and what of it may be published.

A result has to name the condition it was produced under precisely enough that
someone else can reproduce it, and it has to do that without carrying account
identifiers that mean nothing outside one subscription.

Those two goals do not conflict, because they want different fields. A
deployment name is individual and useless to anyone else. The model version
behind it is generic and pins the run. The record carries
both, and :meth:`RunRecord.publish` keeps the second.

The split is enforced by an allowlist instead of a convention, so a field
added later cannot leak by being forgotten.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "PUBLISHED_ENVIRONMENT_FIELDS",
    "PUBLISHED_MODEL_FIELDS",
    "Environment",
    "ModelSpec",
    "RunRecord",
    "config_hash",
]

PUBLISHED_MODEL_FIELDS = frozenset({
    "model",
    "model_version",
    "provider_profile",
    "region",
    "temperature",
    "max_tokens",
    "seed",
})
"""Model fields that reproduce a run and identify nobody."""

PUBLISHED_ENVIRONMENT_FIELDS = frozenset({
    "benchmark_version",
    "benchmark_sha",
    "oold_version",
    "python_version",
    "platform",
    "started_at",
})

_WITHHELD = frozenset({
    "deployment",
    "endpoint",
    "subscription",
    "resource_group",
    "resource",
    "tenant",
    "api_key",
    "api_version",
})
"""Account-scoped fields. Kept locally so a run can be repeated on the same
infrastructure, never published, because they identify one subscription and
help nobody reproduce anything."""


def config_hash(value: Any) -> str:
    """Stable hash of any JSON-serialisable configuration."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    """One model, as configured and as resolved.

    ``model_version`` is the version the provider reports, not the one the
    config asked for. Stable aliases move underneath a deployment, so the
    version the config asked for does not pin the run.
    """

    model: str
    provider_profile: str
    model_version: str | None = None
    region: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None

    deployment: str | None = None
    endpoint: str | None = None
    subscription: str | None = None
    resource_group: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_version": self.model_version,
            "provider_profile": self.provider_profile,
            "region": self.region,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "deployment": self.deployment,
            "endpoint": self.endpoint,
            "subscription": self.subscription,
            "resource_group": self.resource_group,
        }

    def publish(self) -> dict[str, Any]:
        return {key: value for key, value in self.describe().items() if key in PUBLISHED_MODEL_FIELDS}


@dataclass(frozen=True)
class Environment:
    """What produced the result, so a rerun can be compared honestly."""

    benchmark_version: str
    benchmark_sha: str | None = None
    oold_version: str | None = None
    python_version: str = field(default_factory=lambda: ".".join(map(str, sys.version_info[:3])))
    platform: str = field(default_factory=platform.platform)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def describe(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "benchmark_sha": self.benchmark_sha,
            "oold_version": self.oold_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "started_at": self.started_at,
        }

    def publish(self) -> dict[str, Any]:
        return {key: value for key, value in self.describe().items() if key in PUBLISHED_ENVIRONMENT_FIELDS}


@dataclass
class RunRecord:
    """One cell of the study. An arm, a model, a corpus, and what came out."""

    run_id: str
    arm: str
    model: ModelSpec
    environment: Environment
    enforcement: dict[str, Any]
    corpus_hash: str
    catalogue_hash: str
    prompt_hash: str
    schema_hash: str | None = None
    split: str = "dev"
    variant: str = "native"
    repetition: int = 1
    scores: list[dict[str, Any]] = field(default_factory=list)
    calls: dict[str, Any] = field(default_factory=dict)
    degradation: dict[str, Any] = field(default_factory=dict)
    selected: dict[str, list[str]] = field(default_factory=dict)
    """The shortlist a two-step orchestration produced, per entity. Empty for
    a single-shot run. Class identifiers are corpus vocabulary, so nothing
    here identifies an account."""
    document_chars: int = 0
    """How long the document was.

    Recorded because a two-step orchestration sends it twice, so the enum it
    saves on the second call is offset by a second copy of the input. Without
    this the orchestration looks cheaper than it is."""
    links: list[dict[str, str]] = field(default_factory=list)
    """Every edge the answer asserted between planned entities.

    Empty for an orchestration that hands out no ids, where a link has nothing
    to name. Recorded because a graph result is not interpretable without it:
    an orchestration that hands out ids can assert edges, and a run recording
    none cannot say whether it produced any."""
    dangling: list[dict[str, str]] = field(default_factory=list)
    """The links whose target was never emitted, a subset of ``links``.

    An answer pointing at an entity it failed to report is a different failure
    from an answer reporting no edge, and one number for both hides which."""
    unpinned: list[str] = field(default_factory=list)
    """Reference properties left open because no planned entity fitted.

    Those slots ran unconstrained. Without the names, a decode-time constraint
    that never engaged and one that engaged and was obeyed look the same."""
    invalid: list[str] = field(default_factory=list)
    """Why the answer still fails the schema after any repair. A conformance
    rate reported without these is a rate nobody can check."""
    repairs: int = 0
    """How many times the answer was sent back with its errors."""
    answer_text: str | None = None
    """What the model actually replied, kept verbatim.

    Local only, never published, because a reply can carry the document back.
    Kept so a later grader fix can be applied to results that already exist
    instead of re-running the grid, which is the only other way to measure
    what a grader got wrong.

    It is one call's reply, so on a multi-call orchestration it is the last
    one and not the answer. Regrade from :attr:`answer_payload`.
    """
    answer_payload: Any | None = None
    """The answer as the grader received it, assembled.

    Local only, for the same reason as :attr:`answer_text`. A segmented or
    multi-step cell assembles its entities across calls and ``answer_text``
    holds only the last reply, so regrading a multi-call cell needs the
    assembled answer.
    """
    notes: str | None = None

    training_match: dict[str, Any] = field(default_factory=dict)
    """How this cell's condition relates to what the model was trained on.

    ``{}`` for a base model with nothing declared. Otherwise carries the
    status and the keys that differ, so a tuned model scored off its training
    condition is visible in the record rather than in whoever remembers.
    See :mod:`oold_llm_bench.runner.training_match`.
    """

    def describe(self) -> dict[str, Any]:
        """Everything, for the local record."""
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "split": self.split,
            "variant": self.variant,
            "repetition": self.repetition,
            "model": self.model.describe(),
            "environment": self.environment.describe(),
            "enforcement": self.enforcement,
            "corpus_hash": self.corpus_hash,
            "catalogue_hash": self.catalogue_hash,
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
            "scores": self.scores,
            "calls": self.calls,
            "degradation": self.degradation,
            "selected": {k: list(v) for k, v in self.selected.items()},
            "document_chars": self.document_chars,
            "links": [dict(edge) for edge in self.links],
            "dangling": [dict(edge) for edge in self.dangling],
            "unpinned": list(self.unpinned),
            "invalid": list(self.invalid),
            "repairs": self.repairs,
            "answer_text": self.answer_text,
            "answer_payload": self.answer_payload,
            "notes": self.notes,
            "training_match": dict(self.training_match),
        }

    def publish(self) -> dict[str, Any]:
        """The same result with account-scoped fields removed.

        Everything a third party needs to repeat the run survives. What
        identifies one subscription does not.
        """
        published = self.describe()
        published["model"] = self.model.publish()
        published["environment"] = self.environment.publish()
        # A reply can quote the document back, and a document may be page text
        # we may read but not redistribute. The local file keeps it.
        published.pop("answer_text", None)
        published.pop("answer_payload", None)
        return published

    def withheld(self) -> set[str]:
        """Fields the published form dropped, so a report can say so."""
        return {
            key
            for key, value in self.model.describe().items()
            if key not in PUBLISHED_MODEL_FIELDS and value is not None
        }
