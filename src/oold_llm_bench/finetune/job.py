"""The job body, and what a poll reports.

Kept apart from the transport so the body can be built and inspected without
an account, which is the only way to be sure what is about to be paid for.

``estimated_finish`` is not a forecast: four jobs it put at 110 to 147 hours
all finished inside three, so do not plan around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oold_llm_bench.finetune.azure import TERMINAL_STATES, DataPlane

__all__ = ["DEVELOPER_TIER", "JobRequest", "poll", "summarise"]

DEVELOPER_TIER = "developerTier"
"""The training tier this study runs on.

Stated rather than left out, because omitting the field does not leave the
tier unset: the service fills in ``standard``. Developer tier offers no
latency guarantee and trains on pre-emptible capacity, which is the right
trade for a job nothing is waiting on.

It buys nothing in queue position. Three jobs created minutes apart on this
account under ``developerTier``, ``standard`` and ``globalStandard`` came back
with estimates 12.3 hours apart in creation order, so ``estimated_finish`` is
a slot count and not a capacity reading, and the tier does not move it.

The field carrying it is a service extension: it is in Microsoft's published
samples and not in the OpenAPI spec, and their own samples spell it four
different ways. This is the spelling that appears in both the REST and the SDK
documentation. The service echoes back ``developerTier``, ``standard`` and
``globalStandard``.
"""


@dataclass(frozen=True)
class JobRequest:
    """One tuning job, as the data plane wants it.

    ``model`` is the base model identifier the account offers, not a
    deployment name. A deployment is a separate, billable object and this
    never creates one.
    """

    model: str
    training_file: str
    validation_file: str | None = None
    suffix: str | None = None
    n_epochs: int | None = None
    seed: int | None = None
    tier: str | None = DEVELOPER_TIER

    def payload(self) -> dict[str, Any]:
        """The request body, with the fields the service was not given left out.

        Omitted rather than sent as null: the hyperparameters default to a
        value the service picks, and sending a null asks it to pick nothing.
        """
        body: dict[str, Any] = {"model": self.model, "training_file": self.training_file}
        if self.validation_file:
            body["validation_file"] = self.validation_file
        if self.suffix:
            body["suffix"] = self.suffix
        if self.seed is not None:
            body["seed"] = self.seed
        if self.tier:
            body["trainingType"] = self.tier
        if self.n_epochs is not None:
            # Top level, not nested under ``method.supervised``. The spec and
            # the how-to disagree and only this form has a wire example.
            body["hyperparameters"] = {"n_epochs": self.n_epochs}
        return body


def summarise(job: dict[str, Any]) -> dict[str, Any]:
    """The fields worth reading off a job, and nothing account-scoped."""
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "model": job.get("model"),
        "trainingType": job.get("trainingType"),
        "fine_tuned_model": job.get("fine_tuned_model"),
        "trained_tokens": job.get("trained_tokens"),
        "created_at": job.get("created_at"),
        "estimated_finish": job.get("estimated_finish"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
        "hyperparameters": job.get("hyperparameters"),
    }


def poll(plane: DataPlane, job_id: str) -> tuple[dict[str, Any], bool]:
    """Read a job once, and say whether it will change again."""
    job = plane.job(job_id)
    return summarise(job), str(job.get("status", "")).lower() in TERMINAL_STATES
