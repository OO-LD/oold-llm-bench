"""The data-plane calls a tuning job needs.

Four of them: upload a file, create a job, read a job, read its events. They
are written against the REST API rather than an SDK because the tier a job
trains on is an Azure-specific field, and a wrapper that does not know about
it would silently train on the wrong one.

Nothing here holds an account. The endpoint and the key are read from the
environment by :class:`~oold_llm_bench.clients.Credentials` and the resource
is never named in code, so this file is publishable and a run is reproducible
by whoever has their own resource.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_API_VERSION",
    "TERMINAL_STATES",
    "DataPlane",
    "FineTuneError",
    "data_plane_endpoint",
]

DEFAULT_API_VERSION = "2025-04-01-preview"
"""The version fine-tuning is documented against.

Declared here and overridable, because the inference endpoint this repo
already reads runs on its own version and the two move independently.
"""

TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
"""Job states that will not change again. Anything else is worth polling."""

_ENDPOINT_ENV = "OOLD_BENCH_FINETUNE_ENDPOINT"
_KEY_ENV = "OOLD_BENCH_API_KEY"
_VERSION_ENV = "OOLD_BENCH_FINETUNE_API_VERSION"
_INFERENCE_ENDPOINT_ENV = "OOLD_BENCH_ENDPOINT"


class FineTuneError(RuntimeError):
    """A data-plane call that did not succeed, with what the service said."""


def data_plane_endpoint(environ: Mapping[str, str] | None = None) -> str:
    """Where the tuning calls go.

    Taken from the environment, and derived from the inference endpoint when
    it is not set. The two differ: model inference for an AI Services account
    answers on the Foundry host under ``/models``, while files and tuning jobs
    answer on the Azure OpenAI host at the root. Deriving one from the other
    saves a second variable holding the same resource under a second name.
    """
    source = os.environ if environ is None else environ
    explicit = source.get(_ENDPOINT_ENV)
    if explicit:
        return explicit.rstrip("/")
    inference = source.get(_INFERENCE_ENDPOINT_ENV)
    if not inference:
        raise FineTuneError(f"set {_ENDPOINT_ENV} or {_INFERENCE_ENDPOINT_ENV}")
    parsed = urllib.parse.urlparse(inference)
    host = (parsed.hostname or "").replace(".services.ai.azure.com", ".openai.azure.com")
    if not host:
        raise FineTuneError(f"{_INFERENCE_ENDPOINT_ENV} does not name a host")
    return f"{parsed.scheme or 'https'}://{host}"


@dataclass
class DataPlane:
    """One account's tuning endpoint, reached with an API key."""

    endpoint: str
    api_key: str
    api_version: str = DEFAULT_API_VERSION
    timeout: int = 300

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> DataPlane:
        source = os.environ if environ is None else environ
        key = source.get(_KEY_ENV)
        if not key:
            raise FineTuneError(f"set {_KEY_ENV}")
        return cls(
            endpoint=data_plane_endpoint(source),
            api_key=key,
            api_version=source.get(_VERSION_ENV) or DEFAULT_API_VERSION,
        )

    def describe(self) -> dict[str, Any]:
        """The host and the version. Never the key."""
        return {
            "host": urllib.parse.urlparse(self.endpoint).hostname,
            "api_version": self.api_version,
        }

    def _url(self, path: str) -> str:
        return f"{self.endpoint}/openai/{path.lstrip('/')}?api-version={self.api_version}"

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        request.add_header("api-key", self.api_key)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - the scheme comes from a configured endpoint
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise FineTuneError(f"{request.get_method()} {error.code}: {detail[:2000]}") from error
        except urllib.error.URLError as error:
            raise FineTuneError(f"{request.get_method()} failed: {error.reason}") from error
        return json.loads(body) if body else {}

    def get(self, path: str) -> dict[str, Any]:
        return self._send(urllib.request.Request(self._url(path), method="GET"))  # noqa: S310 - the scheme comes from a configured endpoint

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310 - the scheme comes from a configured endpoint
            self._url(path),
            data=json.dumps(payload).encode("utf-8") if payload is not None else b"",
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        return self._send(request)

    def upload(self, path: Path, purpose: str = "fine-tune") -> dict[str, Any]:
        """Send one JSONL file and return what the service recorded for it."""
        boundary = f"----oold{uuid.uuid4().hex}"
        body = _multipart(boundary, purpose, path.name, path.read_bytes(), "application/json")
        request = urllib.request.Request(self._url("files"), data=body, method="POST")  # noqa: S310 - the scheme comes from a configured endpoint
        request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        return self._send(request)

    def file(self, file_id: str) -> dict[str, Any]:
        return self.get(f"files/{file_id}")

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.post("fine_tuning/jobs", payload)

    def job(self, job_id: str) -> dict[str, Any]:
        return self.get(f"fine_tuning/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Stop a job. Training already paid for is not refunded.

        A job cancelled before training begins is not billed at all, which is
        what makes a probe job cheap enough to be worth creating.
        """
        return self.post(f"fine_tuning/jobs/{job_id}/cancel")

    def events(self, job_id: str, limit: int = 20) -> list[dict[str, Any]]:
        found = self.get(f"fine_tuning/jobs/{job_id}/events")
        return list(found.get("data", []))[:limit]

    def jobs(self) -> list[dict[str, Any]]:
        return list(self.get("fine_tuning/jobs").get("data", []))


def _multipart(boundary: str, purpose: str, filename: str, content: bytes, content_type: str) -> bytes:
    """One multipart body: the purpose, then the file.

    Written out rather than delegated, because the file part has to carry a
    filename and the purpose part has to not carry one, and the difference is
    what the service reads to tell the two apart.
    """
    marker = f"--{boundary}".encode()
    parts = [
        marker,
        b'Content-Disposition: form-data; name="purpose"',
        b"",
        purpose.encode("utf-8"),
        marker,
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
        f"Content-Type: {content_type}".encode(),
        b"",
        content,
        f"--{boundary}--".encode(),
        b"",
    ]
    return b"\r\n".join(parts)
