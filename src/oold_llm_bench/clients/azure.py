"""Reaching the deployed models.

Three transports serve the nine models in the ladder. The catalogue records
which model uses which transport, instead of a substring check on the model
name at call time. The predecessor did that check, and an OpenAI-compatible
endpoint serving something else got OpenAI-shaped requests.

Credentials are read from the environment and never written to a record.
:class:`Credentials` holds them for the length of a run and
:meth:`Credentials.describe` reports which variables were found, not what was
in them.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from oold_llm_bench.config.models import ModelEntry

if TYPE_CHECKING:
    from oold.agent.langchain_client import LangChainClient

__all__ = ["TRANSPORTS", "Credentials", "build_client", "client_factory"]

TRANSPORTS = ("openai", "anthropic", "mistral", "unsloth", "mistral_api", "vllm_4b", "vllm_9b", "vllm_27b")
"""Every transport the catalogue may name. A model pointing at anything else
fails when the client is built, before a request is sent."""

_ENV = {
    "api_key": "OOLD_BENCH_API_KEY",
    "endpoint": "OOLD_BENCH_ENDPOINT",
    "anthropic_endpoint": "OOLD_BENCH_ANTHROPIC_ENDPOINT",
    "mistral_endpoint": "OOLD_BENCH_MISTRAL_ENDPOINT",
    "api_version": "OOLD_BENCH_API_VERSION",
}

_HOSTED = {
    "unsloth": ("UNSLOTH_API_ENDPOINT", "UNSLOTH_API_KEY"),
    "mistral_api": ("MISTRALAI_API_ENDPOINT", "MISTRALAI_API_KEY"),
    "vllm_4b": ("VLLM_4B_ENDPOINT", "VLLM_API_KEY"),
    "vllm_9b": ("VLLM_9B_ENDPOINT", "VLLM_API_KEY"),
    "vllm_27b": ("VLLM_27B_ENDPOINT", "VLLM_API_KEY"),
}
"""Transports that are a different provider rather than a different deployment.

Azure serves several families behind one key, so one key and a per-transport
endpoint was enough. These two are separate accounts with separate keys, and
sending the Azure key to them authenticates nothing. Read under their own
names, because that is what they are called in the environment every run
script already loads.
"""


@dataclass
class Credentials:
    """Endpoint, key and API version for a request."""

    api_key: str
    endpoint: str
    anthropic_endpoint: str | None = None
    mistral_endpoint: str | None = None
    hosted: dict[str, tuple[str, str]] = field(default_factory=dict)
    """Endpoint and key per transport, for providers that are not Azure."""
    api_version: str = "2024-02-15"
    local_timeout: int = 300
    """Seconds before a call to a self-hosted endpoint is abandoned.

    Longer than :attr:`timeout` because the two fail differently. A cloud
    endpoint under load returns 429 and the runner retries; one GPU under load
    queues, so an interactive call waits behind whatever batch is running.

    It was 900, calibrated against a model that was thinking before it
    answered. With ``Condition.reasoning`` set to ``"off"`` a grid cell
    answers in about five seconds idle, so 900 no longer bounds anything: it
    only decides how long a hung call holds the record stream shut. One did,
    for twenty minutes, on 2026-10-03.

    It is a ceiling and not a budget. A call that genuinely hangs still ends;
    it just stops being reported as a failure of the model."""

    max_retries: int = 1
    """How many times a failed call is tried again, counted as retries.

    Set explicitly because the default is not zero. LangChain retries twice,
    so a timeout is a floor on the wait and not the wait: 900 seconds became
    three attempts and three quarters of an hour, during which the runner
    reported nothing and the GPU sat idle. Whatever the ceiling is, it has to
    be the ceiling."""

    timeout: int = 180
    """Seconds before a call is abandoned.

    The predecessor used 900. A runner that consumes outcomes in declared
    order stalls its whole record stream on one hung call, so a long timeout
    turns a single slow request into a quarter hour of apparent deadlock. A
    cell that has not answered in three minutes has failed."""

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, transports: Iterable[str] | None = None) -> Credentials:
        """Read credentials, naming what is missing.

        If a variable is unset, the run fails at the first call, and the grid
        up to that point has already cost money.

        ``transports`` narrows what is required to what the chosen models
        actually answer on. Azure serves several families behind one key, so
        its pair is required whenever a model names one of those; a grid that
        runs entirely on a self-hosted endpoint needs neither, and demanding
        them would put an account between a third party and a reproduction.
        """
        source = os.environ if environ is None else environ
        wanted = set(transports) if transports is not None else set()
        unknown = sorted(wanted - set(TRANSPORTS))
        if unknown:
            raise RuntimeError(f"unknown transport(s): {', '.join(unknown)}, expected one of {', '.join(TRANSPORTS)}")
        # Azure unless the caller said which transports it needs and none of
        # them is Azure. A hosted pair is required only when named, so an
        # unnamed one stays optional and is picked up below if it is set.
        needs_azure = transports is None or bool(wanted - set(_HOSTED))
        missing = [_ENV[key] for key in ("api_key", "endpoint") if needs_azure and not source.get(_ENV[key])]
        missing += [
            name for transport in sorted(wanted & set(_HOSTED)) for name in _HOSTED[transport] if not source.get(name)
        ]
        if missing:
            raise RuntimeError(f"missing credentials: {', '.join(dict.fromkeys(missing))}")
        endpoint = source.get(_ENV["endpoint"], "")
        return cls(
            api_key=source.get(_ENV["api_key"], ""),
            endpoint=endpoint,
            anthropic_endpoint=source.get(_ENV["anthropic_endpoint"]) or None,
            mistral_endpoint=source.get(_ENV["mistral_endpoint"]) or endpoint,
            api_version=source.get(_ENV["api_version"]) or "2024-02-15",
            hosted={
                name: (source[url], source[key])
                for name, (url, key) in _HOSTED.items()
                if source.get(url) and source.get(key)
            },
        )

    def timeout_for(self, transport: str) -> int:
        """How long this transport is given, which is a fact about the host."""
        return self.local_timeout if transport.startswith(("unsloth", "vllm")) else self.timeout

    def key_for(self, transport: str) -> str:
        """The key this transport authenticates with.

        Falls back to the Azure key, because every transport but the hosted
        ones is a deployment behind it.
        """
        pair = self.hosted.get(transport)
        return pair[1] if pair else self.api_key

    def endpoint_for(self, transport: str) -> str:
        pair = self.hosted.get(transport)
        if pair:
            url = pair[0].rstrip("/")
            # An OpenAI-shaped client appends `/chat/completions`, so its base
            # has to carry the version segment. The variable names a host,
            # because that is also what the training API is reached at, and
            # guessing wrong here is a 404 rather than a visible failure.
            if transport.startswith(("unsloth", "vllm")) and not url.endswith("/v1"):
                url = url + "/v1"
            return url
        if transport in _HOSTED:
            url, key = _HOSTED[transport]
            raise RuntimeError(f"the {transport} transport needs {url} and {key} in the environment")
        if transport == "anthropic":
            if not self.anthropic_endpoint:
                raise RuntimeError(
                    f"the anthropic transport needs {_ENV['anthropic_endpoint']}, "
                    f"which is a different host from the OpenAI-compatible one"
                )
            return self.anthropic_endpoint
        if transport == "mistral":
            return self.mistral_endpoint or self.endpoint
        return self.endpoint

    def describe(self) -> dict[str, bool]:
        """Which variables were found. Never their contents."""
        return {name: bool(getattr(self, key, None)) for key, name in _ENV.items() if key != "api_version"}


def _llm_for(
    entry: ModelEntry, credentials: Credentials, temperature: float | None, reasoning: str | None = None
) -> Any:
    headers = {"x-ms-api-version": credentials.api_version}
    base_url = credentials.endpoint_for(entry.transport)
    api_key = credentials.key_for(entry.transport)
    timeout = credentials.timeout_for(entry.transport)
    retries = credentials.max_retries
    # A chat template decides whether a model thinks before it answers, so the
    # switch rides on the request rather than on the schema. Only the local
    # engines read it; a provider that does not simply ignores the field.
    thinking_off = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    tuning_extra: dict[str, Any] = thinking_off if reasoning == "off" else {}
    # Omitted, not defaulted. A model that rejects the parameter fails the
    # call outright, so sending it anyway would lose every cell for that model.
    tuning: dict[str, Any] = (
        {"temperature": temperature} if entry.accepts_temperature and temperature is not None else {}
    )

    if entry.transport == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=entry.model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=retries,
            **tuning,
            default_headers=headers,
        )

    if entry.transport in ("mistral", "mistral_api"):
        from langchain_mistralai.chat_models import ChatMistralAI

        return ChatMistralAI(
            model=entry.model,
            api_key=api_key,
            endpoint=base_url,
            timeout=timeout,
            max_retries=retries,
            **tuning,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=entry.model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=retries,
        **tuning,
        **tuning_extra,
        default_headers=headers,
    )


def build_client(
    entry: ModelEntry,
    credentials: Credentials,
    *,
    temperature: float | None = 0.0,
    reasoning: str | None = None,
) -> LangChainClient:
    """Build the client for this model.

    ``entry.model`` is recorded as the identifier, not the deployment name.
    The two happen to match on this resource and will not on another.
    """
    from oold.agent.langchain_client import LangChainClient

    if entry.transport not in TRANSPORTS:
        raise ValueError(f"{entry.model} names transport {entry.transport!r}, expected one of {list(TRANSPORTS)}")
    return LangChainClient(
        _llm_for(entry, credentials, temperature, reasoning),
        model=entry.model,
        fold_system=not entry.system_role,
    )


def client_factory(credentials: Credentials) -> Any:
    """A ``client_for`` callback for
    :func:`~oold_llm_bench.runner.adapter.agent_factory`.

    Builds one client per cell. Reusing a single client across cells would
    share provider-side state that nothing in the record accounts for.
    """
    from oold_llm_bench.config.models import entry_for

    def client_for(cell: Any) -> LangChainClient:
        return build_client(
            entry_for(cell.model.model),
            credentials,
            temperature=cell.model.temperature,
            reasoning=getattr(cell.condition, "reasoning", None),
        )

    return client_for
