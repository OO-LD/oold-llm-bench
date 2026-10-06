"""The models a study runs, and the versions they were pinned at.

Only generic facts may be committed here: model, version, provider profile. A
third party needs these to repeat the run. Deployment, endpoint and
subscription are account-specific and useless to anyone else, so the caller
supplies them at run time.

Versions are pinned because a stable alias can point to a new version without
notice. :func:`drifted` compares what the provider reports now against what is
recorded here. A run that finds a difference stops instead of measuring a
different model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from oold_llm_bench.results.record import ModelSpec

__all__ = ["LADDER", "ModelEntry", "drifted", "entry_for", "load_ladder", "spec_for"]


@dataclass(frozen=True)
class ModelEntry:
    """One model as the study names it, with nothing account-scoped in it."""

    model: str
    model_version: str
    """The version the provider reported when this entry was recorded."""
    provider_profile: str
    """Declared, never derived from the model name. The predecessor used a
    substring check on the name and sent OpenAI-shaped schemas to an
    OpenAI-compatible endpoint that served something else."""
    tier: str
    family: str
    transport: str = "openai"
    """Which client reaches this model, a separate question from which JSON
    Schema subset it accepts. ``mistral-small-2503`` answers on the Mistral
    transport and takes the OpenAI schema subset, so one field cannot carry
    both."""
    trained_on: dict[str, object] | None = None
    """The condition this model was fine-tuned under, or ``None`` for a base.

    A tuned model answers a question about the prompt it was trained on. Shown
    a different one it is not a tuned model under test, it is a model out of
    distribution, and the number means something else.

    Recorded here and checked against the evaluation condition because
    remembering the rule is what failed. The Azure H4 encoded the pairing in
    the cell name, ``tuned-lean+lean``, so the mismatch could not be written
    down; the local run reused the general grid, which has its own condition
    axis and no idea a tune happened. The adapters were trained on a 2,292
    character catalogue of bare names and evaluated against a 38,654 character
    described one, and the first sign of it was a score moving the wrong way.

    Compared key by key. Any key absent here is not claimed about and does not
    count as a mismatch, so adding an axis later does not retroactively
    invalidate a record."""

    system_role: bool = True
    """Whether the deployment actually uses a system message. The
    ``phi-4-mini-instruct`` deployment accepts one, returns 200 and discards
    it: a prompt carrying a schema arrives counted at five tokens. Where this
    is ``False`` the system content is folded into the user turn, so the model
    is asked the same question the others are asked."""
    accepts_temperature: bool = True
    """Some newer models reject the parameter outright. Sending it fails the
    call, and recording a temperature that was never sent would be worse: the
    record would name a setting the run did not use."""
    notes: str | None = None


LADDER: tuple[ModelEntry, ...] = (
    ModelEntry(
        model="phi-4-mini-instruct",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="phi",
        system_role=False,
        notes=(
            "Two separate limits, measured 2026-09-27. The endpoint refuses "
            "response_format json_schema outright, accepting only text or "
            "json_object, so this model cannot run a flat-enforced arm. Separately, the "
            "deployment discards the system message, so the instruction has "
            "to be folded into the user turn or the model is asked nothing."
        ),
    ),
    ModelEntry(
        model="mistral-small-2503",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="mistral",
        transport="mistral",
    ),
    ModelEntry(
        model="qwen3.5-4b",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "Qwen3.5-4B on vLLM 0.30.0, guidance backend. Measured 2026-10-03: "
            "free decoding answers 'metres', json_schema answers 'meter', so "
            "the grammar is real and ``-strict`` means here what it means on no "
            "Azure deployment."
        ),
    ),
    ModelEntry(
        model="unsloth/Qwen3.5-4B",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "The same weights as qwen3.5-4b, under the name a Hugging Face "
            "endpoint serves them by. An endpoint built from a repository "
            "answers to the repository id unless --served-model-name is "
            "passed at creation, and that is a startup flag. Two entries for "
            "one model is the honest way to say that, rather than rewriting "
            "the id in the client and losing which host answered."
        ),
    ),
    ModelEntry(
        model="qwen3.5-4b-lean",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "OO-LD/oold-quantities-lean-qwen4b-r16-s1000, trained on HF Jobs at "
            "r=16 for 1000 steps with target_modules all-linear. The recipe "
            "is identical for Phi-4 and the DeepSeek distill, which is the "
            "point: the earlier adapters named Qwen's module list, which "
            "matches nothing on Phi-4, so a cross-model comparison needed one "
            "rule that resolves on every architecture."
        ),
    ),
    ModelEntry(
        model="qwen3.5-4b-tuned",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "Qwen3.5-4B with the LoRA adapter "
            "SimonTaurus/Qwen3.5-4B-oold-quantities-bf16-e2, r=16, served by "
            "the same vLLM process as the base under --lora-modules. The pair "
            "differs in the adapter and in nothing else, which is what makes "
            "the comparison a comparison. Trained on quantities under the gated arm with no schema shown, "
            "so a schema.org number from it measures transfer and not the "
            "tune. Not published. It belongs to a sweep this benchmark does not report, and stays private rather than being deleted."
        ),
    ),
    ModelEntry(
        model="qwen3.5-4b-r64-s2000",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "SimonTaurus/Qwen3.5-4B-oold-lean-r64-s2000, trained on HF "
            "Jobs on the same lean dataset as the r=16 adapter. The sweep "
            "exists because that one reached a class accuracy of 1.00 and a "
            "unit accuracy of 0.07: it learned the task and not the 2,864 "
            "unit identifiers, and rank and exposure are the two things that "
            "could be the reason. Not published. It belongs to a sweep this benchmark does not report, and stays private rather than being deleted."
        ),
    ),
    ModelEntry(
        model="qwen3.5-4b-r128-s2000",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "SimonTaurus/Qwen3.5-4B-oold-lean-r128-s2000, trained on HF "
            "Jobs on the same lean dataset as the r=16 adapter. The sweep "
            "exists because that one reached a class accuracy of 1.00 and a "
            "unit accuracy of 0.07: it learned the task and not the 2,864 "
            "unit identifiers, and rank and exposure are the two things that "
            "could be the reason. Not published. It belongs to a sweep this benchmark does not report, and stays private rather than being deleted."
        ),
    ),
    ModelEntry(
        model="qwen3.5-4b-r64-s6000",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "SimonTaurus/Qwen3.5-4B-oold-lean-r64-s6000, trained on HF "
            "Jobs on the same lean dataset as the r=16 adapter. The sweep "
            "exists because that one reached a class accuracy of 1.00 and a "
            "unit accuracy of 0.07: it learned the task and not the 2,864 "
            "unit identifiers, and rank and exposure are the two things that "
            "could be the reason. Not published. It belongs to a sweep this benchmark does not report, and stays private rather than being deleted."
        ),
    ),
    ModelEntry(
        model="qwen3.5-4b-r64-s1000",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "OO-LD/oold-quantities-lean-qwen4b-r64-s1000, trained on the box. "
            "Rank raised fourfold, exposure held. Final training loss 0.8319 against the baseline's 0.8320. "
            "One of a pair that moves a single axis each against the r=16 "
            "1000-step adapter, which reached class accuracy 1.00 and unit "
            "accuracy 0.07: it learned the task and the class names and not "
            "the 2,864 unit identifiers."
        ),
    ),
    ModelEntry(
        model="qwen3.5-4b-r16-s4000",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "OO-LD/oold-quantities-lean-qwen4b-r16-s4000, trained on the box. "
            "Exposure raised fourfold, rank held. Final training loss 0.7689 against the baseline's 0.8320. "
            "One of a pair that moves a single axis each against the r=16 "
            "1000-step adapter, which reached class accuracy 1.00 and unit "
            "accuracy 0.07: it learned the task and the class names and not "
            "the 2,864 unit identifiers."
        ),
    ),
    ModelEntry(
        model="qwen3.5-4b-20k",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_4b",
        notes=(
            "OO-LD/oold-quantities-lean-20k-qwen4b-r16-s10000. Twenty thousand "
            "examples, one pass, r=16. Rank and repetition were both ruled "
            "out first: 16 to 64 left F1 at 0.09 against 0.10, and four "
            "passes took it to 0.03 while the training loss fell, which is "
            "overfitting rather than exposure. What moves here is coverage, "
            "393 distinct units at a median of two uses each against 647 at "
            "eighteen."
        ),
    ),
    ModelEntry(
        model="unsloth/Qwen3.5-9B",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_9b",
        notes=(
            "The same weights as qwen3.5-9b, under the name a Hugging Face "
            "endpoint built from the repository answers to. See the 4B entry "
            "for why two names for one model is the honest way to say it."
        ),
    ),
    ModelEntry(
        model="qwen3.5-9b-r16-s1000",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_9b",
        notes=(
            "OO-LD/oold-quantities-lean-qwen9b-r16-s1000. the baseline recipe, retrained with all-linear so the sweep differs in one thing. Swept at the context "
            "floor only, because that is the one rung where a tune is worth "
            "anything: with the catalogue present every adapter measured so "
            "far costs rather than buys."
        ),
    ),
    ModelEntry(
        model="qwen3.5-9b-r16-s1000-matched",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_9b",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        notes=(
            "OO-LD/oold-quantities-lean-qwen9b-r16-s1000-matched. The exposure control: "
            "byte-identical to the 4000-step payload but for max_steps, and "
            "checked at the tensor level rather than from a config field. "
            "qwen3.5-9b-r16-s1000 cannot serve as this control: 716 tensors "
            "against 256, because it also adapts linear_attn and the whole "
            "27-block vision tower, which never sees an image here."
        ),
    ),
    ModelEntry(
        model="qwen3.5-9b-r16-s1000-linattn",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_9b",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        notes=(
            "OO-LD/oold-quantities-lean-qwen9b-r16-s1000-linattn. The middle "
            "rung of the module ladder: text projections plus linear_attn, no "
            "vision tower. 496 tensors against the text-only control's 256 and "
            "the published adapter's 716, so what the visual encoder is worth "
            "is the difference between this and that."
        ),
    ),
    ModelEntry(
        model="qwen3.5-9b-r16-s4000",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_9b",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        notes=(
            "OO-LD/oold-quantities-lean-qwen9b-r16-s4000. Exposure raised fourfold "
            "against qwen3.5-9b-r16-s1000-matched, every other field held. "
            "Final loss 0.7307 against the control's 0.8298."
        ),
    ),
    ModelEntry(
        model="qwen3.5-9b-r64-s1000",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_9b",
        notes=(
            "OO-LD/oold-quantities-lean-qwen9b-r64-s1000. rank raised fourfold, exposure held. Swept at the context "
            "floor only, because that is the one rung where a tune is worth "
            "anything: with the catalogue present every adapter measured so "
            "far costs rather than buys."
        ),
    ),
    ModelEntry(
        model="qwen3.5-9b-tuned",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_9b",
        notes=(
            "Qwen3.5-9B with the LoRA adapter "
            "OO-LD/oold-quantities-qwen9b-r16-e2-modulelist, r=16. Same "
            "process as the base, same caveat on corpus."
        ),
    ),
    ModelEntry(
        model="qwen3.5-9b",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="qwen",
        transport="vllm_9b",
        notes="Qwen3.5-9B on vLLM 0.30.0, guidance backend, same grammar as the 4B.",
    ),
    ModelEntry(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        model_version="1",
        provider_profile="openai",
        tier="mid",
        family="deepseek",
        transport="vllm_27b",
        notes=(
            "32B, Qwen2ForCausalLM. A cross-family datapoint and not a rung "
            "of the size ladder: it is a DeepSeek distill of Qwen2.5, so it "
            "shares neither the generation nor the pretraining of the 4B and "
            "9B beside it. Qwen3.5 has no 32B; its third rung is the 27B."
        ),
    ),
    ModelEntry(
        model="dsr1-32b-lean",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="mid",
        family="deepseek",
        transport="vllm_27b",
        notes=(
            "OO-LD/oold-quantities-lean-dsr1-r16-s1000, the same r=16 all-linear "
            "recipe and thousand examples as every other adapter here. 448 "
            "adapted modules against Phi-4's 160 and the 4B's 346, because "
            "all-linear resolves to what each architecture actually has."
        ),
    ),
    ModelEntry(
        model="qwen3.5-27b",
        model_version="1",
        provider_profile="openai",
        tier="mid",
        family="qwen",
        transport="vllm_27b",
        notes=(
            "unsloth/Qwen3.5-27B on vLLM 0.30.0, bf16, tensor parallel over "
            "both A40s. The third rung of a same-family ladder: the 4B and "
            "the 9B both answer 0.03 on unit accuracy with a bare catalogue, "
            "so there is no size trend between them at all, and this decides "
            "whether one appears by 27B or whether the unit vocabulary is "
            "absent from the family. bf16 rather than FP8 because the "
            "question is what the weights memorised, and quantisation "
            "degrades exactly that first."
        ),
    ),
    ModelEntry(
        model="unsloth/Qwen3.5-27B",
        model_version="1",
        provider_profile="openai",
        tier="mid",
        family="qwen",
        transport="vllm_27b",
        notes="The third rung of the same-family ladder, as the endpoint names it.",
    ),
    ModelEntry(
        model="qwen3.5-27b-lean",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="mid",
        family="qwen",
        transport="vllm_27b",
        notes=(
            "OO-LD/oold-quantities-lean-qwen27b-r16-s1000, the same r=16 "
            "all-linear recipe as every other adapter here. Completes the "
            "same-family size series 4B, 9B, 27B at the context floor."
        ),
    ),
    ModelEntry(
        model="qwen3.8-27b",
        model_version="1",
        provider_profile="openai",
        tier="mid",
        family="qwen",
        transport="vllm_27b",
        notes=(
            "unsloth/Qwen3.8-27B on vLLM 0.30.0, bf16, tensor parallel over "
            "both A40s. A second generation rather than a second family, so "
            "it narrows the claim the Qwen3.5 ladder could not support on "
            "its own: 4B, 9B and 27B all answer 0.03 on unit accuracy, which "
            "says nothing yet about models trained on other corpora."
        ),
    ),
    ModelEntry(
        model="unsloth/gemma-4-12B-it-qat-GGUF",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="gemma",
        transport="unsloth",
        notes=(
            "12B, quantisation-aware GGUF, served through Studio's llama.cpp "
            "backend. The GGUF matters: Studio's other path is Transformers, "
            "which has no grammar engine at all, so a safetensors model there "
            "would answer every enforcement arm unenforced and record it. "
            "Probed 2026-10-04, free decoding answers 'metres' and "
            "json_schema answers 'meter'. A sixth family for the question of "
            "who holds the QUDT identifiers; the first five all answer 0.02."
        ),
    ),
    ModelEntry(
        model="unsloth/Qwen3.8-27B-GGUF",
        model_version="1",
        provider_profile="openai",
        tier="mid",
        family="qwen",
        transport="unsloth",
        notes=(
            "Served locally on two A40s, BF16, 262k context. The only model "
            "in this ladder whose endpoint enforces a schema as a grammar "
            "rather than sending it as a hint: measured 2026-10-02, the free "
            "and json_object modes answer 'metres' where json_schema answers "
            "'meter'. So ``-strict`` means something here that it does not "
            "mean on Azure, and the two must not be pooled."
        ),
    ),
    ModelEntry(
        model="mistral-small-2603",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="mistral",
        transport="mistral_api",
        notes=(
            "Mistral's own API rather than the Azure deployment. Devstral is "
            "not served there at all, 53 models and no match, and Mistral "
            "documents Small 4 as unifying Instruct, Magistral and Devstral "
            "into one model, so this is the nearest thing to a Devstral "
            "number that can be had without fine-tuning one."
        ),
    ),
    ModelEntry(
        model="llama-3.3-70b-instruct",
        model_version="5",
        provider_profile="openai",
        tier="mid",
        family="llama",
    ),
    ModelEntry(
        model="microsoft/phi-4",
        model_version="1",
        provider_profile="openai",
        tier="small",
        family="phi",
        transport="vllm_4b",
        notes=(
            "14B on a Hugging Face endpoint running vLLM 0.30.0, which is "
            "the reason it is here twice. The Azure deployment of the same "
            "weights rejects response_format json_schema outright, so it "
            "cannot run an enforcement arm at all; this one can."
        ),
    ),
    ModelEntry(
        model="phi-4-lean",
        model_version="1",
        trained_on={
            "corpus": "quantities",
            "signal": "named",
            "describe_catalogue": False,
            "pin_units": False,
            "catalogue_size": 100,
        },
        provider_profile="openai",
        tier="small",
        family="phi",
        transport="vllm_4b",
        notes=(
            "OO-LD/oold-quantities-lean-phi4-r16-s1000. The same r=16 "
            "all-linear recipe and the same thousand examples as the Qwen "
            "adapters, which is what makes the two comparable: a module list "
            "naming Qwen's projections matches nothing on Phi-4, where "
            "attention is fused into qkv_proj."
        ),
    ),
    ModelEntry(
        model="phi-4",
        model_version="7",
        provider_profile="openai",
        tier="small",
        family="phi",
        transport="openai",
        notes=(
            "14B. On the shortlist because the Structured Output Benchmark "
            "(arXiv:2604.25359) ranks it second among open weights at 0.831 "
            "overall. That benchmark hands the model a schema and asks it to "
            "fill one, with no catalogue to select from and no closed "
            "vocabulary, so it says the model is good at structured output "
            "and nothing about whether it holds QUDT identifiers."
        ),
    ),
    ModelEntry(
        model="gpt-oss-120b",
        model_version="1",
        provider_profile="openai",
        tier="large",
        family="gpt-oss",
    ),
    ModelEntry(
        model="deepseek-v3.2",
        model_version="1",
        provider_profile="openai",
        tier="large",
        family="deepseek",
    ),
    ModelEntry(
        model="gpt-5-nano",
        model_version="2025-08-07",
        provider_profile="openai",
        tier="small",
        family="gpt-5",
    ),
    ModelEntry(
        model="gpt-5-mini",
        model_version="2025-08-07",
        provider_profile="openai",
        tier="mid",
        family="gpt-5",
    ),
    ModelEntry(
        model="claude-haiku-4-5",
        model_version="20251001",
        provider_profile="anthropic",
        tier="mid",
        family="claude",
        transport="anthropic",
    ),
    ModelEntry(
        model="claude-sonnet-5",
        model_version="2",
        provider_profile="anthropic",
        tier="frontier",
        family="claude",
        transport="anthropic",
        accepts_temperature=False,
        notes="Rejects temperature outright: 'temperature is deprecated for this model'.",
    ),
)
"""The size ladder, plus frontier anchors.

Five open models carry H3: does a decode-time constraint help most where the
model is weakest? Four hosted models anchor the top so a ladder effect can be
told apart from a general one.

Transports were taken from the predecessor's config, where each one is
recorded against a model that answered on it.

Versions were read from the provider on 2026-09-27. Re-read them before a
run: see :func:`drifted`.
"""


def entry_for(model: str) -> ModelEntry:
    for entry in LADDER:
        if entry.model == model:
            return entry
    raise KeyError(f"unknown model {model!r}, expected one of {sorted(e.model for e in LADDER)}")


def spec_for(
    entry: ModelEntry,
    *,
    deployment: str | None = None,
    endpoint: str | None = None,
    region: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = 1024,
    seed: int | None = None,
) -> ModelSpec:
    """One entry plus the account details needed to reach it.

    ``max_tokens`` is a bound and not a budget. It defaulted to ``None``,
    which hands the model the whole context window: a Qwen3.5-9B that began
    restating a schema shown in its prompt then generated sixteen thousand tokens
    per call and never stopped, and because the runner consumes outcomes in
    declared order three such calls shut the record stream for forty minutes
    while the card stayed busy. An answer here is tens of tokens. A truncated
    one is recorded as a parse error, which is the honest outcome, and is
    cheaper than an unbounded one.

    Temperature defaults to zero and is recorded. Without it, two runs can
    differ and nothing in the record explains why. A model that rejects the
    parameter records ``None``, which says it was not sent.
    """
    return ModelSpec(
        model=entry.model,
        provider_profile=entry.provider_profile,
        model_version=entry.model_version,
        region=region,
        temperature=temperature if entry.accepts_temperature else None,
        max_tokens=max_tokens,
        seed=seed,
        deployment=deployment or entry.model,
        endpoint=endpoint,
    )


def load_ladder(
    resolve: Callable[[ModelEntry], Mapping[str, Any]] | None = None,
    *,
    tiers: tuple[str, ...] | None = None,
    families: tuple[str, ...] | None = None,
) -> list[ModelSpec]:
    """The ladder as specs, with account details from the caller.

    ``resolve`` supplies whatever is individual: deployment, endpoint,
    region. Leaving it out gives a ladder that names models and versions but
    reaches nothing. That is enough for a test or a dry run.
    """
    chosen = [e for e in LADDER if (tiers is None or e.tier in tiers) and (families is None or e.family in families)]
    specs: list[ModelSpec] = []
    for entry in chosen:
        extra: dict[str, Any] = dict(resolve(entry)) if resolve is not None else {}
        specs.append(
            spec_for(
                entry,
                deployment=extra.get("deployment"),
                endpoint=extra.get("endpoint"),
                region=extra.get("region"),
                temperature=float(extra.get("temperature", 0.0)),
                max_tokens=extra.get("max_tokens"),
                seed=extra.get("seed"),
            )
        )
    return specs


def drifted(live: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    """Models whose version no longer matches what is recorded here.

    ``live`` maps model name to the version the provider reports now. A
    non-empty result has to stop a run. The predecessor's config never changed
    while the provider drifted for seven months, so numbers from that period
    cannot be matched to a version.

    Models the provider does not report are missing, not drifted. The caller
    has to notice that.
    """
    changes: dict[str, tuple[str, str]] = {}
    for entry in LADDER:
        reported = live.get(entry.model)
        if reported is not None and reported != entry.model_version:
            changes[entry.model] = (entry.model_version, reported)
    return changes
