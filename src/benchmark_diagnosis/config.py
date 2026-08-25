"""Configuration loading and validation.

`Settings` mirrors ``config/default.yaml``; a user YAML is deep-merged over the
defaults so partial overrides are possible. Environment variables prefixed with
``BMD_`` take precedence over file values (e.g. ``BMD_LLM__API_KEY``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"


class StorageConfig(BaseModel):
    db_path: str = "data/benchmark_diagnosis.db"
    data_dir: str = "data"


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout_seconds: float = 120


class EvaluationConfig(BaseModel):
    harness_cmd: str = "lm_eval"
    num_fewshot: int | None = None
    batch_size: str = "auto"
    limit: int | None = None
    output_dir: str = "data/eval_runs"
    # Random-sampling evaluation: ``repeats`` samples each generative prompt
    # this many times and the harness averages the metric over the samples
    # (e.g. 5 runs at temperature=0.7). None/1 keeps the default single
    # (greedy) pass. Only generative tasks are affected — log-likelihood
    # tasks (mmlu etc.) are deterministic. Requires matching ``gen_kwargs``
    # (e.g. temperature > 0) for the runs to actually differ.
    repeats: int | None = None
    gen_kwargs: dict[str, Any] | None = None
    # Endpoint evaluation knobs (ignored for local HF weights runs except
    # ``apply_chat_template``). ``tokenizer`` is the local HF tokenizer whose
    # chat template ``apply_chat_template`` renders client-side — required
    # when ``apply_chat_template`` is on. Chat/RL-trained models degrade on
    # bare prompts; the wire protocol stays the raw completions endpoint so
    # log-likelihood tasks (mmlu etc.) keep working.
    tokenizer: str | None = None
    max_gen_toks: int | None = None
    num_concurrent: int | None = None
    max_length: int | None = None
    timeout: int | None = None
    confirm_run_unsafe_code: bool = False
    apply_chat_template: bool = False
    # Chat/RL-trained models (especially ``apply_chat_template``) often wrap
    # answers so lm-eval's ``get-answer`` filter drops them (``filtered_resps
    # == ["[invalid]"]``), understating real capability. When true, scores are
    # re-derived from the raw sample logs in ``output_dir`` with the strict
    # reparse scorer. Only tasks whose samples are dominated by ``[invalid]``
    # filtered outputs are rewritten; tasks lm-eval parsed correctly are left
    # untouched. No-op when no ``samples_*.jsonl`` exist for a task.
    reparse_scores: bool = True


class ServingConfig(BaseModel):
    engine: str = "vllm"
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    extra_args: list[str] = Field(default_factory=list)


class CurvesConfig(BaseModel):
    percentile_threshold: float = 25.0
    z_threshold: float = -1.0
    min_arch_points: int = 5


class LLMAgentDiagnosisConfig(BaseModel):
    """LLM-agent-base diagnosis (design: ``diagnosis.engine: llm_agent``).

    The path is only usable when *all three* are configured: ``enabled: true``
    plus a harness start command and an interaction command. ``harness_cmd``
    launches the harness agent with the case pack (``{case_pack}`` placeholder,
    or the ``BMD_CASE_PACK`` env var); ``interact_cmd`` sends a follow-up
    message file to the running harness (``{case_pack}`` / ``{message}``
    placeholders, or ``BMD_CASE_PACK`` / ``BMD_MESSAGE``). ``eval_cmd`` is the
    dataset-verification tool the harness calls inside the loop (default: this
    CLI's ``eval-task``).
    """

    enabled: bool = False
    harness_cmd: str | None = None
    interact_cmd: str | None = None
    eval_cmd: str | None = None
    skill_path: str | None = None
    max_rounds: int = 5
    timeout_seconds: int = 7200


class DiagnosisConfig(BaseModel):
    # Auto-diagnosis is OFF by default: evaluation always archives its results
    # and bad cases, but the diagnosis engines only run when ``enabled`` is
    # true (CLI ``--diagnose``, or ``diagnosis.enabled: true``).
    enabled: bool = False
    # Which diagnosis path runs: ``rule`` (default; deterministic statistics)
    # or ``llm_agent`` (rule base + bad-case analysis + harness loop).
    engine: Literal["rule", "llm_agent"] = "rule"
    llm_agent: LLMAgentDiagnosisConfig = Field(default_factory=LLMAgentDiagnosisConfig)
    sample_size: int = 50
    taxonomy_path: str | None = None
    # Unified intelligent diagnosis pipeline (design doc v2, stages 1-7)
    # thresholds. Stages 1-7 always run; the per-stage guards below decide
    # which evidence channels fire (e.g. coarse vs fine Stage 1, pass@k tests).
    min_items_per_capability: int = 8
    min_peers: int = 5
    min_passk_samples: int = 8
    passk_gap_threshold: float = 0.5
    pass1_high_threshold: float = 0.5
    saturation_window: int = 20
    probe_registry_path: str | None = None
    cost_table_path: str | None = None


class RecommendationConfig(BaseModel):
    retrieval_enabled: bool = False
    max_external_sources: int = 5
    advisor_mode: Literal["auto", "llm_rules", "rules"] = "auto"
    max_actions: int = 3
    experience_path: str | None = None


class RunModelConfig(BaseModel):
    """Model source + metadata for a unified ``run``.

    ``source`` selects where the evaluated model comes from: ``weights``
    (auto-deploy via vLLM), ``endpoint`` (an existing OpenAI-compatible
    inference service IP), or ``scores`` (skip evaluation, read a JSON scores
    file). When ``None`` it is auto-derived from whichever of ``weights`` /
    ``base_url`` / ``scores_file`` is set.

    ``benchmarks`` optionally narrows the evaluation to a subset of the
    representative portfolio (e.g. ``["mmlu_pro", "math"]``); only effective on
    the evaluation path (``weights`` / ``endpoint``). It is ignored when
    ``source=scores`` (the scores file already carries its own benchmark set).
    """

    name: str | None = None
    source: Literal["weights", "endpoint", "scores"] | None = None
    weights: str | None = None
    base_url: str | None = None
    scores_file: str | None = None
    benchmarks: list[str] | None = None
    arch: str | None = None
    params: float | None = None
    release_date: str | None = None


class RunOutputConfig(BaseModel):
    dir: str = "data/run_output"


class RunConfig(BaseModel):
    mode: Literal["benchmark", "analyze", "full"] = "full"
    model: RunModelConfig = Field(default_factory=RunModelConfig)
    output: RunOutputConfig = Field(default_factory=RunOutputConfig)


class Settings(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    curves: CurvesConfig = Field(default_factory=CurvesConfig)
    diagnosis: DiagnosisConfig = Field(default_factory=DiagnosisConfig)
    recommendation: RecommendationConfig = Field(default_factory=RecommendationConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returning a new dict)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply ``BMD_<SECTION>__<FIELD>`` environment variables.

    Double underscore separates nested keys, so ``BMD_RUN__MODEL__NAME`` walks
    ``data["run"]["model"]["name"]``; single underscore stays inside a key.
    Primitives are coerced YAML-style (booleans/numbers/None).
    """
    for env_key, raw in os.environ.items():
        if not env_key.startswith("BMD_"):
            continue
        parts = [p for p in env_key[4:].lower().split("__") if p]
        if len(parts) < 2:
            continue
        node = data
        for part in parts[:-1]:
            if isinstance(node, dict) and part in node and isinstance(node[part], dict):
                node = node[part]
            else:
                break
        else:
            if isinstance(node, dict) and parts[-1] in node:
                node[parts[-1]] = _coerce(raw)
    return data


def _coerce(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def resolve_advisor_mode(config: Settings, requested: str | None = None) -> str:
    """Resolve the effective recommendation advisor mode.

    ``requested`` (or ``config.recommendation.advisor_mode``) is one of
    ``auto`` / ``llm_rules`` / ``rules``:

    * ``rules`` — deterministic recommendation engine only: score the
      tool-maintained experience-base interventions against the diagnosed
      capability deficit and emit concrete datasets / hyperparameters / reason
      chain (a configured LLM is ignored and failure-mode classification is
      skipped).
    * ``llm_rules`` — the engine's Stage 2: the analyst LLM re-ranks the
      experience-base candidates (grounded, never free generation). Requires
      ``config.llm.model``; raises ``ValueError`` if no analyst LLM is
      configured so misconfiguration fails before evaluation.
    * ``auto`` — ``llm_rules`` when an analyst LLM is configured, else ``rules``.

    Returns the resolved mode (``"llm_rules"`` or ``"rules"``).
    """
    mode = requested or config.recommendation.advisor_mode
    has_llm = bool(config.llm.model)
    if mode == "rules":
        return "rules"
    if mode == "llm_rules":
        if not has_llm:
            raise ValueError(
                "advisor_mode='llm_rules' requires an analyst LLM: set "
                "llm.model (and base_url) in the config, or use "
                "advisor_mode='auto'|'rules'."
            )
        return "llm_rules"
    if mode == "auto":
        return "llm_rules" if has_llm else "rules"
    raise ValueError(f"unknown advisor_mode {mode!r}; expected auto|llm_rules|rules")


def resolve_diagnosis_engine(config: Settings, requested: str | None = None) -> str:
    """Resolve and validate the effective diagnosis engine.

    ``requested`` (or ``config.diagnosis.engine``) is one of ``rule`` /
    ``llm_agent``:

    * ``rule`` — the deterministic rule-base path (default). No extra config.
    * ``llm_agent`` — rule base + bad-case analysis + a harness loop. Only
      usable when the LLM-agent switch AND the harness start/interaction
      commands are all configured (``diagnosis.llm_agent.enabled`` +
      ``harness_cmd`` + ``interact_cmd``); otherwise raises ``ValueError`` so
      misconfiguration fails before evaluation.

    Returns the resolved engine (``"rule"`` or ``"llm_agent"``).
    """
    engine = requested or config.diagnosis.engine
    if engine == "rule":
        return "rule"
    if engine == "llm_agent":
        agent = config.diagnosis.llm_agent
        missing = [
            name
            for name, ok in (
                ("diagnosis.llm_agent.enabled", agent.enabled),
                ("diagnosis.llm_agent.harness_cmd", bool(agent.harness_cmd)),
                ("diagnosis.llm_agent.interact_cmd", bool(agent.interact_cmd)),
            )
            if not ok
        ]
        if missing:
            raise ValueError(
                "diagnosis.engine='llm_agent' requires the LLM-agent switch and "
                "the harness start/interact commands: configure "
                + ", ".join(missing)
                + " (diagnosis.llm_agent.enabled: true + harness_cmd + "
                "interact_cmd) in the config, or use diagnosis.engine='rule'."
            )
        return "llm_agent"
    raise ValueError(f"unknown diagnosis engine {engine!r}; expected rule|llm_agent")


def load_config(path: str | Path | None = None) -> Settings:
    """Load configuration, deep-merging defaults with an optional user YAML.

    Args:
        path: Optional path to a user YAML file overriding defaults.

    Returns:
        A validated :class:`Settings` instance.

    Raises:
        ValueError: if the merged config still carries the removed
            ``diagnosis.intelligent`` flag (the diagnostic paths have been
            merged — the stages 1-7 pipeline is now always on, so the flag is
            no longer meaningful).
    """
    with open(_DEFAULT_CONFIG, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}

    if path is not None:
        user_path = Path(path)
        with open(user_path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        base = _deep_merge(base, user)

    base = _apply_env_overrides(base)

    diagnosis_cfg = base.get("diagnosis") if isinstance(base, dict) else None
    if isinstance(diagnosis_cfg, dict) and "intelligent" in diagnosis_cfg:
        raise ValueError(
            "config key 'diagnosis.intelligent' has been removed: the stages 1-7 "
            "intelligent diagnosis pipeline is now always on. Drop this key from "
            "your config (and the --intelligent flag from the CLI) to continue."
        )

    return Settings.model_validate(base)
