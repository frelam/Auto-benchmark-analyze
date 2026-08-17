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


class DiagnosisConfig(BaseModel):
    sample_size: int = 50
    taxonomy_path: str | None = None


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
    """

    name: str | None = None
    source: Literal["weights", "endpoint", "scores"] | None = None
    weights: str | None = None
    base_url: str | None = None
    scores_file: str | None = None
    arch: str | None = None
    params: float | None = None
    release_date: str | None = None


class RunOutputConfig(BaseModel):
    dir: str = "data/run_output"


class RunConfig(BaseModel):
    mode: Literal["analyze", "full"] = "full"
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


def load_config(path: str | Path | None = None) -> Settings:
    """Load configuration, deep-merging defaults with an optional user YAML.

    Args:
        path: Optional path to a user YAML file overriding defaults.

    Returns:
        A validated :class:`Settings` instance.
    """
    with open(_DEFAULT_CONFIG, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}

    if path is not None:
        user_path = Path(path)
        with open(user_path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        base = _deep_merge(base, user)

    base = _apply_env_overrides(base)
    return Settings.model_validate(base)
