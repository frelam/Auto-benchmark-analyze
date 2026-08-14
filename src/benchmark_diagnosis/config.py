"""Configuration loading and validation.

`Settings` mirrors ``config/default.yaml``; a user YAML is deep-merged over the
defaults so partial overrides are possible. Environment variables prefixed with
``BMD_`` take precedence over file values (e.g. ``BMD_LLM__API_KEY``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


class Settings(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    curves: CurvesConfig = Field(default_factory=CurvesConfig)
    diagnosis: DiagnosisConfig = Field(default_factory=DiagnosisConfig)
    recommendation: RecommendationConfig = Field(default_factory=RecommendationConfig)


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
    """Apply ``BMD_<SECTION>__<FIELD>`` environment variables (double underscore
    separates section from field; single underscore stays inside a key)."""
    for env_key, raw in os.environ.items():
        if not env_key.startswith("BMD_"):
            continue
        parts = env_key[4:].lower().split("__")
        if len(parts) != 2:
            continue
        section, field = parts
        if section in data and isinstance(data[section], dict):
            # Best-effort primitive coercion (YAML-like booleans/numbers).
            data[section][field] = _coerce(raw)
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
