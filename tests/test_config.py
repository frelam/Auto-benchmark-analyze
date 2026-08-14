"""Tests for run-profile config, nested env overrides, and advisor resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark_diagnosis.config import Settings, load_config, resolve_advisor_mode


def test_defaults_include_run_and_advisor() -> None:
    settings = load_config()
    assert settings.run.mode == "full"
    assert settings.run.output.dir == "data/run_output"
    assert settings.run.model.name is None
    assert settings.run.model.source is None
    assert settings.recommendation.advisor_mode == "auto"


def test_run_mode_rejects_bogus_literal() -> None:
    with pytest.raises(ValidationError):
        Settings(run={"mode": "everything"})


def test_advisor_mode_rejects_bogus_literal() -> None:
    with pytest.raises(ValidationError):
        Settings(recommendation={"advisor_mode": "magic"})


def test_env_override_walks_nested_keys(monkeypatch) -> None:
    monkeypatch.setenv("BMD_RUN__MODEL__NAME", "env-model")
    monkeypatch.setenv("BMD_RUN__MODE", "analyze")
    monkeypatch.setenv("BMD_RECOMMENDATION__ADVISOR_MODE", "rules")
    settings = load_config()
    assert settings.run.model.name == "env-model"
    assert settings.run.mode == "analyze"
    assert settings.recommendation.advisor_mode == "rules"


def test_env_override_does_not_corrupt_missing_section() -> None:
    settings = load_config()
    assert settings.run.model.params is None


def _settings_with_llm(*, model: str | None) -> Settings:
    return Settings(llm={"model": model, "base_url": "http://x:8000/v1"})


def test_resolve_auto_with_llm_is_llm_rules() -> None:
    config = _settings_with_llm(model="qwen2.5-32b")
    assert resolve_advisor_mode(config) == "llm_rules"


def test_resolve_auto_without_llm_is_rules() -> None:
    config = _settings_with_llm(model=None)
    assert resolve_advisor_mode(config) == "rules"


def test_resolve_rules_ignores_configured_llm() -> None:
    config = _settings_with_llm(model="qwen2.5-32b")
    assert resolve_advisor_mode(config, requested="rules") == "rules"


def test_resolve_llm_rules_without_model_raises() -> None:
    config = _settings_with_llm(model=None)
    with pytest.raises(ValueError, match="requires an analyst LLM"):
        resolve_advisor_mode(config, requested="llm_rules")


def test_resolve_requested_overrides_config() -> None:
    config = _settings_with_llm(model=None)
    assert resolve_advisor_mode(config, requested="auto") == "rules"
    config = _settings_with_llm(model="qwen2.5-32b")
    assert resolve_advisor_mode(config, requested="rules") == "rules"
