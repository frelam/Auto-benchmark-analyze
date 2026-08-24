"""Tests for the diagnosis engines: rule base (2.1) and llm agent (2.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_diagnosis.config import Settings, load_config
from benchmark_diagnosis.core import db
from benchmark_diagnosis.core.schema import ModelRecord
from benchmark_diagnosis.data import ingestion
from benchmark_diagnosis.diagnosis_engine.llm_agent import (
    AgentRunResult,
    prepare_case_pack,
    run_llm_agent_diagnosis,
)
from benchmark_diagnosis.diagnosis_engine.rule_base import (
    infer_missing_capabilities,
    run_rule_base,
)
from benchmark_diagnosis.intelligent_diagnosis.capability_taxonomy import (
    load_taxonomy,
)
from benchmark_diagnosis.pipeline import build_offline, diagnose_model

SCORES = {"mmlu_pro": 30.0, "math": 25.0, "swe_bench": 15.0, "aime24": 5.0}


@pytest.fixture()
def seeded_db(tmp_path):
    settings = load_config()
    settings.storage.db_path = str(tmp_path / "test.db")
    settings.run.output.dir = str(tmp_path / "out")
    engine = db.make_engine(settings.storage.db_path)
    db.init_db(engine)
    session = db.session_factory(engine)()
    ingestion.load_seed(session)
    build_offline(session, settings)
    yield session, settings
    session.close()


def _model() -> ModelRecord:
    return ModelRecord(
        model_id="llama-3-8b",
        name="Llama-3-8B",
        arch_type="dense",
        total_params=8.0,
        active_params=8.0,
        release_date=None,
    )


# ------------------------------------------------------------------ rule base


def test_rule_base_end_to_end(seeded_db):
    session, settings = seeded_db
    result = run_rule_base(
        session=session, model=_model(), raw_scores=SCORES, config=settings
    )
    assert result.engine == "rule"
    assert result.low_score_benchmarks, "some scores must be below percentile"
    for b in result.low_score_benchmarks:
        assert b["percentile"] < settings.curves.percentile_threshold
        assert b["curve_basis"]  # 同等参数量 / 激活参数量 label
    assert result.missing_capabilities
    for cap in result.missing_capabilities:
        assert cap["capability_id"] and 0 < cap["evidence"] <= 1.0
        assert cap["sources"], "every missing capability must cite low benchmarks"
    # suggestions map capabilities to datasets from the experience base
    by_cap = {s["capability_id"]: s for s in result.dataset_suggestions}
    assert set(by_cap) == {c["capability_id"] for c in result.missing_capabilities}
    assert any(s["datasets"] for s in result.dataset_suggestions)
    # JSON-serializable (metrics.json path)
    json.dumps(result.__dict__)


def test_infer_missing_capabilities_collapses_explained_ancestors():
    taxonomy = load_taxonomy()
    coverage = [
        {"benchmark_id": "aime24", "design_goal_tags": ["math", "reasoning"],
         "reliability_score": 1.0, "design_goal_agreement_score": 1.0,
         "saturated_flag": False},
    ]
    low = [
        {"benchmark_id": "aime24", "percentile": 5.0, "shortfall": 0.4,
         "score": 0.1, "residual": -0.4},
    ]
    missing = infer_missing_capabilities(low, coverage, taxonomy)
    ids = {m["capability_id"] for m in missing}
    # reasoning.math and reasoning both receive evidence; reasoning is NOT
    # collapsed because reasoning.math (0.5x) < 0.8 * reasoning (1.0).
    assert "reasoning.math" in ids
    assert "reasoning" in ids
    assert all(len(m["sources"]) == 1 for m in missing)  # deduped sources


def test_infer_missing_capabilities_noise_floor_and_dedup():
    taxonomy = load_taxonomy()
    coverage = [
        {"benchmark_id": "b1", "design_goal_tags": ["math"],
         "reliability_score": 1.0, "design_goal_agreement_score": 1.0,
         "saturated_flag": False},
        {"benchmark_id": "b2", "design_goal_tags": ["code"],
         "reliability_score": 0.02, "design_goal_agreement_score": 1.0,
         "saturated_flag": False},  # unreliable tag -> weak signal
    ]
    low = [
        {"benchmark_id": "b1", "percentile": 1.0, "shortfall": 0.5, "residual": -0.5},
        {"benchmark_id": "b2", "percentile": 2.0, "shortfall": 0.5, "residual": -0.5},
    ]
    missing = infer_missing_capabilities(low, coverage, taxonomy)
    ids = {m["capability_id"] for m in missing}
    assert "reasoning.math" in ids  # strong signal
    # code's 0.025 evidence falls below the 5% noise floor of the max
    assert "code" not in ids


# ------------------------------------------------------------------ llm agent


def _llm_agent_settings(**overrides) -> Settings:
    settings = load_config()
    settings.diagnosis.llm_agent.enabled = True
    settings.diagnosis.llm_agent.harness_cmd = "fake-harness {case_pack}"
    settings.diagnosis.llm_agent.interact_cmd = "fake-interact {case_pack} {message}"
    settings.diagnosis.llm_agent.max_rounds = 3
    settings.diagnosis.llm_agent.timeout_seconds = 30
    for key, value in overrides.items():
        setattr(settings.diagnosis.llm_agent, key, value)
    return settings


class _FakeProc:
    """subprocess.Popen stand-in: iterable stdout + poll/wait/terminate."""

    def __init__(self, on_construct=None, *, done: bool = False) -> None:
        self.stdout = []  # iterable; nothing to stream
        self.terminated = False
        self.killed = False
        self._done = done
        self._waited = False
        if on_construct is not None:
            on_construct()

    def poll(self):
        return 0 if self._done else None

    def wait(self, timeout: float | None = None):
        self._waited = True
        self._done = True
        return 0

    def terminate(self) -> None:
        self.terminated = True
        self._done = True

    def kill(self) -> None:
        self.killed = True
        self._done = True


def _write_conclusion(pack: Path, *, status: str, round_no: int = 1) -> None:
    out = pack / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "conclusion.json").write_text(
        json.dumps(
            {
                "status": status,
                "round": round_no,
                "summary": f"conclusion-{status}",
                "conclusions": [
                    {"capability_id": "reasoning.math", "confidence": "high",
                     "evidence": "test", "verified_by": []}
                ],
                "suggestions": [],
                "bad_case_analysis": {"n_cases": 0, "root_causes": {}},
            }
        ),
        encoding="utf-8",
    )


def _agent_env(env: dict) -> Path:
    return Path(env["BMD_CASE_PACK"])


def test_llm_agent_requires_full_config(seeded_db):
    session, settings = seeded_db
    settings.diagnosis.llm_agent.enabled = False
    with pytest.raises(ValueError, match="llm_agent"):
        run_llm_agent_diagnosis(
            session=session, model=_model(), raw_scores=SCORES, config=settings,
            rule_result={}, output_dir=Path("."), base_url=None, bad_cases_dir=None,
        )


def test_llm_agent_loop_final_on_first_round(seeded_db, tmp_path):
    session, settings = seeded_db
    settings.diagnosis.llm_agent.enabled = True
    settings.diagnosis.llm_agent.harness_cmd = "fake-harness {case_pack}"
    settings.diagnosis.llm_agent.interact_cmd = "fake-interact {case_pack} {message}"
    calls: list[dict] = []

    def popen(cmd, env):
        pack = _agent_env(env)
        _write_conclusion(pack, status="final", round_no=1)
        calls.append({"cmd": cmd, "message": env.get("BMD_MESSAGE")})
        return _FakeProc(done=True)

    result = run_llm_agent_diagnosis(
        session=session, model=_model(), raw_scores=SCORES, config=settings,
        rule_result={"engine": "rule"}, output_dir=tmp_path, base_url=None,
        bad_cases_dir=None, popen=popen,
    )
    assert result.concluded is True
    assert result.rounds == 1
    assert result.conclusion["status"] == "final"
    assert len(calls) == 1  # no interaction round needed


def test_llm_agent_loop_draft_then_interact_rounds(seeded_db, tmp_path):
    session, settings = seeded_db
    settings.diagnosis.llm_agent.enabled = True
    settings.diagnosis.llm_agent.harness_cmd = "fake-harness {case_pack}"
    settings.diagnosis.llm_agent.interact_cmd = "fake-interact {case_pack} {message}"
    calls: list[dict] = []

    def popen(cmd, env):
        pack = _agent_env(env)
        message = env.get("BMD_MESSAGE")
        if message:
            _write_conclusion(pack, status="final", round_no=2)
        else:
            _write_conclusion(pack, status="draft", round_no=1)
        calls.append({"cmd": cmd, "message": message})
        return _FakeProc(done=True)

    result = run_llm_agent_diagnosis(
        session=session, model=_model(), raw_scores=SCORES, config=settings,
        rule_result={"engine": "rule"}, output_dir=tmp_path, base_url=None,
        bad_cases_dir=None, popen=popen,
    )
    assert result.rounds == 2
    assert result.concluded is True
    assert len(calls) == 2
    assert calls[0]["message"] is None          # harness_cmd launch
    assert calls[1]["message"] is not None      # interact_cmd follow-up
    assert (tmp_path / "agent_run" / "input" / "followup_2.json").exists()


def test_llm_agent_loop_timeout_terminates(seeded_db, tmp_path):
    session, settings = seeded_db
    settings.diagnosis.llm_agent.enabled = True
    settings.diagnosis.llm_agent.harness_cmd = "fake-harness {case_pack}"
    settings.diagnosis.llm_agent.interact_cmd = "fake-interact {case_pack} {message}"
    settings.diagnosis.llm_agent.timeout_seconds = 1  # tiny timeout
    procs: list[_FakeProc] = []

    def popen(cmd, env):
        proc = _FakeProc(done=False)  # never finishes, never writes conclusion
        procs.append(proc)
        return proc

    result = run_llm_agent_diagnosis(
        session=session, model=_model(), raw_scores=SCORES, config=settings,
        rule_result={"engine": "rule"}, output_dir=tmp_path, base_url=None,
        bad_cases_dir=None, popen=popen,
    )
    assert result.timed_out is True
    assert result.concluded is False
    assert result.conclusion is None
    assert procs and procs[0].terminated


def test_prepare_case_pack_contents(seeded_db, tmp_path):
    session, settings = seeded_db
    bad_dir = tmp_path / "bad_cases"
    bad_dir.mkdir()
    (bad_dir / "math.jsonl").write_text("{}", encoding="utf-8")
    pack = prepare_case_pack(
        tmp_path,
        model=_model(),
        raw_scores=SCORES,
        rule_result={"engine": "rule", "missing_capabilities": []},
        bad_cases_dir=bad_dir,
        base_url="http://host:8000/v1",
        config=settings,
    )
    assert (pack / "input" / "scores.json").exists()
    assert (pack / "input" / "rule_base_result.json").exists()
    assert (pack / "input" / "bad_cases" / "math.jsonl").exists()
    assert (pack / "input" / "context.md").exists()
    target = json.loads((pack / "input" / "evaluation_target.json").read_text())
    assert target["available"] is True and target["base_url"] == "http://host:8000/v1"
    assert (pack / "skill" / "SKILL.md").exists()
    # no endpoint -> target says unavailable
    pack2 = prepare_case_pack(
        tmp_path / "out2", model=_model(), raw_scores=SCORES,
        rule_result={}, bad_cases_dir=None, base_url=None, config=settings,
    )
    target2 = json.loads((pack2 / "input" / "evaluation_target.json").read_text())
    assert target2["available"] is False


# ------------------------------------------------------------------ pipeline


def test_diagnose_model_rule_engine(seeded_db):
    session, settings = seeded_db
    report = diagnose_model(session, _model(), SCORES, settings, engine="rule")
    assert report["engine"] == "rule"
    block = report["diagnosis"]
    assert block["engine"] == "rule"
    assert block["rule_base"]["low_score_benchmarks"]
    assert block["rule_base"]["missing_capabilities"]
    assert report["clusters"]
    json.dumps(report)  # metrics.json path


def test_diagnose_model_llm_agent_engine(seeded_db, tmp_path, monkeypatch):
    session, settings = seeded_db
    settings.diagnosis.llm_agent.enabled = True
    settings.diagnosis.llm_agent.harness_cmd = "x {case_pack}"
    settings.diagnosis.llm_agent.interact_cmd = "y {message}"
    settings.run.output.dir = str(tmp_path)

    from benchmark_diagnosis import diagnosis_engine

    captured: dict = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return AgentRunResult(
            case_pack=str(tmp_path / "agent_run"),
            rounds=1,
            max_rounds=1,
            concluded=True,
            conclusion={
                "status": "final", "round": 1, "summary": "s",
                "conclusions": [], "suggestions": [], "bad_case_analysis": {},
            },
        )

    monkeypatch.setattr(
        diagnosis_engine.llm_agent, "run_llm_agent_diagnosis", fake_agent
    )
    report = diagnose_model(
        session, _model(), SCORES, settings, engine="llm_agent",
        base_url="http://host:8000/v1", bad_cases_dir=tmp_path / "bad_cases",
        output_dir=tmp_path,
    )
    block = report["diagnosis"]
    assert block["engine"] == "llm_agent"
    assert block["rule_base"]["missing_capabilities"]
    assert block["agent"]["concluded"] is True
    # the agent receives the rule-base block + endpoint + case pack location
    assert captured["base_url"] == "http://host:8000/v1"
    assert captured["rule_result"]["engine"] == "rule"
    json.dumps(report)


def test_diagnose_model_llm_agent_requires_config(seeded_db):
    session, settings = seeded_db  # llm_agent not configured
    with pytest.raises(ValueError, match="llm_agent"):
        diagnose_model(session, _model(), SCORES, settings, engine="llm_agent")


def test_skill_mirrors_are_in_sync():
    """The packaged skill and the repo-level DSH mirror must not drift."""
    repo_root = Path(__file__).resolve().parent.parent
    packaged = (
        repo_root / "src" / "benchmark_diagnosis" / "diagnosis_engine"
        / "skill" / "benchmark-diagnosis" / "SKILL.md"
    )
    mirror = repo_root / "skills" / "benchmark-diagnosis" / "SKILL.md"
    assert packaged.read_text(encoding="utf-8") == mirror.read_text(encoding="utf-8")
