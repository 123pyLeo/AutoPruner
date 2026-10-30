import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_pruning import driver
from agentic_pruning.agents.base import AgentContext, AgentDecision


@pytest.mark.parametrize("rounds", [1, 2])
def test_only_agent_accepted_diagnoses_enter_store(tmp_path, monkeypatch, rounds):
    diagnoses = [
        {"pattern_id": f"review_test_{i}", "rho": f"review_test_{i}",
         "evidence": "test outcome", "problem": "Test diagnosis",
         "scope": {"level": "model_specific"}, "strategy": "Review evidence"}
        for i in range(rounds)
    ]
    modules = SimpleNamespace(
        probe=SimpleNamespace(suite="review_test", dense_sr=0.95),
        mask_search=Mock(total_layers=32),
        recovery=Mock(),
        eval=Mock(),
        diagnosis=Mock(),
    )
    modules.recovery.is_valid.return_value = True
    modules.mask_search.is_tried.return_value = False
    modules.eval.lookup_cached.return_value = {"sr": 0.9, "n_trials": 10}
    modules.diagnosis.diagnose.side_effect = [[d] for d in diagnoses]

    def context(modules, rules, i, proposal, result, last_diagnoses, **kwargs):
        return AgentContext(i, rules, proposal, result, last_diagnoses)

    def propose(agent, ctx, *args, **kwargs):
        decision = AgentDecision(
            mask=[ctx.iter_idx], recovery="direct", rationale="test",
            success_gate={}, stop_gate={}, evidence_used=[],
            accept_diagnoses=[d["pattern_id"] for d in ctx.last_diagnoses],
        )
        return decision, {"probe_rounds_used": 0, "probe_calls": [],
                          "intent_rationale": "", "validation": {}}

    monkeypatch.setattr(driver, "build_context", context)
    monkeypatch.setattr(driver, "propose_with_probe_tool_loop", propose)
    summary = driver.run_loop(
        SimpleNamespace(name="review_test"), modules, rounds, tmp_path / "run",
        experience_dir=tmp_path / "experience", no_probe=True,
    )
    stored = {r["rho"] for r in summary["rule_store"]}
    assert diagnoses[-1]["pattern_id"] not in stored
    if rounds == 2:
        assert diagnoses[0]["pattern_id"] in stored
    logged = json.loads(
        (tmp_path / "run" / f"iter_{rounds - 1:03d}" / "diagnoses.json").read_text()
    )
    assert logged == [diagnoses[-1]]
    reloaded = driver.RuleStore(tmp_path / "experience", "review_test")
    assert {r["rho"] for r in reloaded.all_rules()} == stored
