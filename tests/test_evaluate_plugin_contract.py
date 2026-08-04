import sys
from pathlib import Path

from evaluate import parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_plugin_defaults_to_50_episodes():
    extension = (ROOT / ".omp/extensions/triforce-evaluation/index.ts").read_text(encoding="utf-8")

    assert "episodes: z.number().int().positive().default(50)" in extension
    assert '"evaluate.py"' in extension
    assert '"--episodes"' in extension


def test_evaluation_plugin_forbids_direct_fallback_guidance():
    extension = (ROOT / ".omp/extensions/triforce-evaluation/index.ts").read_text(encoding="utf-8")
    skill = (ROOT / ".agents/skills/triforce-evaluation/SKILL.md").read_text(encoding="utf-8")

    assert "Do not run evaluate.py directly" in extension
    assert "Do not call evaluate.py directly" in skill


def test_training_skill_uses_evaluation_plugin():
    skill = (ROOT / ".agents/skills/triforce-experiment/SKILL.md").read_text(encoding="utf-8")

    assert "triforce_evaluation_start" in skill
    assert "triforce_evaluation_compare" in skill
    assert "python evaluate.py <final_model_or_run_dir>" not in skill


def test_evaluation_skills_forbid_sleep_poll_wait_loop():
    evaluation_skill = (ROOT / ".agents/skills/triforce-evaluation/SKILL.md").read_text(encoding="utf-8")
    training_skill = (ROOT / ".agents/skills/triforce-experiment/SKILL.md").read_text(encoding="utf-8")

    for skill in (evaluation_skill, training_skill):
        assert "Do not sleep, poll, wait" in skill
        assert "sleep to wait" not in skill
        assert "call status in a loop" in skill or "call evaluation status in a loop" in skill


def test_evaluate_default_remains_100_for_cli_backcompat(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "model.pt", "scenario-name"])

    args = parse_args()

    assert args.episodes == 100
