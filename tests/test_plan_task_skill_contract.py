# pylint: disable=all
"""Contract tests for the triforce-plan-task skill.

These assert the load-bearing rules of the plan-task dev loop, which are easy to
silently drop when editing prose: the item gets checked off on every path, code
lands only on its own merit, long training runs are gated, and nothing is pushed
straight to main.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents/skills/triforce-plan-task/SKILL.md"


@pytest.fixture(scope="module")
def skill():
    return SKILL.read_text(encoding="utf-8")


def test_skill_exists_with_frontmatter_name(skill):
    assert skill.startswith("---\n")
    assert "name: triforce-plan-task" in skill


def test_description_covers_the_invocation_phrasings(skill):
    # The user invokes this with loose phrasing; the description drives matching.
    description = skill.split("---")[1]
    for phrase in ("next proposed experiment", "next task", "plan"):
        assert phrase in description, f"description must mention {phrase!r}"


def test_one_task_per_invocation(skill):
    assert "exactly one" in skill.lower()
    assert "Do not begin the next task." in skill


def test_plan_doc_update_happens_on_every_path(skill):
    # The checkoff must not be conditional on the code landing.
    assert "Record the outcome in the plan doc (ALWAYS)" in skill
    assert "The plan-doc update always lands" in skill


def test_all_four_status_markers_are_defined(skill):
    for marker in ("`[x]`", "`[~]`", "`[!]`"):
        assert marker in skill, f"missing status marker {marker}"
    # The easiest mistake: treating a refuted hypothesis as an abandoned task.
    assert "refuted its own hypothesis is `[x]`" in skill


def test_experiment_class_tasks_are_gated(skill):
    assert "EXPERIMENT-class tasks are gated" in skill
    assert "stop and ask the user" in skill
    assert "triforce-experiment" in skill


def test_forbids_pushing_to_main_and_requires_green_ci(skill):
    assert "Never commit or push to `main`" in skill
    assert "Merge only when CI is green." in skill


def test_requires_the_uncommitted_work_check(skill):
    # A branch with zero commits reports "up to date with origin/main"; this
    # repo has already lost work to that illusion.
    assert "git log origin/main..HEAD" in skill


def test_evaluation_plugin_rule_is_preserved(skill):
    assert "Never call `evaluate.py` directly for a number you will record." in skill
    assert "triforce_evaluation_start" in skill


def test_discarded_code_path_still_opens_a_pr(skill):
    assert "When the code is discarded" in skill
    assert "Still open a PR" in skill


def test_default_plan_document_is_referenced(skill):
    assert "docs/experiments/dungeon1-completion-plan.md" in skill
    assert (ROOT / "docs/experiments/dungeon1-completion-plan.md").exists()
