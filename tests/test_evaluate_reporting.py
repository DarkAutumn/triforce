# pylint: disable=all
"""Evaluation reporting tests."""

import json

from evaluate import convert_eval_json_to_md


def _write_eval_json(path, metrics=None):
    data = {
        "episodes": 100,
        "scenario": "dungeon1-wallmaster-north-exit",
        "progress_values": [9] * 76 + [10] * 24,
        "max_progress": 11,
        "metrics": metrics,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_eval_markdown_uses_scenario_success_rate(tmp_path):
    json_path = tmp_path / "wallmaster.eval.json"
    _write_eval_json(json_path, {"success-rate": 0.24})

    md_path = convert_eval_json_to_md(str(json_path))

    markdown = (tmp_path / "wallmaster.eval.md").read_text(encoding="utf-8")
    assert md_path == str(tmp_path / "wallmaster.eval.md")
    assert "Success rate**: 24/100 (24%) (scenario success-rate; reached milestone 11)" in markdown


def test_eval_markdown_keeps_progress_success_without_metric(tmp_path):
    json_path = tmp_path / "wallmaster.eval.json"
    _write_eval_json(json_path)

    convert_eval_json_to_md(str(json_path))

    markdown = (tmp_path / "wallmaster.eval.md").read_text(encoding="utf-8")
    assert "Success rate**: 0/100 (0%) (reached milestone 11)" in markdown
