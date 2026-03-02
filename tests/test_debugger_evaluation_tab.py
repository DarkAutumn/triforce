"""Tests for triforce_debugger.evaluation_tab — headless (QT_QPA_PLATFORM=offscreen)."""

import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import pytest

from triforce_debugger.evaluation_tab import EvalData, EvaluationTab, eval_json_path


# ── Ensure a QApplication exists ──────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ── Fixtures ──────────────────────────────────────────────────────────
SAMPLE_EVAL = {
    "episodes": 100,
    "scenario": "test-scenario",
    "progress_values": [3, 5, 5, 7, 7, 7, 7, 10, 10, 10],
    "max_progress": 10,
    "metrics": {"success-rate": 0.3, "room-progress": 7.1},
}


@pytest.fixture()
def eval_json_file(tmp_path):
    """Write sample eval JSON and return its path."""
    p = tmp_path / "model_100000.eval.json"
    p.write_text(json.dumps(SAMPLE_EVAL), encoding="utf-8")
    return str(p)


@pytest.fixture()
def model_pt_file(tmp_path):
    """Create a dummy .pt file and a matching .eval.json sidecar."""
    pt = tmp_path / "model_100000.pt"
    pt.write_bytes(b"dummy")
    sidecar = tmp_path / "model_100000.eval.json"
    sidecar.write_text(json.dumps(SAMPLE_EVAL), encoding="utf-8")
    return str(pt)


# ── EvalData unit tests ──────────────────────────────────────────────

class TestEvalData:
    def test_from_file(self, eval_json_file):
        data = EvalData.from_file(eval_json_file)
        assert data.episodes == 100
        assert data.scenario == "test-scenario"
        assert data.max_progress == 10
        assert len(data.progress_values) == 10

    def test_success_count(self):
        data = EvalData(10, "s", [5, 5, 10, 10, 10], 10)
        assert data.success_count() == 3

    def test_success_rate(self):
        data = EvalData(4, "s", [10, 10, 5, 5], 10)
        assert data.success_rate() == 0.5

    def test_success_rate_empty(self):
        data = EvalData(0, "s", [], 10)
        assert data.success_rate() == 0.0

    def test_percentile(self):
        # sorted: [1,2,3,4,5,6,7,8,9,10]
        data = EvalData(10, "s", list(range(1, 11)), 10)
        assert data.percentile(0.50) == 5
        assert data.percentile(0.25) == 3  # ceil(10*0.25)-1 = 2 → index 2 → val 3
        assert data.percentile(0.90) == 9
        assert data.percentile(1.0) == 10

    def test_milestone_histogram(self):
        data = EvalData(5, "s", [0, 0, 1, 2, 2], 3)
        hist = data.milestone_histogram()
        assert len(hist) == 4  # milestones 0,1,2,3
        assert hist[0] == (0, 2)
        assert hist[1] == (1, 1)
        assert hist[2] == (2, 2)
        assert hist[3] == (3, 0)


# ── eval_json_path helper ────────────────────────────────────────────

class TestEvalJsonPath:
    def test_derives_sidecar_path(self):
        assert eval_json_path("/foo/bar/model_123.pt") == "/foo/bar/model_123.eval.json"

    def test_handles_multiple_dots(self):
        assert eval_json_path("/a/b.c/model.v2.pt") == "/a/b.c/model.v2.eval.json"


# ── EvaluationTab widget tests ───────────────────────────────────────

class TestEvaluationTab:
    def test_instantiation(self):
        tab = EvaluationTab()
        assert tab.objectName() == "evaluation_tab"

    def test_default_shows_empty(self):
        tab = EvaluationTab()
        assert not tab.is_showing_results

    def test_set_model_with_eval(self, model_pt_file):
        tab = EvaluationTab()
        tab.set_model(model_pt_file, "test-model", "test-scenario")
        assert tab.is_showing_results

    def test_set_model_without_eval(self, tmp_path):
        pt = tmp_path / "no_eval_model.pt"
        pt.write_bytes(b"dummy")
        tab = EvaluationTab()
        tab.set_model(str(pt), "m", "s")
        assert not tab.is_showing_results

    def test_set_model_none(self):
        tab = EvaluationTab()
        tab.set_model(None)
        assert not tab.is_showing_results

    def test_load_eval_file(self, eval_json_file):
        tab = EvaluationTab()
        tab.load_eval_file(eval_json_file)
        assert tab.is_showing_results

    def test_results_widget_populated(self, eval_json_file):
        tab = EvaluationTab()
        tab.load_eval_file(eval_json_file)
        # The summary table should have rows
        results = tab._results_widget  # pylint: disable=protected-access
        summary = results._summary  # pylint: disable=protected-access
        assert summary.rowCount() == 7  # Episodes, Success Rate, P25-P90, Max

    def test_histogram_populated(self, eval_json_file):
        tab = EvaluationTab()
        tab.load_eval_file(eval_json_file)
        results = tab._results_widget  # pylint: disable=protected-access
        histogram = results._histogram  # pylint: disable=protected-access
        # max_progress=10, so milestones 0..10 = 11 rows
        assert histogram.rowCount() == 11

    def test_bad_json_shows_error(self, tmp_path):
        bad = tmp_path / "bad.eval.json"
        bad.write_text("not json", encoding="utf-8")
        tab = EvaluationTab()
        tab.load_eval_file(str(bad))
        assert not tab.is_showing_results

    def test_is_running_default_false(self):
        tab = EvaluationTab()
        assert not tab.is_running

    def test_empty_widget_run_button_exists(self):
        tab = EvaluationTab()
        empty = tab._empty_widget  # pylint: disable=protected-access
        assert empty.run_button is not None
        assert empty.episode_count == 100  # default

    def test_empty_widget_spinner_range(self):
        tab = EvaluationTab()
        empty = tab._empty_widget  # pylint: disable=protected-access
        spinner = empty._episode_spinner  # pylint: disable=protected-access
        assert spinner.minimum() == 1
        assert spinner.maximum() == 10000
