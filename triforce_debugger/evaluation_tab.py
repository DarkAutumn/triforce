"""Evaluation tab for the detail panel.

If a ``.eval.json`` sidecar file exists for the selected model checkpoint,
shows a data grid with: episodes, success rate, progress percentiles
(p25/p50/p75/p90/max), and a milestone histogram.

If no evaluation exists, shows a "Run Evaluation" button with an episode
count spinner.  Clicking runs ``evaluate.py`` in a QProcess
(non-blocking), displays progress, and auto-populates on completion.
"""

import json
import os
import sys
from collections import Counter
from math import ceil

from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QPushButton,
    QSpinBox,
    QProgressBar,
    QStackedWidget,
)


# ── Data parsing ─────────────────────────────────────────────────────

class EvalData:
    """Parsed contents of a ``.eval.json`` file."""

    def __init__(self, episodes, scenario, progress_values, max_progress, metrics=None):
        self.episodes: int = episodes
        self.scenario: str = scenario
        self.progress_values: list[int] = progress_values
        self.max_progress: int = max_progress
        self.metrics: dict[str, float] = metrics or {}

    @classmethod
    def from_file(cls, path: str) -> "EvalData":
        """Load from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            episodes=data["episodes"],
            scenario=data["scenario"],
            progress_values=data["progress_values"],
            max_progress=data["max_progress"],
            metrics=data.get("metrics"),
        )

    def success_count(self) -> int:
        """Number of episodes that reached max_progress."""
        return sum(1 for v in self.progress_values if v >= self.max_progress)

    def success_rate(self) -> float:
        """Fraction of episodes that reached max_progress."""
        if not self.progress_values:
            return 0.0
        return self.success_count() / len(self.progress_values)

    def percentile(self, pct: float) -> int:
        """Return the value at the given percentile (0–1)."""
        vals = sorted(self.progress_values)
        if not vals:
            return 0
        idx = max(0, ceil(len(vals) * pct) - 1)
        return vals[idx]

    def milestone_histogram(self) -> list[tuple[int, int]]:
        """Return (milestone, count) pairs from 0..max_progress."""
        counts = Counter(self.progress_values)
        return [(m, counts.get(m, 0)) for m in range(self.max_progress + 1)]


def eval_json_path(model_path: str) -> str:
    """Derive the ``.eval.json`` sidecar path from a ``.pt`` model path."""
    return model_path.rsplit(".", 1)[0] + ".eval.json"


# ── Results widget ───────────────────────────────────────────────────

class _EvalResultsWidget(QWidget):
    """Shows evaluation results from an EvalData object."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel("Evaluation Results")
        self._header.setObjectName("eval_header")
        layout.addWidget(self._header)

        # Summary table (key/value pairs)
        self._summary = QTableWidget(0, 2)
        self._summary.setObjectName("eval_summary")
        self._summary.setHorizontalHeaderLabels(["Metric", "Value"])
        self._summary.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._summary.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._summary.verticalHeader().setVisible(False)
        sh = self._summary.horizontalHeader()
        sh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        sh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._summary)

        # Histogram table
        hist_label = QLabel("Milestone Histogram")
        layout.addWidget(hist_label)
        self._histogram = QTableWidget(0, 3)
        self._histogram.setObjectName("eval_histogram")
        self._histogram.setHorizontalHeaderLabels(["Milestone", "Count", "Distribution"])
        self._histogram.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._histogram.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._histogram.verticalHeader().setVisible(False)
        hh = self._histogram.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(True)
        layout.addWidget(self._histogram, stretch=1)

    def populate(self, data: EvalData):
        """Fill the summary and histogram from *data*."""
        self._header.setText(f"Evaluation: {data.scenario} ({data.episodes} episodes)")

        # Summary rows
        rows = [
            ("Episodes", str(data.episodes)),
            ("Success Rate", f"{data.success_count()}/{data.episodes}"
                             f" ({data.success_rate() * 100:.0f}%)"),
            ("P25", str(data.percentile(0.25))),
            ("P50 (Median)", str(data.percentile(0.50))),
            ("P75", str(data.percentile(0.75))),
            ("P90", str(data.percentile(0.90))),
            ("Max", str(data.percentile(1.0))),
        ]
        self._summary.setRowCount(len(rows))
        for row_idx, (key, val) in enumerate(rows):
            self._summary.setItem(row_idx, 0, QTableWidgetItem(key))
            vi = QTableWidgetItem(val)
            vi.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._summary.setItem(row_idx, 1, vi)

        # Histogram
        histogram = data.milestone_histogram()
        max_count = max((c for _, c in histogram), default=1) or 1
        self._histogram.setRowCount(len(histogram))
        for row_idx, (milestone, count) in enumerate(histogram):
            mi = QTableWidgetItem(str(milestone))
            mi.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            ci = QTableWidgetItem(str(count))
            ci.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            bar_len = round(count / max_count * 20)
            bi = QTableWidgetItem("█" * bar_len)
            self._histogram.setItem(row_idx, 0, mi)
            self._histogram.setItem(row_idx, 1, ci)
            self._histogram.setItem(row_idx, 2, bi)


# ── Empty / run widget ───────────────────────────────────────────────

class _EvalEmptyWidget(QWidget):
    """Shown when no .eval.json exists — offers to run evaluation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._message = QLabel("No evaluation found for this model.")
        self._message.setObjectName("eval_empty_message")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._message, stretch=1)

        # Run controls
        controls = QHBoxLayout()
        controls.addStretch()
        self._episode_spinner = QSpinBox()
        self._episode_spinner.setObjectName("eval_episode_spinner")
        self._episode_spinner.setRange(1, 10000)
        self._episode_spinner.setValue(100)
        self._episode_spinner.setSuffix(" episodes")
        controls.addWidget(self._episode_spinner)

        self._run_button = QPushButton("Run Evaluation")
        self._run_button.setObjectName("eval_run_button")
        controls.addWidget(self._run_button)
        controls.addStretch()
        layout.addLayout(controls)

        # Progress bar (hidden until running)
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("eval_progress_bar")
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setObjectName("eval_status")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

    @property
    def run_button(self) -> QPushButton:
        """The 'Run Evaluation' button."""
        return self._run_button

    @property
    def episode_count(self) -> int:
        """Currently selected episode count from the spinner."""
        return self._episode_spinner.value()

    @property
    def progress_bar(self) -> QProgressBar:
        """The progress bar widget."""
        return self._progress_bar

    @property
    def status_label(self) -> QLabel:
        """The status text label."""
        return self._status_label

    def set_running(self, episodes: int):
        """Switch to running state."""
        self._run_button.setEnabled(False)
        self._episode_spinner.setEnabled(False)
        self._progress_bar.setMaximum(episodes)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._status_label.setText("Running evaluation...")

    def set_idle(self):
        """Reset to idle state."""
        self._run_button.setEnabled(True)
        self._episode_spinner.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText("")

    def set_error(self, message: str):
        """Show error state."""
        self._run_button.setEnabled(True)
        self._episode_spinner.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Error: {message}")


# ── Main evaluation tab ──────────────────────────────────────────────

class EvaluationTab(QWidget):
    """Evaluation detail tab.

    Switches between a results view (when ``.eval.json`` exists) and an
    empty/run-evaluation view (when it doesn't).
    """

    # Stack indices
    _PAGE_EMPTY = 0
    _PAGE_RESULTS = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("evaluation_tab")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self._stack.setObjectName("eval_stack")

        self._empty_widget = _EvalEmptyWidget()
        self._results_widget = _EvalResultsWidget()

        self._stack.addWidget(self._empty_widget)   # index 0
        self._stack.addWidget(self._results_widget)  # index 1

        layout.addWidget(self._stack)

        # Current model info for running evaluation
        self._model_path: str | None = None
        self._model_name: str | None = None
        self._scenario_name: str | None = None

        # QProcess for background evaluation
        self._process: QProcess | None = None

        # Wire run button
        self._empty_widget.run_button.clicked.connect(self._on_run_clicked)  # pylint: disable=no-member

    # ── Public API ────────────────────────────────────────────

    def set_model(self, model_path: str | None, model_name: str | None = None,
                  scenario_name: str | None = None):
        """Update the model being viewed.  Checks for ``.eval.json`` and
        shows results or the run-evaluation form."""
        self._model_path = model_path
        self._model_name = model_name
        self._scenario_name = scenario_name

        if model_path and os.path.isfile(eval_json_path(model_path)):
            self._show_results(eval_json_path(model_path))
        else:
            self._stack.setCurrentIndex(self._PAGE_EMPTY)
            self._empty_widget.set_idle()

    def load_eval_file(self, path: str):
        """Directly load and display an eval JSON file."""
        self._show_results(path)

    @property
    def is_showing_results(self) -> bool:
        """True when displaying evaluation results rather than the empty/run form."""
        return self._stack.currentIndex() == self._PAGE_RESULTS

    @property
    def is_running(self) -> bool:
        """True when an evaluation process is currently running."""
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    # ── Internals ─────────────────────────────────────────────

    def _show_results(self, json_path: str):
        """Load and display results from a .eval.json file."""
        try:
            data = EvalData.from_file(json_path)
        except (json.JSONDecodeError, KeyError, OSError):
            self._stack.setCurrentIndex(self._PAGE_EMPTY)
            self._empty_widget.set_error("Failed to parse evaluation file.")
            return

        self._results_widget.populate(data)
        self._stack.setCurrentIndex(self._PAGE_RESULTS)

    def _on_run_clicked(self):
        """Launch evaluate.py as a background QProcess."""
        if not self._model_path or not self._model_name or not self._scenario_name:
            self._empty_widget.set_error("No model/scenario selected.")
            return

        episodes = self._empty_widget.episode_count
        model_dir = os.path.dirname(self._model_path)

        self._empty_widget.set_running(episodes)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_process_output)  # pylint: disable=no-member
        self._process.finished.connect(self._on_process_finished)  # pylint: disable=no-member

        evaluate_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluate.py")
        args = [
            evaluate_script,
            model_dir,
            self._model_name,
            self._scenario_name,
            "--episodes", str(episodes),
        ]
        self._process.start(sys.executable, args)

    def _on_process_output(self):
        """Read process output to update progress."""
        if self._process is None:
            return

        raw = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        # tqdm outputs percentage — try to parse it for progress
        self._empty_widget.status_label.setText(raw.strip()[-120:] if raw.strip() else "Running...")

    def _on_process_finished(self, exit_code, _exit_status):
        """Handle process completion."""
        self._process = None

        if exit_code == 0 and self._model_path:
            json_path = eval_json_path(self._model_path)
            if os.path.isfile(json_path):
                self._show_results(json_path)
                return

        if exit_code != 0:
            self._empty_widget.set_error(f"evaluate.py exited with code {exit_code}")
        else:
            self._empty_widget.set_idle()
