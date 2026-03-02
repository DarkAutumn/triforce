"""Rewards tab for the detail panel.

Shows running reward/penalty totals accumulated across the episode, the episode
total, and an endings summary.  When time-travelling to a historical step the
tab switches to display that single step's reward breakdown.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)


# ── Colours ───────────────────────────────────────────────────────────
_COLOR_POSITIVE = QColor(0, 160, 0)
_COLOR_NEGATIVE = QColor(200, 0, 0)
_COLOR_NEUTRAL = QColor(128, 128, 128)


def _value_color(value: float) -> QColor:
    if value > 0:
        return _COLOR_POSITIVE
    if value < 0:
        return _COLOR_NEGATIVE
    return _COLOR_NEUTRAL


# ── Widget ────────────────────────────────────────────────────────────

class RewardsTab(QWidget):
    """Rewards detail tab showing running totals or per-step breakdown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rewards_tab")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header label — switches between "Running Rewards" and "Step #N Rewards"
        self._header = QLabel("Running Rewards")
        self._header.setObjectName("rewards_header")
        layout.addWidget(self._header)

        # Table: NAME | COUNT | TOTAL VALUE
        self._table = QTableWidget(0, 3)
        self._table.setObjectName("rewards_table")
        self._table.setHorizontalHeaderLabels(["Name", "Count", "Total Value"])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._table, stretch=1)

        # Episode total
        self._total_label = QLabel("Total: 0.000")
        self._total_label.setObjectName("rewards_total")
        layout.addWidget(self._total_label)

        # Endings summary
        self._endings_label = QLabel("Endings: {}")
        self._endings_label.setObjectName("rewards_endings")
        layout.addWidget(self._endings_label)

        # ── Internal state ────────────────────────────────────
        # Running totals: {outcome_name: [count, total_value]}
        self._running: dict[str, list] = {}
        self._episode_total: float = 0.0
        self._endings: dict[str, int] = {}

        # True when we're displaying a single step, not running totals
        self._showing_step = False

    # ── Public API ────────────────────────────────────────────

    def add_step_rewards(self, rewards) -> None:
        """Accumulate a step's rewards into running totals.

        *rewards* should be iterable of outcomes (e.g. ``StepRewards``),
        each having ``.name``, ``.value``, and ``.count`` attributes, and
        a ``.value`` property for the total.
        """
        step_total = rewards.value if hasattr(rewards, 'value') else 0.0
        self._episode_total += step_total

        for outcome in rewards:
            name = outcome.name
            if name in self._running:
                self._running[name][0] += outcome.count
                self._running[name][1] += outcome.value
            else:
                self._running[name] = [outcome.count, outcome.value]

        # Record ending if present
        ending = getattr(rewards, 'ending', None)
        if ending is not None:
            self._endings[ending] = self._endings.get(ending, 0) + 1

        if not self._showing_step:
            self._refresh_running()

    def show_step_rewards(self, rewards, step_number: int = 0) -> None:
        """Switch to displaying a single step's reward breakdown.

        Used when time-travelling to a historical step.
        """
        self._showing_step = True
        self._header.setText(f"Step #{step_number} Rewards")

        outcomes = list(rewards) if hasattr(rewards, '__iter__') else []
        step_total = rewards.value if hasattr(rewards, 'value') else 0.0

        self._populate_table(outcomes, single_step=True)
        self._total_label.setText(f"Step Total: {step_total:+.3f}")

        color = _value_color(step_total)
        self._total_label.setStyleSheet(f"color: {color.name()};")

    def show_running(self) -> None:
        """Switch back to displaying running totals (exit time-travel)."""
        self._showing_step = False
        self._refresh_running()

    def clear(self) -> None:
        """Clear all accumulated data (episode reset)."""
        self._running.clear()
        self._episode_total = 0.0
        self._endings.clear()
        self._showing_step = False
        self._refresh_running()

    @property
    def running_totals(self) -> dict[str, list]:
        """Return the current running totals dict (for testing)."""
        return dict(self._running)

    @property
    def episode_total(self) -> float:
        """The cumulative episode reward total."""
        return self._episode_total

    @property
    def endings(self) -> dict[str, int]:
        """Endings counter dict."""
        return dict(self._endings)

    # ── Internals ─────────────────────────────────────────────

    def _refresh_running(self) -> None:
        """Repopulate the table with running totals."""
        self._header.setText("Running Rewards")

        # Build flat outcome-like tuples sorted by absolute value descending
        sorted_items = sorted(
            self._running.items(), key=lambda kv: abs(kv[1][1]), reverse=True
        )

        self._table.setRowCount(len(sorted_items))
        for row, (name, (count, total_val)) in enumerate(sorted_items):
            name_item = QTableWidgetItem(name)
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_item = QTableWidgetItem(f"{total_val:+.3f}")
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_item.setForeground(_value_color(total_val))

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, count_item)
            self._table.setItem(row, 2, value_item)

        self._total_label.setText(f"Total: {self._episode_total:+.3f}")
        color = _value_color(self._episode_total)
        self._total_label.setStyleSheet(f"color: {color.name()};")

        self._endings_label.setText(f"Endings: {self._endings}")

    def _populate_table(self, outcomes, single_step=False) -> None:
        """Fill the table from a list of outcomes."""
        sorted_outcomes = sorted(outcomes, key=lambda o: abs(o.value), reverse=True)

        self._table.setRowCount(len(sorted_outcomes))
        for row, outcome in enumerate(sorted_outcomes):
            name_item = QTableWidgetItem(outcome.name)
            count_item = QTableWidgetItem(str(outcome.count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            val = outcome.value
            value_item = QTableWidgetItem(f"{val:+.3f}")
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_item.setForeground(_value_color(val))

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, count_item)
            self._table.setItem(row, 2, value_item)

        if not single_step:
            self._endings_label.setText(f"Endings: {self._endings}")
