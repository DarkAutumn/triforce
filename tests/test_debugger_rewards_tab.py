"""Tests for triforce_debugger.rewards_tab — headless (QT_QPA_PLATFORM=offscreen)."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass
from PySide6.QtWidgets import QApplication
import pytest

from triforce_debugger.rewards_tab import RewardsTab

# ── Ensure a QApplication exists ──────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ── Minimal outcome stubs ─────────────────────────────────────────────
@dataclass(frozen=True)
class _Outcome:
    name: str
    value: float
    count: int = 1


class _StepRewards:
    """Minimal mock matching StepRewards interface."""
    def __init__(self, outcomes, ending=None):
        self._outcomes = list(outcomes)
        self.ending = ending

    @property
    def value(self):
        return sum(o.value for o in self._outcomes)

    def __iter__(self):
        return iter(self._outcomes)

    def __len__(self):
        return len(self._outcomes)


# ── Tests ─────────────────────────────────────────────────────────────

class TestRewardsTabCreation:
    def test_widget_creates(self):
        tab = RewardsTab()
        assert tab.objectName() == "rewards_tab"
        assert tab._table is not None
        assert tab._header.text() == "Running Rewards"

    def test_initial_state(self):
        tab = RewardsTab()
        assert tab.running_totals == {}
        assert tab.episode_total == 0.0
        assert tab.endings == {}
        assert tab._table.rowCount() == 0


class TestAddStepRewards:
    def test_single_step(self):
        tab = RewardsTab()
        rewards = _StepRewards([
            _Outcome("reward-move-closer", 0.25),
            _Outcome("penalty-health", -0.50),
        ])
        tab.add_step_rewards(rewards)

        assert tab.running_totals == {
            "reward-move-closer": [1, 0.25],
            "penalty-health": [1, -0.50],
        }
        assert tab.episode_total == pytest.approx(-0.25)
        assert tab._table.rowCount() == 2

    def test_accumulation_across_steps(self):
        tab = RewardsTab()
        r1 = _StepRewards([
            _Outcome("reward-move-closer", 0.25),
            _Outcome("penalty-health", -0.10),
        ])
        r2 = _StepRewards([
            _Outcome("reward-move-closer", 0.25, count=1),
            _Outcome("reward-hit-enemy", 0.50),
        ])
        tab.add_step_rewards(r1)
        tab.add_step_rewards(r2)

        totals = tab.running_totals
        assert totals["reward-move-closer"] == [2, pytest.approx(0.50)]
        assert totals["penalty-health"] == [1, pytest.approx(-0.10)]
        assert totals["reward-hit-enemy"] == [1, pytest.approx(0.50)]
        assert tab.episode_total == pytest.approx(0.90)

    def test_table_sorted_by_abs_value(self):
        tab = RewardsTab()
        rewards = _StepRewards([
            _Outcome("reward-tiny", 0.01),
            _Outcome("penalty-big", -0.80),
            _Outcome("reward-medium", 0.40),
        ])
        tab.add_step_rewards(rewards)

        # Sorted by absolute value descending
        assert tab._table.rowCount() == 3
        assert tab._table.item(0, 0).text() == "penalty-big"
        assert tab._table.item(1, 0).text() == "reward-medium"
        assert tab._table.item(2, 0).text() == "reward-tiny"

    def test_value_column_formatting(self):
        tab = RewardsTab()
        rewards = _StepRewards([
            _Outcome("reward-a", 0.25),
            _Outcome("penalty-b", -0.10),
        ])
        tab.add_step_rewards(rewards)

        # Check formatting (sorted: reward-a first since |0.25| > |0.10|)
        assert tab._table.item(0, 2).text() == "+0.250"
        assert tab._table.item(1, 2).text() == "-0.100"


class TestEndings:
    def test_endings_tracked(self):
        tab = RewardsTab()
        r1 = _StepRewards([_Outcome("reward-a", 0.1)], ending="timeout")
        r2 = _StepRewards([_Outcome("reward-a", 0.1)], ending="timeout")
        r3 = _StepRewards([_Outcome("reward-a", 0.1)], ending="gameover")

        tab.add_step_rewards(r1)
        tab.add_step_rewards(r2)
        tab.add_step_rewards(r3)

        assert tab.endings == {"timeout": 2, "gameover": 1}

    def test_no_ending(self):
        tab = RewardsTab()
        rewards = _StepRewards([_Outcome("reward-a", 0.1)])
        tab.add_step_rewards(rewards)
        assert tab.endings == {}


class TestClear:
    def test_clear_resets_everything(self):
        tab = RewardsTab()
        rewards = _StepRewards([
            _Outcome("reward-a", 0.25),
        ], ending="timeout")
        tab.add_step_rewards(rewards)

        tab.clear()

        assert tab.running_totals == {}
        assert tab.episode_total == 0.0
        assert tab.endings == {}
        assert tab._table.rowCount() == 0
        assert tab._header.text() == "Running Rewards"


class TestShowStepRewards:
    def test_shows_single_step(self):
        tab = RewardsTab()
        rewards = _StepRewards([
            _Outcome("reward-hit-enemy", 0.50),
            _Outcome("penalty-health", -0.12),
        ])

        tab.show_step_rewards(rewards, step_number=42)

        assert tab._header.text() == "Step #42 Rewards"
        assert tab._table.rowCount() == 2
        # Sorted by abs value: hit-enemy (0.50) > health (0.12)
        assert tab._table.item(0, 0).text() == "reward-hit-enemy"
        assert tab._table.item(0, 2).text() == "+0.500"
        assert tab._table.item(1, 0).text() == "penalty-health"
        assert tab._table.item(1, 2).text() == "-0.120"

    def test_step_total_displayed(self):
        tab = RewardsTab()
        rewards = _StepRewards([
            _Outcome("reward-a", 0.30),
            _Outcome("penalty-b", -0.10),
        ])
        tab.show_step_rewards(rewards, step_number=10)

        assert "Step Total:" in tab._total_label.text()
        assert "+0.200" in tab._total_label.text()


class TestShowRunning:
    def test_switches_back_to_running(self):
        tab = RewardsTab()
        r1 = _StepRewards([_Outcome("reward-a", 0.25)])
        tab.add_step_rewards(r1)

        # Switch to step view
        r_step = _StepRewards([_Outcome("penalty-b", -0.10)])
        tab.show_step_rewards(r_step, step_number=5)
        assert tab._header.text() == "Step #5 Rewards"

        # Switch back to running
        tab.show_running()
        assert tab._header.text() == "Running Rewards"
        assert tab._table.rowCount() == 1
        assert tab._table.item(0, 0).text() == "reward-a"

    def test_new_steps_dont_update_table_during_step_view(self):
        tab = RewardsTab()
        r1 = _StepRewards([_Outcome("reward-a", 0.25)])
        tab.add_step_rewards(r1)

        # Enter step view
        tab.show_step_rewards(_StepRewards([_Outcome("penalty-x", -0.10)]), step_number=1)

        # Add another step while in step view — running totals update but table doesn't
        r2 = _StepRewards([_Outcome("reward-b", 0.50)])
        tab.add_step_rewards(r2)

        # Table should still show the step view (penalty-x)
        assert tab._table.rowCount() == 1
        assert tab._table.item(0, 0).text() == "penalty-x"

        # But running totals ARE accumulated
        assert "reward-b" in tab.running_totals

        # Switch back — now shows running totals
        tab.show_running()
        assert tab._table.rowCount() == 2


class TestTotalLabel:
    def test_total_label_format(self):
        tab = RewardsTab()
        rewards = _StepRewards([_Outcome("reward-a", 0.25)])
        tab.add_step_rewards(rewards)
        assert "+0.250" in tab._total_label.text()

    def test_negative_total(self):
        tab = RewardsTab()
        rewards = _StepRewards([_Outcome("penalty-a", -0.80, count=1)])
        # Need an outcome type that doesn't assert on prefix — use our mock
        tab.add_step_rewards(rewards)
        assert "-0.800" in tab._total_label.text()
