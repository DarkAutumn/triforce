"""Tests for QT-07: Scenario selector dropdown."""

import sys

from PySide6.QtWidgets import QApplication

from triforce import TrainingScenarioDefinition
from triforce_debugger.scenario_selector import ScenarioSelector


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _expected_scenario_names():
    """Return scenario names from triforce.yaml for comparison."""
    return [s.name for s in TrainingScenarioDefinition.get_all()]


class TestScenarioSelector:
    """Tests for the ScenarioSelector widget."""

    def test_populated_from_triforce_json(self):
        """Combobox contains all scenarios from triforce.yaml."""
        _app = get_app()
        widget = ScenarioSelector()
        expected = _expected_scenario_names()
        assert len(expected) > 0, "triforce.yaml should have scenarios"
        assert widget.scenario_names == expected

    def test_current_scenario_name(self):
        """current_scenario_name returns the first scenario by default."""
        _app = get_app()
        widget = ScenarioSelector()
        expected = _expected_scenario_names()
        assert widget.current_scenario_name == expected[0]

    def test_current_scenario_object(self):
        """current_scenario returns a TrainingScenarioDefinition."""
        _app = get_app()
        widget = ScenarioSelector()
        scenario = widget.current_scenario
        assert isinstance(scenario, TrainingScenarioDefinition)
        assert scenario.name == widget.current_scenario_name

    def test_set_scenario_programmatic(self):
        """set_scenario changes the selection."""
        _app = get_app()
        widget = ScenarioSelector()
        names = widget.scenario_names
        assert len(names) >= 2, "Need at least two scenarios for this test"
        widget.set_scenario(names[1])
        assert widget.current_scenario_name == names[1]

    def test_scenario_changed_signal(self):
        """scenario_changed fires when the user picks a different scenario."""
        _app = get_app()
        widget = ScenarioSelector()
        names = widget.scenario_names
        assert len(names) >= 2

        received = []
        widget.scenario_changed.connect(received.append)

        # Simulate user selection
        widget.set_scenario(names[1])
        assert received == [names[1]]

    def test_no_signal_on_same_selection(self):
        """No signal when setting the already-selected scenario."""
        _app = get_app()
        widget = ScenarioSelector()
        first = widget.current_scenario_name

        received = []
        widget.scenario_changed.connect(received.append)
        widget.set_scenario(first)
        assert received == []

    def test_object_name(self):
        """Widget has the expected object name."""
        _app = get_app()
        widget = ScenarioSelector()
        assert widget.objectName() == "scenario_selector"

    def test_scenario_in_main_window(self):
        """ScenarioSelector is wired into the main window right panel."""
        _app = get_app()
        from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
        window = MainWindow()
        assert isinstance(window.scenario_selector, ScenarioSelector)
        assert len(window.scenario_selector.scenario_names) > 0
        window.close()
