"""Scenario selector dropdown for the Triforce Debugger.

Populates a QComboBox from triforce.json scenarios.  Emits a signal when the
user picks a different scenario so the environment can be reset.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox

from triforce import TrainingScenarioDefinition


class ScenarioSelector(QWidget):
    """Dropdown listing all scenarios from triforce.json.

    Signals:
        scenario_changed(str): Emitted with the new scenario name when the
            user selects a different scenario.
    """

    scenario_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("scenario_selector")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel("Scenario:")
        self._combo = QComboBox()
        self._combo.setObjectName("scenario_combo")

        layout.addWidget(label)
        layout.addWidget(self._combo)

        self._scenarios: list[TrainingScenarioDefinition] = []
        self._populate()

        self._combo.currentTextChanged.connect(self._on_selection_changed)  # pylint: disable=no-member

    # ── Public API ────────────────────────────────────────────

    @property
    def current_scenario_name(self) -> str | None:
        """The name currently shown in the dropdown, or None if empty."""
        text = self._combo.currentText()
        return text if text else None

    @property
    def current_scenario(self) -> TrainingScenarioDefinition | None:
        """The TrainingScenarioDefinition for the current selection."""
        name = self.current_scenario_name
        if name is None:
            return None
        for s in self._scenarios:
            if s.name == name:
                return s
        return None

    @property
    def scenario_names(self) -> list[str]:
        """All scenario names in the dropdown."""
        return [self._combo.itemText(i) for i in range(self._combo.count())]

    def set_scenario(self, name: str):
        """Programmatically select a scenario by name (no signal if already selected)."""
        idx = self._combo.findText(name)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

    # ── Internal ──────────────────────────────────────────────

    def _populate(self):
        """Load scenarios from triforce.json and fill the combobox."""
        self._scenarios = TrainingScenarioDefinition.get_all()
        self._combo.blockSignals(True)
        self._combo.clear()
        for s in self._scenarios:
            self._combo.addItem(s.name)
        self._combo.blockSignals(False)

    def _on_selection_changed(self, text: str):
        """Emit scenario_changed when the user picks a different scenario."""
        if text:
            self.scenario_changed.emit(text)
