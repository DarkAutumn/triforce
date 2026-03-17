"""Tests for QT-10: Observation panel — image, vectors, booleans, directional circles."""

import sys

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from triforce.zelda_enums import Direction


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_panel():
    from triforce_debugger.observation_panel import ObservationPanel  # pylint: disable=import-outside-toplevel
    _app = get_app()
    return ObservationPanel()


def _make_obs(**overrides):
    """Build a mock observation dict with sensible defaults."""
    obs = {
        "image": torch.zeros(4, 1, 84, 84),
        "entities": torch.zeros(12, 7),
        "entity_types": torch.zeros(12).long(),
        "information": torch.zeros(15),
    }
    obs.update(overrides)
    return obs


# ── Panel creation ─────────────────────────────────────────────


def test_panel_creates_without_crash():
    """ObservationPanel instantiates headlessly."""
    panel = _make_panel()
    assert panel.objectName() == "observation_panel"
    panel.close()


def test_panel_has_twelve_entity_rows():
    """Panel has 12 entity row widgets."""
    panel = _make_panel()
    assert len(panel.entity_rows) == 12
    panel.close()


def test_panel_has_directional_circles():
    """Panel has Objective and Source directional circles."""
    panel = _make_panel()
    assert panel.objective_circle is not None
    assert panel.objective_circle.label == "Objective"
    assert panel.source_circle is not None
    assert panel.source_circle.label == "Source"
    panel.close()


def test_panel_has_five_boolean_indicators():
    """Panel has 5 boolean indicators."""
    panel = _make_panel()
    assert len(panel.bool_indicators) == 5
    expected = {"Enemies", "Beams", "Low HP", "Full HP", "Clock"}
    assert set(panel.bool_indicators.keys()) == expected
    panel.close()




# ── DirectionalCircleWidget ──────────────────────────────────


def test_directional_circle_label():
    """DirectionalCircleWidget stores and exposes its label."""
    from triforce_debugger.observation_panel import DirectionalCircleWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    w = DirectionalCircleWidget("Objective")
    assert w.label == "Objective"
    w.close()


def test_directional_circle_set_directions():
    """Setting directions updates the state."""
    from triforce_debugger.observation_panel import DirectionalCircleWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    w = DirectionalCircleWidget("Source")
    w.set_directions([Direction.N, Direction.E])
    assert w.directions == [Direction.N, Direction.E]
    w.close()


def test_directional_circle_paints():
    """DirectionalCircleWidget paints without crash."""
    from triforce_debugger.observation_panel import DirectionalCircleWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    w = DirectionalCircleWidget("Objective")
    w.set_directions([Direction.N, Direction.S, Direction.E, Direction.W])
    w.show()
    w.repaint()
    w.close()


# ── BooleanIndicator ─────────────────────────────────────────


def test_boolean_indicator_default_inactive():
    """BooleanIndicator starts inactive (white)."""
    from triforce_debugger.observation_panel import BooleanIndicator  # pylint: disable=import-outside-toplevel
    _app = get_app()
    ind = BooleanIndicator("Enemies")
    assert not ind.active
    assert "255, 255, 255" in ind.styleSheet()
    ind.close()


def test_boolean_indicator_set_active():
    """Setting active changes color to dark blue."""
    from triforce_debugger.observation_panel import BooleanIndicator  # pylint: disable=import-outside-toplevel
    _app = get_app()
    ind = BooleanIndicator("Beams")
    ind.set_active(True)
    assert ind.active
    assert "0, 0, 160" in ind.styleSheet()
    ind.close()


def test_boolean_indicator_set_inactive():
    """Toggling back to inactive restores white."""
    from triforce_debugger.observation_panel import BooleanIndicator  # pylint: disable=import-outside-toplevel
    _app = get_app()
    ind = BooleanIndicator("Low HP")
    ind.set_active(True)
    ind.set_active(False)
    assert not ind.active
    assert "255, 255, 255" in ind.styleSheet()
    ind.close()


# ── update_observation integration ────────────────────────────


def test_update_observation_sets_booleans():
    """update_observation correctly sets boolean indicators."""
    panel = _make_panel()
    info = torch.zeros(15)
    info[10] = 1.0  # Enemies active
    info[13] = 1.0  # Full HP
    info[14] = 1.0  # Clock
    obs = _make_obs(information=info)
    panel.update_observation(obs)

    assert panel.bool_indicators["Enemies"].active
    assert not panel.bool_indicators["Beams"].active
    assert not panel.bool_indicators["Low HP"].active
    assert panel.bool_indicators["Full HP"].active
    assert panel.bool_indicators["Clock"].active
    panel.close()


def test_update_observation_sets_objective_directions():
    """update_observation sets objective directional circle."""
    panel = _make_panel()
    info = torch.zeros(15)
    info[0] = 1.0  # Objective North
    info[2] = 1.0  # Objective East
    obs = _make_obs(information=info)
    panel.update_observation(obs)

    assert Direction.N in panel.objective_circle.directions
    assert Direction.E in panel.objective_circle.directions
    assert Direction.S not in panel.objective_circle.directions
    panel.close()


def test_update_observation_sets_source_directions():
    """update_observation sets source directional circle."""
    panel = _make_panel()
    info = torch.zeros(15)
    info[7] = 1.0  # Source South
    obs = _make_obs(information=info)
    panel.update_observation(obs)

    assert Direction.S in panel.source_circle.directions
    assert Direction.N not in panel.source_circle.directions
    panel.close()


def test_update_observation_sets_entity_rows():
    """update_observation populates entity rows."""
    panel = _make_panel()
    panel.show()
    entities = torch.zeros(12, 7)
    entities[0, 0] = 1.0    # presence
    entities[0, 1] = 0.5    # dir_x
    entities[0, 2] = -0.3   # dir_y
    entity_types = torch.zeros(12).long()
    entity_types[0] = 1
    obs = _make_obs(entities=entities, entity_types=entity_types)
    panel.update_observation(obs)
    assert panel.entity_rows[0].isVisible()
    assert not panel.entity_rows[1].isVisible()
    panel.close()


def test_update_observation_none_is_safe():
    """update_observation(None) does nothing."""
    panel = _make_panel()
    panel.update_observation(None)
    panel.close()


def test_panel_paints_without_crash():
    """Full panel renders without crash."""
    panel = _make_panel()
    obs = _make_obs()
    panel.update_observation(obs)
    panel.show()
    panel.repaint()
    panel.close()
