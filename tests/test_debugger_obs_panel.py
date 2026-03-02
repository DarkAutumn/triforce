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
        "enemy_features": torch.zeros(4, 6),
        "item_features": torch.zeros(2, 4),
        "projectile_features": torch.zeros(2, 5),
        "information": torch.zeros(14),
    }
    obs.update(overrides)
    return obs


# ── Panel creation ─────────────────────────────────────────────


def test_panel_creates_without_crash():
    """ObservationPanel instantiates headlessly."""
    panel = _make_panel()
    assert panel.objectName() == "observation_panel"
    panel.close()


def test_panel_has_obs_image():
    """Panel contains an ObsImageWidget."""
    panel = _make_panel()
    assert panel.obs_image is not None
    assert panel.obs_image.objectName() == "obs_image"
    panel.close()


def test_panel_has_eight_vector_widgets():
    """Panel has 8 vector circle widgets (4 enemy + 2 proj + 2 item)."""
    panel = _make_panel()
    assert len(panel.vector_widgets) == 8
    expected = {"Enemy 1", "Enemy 2", "Enemy 3", "Enemy 4",
                "Proj 1", "Proj 2", "Item 1", "Item 2"}
    assert set(panel.vector_widgets.keys()) == expected
    panel.close()


def test_panel_has_directional_circles():
    """Panel has Objective and Source directional circles."""
    panel = _make_panel()
    assert panel.objective_circle is not None
    assert panel.objective_circle.label == "Objective"
    assert panel.source_circle is not None
    assert panel.source_circle.label == "Source"
    panel.close()


def test_panel_has_four_boolean_indicators():
    """Panel has 4 boolean indicators."""
    panel = _make_panel()
    assert len(panel.bool_indicators) == 4
    expected = {"Enemies", "Beams", "Low HP", "Full HP"}
    assert set(panel.bool_indicators.keys()) == expected
    panel.close()


# ── ObsImageWidget ────────────────────────────────────────────


def test_obs_image_initial_state():
    """ObsImageWidget starts with no image."""
    from triforce_debugger.observation_panel import ObsImageWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    widget = ObsImageWidget()
    assert widget.current_image is None
    widget.close()


def test_obs_image_set_tensor():
    """Setting an image tensor creates a QImage."""
    from triforce_debugger.observation_panel import ObsImageWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    widget = ObsImageWidget()
    img = torch.rand(4, 1, 84, 84)
    widget.set_image(img)
    assert widget.current_image is not None
    assert widget.current_image.width() == 84
    assert widget.current_image.height() == 84
    widget.close()


def test_obs_image_numpy_array():
    """Setting a numpy array works too."""
    from triforce_debugger.observation_panel import ObsImageWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    widget = ObsImageWidget()
    img = np.random.rand(4, 1, 84, 84).astype(np.float32)
    widget.set_image(img)
    assert widget.current_image is not None
    assert widget.current_image.width() == 84
    widget.close()


def test_obs_image_set_none_clears():
    """Setting None clears the image."""
    from triforce_debugger.observation_panel import ObsImageWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    widget = ObsImageWidget()
    widget.set_image(torch.rand(4, 1, 84, 84))
    assert widget.current_image is not None
    widget.set_image(None)
    assert widget.current_image is None
    widget.close()


# ── VectorCircleWidget ────────────────────────────────────────


def test_vector_circle_label():
    """VectorCircleWidget stores and exposes its label."""
    from triforce_debugger.observation_panel import VectorCircleWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    w = VectorCircleWidget("Enemy 1")
    assert w.label == "Enemy 1"
    w.close()


def test_vector_circle_set_vector():
    """Setting vector and scale doesn't crash."""
    from triforce_debugger.observation_panel import VectorCircleWidget  # pylint: disable=import-outside-toplevel
    _app = get_app()
    w = VectorCircleWidget("Test")
    w.set_vector(np.array([0.5, -0.3]), 0.8)
    w.show()
    w.repaint()
    w.close()


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
    info = torch.zeros(14)
    info[10] = 1.0  # Enemies active
    info[13] = 1.0  # Full HP
    obs = _make_obs(information=info)
    panel.update_observation(obs)

    assert panel.bool_indicators["Enemies"].active
    assert not panel.bool_indicators["Beams"].active
    assert not panel.bool_indicators["Low HP"].active
    assert panel.bool_indicators["Full HP"].active
    panel.close()


def test_update_observation_sets_objective_directions():
    """update_observation sets objective directional circle."""
    panel = _make_panel()
    info = torch.zeros(14)
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
    info = torch.zeros(14)
    info[7] = 1.0  # Source South
    obs = _make_obs(information=info)
    panel.update_observation(obs)

    assert Direction.S in panel.source_circle.directions
    assert Direction.N not in panel.source_circle.directions
    panel.close()


def test_update_observation_sets_enemy_vectors():
    """update_observation populates enemy vector widgets."""
    panel = _make_panel()
    enemy_features = torch.zeros(4, 6)
    enemy_features[0, 1] = 0.3   # closeness → scale = 0.7
    enemy_features[0, 2] = 0.5   # direction x
    enemy_features[0, 3] = -0.5  # direction y
    obs = _make_obs(enemy_features=enemy_features)
    panel.update_observation(obs)
    # No crash — visual verification would need manual inspection
    panel.close()


def test_update_observation_sets_image():
    """update_observation populates the observation image."""
    panel = _make_panel()
    obs = _make_obs(image=torch.rand(4, 1, 84, 84))
    panel.update_observation(obs)
    assert panel.obs_image.current_image is not None
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
