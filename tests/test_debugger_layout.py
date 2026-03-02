"""Tests for QT-03: Main window layout — splitter, panels, menus."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QSplitter, QTabWidget, QLabel


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_window():
    """Create a MainWindow for testing."""
    from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
    _app = get_app()
    return MainWindow()


# ── Splitter & overall structure ──────────────────────────────


def test_main_splitter_exists():
    """Main window has a vertical QSplitter as central widget."""
    window = _make_window()
    splitter = window.main_splitter
    assert isinstance(splitter, QSplitter)
    assert splitter.orientation() == Qt.Orientation.Vertical
    assert splitter.count() == 2
    window.close()


def test_splitter_bottom_ge_half():
    """Bottom section should get ≥50% of the initial splitter sizes."""
    window = _make_window()
    # Use a tall window so real widget minimum sizes don't dominate the split
    window.resize(1280, 1600)
    window.show()
    sizes = window.main_splitter.sizes()
    assert sizes[1] >= sizes[0], f"Bottom ({sizes[1]}) should be >= top ({sizes[0]})"
    window.close()


# ── Top section panels ────────────────────────────────────────


def test_top_section_has_three_panels():
    """Top section contains obs panel, game view, and right panel."""
    window = _make_window()
    from triforce_debugger.observation_panel import ObservationPanel  # pylint: disable=import-outside-toplevel
    from triforce_debugger.game_view import GameView  # pylint: disable=import-outside-toplevel
    assert isinstance(window.obs_panel, ObservationPanel)
    assert isinstance(window.game_view, GameView)
    assert window.right_panel.objectName() == "right_panel"
    window.close()


def test_right_panel_sub_widgets():
    """Right panel contains model browser, scenario selector, action table."""
    window = _make_window()
    from triforce_debugger.model_browser import ModelBrowser  # pylint: disable=import-outside-toplevel
    assert isinstance(window.model_browser, ModelBrowser)
    from triforce_debugger.scenario_selector import ScenarioSelector  # pylint: disable=import-outside-toplevel
    assert isinstance(window.scenario_selector, ScenarioSelector)
    from triforce_debugger.action_table import ActionTable  # pylint: disable=import-outside-toplevel
    assert isinstance(window.action_table, ActionTable)
    window.close()


# ── Bottom section panels ─────────────────────────────────────


def test_bottom_section_has_step_history_and_tabs():
    """Bottom section has step history widget and tabbed detail panel."""
    window = _make_window()
    from triforce_debugger.step_history import StepHistoryWidget  # pylint: disable=import-outside-toplevel
    assert isinstance(window.step_history, StepHistoryWidget)
    assert isinstance(window.detail_tabs, QTabWidget)
    window.close()


def test_detail_tabs_have_three_tabs():
    """Detail tab widget has Rewards, State, and Evaluation tabs."""
    window = _make_window()
    tabs = window.detail_tabs
    assert tabs.count() == 3
    assert tabs.tabText(0) == "Rewards"
    assert tabs.tabText(1) == "State"
    assert tabs.tabText(2) == "Evaluation"
    window.close()


# ── Menu bar ──────────────────────────────────────────────────


def test_menu_bar_has_file_view_run():
    """Menu bar contains File, View, and Run menus."""
    window = _make_window()
    assert window.file_menu is not None
    assert window.view_menu is not None
    assert window.run_menu is not None
    window.close()


def test_file_menu_actions():
    """File menu has Open Directory and Exit actions."""
    window = _make_window()
    action_texts = [a.text() for a in window.file_menu.actions() if not a.isSeparator()]
    assert "Open Directory..." in action_texts
    assert "Exit" in action_texts
    window.close()


def test_view_menu_overlay_toggles():
    """View menu has checkable overlay toggles and Uncap FPS."""
    window = _make_window()
    assert window.action_overlay_wavefront.isCheckable()
    assert window.action_overlay_tile_ids.isCheckable()
    assert window.action_overlay_walkability.isCheckable()
    assert window.action_overlay_coordinates.isCheckable()
    assert window.action_uncap_fps.isCheckable()
    window.close()


def test_run_menu_actions():
    """Run menu has Continue, Pause, Step, and Restart actions."""
    window = _make_window()
    action_texts = [a.text() for a in window.run_menu.actions()]
    assert "Continue" in action_texts
    assert "Pause" in action_texts
    assert "Step" in action_texts
    assert "Restart" in action_texts
    window.close()


def test_run_menu_shortcuts():
    """Run menu actions have correct keyboard shortcuts."""
    window = _make_window()
    assert window.action_continue.shortcut().toString() == "F5"
    assert window.action_pause.shortcut().toString() == "Shift+F5"
    assert window.action_step.shortcut().toString() == "F10"
    assert window.action_restart.shortcut().toString() == "Ctrl+Shift+F5"
    window.close()
