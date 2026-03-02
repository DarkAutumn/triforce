"""Tests for QT-22: Manual input (arrows + attack) wiring.

Verifies that arrow keys produce MOVE actions, A+arrow produces SWORD/BEAMS actions,
and that the manual step handler validates, pauses, and steps correctly.
"""

import sys
from unittest.mock import MagicMock

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from triforce.zelda_enums import ActionKind, Direction


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


def _send_key(window, key, event_type=QKeyEvent.Type.KeyPress, auto_repeat=False):
    """Send a synthetic key event to the window."""
    event = QKeyEvent(event_type, key, Qt.KeyboardModifier.NoModifier, "", auto_repeat)
    if event_type == QKeyEvent.Type.KeyPress:
        window.keyPressEvent(event)
    else:
        window.keyReleaseEvent(event)


# ── Direction mapping ────────────────────────────────────────


def test_direction_mapping_all():
    """_DIR_STR_TO_ENUM maps all four direction strings correctly."""
    from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
    assert MainWindow._DIR_STR_TO_ENUM == {
        'N': Direction.N,
        'S': Direction.S,
        'E': Direction.E,
        'W': Direction.W,
    }


# ── Manual move produces correct action tuples ───────────────


def test_manual_move_all_directions():
    """Each direction string maps to the correct MOVE action tuple."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    for dir_str in ['N', 'S', 'E', 'W']:
        window._on_manual_move(dir_str)

    assert actions == [
        (ActionKind.MOVE, Direction.N),
        (ActionKind.MOVE, Direction.S),
        (ActionKind.MOVE, Direction.E),
        (ActionKind.MOVE, Direction.W),
    ]
    window.close()


def test_manual_move_invalid_direction_ignored():
    """Invalid direction string does not produce a step."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    window._on_manual_move('X')
    assert actions == []
    window.close()


# ── Manual attack: SWORD vs BEAMS ────────────────────────────


def test_manual_attack_sword_no_bridge():
    """Without a bridge, attack defaults to SWORD."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    window._on_manual_attack('E')
    assert actions == [(ActionKind.SWORD, Direction.E)]
    window.close()


def test_manual_attack_sword_no_beams_in_space():
    """Attack uses SWORD when BEAMS not in action space."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    bridge = MagicMock()
    bridge.action_space.actions_allowed = [ActionKind.MOVE, ActionKind.SWORD]
    window._bridge = bridge

    window._on_manual_attack('W')
    assert actions == [(ActionKind.SWORD, Direction.W)]
    window.close()


def test_manual_attack_sword_when_no_beams_available():
    """Attack uses SWORD when BEAMS in space but link doesn't have beams."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    bridge = MagicMock()
    bridge.action_space.actions_allowed = [ActionKind.MOVE, ActionKind.SWORD, ActionKind.BEAMS]
    window._bridge = bridge

    state = MagicMock()
    state.link.has_beams = False
    window._last_zelda_state = state

    window._on_manual_attack('N')
    assert actions == [(ActionKind.SWORD, Direction.N)]
    window.close()


def test_manual_attack_beams_when_available():
    """Attack uses BEAMS when BEAMS in space and link has beams."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    bridge = MagicMock()
    bridge.action_space.actions_allowed = [ActionKind.MOVE, ActionKind.SWORD, ActionKind.BEAMS]
    window._bridge = bridge

    state = MagicMock()
    state.link.has_beams = True
    window._last_zelda_state = state

    window._on_manual_attack('S')
    assert actions == [(ActionKind.BEAMS, Direction.S)]
    window.close()


def test_manual_attack_invalid_direction_ignored():
    """Invalid direction string for attack does not produce a step."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    window._on_manual_attack('Z')
    assert actions == []
    window.close()


# ── _do_manual_step validation and behavior ──────────────────


def test_manual_step_no_bridge_noop():
    """Manual step with no bridge does nothing."""
    window = _make_window()
    window._do_manual_step((ActionKind.MOVE, Direction.N))
    window.close()


def test_manual_step_historical_noop():
    """Manual step while viewing historical step does nothing."""
    window = _make_window()
    bridge = MagicMock()
    window._bridge = bridge
    window._viewing_historical = True

    window._do_manual_step((ActionKind.MOVE, Direction.N))
    bridge.step.assert_not_called()
    bridge.is_valid_action.assert_not_called()
    window.close()


def test_manual_step_invalid_action_skipped():
    """Manual step with invalid action does not call bridge.step()."""
    window = _make_window()
    bridge = MagicMock()
    bridge.is_valid_action.return_value = False
    window._bridge = bridge

    window._do_manual_step((ActionKind.MOVE, Direction.N))
    bridge.step.assert_not_called()
    window.close()


def test_manual_step_pauses_timer():
    """Manual step pauses the game timer."""
    window = _make_window()

    bridge = MagicMock()
    bridge.is_valid_action.return_value = True
    step_result = MagicMock()
    step_result.completed = False
    step_result.frames = []
    step_result.state = None
    step_result.observation = None
    step_result.rewards = MagicMock()
    step_result.action_mask = None
    step_result.action_mask_desc = None
    step_result.terminated = False
    step_result.truncated = False
    step_result.state_change = MagicMock()
    step_result.state_change.action = None
    bridge.step.return_value = step_result
    bridge.get_probabilities.return_value = None
    bridge.model_details = "test"
    window._bridge = bridge

    # Start the timer running
    window.game_timer._running = True  # pylint: disable=protected-access

    window._do_manual_step((ActionKind.MOVE, Direction.N))
    assert not window.game_timer.is_running
    window.close()


def test_manual_step_calls_bridge_step():
    """Manual step calls bridge.step() with the action tuple."""
    window = _make_window()

    bridge = MagicMock()
    bridge.is_valid_action.return_value = True
    step_result = MagicMock()
    step_result.completed = False
    step_result.frames = []
    step_result.state = None
    step_result.observation = None
    step_result.rewards = MagicMock()
    step_result.action_mask = None
    step_result.action_mask_desc = None
    step_result.terminated = False
    step_result.truncated = False
    step_result.state_change = MagicMock()
    step_result.state_change.action = None
    bridge.step.return_value = step_result
    bridge.get_probabilities.return_value = None
    bridge.model_details = "test"
    window._bridge = bridge

    action = (ActionKind.SWORD, Direction.E)
    window._do_manual_step(action)
    bridge.step.assert_called_once_with(action=action)
    window.close()


# ── End-to-end: key → signal → handler ──────────────────────


def test_arrow_key_end_to_end():
    """Pressing Up arrow key fires signal → _on_manual_move → _do_manual_step."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    _send_key(window, Qt.Key.Key_Up)
    assert actions == [(ActionKind.MOVE, Direction.N)]
    window.close()


def test_a_arrow_key_end_to_end():
    """Holding A + pressing Right arrow → _on_manual_attack → SWORD action."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    _send_key(window, Qt.Key.Key_A)
    _send_key(window, Qt.Key.Key_Right)
    assert actions == [(ActionKind.SWORD, Direction.E)]
    window.close()


def test_a_arrow_beams_end_to_end():
    """A+arrow with beams available produces BEAMS action."""
    window = _make_window()
    actions = []
    window._do_manual_step = lambda t: actions.append(t)

    bridge = MagicMock()
    bridge.action_space.actions_allowed = [ActionKind.MOVE, ActionKind.SWORD, ActionKind.BEAMS]
    window._bridge = bridge

    state = MagicMock()
    state.link.has_beams = True
    window._last_zelda_state = state

    _send_key(window, Qt.Key.Key_A)
    _send_key(window, Qt.Key.Key_Down)
    assert actions == [(ActionKind.BEAMS, Direction.S)]
    window.close()
