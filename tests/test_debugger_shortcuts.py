"""Tests for QT-06: Global keyboard shortcuts — application-level shortcuts and arrow/attack keys."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication


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


# ── Run menu shortcuts have ApplicationShortcut context ──────


def test_continue_shortcut_context():
    """F5 (Continue) has ApplicationShortcut context."""
    window = _make_window()
    assert window.action_continue.shortcutContext() == Qt.ShortcutContext.ApplicationShortcut
    window.close()


def test_pause_shortcut_context():
    """Shift+F5 (Pause) has ApplicationShortcut context."""
    window = _make_window()
    assert window.action_pause.shortcutContext() == Qt.ShortcutContext.ApplicationShortcut
    window.close()


def test_step_shortcut_context():
    """F10 (Step) has ApplicationShortcut context."""
    window = _make_window()
    assert window.action_step.shortcutContext() == Qt.ShortcutContext.ApplicationShortcut
    window.close()


def test_restart_shortcut_context():
    """Ctrl+Shift+F5 (Restart) has ApplicationShortcut context."""
    window = _make_window()
    assert window.action_restart.shortcutContext() == Qt.ShortcutContext.ApplicationShortcut
    window.close()


# ── Shortcut key sequences are still correct ─────────────────


def test_shortcut_keys_unchanged():
    """Run menu shortcuts retain their correct key sequences."""
    window = _make_window()
    assert window.action_continue.shortcut().toString() == "F5"
    assert window.action_pause.shortcut().toString() == "Shift+F5"
    assert window.action_step.shortcut().toString() == "F10"
    assert window.action_restart.shortcut().toString() == "Ctrl+Shift+F5"
    window.close()


# ── Manual move signals from arrow keys ──────────────────────


def _send_key(window, key, event_type=QKeyEvent.Type.KeyPress, auto_repeat=False):
    """Send a synthetic key event through the application event filter."""
    app = QApplication.instance()
    event = QKeyEvent(event_type, key, Qt.KeyboardModifier.NoModifier, "", auto_repeat)
    app.sendEvent(window, event)


def test_arrow_up_emits_move_north():
    """Pressing Up arrow emits manual_move_requested('N')."""
    window = _make_window()
    moves = []
    window.manual_move_requested.connect(moves.append)
    _send_key(window, Qt.Key.Key_Up)
    assert moves == ['N']
    window.close()


def test_arrow_down_emits_move_south():
    """Pressing Down arrow emits manual_move_requested('S')."""
    window = _make_window()
    moves = []
    window.manual_move_requested.connect(moves.append)
    _send_key(window, Qt.Key.Key_Down)
    assert moves == ['S']
    window.close()


def test_arrow_left_emits_move_west():
    """Pressing Left arrow emits manual_move_requested('W')."""
    window = _make_window()
    moves = []
    window.manual_move_requested.connect(moves.append)
    _send_key(window, Qt.Key.Key_Left)
    assert moves == ['W']
    window.close()


def test_arrow_right_emits_move_east():
    """Pressing Right arrow emits manual_move_requested('E')."""
    window = _make_window()
    moves = []
    window.manual_move_requested.connect(moves.append)
    _send_key(window, Qt.Key.Key_Right)
    assert moves == ['E']
    window.close()


# ── Manual attack signals from A+arrow ───────────────────────


def test_a_plus_up_emits_attack_north():
    """Holding A + pressing Up arrow emits manual_attack_requested('N')."""
    window = _make_window()
    attacks = []
    window.manual_attack_requested.connect(attacks.append)
    _send_key(window, Qt.Key.Key_A)   # hold A
    _send_key(window, Qt.Key.Key_Up)   # press arrow
    assert attacks == ['N']
    window.close()


def test_a_plus_down_emits_attack_south():
    """Holding A + pressing Down arrow emits manual_attack_requested('S')."""
    window = _make_window()
    attacks = []
    window.manual_attack_requested.connect(attacks.append)
    _send_key(window, Qt.Key.Key_A)
    _send_key(window, Qt.Key.Key_Down)
    assert attacks == ['S']
    window.close()


def test_a_plus_left_emits_attack_west():
    """Holding A + pressing Left arrow emits manual_attack_requested('W')."""
    window = _make_window()
    attacks = []
    window.manual_attack_requested.connect(attacks.append)
    _send_key(window, Qt.Key.Key_A)
    _send_key(window, Qt.Key.Key_Left)
    assert attacks == ['W']
    window.close()


def test_a_plus_right_emits_attack_east():
    """Holding A + pressing Right arrow emits manual_attack_requested('E')."""
    window = _make_window()
    attacks = []
    window.manual_attack_requested.connect(attacks.append)
    _send_key(window, Qt.Key.Key_A)
    _send_key(window, Qt.Key.Key_Right)
    assert attacks == ['E']
    window.close()


# ── A key release switches back to move mode ─────────────────


def test_a_release_switches_to_move():
    """After releasing A, arrow keys emit move again, not attack."""
    window = _make_window()
    moves = []
    attacks = []
    window.manual_move_requested.connect(moves.append)
    window.manual_attack_requested.connect(attacks.append)

    _send_key(window, Qt.Key.Key_A)  # hold A
    _send_key(window, Qt.Key.Key_Up)  # attack N
    _send_key(window, Qt.Key.Key_A, QKeyEvent.Type.KeyRelease)  # release A
    _send_key(window, Qt.Key.Key_Up)  # now should be move N

    assert attacks == ['N']
    assert moves == ['N']
    window.close()


# ── Auto-repeat arrows are ignored ──────────────────────────


def test_auto_repeat_arrow_ignored():
    """Auto-repeat arrow key events are ignored (no duplicate signals)."""
    window = _make_window()
    moves = []
    window.manual_move_requested.connect(moves.append)
    _send_key(window, Qt.Key.Key_Up)  # real press
    _send_key(window, Qt.Key.Key_Up, auto_repeat=True)  # auto-repeat
    _send_key(window, Qt.Key.Key_Up, auto_repeat=True)  # auto-repeat
    assert len(moves) == 1
    window.close()


# ── Signals exist on MainWindow ──────────────────────────────


def test_manual_move_signal_exists():
    """MainWindow has manual_move_requested signal."""
    window = _make_window()
    assert hasattr(window, 'manual_move_requested')
    window.close()


def test_manual_attack_signal_exists():
    """MainWindow has manual_attack_requested signal."""
    window = _make_window()
    assert hasattr(window, 'manual_attack_requested')
    window.close()


# ── No move/attack for non-arrow keys ────────────────────────


def test_non_arrow_key_no_signal():
    """Non-arrow keys (e.g., space) don't emit move or attack."""
    window = _make_window()
    moves = []
    attacks = []
    window.manual_move_requested.connect(moves.append)
    window.manual_attack_requested.connect(attacks.append)
    _send_key(window, Qt.Key.Key_Space)
    assert moves == []
    assert attacks == []
    window.close()
