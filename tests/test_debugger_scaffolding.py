"""Tests for QT-01: Project scaffolding — debug.py and triforce_debugger package."""

import subprocess
import sys

from PySide6.QtWidgets import QApplication, QMainWindow


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_debugger_package_imports():
    """triforce_debugger package is importable."""
    import triforce_debugger  # pylint: disable=import-outside-toplevel,unused-import


def test_debug_module_imports():
    """debug.py module is importable and has expected symbols."""
    from debug import DebuggerWindow, main, parse_args  # pylint: disable=import-outside-toplevel,unused-import
    assert callable(main)
    assert callable(parse_args)


def test_debugger_window_creation():
    """DebuggerWindow can be instantiated headlessly."""
    _app = get_app()
    from debug import DebuggerWindow  # pylint: disable=import-outside-toplevel
    window = DebuggerWindow()
    assert isinstance(window, QMainWindow)
    assert window.windowTitle() == "Triforce Debugger"
    assert window.width() >= 1280
    assert window.height() >= 900
    window.close()


def test_parse_args_defaults():
    """parse_args returns correct defaults."""
    from debug import parse_args  # pylint: disable=import-outside-toplevel
    sys.argv = ["debug.py"]
    args = parse_args()
    assert args.path == "."


def test_parse_args_custom_path():
    """parse_args respects --path."""
    from debug import parse_args  # pylint: disable=import-outside-toplevel
    sys.argv = ["debug.py", "--path", "training/run-009"]
    args = parse_args()
    assert args.path == "training/run-009"


def test_debug_script_syntax():
    """debug.py has no syntax errors (import check via subprocess)."""
    result = subprocess.run(
        [sys.executable, "-c", "import debug"],
        capture_output=True, text=True, timeout=10,
        env={**__import__('os').environ, "QT_QPA_PLATFORM": "offscreen"}
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
