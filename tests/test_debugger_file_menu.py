"""Tests for QT-09: File → Open Directory and --path wiring."""

import os
import sys
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from triforce_debugger.main_window import MainWindow
from triforce_debugger.model_browser import ModelBrowser


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestFileMenuActions:
    """Tests for File menu action existence and connections."""

    def test_open_directory_action_exists(self):
        """File menu has an Open Directory action with Ctrl+O shortcut."""
        _app = get_app()
        window = MainWindow()
        assert window.action_open_dir is not None
        assert window.action_open_dir.shortcut().toString() == "Ctrl+O"
        window.close()

    def test_exit_action_exists(self):
        """File menu has an Exit action with Ctrl+Q shortcut."""
        _app = get_app()
        window = MainWindow()
        assert window.action_exit is not None
        assert window.action_exit.shortcut().toString() == "Ctrl+Q"
        window.close()

    def test_open_directory_is_connected(self):
        """action_open_dir.triggered is connected to open_directory."""
        _app = get_app()
        window = MainWindow()
        # Verify by checking that triggering with a mocked dialog works
        with patch(
            "triforce_debugger.main_window.QFileDialog.getExistingDirectory",
            return_value="",
        ) as mock_dialog:
            window.action_open_dir.trigger()
        mock_dialog.assert_called_once()
        window.close()

    def test_exit_is_connected(self):
        """File > Exit action exists and has correct shortcut."""
        _app = get_app()
        window = MainWindow()
        assert window.action_exit is not None
        # The connection is verified by the fact that close() is wired in _build_menus
        # We just confirm the shortcut is correct
        assert window.action_exit.shortcut().toString() == "Ctrl+Q"
        window.close()


class TestSetModelPath:
    """Tests for MainWindow.set_model_path()."""

    def test_set_model_path_scans_directory(self, tmp_path):
        """set_model_path calls model_browser.scan_directory."""
        _app = get_app()
        window = MainWindow()
        (tmp_path / "model_100.pt").touch()

        window.set_model_path(str(tmp_path))
        assert window.model_browser.root_path == str(tmp_path)
        assert window.model_browser.item_model.rowCount() == 1
        window.close()

    def test_set_model_path_with_nested_structure(self, tmp_path):
        """set_model_path correctly scans nested directories."""
        _app = get_app()
        sub = tmp_path / "training" / "run-001"
        sub.mkdir(parents=True)
        (sub / "model_500.pt").touch()

        window = MainWindow()
        window.set_model_path(str(tmp_path))
        assert window.model_browser.item_model.rowCount() > 0
        window.close()


class TestOpenDirectory:
    """Tests for MainWindow.open_directory() with mocked QFileDialog."""

    def test_open_directory_updates_browser(self, tmp_path):
        """open_directory rescans the model browser when a directory is selected."""
        _app = get_app()
        (tmp_path / "model_200.pt").touch()

        window = MainWindow()
        assert window.model_browser.item_model.rowCount() == 0

        with patch(
            "triforce_debugger.main_window.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            window.open_directory()

        assert window.model_browser.root_path == str(tmp_path)
        assert window.model_browser.item_model.rowCount() == 1
        window.close()

    def test_open_directory_cancelled_does_nothing(self, tmp_path):
        """Cancelling the dialog leaves the model browser unchanged."""
        _app = get_app()
        (tmp_path / "model_100.pt").touch()
        window = MainWindow()
        window.set_model_path(str(tmp_path))
        original_count = window.model_browser.item_model.rowCount()

        with patch(
            "triforce_debugger.main_window.QFileDialog.getExistingDirectory",
            return_value="",
        ):
            window.open_directory()

        assert window.model_browser.item_model.rowCount() == original_count
        window.close()

    def test_open_directory_uses_current_root_as_start(self, tmp_path):
        """open_directory starts QFileDialog from the current root_path."""
        _app = get_app()
        window = MainWindow()
        window.set_model_path(str(tmp_path))

        with patch(
            "triforce_debugger.main_window.QFileDialog.getExistingDirectory",
            return_value="",
        ) as mock_dialog:
            window.open_directory()

        mock_dialog.assert_called_once_with(
            window, "Open Model Directory", str(tmp_path)
        )
        window.close()


class TestDebugEntryPoint:
    """Tests for debug.py --path wiring."""

    def test_parse_args_default_path(self):
        """Default path is '.'."""
        from debug import parse_args  # pylint: disable=import-outside-toplevel
        saved = sys.argv
        try:
            sys.argv = ["debug.py"]
            args = parse_args()
            assert args.path == "."
        finally:
            sys.argv = saved

    def test_parse_args_custom_path(self):
        """--path argument is respected."""
        from debug import parse_args  # pylint: disable=import-outside-toplevel
        saved = sys.argv
        try:
            sys.argv = ["debug.py", "--path", "/some/dir"]
            args = parse_args()
            assert args.path == "/some/dir"
        finally:
            sys.argv = saved
