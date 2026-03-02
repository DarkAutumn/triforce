"""Tests for QT-08: Model browser tree view."""

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeView

from triforce_debugger.model_browser import (
    ModelBrowser,
    PATH_ROLE,
    STEPS_ROLE,
    format_step_count,
    parse_step_count,
)


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ── Step count parsing ────────────────────────────────────────


class TestParseStepCount:
    """Tests for parse_step_count()."""

    def test_standard_checkpoint_name(self):
        """Standard checkpoint: model-name-scenario_1501764.pt → 1501764."""
        assert parse_step_count("sword-and-beams-skip-sword-to-triforce_1501764.pt") == 1501764

    def test_simple_name(self):
        """Simple name with underscore: model_50000.pt → 50000."""
        assert parse_step_count("model_50000.pt") == 50000

    def test_no_step_count(self):
        """No underscore-number pattern: overworld.pt → None."""
        assert parse_step_count("overworld.pt") is None

    def test_multiple_underscores(self):
        """Multiple underscores — uses last one: a_b_300.pt → 300."""
        assert parse_step_count("a_b_300.pt") == 300

    def test_zero_steps(self):
        """Zero is a valid step count: model_0.pt → 0."""
        assert parse_step_count("model_0.pt") == 0

    def test_non_numeric_suffix(self):
        """Non-numeric after underscore: model_best.pt → None."""
        assert parse_step_count("model_best.pt") is None


class TestFormatStepCount:
    """Tests for format_step_count()."""

    def test_with_separators(self):
        assert format_step_count(1501764) == "1,501,764 steps"

    def test_small_number(self):
        assert format_step_count(500) == "500 steps"

    def test_zero(self):
        assert format_step_count(0) == "0 steps"


# ── Tree building ─────────────────────────────────────────────


def _make_tree(tmp_path):
    """Build a mock directory tree and return the root path.

    Structure:
        root/
          training/
            run-001/
              model/
                model_100000.pt
                model_500000.pt
            run-002/
              model/
                model_200000.pt
          models/
            overworld.pt
            dungeon.pt
    """
    training = tmp_path / "training"
    run1 = training / "run-001" / "model"
    run1.mkdir(parents=True)
    (run1 / "model_100000.pt").touch()
    (run1 / "model_500000.pt").touch()

    run2 = training / "run-002" / "model"
    run2.mkdir(parents=True)
    (run2 / "model_200000.pt").touch()

    models = tmp_path / "models"
    models.mkdir()
    (models / "overworld.pt").touch()
    (models / "dungeon.pt").touch()

    return str(tmp_path)


class TestModelBrowserTree:
    """Tests for ModelBrowser tree construction."""

    def test_widget_structure(self):
        """ModelBrowser contains a QTreeView."""
        _app = get_app()
        browser = ModelBrowser()
        assert isinstance(browser.tree_view, QTreeView)
        assert browser.objectName() == "model_browser"

    def test_scan_empty_dir(self, tmp_path):
        """Scanning an empty directory produces an empty tree."""
        _app = get_app()
        browser = ModelBrowser()
        browser.scan_directory(str(tmp_path))
        assert browser.item_model.rowCount() == 0

    def test_scan_nonexistent_dir(self):
        """Scanning a non-existent directory produces an empty tree."""
        _app = get_app()
        browser = ModelBrowser()
        browser.scan_directory("/nonexistent/path/abc123")
        assert browser.item_model.rowCount() == 0

    def test_scan_directory_structure(self, tmp_path):
        """Scanning builds correct folder/file hierarchy."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)

        model = browser.item_model
        root_item = model.invisibleRootItem()

        # Should have 2 top-level entries: "models" and "training"
        top_names = sorted(root_item.child(i).text() for i in range(root_item.rowCount()))
        assert "models" in top_names
        assert "training" in top_names

    def test_pt_files_are_leaves_with_path(self, tmp_path):
        """Each .pt file stores its full path in PATH_ROLE."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)

        # Collect all leaf paths
        paths = []
        _collect_leaf_paths(browser.item_model.invisibleRootItem(), paths)

        assert len(paths) == 5  # 2 in run-001, 1 in run-002, 2 in models
        for p in paths:
            assert p.endswith('.pt')
            assert os.path.isabs(p)

    def test_step_count_displayed(self, tmp_path):
        """Leaves with step counts show formatted text (e.g. '100,000 steps')."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)

        texts = []
        _collect_leaf_texts(browser.item_model.invisibleRootItem(), texts)

        # Check that step-count leaves show formatted text
        assert "100,000 steps" in texts
        assert "500,000 steps" in texts
        assert "200,000 steps" in texts

    def test_no_step_count_shows_filename(self, tmp_path):
        """Leaves without step counts show the raw filename."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)

        texts = []
        _collect_leaf_texts(browser.item_model.invisibleRootItem(), texts)

        assert "overworld.pt" in texts
        assert "dungeon.pt" in texts

    def test_steps_role_stored(self, tmp_path):
        """Leaves store the integer step count in STEPS_ROLE."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)

        steps_values = []
        _collect_steps_roles(browser.item_model.invisibleRootItem(), steps_values)

        assert 100000 in steps_values
        assert 500000 in steps_values
        assert None in steps_values  # for overworld.pt / dungeon.pt

    def test_empty_dirs_excluded(self, tmp_path):
        """Directories with no .pt files (recursively) are not shown."""
        _app = get_app()
        empty_dir = tmp_path / "empty_folder"
        empty_dir.mkdir()
        (tmp_path / "model.pt").touch()

        browser = ModelBrowser()
        browser.scan_directory(str(tmp_path))

        root_item = browser.item_model.invisibleRootItem()
        names = [root_item.child(i).text() for i in range(root_item.rowCount())]
        assert "empty_folder" not in names

    def test_hidden_dirs_excluded(self, tmp_path):
        """Directories starting with '.' are excluded."""
        _app = get_app()
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "model_100.pt").touch()
        (tmp_path / "visible.pt").touch()

        browser = ModelBrowser()
        browser.scan_directory(str(tmp_path))

        root_item = browser.item_model.invisibleRootItem()
        names = [root_item.child(i).text() for i in range(root_item.rowCount())]
        assert ".hidden" not in names

    def test_rescan_clears_old_data(self, tmp_path):
        """Rescanning clears the previous tree before populating."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)
        assert browser.item_model.rowCount() > 0

        # Rescan an empty dir
        empty = tmp_path / "empty"
        empty.mkdir()
        browser.scan_directory(str(empty))
        assert browser.item_model.rowCount() == 0


# ── Bold highlighting ─────────────────────────────────────────


class TestModelBrowserBold:
    """Tests for set_loaded_model() bolding."""

    def test_loaded_model_is_bold(self, tmp_path):
        """The leaf matching the loaded model path is bold."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)

        target = str(tmp_path / "models" / "overworld.pt")
        browser.set_loaded_model(target)

        bold_items = []
        _collect_bold_items(browser.item_model.invisibleRootItem(), bold_items)
        assert len(bold_items) == 1
        assert bold_items[0].data(PATH_ROLE) == target

    def test_changing_loaded_model_unbolds_previous(self, tmp_path):
        """Changing the loaded model un-bolds the previous one."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)

        first = str(tmp_path / "models" / "overworld.pt")
        second = str(tmp_path / "models" / "dungeon.pt")

        browser.set_loaded_model(first)
        browser.set_loaded_model(second)

        bold_items = []
        _collect_bold_items(browser.item_model.invisibleRootItem(), bold_items)
        assert len(bold_items) == 1
        assert bold_items[0].data(PATH_ROLE) == second

    def test_set_none_unbolds_all(self, tmp_path):
        """Setting loaded model to None un-bolds everything."""
        _app = get_app()
        root = _make_tree(tmp_path)
        browser = ModelBrowser()
        browser.scan_directory(root)

        browser.set_loaded_model(str(tmp_path / "models" / "overworld.pt"))
        browser.set_loaded_model(None)

        bold_items = []
        _collect_bold_items(browser.item_model.invisibleRootItem(), bold_items)
        assert len(bold_items) == 0


# ── Signal emission ───────────────────────────────────────────


class TestModelBrowserSignal:
    """Tests for the model_selected signal."""

    def test_double_click_emits_signal(self, tmp_path):
        """Double-clicking a .pt leaf emits model_selected with the path."""
        _app = get_app()
        (tmp_path / "test_model_500.pt").touch()

        browser = ModelBrowser()
        browser.scan_directory(str(tmp_path))

        received = []
        browser.model_selected.connect(received.append)

        # Simulate double-click on the first (only) item
        index = browser.item_model.index(0, 0)
        browser._on_double_click(index)  # pylint: disable=protected-access

        assert len(received) == 1
        assert received[0].endswith("test_model_500.pt")

    def test_double_click_folder_no_signal(self, tmp_path):
        """Double-clicking a folder node does NOT emit model_selected."""
        _app = get_app()
        sub = tmp_path / "subfolder"
        sub.mkdir()
        (sub / "model_100.pt").touch()

        browser = ModelBrowser()
        browser.scan_directory(str(tmp_path))

        received = []
        browser.model_selected.connect(received.append)

        # The first item is the folder "subfolder"
        index = browser.item_model.index(0, 0)
        browser._on_double_click(index)  # pylint: disable=protected-access

        assert len(received) == 0


# ── Integration with main window ──────────────────────────────


class TestModelBrowserInMainWindow:
    """Tests that ModelBrowser is wired into the main window."""

    def test_main_window_has_model_browser(self):
        """MainWindow has a ModelBrowser instance."""
        _app = get_app()
        from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
        window = MainWindow()
        assert isinstance(window.model_browser, ModelBrowser)
        window.close()


# ── Helpers ───────────────────────────────────────────────────


def _collect_leaf_paths(parent, paths):
    """Recursively collect PATH_ROLE values from leaf items."""
    for row in range(parent.rowCount()):
        child = parent.child(row)
        if child is None:
            continue
        path = child.data(PATH_ROLE)
        if path is not None:
            paths.append(path)
        if child.rowCount() > 0:
            _collect_leaf_paths(child, paths)


def _collect_leaf_texts(parent, texts):
    """Recursively collect display text from leaf items (items with a PATH_ROLE)."""
    for row in range(parent.rowCount()):
        child = parent.child(row)
        if child is None:
            continue
        if child.data(PATH_ROLE) is not None:
            texts.append(child.text())
        if child.rowCount() > 0:
            _collect_leaf_texts(child, texts)


def _collect_steps_roles(parent, values):
    """Recursively collect STEPS_ROLE values from leaf items."""
    for row in range(parent.rowCount()):
        child = parent.child(row)
        if child is None:
            continue
        if child.data(PATH_ROLE) is not None:
            values.append(child.data(STEPS_ROLE))
        if child.rowCount() > 0:
            _collect_steps_roles(child, values)


def _collect_bold_items(parent, bold_items):
    """Recursively collect items that are bold."""
    for row in range(parent.rowCount()):
        child = parent.child(row)
        if child is None:
            continue
        if child.font().bold():
            bold_items.append(child)
        if child.rowCount() > 0:
            _collect_bold_items(child, bold_items)
